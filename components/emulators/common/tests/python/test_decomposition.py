"""Tests for the column <-> grid decomposition.

Run directly (``python3 test_decomposition.py``) or under ctest.  Needs numpy
and nothing else.
"""

from __future__ import annotations

import sys
import unittest

import numpy as np

from e3sm_emulator.comm import SerialComm, run_where
from e3sm_emulator.context import Context
from e3sm_emulator.decomposition import (
    PermutationExchange,
    ReplicaExchange,
    Tiling,
    cell_indices,
    split_bounds,
)
from fake_cluster import run_ranks

NY, NX = 6, 8


def round_robin_gids(rank: int, size: int, total: int) -> np.ndarray:
    """A deliberately awkward decomposition: strided, so no rank owns a
    rectangle and nothing lines up with the model's tiling by luck."""
    return np.arange(1 + rank, total + 1, size, dtype=np.int64)


def field_from_gids(gids: np.ndarray, offset: float = 0.0) -> np.ndarray:
    """A value that identifies its own column, so a misplaced one shows."""
    return gids.astype(np.float64) * 10.0 + offset


def torch_harmonics_split_shapes(size: int, num_chunks: int) -> list:
    """`torch_harmonics.distributed.compute_split_shapes`, reproduced.

    ACE slices its spatial dimensions with this exact function, so our tiling
    has to agree with it or our columns land in the wrong rows of the model's
    tensor — a wrong answer, not an error.  Copied here (rather than imported)
    so the property is pinned in CI on a machine with no torch installed.
    """
    base, remainder = divmod(size, num_chunks)
    return [base + 1] * remainder + [base] * (num_chunks - remainder)


class TestIndexing(unittest.TestCase):
    def test_split_bounds_covers_everything(self):
        bounds = split_bounds(10, 3)
        self.assertEqual(bounds[0], 0)
        self.assertEqual(bounds[-1], 10)

    def test_split_bounds_matches_torch_harmonics(self):
        # The remainder goes to the LOW-numbered ranks. Getting this backwards
        # is invisible on an even split and wrong on every uneven one.
        for size, parts in [(10, 3), (10, 4), (180, 8), (360, 7), (721, 6)]:
            sizes = list(np.diff(split_bounds(size, parts)))
            self.assertEqual(
                sizes,
                torch_harmonics_split_shapes(size, parts),
                f"split_bounds({size}, {parts}) disagrees with torch_harmonics",
            )

    def test_split_bounds_matches_numpy_array_split(self):
        # Same convention, and a second independent statement of it.
        for size, parts in [(10, 3), (10, 4), (97, 5)]:
            sizes = list(np.diff(split_bounds(size, parts)))
            expected = [len(c) for c in np.array_split(np.arange(size), parts)]
            self.assertEqual(sizes, expected)

    def test_split_bounds_refuses_empty_tiles(self):
        with self.assertRaises(ValueError):
            split_bounds(3, 4)

    def test_cell_indices_round_trip(self):
        gids = np.arange(1, NY * NX + 1)
        j, i = cell_indices(gids, NY, NX, lon_fastest=True)
        np.testing.assert_array_equal(j * NX + i + 1, gids)

    def test_cell_indices_rejects_a_foreign_grid(self):
        with self.assertRaises(ValueError) as caught:
            cell_indices(np.array([NY * NX + 1]), NY, NX)
        self.assertIn("do not agree", str(caught.exception))

    def test_tiling_partitions_the_grid_exactly(self):
        tiling = Tiling(NY, NX, h=2, w=4)
        seen = {}
        for j in range(NY):
            for i in range(NX):
                rank, offset = tiling.locate(np.array([j]), np.array([i]))
                key = (int(rank[0]), int(offset[0]))
                self.assertNotIn(key, seen, "two cells claim the same slot")
                seen[key] = (j, i)
        self.assertEqual(len(seen), NY * NX)
        for rank in range(tiling.size):
            nj, ni = tiling.tile_shape(rank)
            self.assertEqual(
                sum(1 for r, _ in seen if r == rank),
                nj * ni,
                "tile size does not match what routes to it",
            )

    def test_ranks_past_the_mesh_own_nothing(self):
        tiling = Tiling(NY, NX, h=1, w=1)
        self.assertEqual(tiling.tile_shape(0), (NY, NX))
        self.assertEqual(tiling.tile_shape(3), (0, 0))

    def test_a_tiling_can_be_built_from_someone_elses_partition(self):
        # Spatial mode takes the partition from ACE rather than choosing one.
        tiling = Tiling.from_bounds(NY, NX, 2, 2, [0, 4, NY], [0, 5, NX])
        self.assertEqual(tiling.tile_shape(0), (4, 5))
        self.assertEqual(tiling.tile_shape(3), (NY - 4, NX - 5))
        self.assertFalse(tiling.agrees_with_even_split())
        self.assertTrue(Tiling(NY, NX, 2, 2).agrees_with_even_split())

    def test_a_foreign_partition_must_cover_the_grid(self):
        with self.assertRaises(ValueError):
            Tiling.from_bounds(NY, NX, 2, 2, [0, 4, NY - 1], [0, 5, NX])


class TestSerialExchange(unittest.TestCase):
    def test_round_trip(self):
        gids = np.arange(1, NY * NX + 1)
        exchange = PermutationExchange(SerialComm(), gids, Tiling(NY, NX))
        values = np.stack([field_from_gids(gids), field_from_gids(gids, 0.5)], axis=1)

        tile = exchange.to_tile(values)
        self.assertEqual(tile.shape, (NY, NX, 2))
        # Column with gid g sits at (j, i); check a couple by hand.
        self.assertAlmostEqual(tile[0, 0, 0], 10.0)
        self.assertAlmostEqual(tile[1, 2, 0], (1 * NX + 2 + 1) * 10.0)

        np.testing.assert_allclose(exchange.to_columns(tile), values)

    def test_grid_mismatch_is_loud(self):
        gids = np.arange(1, NY * NX)  # one column short
        with self.assertRaises(ValueError) as caught:
            PermutationExchange(SerialComm(), gids, Tiling(NY, NX))
        self.assertIn("same globe", str(caught.exception))


class TestDistributedExchange(unittest.TestCase):
    """The cases that only appear once the two decompositions disagree."""

    def _round_trip(self, size: int, h: int, w: int):
        total = NY * NX

        def body(comm, rank):
            gids = round_robin_gids(rank, size, total)
            exchange = PermutationExchange(comm, gids, Tiling(NY, NX, h, w))
            values = field_from_gids(gids).reshape(-1, 1)
            tile = exchange.to_tile(values)

            # Every cell this rank received must carry the value of the
            # column that actually lives there.
            nj, ni = exchange.tile_shape
            j0, i0 = Tiling(NY, NX, h, w).tile_origin(rank) if rank < h * w else (0, 0)
            for j in range(nj):
                for i in range(ni):
                    gid = (j0 + j) * NX + (i0 + i) + 1
                    assert tile[j, i, 0] == gid * 10.0, (
                        f"rank {rank} cell ({j},{i}) holds {tile[j, i, 0]}, "
                        f"expected column {gid}"
                    )

            back = exchange.to_columns(tile)
            np.testing.assert_allclose(back, values)
            return int(nj * ni)

        tiles = run_ranks(size, body)
        self.assertEqual(sum(tiles), total, "the tiles do not cover the globe")

    def test_one_rank_gathers_the_globe(self):
        # 1x1 mesh on four ranks: rank 0 assembles everything, the rest hand
        # their columns over and take the answers back.
        self._round_trip(size=4, h=1, w=1)

    def test_spatial_mesh(self):
        self._round_trip(size=4, h=2, w=2)

    def test_lopsided_mesh(self):
        self._round_trip(size=6, h=2, w=3)

    def test_replica_exchange_gives_everyone_the_globe(self):
        size = 3
        total = NY * NX

        def body(comm, rank):
            gids = round_robin_gids(rank, size, total)
            exchange = ReplicaExchange(comm, gids, NY, NX)
            grid = exchange.to_grid(field_from_gids(gids).reshape(-1, 1))
            assert grid.shape == (NY, NX, 1)
            for j in range(NY):
                for i in range(NX):
                    gid = j * NX + i + 1
                    assert grid[j, i, 0] == gid * 10.0
            # The return trip is local: each rank keeps its own columns.
            np.testing.assert_allclose(
                exchange.to_columns(grid), field_from_gids(gids).reshape(-1, 1)
            )
            return True

        self.assertEqual(run_ranks(size, body), [True] * size)


class TestContext(unittest.TestCase):
    def test_export_publishes_the_component_not_the_job(self):
        import os

        os.environ["SLURM_PROCID"] = "512"  # what the coupled job thinks
        os.environ["SLURM_NTASKS"] = "1024"
        context = Context(
            rank=3, world_size=8, local_rank=3, master_addr="nid001", master_port=41111
        )
        context.export(device_ordinal=0)
        self.assertEqual(os.environ["RANK"], "3")
        self.assertEqual(os.environ["WORLD_SIZE"], "8")
        # LOCAL_RANK is a *device ordinal*, not a rank: every consumer of it in
        # ACE and PhysicsNeMo feeds it to torch.cuda.set_device. Under the
        # recommended one-GPU-per-rank binding that is 0 on every rank, and
        # publishing the component-local rank 3 here would ask for a device
        # this process cannot see.
        self.assertEqual(os.environ["LOCAL_RANK"], "0")
        self.assertEqual(os.environ["MASTER_ADDR"], "nid001")
        self.assertEqual(os.environ["MASTER_PORT"], "41111")
        # ACE reads SLURM_* only when FME_USE_SRUN says to; it must not.
        self.assertEqual(os.environ["FME_USE_SRUN"], "0")

    def test_from_dict_matches_the_cxx_payload(self):
        context = Context.from_dict(
            {
                "rank": 2,
                "world_size": 4,
                "nx": NX,
                "ny": NY,
                "num_global_cols": NY * NX,
                "col_gids": [1, 5, 9],
                "lat": [0.0, 1.0, 2.0],
                "lon": [0.0, 1.0, 2.0],
            }
        )
        self.assertEqual(context.num_local_cols, 3)
        self.assertFalse(context.is_root)
        self.assertIn("3 of 48 columns", context.describe())


class TestAgreement(unittest.TestCase):
    """One rank failing must stop them all, not hang them all."""

    def test_nobody_unhappy_is_a_no_op(self):
        def body(comm, rank):
            comm.agree("")
            return "survived"

        self.assertEqual(run_ranks(3, body), ["survived"] * 3)

    def test_one_bad_rank_raises_everywhere(self):
        # The case that matters: in single mode only rank 0 loads the
        # checkpoint, so a bad path raises there while every other rank
        # finishes construction and then hangs on the next collective.
        def body(comm, rank):
            comm.agree("checkpoint is corrupt" if rank == 0 else "")
            return "survived"

        with self.assertRaises(RuntimeError) as caught:
            run_ranks(4, body)
        self.assertIn("checkpoint is corrupt", str(caught.exception))

    def test_a_healthy_rank_learns_the_real_reason(self):
        # A bare "some other rank failed" would send someone to the wrong
        # log, so the failing rank's message travels with the flag.
        errors = {}

        def body(comm, rank):
            try:
                comm.agree("disk on fire" if rank == 2 else "")
            except RuntimeError as exc:
                errors[rank] = str(exc)
            return True

        run_ranks(4, body)
        self.assertEqual(sorted(errors), [0, 1, 2, 3])
        self.assertEqual(errors[2], "disk on fire")
        for rank in (0, 1, 3):
            self.assertIn("rank 2", errors[rank])
            self.assertIn("disk on fire", errors[rank])
            self.assertIn("rather than hanging", errors[rank])

    def test_the_failing_rank_keeps_its_own_exception(self):
        # The rank that hit the problem should see the real type and the
        # original traceback, not a RuntimeError wrapper that hides both.
        original = ValueError("the grid does not match")

        def body(comm, rank):
            try:
                comm.agree("rank 0 says no" if rank == 0 else "", original)
            except BaseException as exc:  # noqa: BLE001 - inspected below
                return type(exc).__name__
            return "no error"

        kinds = run_ranks(3, body)
        self.assertEqual(kinds[0], "ValueError")
        self.assertEqual(kinds[1:], ["RuntimeError", "RuntimeError"])

    def test_a_rank_local_gid_error_is_agreed_not_hung(self):
        # cell_indices() inspects this rank's own global ids and runs before
        # the exchange's collectives, so a bad one on a single rank would
        # otherwise leave the others waiting in exchange_counts().
        size = 3
        total = NY * NX

        def body(comm, rank):
            gids = round_robin_gids(rank, size, total)
            if rank == 1:
                gids = gids + total  # off the grid entirely, on one rank only
            PermutationExchange(comm, gids, Tiling(NY, NX, 1, 1))
            return True

        with self.assertRaises(RuntimeError) as caught:
            run_ranks(size, body)
        self.assertIn("rank 1", str(caught.exception))


class TestOwnerOnlyWork(unittest.TestCase):
    """The shape of every step: gather -> owner computes -> scatter."""

    def _step(self, size, owner_raises):
        """Mimic AceEmulator.infer: a collective, owner-only work, a collective."""
        total = NY * NX

        def body(comm, rank):
            gids = round_robin_gids(rank, size, total)
            exchange = PermutationExchange(comm, gids, Tiling(NY, NX, 1, 1))
            owns = exchange.tile_shape[0] > 0

            # Gather (collective).
            tile = exchange.to_tile(field_from_gids(gids).reshape(-1, 1))

            def model():
                if owner_raises:
                    raise RuntimeError("CUDA out of memory")
                return tile * 2.0

            result = run_where(comm, owns, model)
            if result is None:
                result = np.empty((0, 0, 1), dtype=np.float64)

            # Scatter (collective). Only safe because run_where already
            # agreed: an owner that died above must not leave us here alone.
            return exchange.to_columns(result)

        return run_ranks(size, body)

    def test_a_clean_step_round_trips(self):
        results = self._step(size=4, owner_raises=False)
        self.assertEqual(len(results), 4)
        for rank, columns in enumerate(results):
            expected = field_from_gids(round_robin_gids(rank, 4, NY * NX)) * 2.0
            np.testing.assert_allclose(columns[:, 0], expected)

    def test_an_owner_failure_stops_every_rank(self):
        # Without the agreement the three non-owner ranks would enter
        # to_columns() and block there forever. The fake cluster reports that
        # as RankDivergence rather than rescuing it, so this test really does
        # fail if run_where stops agreeing.
        with self.assertRaises(RuntimeError) as caught:
            self._step(size=4, owner_raises=True)
        self.assertIn("CUDA out of memory", str(caught.exception))


class TestRestartExport(unittest.TestCase):
    """Restart state lives on one rank but has to come back on all of them."""

    def test_export_when_only_the_owner_holds_state(self):
        # The deadlock this guards against: non-owners see an empty state,
        # return early, and leave rank 0 alone inside to_columns().
        size = 3
        total = NY * NX

        def body(comm, rank):
            gids = round_robin_gids(rank, size, total)
            exchange = PermutationExchange(comm, gids, Tiling(NY, NX, 1, 1))
            owns = exchange.tile_shape[0] > 0
            state = {"air_temperature": 1.0} if owns else {}

            # The name list is agreed before anybody touches the exchange.
            announced = comm.allgather_text(
                "\n".join(sorted(state)) if owns else ""
            )
            names = sorted({n for b in announced for n in b.split("\n") if n})
            assert names == ["air_temperature"], f"rank {rank} saw {names}"
            if not names:
                return {}

            nj, ni = exchange.tile_shape
            stacked = np.stack(
                [
                    np.full((nj, ni), state[n]) if owns else np.empty((nj, ni))
                    for n in names
                ],
                axis=-1,
            )
            columns = exchange.to_columns(stacked)
            return {n: columns[:, k] for k, n in enumerate(names)}

        results = run_ranks(size, body)
        for rank, exported in enumerate(results):
            self.assertEqual(sorted(exported), ["air_temperature"])
            # Every rank gets its own columns back, owner or not.
            self.assertEqual(
                exported["air_temperature"].size,
                round_robin_gids(rank, size, total).size,
            )
            np.testing.assert_allclose(exported["air_temperature"], 1.0)


class TestBridge(unittest.TestCase):
    def test_unknown_emulator_names_the_alternatives(self):
        from e3sm_emulator.bridge import create_emulator

        with self.assertRaises(ValueError) as caught:
            create_emulator({"emulator": "nope"})
        message = str(caught.exception)
        self.assertIn("ace", message)
        self.assertIn("generic", message)
        self.assertIn("python_module", message)


class TestAceModeResolution(unittest.TestCase):
    """Mode selection, tested without ACE, torch or a checkpoint."""

    def setUp(self):
        from e3sm_emulator.ace import SPATIAL_OVERRIDE, resolve_mode

        self.resolve = resolve_mode
        self.override = SPATIAL_OVERRIDE

    def test_serial_runs_undistributed(self):
        self.assertEqual(self.resolve({}, Context(world_size=1)), ("single", 1, 1))

    def test_auto_will_not_invent_a_mesh(self):
        # Eight ranks and no mesh declared: gather to one rank rather than
        # guess a factorization that might be wrong or slow.
        self.assertEqual(self.resolve({}, Context(world_size=8)), ("single", 1, 1))

    def test_auto_uses_a_declared_mesh(self):
        self.assertEqual(
            self.resolve(
                {
                    "ace_h": "2",
                    "ace_w": "4",
                    self.override: "true",
                },
                Context(world_size=8),
            ),
            ("spatial", 2, 4),
        )

    def test_spatial_is_gated_on_an_explicit_opt_in(self):
        # The SFNO builders a deterministic ACE2 checkpoint instantiates still
        # call non-distributed spherical transforms, so a sharded input would
        # produce plausible numbers rather than an error. Refuse by default.
        with self.assertRaises(ValueError) as caught:
            self.resolve(
                {"ace_h": "2", "ace_w": "4"},
                Context(world_size=8),
            )
        self.assertIn("RealSHT", str(caught.exception))
        self.assertIn(self.override, str(caught.exception))

    def test_spatial_needs_every_rank(self):
        with self.assertRaises(ValueError) as caught:
            self.resolve(
                {"ace_mode": "spatial", "ace_h": "2", "ace_w": "3"},
                Context(world_size=8),
            )
        self.assertIn("must own a rectangle", str(caught.exception))

    def test_spatial_needs_a_mesh(self):
        with self.assertRaises(ValueError):
            self.resolve({"ace_mode": "spatial"}, Context(world_size=8))

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            self.resolve({"ace_mode": "magic"}, Context(world_size=1))

    def test_torch_distributed_is_refused_with_a_reason(self):
        # It splits a batch of globes across ranks; a coupled run has one.
        with self.assertRaises(ValueError) as caught:
            self.resolve({"ace_mode": "ensemble"}, Context(world_size=4))
        self.assertIn("one globe", str(caught.exception))


class TestDeviceContract(unittest.TestCase):
    """GPU ownership is not inferred from the component's rank numbering."""

    def test_one_visible_device_means_ordinal_zero(self):
        # The whole point. Under --gpus-per-task=1 every rank sees exactly one
        # device and must select logical 0; the fourth rank on a node asking
        # for device 3 is out of range. This is what LOCAL_RANK carries, and
        # what ACE feeds to torch.cuda.set_device.
        context = Context(rank=7, world_size=8, local_rank=3, local_size=4)
        self.assertEqual(context.device_ordinal(visible_devices=1), 0)

    def test_ambiguous_binding_is_refused(self):
        # Another component's ranks may hold some of those devices, and our
        # local_rank says nothing about which.
        context = Context(rank=1, world_size=4, local_rank=1, local_size=4)
        with self.assertRaises(ValueError) as caught:
            context.device_ordinal(visible_devices=4)
        message = str(caught.exception)
        self.assertIn("gpus-per-task", message)
        self.assertIn("device_id", message)

    def test_being_alone_in_this_component_is_not_ownership(self):
        # Only one rank of *this* component on the node says nothing about the
        # ocean and land ranks sharing it, one of which may already hold
        # device 0. The rule has to be the same one applied above, or it is
        # not a rule.
        context = Context(rank=1, world_size=4, local_rank=0, local_size=1)
        with self.assertRaises(ValueError):
            context.device_ordinal(visible_devices=4)

    def test_an_explicit_device_is_honoured_and_range_checked(self):
        context = Context(rank=1, world_size=4, local_rank=1, local_size=4)
        self.assertEqual(context.device_ordinal(2, visible_devices=4), 2)
        with self.assertRaises(ValueError):
            context.device_ordinal(9, visible_devices=4)

    def test_no_gpu_is_not_an_error(self):
        context = Context(rank=1, world_size=4, local_rank=1, local_size=4)
        self.assertEqual(context.device_ordinal(visible_devices=0), 0)


class TestMeshOrdering(unittest.TestCase):
    """The reconstructed (h, w) rank order has to be ACE's, not a guess."""

    def setUp(self):
        from e3sm_emulator.ace import _check_mesh_ordering

        self.check = _check_mesh_ordering
        self.tiling = Tiling(NY, NX, 2, 2)  # NY=6, NX=8 -> four 3x4 tiles

    def _reported(self, order):
        """Slices as if rank r held the tile of `order[r]`."""
        rows = []
        for rank in order:
            j0, i0 = self.tiling.tile_origin(rank)
            nj, ni = self.tiling.tile_shape(rank)
            rows.append([j0, j0 + nj, i0, i0 + ni])
        return np.array(rows, dtype=np.int64)

    def test_the_assumed_order_passes(self):
        self.check(self._reported([0, 1, 2, 3]), self.tiling, NY, NX, 2, 2)

    def test_a_permuted_rank_order_is_caught(self):
        # Every tile here is 3x4, so comparing shapes alone would pass while
        # each rank worked on somebody else's piece of the planet — and a
        # global model fed a rotated globe returns a plausible field.
        for rank in range(4):
            nj, ni = self.tiling.tile_shape(rank)
            self.assertEqual((nj, ni), (3, 4))
        with self.assertRaises(ValueError) as caught:
            self.check(self._reported([0, 2, 1, 3]), self.tiling, NY, NX, 2, 2)
        self.assertIn("does not match ACE's", str(caught.exception))

    def test_two_ranks_claiming_one_rectangle_are_caught(self):
        # The per-rank coordinate check fires first here, before the coverage
        # check does. That is the honest description: once every rank matches
        # the reconstruction, complete coverage follows by construction, so
        # the coverage check in _check_mesh_ordering is belt-and-braces
        # against a partition shape nobody has anticipated.
        reported = self._reported([0, 1, 2, 3])
        reported[3] = reported[0]  # rank 3 claims rank 0's rectangle
        with self.assertRaises(ValueError):
            self.check(reported, self.tiling, NY, NX, 2, 2)


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False).result.wasSuccessful() else 1)
