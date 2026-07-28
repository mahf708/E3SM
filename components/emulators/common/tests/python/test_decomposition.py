"""Tests for the column <-> grid decomposition and the collectives under it.

Run directly (``python3 test_decomposition.py``) or under ctest.  Needs numpy
and nothing else: the multi-rank cases run N logical ranks as threads over
the same Comm interface the real implementation satisfies.
"""

from __future__ import annotations

import sys
import unittest

import numpy as np

from e3sm_emulator.comm import SerialComm, TorchComm, run_where
from e3sm_emulator.decomposition import (
    PermutationExchange,
    Tiling,
    cell_indices,
    split_bounds,
)
from fake_cluster import run_ranks

NY, NX = 6, 8


def round_robin_gids(rank: int, size: int, total: int = NY * NX) -> np.ndarray:
    """A deliberately awkward decomposition: strided, so no rank owns a
    rectangle and nothing lines up with the model's tiling by luck."""
    return np.arange(1 + rank, total + 1, size, dtype=np.int64)


def field_from_gids(gids: np.ndarray) -> np.ndarray:
    """A value that identifies its own column, so a misplaced one shows."""
    return (gids.astype(np.float64) * 10.0).reshape(-1, 1)


class TestIndexing(unittest.TestCase):
    def test_split_bounds_matches_torch_harmonics(self):
        # torch_harmonics' compute_split_shapes gives the remainder to the
        # LOW-numbered ranks, which is also numpy.array_split's convention.
        # Getting this backwards is invisible on an even split and silently
        # wrong on every uneven one.
        for n, parts in [(10, 3), (6, 4), (8, 8), (97, 5)]:
            base, remainder = divmod(n, parts)
            expected = [base + 1] * remainder + [base] * (parts - remainder)
            sizes = np.diff(split_bounds(n, parts)).tolist()
            self.assertEqual(sizes, expected, f"{n} over {parts}")
            self.assertEqual(
                sizes, [len(c) for c in np.array_split(np.arange(n), parts)]
            )

    def test_split_bounds_refuses_empty_tiles(self):
        # A model with a spatial receptive field cannot work with one.
        with self.assertRaises(ValueError):
            split_bounds(3, 4)

    def test_cell_indices_round_trip(self):
        gids = np.arange(1, NY * NX + 1)
        j, i = cell_indices(gids, NY, NX)
        np.testing.assert_array_equal(j * NX + i + 1, gids)

        # The other numbering convention transposes the sweep.
        j, i = cell_indices(gids, NY, NX, lon_fastest=False)
        np.testing.assert_array_equal(i * NY + j + 1, gids)

    def test_cell_indices_rejects_a_foreign_grid(self):
        with self.assertRaises(ValueError) as caught:
            cell_indices(np.array([NY * NX + 1]), NY, NX)
        self.assertIn("same globe", str(caught.exception))

    def test_a_tiling_partitions_the_grid_exactly(self):
        for h, w in [(1, 1), (2, 2), (2, 3), (3, 1)]:
            tiling = Tiling(NY, NX, h, w)
            covered = np.zeros((NY, NX), dtype=int)
            for rank in range(h * w):
                j0, i0 = tiling.tile_origin(rank)
                nj, ni = tiling.tile_shape(rank)
                covered[j0 : j0 + nj, i0 : i0 + ni] += 1
            np.testing.assert_array_equal(covered, 1, f"{h}x{w} does not tile")

    def test_ranks_past_the_mesh_own_nothing(self):
        # 1x1 on many ranks: one rank holds the globe, the rest only take
        # part in the exchange.
        self.assertEqual(Tiling(NY, NX, 1, 1).tile_shape(3), (0, 0))

    def test_a_foreign_partition_must_still_cover_the_grid(self):
        # ACE's partition is authoritative, but it still has to be a
        # partition: one that starts late, ends early, runs backwards or has
        # the wrong number of parts would leave a field with holes in it, and
        # a model consumes that happily.
        Tiling.from_bounds(NY, NX, 2, 1, [0, 2, NY], [0, NX])  # fine
        for bad in (
            [0, 2, NY - 1],  # ends early
            [1, 2, NY],  # starts late
            [0, NY, NY],  # a part owning nothing
            [0, 2, 4, NY],  # more parts than the mesh has ranks
        ):
            with self.assertRaises(ValueError, msg=f"{bad} was accepted"):
                Tiling.from_bounds(NY, NX, 2, 1, bad, [0, NX])

    def test_a_tiling_knows_when_it_is_not_the_even_split(self):
        self.assertTrue(Tiling(NY, NX, 2, 2).agrees_with_even_split())
        self.assertFalse(
            Tiling.from_bounds(NY, NX, 2, 1, [0, 2, NY], [0, NX])
            .agrees_with_even_split()
        )


class TestExchange(unittest.TestCase):
    """Round trips, on one rank and on many."""

    def test_serial_round_trip(self):
        gids = np.arange(1, NY * NX + 1)
        exchange = PermutationExchange(SerialComm(), gids, Tiling(NY, NX))
        values = field_from_gids(gids)

        tile = exchange.to_tile(values)
        self.assertEqual(tile.shape, (NY, NX, 1))
        for j in range(NY):
            for i in range(NX):
                self.assertEqual(tile[j, i, 0], (j * NX + i + 1) * 10.0)
        np.testing.assert_allclose(exchange.to_columns(tile), values)

    def test_a_grid_mismatch_is_loud(self):
        gids = np.arange(1, NY * NX)  # one column short
        with self.assertRaises(ValueError) as caught:
            PermutationExchange(SerialComm(), gids, Tiling(NY, NX))
        self.assertIn("same globe", str(caught.exception))

    def _round_trip(self, size: int, h: int, w: int):
        reference = Tiling(NY, NX, h, w)

        def body(comm, rank):
            gids = round_robin_gids(rank, size)
            exchange = PermutationExchange(comm, gids, Tiling(NY, NX, h, w))
            values = field_from_gids(gids)
            tile = exchange.to_tile(values)

            # Every cell this rank received must carry the value of the
            # column that actually lives there.
            nj, ni = exchange.tile_shape
            j0, i0 = reference.tile_origin(rank) if rank < h * w else (0, 0)
            for j in range(nj):
                for i in range(ni):
                    gid = (j0 + j) * NX + (i0 + i) + 1
                    assert tile[j, i, 0] == gid * 10.0, (
                        f"rank {rank} cell ({j},{i}) holds {tile[j, i, 0]}, "
                        f"expected column {gid}"
                    )

            np.testing.assert_allclose(exchange.to_columns(tile), values)
            return nj * ni

        self.assertEqual(
            sum(run_ranks(size, body)), NY * NX, "the tiles do not cover the globe"
        )

    def test_one_rank_gathers_the_globe(self):
        # 1x1 on four ranks: rank 0 assembles everything, the rest hand their
        # columns over and take the answers back.
        self._round_trip(size=4, h=1, w=1)

    def test_a_spatial_mesh(self):
        self._round_trip(size=4, h=2, w=2)

    def test_a_lopsided_mesh(self):
        # 6 rows over 2 and 8 columns over 3: the uneven split is where a
        # partition convention that merely looks balanced goes wrong.
        self._round_trip(size=6, h=2, w=3)


class TestAgreement(unittest.TestCase):
    """Every rank stops, or none does -- never one rank leaving the others."""

    def test_nobody_unhappy_is_a_no_op(self):
        self.assertEqual(run_ranks(3, lambda comm, rank: comm.agree("")), [None] * 3)

    def test_one_bad_rank_stops_every_rank(self):
        def body(comm, rank):
            comm.agree("the grid is wrong" if rank == 1 else "")

        with self.assertRaises(RuntimeError) as caught:
            run_ranks(3, body)
        self.assertIn("the grid is wrong", str(caught.exception))

    def test_a_healthy_rank_learns_the_real_reason(self):
        # A bare "some other rank failed" would send you to the wrong log.
        def body(comm, rank):
            try:
                comm.agree("checkpoint missing" if rank == 2 else "")
            except RuntimeError as exc:
                return str(exc)
            return None

        messages = run_ranks(3, body)
        for rank, message in enumerate(messages):
            self.assertIsNotNone(message, f"rank {rank} did not stop")
            self.assertIn("checkpoint missing", message)
        self.assertIn("rank 2", messages[0])

    def test_the_failing_rank_keeps_its_own_exception(self):
        # Type and traceback intact where it actually happened.
        def body(comm, rank):
            error = FileNotFoundError("no such checkpoint") if rank == 1 else None
            try:
                comm.agree("rank 1: no such checkpoint" if error else "", error)
            except BaseException as exc:
                return type(exc).__name__
            return None

        kinds = run_ranks(3, body)
        self.assertEqual(kinds[1], "FileNotFoundError")
        self.assertEqual(kinds[0], "RuntimeError")

    def test_a_rank_local_gid_error_is_agreed_not_hung(self):
        # cell_indices() runs before the exchange's own collectives, so a bad
        # global id on one rank would otherwise strand the others in them.
        def body(comm, rank):
            gids = round_robin_gids(rank, 3)
            if rank == 1:
                gids = gids + NY * NX  # off this globe entirely
            return PermutationExchange(comm, gids, Tiling(NY, NX))

        with self.assertRaises(RuntimeError) as caught:
            run_ranks(3, body)
        self.assertIn("rank 1", str(caught.exception))


class TestOwnerOnlyWork(unittest.TestCase):
    """The shape of every step: gather -> owner computes -> scatter."""

    def _step(self, size, model):
        def body(comm, rank):
            gids = round_robin_gids(rank, size)
            exchange = PermutationExchange(comm, gids, Tiling(NY, NX, 1, 1))
            owns = exchange.tile_shape[0] > 0

            tile = exchange.to_tile(field_from_gids(gids))  # collective
            result = run_where(comm, owns, lambda: model(tile, exchange.tile_shape))
            if result is None:
                # This rank owns no part of the globe.  It still has to take
                # part in the exchange, because infer() is collective.
                result = np.empty((*exchange.tile_shape, 1), dtype=np.float64)
            return exchange.to_columns(result)  # collective

        return run_ranks(size, body)

    def test_a_clean_step_round_trips(self):
        results = self._step(4, lambda tile, shape: tile * 2.0)
        for rank, columns in enumerate(results):
            expected = field_from_gids(round_robin_gids(rank, 4)) * 2.0
            np.testing.assert_allclose(columns, expected)

    def test_an_owner_failure_stops_every_rank(self):
        # Without the agreement the three non-owner ranks would enter
        # to_columns() and block there forever.  The fake cluster reports that
        # as RankDivergence rather than rescuing it, so this really does fail
        # if run_where stops agreeing -- and RuntimeError rather than
        # RankDivergence is what says they failed together.
        def boom(tile, shape):
            raise RuntimeError("CUDA out of memory")

        with self.assertRaises(RuntimeError) as caught:
            self._step(4, boom)
        self.assertIn("CUDA out of memory", str(caught.exception))

    def test_a_wrong_sized_model_output_stops_every_rank(self):
        # A model can return a correctly shaped-looking tile of the wrong
        # size; caught inside the guard that is an error everywhere, caught in
        # the scatter it is a hang.  ValueError, not RankDivergence: the owner
        # failed *with* the others rather than ahead of them, and it kept its
        # own exception type on the way out.
        with self.assertRaises(ValueError) as caught:
            self._step(4, lambda tile, shape: np.zeros((shape[0], shape[1] + 1, 1)))
        self.assertIn("rows of tile values", str(caught.exception))

    def test_a_wrong_length_input_stops_every_rank(self):
        # The row-count check lives inside the exchange, immediately before
        # the all-to-all, so it has to be settled collectively too.
        def body(comm, rank):
            gids = round_robin_gids(rank, 3)
            exchange = PermutationExchange(comm, gids, Tiling(NY, NX))
            values = field_from_gids(gids)
            if rank == 2:
                values = values[:-1]  # one column short, on one rank only
            return exchange.to_tile(values)

        with self.assertRaises(RuntimeError) as caught:
            run_ranks(3, body)
        self.assertIn("rows of column values", str(caught.exception))
        self.assertIn("rank 2", str(caught.exception))


class TestRestartSchema(unittest.TestCase):
    """A field name list is data, not a diagnostic: it must survive intact."""

    def test_a_long_schema_is_not_truncated(self):
        # Silently dropping a state variable because the name list ran past a
        # fixed buffer would lose part of the atmosphere at every restart.
        names = [f"prognostic_field_number_{i:04d}" for i in range(200)]
        schema = "\n".join(names)
        self.assertGreater(len(schema.encode()), 2048)

        def body(comm, rank):
            announced = comm.allgather_text(schema if rank == 0 else "")
            return sorted({n for b in announced for n in b.split("\n") if n})

        for got in run_ranks(3, body):
            self.assertEqual(got, sorted(names))


class _FakeTensor:
    """A numpy array wearing just enough of the torch.Tensor interface."""

    def __init__(self, array):
        self.array = np.asarray(array)

    @property
    def dtype(self):
        return self.array.dtype

    def __setitem__(self, index, value):
        self.array[index] = value

    def tolist(self):
        return self.array.tolist()

    def item(self):
        return self.array.reshape(-1)[0].item()

    def numpy(self):
        return self.array


class _FakeTorch:
    """The handful of torch entry points TorchComm actually calls."""

    int64 = np.int64

    @staticmethod
    def zeros(n, dtype=None):
        return _FakeTensor(np.zeros(n, dtype=dtype or np.int64))

    @staticmethod
    def tensor(values, dtype=None):
        return _FakeTensor(np.asarray(values, dtype=dtype or np.int64))

    @staticmethod
    def from_numpy(array):
        return _FakeTensor(array)

    @staticmethod
    def empty_like(tensor):
        return _FakeTensor(np.zeros_like(tensor.array))


class _RecordingDist:
    """torch.distributed for a two-rank world in which rank 0 has failed.

    Complete enough for TorchComm's length-then-payload gather to run all the
    way through and produce the right strings: a stub that threw part way
    would let the test claim it exercised the failure path when it had only
    reached the first step of it.  This rank is rank 1, and healthy.
    """

    class ReduceOp:
        MAX = "max"

    def __init__(self, peer_message: str):
        self.peer = peer_message.encode("utf-8")
        self.calls = []

    def all_reduce(self, tensor, op=None, group=None):
        self.calls.append("all_reduce")
        if op == self.ReduceOp.MAX:
            # any_true: rank 0 failed, so the maximum is 1.
            tensor[0] = max(int(tensor.item()), 1 if self.peer else 0)
        else:
            # allgather sums the per-rank row counts; rank 0 contributes as
            # many rows as we do.
            tensor[0] = tensor.array[0] + int(tensor.array[1])

    def all_gather(self, out_list, tensor, group=None):
        self.calls.append("all_gather")
        block = tensor.array
        out_list[1] = _FakeTensor(np.array(block, copy=True))  # ours
        peer = np.zeros_like(block)
        if block.dtype == np.int64:
            peer[0, 0] = len(self.peer)  # the length gather
        else:
            payload = np.frombuffer(self.peer, dtype=np.uint8)
            peer[: len(payload), 0] = payload  # the payload gather
        out_list[0] = _FakeTensor(peer)


class TestAgreementCost(unittest.TestCase):
    """What a working run pays, every step, for the safety above.

    Counted at the *torch* level, which is where it is actually paid: one
    Comm.allgather is an all_reduce (to exchange the block sizes) *plus* an
    all_gather, so counting Comm-level calls would show no difference at all
    and pin nothing.
    """

    def _torch_comm(self, peer_message=""):
        recorder = _RecordingDist(peer_message)
        comm = TorchComm.__new__(TorchComm)  # bypass group creation
        comm._dist = recorder
        comm._group = None
        comm._owns_group = False
        comm.rank, comm.size = 1, 2
        return comm, recorder

    def test_a_healthy_agreement_is_one_reduction(self):
        import unittest.mock as mock

        comm, recorder = self._torch_comm()
        with mock.patch.dict(sys.modules, {"torch": _FakeTorch}):
            comm.agree("")
        self.assertEqual(recorder.calls, ["all_reduce"])

    def test_a_failing_agreement_gathers_the_diagnostic_and_raises(self):
        import unittest.mock as mock

        comm, recorder = self._torch_comm(peer_message="something broke")
        with mock.patch.dict(sys.modules, {"torch": _FakeTorch}):
            with self.assertRaises(RuntimeError) as caught:
                comm.agree("")

        message = str(caught.exception)
        self.assertIn("rank 0", message)
        self.assertIn("something broke", message)
        # One reduction to discover the failure, then the length gather and
        # the payload gather.
        self.assertEqual(recorder.calls[0], "all_reduce")
        self.assertEqual(recorder.calls.count("all_gather"), 2)


if __name__ == "__main__":
    unittest.main()
