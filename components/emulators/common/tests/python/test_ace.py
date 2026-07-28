"""Tests for the ACE adapter's decisions, without ACE, torch or a checkpoint.

What can be tested here is everything that happens *before* a weight is read:
mode selection, the mesh ordering, the output contract and the restart
export.  What cannot is whether a real checkpoint gives the right answer at
several ranks, which needs a machine and is the decisive next step.
"""

from __future__ import annotations

import unittest

import numpy as np

from e3sm_emulator.ace import (
    SPATIAL_OVERRIDE,
    check_mesh_ordering,
    empty_output_tiles,
    resolve_mode,
    stack_output_tiles,
)
from e3sm_emulator.context import Context
from e3sm_emulator.decomposition import PermutationExchange, Tiling
from fake_cluster import run_ranks

NY, NX = 6, 8


def context(world_size: int) -> Context:
    return Context(rank=0, world_size=world_size, ny=NY, nx=NX)


class TestModeResolution(unittest.TestCase):
    def test_serial_runs_undistributed(self):
        self.assertEqual(resolve_mode({}, context(1)), ("single", 1, 1))

    def test_auto_will_not_invent_a_mesh(self):
        # Which factorization is right depends on the model's transforms and
        # on the machine; guessing is how a run ends up slow or wrong.
        self.assertEqual(resolve_mode({}, context(8)), ("single", 1, 1))

    def test_auto_uses_a_declared_mesh(self):
        config = {"ace_h": 2, "ace_w": 4, SPATIAL_OVERRIDE: "true"}
        self.assertEqual(resolve_mode(config, context(8)), ("spatial", 2, 4))

    def test_spatial_is_gated_on_an_explicit_opt_in(self):
        config = {"ace_mode": "spatial", "ace_h": 2, "ace_w": 4}
        with self.assertRaises(ValueError) as caught:
            resolve_mode(config, context(8))
        message = str(caught.exception)
        # The gate has to say what it is protecting against, or it reads as
        # bureaucracy and gets switched off.
        self.assertIn("global transforms", message)
        self.assertIn(SPATIAL_OVERRIDE, message)

    def test_spatial_needs_every_rank(self):
        config = {"ace_mode": "spatial", "ace_h": 2, "ace_w": 2,
                  SPATIAL_OVERRIDE: "true"}
        with self.assertRaises(ValueError) as caught:
            resolve_mode(config, context(8))
        self.assertIn("rank count", str(caught.exception))

    def test_spatial_needs_a_mesh(self):
        with self.assertRaises(ValueError):
            resolve_mode({"ace_mode": "spatial"}, context(8))

    def test_torch_distributed_is_refused_with_a_reason(self):
        with self.assertRaises(ValueError) as caught:
            resolve_mode({"ace_mode": "torch_distributed"}, context(8))
        self.assertIn("one globe", str(caught.exception))


class TestOutputTiles(unittest.TestCase):
    def test_the_named_outputs_are_stacked_in_order(self):
        produced = {"b": np.full((2, 3), 2.0), "a": np.full((2, 3), 1.0)}
        stacked = stack_output_tiles(produced, ["a", "b"], (2, 3))
        self.assertEqual(stacked.shape, (2, 3, 2))
        self.assertTrue((stacked[..., 0] == 1.0).all())
        self.assertTrue((stacked[..., 1] == 2.0).all())

    def test_a_missing_output_is_named(self):
        with self.assertRaises(ValueError) as caught:
            stack_output_tiles({"a": np.zeros((2, 3))}, ["a", "b"], (2, 3))
        self.assertIn("'b'", str(caught.exception))

    def test_a_wrong_sized_output_is_caught_before_the_scatter(self):
        # .numpy()[0] does not check (nj, ni), so without this a model that
        # returned a correctly named field of the wrong size would fail alone
        # inside the scatter, with every other rank already waiting.
        with self.assertRaises(ValueError) as caught:
            stack_output_tiles({"a": np.zeros((2, 4))}, ["a"], (2, 3))
        self.assertIn("owns a 2x3 tile", str(caught.exception))

    def test_a_rank_with_no_tile_contributes_a_shaped_nothing(self):
        self.assertEqual(empty_output_tiles(["a", "b"], (0, 0)).shape, (0, 0, 2))


class TestMeshOrdering(unittest.TestCase):
    """The rank ordering, checked by coordinates rather than by shape."""

    def _reported(self, tiling: Tiling, h: int, w: int, order=None):
        order = order or list(range(h * w))
        rows = []
        for rank in order:
            j0, i0 = tiling.tile_origin(rank)
            nj, ni = tiling.tile_shape(rank)
            rows.append([j0, j0 + nj, i0, i0 + ni])
        return np.array(rows, dtype=np.int64)

    def test_the_assumed_order_passes(self):
        tiling = Tiling(NY, NX, 2, 2)
        check_mesh_ordering(self._reported(tiling, 2, 2), tiling, 2, 2)

    def test_a_permuted_rank_order_is_caught(self):
        # Every tile is 3x4 here, so comparing shapes would pass while each
        # rank worked on somebody else's piece of the planet -- and a global
        # model fed a rotated globe returns a plausible field, not an error.
        tiling = Tiling(NY, NX, 2, 2)
        reported = self._reported(tiling, 2, 2, order=[1, 0, 3, 2])
        with self.assertRaises(ValueError) as caught:
            check_mesh_ordering(reported, tiling, 2, 2)
        self.assertIn("wrong part of the globe", str(caught.exception))

    def test_overlapping_rectangles_are_caught(self):
        tiling = Tiling(NY, NX, 2, 1)
        reported = np.array([[0, 4, 0, NX], [2, NY, 0, NX]], dtype=np.int64)
        with self.assertRaises(ValueError) as caught:
            check_mesh_ordering(reported, tiling, 2, 1)
        self.assertIn("rows 0:4", str(caught.exception))


class TestRestartExport(unittest.TestCase):
    """Restart state lives on one rank but has to come back on all of them."""

    def test_export_when_only_the_owner_holds_state(self):
        # The deadlock this guards against: non-owners see an empty state,
        # return early, and leave the owner alone inside to_columns().
        size = 3

        def body(comm, rank):
            gids = np.arange(1 + rank, NY * NX + 1, size, dtype=np.int64)
            exchange = PermutationExchange(comm, gids, Tiling(NY, NX))
            owns = exchange.tile_shape[0] > 0
            state = {"air_temperature": 1.0} if owns else {}

            # The name list is agreed before anybody touches the exchange.
            announced = comm.allgather_text("\n".join(sorted(state)) if owns else "")
            names = sorted({n for b in announced for n in b.split("\n") if n})
            assert names == ["air_temperature"], f"rank {rank} saw {names}"

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

        for rank, exported in enumerate(run_ranks(size, body)):
            self.assertEqual(sorted(exported), ["air_temperature"])
            # Every rank gets its own columns back, owner or not.
            self.assertEqual(
                exported["air_temperature"].size,
                np.arange(1 + rank, NY * NX + 1, size).size,
            )
            np.testing.assert_allclose(exported["air_temperature"], 1.0)


if __name__ == "__main__":
    unittest.main()
