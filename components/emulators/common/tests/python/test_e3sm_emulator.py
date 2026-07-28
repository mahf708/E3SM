"""Tests for the python emulator package.

Run directly (``python3 test_e3sm_emulator.py``) or under ctest.  Needs numpy
and nothing else: no MPI, no launcher, no torch, no checkpoint.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from e3sm_emulator.bridge import create_emulator
from e3sm_emulator.context import Context


class TestContext(unittest.TestCase):
    def test_from_dict_matches_the_cxx_payload(self):
        # The keys here are exactly what python_inference_backend.cpp writes.
        context = Context.from_dict(
            {
                "rank": 2,
                "world_size": 4,
                "local_rank": 2,
                "local_size": 4,
                "node_name": "nid001",
                "master_addr": "nid000",
                "master_port": 41234,
                "nx": 8,
                "ny": 6,
                "num_global_cols": 48,
                "col_gids": np.array([1, 5, 9], dtype=np.int32),
                "lat": np.array([-45.0, 0.0, 45.0]),
                "lon": np.array([0.0, 120.0, 240.0]),
            }
        )
        self.assertEqual(context.num_local_cols, 3)
        self.assertFalse(context.is_root)
        self.assertEqual(context.col_gids.dtype, np.int64)
        self.assertIn("3 of 48 columns", context.describe())

    def test_defaults_describe_a_serial_run(self):
        context = Context.from_dict({})
        self.assertTrue(context.is_root)
        self.assertEqual(context.world_size, 1)
        self.assertEqual(context.num_local_cols, 0)

    def test_export_publishes_the_component_not_the_job(self):
        # What the coupled job thinks: 512 ranks, this one number 300.  A
        # process group built from these would wait for ocean and land ranks.
        os.environ["SLURM_PROCID"] = "300"
        os.environ["SLURM_NTASKS"] = "512"

        Context(rank=1, world_size=4, master_addr="nid000", master_port=41234).export(
            device_ordinal=0
        )

        self.assertEqual(os.environ["RANK"], "1")
        self.assertEqual(os.environ["WORLD_SIZE"], "4")
        self.assertEqual(os.environ["MASTER_ADDR"], "nid000")
        self.assertEqual(os.environ["MASTER_PORT"], "41234")
        # LOCAL_RANK is a device ordinal, not a rank.
        self.assertEqual(os.environ["LOCAL_RANK"], "0")
        # And ACE must not fall back to reading SLURM_* instead.
        self.assertEqual(os.environ["FME_USE_SRUN"], "0")

    def test_a_missing_port_still_yields_a_usable_rendezvous(self):
        Context(rank=0, world_size=2).export()
        self.assertEqual(os.environ["MASTER_ADDR"], "127.0.0.1")
        self.assertEqual(os.environ["MASTER_PORT"], "29500")


class TestDeviceContract(unittest.TestCase):
    """One visible device per rank, or say which one -- never guess."""

    def setUp(self):
        self.context = Context(rank=3, world_size=8, local_rank=3, local_size=4)

    def test_one_visible_device_means_ordinal_zero(self):
        # The recommended binding: --gpus-per-task=1. The component-local rank
        # would be 3 here, which is not a device this process can see.
        self.assertEqual(self.context.device_ordinal(visible_devices=1), 0)

    def test_ambiguous_binding_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.context.device_ordinal(visible_devices=4)
        self.assertIn("not ours to decide", str(caught.exception))

    def test_being_alone_in_this_component_is_not_ownership(self):
        # The ocean and land ranks sharing this node are invisible to us, and
        # one of them may already hold device 0.
        alone = Context(rank=0, world_size=1, local_rank=0, local_size=1)
        with self.assertRaises(ValueError):
            alone.device_ordinal(visible_devices=4)

    def test_an_explicit_device_is_honoured_and_range_checked(self):
        self.assertEqual(self.context.device_ordinal(2, visible_devices=4), 2)
        with self.assertRaises(ValueError):
            self.context.device_ordinal(9, visible_devices=4)

    def test_no_gpu_is_not_an_error(self):
        self.assertEqual(self.context.device_ordinal(visible_devices=0), 0)


class TestBridge(unittest.TestCase):
    def test_unknown_emulator_names_the_alternatives(self):
        with self.assertRaises(ValueError) as caught:
            create_emulator({"emulator": "nope"})
        message = str(caught.exception)
        self.assertIn("ace", message)
        self.assertIn("generic", message)
        self.assertIn("python_module", message)


if __name__ == "__main__":
    unittest.main()
