"""Tests for the column-local emulator's packing and output splitting.

Torch is faked rather than required, so what is covered is the part that is
ours: laying several named fields out side by side, splitting the answer back
over the declared outputs, and the widths that have to agree for either to be
right.  Loading a real TorchScript module is not covered here and needs torch.
"""

from __future__ import annotations

import contextlib
import sys
import unittest
import unittest.mock as mock

import numpy as np

from e3sm_emulator.context import Context
from e3sm_emulator.generic import GenericEmulator


class _FakeTensor:
    """A numpy array wearing just enough of the torch.Tensor interface."""

    def __init__(self, array):
        self.array = np.asarray(array)

    def to(self, *args, **kwargs):
        return self

    def detach(self):
        return self

    def numpy(self):
        return self.array


class _FakeTorch:
    """The handful of torch entry points GenericEmulator.infer calls."""

    float32 = np.float32
    float64 = np.float64

    @staticmethod
    def no_grad():
        return contextlib.nullcontext()

    @staticmethod
    def from_numpy(array):
        return _FakeTensor(array)


def emulator(model, inputs, outputs) -> GenericEmulator:
    """A GenericEmulator with the torch-dependent construction skipped."""
    built = GenericEmulator.__new__(GenericEmulator)
    built.context = Context()
    built.verbose = False
    built.inputs = inputs
    built.outputs = outputs
    built.dtype = np.float32
    built.device = "cpu"
    built.model = model
    return built


def run(built: GenericEmulator, inputs: dict, outputs: dict) -> None:
    with mock.patch.dict(sys.modules, {"torch": _FakeTorch}):
        built.infer(inputs, outputs)


class TestGenericEmulator(unittest.TestCase):
    def test_fields_are_packed_side_by_side_and_split_back(self):
        # T is one value per column, q is three: the model must see them laid
        # out as [ncol, 4], in the declared order.
        seen = {}

        def model(x):
            seen["packed"] = np.array(x.array, copy=True)
            # Two outputs of width 1 and 2, in the order they are declared.
            return _FakeTensor(np.column_stack([x.array[:, 0] * 2.0,
                                                x.array[:, 1:3] * -1.0]))

        T = np.arange(4.0)
        q = np.arange(12.0).reshape(4, 3)
        dT = np.zeros(4)
        dq = np.zeros((4, 2))

        run(emulator(model, ["T", "q"], ["dT", "dq"]),
            {"T": T, "q": q}, {"dT": dT, "dq": dq})

        self.assertEqual(seen["packed"].shape, (4, 4))
        np.testing.assert_allclose(seen["packed"][:, 0], T)
        np.testing.assert_allclose(seen["packed"][:, 1:], q)
        np.testing.assert_allclose(dT, T * 2.0)
        np.testing.assert_allclose(dq, q[:, :2] * -1.0)

    def test_a_rank_with_no_columns_does_nothing(self):
        # Normal on a large layout, and the reason this is not simply left to
        # numpy: reshape(0, -1) cannot infer the trailing extent, so every
        # packing step would raise.  A column-local model communicates
        # nothing, so there is no collective this rank fails to reach.
        called = []
        built = emulator(lambda x: called.append(x) or x, ["T"], ["dT"])
        run(built, {"T": np.zeros(0)}, {"dT": np.zeros(0)})
        self.assertEqual(called, [], "the model ran on an empty batch")

    def test_a_model_that_returns_too_little_is_named(self):
        built = emulator(lambda x: _FakeTensor(x.array[:, :1]), ["T"],
                         ["dT", "dq"])
        with self.assertRaises(ValueError) as caught:
            run(built, {"T": np.zeros(4)},
                {"dT": np.zeros(4), "dq": np.zeros(4)})
        self.assertIn("'dq'", str(caught.exception))

    def test_a_model_that_returns_too_much_is_caught(self):
        # Silently dropping the tail would leave a field the namelist never
        # mentioned quietly unused.
        built = emulator(lambda x: _FakeTensor(np.zeros((4, 3))), ["T"], ["dT"])
        with self.assertRaises(ValueError) as caught:
            run(built, {"T": np.zeros(4)}, {"dT": np.zeros(4)})
        self.assertIn("consume 1", str(caught.exception))

    def test_inputs_that_disagree_about_the_columns_are_refused(self):
        built = emulator(lambda x: x, ["T", "q"], ["dT"])
        with self.assertRaises(ValueError) as caught:
            run(built, {"T": np.zeros(4), "q": np.zeros(3)},
                {"dT": np.zeros(4)})
        self.assertIn("disagree about the column count", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
