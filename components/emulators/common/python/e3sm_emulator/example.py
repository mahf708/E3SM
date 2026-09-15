"""A worked example of the emulator contract, in numpy alone.

Reference for what an emulator has to do, not a model to run for real: the
whole path -- namelist, C++ backend, embedded interpreter, zero-copy numpy
views -- is exercised with nothing installed but numpy.
"""

from __future__ import annotations

import numpy as np

from .context import Context


def build(config: dict, context: Context) -> "ExampleEmulator":
    return ExampleEmulator(config, context)


class ExampleEmulator:
    """Applies an affine map to this rank's columns."""

    def __init__(self, config: dict, context: Context):
        self.context = context
        self.verbose = bool(config.get("verbose", False))
        self.inputs = list(config.get("inputs") or [])
        self.outputs = list(config.get("outputs") or [])

        model_path = config.get("model_path") or ""
        if model_path:
            with np.load(model_path) as weights:
                self.weight = np.asarray(weights["W"], dtype=np.float64)
                self.bias = np.asarray(weights["b"], dtype=np.float64)
        else:
            # No weights: fall back to a scalar affine map.
            self.weight = None
            self.scale = float(config.get("scale", 1.0))
            self.offset = float(config.get("offset", 0.0))

        if self.verbose and context.is_root:
            what = model_path if model_path else f"{self.scale}x + {self.offset}"
            print(
                f"[e3sm_emulator.example] {what}, "
                f"{len(self.inputs)} input(s) -> {len(self.outputs)} output(s)",
                flush=True,
            )

    def infer(self, inputs: dict, outputs: dict) -> None:
        names_in = self.inputs or sorted(inputs)
        names_out = self.outputs or sorted(outputs)

        columns = _column_count(inputs, names_in)
        if columns == 0:
            # A rank with no columns returns early: numpy can't reshape a zero-sized array.
            return

        # Flatten each field to [ncol, k] and lay them side by side: the layout a column network expects.
        packed = np.concatenate(
            [np.asarray(inputs[n]).reshape(columns, -1) for n in names_in], axis=1
        )

        if self.weight is not None:
            y = packed @ self.weight + self.bias
        else:
            y = packed * self.scale + self.offset
        y = np.asarray(y, dtype=np.float64).reshape(columns, -1)

        # Split the result back over the named outputs, in order and width, as E3SM declared them.
        start = 0
        for name in names_out:
            target = outputs[name]
            width = int(np.asarray(target).size // columns)
            stop = start + width
            if stop > y.shape[1]:
                raise ValueError(
                    f"The model returned {y.shape[1]} values per column, which "
                    f"runs out while filling '{name}'. Check `inference.output` "
                    "against the model's actual output width."
                )
            target.reshape(columns, -1)[:] = y[:, start:stop]
            start = stop
        if start != y.shape[1]:
            raise ValueError(
                f"The model returned {y.shape[1]} values per column but the "
                f"declared outputs consume {start}."
            )


def _column_count(fields: dict, names) -> int:
    if not names:
        raise ValueError("No input fields were declared or supplied.")
    counts = {int(np.asarray(fields[n]).shape[0]) for n in names}
    if len(counts) != 1:
        raise ValueError(
            f"Input fields disagree about the column count: {sorted(counts)}."
        )
    return counts.pop()
