"""Return style python emulator used by the python-backend test.

`infer(inputs)` takes one argument and hands back its results, which the
backend copies into the destination tensors (one copy per output).  This is
the more convenient style, and the one that a model written without E3SM in
mind will already have.
"""

import numpy as np


def create_emulator(config):
    return SumEmulator(config)


class SumEmulator:
    """Two inputs of different precision in, two outputs out."""

    def __init__(self, config):
        self.verbose = bool(config.get("verbose", False))

    def infer(self, inputs):
        total = inputs["T"].sum(axis=1) + inputs["ps"]
        return {
            # float64 result for a float32 destination: the backend converts.
            "total": total.astype(np.float64),
            "doubled": 2.0 * inputs["T"],
        }
