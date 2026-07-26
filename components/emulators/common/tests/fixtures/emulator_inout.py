"""In-place style python emulator used by the python-backend test.

`infer(inputs, outputs)` receives dicts of numpy arrays.  The arrays are
zero-copy views of the caller's memory, so writing into `outputs` writes
directly into the C++/Fortran field: this is the no-copy path.

The arrays are only valid for the duration of the call — never keep a
reference to them past `infer`.
"""


def create_emulator(config):
    """Factory picked up by convention (no `python_factory` option needed)."""
    return AffineEmulator(config)


class AffineEmulator:
    """Stands in for a trained model: dT = scale * T + offset."""

    def __init__(self, config):
        self.scale = float(config.get("scale", 2.0))
        self.offset = float(config.get("offset", 1.0))
        self.model_path = config.get("model_path", "")
        self.calls = 0
        self.finalized = False

    def infer(self, inputs, outputs):
        self.calls += 1
        # `[:]` writes through the view; `outputs["dT"] = ...` would only
        # rebind the local name and lose the result.
        outputs["dT"][:] = self.scale * inputs["T"] + self.offset

    def finalize(self):
        self.finalized = True
