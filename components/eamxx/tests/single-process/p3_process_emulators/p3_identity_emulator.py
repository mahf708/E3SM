"""Python-backend test emulator: returns its inputs, which are P3 process rates.

Run on all the rates (or any subset), P3 must give the same answers as without it.
"""


class Identity:
    def infer(self, inputs, outputs):
        for name, out in outputs.items():
            out[...] = inputs[name]


def create_emulator(config):
    return Identity()
