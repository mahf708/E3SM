"""Python-backend test emulator: returns zero for every output.

With physics: skip on rain sedimentation tendencies, rain does not sediment.
"""


class Zero:
    def infer(self, inputs, outputs):
        for out in outputs.values():
            out[...] = 0


def create_emulator(config):
    return Zero()
