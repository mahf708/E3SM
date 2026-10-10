"""Python-backend test emulator of whole processes: returns, for each output X,
its input X_before if there is one (the value of X before the process ran),
else its input X.
"""


class Identity:
    def infer(self, inputs, outputs):
        for name, out in outputs.items():
            before = name + "_before"
            out[...] = inputs[before] if before in inputs else inputs[name]


def create_emulator(config):
    return Identity()
