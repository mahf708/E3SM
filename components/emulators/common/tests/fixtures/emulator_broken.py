"""Python emulator that fails on purpose, to check error reporting."""


def create_emulator(config):
    if config.get("fail_at") == "create":
        raise ValueError("deliberate failure in create_emulator")
    return BrokenEmulator()


class BrokenEmulator:
    def infer(self, inputs, outputs):
        raise RuntimeError("deliberate failure in infer")
