"""Smallest possible python emulator: a module-level infer(), no class.

With neither `python_factory` nor `create_emulator` present, the module itself
acts as the emulator.
"""


def infer(inputs, outputs):
    outputs["y"][:] = -inputs["x"]
