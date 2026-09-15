"""Deterministic ACE-shaped test model. No climate skill is implied."""
import numpy as np


def create_emulator(config):
    return AtmosphereFixture()


class AtmosphereFixture:
    def infer(self, inputs, outputs):
        x = inputs['inputs']
        y = outputs['outputs']
        y[:, :34] = x[:, 5:39]
        y[:, 0] += 10.0
        y[:, 2:10] += 0.125
        flux = np.array([80, 20, 1e-5, 350, 240, 300, 150, 30, 100, 0])
        y[:, 34:] = flux[None, :, None, None]
