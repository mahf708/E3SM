import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from e3sm_emulator.torchscript import TorchScript


class Noisy(torch.nn.Module):
    def forward(self, x):
        return x + torch.randn_like(x)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        path = Path(self.directory.name) / 'model.pt'
        torch.jit.script(Noisy()).save(str(path))
        self.config = {'model_path': str(path), 'seed': '2026'}

    def test_restored_step_repeats_noise(self):
        x = np.ones((1, 2, 3, 4))
        x.flags.writeable = False
        y = np.empty_like(x)
        model = TorchScript(self.config)
        model.set_step(19)
        model.infer({'x': x}, {'y': y})
        expected = y.copy()
        model.set_step(20)
        model.infer({'x': x}, {'y': y})
        self.assertFalse(np.array_equal(y, expected))
        restored = TorchScript(self.config)
        restored.set_step(19)
        restored.infer({'x': x}, {'y': y})
        np.testing.assert_array_equal(y, expected)
        np.testing.assert_array_equal(x, np.ones_like(x))

    def test_seed_requires_step(self):
        model = TorchScript(self.config)
        with self.assertRaisesRegex(ValueError, 'set_step'):
            model.infer({'x': np.ones(2)}, {'y': np.empty(2)})

    def test_shape_mismatch_does_not_write_output(self):
        model = TorchScript(self.config)
        model.set_step(1)
        y = np.full(3, -999.)
        with self.assertRaisesRegex(ValueError, 'shape'):
            model.infer({'x': np.ones(2)}, {'y': y})
        np.testing.assert_array_equal(y, [-999.] * 3)

    def test_invalid_seed(self):
        with self.assertRaisesRegex(ValueError, 'uint64'):
            TorchScript(dict(self.config, seed='-1'))


if __name__ == '__main__':
    unittest.main()
