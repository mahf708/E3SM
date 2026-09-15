"""TorchScript adapter for the component's ordered NCHW tensors."""

import numpy as np
import torch


def step_seed(seed, step):
    mask = (1 << 64) - 1

    def mix(z):
        z = (z + 0x9E3779B97F4A7C15) & mask
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        return z ^ (z >> 31)

    return mix(seed ^ mix(step))


def create_emulator(config):
    return TorchScript(config)


class TorchScript:
    def __init__(self, config):
        self.device = torch.device(config.get("device", "cpu"))
        dtype = config.get("dtype", "float32")
        if dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        self.dtype = getattr(torch, dtype)
        self.optimize = str(config.get("jit_optimize", "false")).lower() in ("true", "1", "yes", "on")
        seed = str(config.get("seed", ""))
        if seed and (not seed.isascii() or not seed.isdigit() or int(seed) >= 1 << 64):
            raise ValueError("seed must be a uint64 integer")
        self.seed = int(seed) if seed else None
        self.step = -1
        threads = int(config.get("num_threads", 0))
        if threads < 0:
            raise ValueError("num_threads must be nonnegative")
        if threads:
            torch.set_num_threads(threads)
        self.model = torch.jit.load(config["model_path"], map_location=self.device).eval()

    def set_step(self, step):
        self.step = int(step)

    def infer(self, inputs, outputs):
        if self.seed is not None:
            if self.step < 0:
                raise ValueError("set_step is required with a seed")
            torch.manual_seed(step_seed(self.seed, self.step))
        # Own the inputs: Torch cannot enforce NumPy's read-only flag.
        args = [torch.tensor(a, dtype=self.dtype, device=self.device) for a in inputs.values()]
        with torch.no_grad(), torch.jit.optimized_execution(self.optimize):
            result = self.model(*args)
        results = [result] if isinstance(result, torch.Tensor) else list(result)
        if len(results) != len(outputs):
            raise ValueError("checkpoint output count differs from the declared tensors")
        staged = []
        for value, target in zip(results, outputs.values()):
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != target.shape:
                raise ValueError("checkpoint output shape differs from the declared tensor")
            array = value.detach().to(device="cpu", dtype=torch.float64).numpy()
            if not np.isfinite(array).all():
                raise ValueError("checkpoint produced non-finite output")
            staged.append(array)
        for value, target in zip(staged, outputs.values()):
            np.copyto(target, value)

    def finalize(self):
        self.model = None
