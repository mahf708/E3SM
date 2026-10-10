"""Python model for the strided and device-memory tests of the Python backend."""
import ctypes

import numpy as np


def create_emulator(config):
    return Model()


def from_cuda_array_interface(obj):
    """A numpy view of what a CUDA array interface describes.

    The tests mark host memory as device memory, so it can be read here.
    """
    cai = obj.__cuda_array_interface__
    assert cai['version'] == 3 and cai['typestr'] == '<f8' and cai['stream'] is None
    ptr, readonly = cai['data']
    shape, strides = cai['shape'], cai['strides']
    span = sum((n - 1) * s for n, s in zip(shape, strides)) // 8 + 1
    buf = (ctypes.c_double * span).from_address(ptr)
    flat = np.frombuffer(buf, dtype=np.float64)
    return np.ndarray(shape, np.float64, buffer=flat, strides=strides), readonly


class Model:
    def infer(self, inputs, outputs):
        x, y = inputs['x'], outputs['y']
        if hasattr(x, '__cuda_array_interface__'):
            x, x_ro = from_cuda_array_interface(x)
            y, y_ro = from_cuda_array_interface(y)
            assert x_ro and not y_ro
        else:
            assert not x.flags.writeable and y.flags.writeable
        y[...] = 2 * x
