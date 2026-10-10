#!/usr/bin/env python3
"""Write a TorchScript module, for the libtorch backend, that returns its inputs.

  make_identity_torchscript.py OUTFILE NUM_INPUTS
"""
import sys

import torch


class Identity(torch.nn.Module):
    # The libtorch backend passes each input tensor as a positional argument,
    # and expects a tuple of output tensors back
    def forward(self, *args):
        return tuple(a.clone() for a in args)


if __name__ == '__main__':
    out_file, n = sys.argv[1], int(sys.argv[2])
    example = tuple(torch.zeros(2, 3, dtype=torch.float64) for _ in range(n))
    torch.jit.trace(Identity().eval(), example).save(out_file)
