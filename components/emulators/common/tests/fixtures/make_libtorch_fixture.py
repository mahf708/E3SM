#!/usr/bin/env python3
"""Write the TorchScript fixtures the libtorch backend test loads: make_libtorch_fixture.py OUTDIR."""

import os
import sys

import torch


class Affine(torch.nn.Module):
    """The arithmetic the test checks: exact in float32, obvious in a log."""

    def forward(self, x):
        return 2.0 * x + 1.0


class TwoHeads(torch.nn.Module):
    """A module with a diagnostic head, i.e. a tuple return."""

    def forward(self, x):
        return 2.0 * x + 1.0, x - 1.0


class ChannelWeights(torch.nn.Module):
    """Fixed to 3 channels, so a wrong shape is an error inside forward().

    A real emulator has this property everywhere (every conv fixes its input
    channels); an elementwise fixture does not, and would happily accept a
    transposed field.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("w", torch.tensor([1.0, 10.0, 100.0]))

    def forward(self, x):
        return (x * self.w.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)


class Noise(torch.nn.Module):
    """A stochastic model in miniature: draws on every forward pass."""

    def forward(self, x):
        return x + torch.randn_like(x)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: make_libtorch_fixture.py OUTDIR")
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)

    example = torch.arange(1.0, 25.0, dtype=torch.float32).reshape(1, 3, 2, 4)

    for name, module in (
        ("libtorch_affine.pt", Affine()),
        ("libtorch_tuple.pt", TwoHeads()),
        ("libtorch_channels.pt", ChannelWeights()),
    ):
        traced = torch.jit.trace(module.eval(), example)
        path = os.path.join(outdir, name)
        torch.jit.save(traced, path)
        print(f"wrote {path} (torch {torch.__version__})")

    path = os.path.join(outdir, "libtorch_noise.pt")
    torch.jit.save(torch.jit.script(Noise().eval()), path)
    print(f"wrote {path} (torch {torch.__version__})")


if __name__ == "__main__":
    main()
