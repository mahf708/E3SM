#!/usr/bin/env python3
"""Write the TorchScript fixtures the libtorch backend test loads.

The C++ test cannot save a TorchScript archive -- `torch.jit.save` lives on
the Python side -- so the fixtures are generated at build time by whatever
interpreter on this machine has torch (see EMULATOR_TORCH_PYTHON in
common/tests/CMakeLists.txt).  Keep them tiny and keep their arithmetic
exact in float32: the point of the test is the plumbing (double -> float ->
double, shapes, device, multi-output), not the numerics of a real model.

    make_libtorch_fixture.py OUTDIR

writes, into OUTDIR:

  libtorch_affine.pt    x -> 2*x + 1, elementwise, any shape
  libtorch_tuple.pt     x -> (2*x + 1, x - 1), the multi-output path
  libtorch_channels.pt  x -> channel-weighted sum over exactly 3 channels,
                        so a wrong channel count fails inside forward()
  libtorch_noise.pt     x -> x + N(0, 1), drawn on every call: the seeding
                        path, scripted rather than traced so the draw stays
                        an op instead of being frozen into a constant

All three are traced on a [1, 3, 2, 4] input, which is the same shape family
as the real ACE / Samudra checkpoints ([1, channels, ny, nx]) at a size that
fits in a test.
"""

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
