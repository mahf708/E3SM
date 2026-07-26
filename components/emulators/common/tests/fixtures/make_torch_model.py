#!/usr/bin/env python3
"""Generate the tiny TorchScript module used by the torch-backend test.

The module mirrors the ONNX fixture so both backends can be checked against
the same arithmetic, and returns a dict so the name-based output path gets
exercised:

    dT    = 2 * T + 1          # [ncol, 3]
    total = T.sum(dim=1, keepdim=True) + ps

Usage: make_torch_model.py <output.pt>
"""

import sys

import torch


class AffineEmulator(torch.nn.Module):
    def forward(
        self, T: torch.Tensor, ps: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {
            "dT": 2.0 * T + 1.0,
            "total": T.sum(dim=1, keepdim=True) + ps,
        }


def main(argv):
    if len(argv) != 2:
        print(f"usage: {argv[0]} <output.pt>", file=sys.stderr)
        return 2
    scripted = torch.jit.script(AffineEmulator())
    scripted.save(argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
