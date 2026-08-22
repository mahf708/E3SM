#!/usr/bin/env python
"""Trace a Samudra-family ocean checkpoint into the TorchScript file EOCN loads.

Driver around the ACE repository's tracing script, same idea as EATM's
tools/trace_eatm_model.py.  Three ocean-specific things it has to do:

  * The ocean stepper's corrector is an ``OceanCorrectorConfig``.  The tracing
    script only knows the atmosphere corrector, so it reports "no correctors"
    and drops the sea-ice-fraction constraint.  This driver re-applies that
    constraint (clamp the fraction to [0,1], zero the ice volume where the
    cell is ice free) in a wrapper module so it ends up inside the graph.

  * The ocean's vertical coordinate is a DepthCoordinate.  The tracing script
    only touches ``atmosphere_vertical_coordinate`` when an atmosphere
    corrector is active, so this is fine, but do not turn correctors on.

  * EOCN addresses channels by position, so the layout is checked against the
    table compiled into eocn_channels_mod.F90 before anything is written.

Usage
-----
    python trace_eocn_model.py CHECKPOINT OUTPUT_BASE --device cuda
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import pathlib
import sys

import yaml

import torch
from torch import nn

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _levels(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{k}" for k in range(n)]


FORCING = [
    "TAUX",
    "TAUY",
    "surface_precipitation_rate",
    "frozen_precipitation_rate",
    "FLUS",
    "FSUS",
    "FLDS",
    "FSDS",
    "LHFLX",
    "SHFLX",
]

STATE = (
    ["sst", "ssh"]
    + _levels("salinityCoarsened_", 19)
    + _levels("temperatureCoarsened_", 19)
    + _levels("velocityZonalCoarsened_", 19)
    + _levels("velocityMeridionalCoarsened_", 19)
    + ["ocean_sea_ice_fraction", "iceVolumeTotal"]
)

EXPECTED = {
    "SamudrACE-E3SMv3": {
        "in": ["LANDFRAC", "sea_surface_fraction"] + FORCING + STATE,
        "out": STATE,
    }
}


class SeaIceCorrected(nn.Module):
    """Traceable wrapper adding the ocean corrector's sea-ice constraint."""

    def __init__(self, inner: nn.Module, sif_idx: int, zero_idx: list[int]):
        super().__init__()
        self.inner = inner
        self.sif_idx = sif_idx
        self.register_buffer("zero_idx", torch.tensor(zero_idx, dtype=torch.long))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.inner(inputs)
        sif = torch.clamp(out[:, self.sif_idx], min=0.0, max=1.0)
        ice = (sif > 0.0).to(out.dtype)
        out = out.clone()
        out[:, self.sif_idx] = sif
        for j in range(self.zero_idx.shape[0]):
            k = self.zero_idx[j].item()
            out[:, k] = out[:, k] * ice
        return out


def _load_ace_tracer(path: pathlib.Path):
    if not path.is_file():
        raise SystemExit(f"ERROR: ACE tracing script not found at {path}")
    spec = importlib.util.spec_from_file_location("ace_trace_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ace_trace_script"] = module
    spec.loader.exec_module(module)
    return module


def _check(metadata: dict, emulator: str) -> None:
    exp = EXPECTED[emulator]
    n_in = metadata["n_input_channels"]
    got_in = [metadata["input_channels"][i] for i in range(n_in)]
    got_out = [metadata["output_channels"][i] for i in range(len(metadata["output_channels"]))]
    for kind, got, want in (("input", got_in, exp["in"]), ("output", got_out, exp["out"])):
        if got != want:
            bad = [
                f"    [{i}] got {got[i] if i < len(got) else '<missing>'!r}, "
                f"expected {want[i] if i < len(want) else '<missing>'!r}"
                for i in range(max(len(got), len(want)))
                if (got[i] if i < len(got) else None) != (want[i] if i < len(want) else None)
            ]
            raise SystemExit(
                f"ERROR: {kind} channels do not match the '{emulator}' table in "
                f"eocn_channels_mod.F90 (got {len(got)}, expected {len(want)}):\n"
                + "\n".join(bad[:25])
            )
    logger.info("channel layout matches '%s' (%d in, %d out)", emulator, n_in, len(got_out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output", nargs="?", default="eocn_traced")
    parser.add_argument("--emulator", default="SamudrACE-E3SMv3", choices=sorted(EXPECTED))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--trace-script",
        default=os.environ.get("EATM_TRACE_SCRIPT", "/pscratch/sd/m/mahf708/test_ace_repo/trace.py"),
    )
    parser.add_argument("--check-trace", action="store_true")
    args = parser.parse_args()

    tracer = _load_ace_tracer(pathlib.Path(args.trace_script))
    from fme.core.distributed import Distributed  # noqa: PLC0415
    import fme.ace.stepper.single_module as _sm  # noqa: PLC0415

    # The tracing script resolves force-positive channels through
    # ``corrector._config``, which only the atmosphere corrector has.  Keep a
    # handle on the stepper so we can read them off the ocean corrector's
    # CorrectorSelector instead.
    stash: dict = {}
    _orig_load = _sm.load_stepper

    def _load(*a, **k):
        stepper = _orig_load(*a, **k)
        stash["stepper"] = stepper
        return stepper

    _sm.load_stepper = _load

    with Distributed.context():
        traceable, metadata = tracer.load_and_build(
            args.checkpoint,
            device=args.device,
            include_normalization=True,
            include_corrector=False,
            include_ocean=False,
        )
        _check(metadata, args.emulator)

        out_names = [metadata["output_channels"][i] for i in range(len(metadata["output_channels"]))]
        if traceable.force_pos_idx.numel() == 0:
            step = tracer._unwrap_step(stash["stepper"])
            sel = getattr(step.config, "corrector", None)
            cfg = getattr(sel, "config", None) or {}
            fp = list(cfg.get("force_positive_names", []))
            idx = [out_names.index(n) for n in fp if n in out_names]
            if not idx:
                raise SystemExit(
                    "ERROR: no force-positive channels resolved from the ocean "
                    "corrector; salinity would be free to go negative."
                )
            traceable.force_pos_idx = torch.tensor(
                idx, dtype=torch.long, device=traceable.force_pos_idx.device
            )
            metadata["force_positive_names"] = fp
            logger.info("restored %d force-positive channels", len(idx))
        sif_idx = out_names.index("ocean_sea_ice_fraction")
        zero_idx = [out_names.index("iceVolumeTotal")]
        wrapped = SeaIceCorrected(traceable, sif_idx, zero_idx).to(args.device).eval()

        # Samudra's ConvNeXt blocks run under torch.utils.checkpoint when
        # checkpoint_strategy == "all".  That is a training-time memory trade
        # and torch.jit.save cannot export the autograd function it inserts.
        # Turning it off is answer-preserving.
        n_off = 0
        for m in wrapped.modules():
            if getattr(m, "checkpoint_strategy", None) is not None:
                m.checkpoint_strategy = None
                n_off += 1
        if n_off:
            logger.info("disabled gradient checkpointing on %d submodules", n_off)

        metadata["corrector_flags"] = {
            "any_active": True,
            "force_positive": True,
            "sea_ice_fraction_clamp": True,
            "zero_where_ice_free": ["iceVolumeTotal"],
        }
        metadata["corrector_enabled"] = True

        n_in = metadata["n_input_channels"]
        n_forcing = metadata["n_forcing_channels"]
        n_lat, n_lon = metadata["n_lat"], metadata["n_lon"]
        example = torch.randn(1, n_in + n_forcing, n_lat, n_lon, device=args.device)
        logger.info(
            "tracing (%d state + %d forcing channels, %dx%d)", n_in, n_forcing, n_lat, n_lon
        )
        with torch.no_grad():
            traced = torch.jit.trace(wrapped, (example,), check_trace=args.check_trace)

        out = pathlib.Path(f"{args.output}_{args.device}")
        pt_path = out.with_suffix(".pt")
        meta_path = out.parent / (out.name + "_metadata.yaml")
        torch.jit.save(traced, str(pt_path))
        metadata["eocn_emulator"] = args.emulator
        with open(meta_path, "w") as fh:
            yaml.dump(metadata, fh, default_flow_style=False, sort_keys=False)
        logger.info("wrote %s", pt_path)
        logger.info("wrote %s", meta_path)


if __name__ == "__main__":
    main()
