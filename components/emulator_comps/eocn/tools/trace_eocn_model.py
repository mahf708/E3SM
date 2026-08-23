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


class OceanWrapper(nn.Module):
    """Traceable wrapper adding the two things the ACE tracing script omits.

    **Input masking.**  fme's step zeroes masked input variables *in normalized
    space* (`_apply_input_mask` in fme/core/step/single_module.py), which in
    physical units means substituting each channel's training mean.  Samudra's
    ocean state is masked by bathymetry -- a separate mask per depth level --
    so on land, and below the sea floor, that is what the network was trained
    to see.  Feeding a physical zero instead lands roughly -25 standard
    deviations away on every masked cell, and a convolutional network with
    circular padding spreads that far inland: without this the traced model
    returns a 440 K global mean sea surface temperature.

    **Sea ice.**  The ocean corrector clamps the sea ice fraction to [0, 1] and
    zeroes the ice volume where the cell is ice free.  The tracing script only
    knows the atmosphere corrector, so it drops this.
    """

    def __init__(self, inner: nn.Module, sif_idx: int, zero_idx: list[int],
                 in_mask: torch.Tensor | None):
        super().__init__()
        self.inner = inner
        self.sif_idx = sif_idx
        self.register_buffer("zero_idx", torch.tensor(zero_idx, dtype=torch.long))
        self.has_mask = in_mask is not None
        if in_mask is not None:
            means = inner.in_means.reshape(1, -1, 1, 1)
            self.register_buffer("in_mask", in_mask)
            self.register_buffer("in_fill", means * (1.0 - in_mask))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.has_mask:
            n = self.in_mask.shape[1]
            state = inputs[:, :n] * self.in_mask + self.in_fill
            inputs = torch.cat([state, inputs[:, n:]], dim=1)
        out = self.inner(inputs)
        sif = torch.clamp(out[:, self.sif_idx], min=0.0, max=1.0)
        ice = (sif > 0.0).to(out.dtype)
        out = out.clone()
        out[:, self.sif_idx] = sif
        for j in range(self.zero_idx.shape[0]):
            k = self.zero_idx[j].item()
            out[:, k] = out[:, k] * ice
        return out


def _mask_name(channel: str) -> str | None:
    """The mask_* variable that says where a given input channel is defined.

    The ten atmospheric fluxes are defined over land too *in the training
    data*, which is exactly why they have to be masked here.  Online they come
    from the coupler as Foxx_*, which is identically zero wherever the ocean
    fraction is zero -- 19908 of 64800 cells, 30.7% of the grid.  Leaving them
    unmasked passes that structural zero straight into the network, where it
    reads as a -1.0 sigma downward longwave and a -0.5 sigma shortwave over a
    third of the domain, and a dilated ConvNeXt stack carries the
    discontinuity well inland of every coast.  Masking them fills those cells
    with each channel's training mean instead, which is what fme's
    _apply_input_mask does to a masked channel and the closest thing to "no
    information" the network has a representation for.

    None still means genuinely present everywhere: the two surface fractions.
    """
    for prefix in ("salinityCoarsened_", "temperatureCoarsened_",
                   "velocityZonalCoarsened_", "velocityMeridionalCoarsened_"):
        if channel.startswith(prefix):
            return "mask_" + channel[len(prefix):]
    if channel in ("sst", "ssh"):
        return "mask_2d"
    if channel in ("ocean_sea_ice_fraction", "iceVolumeTotal"):
        return "mask_" + channel
    if channel in FORCING:
        return "mask_2d"
    return None


def _build_input_mask(path: str, in_names: list[str], device: str):
    if not path:
        logger.warning("tracing without input masking; the network will see "
                       "whatever fills the masked cells")
        return None

    import netCDF4  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    d = netCDF4.Dataset(path)
    nlat = len(d.dimensions["lat"])
    nlon = len(d.dimensions["lon"])
    mask = np.ones((len(in_names), nlat, nlon), dtype=np.float32)
    n_masked = 0
    for i, name in enumerate(in_names):
        mname = _mask_name(name)
        if mname is None:
            continue
        if mname not in d.variables:
            raise SystemExit(
                f"ERROR: {path} has no mask variable {mname!r} for channel {name!r}")
        mask[i] = np.asarray(d[mname][:], dtype=np.float32)
        n_masked += 1
    d.close()
    logger.info("built an input mask for %d of %d channels from %s",
                n_masked, len(in_names), path)
    return torch.from_numpy(mask[None]).to(device)


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
    parser.add_argument(
        "--mask-file",
        default=("/pscratch/sd/m/mahf708/SamudrACE-E3SMv3/initial_conditions/"
                 "SamudrACE-E3SMv3-ICx3-train_ocean_ic.nc"),
        help="file carrying the mask_* variables that say where each ocean "
        "channel is defined; pass an empty string to trace without masking",
    )
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

        n_in_state = metadata["n_input_channels"]
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

        in_names = [metadata["input_channels"][i] for i in range(n_in_state)]
        in_mask = _build_input_mask(args.mask_file, in_names, args.device)
        wrapped = OceanWrapper(traceable, sif_idx, zero_idx, in_mask)
        wrapped = wrapped.to(args.device).eval()
        metadata["input_masking"] = args.mask_file or "none"

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
