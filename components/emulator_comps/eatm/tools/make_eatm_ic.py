#!/usr/bin/env python
"""Build an EATM initial-condition file from an ACE initial-condition dataset.

EATM's startup path reads one 2D (lat, lon) field per emulator *input* channel,
named exactly as the channel is named in the checkpoint, from the file named by
``eatm_ic_file``.  The initial conditions published with the ACE checkpoints
carry a leading ``time`` dimension (SamudrACE-E3SMv3 ships three), plus a lot of
fields EATM does not read.  This script selects one time index and writes out
just what EATM needs.

Usage
-----
    python make_eatm_ic.py \\
        --emulator SamudrACE-E3SMv3 \\
        --source $PSCRATCH/SamudrACE-E3SMv3/initial_conditions/SamudrACE-E3SMv3-ICx3-train_atmosphere_ic.nc \\
        --time-index 0 \\
        --output $PSCRATCH/SamudrACE-E3SMv3/eatm/samudrace_atm_ic_ic0.nc

Fields that are constant in time in the source (LANDFRAC, PHIS) are copied as
is.  The output keeps float32 and the source's lat/lon coordinates, so it lines
up with the gaussian_180x360 SCRIP mesh EATM builds its grid from.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import xarray as xr


def _levels(prefix: str) -> list[str]:
    return [f"{prefix}{k}" for k in range(8)]


# Emulator input channel lists.  Keep in sync with eatm_channels_mod.F90.
IN_NAMES: dict[str, list[str]] = {
    "ACE2-EAMv3": (
        ["LANDFRAC", "OCNFRAC", "ICEFRAC", "PHIS", "SOLIN", "PS", "TS"]
        + _levels("T_")
        + _levels("specific_total_water_")
        + _levels("U_")
        + _levels("V_")
    ),
    "SamudrACE-E3SMv3": (
        ["LANDFRAC", "OCNFRAC", "ICEFRAC", "PHIS", "SOLIN", "PS", "TS"]
        + _levels("T_")
        + _levels("STW_")
        + _levels("U_")
        + _levels("V_")
        + ["Qat2m", "Uat10m", "Vat10m", "Tat2m"]
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--emulator", required=True, choices=sorted(IN_NAMES))
    parser.add_argument("--source", required=True, help="ACE initial condition file")
    parser.add_argument(
        "--forcing",
        default=None,
        help="optional forcing file to pull time-invariant fields "
        "(LANDFRAC, PHIS) from when the IC file lacks them",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=0,
        help="which initial condition to extract (default: 0)",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    names = IN_NAMES[args.emulator]

    src = xr.open_dataset(args.source, decode_times=False)
    extra = (
        xr.open_dataset(args.forcing, decode_times=False)
        if args.forcing is not None
        else None
    )

    out = {}
    missing = []
    for name in names:
        da = None
        if name in src:
            da = src[name]
        elif extra is not None and name in extra:
            da = extra[name]
        else:
            missing.append(name)
            continue

        if "time" in da.dims:
            if da.sizes["time"] <= args.time_index:
                sys.exit(
                    f"ERROR: --time-index {args.time_index} is out of range for "
                    f"{name} (time has {da.sizes['time']} entries)"
                )
            da = da.isel(time=args.time_index, drop=True)

        if da.dims != ("lat", "lon"):
            sys.exit(f"ERROR: {name} has dims {da.dims}, expected ('lat', 'lon')")

        out[name] = da.astype(np.float32)

    # A single non-finite value anywhere in the input block is fatal, not local:
    # the spherical harmonic transform inside an SFNO is global, so one NaN
    # makes every output channel NaN, and the emulator is autoregressive, so it
    # never recovers.  The published SamudrACE-E3SMv3 initial conditions carry
    # NaN in ICEFRAC over every cell without sea ice -- 60% of the globe --
    # because sea-ice concentration is undefined there in the source dataset.
    # Zero is the right fill for a fraction meaning "none here".
    for name, da in out.items():
        bad = ~np.isfinite(da.values)
        nbad = int(bad.sum())
        if nbad:
            print(
                f"  WARNING {name}: {nbad} of {da.size} values non-finite in the "
                f"source, filled with 0"
            )
            out[name] = da.where(np.isfinite(da), 0.0).astype(np.float32)

    if missing:
        sys.exit(
            "ERROR: the source file is missing these emulator input channels:\n  "
            + "\n  ".join(missing)
            + "\nPass --forcing if the time-invariant ones live in the forcing file."
        )

    ds = xr.Dataset(out)
    ds = ds.assign_coords(lat=src["lat"], lon=src["lon"])
    ds.attrs["source_file"] = args.source
    ds.attrs["source_time_index"] = args.time_index
    ds.attrs["eatm_emulator"] = args.emulator
    ds.attrs["history"] = (
        "created by components/emulator_comps/eatm/tools/make_eatm_ic.py"
    )
    if "time" in src.coords and src["time"].size > args.time_index:
        ds.attrs["source_time"] = str(src["time"].values[args.time_index])

    ds.to_netcdf(args.output)
    print(f"wrote {args.output}")
    print(f"  {len(out)} channels, grid {ds.sizes['lat']} x {ds.sizes['lon']}")
    print("Set in user_nl_eatm:")
    print(f"  eatm_ic_file = '{args.output}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
