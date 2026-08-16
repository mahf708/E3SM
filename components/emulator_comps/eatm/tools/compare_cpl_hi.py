#!/usr/bin/env python
"""Compare the atmosphere fields in two coupler history files.

EATM's effect on the coupled system is entirely carried by the ``a2x`` fields,
and the coupler writes them to ``*.cpl.hi.*.nc``.  Diffing the same-dated file
from two runs is therefore the most direct way to see what a change to EATM
actually did -- more direct than any component's own history, which mixes the
atmosphere's effect with the ocean's response.

Two runs line up file-for-file when they share ``RUN_STARTDATE`` and
``HIST_OPTION`` / ``HIST_N``.  The reference runs under
``/pscratch/sd/a/anolan/e3sm_scratch/pm-gpu/`` use ``ndays`` / ``10`` starting
at ``0001-01-01``; see REVIEW.md "Reference runs".

Usage
-----
    python compare_cpl_hi.py REFERENCE.nc NEW.nc
    python compare_cpl_hi.py REFERENCE.nc NEW.nc --fields Sa_shum Sa_z

Reports the area-weighted global mean, min and max of each field in both files,
and the change.  Area weights come from ``doma_area`` when present.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import xarray as xr

# The a2x fields EATM sets, in the order they matter for the review.
DEFAULT_FIELDS = [
    "a2x_Sa_z",
    "a2x_Sa_topo",
    "a2x_Sa_tbot",
    "a2x_Sa_shum",
    "a2x_Sa_pbot",
    "a2x_Sa_pslv",
    "a2x_Sa_dens",
    "a2x_Sa_ptem",
    "a2x_Sa_u",
    "a2x_Sa_v",
    "a2x_Faxa_swnet",
    "a2x_Faxa_swvdr",
    "a2x_Faxa_lwdn",
    "a2x_Faxa_rainl",
    "a2x_Faxa_snowl",
]


def _weights(ds: xr.Dataset, like: xr.DataArray) -> np.ndarray | None:
    for name in ("doma_area", "domain_area", "area"):
        if name in ds:
            w = np.squeeze(np.asarray(ds[name].values, dtype=float))
            if w.shape == np.squeeze(np.asarray(like.values)).shape:
                return w
    return None


def _stats(ds: xr.Dataset, name: str):
    if name not in ds:
        return None
    da = ds[name]
    v = np.squeeze(np.asarray(da.values, dtype=float))
    w = _weights(ds, da)
    finite = np.isfinite(v)
    if w is not None and np.isfinite(w).all() and w.sum() > 0:
        mean = float((v[finite] * w[finite]).sum() / w[finite].sum())
    else:
        mean = float(v[finite].mean())
    return mean, float(v[finite].min()), float(v[finite].max())


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("reference", help="cpl.hi file from the baseline run")
    p.add_argument("new", help="cpl.hi file from the run under test")
    p.add_argument("--fields", nargs="*", default=None)
    p.add_argument(
        "--all", action="store_true", help="compare every a2x_ field present in both"
    )
    args = p.parse_args()

    ref = xr.open_dataset(args.reference, decode_times=False)
    new = xr.open_dataset(args.new, decode_times=False)

    if args.all:
        fields = sorted(
            n for n in ref.data_vars if n.startswith("a2x_") and n in new.data_vars
        )
    else:
        fields = args.fields or DEFAULT_FIELDS

    print(f"reference : {args.reference}")
    print(f"new       : {args.new}")
    print()
    print(
        f"{'field':<20} {'ref mean':>13} {'new mean':>13} {'change':>11} "
        f"{'ref min':>12} {'new min':>12} {'ref max':>12} {'new max':>12}"
    )
    print("-" * 112)

    missing = []
    for f in fields:
        a, b = _stats(ref, f), _stats(new, f)
        if a is None or b is None:
            missing.append(f)
            continue
        am, amin, amax = a
        bm, bmin, bmax = b
        if am != 0:
            chg = f"{100.0 * (bm - am) / abs(am):+.1f}%"
        else:
            chg = f"{bm - am:+.3e}"
        print(
            f"{f:<20} {am:13.5e} {bm:13.5e} {chg:>11} "
            f"{amin:12.4e} {bmin:12.4e} {amax:12.4e} {bmax:12.4e}"
        )

    if missing:
        print()
        print("absent from one or both files:", ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
