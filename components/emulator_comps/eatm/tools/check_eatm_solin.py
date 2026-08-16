#!/usr/bin/env python3
"""Check the SOLIN field an EATM run actually fed its emulator.

EATM writes the last SOLIN it computed into its restart file, so a restart is a
direct record of the insolation the emulator was handed.  Both emulators were
trained on an E3SMv3 stream in which SOLIN carries
``cell_methods = "time: mean"`` -- the mean over the 6 h leading up to its
timestamp -- so a run feeding the instantaneous value at the interval end is
forcing the model with a field it has never seen.  The two have the same global
mean, 342 W/m2, because the lit hemisphere is always the same fraction of the
globe, so a budget check cannot find this.  These two signatures can, and
neither needs orbital parameters, a calendar, or the model grid:

  dark fraction   An instantaneous insolation field is zero over exactly the
                  unlit hemisphere: half the globe, to machine precision.  A
                  6 h mean is zero only where the sun stays down for the whole
                  window, and the terminator sweeps 90 degrees of longitude in
                  6 h, so it comes out near a quarter -- plus polar night.

  peak value      The instantaneous field peaks at S0*eccf at the subsolar
                  point.  Averaging across a window in which the sun moves 45
                  degrees either side knocks that down by roughly a quarter.

Usage:

    check_eatm_solin.py <case>.eatm.r.<date>.nc [...]

A dark fraction of 0.5000 means the run had the pre-fix behaviour.
"""

import sys

import numpy as np
import netCDF4 as nc


def signature(path):
    with nc.Dataset(path) as d:
        if "SOLIN" not in d.variables:
            raise SystemExit(f"{path}: no SOLIN variable -- is this an EATM restart?")
        solin = np.asarray(d.variables["SOLIN"][:]).astype(float)

    if solin.ndim != 2:
        raise SystemExit(f"{path}: expected a 2D SOLIN, got shape {solin.shape}")

    ny, nx = solin.shape
    lat = np.deg2rad(np.linspace(-90 + 90 / ny, 90 - 90 / ny, ny))
    w = np.repeat(np.cos(lat)[:, None], nx, axis=1)
    w /= w.sum()

    return dict(
        dark=float(np.sum(w * (solin <= 1e-6))),
        peak=float(solin.max()),
        gmean=float(np.sum(w * solin)),
    )


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)

    print(f"{'restart':<52} {'dark frac':>10} {'peak':>9} {'gmean':>8}  verdict")
    print("-" * 100)

    bad = 0
    for path in argv[1:]:
        s = signature(path)
        # 0.5 is the instantaneous signature; anything near a quarter is a mean.
        if abs(s["dark"] - 0.5) < 0.02:
            verdict = "INSTANTANEOUS -- emulator forced with an untrained field"
            bad += 1
        elif s["dark"] < 0.40:
            verdict = "window mean -- correct"
        else:
            verdict = "unclear, inspect by hand"
        name = path.rsplit("/", 1)[-1]
        print(
            f"{name:<52} {s['dark']:>10.4f} {s['peak']:>9.1f} "
            f"{s['gmean']:>8.2f}  {verdict}"
        )

    print("-" * 100)
    print(f"{'expected, instantaneous':<52} {0.5:>10.4f} {'~1365':>9}")
    print(f"{'expected, 6 h window mean':<52} {'~0.27':>10} {'~1050':>9}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
