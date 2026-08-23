#!/usr/bin/env python
"""Build EOCN's initial condition and the domain files for its grid.

The published SamudrACE ocean initial condition is not directly loadable by
EOCN for three reasons, all of them fixed here:

  * it bundles three initial conditions on a leading record dimension, and
    carries the static fields (LANDFRAC, sea_surface_fraction) without one;
  * every ocean state variable is NaN over land, because an ocean state is
    undefined there.  A convolutional emulator spreads one NaN outward on
    every layer, and EOCN is autoregressive, so a single NaN at step zero is a
    NaN field for the rest of the run.  fme's own data loader substitutes zero
    and the network was trained against that substitution;
  * it carries a great many variables EOCN never reads.

The domain files describe the same 180x360 Gaussian grid the SCRIP mesh does.
The ocean domain's fraction is the emulator's own sea surface fraction, so the
coupler weights the ocean exactly where the checkpoint believes there is one.

Usage
-----
    python make_eocn_input.py --ic-out samudra_ocn_ic_0.nc \\
        --domain-dir $DIN_LOC_ROOT/share/domains
"""

from __future__ import annotations

import argparse
import datetime
import os

import netCDF4 as nc
import numpy as np

SRC_IC = ("/pscratch/sd/m/mahf708/SamudrACE-E3SMv3/initial_conditions/"
          "SamudrACE-E3SMv3-ICx3-train_ocean_ic.nc")
SCRIP = ("/global/cfs/cdirs/e3sm/inputdata/share/meshes/"
         "gaussian_180x360_latlon.scrip.20260127.nc")

STATIC = ["LANDFRAC", "sea_surface_fraction"]
# Not emulator channels, but the two masks that say where its output means
# anything.  mask_2d is where the ocean state is defined at all: the sea
# surface fraction is nonzero on ~3200 more coastal cells than it covers, and
# on those the emulator has no ocean to predict, so the mask and not the
# fraction is what EOCN may export.  mask_ocean_sea_ice_fraction is tighter
# still -- 25,923 cells against 44,892 -- and bounds the sea ice channels,
# which were masked out of both the inputs and the targets everywhere else and
# so carry no prediction there.
EXTRA = ["mask_2d", "mask_ocean_sea_ice_fraction"]
FORCING = ["TAUX", "TAUY", "surface_precipitation_rate",
           "frozen_precipitation_rate", "FLUS", "FSUS", "FLDS", "FSDS",
           "LHFLX", "SHFLX"]
STATE = (["sst", "ssh"]
         + [f"salinityCoarsened_{k}" for k in range(19)]
         + [f"temperatureCoarsened_{k}" for k in range(19)]
         + [f"velocityZonalCoarsened_{k}" for k in range(19)]
         + [f"velocityMeridionalCoarsened_{k}" for k in range(19)]
         + ["ocean_sea_ice_fraction", "iceVolumeTotal"])


def build_ic(src: str, out: str, record: int) -> None:
    d = nc.Dataset(src)
    nlat = len(d.dimensions["lat"])
    nlon = len(d.dimensions["lon"])

    o = nc.Dataset(out, "w", format="NETCDF4")
    o.createDimension("lat", nlat)
    o.createDimension("lon", nlon)
    for name, src_name in (("lat", "lat"), ("lon", "lon")):
        v = o.createVariable(name, "f8", (name,))
        v[:] = d[src_name][:]
        for a in d[src_name].ncattrs():
            if a == "_FillValue":
                continue
            v.setncattr(a, d[src_name].getncattr(a))

    nfilled = 0
    for name in STATIC + EXTRA + FORCING + STATE:
        src_var = d[name]
        arr = np.array(src_var[:], dtype=np.float64)
        if src_var.dimensions and src_var.dimensions[0] == "time":
            arr = arr[record]
        bad = ~np.isfinite(arr)
        nfilled += int(bad.sum())
        arr = np.where(bad, 0.0, arr)
        v = o.createVariable(name, "f8", ("lat", "lon"))
        v[:] = arr
        for a in src_var.ncattrs():
            if a in ("_FillValue",):
                continue
            v.setncattr(a, src_var.getncattr(a))

    o.title = "EOCN initial condition"
    o.source = src
    o.comment = (
        "record {} of the published SamudrACE ocean initial conditions; "
        "non-finite values replaced with zero, as fme's data loader does"
    ).format(record)
    o.history = "created by make_eocn_input.py on {}".format(
        datetime.date.today().isoformat())
    o.close()
    d.close()
    print("wrote {} ({} non-finite values zero-filled)".format(out, nfilled))


def build_domains(scrip: str, ic: str, outdir: str, stamp: str) -> None:
    s = nc.Dataset(scrip)
    ni, nj = [int(x) for x in s["grid_dims"][:]]
    xc = np.array(s["grid_center_lon"][:]).reshape(nj, ni)
    yc = np.array(s["grid_center_lat"][:]).reshape(nj, ni)
    xv = np.array(s["grid_corner_lon"][:]).reshape(nj, ni, 4)
    yv = np.array(s["grid_corner_lat"][:]).reshape(nj, ni, 4)
    area = np.array(s["grid_area"][:]).reshape(nj, ni)
    s.close()

    d = nc.Dataset(ic)
    # Binary, matching what EOCN hands the coupler: seq_domain_mct derives the
    # ocean fraction on the atmosphere grid from the *mask*, so a continuous
    # fraction here would disagree with the land model on every coastal cell.
    ofrac = (np.array(d["mask_2d"][:]) > 0.5).astype(float)
    d.close()

    for kind, frac in (("ocn", ofrac), ("lnd", np.ones((nj, ni)))):
        path = os.path.join(
            outdir, "domain.{}.gauss180x360_gauss180x360.{}.nc".format(kind, stamp))
        o = nc.Dataset(path, "w", format="NETCDF4")
        o.createDimension("n", ni * nj)
        o.createDimension("ni", ni)
        o.createDimension("nj", nj)
        o.createDimension("nv", 4)

        def put(name, dtype, dims, data, **attrs):
            v = o.createVariable(name, dtype, dims)
            v[:] = data
            for k, val in attrs.items():
                v.setncattr(k, val)

        put("xc", "f8", ("nj", "ni"), xc,
            long_name="longitude of grid cell center", units="degrees_east",
            bounds="xv")
        put("yc", "f8", ("nj", "ni"), yc,
            long_name="latitude of grid cell center", units="degrees_north",
            bounds="yv")
        put("xv", "f8", ("nj", "ni", "nv"), xv,
            long_name="longitude of grid cell verticies", units="degrees_east")
        put("yv", "f8", ("nj", "ni", "nv"), yv,
            long_name="latitude of grid cell verticies", units="degrees_north")
        put("mask", "i4", ("nj", "ni"), (frac > 0.0).astype(np.int32),
            long_name="domain mask", note="unitless", coordinates="xc yc",
            comment="0 value indicates cell is not active")
        put("area", "f8", ("nj", "ni"), area,
            long_name="area of grid cell in radians squared",
            coordinates="xc yc", units="radian2")
        put("frac", "f8", ("nj", "ni"), frac,
            long_name="fraction of grid cell that is active",
            coordinates="xc yc", note="unitless")

        o.title = "E3SM domain data:"
        o.Conventions = "CF-1.0"
        o.source = "{} + {}".format(os.path.basename(scrip), os.path.basename(ic))
        o.comment = ("identity domain for the gauss180x360 grid; the ocean "
                     "fraction is the SamudrACE ocean emulator's own "
                     "sea_surface_fraction")
        o.history = "created by make_eocn_input.py on {}".format(
            datetime.date.today().isoformat())
        o.close()
        print("wrote {} (active fraction {:.4f})".format(
            path, float(np.sum(frac * area) / np.sum(area))))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-ic", default=SRC_IC)
    p.add_argument("--scrip", default=SCRIP)
    p.add_argument("--record", type=int, default=0)
    p.add_argument("--ic-out", required=True)
    p.add_argument("--domain-dir", default=None)
    p.add_argument("--stamp", default=datetime.date.today().strftime("%Y%m%d"))
    args = p.parse_args()

    build_ic(args.src_ic, args.ic_out, args.record)
    if args.domain_dir:
        build_domains(args.scrip, args.ic_out, args.domain_dir, args.stamp)


if __name__ == "__main__":
    main()
