#!/usr/bin/env python
"""Build the atmosphere-side domain file for a grid pair that uses EOCN's mask.

gen_domain's job, done in numpy: the atmosphere's land fraction is one minus
the ocean fraction mapped conservatively onto the atmosphere grid, and its
coordinates and areas come from the atmosphere's SCRIP description.

    python make_atm_domain.py --map map_gauss180x360_to_ne30pg2_traave.nc \\
        --atm-scrip ne30pg2_scrip_20200209.nc \\
        --ocn-domain domain.ocn.gauss180x360_gauss180x360.nc \\
        --out domain.lnd.ne30pg2_gauss180x360.nc
"""

from __future__ import annotations

import argparse
import datetime
import os

import netCDF4 as nc
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map", required=True)
    p.add_argument("--atm-scrip", required=True)
    p.add_argument("--ocn-domain", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    m = nc.Dataset(args.map)
    n_a = len(m.dimensions["n_b"])       # destination = atmosphere
    n_o = len(m.dimensions["n_a"])       # source      = ocean
    row = np.array(m["row"][:]) - 1
    col = np.array(m["col"][:]) - 1
    S = np.array(m["S"][:])
    m.close()

    d = nc.Dataset(args.ocn_domain)
    ofrac_o = np.array(d["frac"][:]).reshape(-1)
    d.close()
    assert ofrac_o.size == n_o, (ofrac_o.size, n_o)

    ofrac_a = np.zeros(n_a)
    np.add.at(ofrac_a, row, S * ofrac_o[col])
    ofrac_a = np.clip(ofrac_a, 0.0, 1.0)
    lfrac_a = 1.0 - ofrac_a

    s = nc.Dataset(args.atm_scrip)
    xc = np.array(s["grid_center_lon"][:])
    yc = np.array(s["grid_center_lat"][:])
    xv = np.array(s["grid_corner_lon"][:])
    yv = np.array(s["grid_corner_lat"][:])
    area = np.array(s["grid_area"][:])
    s.close()
    nv = xv.shape[1]

    o = nc.Dataset(args.out, "w", format="NETCDF4")
    o.createDimension("n", n_a)
    o.createDimension("ni", n_a)
    o.createDimension("nj", 1)
    o.createDimension("nv", nv)

    def put(name, dtype, dims, data, **attrs):
        v = o.createVariable(name, dtype, dims)
        v[:] = data
        for k, val in attrs.items():
            v.setncattr(k, val)

    put("xc", "f8", ("nj", "ni"), xc.reshape(1, -1),
        long_name="longitude of grid cell center", units="degrees_east", bounds="xv")
    put("yc", "f8", ("nj", "ni"), yc.reshape(1, -1),
        long_name="latitude of grid cell center", units="degrees_north", bounds="yv")
    put("xv", "f8", ("nj", "ni", "nv"), xv.reshape(1, n_a, nv),
        long_name="longitude of grid cell verticies", units="degrees_east")
    put("yv", "f8", ("nj", "ni", "nv"), yv.reshape(1, n_a, nv),
        long_name="latitude of grid cell verticies", units="degrees_north")
    put("mask", "i4", ("nj", "ni"), np.ones((1, n_a), dtype=np.int32),
        long_name="domain mask", note="unitless", coordinates="xc yc",
        comment="0 value indicates cell is not active")
    put("area", "f8", ("nj", "ni"), area.reshape(1, -1),
        long_name="area of grid cell in radians squared", coordinates="xc yc",
        units="radian2")
    put("frac", "f8", ("nj", "ni"), lfrac_a.reshape(1, -1),
        long_name="fraction of grid cell that is active", coordinates="xc yc",
        note="unitless")

    o.title = "E3SM domain data:"
    o.Conventions = "CF-1.0"
    o.source = "{} + {}".format(os.path.basename(args.map),
                                os.path.basename(args.ocn_domain))
    o.comment = ("land fraction is 1 minus the EOCN ocean fraction mapped "
                 "conservatively onto the atmosphere grid")
    o.history = "created by make_atm_domain.py on {}".format(
        datetime.date.today().isoformat())
    o.close()
    print("wrote {} (global land fraction {:.4f})".format(
        args.out, float(np.sum(lfrac_a * area) / np.sum(area))))


if __name__ == "__main__":
    main()
