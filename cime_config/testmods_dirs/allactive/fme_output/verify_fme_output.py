#!/usr/bin/env python3
"""
Verify and visualize FME online output against ACE/Samudra requirements.

Checks that all fields required by ACE (github.com/ai2cm/ace, branch exp/e3sm)
are present and physically reasonable, then produces diagnostic figures saved
to an output directory for online browsing.

ACE variable mapping reference:
  configs/experiments/2025-11-05-e3smv3-piControl-100yr/atmosphere/config-train.yaml
  configs/experiments/2025-11-05-e3smv3-piControl-100yr/ocean/config-train.yaml
  scripts/data_process/configs/e3smv3-coupled-atm-1deg.yaml
  scripts/data_process/configs/e3smv3-ocean-1deg.yaml

Usage:
    micromamba run -n xgns python verify_fme_output.py \\
        --rundir /path/to/RUNDIR \\
        --outdir /pscratch/sd/m/mahf708/aifigs_v3LRpiControlSamudrACE
"""

import argparse
import os
import sys
import glob
import textwrap
from pathlib import Path

import numpy as np

# ── optional imports ──────────────────────────────────────────────────────────
try:
    import xarray as xr
    HAS_XR = True
except ImportError:
    HAS_XR = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

try:
    from netCDF4 import Dataset as NC4Dataset
    HAS_NC4 = True
except ImportError:
    HAS_NC4 = False

# ── ACE variable requirements ─────────────────────────────────────────────────
# Fields ACE atmosphere model needs from EAM (after offline renaming).
# key = EAM output name,  value = ACE canonical name
ACE_ATM_FIELDS = {
    "PHIS": "PHIS",
    "PS": "PS",
    "TS": "TS",
    "TAUX": "TAUX",
    "TAUY": "TAUY",
    "LHFLX": "LHFLX",
    "SHFLX": "SHFLX",
    "PRECT": "surface_precipitation_rate",  # derived: PRECC+PRECL
    "FLDS": "FLDS",
    "FSDS": "FSDS",
    "FLNS": "FLNS",   # -> surface_upward_longwave_flux = FLDS - FLNS
    "FSNS": "FSNS",   # -> surface_upward_shortwave_flux = FSDS - FSNS
    "FSNTOA": "FSNTOA",  # -> top_of_atmos_upward_shortwave_flux = SOLIN - FSNTOA
    "FLUT": "FLUT",
    "SOLIN": "SOLIN",
    "ICEFRAC": "sea_ice_fraction",
    "LANDFRAC": "land_fraction",
    "OCNFRAC": "ocean_fraction",
    # 3D coarsened (tape 3): T_0..T_7, Q_0..Q_7, U_0..U_7, V_0..V_7,
    #                         TOTAL_WATER_0..7 -> specific_total_water_0..7
}
ACE_ATM_3D_COARSENED = ["T", "Q", "U", "V", "TOTAL_WATER"]
N_ATM_LAYERS = 8  # ACE uses 8 pressure layers

# Fields ACE ocean model needs from MPAS-O fmeDepthCoarsening (19 layers).
ACE_OCN_COARSENED = {
    "temperatureCoarsened": "thetao",
    "salinityCoarsened": "so",
    "velocityZonalCoarsened": "uo",
    "velocityMeridionalCoarsened": "vo",
}
N_OCN_LAYERS = 19  # ACE uses 19 depth layers

# Fields ACE ocean model needs from MPAS-O fmeDerivedFields.
ACE_OCN_DERIVED = {
    "sst": "sst",
    "sss": "sss",
    "ssh": "zos",
    "windStressZonal": "tauvo",
    "windStressMeridional": "tauuo",
    "shortWaveHeatFlux": "sw_heat_flux",
    "longWaveHeatFluxDown": "lw_heat_flux_down",
    "sensibleHeatFlux": "sensible_heat_flux",
    "surfaceThicknessFlux": "evap_flux",
}

# Fields ACE needs from MPAS-SI fmeSeaiceDerivedFields.
ACE_ICE_DERIVED = {
    "iceAreaTotal": "ocean_sea_ice_fraction",
    "iceVolumeTotal": "sea_ice_volume",
}

# Physical range checks: (vmin, vmax)
RANGE_CHECKS = {
    "temperatureCoarsened": (-5, 40),
    "salinityCoarsened": (0, 42),
    "velocityZonalCoarsened": (-5, 5),
    "velocityMeridionalCoarsened": (-5, 5),
    "sst": (-5, 40),
    "sss": (0, 42),
    "iceAreaTotal": (0, 1),
    "iceVolumeTotal": (0, None),
    "T": (150, 350),
    "Q": (0, 0.05),
    "PS": (40000, 110000),
    "TS": (200, 340),
    "PHIS": (-1000, 60000),
}


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_files(rundir, pattern):
    return sorted(glob.glob(os.path.join(rundir, pattern)))


def safe_open(path):
    """Open a NetCDF file with xarray (preferred) or netCDF4."""
    if HAS_XR:
        try:
            return xr.open_dataset(path, decode_times=False)
        except Exception:
            pass
    if HAS_NC4:
        return NC4Dataset(path, "r")
    raise RuntimeError("Neither xarray nor netCDF4 is available.")


def get_var(ds, name):
    """Return numpy array from xarray Dataset or netCDF4 Dataset."""
    if HAS_XR and isinstance(ds, xr.Dataset):
        if name in ds:
            return ds[name].values
        return None
    if HAS_NC4 and isinstance(ds, NC4Dataset):
        if name in ds.variables:
            return np.array(ds.variables[name][:])
        return None
    return None


def valid_data(arr, fill=1e33):
    """Return non-fill, finite values."""
    if arr is None:
        return None
    flat = arr.ravel().astype(float)
    mask = (np.abs(flat) < fill) & np.isfinite(flat)
    return flat[mask] if mask.any() else None


def check_range(arr, name, vmin=None, vmax=None):
    v = valid_data(arr)
    issues = []
    if v is None or v.size == 0:
        issues.append(f"  {name}: no valid data")
        return issues
    if not np.isfinite(v).all():
        issues.append(f"  {name}: NaN/Inf present")
    if vmin is not None and v.min() < vmin:
        issues.append(f"  {name}: min={v.min():.4g} below {vmin}")
    if vmax is not None and v.max() > vmax:
        issues.append(f"  {name}: max={v.max():.4g} above {vmax}")
    return issues


def summary_stats(arr):
    v = valid_data(arr)
    if v is None or v.size == 0:
        return "no valid data"
    return f"mean={v.mean():.4g}  min={v.min():.4g}  max={v.max():.4g}  n={v.size}"


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def savefig(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def global_map(data, lons, lats, title, cmap="RdBu_r", vmin=None, vmax=None,
               outdir=None, fname=None, units=""):
    """
    Plot a global filled map.  data is 2-D (lat x lon) or 1-D (nCells) on an
    unstructured grid (scatter plot fallback).
    """
    if not HAS_MPL:
        return None

    fig = plt.figure(figsize=(10, 5))
    if HAS_CARTOPY:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.set_global()
        if data.ndim == 2:
            ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(),
                          cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        else:  # unstructured
            sc = ax.scatter(lons, lats, c=data, s=1, cmap=cmap,
                            vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
            plt.colorbar(sc, ax=ax, shrink=0.6, label=units)
    else:
        ax = fig.add_subplot(1, 1, 1)
        if data.ndim == 2:
            im = ax.pcolormesh(lons, lats, data, cmap=cmap,
                               vmin=vmin, vmax=vmax, shading="auto")
        else:
            im = ax.scatter(lons, lats, c=data, s=1, cmap=cmap,
                            vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8, label=units)
        ax.set_xlabel("lon"); ax.set_ylabel("lat")

    ax.set_title(title, fontsize=11)
    if outdir and fname:
        return savefig(fig, outdir, fname)
    plt.show()
    return None


def layer_profiles(data3d, label, outdir, fname, ylabel="Level index"):
    """Plot global-mean vertical profile from (nlev x ncells) array."""
    if not HAS_MPL:
        return
    means = []
    for lev in range(data3d.shape[0]):
        v = valid_data(data3d[lev])
        means.append(v.mean() if v is not None and v.size > 0 else np.nan)
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.plot(means, np.arange(len(means)), "o-")
    ax.invert_yaxis()
    ax.set_xlabel(label)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Global-mean vertical profile: {label}")
    ax.grid(True, alpha=0.4)
    savefig(fig, outdir, fname)


def time_series(values, times_label, title, ylabel, outdir, fname):
    """Plot a simple time series."""
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(values, ".-")
    ax.set_xlabel(times_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    savefig(fig, outdir, fname)


# ─────────────────────────────────────────────────────────────────────────────
# Per-component verification + visualization
# ─────────────────────────────────────────────────────────────────────────────

def check_eam(rundir, outdir, verbose):
    print("\n=== EAM ===")
    issues = []
    plots = []

    # ── tape 1: averaged 2D fields ────────────────────────────────────────────
    t1_files = find_files(rundir, "*.eam.h0.*.nc") or find_files(rundir, "*.cam.h0.*.nc")
    if not t1_files:
        print("  SKIP tape 1: no h0 files found")
    else:
        ds = safe_open(t1_files[0])
        for var in ["PHIS", "PS", "TS", "LHFLX", "SHFLX", "FLDS", "FSDS",
                    "FLNS", "FSNS", "FSNTOA", "FLUT", "SOLIN",
                    "ICEFRAC", "LANDFRAC", "OCNFRAC", "TAUX", "TAUY"]:
            arr = get_var(ds, var)
            if arr is None:
                issues.append(f"  tape1 missing: {var}")
            else:
                vmin, vmax = RANGE_CHECKS.get(var, (None, None))
                issues += check_range(arr, f"tape1/{var}", vmin, vmax)
                if verbose:
                    print(f"  tape1/{var}: {summary_stats(arr)}")

        # Maps of key 2D fields on lat-lon grid
        if HAS_MPL and HAS_XR and isinstance(ds, xr.Dataset):
            lon_name = next((d for d in ["lon", "ncol", "longitude"] if d in ds.dims), None)
            lat_name = next((d for d in ["lat", "latitude"] if d in ds.dims), None)
            for var, cmap, vmin, vmax, units in [
                ("TS",    "RdBu_r",  220, 320, "K"),
                ("LHFLX","viridis",    0, 300, "W/m²"),
                ("SHFLX","RdBu_r", -100, 200, "W/m²"),
                ("FLUT", "inferno",  100, 320, "W/m²"),
                ("PRECT","Blues",      0, 1e-3,"kg/m²/s"),
                ("ICEFRAC","Blues",    0,   1, "fraction"),
            ]:
                arr = get_var(ds, var)
                if arr is None:
                    continue
                data = arr[0] if arr.ndim >= 2 else arr  # first time step
                if lat_name and lon_name and data.ndim == 2:
                    lons = ds[lon_name].values
                    lats = ds[lat_name].values
                    if lons.ndim == 1:
                        lons, lats = np.meshgrid(lons, lats)
                    p = global_map(data, lons, lats, f"EAM {var} (tape1, t=0)",
                                   cmap=cmap, vmin=vmin, vmax=vmax,
                                   outdir=outdir, fname=f"eam_tape1_{var}.png",
                                   units=units)
                    if p: plots.append(p)
        if HAS_XR and isinstance(ds, xr.Dataset):
            ds.close()

    # ── tape 3: vertically coarsened 3D state ────────────────────────────────
    t3_files = find_files(rundir, "*.eam.h2.*.nc") or find_files(rundir, "*.cam.h2.*.nc")
    if not t3_files:
        print("  SKIP tape 3: no h2 files found")
    else:
        ds3 = safe_open(t3_files[0])
        for base in ACE_ATM_3D_COARSENED:
            found_layers = []
            for k in range(N_ATM_LAYERS):
                vname = f"{base}_{k}"
                arr = get_var(ds3, vname)
                if arr is not None:
                    found_layers.append(k)
                    vmin, vmax = RANGE_CHECKS.get(base, (None, None))
                    issues += check_range(arr, f"tape3/{vname}", vmin, vmax)
            if len(found_layers) == N_ATM_LAYERS:
                print(f"  tape3/{base}: all {N_ATM_LAYERS} layers present ✓")
            elif found_layers:
                print(f"  tape3/{base}: {len(found_layers)}/{N_ATM_LAYERS} layers found "
                      f"(indices {found_layers})")
                issues.append(f"  tape3/{base}: only {len(found_layers)}/{N_ATM_LAYERS} layers")
            else:
                # EAM may name them differently; check alternate patterns
                alts = [v for v in (ds3.data_vars if HAS_XR else ds3.variables)
                        if v.startswith(base + "_") or v.startswith(base.upper() + "_")]
                if alts:
                    print(f"  tape3/{base}: found alternate names: {alts[:5]}")
                else:
                    issues.append(f"  tape3 missing coarsened field: {base}_0..{base}_{N_ATM_LAYERS-1}")
        if HAS_XR and isinstance(ds3, xr.Dataset):
            ds3.close()

    _report("EAM", issues)
    return issues, plots


def check_mpaso_depth_coarsening(rundir, outdir, verbose):
    print("\n=== MPAS-O Depth Coarsening ===")
    issues = []
    plots = []

    files = find_files(rundir, "analysis_members/fmeDepthCoarsening.*.nc")
    if not files:
        print("  SKIP: no fmeDepthCoarsening files found")
        return issues, plots

    ds = safe_open(files[0])

    for var, ace_name in ACE_OCN_COARSENED.items():
        arr = get_var(ds, var)
        if arr is None:
            issues.append(f"  missing: {var} (ACE: {ace_name})")
            continue
        # arr shape: (Time, nFmeDepthLevels, nCells)
        n_levels = arr.shape[1] if arr.ndim == 3 else arr.shape[0]
        if n_levels != N_OCN_LAYERS:
            issues.append(f"  {var}: {n_levels} depth layers, expected {N_OCN_LAYERS}")
        vmin, vmax = RANGE_CHECKS.get(var, (None, None))
        issues += check_range(arr, var, vmin, vmax)
        if verbose:
            print(f"  {var} ({ace_name}): {summary_stats(arr)}")

        # Vertical profile plot (global mean across all cells, first timestep)
        if HAS_MPL and arr.ndim == 3:
            layer_profiles(arr[0], f"{var} [{ace_name}]", outdir,
                           f"mpaso_depth_profile_{var}.png",
                           ylabel="Depth layer (0=surface)")

    # Surface map of SST layer (layer 0 of temperatureCoarsened)
    if HAS_MPL:
        arr = get_var(ds, "temperatureCoarsened")
        lon = get_var(ds, "lonCell")
        lat = get_var(ds, "latCell")
        if arr is not None and lon is not None and lat is not None:
            data = arr[0, 0, :]  # first time, surface layer, all cells
            lon_deg = np.degrees(lon)
            lat_deg = np.degrees(lat)
            p = global_map(data, lon_deg, lat_deg,
                           "MPAS-O temperatureCoarsened layer 0 (surface)",
                           cmap="RdBu_r", vmin=-2, vmax=30,
                           outdir=outdir, fname="mpaso_temp_surf.png", units="°C")
            if p: plots.append(p)

        arr = get_var(ds, "salinityCoarsened")
        if arr is not None and lon is not None:
            data = arr[0, 0, :]
            p = global_map(data, lon_deg, lat_deg,
                           "MPAS-O salinityCoarsened layer 0 (surface)",
                           cmap="viridis", vmin=30, vmax=40,
                           outdir=outdir, fname="mpaso_sal_surf.png", units="PSU")
            if p: plots.append(p)

    if HAS_XR and isinstance(ds, xr.Dataset):
        ds.close()

    _report("MPAS-O depth coarsening", issues)
    return issues, plots


def check_mpaso_derived(rundir, outdir, verbose):
    print("\n=== MPAS-O Derived Fields ===")
    issues = []
    plots = []

    files = find_files(rundir, "analysis_members/fmeDerivedFields.*.nc")
    if not files:
        print("  SKIP: no fmeDerivedFields files found")
        return issues, plots

    ds = safe_open(files[0])

    for var, ace_name in ACE_OCN_DERIVED.items():
        arr = get_var(ds, var)
        if arr is None:
            issues.append(f"  missing: {var} (ACE: {ace_name})")
            continue
        vmin, vmax = RANGE_CHECKS.get(var, (None, None))
        issues += check_range(arr, var, vmin, vmax)
        if verbose:
            print(f"  {var} ({ace_name}): {summary_stats(arr)}")

    # Cross-validate surfaceHeatFluxTotal if components present
    component_vars = ["shortWaveHeatFlux", "longWaveHeatFluxDown",
                      "sensibleHeatFlux", "surfaceThicknessFlux"]
    shf_total = get_var(ds, "surfaceHeatFluxTotal")
    if shf_total is not None and all(get_var(ds, v) is not None for v in component_vars):
        comp_sum = sum(get_var(ds, v) for v in component_vars)
        v_tot = valid_data(shf_total)
        v_sum = valid_data(comp_sum)
        if v_tot is not None and v_sum is not None:
            diff = np.abs(v_tot - v_sum).max()
            if diff > 1.0:
                issues.append(f"  surfaceHeatFluxTotal vs component sum: max diff={diff:.4g}")
            elif verbose:
                print(f"  surfaceHeatFluxTotal cross-validation: max diff={diff:.2e} ✓")

    # Maps
    if HAS_MPL:
        lon = get_var(ds, "lonCell")
        lat = get_var(ds, "latCell")
        if lon is not None:
            lon_deg = np.degrees(lon)
            lat_deg = np.degrees(lat)
            for var, cmap, vmin, vmax, units in [
                ("sst",  "RdBu_r",  -2, 30, "°C"),
                ("ssh",  "RdBu_r",  -2,  2, "m"),
                ("shortWaveHeatFlux", "YlOrRd", 0, 300, "W/m²"),
            ]:
                arr = get_var(ds, var)
                if arr is None:
                    continue
                data = arr[0] if arr.ndim > 1 else arr
                p = global_map(data, lon_deg, lat_deg,
                               f"MPAS-O {var} (t=0)",
                               cmap=cmap, vmin=vmin, vmax=vmax,
                               outdir=outdir, fname=f"mpaso_derived_{var}.png",
                               units=units)
                if p: plots.append(p)

    if HAS_XR and isinstance(ds, xr.Dataset):
        ds.close()

    _report("MPAS-O derived fields", issues)
    return issues, plots


def check_mpaso_vertical_reduce(rundir, outdir, verbose):
    print("\n=== MPAS-O Vertical Reduction ===")
    issues = []

    files = find_files(rundir, "analysis_members/fmeVerticalReduce.*.nc")
    if not files:
        print("  SKIP: no fmeVerticalReduce files found")
        return issues, []

    ds = safe_open(files[0])
    for var, (vmin, vmax) in [("oceanHeatContent", (None, None)),
                               ("freshwaterContent", (-1000, 1000)),
                               ("kineticEnergy", (0, None))]:
        arr = get_var(ds, var)
        if arr is None:
            issues.append(f"  missing: {var}")
        else:
            issues += check_range(arr, var, vmin, vmax)
            if verbose:
                print(f"  {var}: {summary_stats(arr)}")

    # Time series of global-mean OHC
    plots = []
    if HAS_MPL:
        ohc = get_var(ds, "oceanHeatContent")
        if ohc is not None and ohc.ndim >= 2:
            area = get_var(ds, "areaCell")
            if area is not None:
                ts = [(ohc[t] * area).sum() / area.sum() for t in range(ohc.shape[0])]
                time_series(ts, "Timestep", "Global-mean OHC",
                            "J m⁻²", outdir, "mpaso_ohc_timeseries.png")
                plots.append(os.path.join(outdir, "mpaso_ohc_timeseries.png"))

    if HAS_XR and isinstance(ds, xr.Dataset):
        ds.close()

    _report("MPAS-O vertical reduction", issues)
    return issues, plots


def check_mpassi_derived(rundir, outdir, verbose):
    print("\n=== MPAS-SI Derived Fields ===")
    issues = []
    plots = []

    files = find_files(rundir, "analysis_members/seaice_fmeSeaiceDerivedFields.*.nc")
    if not files:
        print("  SKIP: no seaice_fmeSeaiceDerivedFields files found")
        return issues, plots

    ds = safe_open(files[0])

    for var, ace_name in ACE_ICE_DERIVED.items():
        arr = get_var(ds, var)
        if arr is None:
            issues.append(f"  missing: {var} (ACE: {ace_name})")
            continue
        vmin, vmax = RANGE_CHECKS.get(var, (None, None))
        issues += check_range(arr, var, vmin, vmax)
        if verbose:
            print(f"  {var} ({ace_name}): {summary_stats(arr)}")

    for var in ["surfaceTemperatureMean", "airStressZonal", "airStressMeridional"]:
        arr = get_var(ds, var)
        if arr is None and verbose:
            print(f"  INFO {var}: not present")
        elif arr is not None:
            issues += check_range(arr, var)

    # Maps
    if HAS_MPL:
        lon = get_var(ds, "lonCell")
        lat = get_var(ds, "latCell")
        if lon is not None:
            lon_deg = np.degrees(lon)
            lat_deg = np.degrees(lat)
            for var, cmap, vmin, vmax, units in [
                ("iceAreaTotal",   "Blues",   0, 1, "fraction"),
                ("iceVolumeTotal", "viridis", 0, 5, "m"),
                ("surfaceTemperatureMean", "RdBu_r", 210, 275, "K"),
            ]:
                arr = get_var(ds, var)
                if arr is None:
                    continue
                data = arr[0] if arr.ndim > 1 else arr
                p = global_map(data, lon_deg, lat_deg,
                               f"MPAS-SI {var} (t=0)",
                               cmap=cmap, vmin=vmin, vmax=vmax,
                               outdir=outdir, fname=f"mpassi_{var}.png",
                               units=units)
                if p: plots.append(p)

    if HAS_XR and isinstance(ds, xr.Dataset):
        ds.close()

    _report("MPAS-SI derived fields", issues)
    return issues, plots


def _report(name, issues):
    if issues:
        print(f"  FAIL ({len(issues)} issues):")
        for msg in issues:
            print(msg)
    else:
        print(f"  PASS")


# ─────────────────────────────────────────────────────────────────────────────
# HTML index (named olpp.html to distinguish it from other stuff)
# ─────────────────────────────────────────────────────────────────────────────

def write_html_index(outdir, all_plots, all_issues):
    index = os.path.join(outdir, "olpp.html")
    n_issues = sum(len(v) for v in all_issues.values())
    status = "PASS" if n_issues == 0 else f"FAIL ({n_issues} issues)"
    status_color = "green" if n_issues == 0 else "red"

    rows_issues = ""
    for comp, issues in all_issues.items():
        for msg in issues:
            rows_issues += f"<tr><td>{comp}</td><td style='color:red'>{msg.strip()}</td></tr>\n"

    img_tags = ""
    for p in all_plots:
        if p and os.path.exists(p):
            rel = os.path.basename(p)
            img_tags += (f'<div style="display:inline-block;margin:8px;">'
                         f'<a href="{rel}"><img src="{rel}" width="480" '
                         f'style="border:1px solid #ccc;"/></a>'
                         f'<br/><small>{rel}</small></div>\n')

    html = textwrap.dedent(f"""\
    <!DOCTYPE html><html><head>
    <meta charset="utf-8"/>
    <title>FME Output Verification</title>
    <style>body{{font-family:sans-serif;margin:20px}}
           table{{border-collapse:collapse}}
           td,th{{border:1px solid #ccc;padding:4px 8px}}
           th{{background:#eee}}</style>
    </head><body>
    <h1>FME Output Verification</h1>
    <p>Overall status: <b style="color:{status_color}">{status}</b></p>
    <h2>Issues</h2>
    <table><tr><th>Component</th><th>Issue</th></tr>
    {rows_issues if rows_issues else '<tr><td colspan=2>None</td></tr>'}
    </table>
    <h2>Diagnostic Figures</h2>
    {img_tags if img_tags else '<p>No figures generated (matplotlib/cartopy not available).</p>'}
    </body></html>
    """)
    with open(index, "w") as fh:
        fh.write(html)
    print(f"\nIndex written: {index}")
    return index


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify FME output and produce diagnostic figures for ACE/Samudra.")
    parser.add_argument("--rundir", required=True,
                        help="CIME RUNDIR (or archive/ocn/hist directory)")
    parser.add_argument("--outdir",
                        default="/pscratch/sd/m/mahf708/aifigs_v3LRpiControlSamudrACE",
                        help="Output directory for figures and HTML index")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.rundir):
        sys.exit(f"ERROR: rundir not found: {args.rundir}")

    os.makedirs(args.outdir, exist_ok=True)
    print(f"FME Verification  |  rundir: {args.rundir}")
    print(f"Output directory  :  {args.outdir}")
    print("=" * 70)

    if not HAS_MPL:
        print("WARNING: matplotlib not available — no figures will be produced")
    if not HAS_CARTOPY:
        print("WARNING: cartopy not available — falling back to scatter maps")
    if not (HAS_XR or HAS_NC4):
        sys.exit("ERROR: need xarray or netCDF4")

    all_issues = {}
    all_plots = []

    issues, plots = check_eam(args.rundir, args.outdir, args.verbose)
    all_issues["EAM"] = issues; all_plots += plots

    issues, plots = check_mpaso_depth_coarsening(args.rundir, args.outdir, args.verbose)
    all_issues["MPAS-O depth"] = issues; all_plots += plots

    issues, plots = check_mpaso_derived(args.rundir, args.outdir, args.verbose)
    all_issues["MPAS-O derived"] = issues; all_plots += plots

    issues, plots = check_mpaso_vertical_reduce(args.rundir, args.outdir, args.verbose)
    all_issues["MPAS-O vertreduce"] = issues; all_plots += plots

    issues, plots = check_mpassi_derived(args.rundir, args.outdir, args.verbose)
    all_issues["MPAS-SI"] = issues; all_plots += plots

    write_html_index(args.outdir, all_plots, all_issues)

    n_total = sum(len(v) for v in all_issues.values())
    print("\n" + "=" * 70)
    print(f"RESULT: {'ALL CHECKS PASSED' if n_total == 0 else f'{n_total} ISSUES FOUND'}")
    sys.exit(0 if n_total == 0 else 1)


if __name__ == "__main__":
    main()
