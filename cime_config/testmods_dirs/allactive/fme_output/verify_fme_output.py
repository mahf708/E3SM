#!/usr/bin/env python3
"""
Verify and visualize FME online output against ACE/Samudra requirements.

Checks that all fields required by ACE (github.com/ai2cm/ace, branch exp/e3sm)
are present and physically reasonable, then produces diagnostic figures saved
to an output directory for online browsing.

Verifies both native (unstructured MPAS) and remapped (lat-lon) output files.

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

# -- optional imports ----------------------------------------------------------
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

# -- ACE variable requirements ------------------------------------------------
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
ACE_ATM_3D_COARSENED = ["T", "Q", "U", "V", "OMEGA", "CLDLIQ", "CLDICE", "RAINQM", "TOTAL_WATER"]
N_ATM_LAYERS = 8  # ACE uses 8 pressure layers (1-based: T_1..T_8)

# Fields ACE ocean model needs from MPAS-O fmeDepthCoarsening (19 layers).
ACE_OCN_COARSENED = {
    "temperatureCoarsened": "thetao",
    "salinityCoarsened": "so",
    "velocityZonalCoarsened": "uo",
    "velocityMeridionalCoarsened": "vo",
    "layerThicknessCoarsened": "layerThickness",
}
N_OCN_LAYERS = 19  # ACE uses 19 depth layers

# Fields ACE ocean model needs from MPAS-O fmeDerivedFields.
ACE_OCN_DERIVED = {
    "sst": "sst",
    "sss": "sss",
    "surfaceHeatFluxTotal": "surfaceHeatFluxTotal",
}

# Fields ACE needs from MPAS-O fmeVerticalReduce.
ACE_OCN_VERTREDUCE = {
    "oceanHeatContent": "oceanHeatContent",
    "freshwaterContent": "freshwaterContent",
    "kineticEnergy": "kineticEnergy",
}

# Fields ACE needs from MPAS-SI fmeSeaiceDerivedFields.
ACE_ICE_DERIVED = {
    "iceAreaTotal": "ocean_sea_ice_fraction",
    "iceVolumeTotal": "sea_ice_volume",
    "snowVolumeTotal": "snowVolumeTotal",
    "iceThicknessMean": "iceThicknessMean",
    "surfaceTemperatureMean": "surfaceTemperatureMean",
    "airStressZonal": "airStressZonal",
    "airStressMeridional": "airStressMeridional",
}

# Expected variables in remapped files (lat-lon 360x180 output)
REMAPPED_DERIVED_VARS = ["sst", "sss", "surfaceHeatFluxTotal"]
REMAPPED_SEAICE_VARS = [
    "iceAreaTotal", "iceVolumeTotal", "snowVolumeTotal",
    "iceThicknessMean", "surfaceTemperatureMean",
    "airStressZonal", "airStressMeridional",
]
REMAPPED_DEPTH_COARSENED_BASES = [
    "temperatureCoarsened", "salinityCoarsened",
    "velocityZonalCoarsened", "velocityMeridionalCoarsened",
    "layerThicknessCoarsened",
]
REMAPPED_VERTREDUCE_VARS = [
    "oceanHeatContent", "freshwaterContent", "kineticEnergy",
]

# Physical range checks: (vmin, vmax)
RANGE_CHECKS = {
    "temperatureCoarsened": (-5, 40),
    "salinityCoarsened": (0, 42),
    "velocityZonalCoarsened": (-5, 5),
    "velocityMeridionalCoarsened": (-5, 5),
    "layerThicknessCoarsened": (0, None),
    "sst": (-5, 40),
    "sss": (0, 42),
    "surfaceHeatFluxTotal": (-1000, 1000),
    "iceAreaTotal": (0, 1),
    "iceVolumeTotal": (0, None),
    "snowVolumeTotal": (0, None),
    "iceThicknessMean": (0, None),
    "surfaceTemperatureMean": (180, 320),
    "oceanHeatContent": (None, None),
    "freshwaterContent": (-1000, 1000),
    "kineticEnergy": (0, None),
    "T": (150, 350),
    "Q": (0, 0.05),
    "PS": (40000, 110000),
    "TS": (200, 340),
    "PHIS": (-1000, 60000),
}

# Expected lat-lon grid dimensions for remapped output
EXPECTED_NLON = 360
EXPECTED_NLAT = 180


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def find_files(rundir, pattern, exclude=None):
    """Find files matching glob pattern, optionally excluding a substring.

    Parameters
    ----------
    rundir : str
        Directory to search in.
    pattern : str
        Glob pattern relative to rundir.
    exclude : str or None
        If set, exclude any file whose basename contains this substring.

    Returns
    -------
    list of str
        Sorted list of matching file paths.
    """
    hits = sorted(glob.glob(os.path.join(rundir, pattern)))
    # Always exclude .base restart files
    hits = [f for f in hits if not f.endswith('.base')]
    if exclude:
        hits = [f for f in hits if exclude not in os.path.basename(f)]
    return hits


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


def get_dims(ds):
    """Return dict of dimension names -> sizes."""
    if HAS_XR and isinstance(ds, xr.Dataset):
        return dict(ds.dims)
    if HAS_NC4 and isinstance(ds, NC4Dataset):
        return {d: len(ds.dimensions[d]) for d in ds.dimensions}
    return {}


def get_varnames(ds):
    """Return list of variable names in the dataset."""
    if HAS_XR and isinstance(ds, xr.Dataset):
        return list(ds.data_vars)
    if HAS_NC4 and isinstance(ds, NC4Dataset):
        return list(ds.variables.keys())
    return []


def close_ds(ds):
    """Close a dataset if applicable."""
    if HAS_XR and isinstance(ds, xr.Dataset):
        ds.close()
    elif HAS_NC4 and isinstance(ds, NC4Dataset):
        ds.close()


def valid_data(arr, fill=1e15):
    """Return non-fill, finite values.

    The threshold is 1e15 because remapped output can contain interpolated
    fill artifacts from neighboring land/ocean cells that are very large
    but not exactly _FillValue=1e34.  No geophysical quantity exceeds 1e15.
    """
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


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def savefig(fig, outdir, name, subdir=None):
    """Save figure, optionally in a subdirectory."""
    if subdir:
        d = os.path.join(outdir, subdir)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, name)
    else:
        path = os.path.join(outdir, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def global_map(data, lons, lats, title, cmap="RdBu_r", vmin=None, vmax=None,
               outdir=None, fname=None, units="", subdir=None):
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
            im = ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(),
                               cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
            plt.colorbar(im, ax=ax, shrink=0.6, label=units)
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
        return savefig(fig, outdir, fname, subdir=subdir)
    plt.show()
    return None


def latlon_map(ds, varname, title, cmap="RdBu_r", vmin=None, vmax=None,
               outdir=None, fname=None, units="", subdir=None):
    """Plot a variable from a remapped (lon, lat, Time) dataset."""
    if not HAS_MPL:
        return None
    arr = get_var(ds, varname)
    if arr is None:
        return None
    # Get first timestep if time dimension present
    if arr.ndim == 3:
        data = arr[0, :, :]
    elif arr.ndim == 2:
        data = arr
    else:
        return None

    lon = get_var(ds, "lon")
    lat = get_var(ds, "lat")
    if lon is None or lat is None:
        return None

    if lon.ndim == 1 and lat.ndim == 1:
        lons, lats = np.meshgrid(lon, lat)
    else:
        lons, lats = lon, lat

    return global_map(data, lons, lats, title, cmap=cmap, vmin=vmin, vmax=vmax,
                      outdir=outdir, fname=fname, units=units, subdir=subdir)


def layer_profiles(data3d, label, outdir, fname, ylabel="Level index", subdir=None):
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
    savefig(fig, outdir, fname, subdir=subdir)


def time_series(values, times_label, title, ylabel, outdir, fname, subdir=None):
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
    savefig(fig, outdir, fname, subdir=subdir)


# -----------------------------------------------------------------------------
# Remapped file verification helpers
# -----------------------------------------------------------------------------

def check_remapped_grid(ds, label):
    """Verify that a remapped dataset has the expected lat-lon grid.

    Returns a list of issues (empty if OK).
    """
    issues = []
    dims = get_dims(ds)

    # Check for lon/lat dimensions
    has_lon = "lon" in dims
    has_lat = "lat" in dims
    if not has_lon:
        issues.append(f"  {label}: missing 'lon' dimension")
    if not has_lat:
        issues.append(f"  {label}: missing 'lat' dimension")
    if has_lon and dims["lon"] != EXPECTED_NLON:
        issues.append(f"  {label}: lon={dims['lon']}, expected {EXPECTED_NLON}")
    if has_lat and dims["lat"] != EXPECTED_NLAT:
        issues.append(f"  {label}: lat={dims['lat']}, expected {EXPECTED_NLAT}")

    # Check for Time dimension
    has_time = "Time" in dims or "time" in dims
    if not has_time:
        issues.append(f"  {label}: missing 'Time' dimension")

    if not issues:
        nlon = dims.get("lon", "?")
        nlat = dims.get("lat", "?")
        print(f"  {label}: lat-lon grid OK ({nlon}x{nlat})")

    return issues


def check_remapped_vars(ds, expected_vars, label):
    """Check that all expected variables are present in a remapped dataset."""
    issues = []
    varnames = get_varnames(ds)
    for var in expected_vars:
        if var not in varnames:
            issues.append(f"  {label}: missing variable '{var}'")
    present = [v for v in expected_vars if v in varnames]
    missing = [v for v in expected_vars if v not in varnames]
    if present:
        print(f"  {label}: {len(present)}/{len(expected_vars)} expected variables present")
    if missing:
        print(f"  {label}: MISSING variables: {missing}")
    return issues


# -----------------------------------------------------------------------------
# Per-component verification + visualization
# -----------------------------------------------------------------------------

def check_eam(rundir, outdir, verbose):
    print("\n=== EAM ===")
    issues = []
    plots = []

    # -- tape 1 (h0): averaged 2D fields -----------------------------------
    t1_files = find_files(rundir, "*.eam.h0.*.nc")
    if not t1_files:
        print("  SKIP tape 1: no *.eam.h0.*.nc files found")
    else:
        print(f"  tape1: found {len(t1_files)} h0 file(s)")
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
            for var, cmap, vm, vx, units in [
                ("TS",    "RdBu_r",  220, 320, "K"),
                ("LHFLX","viridis",    0, 300, "W/m2"),
                ("SHFLX","RdBu_r", -100, 200, "W/m2"),
                ("FLUT", "inferno",  100, 320, "W/m2"),
                ("PRECT","Blues",      0, 1e-3,"kg/m2/s"),
                ("ICEFRAC","Blues",    0,   1, "fraction"),
            ]:
                arr = get_var(ds, var)
                if arr is None or arr.size == 0:
                    continue
                data = arr[0] if arr.ndim >= 2 and arr.shape[0] > 0 else arr
                if lat_name and lon_name and data.ndim == 2:
                    lons = ds[lon_name].values
                    lats = ds[lat_name].values
                    if lons.ndim == 1:
                        lons, lats = np.meshgrid(lons, lats)
                    p = global_map(data, lons, lats, f"EAM {var} (tape1, t=0)",
                                   cmap=cmap, vmin=vm, vmax=vx,
                                   outdir=outdir, fname=f"eam_tape1_{var}.png",
                                   units=units)
                    if p: plots.append(p)
        close_ds(ds)

    # -- tape 2 (h1): additional 2D fields ---------------------------------
    t2_files = find_files(rundir, "*.eam.h1.*.nc")
    if not t2_files:
        print("  SKIP tape 2: no *.eam.h1.*.nc files found")
    else:
        print(f"  tape2: found {len(t2_files)} h1 file(s)")
        ds2 = safe_open(t2_files[0])
        varnames = get_varnames(ds2)
        print(f"  tape2: {len(varnames)} variables in file")
        if verbose:
            print(f"  tape2 variables: {varnames[:30]}{'...' if len(varnames)>30 else ''}")
        close_ds(ds2)

    # -- tape 3 (h2): vertically coarsened 3D state -------------------------
    t3_files = find_files(rundir, "*.eam.h2.*.nc")
    if not t3_files:
        print("  SKIP tape 3: no *.eam.h2.*.nc files found")
    else:
        print(f"  tape3: found {len(t3_files)} h2 file(s)")
        ds3 = safe_open(t3_files[0])
        for base in ACE_ATM_3D_COARSENED:
            found_layers = []
            for k in range(1, N_ATM_LAYERS + 1):  # 1-based: T_1..T_8
                vname = f"{base}_{k}"
                arr = get_var(ds3, vname)
                if arr is not None:
                    found_layers.append(k)
                    vmin, vmax = RANGE_CHECKS.get(base, (None, None))
                    issues += check_range(arr, f"tape3/{vname}", vmin, vmax)
            if len(found_layers) == N_ATM_LAYERS:
                print(f"  tape3/{base}: all {N_ATM_LAYERS} layers present")
            elif found_layers:
                print(f"  tape3/{base}: {len(found_layers)}/{N_ATM_LAYERS} layers found "
                      f"(indices {found_layers})")
                issues.append(f"  tape3/{base}: only {len(found_layers)}/{N_ATM_LAYERS} layers")
            else:
                # EAM may name them differently; check alternate patterns
                alts = [v for v in get_varnames(ds3)
                        if v.startswith(base + "_") or v.startswith(base.upper() + "_")]
                if alts:
                    print(f"  tape3/{base}: found alternate names: {alts[:5]}")
                else:
                    issues.append(f"  tape3 missing coarsened field: {base}_1..{base}_{N_ATM_LAYERS}")
        close_ds(ds3)

    _report("EAM", issues)
    return issues, plots


def check_mpaso_depth_coarsening(rundir, outdir, verbose):
    print("\n=== MPAS-O Depth Coarsening (native) ===")
    issues = []
    plots = []

    # Native files: *.mpaso.hist.am.fmeDepthCoarsening.*.nc, excluding .remapped.
    files = find_files(rundir, "*.mpaso.hist.am.fmeDepthCoarsening.*.nc",
                       exclude=".remapped.")
    if not files:
        print("  SKIP: no native fmeDepthCoarsening files found")
        return issues, plots

    print(f"  Found {len(files)} native file(s)")
    ds = safe_open(files[0])

    for var, ace_name in ACE_OCN_COARSENED.items():
        arr = get_var(ds, var)
        if arr is None:
            issues.append(f"  native missing: {var} (ACE: {ace_name})")
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
                           "MPAS-O temperatureCoarsened layer 0 (native)",
                           cmap="RdBu_r", vmin=-2, vmax=30,
                           outdir=outdir, fname="mpaso_temp_surf_native.png", units="degC")
            if p: plots.append(p)

        arr = get_var(ds, "salinityCoarsened")
        if arr is not None and lon is not None:
            lon_deg = np.degrees(lon)
            lat_deg = np.degrees(lat)
            data = arr[0, 0, :]
            p = global_map(data, lon_deg, lat_deg,
                           "MPAS-O salinityCoarsened layer 0 (native)",
                           cmap="viridis", vmin=30, vmax=40,
                           outdir=outdir, fname="mpaso_sal_surf_native.png", units="PSU")
            if p: plots.append(p)

    close_ds(ds)

    _report("MPAS-O depth coarsening (native)", issues)
    return issues, plots


def check_mpaso_depth_coarsening_remapped(rundir, outdir, verbose):
    print("\n=== MPAS-O Depth Coarsening (remapped) ===")
    issues = []
    plots = []

    files = find_files(rundir, "*.mpaso.hist.am.fmeDepthCoarsening.*.remapped.nc")
    if not files:
        print("  SKIP: no remapped fmeDepthCoarsening files found")
        return issues, plots

    print(f"  Found {len(files)} remapped file(s)")
    ds = safe_open(files[0])

    # Verify lat-lon grid
    issues += check_remapped_grid(ds, "depth coarsening remapped")

    # Check for per-level variables: temperatureCoarsened_0..18, etc.
    expected_vars = []
    for base in REMAPPED_DEPTH_COARSENED_BASES:
        for k in range(N_OCN_LAYERS):
            expected_vars.append(f"{base}_{k}")
    issues += check_remapped_vars(ds, expected_vars, "depth coarsening remapped")

    # Range checks on per-level variables
    varnames = get_varnames(ds)
    for base in REMAPPED_DEPTH_COARSENED_BASES:
        found_levels = []
        for k in range(N_OCN_LAYERS):
            vname = f"{base}_{k}"
            if vname in varnames:
                found_levels.append(k)
                arr = get_var(ds, vname)
                vmin, vmax = RANGE_CHECKS.get(base, (None, None))
                issues += check_range(arr, f"remapped/{vname}", vmin, vmax)
                if verbose and k == 0:
                    print(f"  remapped/{vname}: {summary_stats(arr)}")
        if found_levels:
            print(f"  {base}: {len(found_levels)}/{N_OCN_LAYERS} levels in remapped file")

    # Maps of surface fields from remapped data
    if HAS_MPL:
        for base, cmap, vm, vx, units in [
            ("temperatureCoarsened", "RdBu_r", -2, 30, "degC"),
            ("salinityCoarsened", "viridis", 30, 40, "PSU"),
        ]:
            vname = f"{base}_0"
            p = latlon_map(ds, vname,
                           f"MPAS-O {base} layer 0 (remapped)",
                           cmap=cmap, vmin=vm, vmax=vx,
                           outdir=outdir,
                           fname=f"mpaso_depth_{base}_0_remapped.png",
                           units=units)
            if p: plots.append(p)

        # Vertical profile from remapped per-level variables
        for base in ["temperatureCoarsened", "salinityCoarsened"]:
            means = []
            for k in range(N_OCN_LAYERS):
                vname = f"{base}_{k}"
                arr = get_var(ds, vname)
                v = valid_data(arr)
                means.append(v.mean() if v is not None and v.size > 0 else np.nan)
            if any(np.isfinite(m) for m in means):
                fig, ax = plt.subplots(figsize=(4, 5))
                ax.plot(means, np.arange(len(means)), "o-")
                ax.invert_yaxis()
                ax.set_xlabel(base)
                ax.set_ylabel("Depth layer (0=surface)")
                ax.set_title(f"Global-mean profile: {base} (remapped)")
                ax.grid(True, alpha=0.4)
                p = savefig(fig, outdir, f"mpaso_depth_profile_{base}_remapped.png")
                if p: plots.append(p)

    close_ds(ds)

    _report("MPAS-O depth coarsening (remapped)", issues)
    return issues, plots


def check_mpaso_derived(rundir, outdir, verbose):
    print("\n=== MPAS-O Derived Fields (native) ===")
    issues = []
    plots = []

    files = find_files(rundir, "*.mpaso.hist.am.fmeDerivedFields.*.nc",
                       exclude=".remapped.")
    if not files:
        print("  SKIP: no native fmeDerivedFields files found")
        return issues, plots

    print(f"  Found {len(files)} native file(s)")
    ds = safe_open(files[0])

    for var, ace_name in ACE_OCN_DERIVED.items():
        arr = get_var(ds, var)
        if arr is None:
            issues.append(f"  native missing: {var} (ACE: {ace_name})")
            continue
        vmin, vmax = RANGE_CHECKS.get(var, (None, None))
        issues += check_range(arr, var, vmin, vmax)
        if verbose:
            print(f"  {var} ({ace_name}): {summary_stats(arr)}")

    # List all variables for reference
    varnames = get_varnames(ds)
    print(f"  Native file has {len(varnames)} variables")
    if verbose:
        print(f"  Variables: {varnames}")

    # Maps
    if HAS_MPL:
        lon = get_var(ds, "lonCell")
        lat = get_var(ds, "latCell")
        if lon is not None and lat is not None:
            lon_deg = np.degrees(lon)
            lat_deg = np.degrees(lat)
            for var, cmap, vm, vx, units in [
                ("sst",  "RdBu_r",  -2, 30, "degC"),
                ("sss",  "viridis",  30, 40, "PSU"),
                ("surfaceHeatFluxTotal", "RdBu_r", -300, 300, "W/m2"),
            ]:
                arr = get_var(ds, var)
                if arr is None:
                    continue
                data = arr[0] if arr.ndim > 1 else arr
                p = global_map(data, lon_deg, lat_deg,
                               f"MPAS-O {var} (native, t=0)",
                               cmap=cmap, vmin=vm, vmax=vx,
                               outdir=outdir, fname=f"mpaso_derived_{var}_native.png",
                               units=units)
                if p: plots.append(p)

    close_ds(ds)

    _report("MPAS-O derived fields (native)", issues)
    return issues, plots


def check_mpaso_derived_remapped(rundir, outdir, verbose):
    print("\n=== MPAS-O Derived Fields (remapped) ===")
    issues = []
    plots = []

    files = find_files(rundir, "*.mpaso.hist.am.fmeDerivedFields.*.remapped.nc")
    if not files:
        print("  SKIP: no remapped fmeDerivedFields files found")
        return issues, plots

    print(f"  Found {len(files)} remapped file(s)")
    ds = safe_open(files[0])

    # Verify lat-lon grid
    issues += check_remapped_grid(ds, "derived fields remapped")

    # Check expected variables
    issues += check_remapped_vars(ds, REMAPPED_DERIVED_VARS, "derived fields remapped")

    # Range checks
    for var in REMAPPED_DERIVED_VARS:
        arr = get_var(ds, var)
        if arr is not None:
            vmin, vmax = RANGE_CHECKS.get(var, (None, None))
            issues += check_range(arr, f"remapped/{var}", vmin, vmax)
            if verbose:
                print(f"  remapped/{var}: {summary_stats(arr)}")

    # Maps from remapped data
    if HAS_MPL:
        for var, cmap, vm, vx, units in [
            ("sst", "RdBu_r", -2, 30, "degC"),
            ("sss", "viridis", 30, 40, "PSU"),
            ("surfaceHeatFluxTotal", "RdBu_r", -300, 300, "W/m2"),
        ]:
            p = latlon_map(ds, var,
                           f"MPAS-O {var} (remapped, t=0)",
                           cmap=cmap, vmin=vm, vmax=vx,
                           outdir=outdir,
                           fname=f"mpaso_derived_{var}_remapped.png",
                           units=units)
            if p: plots.append(p)

    close_ds(ds)

    _report("MPAS-O derived fields (remapped)", issues)
    return issues, plots


def check_mpaso_vertical_reduce(rundir, outdir, verbose):
    print("\n=== MPAS-O Vertical Reduction (native) ===")
    issues = []
    plots = []

    files = find_files(rundir, "*.mpaso.hist.am.fmeVerticalReduce.*.nc",
                       exclude=".remapped.")
    if not files:
        print("  SKIP: no native fmeVerticalReduce files found")
        return issues, plots

    print(f"  Found {len(files)} native file(s)")
    ds = safe_open(files[0])
    for var, (vmin, vmax) in [("oceanHeatContent", (None, None)),
                               ("freshwaterContent", (-1000, 1000)),
                               ("kineticEnergy", (0, None))]:
        arr = get_var(ds, var)
        if arr is None:
            issues.append(f"  native missing: {var}")
        else:
            issues += check_range(arr, var, vmin, vmax)
            if verbose:
                print(f"  {var}: {summary_stats(arr)}")

    # Time series of global-mean OHC
    if HAS_MPL:
        ohc = get_var(ds, "oceanHeatContent")
        if ohc is not None and ohc.ndim >= 2:
            area = get_var(ds, "areaCell")
            if area is not None:
                ts = [(ohc[t] * area).sum() / area.sum() for t in range(ohc.shape[0])]
                time_series(ts, "Timestep", "Global-mean OHC (native)",
                            "J/m2", outdir, "mpaso_ohc_timeseries_native.png")
                plots.append(os.path.join(outdir, "mpaso_ohc_timeseries_native.png"))

    close_ds(ds)

    _report("MPAS-O vertical reduction (native)", issues)
    return issues, plots


def check_mpaso_vertical_reduce_remapped(rundir, outdir, verbose):
    print("\n=== MPAS-O Vertical Reduction (remapped) ===")
    issues = []
    plots = []

    files = find_files(rundir, "*.mpaso.hist.am.fmeVerticalReduce.*.remapped.nc")
    if not files:
        print("  SKIP: no remapped fmeVerticalReduce files found")
        return issues, plots

    print(f"  Found {len(files)} remapped file(s)")
    ds = safe_open(files[0])

    # Verify lat-lon grid
    issues += check_remapped_grid(ds, "vertical reduce remapped")

    # Check expected variables
    issues += check_remapped_vars(ds, REMAPPED_VERTREDUCE_VARS, "vertical reduce remapped")

    # Range checks
    for var in REMAPPED_VERTREDUCE_VARS:
        arr = get_var(ds, var)
        if arr is not None:
            vmin, vmax = RANGE_CHECKS.get(var, (None, None))
            issues += check_range(arr, f"remapped/{var}", vmin, vmax)
            if verbose:
                print(f"  remapped/{var}: {summary_stats(arr)}")

    # Maps from remapped data
    if HAS_MPL:
        for var, cmap, vm, vx, units in [
            ("oceanHeatContent", "inferno", None, None, "J/m2"),
            ("freshwaterContent", "RdBu_r", -500, 500, "m"),
            ("kineticEnergy", "YlOrRd", 0, None, "J/m2"),
        ]:
            p = latlon_map(ds, var,
                           f"MPAS-O {var} (remapped, t=0)",
                           cmap=cmap, vmin=vm, vmax=vx,
                           outdir=outdir,
                           fname=f"mpaso_vertreduce_{var}_remapped.png",
                           units=units)
            if p: plots.append(p)

    close_ds(ds)

    _report("MPAS-O vertical reduction (remapped)", issues)
    return issues, plots


def check_mpassi_derived(rundir, outdir, verbose):
    print("\n=== MPAS-SI Derived Fields (native) ===")
    issues = []
    plots = []

    files = find_files(rundir, "*.mpassi.hist.am.fmeSeaiceDerivedFields.*.nc",
                       exclude=".remapped.")
    if not files:
        print("  SKIP: no native fmeSeaiceDerivedFields files found")
        return issues, plots

    print(f"  Found {len(files)} native file(s)")
    ds = safe_open(files[0])

    for var, ace_name in ACE_ICE_DERIVED.items():
        arr = get_var(ds, var)
        if arr is None:
            issues.append(f"  native missing: {var} (ACE: {ace_name})")
            continue
        vmin, vmax = RANGE_CHECKS.get(var, (None, None))
        issues += check_range(arr, var, vmin, vmax)
        if verbose:
            print(f"  {var} ({ace_name}): {summary_stats(arr)}")

    # Maps
    if HAS_MPL:
        lon = get_var(ds, "lonCell")
        lat = get_var(ds, "latCell")
        if lon is not None and lat is not None:
            lon_deg = np.degrees(lon)
            lat_deg = np.degrees(lat)
            for var, cmap, vm, vx, units in [
                ("iceAreaTotal",   "Blues",   0, 1, "fraction"),
                ("iceVolumeTotal", "viridis", 0, 5, "m"),
                ("surfaceTemperatureMean", "RdBu_r", 210, 275, "K"),
            ]:
                arr = get_var(ds, var)
                if arr is None:
                    continue
                data = arr[0] if arr.ndim > 1 else arr
                p = global_map(data, lon_deg, lat_deg,
                               f"MPAS-SI {var} (native, t=0)",
                               cmap=cmap, vmin=vm, vmax=vx,
                               outdir=outdir, fname=f"mpassi_{var}_native.png",
                               units=units)
                if p: plots.append(p)

    close_ds(ds)

    _report("MPAS-SI derived fields (native)", issues)
    return issues, plots


def check_mpassi_derived_remapped(rundir, outdir, verbose):
    print("\n=== MPAS-SI Derived Fields (remapped) ===")
    issues = []
    plots = []

    files = find_files(rundir, "*.mpassi.hist.am.fmeSeaiceDerivedFields.*.remapped.nc")
    if not files:
        print("  SKIP: no remapped fmeSeaiceDerivedFields files found")
        return issues, plots

    print(f"  Found {len(files)} remapped file(s)")
    ds = safe_open(files[0])

    # Verify lat-lon grid
    issues += check_remapped_grid(ds, "sea ice derived remapped")

    # Check expected variables
    issues += check_remapped_vars(ds, REMAPPED_SEAICE_VARS, "sea ice derived remapped")

    # Range checks
    for var in REMAPPED_SEAICE_VARS:
        arr = get_var(ds, var)
        if arr is not None:
            vmin, vmax = RANGE_CHECKS.get(var, (None, None))
            issues += check_range(arr, f"remapped/{var}", vmin, vmax)
            if verbose:
                print(f"  remapped/{var}: {summary_stats(arr)}")

    # Maps from remapped data
    if HAS_MPL:
        for var, cmap, vm, vx, units in [
            ("iceAreaTotal", "Blues", 0, 1, "fraction"),
            ("iceVolumeTotal", "viridis", 0, 5, "m"),
            ("surfaceTemperatureMean", "RdBu_r", 210, 275, "K"),
            ("snowVolumeTotal", "PuBu", 0, 2, "m"),
        ]:
            p = latlon_map(ds, var,
                           f"MPAS-SI {var} (remapped, t=0)",
                           cmap=cmap, vmin=vm, vmax=vx,
                           outdir=outdir,
                           fname=f"mpassi_{var}_remapped.png",
                           units=units)
            if p: plots.append(p)

    close_ds(ds)

    _report("MPAS-SI derived fields (remapped)", issues)
    return issues, plots


def _report(name, issues):
    if issues:
        print(f"  FAIL ({len(issues)} issues):")
        for msg in issues:
            print(msg)
    else:
        print(f"  PASS")


# -----------------------------------------------------------------------------
# Timing summary
# -----------------------------------------------------------------------------

def read_timing_summary(rundir):
    """Read model_timing_stats if present and return summary lines."""
    timing_path = os.path.join(rundir, "timing", "model_timing_stats")
    if not os.path.exists(timing_path):
        return None

    try:
        with open(timing_path, "r") as fh:
            lines = fh.readlines()
    except Exception:
        return None

    if not lines:
        return None

    # Extract key timing info: look for overall model time and component times
    summary = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Keep header lines and lines with timing data
        # Typical format: component-name  seconds  seconds  seconds
        if any(kw in stripped.lower() for kw in [
            "total", "atm", "lnd", "ocn", "ice", "cpl", "rof", "glc", "wav",
            "init", "run", "final", "overall", "model",
        ]):
            summary.append(stripped)
        elif len(summary) == 0:
            # Keep initial header/description lines
            summary.append(stripped)

    return summary[:40] if summary else None


# -----------------------------------------------------------------------------
# HTML index
# -----------------------------------------------------------------------------

def write_html_index(outdir, all_plots_by_comp, all_issues, timing_summary=None):
    index = os.path.join(outdir, "fme_output.html")
    n_issues = sum(len(v) for v in all_issues.values())
    n_pass = sum(1 for v in all_issues.values() if not v)
    n_total = len(all_issues)
    status = "ALL PASS" if n_issues == 0 else f"{n_issues} issues in {n_total - n_pass}/{n_total} components"
    status_color = "#2a2" if n_issues == 0 else "#c33"

    # Summary table
    summary_rows = ""
    for comp, issues in all_issues.items():
        if not issues:
            summary_rows += (f'<tr><td>{comp}</td><td class="pass">PASS</td>'
                             f'<td>0</td></tr>\n')
        else:
            summary_rows += (f'<tr><td>{comp}</td><td class="fail">FAIL</td>'
                             f'<td>{len(issues)}</td></tr>\n')

    # Detailed issues (collapsible)
    detail_rows = ""
    for comp, issues in all_issues.items():
        for msg in issues:
            detail_rows += f'<tr><td>{comp}</td><td>{msg.strip()}</td></tr>\n'

    # Group plots: pair native/remapped side by side where possible
    fig_sections = ""
    # Identify paired components
    paired = [
        ("MPAS-O Depth Coarsening", "native", "remapped"),
        ("MPAS-O Derived Fields", "native", "remapped"),
        ("MPAS-O Vertical Reduce", "native", "remapped"),
        ("MPAS-SI Derived Fields", "native", "remapped"),
    ]
    shown_comps = set()

    for base_name, native_suffix, remap_suffix in paired:
        native_key = f"{base_name} ({native_suffix})"
        remap_key = f"{base_name} ({remap_suffix})"
        native_plots = all_plots_by_comp.get(native_key, [])
        remap_plots = all_plots_by_comp.get(remap_key, [])
        if not native_plots and not remap_plots:
            continue
        anchor = base_name.replace(" ", "_").replace("-", "_")
        fig_sections += f'<h3 id="{anchor}">{base_name}</h3>\n'
        fig_sections += '<div class="comparison">\n'
        if native_plots:
            fig_sections += '<div class="col"><h4>Native Grid</h4>\n'
            for p in native_plots:
                if p and os.path.exists(p):
                    rel = os.path.relpath(p, outdir)
                    fig_sections += (f'<div class="fig"><a href="{rel}">'
                                     f'<img src="{rel}"/></a>'
                                     f'<span>{os.path.basename(p)}</span></div>\n')
            fig_sections += '</div>\n'
        if remap_plots:
            fig_sections += '<div class="col"><h4>Remapped (lat-lon)</h4>\n'
            for p in remap_plots:
                if p and os.path.exists(p):
                    rel = os.path.relpath(p, outdir)
                    fig_sections += (f'<div class="fig"><a href="{rel}">'
                                     f'<img src="{rel}"/></a>'
                                     f'<span>{os.path.basename(p)}</span></div>\n')
            fig_sections += '</div>\n'
        fig_sections += '</div>\n'
        shown_comps.add(native_key)
        shown_comps.add(remap_key)

    # Show remaining (unpaired) components
    for comp, plots in all_plots_by_comp.items():
        if comp in shown_comps or not plots:
            continue
        anchor = comp.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
        fig_sections += f'<h3 id="{anchor}">{comp}</h3>\n<div class="gallery">\n'
        for p in plots:
            if p and os.path.exists(p):
                rel = os.path.relpath(p, outdir)
                fig_sections += (f'<div class="fig"><a href="{rel}">'
                                 f'<img src="{rel}"/></a>'
                                 f'<span>{os.path.basename(p)}</span></div>\n')
        fig_sections += '</div>\n'

    # Navigation
    nav_links = ""
    nav_items = ["Summary"]
    for base_name, _, _ in paired:
        nav_items.append(base_name)
    for comp in all_plots_by_comp:
        if comp not in shown_comps and all_plots_by_comp[comp]:
            nav_items.append(comp)
    if timing_summary:
        nav_items.append("Timing")
    for item in nav_items:
        anchor = item.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
        nav_links += f'<a href="#{anchor}">{item}</a>\n'

    timing_html = ""
    if timing_summary:
        timing_lines = "\n".join(timing_summary[:30])
        timing_html = f'''
    <h2 id="Timing">Performance Timing</h2>
    <pre class="timing">{timing_lines}</pre>
'''

    html = textwrap.dedent(f"""\
    <!DOCTYPE html><html><head>
    <meta charset="utf-8"/>
    <title>FME Output Verification</title>
    <style>
      * {{ box-sizing: border-box; }}
      body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
             margin: 0; padding: 20px 30px; background: #fafafa; color: #333; }}
      h1 {{ color: #1a3a5c; border-bottom: 3px solid #1a3a5c; padding-bottom: 8px; }}
      h2 {{ color: #1a3a5c; margin-top: 35px; border-bottom: 2px solid #ddd; padding-bottom: 6px; }}
      h3 {{ color: #444; margin-top: 25px; }}
      h4 {{ color: #666; margin: 8px 0 4px; font-size: 0.95em; }}
      .status {{ font-size: 1.3em; font-weight: bold; color: {status_color}; }}
      nav {{ background: #fff; padding: 12px 16px; border-radius: 6px;
             box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 15px 0;
             display: flex; flex-wrap: wrap; gap: 8px; }}
      nav a {{ color: #1a3a5c; text-decoration: none; padding: 4px 10px;
               background: #e8eef4; border-radius: 4px; font-size: 0.9em; }}
      nav a:hover {{ background: #1a3a5c; color: #fff; }}
      table {{ border-collapse: collapse; width: 100%; background: #fff;
               box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
      td, th {{ border: 1px solid #ddd; padding: 6px 12px; text-align: left; }}
      th {{ background: #f0f3f6; font-weight: 600; }}
      .pass {{ color: #2a2; font-weight: bold; }}
      .fail {{ color: #c33; font-weight: bold; }}
      .comparison {{ display: flex; gap: 20px; flex-wrap: wrap; }}
      .col {{ flex: 1; min-width: 400px; }}
      .gallery {{ display: flex; flex-wrap: wrap; gap: 10px; }}
      .fig {{ display: inline-block; margin: 4px; vertical-align: top; }}
      .fig img {{ width: 400px; border: 1px solid #ccc; border-radius: 4px;
                  transition: transform 0.2s; }}
      .fig img:hover {{ transform: scale(1.02); box-shadow: 0 2px 8px rgba(0,0,0,0.15); }}
      .fig span {{ display: block; font-size: 0.75em; color: #888; margin-top: 2px; }}
      details {{ margin: 10px 0; }}
      summary {{ cursor: pointer; font-weight: 600; color: #1a3a5c; }}
      .timing {{ background: #fff; padding: 12px; border: 1px solid #ddd;
                 border-radius: 4px; font-size: 11px; overflow-x: auto;
                 max-height: 400px; overflow-y: auto; }}
      footer {{ margin-top: 40px; padding-top: 10px; border-top: 1px solid #ddd;
                color: #999; font-size: 0.8em; }}
    </style>
    </head><body>
    <h1>FME Online Output Verification</h1>
    <p class="status">{status}</p>

    <nav>{nav_links}</nav>

    <h2 id="Summary">Component Summary</h2>
    <table>
    <tr><th>Component</th><th>Status</th><th>Issues</th></tr>
    {summary_rows}
    </table>

    {"<details><summary>Show all " + str(n_issues) + " issues</summary><table><tr><th>Component</th><th>Issue</th></tr>" + detail_rows + "</table></details>" if detail_rows else ""}

    <h2>Diagnostic Figures</h2>
    {fig_sections if fig_sections else '<p>No figures generated.</p>'}

    {timing_html}
    <footer>Generated by verify_fme_output.py &mdash; FME Online Output Processing for E3SM</footer>
    </body></html>
    """)
    with open(index, "w") as fh:
        fh.write(html)
    print(f"\nIndex written: {index}")
    return index


# -----------------------------------------------------------------------------
# File inventory
# -----------------------------------------------------------------------------

def print_file_inventory(rundir):
    """Print a summary of all FME-related files in the run directory."""
    print("\n=== File Inventory ===")
    categories = [
        ("EAM h0 (tape1)",             "*.eam.h0.*.nc",    None),
        ("EAM h1 (tape2)",             "*.eam.h1.*.nc",    None),
        ("EAM h2 (tape3)",             "*.eam.h2.*.nc",    None),
        ("MPAS-O depth coarsening",    "*.mpaso.hist.am.fmeDepthCoarsening.*.nc",      ".remapped."),
        ("MPAS-O depth coarsening (R)","*.mpaso.hist.am.fmeDepthCoarsening.*.remapped.nc", None),
        ("MPAS-O derived fields",      "*.mpaso.hist.am.fmeDerivedFields.*.nc",        ".remapped."),
        ("MPAS-O derived fields (R)",  "*.mpaso.hist.am.fmeDerivedFields.*.remapped.nc", None),
        ("MPAS-O vertical reduce",     "*.mpaso.hist.am.fmeVerticalReduce.*.nc",       ".remapped."),
        ("MPAS-O vertical reduce (R)", "*.mpaso.hist.am.fmeVerticalReduce.*.remapped.nc", None),
        ("MPAS-SI derived fields",     "*.mpassi.hist.am.fmeSeaiceDerivedFields.*.nc", ".remapped."),
        ("MPAS-SI derived fields (R)", "*.mpassi.hist.am.fmeSeaiceDerivedFields.*.remapped.nc", None),
    ]
    total = 0
    for label, pattern, exclude in categories:
        hits = find_files(rundir, pattern, exclude=exclude)
        total += len(hits)
        status = f"{len(hits)} file(s)" if hits else "NONE"
        print(f"  {label:38s} {status}")
    print(f"  {'TOTAL':38s} {total} FME-related file(s)")
    return total


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

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
        print("WARNING: matplotlib not available -- no figures will be produced")
    if not HAS_CARTOPY:
        print("WARNING: cartopy not available -- falling back to scatter maps")
    if not (HAS_XR or HAS_NC4):
        sys.exit("ERROR: need xarray or netCDF4")

    # File inventory
    print_file_inventory(args.rundir)

    all_issues = {}
    all_plots_by_comp = {}

    # All figures go under fme_output/ subdir so fme_output.html can reference them
    fig_root = os.path.join(args.outdir, "fme_output")
    os.makedirs(fig_root, exist_ok=True)

    def run_check(name, func, subdir, *a):
        comp_outdir = os.path.join(fig_root, subdir)
        os.makedirs(comp_outdir, exist_ok=True)
        issues, plots = func(*a[:1], comp_outdir, *a[1:])
        all_issues[name] = issues
        all_plots_by_comp[name] = plots

    run_check("EAM", check_eam, "eam",
              args.rundir, args.verbose)
    run_check("MPAS-O Depth Coarsening (native)", check_mpaso_depth_coarsening, "mpaso_native",
              args.rundir, args.verbose)
    run_check("MPAS-O Depth Coarsening (remapped)", check_mpaso_depth_coarsening_remapped, "mpaso_remapped",
              args.rundir, args.verbose)
    run_check("MPAS-O Derived Fields (native)", check_mpaso_derived, "mpaso_native",
              args.rundir, args.verbose)
    run_check("MPAS-O Derived Fields (remapped)", check_mpaso_derived_remapped, "mpaso_remapped",
              args.rundir, args.verbose)
    run_check("MPAS-O Vertical Reduce (native)", check_mpaso_vertical_reduce, "mpaso_native",
              args.rundir, args.verbose)
    run_check("MPAS-O Vertical Reduce (remapped)", check_mpaso_vertical_reduce_remapped, "mpaso_remapped",
              args.rundir, args.verbose)
    run_check("MPAS-SI Derived Fields (native)", check_mpassi_derived, "mpassi_native",
              args.rundir, args.verbose)
    run_check("MPAS-SI Derived Fields (remapped)", check_mpassi_derived_remapped, "mpassi_remapped",
              args.rundir, args.verbose)

    # Timing summary
    timing_summary = read_timing_summary(args.rundir)
    if timing_summary:
        print("\n=== Performance Timing ===")
        for line in timing_summary[:10]:
            print(f"  {line}")
        if len(timing_summary) > 10:
            print(f"  ... ({len(timing_summary) - 10} more lines in HTML report)")

    write_html_index(args.outdir, all_plots_by_comp, all_issues,
                     timing_summary=timing_summary)

    n_total = sum(len(v) for v in all_issues.values())
    print("\n" + "=" * 70)
    print(f"RESULT: {'ALL CHECKS PASSED' if n_total == 0 else f'{n_total} ISSUES FOUND'}")
    sys.exit(0 if n_total == 0 else 1)


if __name__ == "__main__":
    main()
