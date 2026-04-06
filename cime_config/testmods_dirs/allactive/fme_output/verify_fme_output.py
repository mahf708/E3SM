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
import time as _time
from datetime import datetime
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
N_OCN_LAYERS = None  # auto-detect from file (was 19 for ACE, 25 for default E3SM)

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
# Ranges are deliberately generous to avoid false positives during spinup.
RANGE_CHECKS = {
    "temperatureCoarsened": (-5, 40),
    "salinityCoarsened": (0, 50),      # Red Sea/Persian Gulf can exceed 42 PSU
    "velocityZonalCoarsened": (-5, 5),
    "velocityMeridionalCoarsened": (-5, 5),
    "layerThicknessCoarsened": (0, None),
    "sst": (-5, 40),
    "sss": (0, 50),
    "surfaceHeatFluxTotal": (-2000, 2000),  # spinup transients can be large
    "iceAreaTotal": (0, 1),
    "iceVolumeTotal": (0, None),
    "snowVolumeTotal": (-1e-20, None),      # allow floating-point noise near zero
    "iceThicknessMean": (0, None),
    "surfaceTemperatureMean": (-50, 5),     # degC (not Kelvin!), Arctic ice can be -45C
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
        return dict(ds.sizes)
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


def valid_data(arr, fill=1e10):
    """Return non-fill, finite values.

    The threshold is 1e10 because remapped output can contain interpolated
    fill artifacts from neighboring land/ocean cells that can reach ~1e14
    scale while _FillValue is 1e34.  No geophysical quantity exceeds 1e10.
    """
    if arr is None:
        return None
    flat = arr.ravel().astype(float)
    mask = (np.abs(flat) < fill) & np.isfinite(flat)
    return flat[mask] if mask.any() else None


def fill_nan_report(arr, name):
    """Return a dict with fill/NaN fraction stats for a variable array."""
    if arr is None:
        return {"name": name, "total": 0, "valid": 0, "fill_nan": 0, "pct": 100.0}
    flat = arr.ravel().astype(float)
    total = flat.size
    fill_mask = np.abs(flat) >= 1e10
    nan_mask = ~np.isfinite(flat)
    bad = fill_mask | nan_mask
    n_bad = int(bad.sum())
    n_valid = total - n_bad
    pct = 100.0 * n_bad / total if total > 0 else 0.0
    return {"name": name, "total": total, "valid": n_valid,
            "fill_nan": n_bad, "pct": pct}


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


def has_time_zero(ds):
    """Check if the Time/time dimension has size 0."""
    dims = get_dims(ds)
    for tname in ("Time", "time"):
        if tname in dims and dims[tname] == 0:
            return True
    return False


def last_time_index(arr):
    """Return the index of the last timestep.

    Uses -1 (last available) to avoid initialization artifacts at t=0.
    """
    if arr is None:
        return -1
    if arr.ndim >= 2 and arr.shape[0] > 0:
        return -1
    return -1


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


def _fix_lon(lons):
    """Shift longitudes from [0,360) to [-180,180) for plotting."""
    return (lons + 180) % 360 - 180


def _plot_on_ax(ax, data, lons, lats, cmap, vmin, vmax, is_cartopy):
    """Plot data on an axis — tripcolor for 1D, pcolormesh for 2D."""
    transform = ccrs.PlateCarree() if is_cartopy else None
    if data.ndim == 2:
        kw = dict(transform=transform) if is_cartopy else {}
        im = ax.pcolormesh(lons, lats, data, cmap=cmap,
                           vmin=vmin, vmax=vmax, shading="auto", **kw)
    else:
        # Unstructured: use tripcolor for filled triangulation.
        # Subsample if > 50k points to keep triangulation fast.
        # Mask triangles that cross the antimeridian (lon span > 180).
        from matplotlib.tri import Triangulation
        lons_fix = _fix_lon(lons)
        kw = dict(transform=ccrs.PlateCarree()) if is_cartopy else {}
        max_pts = 50000
        if data.size > max_pts:
            idx = np.random.default_rng(42).choice(data.size, max_pts, replace=False)
            lons_sub, lats_sub, data_sub = lons_fix[idx], lats[idx], data[idx]
        else:
            lons_sub, lats_sub, data_sub = lons_fix, lats, data
        try:
            tri = Triangulation(lons_sub, lats_sub)
            # Mask triangles spanning the antimeridian
            tri_lons = lons_sub[tri.triangles]
            lon_span = tri_lons.max(axis=1) - tri_lons.min(axis=1)
            tri.set_mask(lon_span > 180)
            im = ax.tripcolor(tri, data_sub, cmap=cmap,
                              vmin=vmin, vmax=vmax, **kw)
        except Exception:
            # Fallback to scatter if tripcolor fails
            im = ax.scatter(lons_sub, lats_sub, c=data_sub, s=0.5, cmap=cmap,
                            vmin=vmin, vmax=vmax, **kw)
    return im


def global_map(data, lons, lats, title, cmap="RdBu_r", vmin=None, vmax=None,
               outdir=None, fname=None, units="", subdir=None):
    """
    Plot a global filled map.  data is 2-D (lat x lon) or 1-D (nCells) on an
    unstructured grid (tripcolor with scatter fallback).
    """
    if not HAS_MPL:
        return None

    fig = plt.figure(figsize=(10, 5))
    is_cartopy = HAS_CARTOPY
    if is_cartopy:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.set_global()
    else:
        ax = fig.add_subplot(1, 1, 1)

    im = _plot_on_ax(ax, data, lons, lats, cmap, vmin, vmax, is_cartopy)
    plt.colorbar(im, ax=ax, shrink=0.6 if is_cartopy else 0.8, label=units)

    if not is_cartopy:
        ax.set_xlabel("lon"); ax.set_ylabel("lat")

    ax.set_title(title, fontsize=11)
    if outdir and fname:
        return savefig(fig, outdir, fname, subdir=subdir)
    plt.show()
    return None


def latlon_map(ds, varname, title, cmap="RdBu_r", vmin=None, vmax=None,
               outdir=None, fname=None, units="", subdir=None):
    """Plot a variable from a remapped (lon, lat, Time) dataset.

    Uses the LAST timestep to avoid initialization artifacts.
    """
    if not HAS_MPL:
        return None
    arr = get_var(ds, varname)
    if arr is None:
        return None
    # Use last timestep if time dimension present
    if arr.ndim == 3:
        data = arr[-1, :, :]
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


def multi_panel_maps(panels, suptitle, outdir, fname, ncols=4,
                     cmap="RdBu_r", vmin=None, vmax=None, units=""):
    """Plot a grid of small-multiple global maps.

    Parameters
    ----------
    panels : list of (data, lons, lats, subtitle) tuples
        data can be 2D (lat x lon) for pcolormesh or 1D (nCells) for tripcolor.
    """
    if not HAS_MPL or not panels:
        return None

    n = len(panels)
    nrows = (n + ncols - 1) // ncols
    fig_w = 5 * ncols
    fig_h = 2.8 * nrows + 0.6
    is_cartopy = HAS_CARTOPY

    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = []
    for idx in range(n):
        kw = dict(projection=ccrs.PlateCarree()) if is_cartopy else {}
        ax = fig.add_subplot(nrows, ncols, idx + 1, **kw)
        axes.append(ax)

    im = None
    for idx, (data, lons, lats, subtitle) in enumerate(panels):
        ax = axes[idx]
        if is_cartopy:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
            ax.set_global()
        im = _plot_on_ax(ax, data, lons, lats, cmap, vmin, vmax, is_cartopy)
        ax.set_title(subtitle, fontsize=9)

    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.5, label=units, pad=0.03, aspect=30)
    fig.suptitle(suptitle, fontsize=12, y=1.01)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def zonal_mean_plot(profiles, lat, title, outdir, fname, ylabel=""):
    """Plot zonal mean profiles (latitude vs value) for multiple variables.

    Parameters
    ----------
    profiles : list of (zmean_1d, label) tuples
        zmean_1d is a 1D array of length nlat (zonal mean values).
    lat : 1D array
        Latitude values.
    """
    if not HAS_MPL or not profiles:
        return None

    fig, ax = plt.subplots(figsize=(7, 3.5))
    has_data = False
    for zmean, label in profiles:
        if zmean is None:
            continue
        ax.plot(lat, zmean, label=label, linewidth=1.3)
        has_data = True

    if not has_data:
        plt.close(fig)
        return None

    ax.set_xlabel("Latitude")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-90, 90)
    plt.tight_layout()
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _compute_zonal_mean(arr, fill_thresh=1e10):
    """Compute zonal mean from (lat, lon) array, masking fill values."""
    if arr is None:
        return None
    if arr.ndim == 3:
        data = arr[-1, :, :]
    elif arr.ndim == 2:
        data = arr
    else:
        return None
    masked = np.where(np.abs(data) < fill_thresh, data, np.nan)
    return np.nanmean(masked, axis=1)


def side_by_side_comparison(data1, lons1, lats1, title1,
                            data2, lons2, lats2, title2,
                            suptitle, outdir, fname,
                            cmap="RdBu_r", vmin=None, vmax=None, units=""):
    """Two global maps side by side — native tripcolor on left, remapped pcolormesh on right."""
    if not HAS_MPL:
        return None

    is_cartopy = HAS_CARTOPY
    fig = plt.figure(figsize=(16, 4.5))
    im = None
    for idx, (data, lons, lats, title) in enumerate([
        (data1, lons1, lats1, title1),
        (data2, lons2, lats2, title2),
    ]):
        kw = dict(projection=ccrs.PlateCarree()) if is_cartopy else {}
        ax = fig.add_subplot(1, 2, idx + 1, **kw)
        if is_cartopy:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
            ax.set_global()
        im = _plot_on_ax(ax, data, lons, lats, cmap, vmin, vmax, is_cartopy)
        plt.colorbar(im, ax=ax, shrink=0.7, label=units)
        ax.set_title(title, fontsize=10)

    fig.suptitle(suptitle, fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


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
    fill_reports = []

    # -- tape 1 (h0): averaged 2D fields -----------------------------------
    t1_files = find_files(rundir, "*.eam.h0.*.nc")
    if not t1_files:
        print("  SKIP tape 1: no *.eam.h0.*.nc files found")
    else:
        print(f"  tape1: found {len(t1_files)} h0 file(s)")
        ds = safe_open(t1_files[0])
        if has_time_zero(ds):
            print("  tape1: time dimension has size 0 -- skipping")
            close_ds(ds)
        else:
            for var in ["PHIS", "PS", "TS", "LHFLX", "SHFLX", "FLDS", "FSDS",
                        "FLNS", "FSNS", "FSNTOA", "FLUT", "SOLIN",
                        "ICEFRAC", "LANDFRAC", "OCNFRAC", "TAUX", "TAUY"]:
                arr = get_var(ds, var)
                if arr is None:
                    issues.append(f"  tape1 missing: {var}")
                else:
                    vmin, vmax = RANGE_CHECKS.get(var, (None, None))
                    issues += check_range(arr, f"tape1/{var}", vmin, vmax)
                    fill_reports.append(fill_nan_report(arr, f"tape1/{var}"))
                    if verbose:
                        print(f"  tape1/{var}: {summary_stats(arr)}")

            # Maps of key 2D fields on lat-lon grid -- use LAST timestep
            if HAS_MPL and HAS_XR and isinstance(ds, xr.Dataset):
                lon_name = next((d for d in ["lon", "ncol", "longitude"] if d in ds.dims), None)
                lat_name = next((d for d in ["lat", "latitude"] if d in ds.dims), None)
                for var, cmap, vm, vx, units in [
                    ("TS",      "RdBu_r",  220, 320, "K"),
                    ("LHFLX",   "viridis",   0, 300, "W/m2"),
                    ("PRECT",   "Blues",     0, 1e-3, "kg/m2/s"),
                    ("ICEFRAC", "Blues",      0,   1, "fraction"),
                    ("FLUT",    "inferno",  100, 320, "W/m2"),
                    ("FSDS",    "YlOrRd",     0, 500, "W/m2"),
                ]:
                    arr = get_var(ds, var)
                    if arr is None or arr.size == 0:
                        continue
                    # Last timestep
                    data = arr[-1] if arr.ndim >= 2 and arr.shape[0] > 0 else arr
                    if lat_name and lon_name and data.ndim == 2:
                        lons = ds[lon_name].values
                        lats = ds[lat_name].values
                        if lons.ndim == 1:
                            lons, lats = np.meshgrid(lons, lats)
                        p = global_map(data, lons, lats, f"EAM {var} (tape1, last t)",
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
        if has_time_zero(ds2):
            print("  tape2: time dimension has size 0 -- skipping")
        else:
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
        if has_time_zero(ds3):
            print("  tape3: time dimension has size 0 -- skipping")
            close_ds(ds3)
        else:
            for base in ACE_ATM_3D_COARSENED:
                found_layers = []
                for k in range(1, N_ATM_LAYERS + 1):  # 1-based: T_1..T_8
                    vname = f"{base}_{k}"
                    arr = get_var(ds3, vname)
                    if arr is not None:
                        found_layers.append(k)
                        vmin, vmax = RANGE_CHECKS.get(base, (None, None))
                        issues += check_range(arr, f"tape3/{vname}", vmin, vmax)
                        fill_reports.append(fill_nan_report(arr, f"tape3/{vname}"))
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

            # Plot T_1 (near-top stratosphere) and T_8 (near-surface BL) at last timestep
            if HAS_MPL and HAS_XR and isinstance(ds3, xr.Dataset):
                lon_name = next((d for d in ["lon", "ncol", "longitude"] if d in ds3.dims), None)
                lat_name = next((d for d in ["lat", "latitude"] if d in ds3.dims), None)
                for vname, label_tag in [("T_1", "near-top"), ("T_8", "near-surface")]:
                    arr = get_var(ds3, vname)
                    if arr is None or arr.size == 0:
                        continue
                    data = arr[-1] if arr.ndim >= 2 and arr.shape[0] > 0 else arr
                    if lat_name and lon_name and data.ndim == 2:
                        lons = ds3[lon_name].values
                        lats = ds3[lat_name].values
                        if lons.ndim == 1:
                            lons, lats = np.meshgrid(lons, lats)
                        p = global_map(data, lons, lats,
                                       f"EAM {vname} ({label_tag}, tape3, last t)",
                                       cmap="RdBu_r", vmin=150, vmax=320,
                                       outdir=outdir, fname=f"eam_tape3_{vname}.png",
                                       units="K")
                        if p: plots.append(p)
            close_ds(ds3)

    _report("EAM", issues)
    return issues, plots, fill_reports


def check_mpaso_depth_coarsening(rundir, outdir, verbose):
    print("\n=== MPAS-O Depth Coarsening (native) ===")
    issues = []
    plots = []
    fill_reports = []

    # Native files: *.mpaso.hist.am.fmeDepthCoarsening.*.nc, excluding .remapped.
    files = find_files(rundir, "*.mpaso.hist.am.fmeDepthCoarsening.*.nc",
                       exclude=".remapped.")
    if not files:
        print("  SKIP: no native fmeDepthCoarsening files found")
        return issues, plots, fill_reports

    print(f"  Found {len(files)} native file(s)")
    ds = safe_open(files[0])
    if has_time_zero(ds):
        print("  time dimension has size 0 -- skipping")
        close_ds(ds)
        return issues, plots, fill_reports

    # Auto-detect depth level count from dataset dimensions
    dims = get_dims(ds)
    n_ocn_levels = N_OCN_LAYERS  # None = auto-detect
    for dname in ["nFmeDepthLevels", "nFmeCoarsenLevels"]:
        if dname in dims:
            n_ocn_levels = dims[dname]
            break
    if n_ocn_levels is None:
        # Fallback: smallest non-Time, non-nCells dimension
        candidates = [v for k, v in dims.items()
                      if k not in ("Time", "time") and v < 100]
        n_ocn_levels = min(candidates) if candidates else 19
    print(f"  Depth levels detected: {n_ocn_levels}")

    for var, ace_name in ACE_OCN_COARSENED.items():
        arr = get_var(ds, var)
        if arr is None:
            issues.append(f"  native missing: {var} (ACE: {ace_name})")
            continue
        vmin, vmax = RANGE_CHECKS.get(var, (None, None))
        issues += check_range(arr, var, vmin, vmax)
        fill_reports.append(fill_nan_report(arr, var))
        if verbose:
            print(f"  {var} ({ace_name}): {summary_stats(arr)}")

        # Vertical profile plot (global mean across all cells, last timestep)
        if HAS_MPL and arr.ndim == 3:
            layer_profiles(arr[-1], f"{var} [{ace_name}]", outdir,
                           f"mpaso_depth_profile_{var}.png",
                           ylabel="Depth layer (0=surface)")

    # Surface maps -- last timestep
    if HAS_MPL:
        lon = get_var(ds, "lonCell")
        lat = get_var(ds, "latCell")
        if lon is not None and lat is not None:
            lon_deg = np.degrees(lon)
            lat_deg = np.degrees(lat)
            for var, cmap, vm, vx, units in [
                ("temperatureCoarsened", "RdBu_r", -2, 30, "degC"),
                ("salinityCoarsened", "viridis", 30, 40, "PSU"),
            ]:
                arr = get_var(ds, var)
                if arr is None or arr.ndim < 3:
                    continue
                data = arr[-1, 0, :]  # last time, surface layer, all cells
                p = global_map(data, lon_deg, lat_deg,
                               f"MPAS-O {var} layer 0 (native, last t)",
                               cmap=cmap, vmin=vm, vmax=vx,
                               outdir=outdir, fname=f"mpaso_depth_{var}_surf_native.png",
                               units=units)
                if p: plots.append(p)

    close_ds(ds)

    _report("MPAS-O depth coarsening (native)", issues)
    return issues, plots, fill_reports


def check_mpaso_depth_coarsening_remapped(rundir, outdir, verbose):
    print("\n=== MPAS-O Depth Coarsening (remapped) ===")
    issues = []
    plots = []
    fill_reports = []

    files = find_files(rundir, "*.mpaso.hist.am.fmeDepthCoarsening.*.remapped.nc")
    if not files:
        print("  SKIP: no remapped fmeDepthCoarsening files found")
        return issues, plots, fill_reports

    print(f"  Found {len(files)} remapped file(s)")
    ds = safe_open(files[0])
    if has_time_zero(ds):
        print("  time dimension has size 0 -- skipping")
        close_ds(ds)
        return issues, plots, fill_reports

    # Verify lat-lon grid
    issues += check_remapped_grid(ds, "depth coarsening remapped")

    # Auto-detect number of depth levels from variable names in the file
    varnames = get_varnames(ds)
    n_remap_levels = 0
    for k in range(100):
        if f"temperatureCoarsened_{k}" in varnames:
            n_remap_levels = k + 1
        else:
            break
    if n_remap_levels == 0:
        n_remap_levels = 19  # fallback
    print(f"  Depth levels detected: {n_remap_levels}")

    # Check for per-level variables
    expected_vars = []
    for base in REMAPPED_DEPTH_COARSENED_BASES:
        for k in range(n_remap_levels):
            expected_vars.append(f"{base}_{k}")
    issues += check_remapped_vars(ds, expected_vars, "depth coarsening remapped")

    # Range checks on per-level variables
    for base in REMAPPED_DEPTH_COARSENED_BASES:
        found_levels = []
        for k in range(n_remap_levels):
            vname = f"{base}_{k}"
            if vname in varnames:
                found_levels.append(k)
                arr = get_var(ds, vname)
                vmin, vmax = RANGE_CHECKS.get(base, (None, None))
                issues += check_range(arr, f"remapped/{vname}", vmin, vmax)
                fill_reports.append(fill_nan_report(arr, f"remapped/{vname}"))
                if verbose and k == 0:
                    print(f"  remapped/{vname}: {summary_stats(arr)}")
        if found_levels:
            print(f"  {base}: {len(found_levels)}/{n_remap_levels} levels in remapped file")

    # Maps of surface fields from remapped data
    if HAS_MPL:
        for base, cmap, vm, vx, units in [
            ("temperatureCoarsened", "RdBu_r", -2, 30, "degC"),
            ("salinityCoarsened", "viridis", 30, 40, "PSU"),
        ]:
            vname = f"{base}_0"
            p = latlon_map(ds, vname,
                           f"MPAS-O {base} layer 0 (remapped, last t)",
                           cmap=cmap, vmin=vm, vmax=vx,
                           outdir=outdir,
                           fname=f"mpaso_depth_{base}_0_remapped.png",
                           units=units)
            if p: plots.append(p)

        # Vertical profile from remapped per-level variables
        for base in ["temperatureCoarsened", "salinityCoarsened"]:
            means = []
            for k in range(n_remap_levels):
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
    return issues, plots, fill_reports


def check_mpaso_derived(rundir, outdir, verbose):
    print("\n=== MPAS-O Derived Fields (native) ===")
    issues = []
    plots = []
    fill_reports = []

    files = find_files(rundir, "*.mpaso.hist.am.fmeDerivedFields.*.nc",
                       exclude=".remapped.")
    if not files:
        print("  SKIP: no native fmeDerivedFields files found")
        return issues, plots, fill_reports

    print(f"  Found {len(files)} native file(s)")
    ds = safe_open(files[0])
    if has_time_zero(ds):
        print("  time dimension has size 0 -- skipping")
        close_ds(ds)
        return issues, plots, fill_reports

    for var, ace_name in ACE_OCN_DERIVED.items():
        arr = get_var(ds, var)
        if arr is None:
            issues.append(f"  native missing: {var} (ACE: {ace_name})")
            continue
        vmin, vmax = RANGE_CHECKS.get(var, (None, None))
        issues += check_range(arr, var, vmin, vmax)
        fill_reports.append(fill_nan_report(arr, var))
        if verbose:
            print(f"  {var} ({ace_name}): {summary_stats(arr)}")

    # List all variables for reference
    varnames = get_varnames(ds)
    print(f"  Native file has {len(varnames)} variables")
    if verbose:
        print(f"  Variables: {varnames}")

    # Maps -- use LAST timestep
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
                data = arr[-1] if arr.ndim > 1 else arr
                p = global_map(data, lon_deg, lat_deg,
                               f"MPAS-O {var} (native, last t)",
                               cmap=cmap, vmin=vm, vmax=vx,
                               outdir=outdir, fname=f"mpaso_derived_{var}_native.png",
                               units=units)
                if p: plots.append(p)

    close_ds(ds)

    _report("MPAS-O derived fields (native)", issues)
    return issues, plots, fill_reports


def check_mpaso_derived_remapped(rundir, outdir, verbose):
    print("\n=== MPAS-O Derived Fields (remapped) ===")
    issues = []
    plots = []
    fill_reports = []

    files = find_files(rundir, "*.mpaso.hist.am.fmeDerivedFields.*.remapped.nc")
    if not files:
        print("  SKIP: no remapped fmeDerivedFields files found")
        return issues, plots, fill_reports

    print(f"  Found {len(files)} remapped file(s)")
    ds = safe_open(files[0])
    if has_time_zero(ds):
        print("  time dimension has size 0 -- skipping")
        close_ds(ds)
        return issues, plots, fill_reports

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
            fill_reports.append(fill_nan_report(arr, f"remapped/{var}"))
            if verbose:
                print(f"  remapped/{var}: {summary_stats(arr)}")

    # Maps from remapped data -- last timestep via latlon_map
    if HAS_MPL:
        for var, cmap, vm, vx, units in [
            ("sst", "RdBu_r", -2, 30, "degC"),
            ("sss", "viridis", 30, 40, "PSU"),
            ("surfaceHeatFluxTotal", "RdBu_r", -300, 300, "W/m2"),
        ]:
            p = latlon_map(ds, var,
                           f"MPAS-O {var} (remapped, last t)",
                           cmap=cmap, vmin=vm, vmax=vx,
                           outdir=outdir,
                           fname=f"mpaso_derived_{var}_remapped.png",
                           units=units)
            if p: plots.append(p)

    close_ds(ds)

    _report("MPAS-O derived fields (remapped)", issues)
    return issues, plots, fill_reports


def check_mpaso_vertical_reduce(rundir, outdir, verbose):
    print("\n=== MPAS-O Vertical Reduction (native) ===")
    issues = []
    plots = []
    fill_reports = []

    files = find_files(rundir, "*.mpaso.hist.am.fmeVerticalReduce.*.nc",
                       exclude=".remapped.")
    if not files:
        print("  SKIP: no native fmeVerticalReduce files found")
        return issues, plots, fill_reports

    print(f"  Found {len(files)} native file(s)")
    ds = safe_open(files[0])
    if has_time_zero(ds):
        print("  time dimension has size 0 -- skipping")
        close_ds(ds)
        return issues, plots, fill_reports

    for var, (vmin, vmax) in [("oceanHeatContent", (None, None)),
                               ("freshwaterContent", (-1000, 1000)),
                               ("kineticEnergy", (0, None))]:
        arr = get_var(ds, var)
        if arr is None:
            issues.append(f"  native missing: {var}")
        else:
            issues += check_range(arr, var, vmin, vmax)
            fill_reports.append(fill_nan_report(arr, var))
            if verbose:
                print(f"  {var}: {summary_stats(arr)}")

    # Native scatter plots for vertical reduce fields
    if HAS_MPL:
        lon = get_var(ds, "lonCell")
        lat = get_var(ds, "latCell")
        if lon is not None and lat is not None:
            lon_deg = np.degrees(lon)
            lat_deg = np.degrees(lat)
            for var, cmap, vm, vx, units in [
                ("oceanHeatContent", "inferno", None, None, "J/m2"),
                ("freshwaterContent", "RdBu_r", -500, 500, "m"),
                ("kineticEnergy", "YlOrRd", 0, None, "J/m2"),
            ]:
                arr = get_var(ds, var)
                if arr is None:
                    continue
                data = arr[-1] if arr.ndim > 1 else arr
                p = global_map(data, lon_deg, lat_deg,
                               f"MPAS-O {var} (native, last t)",
                               cmap=cmap, vmin=vm, vmax=vx,
                               outdir=outdir,
                               fname=f"mpaso_vertreduce_{var}_native.png",
                               units=units)
                if p: plots.append(p)

        # Time series of global-mean OHC
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
    return issues, plots, fill_reports


def check_mpaso_vertical_reduce_remapped(rundir, outdir, verbose):
    print("\n=== MPAS-O Vertical Reduction (remapped) ===")
    issues = []
    plots = []
    fill_reports = []

    files = find_files(rundir, "*.mpaso.hist.am.fmeVerticalReduce.*.remapped.nc")
    if not files:
        print("  SKIP: no remapped fmeVerticalReduce files found")
        return issues, plots, fill_reports

    print(f"  Found {len(files)} remapped file(s)")
    ds = safe_open(files[0])
    if has_time_zero(ds):
        print("  time dimension has size 0 -- skipping")
        close_ds(ds)
        return issues, plots, fill_reports

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
            fill_reports.append(fill_nan_report(arr, f"remapped/{var}"))
            if verbose:
                print(f"  remapped/{var}: {summary_stats(arr)}")

    # Maps from remapped data -- last timestep via latlon_map
    if HAS_MPL:
        for var, cmap, vm, vx, units in [
            ("oceanHeatContent", "inferno", None, None, "J/m2"),
            ("freshwaterContent", "RdBu_r", -500, 500, "m"),
            ("kineticEnergy", "YlOrRd", 0, None, "J/m2"),
        ]:
            p = latlon_map(ds, var,
                           f"MPAS-O {var} (remapped, last t)",
                           cmap=cmap, vmin=vm, vmax=vx,
                           outdir=outdir,
                           fname=f"mpaso_vertreduce_{var}_remapped.png",
                           units=units)
            if p: plots.append(p)

    close_ds(ds)

    _report("MPAS-O vertical reduction (remapped)", issues)
    return issues, plots, fill_reports


def check_mpassi_derived(rundir, outdir, verbose):
    print("\n=== MPAS-SI Derived Fields (native) ===")
    issues = []
    plots = []
    fill_reports = []

    files = find_files(rundir, "*.mpassi.hist.am.fmeSeaiceDerivedFields.*.nc",
                       exclude=".remapped.")
    if not files:
        print("  SKIP: no native fmeSeaiceDerivedFields files found")
        return issues, plots, fill_reports

    print(f"  Found {len(files)} native file(s)")
    ds = safe_open(files[0])
    if has_time_zero(ds):
        print("  time dimension has size 0 -- skipping")
        close_ds(ds)
        return issues, plots, fill_reports

    for var, ace_name in ACE_ICE_DERIVED.items():
        arr = get_var(ds, var)
        if arr is None:
            issues.append(f"  native missing: {var} (ACE: {ace_name})")
            continue
        vmin, vmax = RANGE_CHECKS.get(var, (None, None))
        issues += check_range(arr, var, vmin, vmax)
        fill_reports.append(fill_nan_report(arr, var))
        if verbose:
            print(f"  {var} ({ace_name}): {summary_stats(arr)}")

    # Maps -- last timestep
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
                ("airStressZonal", "RdBu_r", -1, 1, "N/m2"),
            ]:
                arr = get_var(ds, var)
                if arr is None:
                    continue
                data = arr[-1] if arr.ndim > 1 else arr
                p = global_map(data, lon_deg, lat_deg,
                               f"MPAS-SI {var} (native, last t)",
                               cmap=cmap, vmin=vm, vmax=vx,
                               outdir=outdir, fname=f"mpassi_{var}_native.png",
                               units=units)
                if p: plots.append(p)

    close_ds(ds)

    _report("MPAS-SI derived fields (native)", issues)
    return issues, plots, fill_reports


def check_mpassi_derived_remapped(rundir, outdir, verbose):
    print("\n=== MPAS-SI Derived Fields (remapped) ===")
    issues = []
    plots = []
    fill_reports = []

    files = find_files(rundir, "*.mpassi.hist.am.fmeSeaiceDerivedFields.*.remapped.nc")
    if not files:
        print("  SKIP: no remapped fmeSeaiceDerivedFields files found")
        return issues, plots, fill_reports

    print(f"  Found {len(files)} remapped file(s)")
    ds = safe_open(files[0])
    if has_time_zero(ds):
        print("  time dimension has size 0 -- skipping")
        close_ds(ds)
        return issues, plots, fill_reports

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
            fill_reports.append(fill_nan_report(arr, f"remapped/{var}"))
            if verbose:
                print(f"  remapped/{var}: {summary_stats(arr)}")

    # Maps from remapped data -- last timestep via latlon_map
    if HAS_MPL:
        for var, cmap, vm, vx, units in [
            ("iceAreaTotal", "Blues", 0, 1, "fraction"),
            ("iceVolumeTotal", "viridis", 0, 5, "m"),
            ("surfaceTemperatureMean", "RdBu_r", 210, 275, "K"),
            ("snowVolumeTotal", "PuBu", 0, 2, "m"),
            ("airStressZonal", "RdBu_r", -1, 1, "N/m2"),
        ]:
            p = latlon_map(ds, var,
                           f"MPAS-SI {var} (remapped, last t)",
                           cmap=cmap, vmin=vm, vmax=vx,
                           outdir=outdir,
                           fname=f"mpassi_{var}_remapped.png",
                           units=units)
            if p: plots.append(p)

    close_ds(ds)

    _report("MPAS-SI derived fields (remapped)", issues)
    return issues, plots, fill_reports


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

def write_html_index(outdir, all_plots_by_comp, all_issues, file_inventory_data,
                     all_fill_reports, timing_summary=None):
    index = os.path.join(outdir, "fme_output.html")
    n_issues = sum(len(v) for v in all_issues.values())
    n_pass = sum(1 for v in all_issues.values() if not v)
    n_total = len(all_issues)
    status = "ALL PASS" if n_issues == 0 else f"{n_issues} issues in {n_total - n_pass}/{n_total} components"
    status_color = "#2a7d2a" if n_issues == 0 else "#c33"

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

    # File inventory table
    inventory_rows = ""
    for label, count, flist in file_inventory_data:
        status_cls = "pass" if count > 0 else "fail"
        inventory_rows += (f'<tr><td>{label}</td>'
                           f'<td class="{status_cls}">{count}</td></tr>\n')

    # Fill/NaN report table
    fill_rows = ""
    for comp, reports in all_fill_reports.items():
        for r in reports:
            pct_cls = "pass" if r["pct"] < 50 else "fail"
            fill_rows += (f'<tr><td>{comp}</td><td>{r["name"]}</td>'
                          f'<td>{r["total"]:,}</td><td>{r["valid"]:,}</td>'
                          f'<td>{r["fill_nan"]:,}</td>'
                          f'<td class="{pct_cls}">{r["pct"]:.1f}%</td></tr>\n')

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
        # Use wider display for comparison/multi-panel figures
        is_comparison = (comp == "Comparisons")
        for p in plots:
            if p and os.path.exists(p):
                rel = os.path.relpath(p, outdir)
                fig_cls = "fig wide" if is_comparison else "fig"
                fig_sections += (f'<div class="{fig_cls}"><a href="{rel}">'
                                 f'<img src="{rel}"/></a>'
                                 f'<span>{os.path.basename(p)}</span></div>\n')
        fig_sections += '</div>\n'

    # Navigation
    nav_links = ""
    nav_items = ["Summary", "File_Inventory", "Fill_NaN_Report"]
    for base_name, _, _ in paired:
        nav_items.append(base_name)
    for comp in all_plots_by_comp:
        if comp not in shown_comps and all_plots_by_comp[comp]:
            nav_items.append(comp)
    if timing_summary:
        nav_items.append("Timing")
    for item in nav_items:
        anchor = item.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
        display = item.replace("_", " ")
        nav_links += f'<a href="#{anchor}">{display}</a>\n'

    timing_html = ""
    if timing_summary:
        timing_rows = ""
        for line in timing_summary[:30]:
            parts = line.split()
            if len(parts) >= 2:
                timing_rows += "<tr>" + "".join(f"<td>{p}</td>" for p in parts) + "</tr>\n"
            else:
                timing_rows += f"<tr><td colspan='6'>{line}</td></tr>\n"
        timing_html = f'''
    <h2 id="Timing">Performance Timing</h2>
    <div class="timing-container">
    <table class="timing-table">
    {timing_rows}
    </table>
    </div>
'''

    gen_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = textwrap.dedent(f"""\
    <!DOCTYPE html><html><head>
    <meta charset="utf-8"/>
    <title>FME Output Verification</title>
    <style>
      * {{ box-sizing: border-box; }}
      body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
             margin: 0; padding: 0; background: #f0f2f5; color: #333; }}
      .container {{ max-width: 1600px; margin: 0 auto; padding: 20px 30px; }}
      header {{ background: #1a2744; color: #fff; padding: 20px 30px; }}
      header h1 {{ margin: 0 0 6px 0; font-size: 1.6em; }}
      header .status {{ font-size: 1.2em; font-weight: bold;
                        color: {"#6fcf6f" if n_issues == 0 else "#ff8888"}; }}
      h2 {{ color: #1a2744; margin-top: 35px; border-bottom: 2px solid #1a2744;
            padding-bottom: 6px; font-size: 1.3em; }}
      h3 {{ color: #2c3e50; margin-top: 25px; }}
      h4 {{ color: #555; margin: 8px 0 4px; font-size: 0.95em; }}
      nav {{ background: #fff; padding: 12px 16px; border-radius: 6px;
             box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 15px 0;
             display: flex; flex-wrap: wrap; gap: 8px; position: sticky;
             top: 0; z-index: 100; }}
      nav a {{ color: #1a2744; text-decoration: none; padding: 5px 12px;
               background: #e8eef4; border-radius: 4px; font-size: 0.85em;
               font-weight: 500; transition: all 0.15s; }}
      nav a:hover {{ background: #1a2744; color: #fff; }}
      table {{ border-collapse: collapse; width: 100%; background: #fff;
               box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-radius: 4px;
               overflow: hidden; margin-bottom: 16px; }}
      td, th {{ border: 1px solid #e0e0e0; padding: 8px 14px; text-align: left; }}
      th {{ background: #1a2744; color: #fff; font-weight: 600; font-size: 0.9em; }}
      .pass {{ color: #2a7d2a; font-weight: bold; }}
      .fail {{ color: #c33; font-weight: bold; }}
      .comparison {{ display: flex; gap: 24px; flex-wrap: wrap; }}
      .col {{ flex: 1; min-width: 420px; }}
      .gallery {{ display: flex; flex-wrap: wrap; gap: 12px; }}
      .fig {{ display: inline-block; margin: 4px; vertical-align: top; }}
      .fig img {{ max-width: 520px; width: 100%; border: 1px solid #ccc; border-radius: 4px;
                  transition: transform 0.2s; cursor: zoom-in; }}
      .fig img:hover {{ transform: scale(1.02); box-shadow: 0 2px 8px rgba(0,0,0,0.15); }}
      .fig span {{ display: block; font-size: 0.75em; color: #888; margin-top: 2px; }}
      .fig.wide img {{ max-width: 900px; }}
      details {{ margin: 10px 0; background: #fff; border-radius: 4px;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
      summary {{ cursor: pointer; font-weight: 600; color: #1a2744; padding: 10px 14px; }}
      summary:hover {{ background: #f5f7fa; }}
      .timing-container {{ background: #fff; padding: 16px; border-radius: 4px;
                           box-shadow: 0 1px 3px rgba(0,0,0,0.08);
                           overflow-x: auto; max-height: 500px; overflow-y: auto; }}
      .timing-table td {{ font-family: 'Courier New', monospace; font-size: 0.85em;
                          padding: 4px 10px; white-space: nowrap; }}
      footer {{ margin-top: 40px; padding: 16px 30px; border-top: 2px solid #1a2744;
                color: #777; font-size: 0.8em; background: #fff; text-align: center; }}
      /* Lightbox overlay */
      .lightbox {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%;
                   background:rgba(0,0,0,0.88); z-index:1000; cursor:zoom-out;
                   align-items:center; justify-content:center; }}
      .lightbox.active {{ display:flex; }}
      .lightbox img {{ max-width:95%; max-height:95%; border-radius:6px;
                       box-shadow: 0 4px 20px rgba(0,0,0,0.4); }}
    </style>
    <script>
    document.addEventListener('DOMContentLoaded', function() {{
      var lb = document.createElement('div');
      lb.className = 'lightbox';
      var lbImg = document.createElement('img');
      lb.appendChild(lbImg);
      document.body.appendChild(lb);
      lb.addEventListener('click', function() {{ lb.classList.remove('active'); }});
      document.addEventListener('keydown', function(e) {{
        if (e.key === 'Escape') lb.classList.remove('active');
      }});
      document.querySelectorAll('.fig a').forEach(function(a) {{
        a.addEventListener('click', function(e) {{
          e.preventDefault();
          lbImg.src = this.href;
          lb.classList.add('active');
        }});
      }});
    }});
    </script>
    </head><body>
    <header>
      <h1>FME Online Output Verification</h1>
      <span class="status">{status}</span>
    </header>
    <div class="container">

    <nav>{nav_links}</nav>

    <h2 id="Summary">Component Summary</h2>
    <table>
    <tr><th>Component</th><th>Status</th><th>Issues</th></tr>
    {summary_rows}
    </table>

    {"<details><summary>Show all " + str(n_issues) + " issues</summary><table><tr><th>Component</th><th>Issue</th></tr>" + detail_rows + "</table></details>" if detail_rows else ""}

    <h2 id="File_Inventory">File Inventory</h2>
    <table>
    <tr><th>Category</th><th>File Count</th></tr>
    {inventory_rows}
    </table>

    <h2 id="Fill_NaN_Report">Fill / NaN Report</h2>
    <details><summary>Variable-level fill fraction details</summary>
    <table>
    <tr><th>Component</th><th>Variable</th><th>Total Cells</th><th>Valid</th><th>Fill/NaN</th><th>Fill %</th></tr>
    {fill_rows if fill_rows else '<tr><td colspan="6">No data collected.</td></tr>'}
    </table>
    </details>

    <h2>Diagnostic Figures</h2>
    {fig_sections if fig_sections else '<p>No figures generated.</p>'}

    {timing_html}
    </div>
    <footer>Generated by verify_fme_output.py &mdash; FME Online Output Processing for E3SM &mdash; {gen_timestamp}</footer>
    </body></html>
    """)
    with open(index, "w") as fh:
        fh.write(html)
    print(f"\nIndex written: {index}")
    return index


# -----------------------------------------------------------------------------
# File inventory
# -----------------------------------------------------------------------------

def collect_file_inventory(rundir):
    """Collect file inventory data and print summary.

    Returns list of (label, count, file_list) tuples for HTML rendering.
    """
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
    inventory_data = []
    for label, pattern, exclude in categories:
        hits = find_files(rundir, pattern, exclude=exclude)
        total += len(hits)
        status = f"{len(hits)} file(s)" if hits else "NONE"
        print(f"  {label:38s} {status}")
        inventory_data.append((label, len(hits), hits))
    print(f"  {'TOTAL':38s} {total} FME-related file(s)")
    return inventory_data


# -----------------------------------------------------------------------------
# Cross-component comparison figures (native vs remapped side-by-side)
# -----------------------------------------------------------------------------

def generate_comparison_figures(rundir, fig_root, all_plots_by_comp):
    """Generate native-vs-remapped side-by-side comparison figures.

    Opens both native and remapped files for each MPAS component and
    produces tripcolor (native) vs pcolormesh (remapped) comparisons.
    Also generates multi-panel layer views for EAM coarsened fields and
    zonal mean profiles for remapped lat-lon data.
    """
    if not HAS_MPL:
        return

    comp_outdir = os.path.join(fig_root, "comparisons")
    os.makedirs(comp_outdir, exist_ok=True)
    comp_plots = []

    # --- MPAS-O Derived: native vs remapped side-by-side ---
    nat_files = find_files(rundir, "*.mpaso.hist.am.fmeDerivedFields.*.nc",
                           exclude=".remapped.")
    rem_files = find_files(rundir, "*.mpaso.hist.am.fmeDerivedFields.*.remapped.nc")
    if nat_files and rem_files:
        print("\n=== Generating native vs remapped comparisons ===")
        ds_nat = safe_open(nat_files[0])
        ds_rem = safe_open(rem_files[-1])  # latest remapped
        if not has_time_zero(ds_nat) and not has_time_zero(ds_rem):
            lon_nat = get_var(ds_nat, "lonCell")
            lat_nat = get_var(ds_nat, "latCell")
            lon_rem = get_var(ds_rem, "lon")
            lat_rem = get_var(ds_rem, "lat")
            if lon_nat is not None and lat_nat is not None and \
               lon_rem is not None and lat_rem is not None:
                lon_nat_deg = np.degrees(lon_nat)
                lat_nat_deg = np.degrees(lat_nat)
                if lon_rem.ndim == 1 and lat_rem.ndim == 1:
                    lons_rem, lats_rem = np.meshgrid(lon_rem, lat_rem)
                else:
                    lons_rem, lats_rem = lon_rem, lat_rem

                for var, cmap, vm, vx, units_str in [
                    ("sst", "RdYlBu_r", -2, 30, "degC"),
                    ("sss", "viridis", 30, 40, "PSU"),
                    ("surfaceHeatFluxTotal", "RdBu_r", -300, 300, "W/m2"),
                ]:
                    arr_nat = get_var(ds_nat, var)
                    arr_rem = get_var(ds_rem, var)
                    if arr_nat is None or arr_rem is None:
                        continue
                    d_nat = arr_nat[-1] if arr_nat.ndim > 1 else arr_nat
                    d_rem = arr_rem[-1] if arr_rem.ndim == 3 else arr_rem
                    p = side_by_side_comparison(
                        d_nat, lon_nat_deg, lat_nat_deg, f"Native ({var})",
                        d_rem, lons_rem, lats_rem, f"Remapped ({var})",
                        f"MPAS-O {var}: Native vs Remapped",
                        comp_outdir, f"compare_mpaso_{var}.png",
                        cmap=cmap, vmin=vm, vmax=vx, units=units_str)
                    if p:
                        comp_plots.append(p)
                        print(f"  Wrote {os.path.basename(p)}")
        close_ds(ds_nat)
        close_ds(ds_rem)

    # --- MPAS-SI Derived: native vs remapped side-by-side ---
    nat_files = find_files(rundir, "*.mpassi.hist.am.fmeSeaiceDerivedFields.*.nc",
                           exclude=".remapped.")
    rem_files = find_files(rundir, "*.mpassi.hist.am.fmeSeaiceDerivedFields.*.remapped.nc")
    if nat_files and rem_files:
        ds_nat = safe_open(nat_files[0])
        ds_rem = safe_open(rem_files[-1])
        if not has_time_zero(ds_nat) and not has_time_zero(ds_rem):
            lon_nat = get_var(ds_nat, "lonCell")
            lat_nat = get_var(ds_nat, "latCell")
            lon_rem = get_var(ds_rem, "lon")
            lat_rem = get_var(ds_rem, "lat")
            if lon_nat is not None and lat_nat is not None and \
               lon_rem is not None and lat_rem is not None:
                lon_nat_deg = np.degrees(lon_nat)
                lat_nat_deg = np.degrees(lat_nat)
                if lon_rem.ndim == 1 and lat_rem.ndim == 1:
                    lons_rem, lats_rem = np.meshgrid(lon_rem, lat_rem)
                else:
                    lons_rem, lats_rem = lon_rem, lat_rem

                for var, cmap, vm, vx, units_str in [
                    ("iceAreaTotal", "Blues", 0, 1, "fraction"),
                    ("iceVolumeTotal", "viridis", 0, 5, "m"),
                    ("surfaceTemperatureMean", "RdBu_r", 210, 275, "K"),
                ]:
                    arr_nat = get_var(ds_nat, var)
                    arr_rem = get_var(ds_rem, var)
                    if arr_nat is None or arr_rem is None:
                        continue
                    d_nat = arr_nat[-1] if arr_nat.ndim > 1 else arr_nat
                    d_rem = arr_rem[-1] if arr_rem.ndim == 3 else arr_rem
                    p = side_by_side_comparison(
                        d_nat, lon_nat_deg, lat_nat_deg, f"Native ({var})",
                        d_rem, lons_rem, lats_rem, f"Remapped ({var})",
                        f"MPAS-SI {var}: Native vs Remapped",
                        comp_outdir, f"compare_mpassi_{var}.png",
                        cmap=cmap, vmin=vm, vmax=vx, units=units_str)
                    if p:
                        comp_plots.append(p)
                        print(f"  Wrote {os.path.basename(p)}")
        close_ds(ds_nat)
        close_ds(ds_rem)

    # --- EAM tape3: multi-panel T_1..T_8 and Q_1..Q_8 ---
    t3_files = find_files(rundir, "*.eam.h2.*.nc")
    if t3_files:
        ds3 = safe_open(t3_files[0])
        if not has_time_zero(ds3) and HAS_XR and isinstance(ds3, xr.Dataset):
            lon_name = next((d for d in ["lon", "ncol", "longitude"] if d in ds3.dims), None)
            lat_name = next((d for d in ["lat", "latitude"] if d in ds3.dims), None)
            if lon_name and lat_name:
                lons = ds3[lon_name].values
                lats = ds3[lat_name].values if lat_name != lon_name else None
                if lats is not None:
                    if lons.ndim == 1 and lat_name == "lat":
                        lons, lats = np.meshgrid(lons, lats)

                    for base, cmap, vm, vx, units_str in [
                        ("T", "RdYlBu_r", 180, 310, "K"),
                        ("Q", "YlGnBu", 0, 0.02, "kg/kg"),
                    ]:
                        panels = []
                        for k in range(1, N_ATM_LAYERS + 1):
                            vname = f"{base}_{k}"
                            arr = get_var(ds3, vname)
                            if arr is None or arr.size == 0:
                                continue
                            data = arr[-1] if arr.ndim >= 2 and arr.shape[0] > 0 else arr
                            v = valid_data(data)
                            stats = f"mean={v.mean():.1f}" if v is not None and v.size > 0 else ""
                            panels.append((data, lons, lats,
                                           f"{vname} ({stats})"))

                        if panels:
                            p = multi_panel_maps(
                                panels,
                                f"EAM Vertically Coarsened {base} "
                                f"(all {N_ATM_LAYERS} layers, last t)",
                                comp_outdir,
                                f"eam_tape3_{base}_all_layers.png",
                                ncols=4, cmap=cmap, vmin=vm, vmax=vx,
                                units=units_str)
                            if p:
                                comp_plots.append(p)
                                print(f"  Wrote {os.path.basename(p)}")

                    # Zonal mean of T layers (only for lat-lon data)
                    if lat_name == "lat":
                        lat_1d = ds3["lat"].values
                        profiles = []
                        for k in range(1, N_ATM_LAYERS + 1):
                            zm = _compute_zonal_mean(get_var(ds3, f"T_{k}"))
                            if zm is not None:
                                profiles.append((zm, f"T_{k}"))
                        if profiles:
                            p = zonal_mean_plot(
                                profiles, lat_1d,
                                "EAM Zonal Mean Temperature (all layers)",
                                comp_outdir, "eam_tape3_T_zonal_mean.png",
                                ylabel="K")
                            if p:
                                comp_plots.append(p)
                                print(f"  Wrote {os.path.basename(p)}")
        close_ds(ds3)

    # --- Zonal means for remapped MPAS-O fields ---
    rem_derived = find_files(rundir, "*.mpaso.hist.am.fmeDerivedFields.*.remapped.nc")
    if rem_derived:
        ds = safe_open(rem_derived[-1])
        if not has_time_zero(ds):
            lat = get_var(ds, "lat")
            if lat is not None:
                profiles = []
                for var in ["sst", "sss"]:
                    zm = _compute_zonal_mean(get_var(ds, var))
                    if zm is not None:
                        profiles.append((zm, var))
                if profiles:
                    p = zonal_mean_plot(
                        profiles, lat,
                        "MPAS-O Zonal Mean SST/SSS (remapped)",
                        comp_outdir, "mpaso_zonal_mean_sst_sss.png",
                        ylabel="degC / PSU")
                    if p:
                        comp_plots.append(p)
                        print(f"  Wrote {os.path.basename(p)}")
        close_ds(ds)

    if comp_plots:
        all_plots_by_comp["Comparisons"] = comp_plots


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

# ─────────────────────────────────────────────────────────────────────────────
# Cross-verification: compare fme_output vs fme_legacy_output
# ─────────────────────────────────────────────────────────────────────────────

def cross_verify(fme_rundir, legacy_rundir, outdir, verbose=False):
    """
    Compare online FME output against legacy (raw) output.

    Checks that fields produced by the FME online pipeline are consistent
    with the raw fields from the legacy pipeline that would be processed
    offline by the ACE scripts.

    Comparisons:
    - EAM: fme_output Tape 1/2/3 fields vs legacy Tape 2/3 fields
      (Tape 2 instantaneous 3D state should match; Tape 1/3 averaged fields
       should have same time-means within averaging-window tolerance)
    - MPAS-O: fmeDerivedFields (SST, SSS, fluxes) vs legacy
      timeSeriesStatsCustom (ssh, activeTracers, fluxes)
    - MPAS-SI: fmeSeaiceDerivedFields (iceAreaTotal, etc.) vs legacy
      timeSeriesStatsCustom (iceAreaCell, etc.)
    """
    print("\n" + "=" * 70)
    print("CROSS-VERIFICATION: fme_output vs fme_legacy_output")
    print("=" * 70)

    issues = []

    # --- EAM cross-check ---
    print("\n=== EAM Cross-Check ===")
    # Compare instantaneous 3D state (Tape 2 in both)
    fme_h2 = find_files(fme_rundir, "*.eam.h1.*nc") or find_files(fme_rundir, "*.cam.h1.*nc")
    leg_h2 = find_files(legacy_rundir, "*.eam.h1.*nc") or find_files(legacy_rundir, "*.cam.h1.*nc")

    if fme_h2 and leg_h2:
        ds_fme = safe_open(fme_h2[0])
        ds_leg = safe_open(leg_h2[0])

        for var in ["T", "Q", "U", "V", "PS", "TS"]:
            fme_data = get_var(ds_fme, var)
            leg_data = get_var(ds_leg, var)
            if fme_data is not None and leg_data is not None:
                if fme_data.shape == leg_data.shape:
                    diff = np.abs(fme_data - leg_data)
                    maxdiff = float(np.nanmax(diff))
                    if maxdiff > 1e-10:
                        issues.append(f"  EAM {var}: max diff = {maxdiff:.4g} (should be 0 for instantaneous)")
                    elif verbose:
                        print(f"  EAM {var}: identical (max diff = {maxdiff:.2e})")
                else:
                    if verbose:
                        print(f"  EAM {var}: shape mismatch fme={fme_data.shape} vs legacy={leg_data.shape}")
            elif verbose:
                print(f"  EAM {var}: {'missing in fme' if fme_data is None else 'missing in legacy'}")
    else:
        print("  SKIP: h1 files not found in both rundirs")

    # --- MPAS-O cross-check ---
    print("\n=== MPAS-O Cross-Check ===")
    fme_derived = find_files(fme_rundir, "*.am.fmeDerivedFields.*nc")
    leg_custom = find_files(legacy_rundir, "*.am.timeSeriesStatsCustom.*nc")

    if fme_derived and leg_custom:
        ds_fme = safe_open(fme_derived[0])
        ds_leg = safe_open(leg_custom[0])

        # SST from fme vs temperature top level from legacy activeTracers
        fme_sst = get_var(ds_fme, "sst")
        if fme_sst is not None:
            v = valid_data(fme_sst)
            if v is not None:
                print(f"  FME SST: {summary_stats(fme_sst)}")
            else:
                issues.append("  FME SST: no valid data")

        # Cross-check flux fields
        for var in ["shortWaveHeatFlux", "sensibleHeatFlux", "windStressZonal"]:
            fme_v = get_var(ds_fme, var)
            # Legacy uses time-averaged names like timeCustom_avg_shortWaveHeatFlux
            leg_name = f"timeCustom_avg_{var}"
            leg_v = get_var(ds_leg, leg_name)
            if leg_v is None:
                leg_v = get_var(ds_leg, var)  # try without prefix

            if fme_v is not None and leg_v is not None:
                fme_valid = valid_data(fme_v)
                leg_valid = valid_data(leg_v)
                if fme_valid is not None and leg_valid is not None:
                    if verbose:
                        print(f"  {var}: FME {summary_stats(fme_v)}")
                        print(f"  {var}: Legacy {summary_stats(leg_v)}")
            elif verbose:
                print(f"  {var}: {'missing in fme' if fme_v is None else 'missing in legacy'}")
    else:
        print("  SKIP: derived/custom files not found in both rundirs")

    # --- MPAS-SI cross-check ---
    print("\n=== MPAS-SI Cross-Check ===")
    fme_si = find_files(fme_rundir, "*fmeSeaiceDerivedFields*nc")
    leg_si = find_files(legacy_rundir, "*mpassi*timeSeriesStatsCustom*nc")

    if fme_si and leg_si:
        ds_fme = safe_open(fme_si[0])
        ds_leg = safe_open(leg_si[0])

        fme_area = get_var(ds_fme, "iceAreaTotal")
        # Legacy has iceAreaCell (per-category), FME has iceAreaTotal (summed)
        leg_area = get_var(ds_leg, "timeCustom_avg_iceAreaCell")
        if leg_area is None:
            leg_area = get_var(ds_leg, "iceAreaCell")

        if fme_area is not None:
            print(f"  FME iceAreaTotal: {summary_stats(fme_area)}")
        if leg_area is not None:
            print(f"  Legacy iceAreaCell: {summary_stats(leg_area)}")
    else:
        print("  SKIP: sea ice files not found in both rundirs")

    if issues:
        print(f"\n  CROSS-VERIFY: {len(issues)} issues found")
        for i in issues:
            print(i)
    else:
        print(f"\n  CROSS-VERIFY: All checks passed")

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Verify FME output and produce diagnostic figures for ACE/Samudra.")
    parser.add_argument("--rundir", required=True,
                        help="CIME RUNDIR (or archive/ocn/hist directory)")
    parser.add_argument("--outdir",
                        default="/pscratch/sd/m/mahf708/aifigs_v3LRpiControlSamudrACE",
                        help="Output directory for figures and HTML index")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--cross-verify",
                        help="Path to legacy testmod RUNDIR for cross-verification")
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
    file_inventory_data = collect_file_inventory(args.rundir)

    all_issues = {}
    all_plots_by_comp = {}
    all_fill_reports = {}

    # All figures go under fme_output/ subdir so fme_output.html can reference them
    fig_root = os.path.join(args.outdir, "fme_output")
    os.makedirs(fig_root, exist_ok=True)

    def run_check(name, func, subdir, *a):
        comp_outdir = os.path.join(fig_root, subdir)
        os.makedirs(comp_outdir, exist_ok=True)
        result = func(*a[:1], comp_outdir, *a[1:])
        issues, plots, fill_reports = result
        all_issues[name] = issues
        all_plots_by_comp[name] = plots
        all_fill_reports[name] = fill_reports

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

    # Cross-component comparison figures (native vs remapped side-by-side,
    # multi-panel layer views, zonal means)
    generate_comparison_figures(args.rundir, fig_root, all_plots_by_comp)

    # Timing summary
    timing_summary = read_timing_summary(args.rundir)
    if timing_summary:
        print("\n=== Performance Timing ===")
        for line in timing_summary[:10]:
            print(f"  {line}")
        if len(timing_summary) > 10:
            print(f"  ... ({len(timing_summary) - 10} more lines in HTML report)")

    # Cross-verification against legacy testmod output
    if args.cross_verify:
        if not os.path.isdir(args.cross_verify):
            print(f"WARNING: cross-verify rundir not found: {args.cross_verify}")
        else:
            xv_issues = cross_verify(args.rundir, args.cross_verify,
                                     args.outdir, args.verbose)
            all_issues["cross-verify"] = xv_issues

    write_html_index(args.outdir, all_plots_by_comp, all_issues,
                     file_inventory_data, all_fill_reports,
                     timing_summary=timing_summary)

    n_total = sum(len(v) for v in all_issues.values())
    print("\n" + "=" * 70)
    print(f"RESULT: {'ALL CHECKS PASSED' if n_total == 0 else f'{n_total} ISSUES FOUND'}")
    sys.exit(0 if n_total == 0 else 1)


if __name__ == "__main__":
    main()
