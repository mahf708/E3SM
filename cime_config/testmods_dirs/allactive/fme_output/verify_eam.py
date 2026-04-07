#!/usr/bin/env python3
"""
Verify EAM FME online output (vcoarsen, derived fields) and cross-verify
against legacy output. Produces HTML dashboard with BFB identity tests,
offline vcoarsen comparison, and diagnostic figures.

Usage:
    python verify_eam.py --rundir /path/to/FME/RUNDIR --outdir /path/to/figs
    python verify_eam.py --rundir /path/to/FME/RUNDIR --legacy-rundir /path/to/LEGACY/RUNDIR --outdir /path/to/figs
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
    # 3D coarsened (all on tape 1): T_0..T_7, U_0..U_7, V_0..V_7,
    #   STW_0..7 -> specific_total_water_0..7 (derived: Q+CLDICE+CLDLIQ+RAINQM)
}
ACE_ATM_3D_COARSENED = ["T", "U", "V", "STW"]
N_ATM_LAYERS = 8  # ACE uses 8 pressure layers (0-based: T_0..T_7)

# Physical range checks: (vmin, vmax)
# Ranges are deliberately generous to avoid false positives during spinup.
RANGE_CHECKS = {
    "T": (150, 350),
    "Q": (0, 0.05),
    "PS": (40000, 110000),
    "TS": (200, 340),
    "PHIS": (-1000, 60000),
}

# -- Vcoarsen constants -------------------------------------------------------
GRAVIT = 9.80616       # m/s2, matches EAM physconst
P0     = 100000.0      # Pa, reference pressure for hybrid coordinates
VCOARSEN_PBOUNDS = np.array([
    10.0, 4803.81, 13913.06, 26856.34,
    43998.31, 59659.67, 76854.15, 90711.83, 101325.0
])  # 9 interfaces -> 8 layers, matching ACE L80 indices [0,25,38,46,52,56,61,69,80]
N_VCOARSEN_LAYERS = len(VCOARSEN_PBOUNDS) - 1  # 8

# Fields that appear in BOTH fme_output and fme_legacy_output for BFB identity checks
# Instantaneous (no :A suffix)
BFB_INST_FIELDS = ["PS", "TS", "PHIS", "LANDFRAC", "OCNFRAC", "ICEFRAC"]
# Averaged (:A suffix in both cases)
BFB_AVG_FIELDS = ["SOLIN", "FSNTOA", "FLUT", "FLDS", "FSDS",
                   "LHFLX", "SHFLX", "TAUX", "TAUY", "PRECT"]

# Raw 3D fields in legacy output that can be vcoarsened offline for comparison
LEGACY_RAW_3D = ["T", "Q", "U", "V", "CLDLIQ", "CLDICE", "RAINQM"]

# FME vcoarsen fields -> legacy raw source (for offline recomputation)
# FME name -> (legacy fields to sum for derived, then vcoarsen)
FME_TO_LEGACY = {
    "T":   ["T"],
    "U":   ["U"],
    "V":   ["V"],
    "STW": ["Q", "CLDICE", "CLDLIQ", "RAINQM"],
}


# -- Offline vcoarsen / column integration ------------------------------------

def compute_pint(ps, hyai, hybi):
    """Compute interface pressures from hybrid coordinates.

    Parameters
    ----------
    ps : ndarray, shape (ncol,)
        Surface pressure in Pa.
    hyai, hybi : ndarray, shape (nlev+1,)
        Hybrid A and B coefficients (interface levels, top to bottom).

    Returns
    -------
    pint : ndarray, shape (ncol, nlev+1)
        Interface pressures in Pa.
    """
    # pint(i, k) = hyai(k) * P0 + hybi(k) * ps(i)
    return hyai[np.newaxis, :] * P0 + hybi[np.newaxis, :] * ps[:, np.newaxis]


def compute_pdel(pint):
    """Compute layer thickness in Pa from interface pressures.

    Parameters
    ----------
    pint : ndarray, shape (ncol, nlev+1)

    Returns
    -------
    pdel : ndarray, shape (ncol, nlev)
    """
    return pint[:, 1:] - pint[:, :-1]


def offline_vcoarsen_avg(field, pint, pbounds=None):
    """Overlap-weighted vertical averaging onto coarsened pressure layers.

    Implements the same algorithm as shr_vcoarsen_avg in shr_vcoarsen_mod.F90.
    For each target layer [pb_top, pb_bot], computes the pressure-weighted mean
    of all model levels that overlap with it.

    Parameters
    ----------
    field : ndarray, shape (ncol, nlev)
        Full-resolution field values.
    pint : ndarray, shape (ncol, nlev+1)
        Interface pressures from compute_pint.
    pbounds : ndarray, shape (nlayers+1,), optional
        Target layer pressure boundaries. Defaults to VCOARSEN_PBOUNDS.

    Returns
    -------
    coarsened : ndarray, shape (ncol, nlayers)
        Coarsened field values.
    """
    if pbounds is None:
        pbounds = VCOARSEN_PBOUNDS
    nlayers = len(pbounds) - 1
    ncol, nlev = field.shape

    coarsened = np.zeros((ncol, nlayers))
    for layer in range(nlayers):
        pb_top = pbounds[layer]
        pb_bot = pbounds[layer + 1]
        layer_dp = pb_bot - pb_top

        weight_sum = np.zeros(ncol)
        for k in range(nlev):
            # Overlap between model level k and target layer
            overlap_top = np.maximum(pb_top, pint[:, k])
            overlap_bot = np.minimum(pb_bot, pint[:, k + 1])
            overlap = np.maximum(0.0, overlap_bot - overlap_top)

            coarsened[:, layer] += field[:, k] * overlap
            weight_sum += overlap

        # Normalize by total overlap (should equal layer_dp for well-formed grids)
        valid = weight_sum > 0
        coarsened[valid, layer] /= weight_sum[valid]
        coarsened[~valid, layer] = np.nan

    return coarsened


def offline_column_integrate(field, pdel):
    """Column integration: sum(field * pdel / g) over all levels.

    Parameters
    ----------
    field : ndarray, shape (ncol, nlev)
    pdel : ndarray, shape (ncol, nlev)

    Returns
    -------
    integrated : ndarray, shape (ncol,)
    """
    return np.sum(field * pdel / GRAVIT, axis=1)


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
    """Plot data on an axis -- tripcolor for 1D, pcolormesh for 2D."""
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
    """Two global maps side by side -- native tripcolor on left, remapped pcolormesh on right."""
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
# EAM verification + visualization
# -----------------------------------------------------------------------------

def check_eam(rundir, outdir, verbose):
    print("\n=== EAM ===")
    issues = []
    plots = []
    fill_reports = []

    # -- tape 1 (h0): all EAM fields (single-tape layout) -------------------
    t1_files = find_files(rundir, "*.eam.h0.*.nc")
    if not t1_files:
        print("  SKIP: no *.eam.h0.*.nc files found")
    else:
        print(f"  h0: found {len(t1_files)} file(s)")
        ds = safe_open(t1_files[0])
        if has_time_zero(ds):
            print("  h0: time dimension has size 0 -- skipping")
            close_ds(ds)
        else:
            for var in ["PHIS", "PS", "TS", "LHFLX", "SHFLX", "FLDS", "FSDS",
                        "FSNTOA", "FLUT", "SOLIN", "FSUS", "FLUS",
                        "ICEFRAC", "LANDFRAC", "OCNFRAC", "TAUX", "TAUY",
                        "PRECT", "DTENDTTW"]:
                arr = get_var(ds, var)
                if arr is None:
                    issues.append(f"  h0 missing: {var}")
                else:
                    vmin, vmax = RANGE_CHECKS.get(var, (None, None))
                    issues += check_range(arr, f"h0/{var}", vmin, vmax)
                    fill_reports.append(fill_nan_report(arr, f"h0/{var}"))
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
                        p = global_map(data, lons, lats, f"EAM {var} (h0, last t)",
                                       cmap=cmap, vmin=vm, vmax=vx,
                                       outdir=outdir, fname=f"eam_h0_{var}.png",
                                       units=units)
                        if p: plots.append(p)
            close_ds(ds)

    # -- vcoarsen layers (on h0 in single-tape config) -----------------------
    if t1_files:
        ds3 = safe_open(t1_files[0])
        if not has_time_zero(ds3):
            for base in ACE_ATM_3D_COARSENED:
                found_layers = []
                for k in range(N_ATM_LAYERS):  # 0-based: T_0..T_7
                    vname = f"{base}_{k}"
                    arr = get_var(ds3, vname)
                    if arr is not None:
                        found_layers.append(k)
                        vmin, vmax = RANGE_CHECKS.get(base, (None, None))
                        issues += check_range(arr, f"h0/{vname}", vmin, vmax)
                        fill_reports.append(fill_nan_report(arr, f"h0/{vname}"))
                if len(found_layers) == N_ATM_LAYERS:
                    print(f"  h0/{base}: all {N_ATM_LAYERS} layers present (0-based)")
                elif found_layers:
                    print(f"  h0/{base}: {len(found_layers)}/{N_ATM_LAYERS} layers found "
                          f"(indices {found_layers})")
                    issues.append(f"  h0/{base}: only {len(found_layers)}/{N_ATM_LAYERS} layers")
                else:
                    alts = [v for v in get_varnames(ds3)
                            if v.startswith(base + "_") or v.startswith(base.upper() + "_")]
                    if alts:
                        print(f"  h0/{base}: found alternate names: {alts[:5]}")
                    else:
                        issues.append(f"  h0 missing coarsened field: {base}_0..{base}_{N_ATM_LAYERS-1}")

            # Plot T_0 (near-top) and T_7 (near-surface) at last timestep
            if HAS_MPL and HAS_XR and isinstance(ds3, xr.Dataset):
                lon_name = next((d for d in ["lon", "ncol", "longitude"] if d in ds3.dims), None)
                lat_name = next((d for d in ["lat", "latitude"] if d in ds3.dims), None)
                for vname, label_tag in [("T_0", "near-top"), ("T_7", "near-surface")]:
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
                                       f"EAM {vname} ({label_tag}, h0, last t)",
                                       cmap="RdBu_r", vmin=150, vmax=320,
                                       outdir=outdir, fname=f"eam_h0_{vname}.png",
                                       units="K")
                        if p: plots.append(p)
        close_ds(ds3)

    _report("EAM", issues)
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
# EAM comparison figures (multi-panel and zonal mean)
# -----------------------------------------------------------------------------

def generate_comparison_figures(rundir, fig_root, all_plots_by_comp):
    """Generate multi-panel layer views and zonal mean profiles for EAM coarsened fields."""
    if not HAS_MPL:
        return

    comp_outdir = os.path.join(fig_root, "comparisons")
    os.makedirs(comp_outdir, exist_ok=True)
    comp_plots = []

    # --- EAM h0: multi-panel T_0..T_7 and STW_0..STW_7 ---
    h0_files = find_files(rundir, "*.eam.h0.*.nc")
    if h0_files:
        ds3 = safe_open(h0_files[0])
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
                        ("STW", "YlGnBu", 0, 0.02, "kg/kg"),
                    ]:
                        panels = []
                        for k in range(N_ATM_LAYERS):  # 0-based
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
                                f"eam_h0_{base}_all_layers.png",
                                ncols=4, cmap=cmap, vmin=vm, vmax=vx,
                                units=units_str)
                            if p:
                                comp_plots.append(p)
                                print(f"  Wrote {os.path.basename(p)}")

                    # Zonal mean of T layers (only for lat-lon data)
                    if lat_name == "lat":
                        lat_1d = ds3["lat"].values
                        profiles = []
                        for k in range(N_ATM_LAYERS):  # 0-based
                            zm = _compute_zonal_mean(get_var(ds3, f"T_{k}"))
                            if zm is not None:
                                profiles.append((zm, f"T_{k}"))
                        if profiles:
                            p = zonal_mean_plot(
                                profiles, lat_1d,
                                "EAM Zonal Mean Temperature (all layers)",
                                comp_outdir, "eam_h0_T_zonal_mean.png",
                                ylabel="K")
                            if p:
                                comp_plots.append(p)
                                print(f"  Wrote {os.path.basename(p)}")
        close_ds(ds3)

    if comp_plots:
        all_plots_by_comp["Comparisons"] = comp_plots


# ─────────────────────────────────────────────────────────────────────────────
# Cross-verification: compare fme_output vs fme_legacy_output
# ─────────────────────────────────────────────────────────────────────────────

def cross_verify(fme_rundir, legacy_rundir, outdir, verbose=False):
    """
    Compare online FME output against legacy (raw) output.

    Both cases run identical model physics from the same ICs -- the only
    difference is what diagnostics get written.  Therefore:

    1. BFB IDENTITY: any field present in both h0 files must match
       bit-for-bit (same model state, same history averaging).

    2. ONLINE vs OFFLINE VCOARSEN: the legacy case outputs raw 3D fields
       (T, Q, U, V, CLDLIQ, CLDICE, RAINQM at full L80 resolution).
       We vcoarsen them offline with the same algorithm / pressure bounds
       and compare against the FME case's online T_0..T_7, STW_0..STW_7,
       U_0..U_7, V_0..V_7.  These should be BFB identical.

    3. ONLINE vs OFFLINE DERIVED: legacy raw Q + CLDICE + CLDLIQ + RAINQM
       should equal FME STW at every level.  After vcoarsening, this tests
       derived + vcoarsen together.

    Returns a dict of {test_name: list_of_result_dicts} for HTML rendering.
    """
    print("\n" + "=" * 70)
    print("CROSS-VERIFICATION: fme_output vs fme_legacy_output")
    print("=" * 70)

    issues = []
    results = []   # list of dicts: {test, field, status, detail}
    xv_plots = []  # comparison figures

    xv_outdir = os.path.join(outdir, "fme_output", "cross_verify")
    os.makedirs(xv_outdir, exist_ok=True)

    # -- locate h0 files in both rundirs --------------------------------------
    fme_h0 = find_files(fme_rundir, "*.eam.h0.*.nc")
    leg_h0 = find_files(legacy_rundir, "*.eam.h0.*.nc")

    if not fme_h0 or not leg_h0:
        print("  SKIP EAM: h0 files not found in both rundirs")
        print(f"    fme: {len(fme_h0)} files,  legacy: {len(leg_h0)} files")
        return issues

    ds_fme = safe_open(fme_h0[0])
    ds_leg = safe_open(leg_h0[0])

    if has_time_zero(ds_fme) or has_time_zero(ds_leg):
        print("  SKIP: time dimension has size 0")
        close_ds(ds_fme); close_ds(ds_leg)
        return issues

    # -- 1. FIELD IDENTITY ------------------------------------------------------
    # Detect grid mismatch (FME may be remapped to lat-lon, legacy on native)
    fme_dims = get_dims(ds_fme)
    leg_dims = get_dims(ds_leg)
    grids_match = ("ncol" in fme_dims) == ("ncol" in leg_dims)

    if grids_match:
        print("\n--- Test 1: BFB Identity (shared fields, same grid) ---")
    else:
        fme_grid = "lat-lon" if "lat" in fme_dims else "native"
        leg_grid = "lat-lon" if "lat" in leg_dims else "native"
        print(f"\n--- Test 1: Global-Mean Comparison (FME={fme_grid}, Legacy={leg_grid}) ---")

    all_bfb_fields = BFB_INST_FIELDS + BFB_AVG_FIELDS
    n_bfb_pass = 0
    n_bfb_fail = 0

    for var in all_bfb_fields:
        fme_data = get_var(ds_fme, var)
        leg_data = get_var(ds_leg, var)
        if fme_data is None or leg_data is None:
            label = "missing-fme" if fme_data is None else "missing-legacy"
            results.append({"test": "BFB", "field": var, "status": "SKIP",
                            "detail": f"{label}"})
            print(f"  {var}: SKIP ({label})")
            continue

        if grids_match and fme_data.shape == leg_data.shape:
            # Same grid: point-by-point BFB
            diff = np.abs(fme_data.astype(np.float64) - leg_data.astype(np.float64))
            maxdiff = float(np.nanmax(diff))
            status = "BFB" if maxdiff < 1e-12 else "DIFF"
            if status == "BFB":
                n_bfb_pass += 1
            else:
                n_bfb_fail += 1
                issues.append(f"  BFB {var}: max|diff| = {maxdiff:.4g}")
            results.append({"test": "BFB", "field": var, "status": status,
                            "detail": f"max|diff| = {maxdiff:.2e}"})
            tag = "PASS" if status == "BFB" else "FAIL"
            print(f"  {var}: {tag}  max|diff| = {maxdiff:.2e}")
        else:
            # Different grids: compare area-weighted global means (last timestep)
            fme_last = fme_data[-1] if fme_data.ndim >= 2 else fme_data
            leg_last = leg_data[-1] if leg_data.ndim >= 2 else leg_data

            # Area-weight lat-lon data with cos(lat); native grid is quasi-equal-area
            fme_flat = fme_last.ravel().astype(np.float64) if hasattr(fme_last, 'ravel') else fme_last
            leg_flat = leg_last.ravel().astype(np.float64) if hasattr(leg_last, 'ravel') else leg_last

            fme_lat = get_var(ds_fme, "lat")
            if fme_lat is not None and "lat" in fme_dims and fme_last.ndim == 2:
                # cos-lat weighting for regular lat-lon grid
                coslat = np.cos(np.deg2rad(fme_lat))
                weights = coslat[:, np.newaxis] * np.ones(fme_last.shape[1])[np.newaxis, :]
                mask = (np.abs(fme_last) < 1e10) & np.isfinite(fme_last)
                fme_mean = float(np.average(fme_last[mask], weights=weights[mask])) if mask.any() else float('nan')
            else:
                fme_v = valid_data(fme_flat)
                fme_mean = float(fme_v.mean()) if fme_v is not None and fme_v.size > 0 else float('nan')

            # Native grid (ne30pg2) is quasi-equal-area, unweighted mean is fine
            leg_v = valid_data(leg_flat)
            leg_mean = float(leg_v.mean()) if leg_v is not None and leg_v.size > 0 else float('nan')

            if np.isnan(fme_mean) or np.isnan(leg_mean):
                results.append({"test": "GMEAN", "field": var, "status": "SKIP",
                                "detail": "no valid data"})
                print(f"  {var}: SKIP (no valid data)")
                continue

            rel_diff = abs(fme_mean - leg_mean) / max(abs(leg_mean), 1e-30)
            # Area-weighted means should agree within remap interpolation error (~2%)
            status = "PASS" if rel_diff < 0.02 else "DIFF"
            if status == "PASS":
                n_bfb_pass += 1
            else:
                n_bfb_fail += 1
                issues.append(f"  GMEAN {var}: rel_diff = {rel_diff:.4g}")
            results.append({"test": "GMEAN", "field": var, "status": status,
                            "detail": f"fme={fme_mean:.6g}, leg={leg_mean:.6g}, rel={rel_diff:.2e}"})
            print(f"  {var}: {status}  fme={fme_mean:.6g}  leg={leg_mean:.6g}  rel={rel_diff:.2e}")

    print(f"  Summary: {n_bfb_pass} pass, {n_bfb_fail} fail, "
          f"{len(all_bfb_fields) - n_bfb_pass - n_bfb_fail} skip")

    # -- Native vs Remapped comparison plots -----------------------------------
    if HAS_MPL and not grids_match:
        print("\n--- Generating native vs remapped comparison plots ---")
        tidx = -1

        # Get native grid coordinates (legacy: lat/lon are 1D arrays of ncol)
        leg_lat = get_var(ds_leg, "lat")
        leg_lon = get_var(ds_leg, "lon")
        # Get remapped grid coordinates (FME: lat/lon are 1D coordinate arrays)
        fme_lat = get_var(ds_fme, "lat")
        fme_lon = get_var(ds_fme, "lon")

        if leg_lat is not None and leg_lon is not None and \
           fme_lat is not None and fme_lon is not None:
            # Native: 1D arrays for tripcolor
            nat_lons = leg_lon if leg_lon.ndim == 1 else leg_lon.ravel()
            nat_lats = leg_lat if leg_lat.ndim == 1 else leg_lat.ravel()
            # Remapped: meshgrid for pcolormesh
            if fme_lat.ndim == 1 and fme_lon.ndim == 1:
                rem_lons, rem_lats = np.meshgrid(fme_lon, fme_lat)
            else:
                rem_lons, rem_lats = fme_lon, fme_lat

            plot_specs = [
                ("TS",       "RdBu_r",   220, 320, "K"),
                ("PS",       "viridis",  50000, 105000, "Pa"),
                ("LHFLX",    "YlOrRd",   0,   300, "W/m2"),
                ("PRECT",    "Blues",     0,   1e-6, "m/s"),
                ("ICEFRAC",  "Blues",     0,   1,   "fraction"),
                ("FLUT",     "inferno",   100, 320, "W/m2"),
                ("SOLIN",    "YlOrRd",   0,   500, "W/m2"),
                ("DTENDTTW", "RdBu_r",   -5e-4, 5e-4, "kg/m2/s"),
            ]
            for var, cmap, vm, vx, units in plot_specs:
                nat_arr = get_var(ds_leg, var)
                rem_arr = get_var(ds_fme, var)
                if nat_arr is None or rem_arr is None:
                    continue
                nat_data = nat_arr[tidx] if nat_arr.ndim >= 2 else nat_arr
                rem_data = rem_arr[tidx] if rem_arr.ndim >= 2 else rem_arr
                # Flatten native to 1D for tripcolor
                nat_flat = nat_data.ravel()
                p = side_by_side_comparison(
                    nat_flat, nat_lons, nat_lats, f"Native ne30pg2 ({var})",
                    rem_data, rem_lons, rem_lats, f"Remapped lat-lon ({var})",
                    f"{var}: Native vs Remapped (last timestep)",
                    xv_outdir, f"xv_native_vs_remap_{var}.png",
                    cmap=cmap, vmin=vm, vmax=vx, units=units)
                if p:
                    xv_plots.append(p)
                    print(f"  Wrote xv_native_vs_remap_{var}.png")

            # Vcoarsen layers: T_7 and STW_7 (near-surface)
            for base, cmap, vm, vx, units in [("T", "RdYlBu_r", 220, 310, "K"),
                                                ("STW", "YlGnBu", 0, 0.025, "kg/kg")]:
                vname = f"{base}_7"
                fme_vc = get_var(ds_fme, vname)
                if fme_vc is not None:
                    fme_data = fme_vc[tidx] if fme_vc.ndim >= 2 else fme_vc
                    # No native vcoarsen to compare, just show remapped
                    p = global_map(fme_data, rem_lons, rem_lats,
                                   f"FME {vname} (remapped, near-surface)",
                                   cmap=cmap, vmin=vm, vmax=vx,
                                   outdir=xv_outdir, fname=f"xv_fme_{vname}_remap.png",
                                   units=units)
                    if p:
                        xv_plots.append(p)
                        print(f"  Wrote xv_fme_{vname}_remap.png")
        else:
            print("  SKIP plots: lat/lon coordinates not found")

    # -- 2. ONLINE vs OFFLINE VCOARSEN ----------------------------------------
    print("\n--- Test 2: Online vs Offline Vcoarsen ---")

    # Need hybrid coords and PS from legacy file to compute interface pressures
    hyai = get_var(ds_leg, "hyai")
    hybi = get_var(ds_leg, "hybi")
    leg_ps = get_var(ds_leg, "PS")

    if hyai is None or hybi is None or leg_ps is None:
        print("  SKIP: hyai/hybi/PS not found in legacy file")
    else:
        # Use the LAST timestep to avoid initialization artifacts
        tidx = -1
        ps_1t = leg_ps[tidx]  # (ncol,)
        pint = compute_pint(ps_1t, hyai, hybi)  # (ncol, nlev+1)

        for fme_base, legacy_sources in FME_TO_LEGACY.items():
            # Build the full-resolution field from legacy
            # (for derived fields like STW, sum the components)
            raw_3d = None
            missing_src = False
            for src in legacy_sources:
                src_data = get_var(ds_leg, src)
                if src_data is None:
                    print(f"  {fme_base}: SKIP (legacy missing {src})")
                    missing_src = True
                    break
                src_1t = src_data[tidx]  # (lev, ncol) or (ncol,) from EAM NetCDF
                if src_1t.ndim < 2:
                    print(f"  {fme_base}: SKIP ({src} is not 3D)")
                    missing_src = True
                    break
                # EAM NetCDF stores 3D as (time, lev, ncol); after time slice: (lev, ncol)
                # offline_vcoarsen_avg expects (ncol, lev) — transpose if needed
                if src_1t.shape[0] < src_1t.shape[1]:
                    src_1t = src_1t.T  # (lev, ncol) -> (ncol, lev)
                if raw_3d is None:
                    raw_3d = src_1t.astype(np.float64)
                else:
                    raw_3d = raw_3d + src_1t.astype(np.float64)

            if missing_src or raw_3d is None:
                continue

            # Offline vcoarsen
            offline_vc = offline_vcoarsen_avg(raw_3d, pint)  # (ncol, 8)

            # Compare with online vcoarsen from FME output
            layer_results = []
            max_err_all = 0.0
            max_rel_all = 0.0
            for k in range(N_VCOARSEN_LAYERS):
                vname = f"{fme_base}_{k}"
                fme_arr = get_var(ds_fme, vname)
                if fme_arr is None:
                    layer_results.append(f"  {vname}: MISSING in FME output")
                    continue

                fme_1t = fme_arr[tidx].astype(np.float64).ravel()  # flatten (lat,lon) or (ncol,)
                off_1t = offline_vc[:, k]

                # Grids may differ (remapped vs native) — compare area-weighted global means
                if fme_1t.size != off_1t.size:
                    # Area-weight the FME (lat-lon) data
                    fme_lat = get_var(ds_fme, "lat")
                    fme_arr_2d = fme_arr[tidx].astype(np.float64)
                    if fme_lat is not None and fme_arr_2d.ndim == 2:
                        coslat = np.cos(np.deg2rad(fme_lat))
                        wt = coslat[:, np.newaxis] * np.ones(fme_arr_2d.shape[1])[np.newaxis, :]
                        m = (np.abs(fme_arr_2d) < 1e10) & np.isfinite(fme_arr_2d)
                        fme_wm = float(np.average(fme_arr_2d[m], weights=wt[m])) if m.any() else float('nan')
                    else:
                        fv = valid_data(fme_1t)
                        fme_wm = float(fv.mean()) if fv is not None else float('nan')
                    off_v = valid_data(off_1t)
                    off_wm = float(off_v.mean()) if off_v is not None else float('nan')
                    if not np.isnan(fme_wm) and not np.isnan(off_wm):
                        rel = abs(fme_wm - off_wm) / max(abs(off_wm), 1e-30)
                        max_rel_all = max(max_rel_all, rel)
                        max_err_all = max(max_err_all, abs(fme_wm - off_wm))
                    continue

                # Same grid: point-by-point comparison
                valid_mask = (np.abs(fme_1t) < 1e10) & (np.abs(off_1t) < 1e10) & \
                             np.isfinite(fme_1t) & np.isfinite(off_1t)

                if not valid_mask.any():
                    layer_results.append(f"  {vname}: no valid data")
                    continue

                diff = np.abs(fme_1t[valid_mask] - off_1t[valid_mask])
                maxdiff = float(diff.max())
                scale = np.maximum(np.abs(off_1t[valid_mask]), 1e-30)
                maxrel = float((diff / scale).max())

                max_err_all = max(max_err_all, maxdiff)
                max_rel_all = max(max_rel_all, maxrel)

            # Report — thresholds depend on whether grids match
            if grids_match:
                bfb_thresh, close_thresh = 1e-10, 1e-5
            else:
                bfb_thresh, close_thresh = 1e-5, 0.02  # remap interpolation ~2%

            if max_rel_all < bfb_thresh:
                status = "BFB"
                tag = "PASS"
            elif max_rel_all < close_thresh:
                status = "CLOSE"
                tag = "PASS"
            else:
                status = "DIFF"
                tag = "FAIL"
                issues.append(f"  VCOARSEN {fme_base}: max|rel| = {max_rel_all:.2e}")

            results.append({"test": "VCOARSEN", "field": fme_base,
                            "status": status,
                            "detail": f"max|diff|={max_err_all:.2e}, max|rel|={max_rel_all:.2e}"})
            print(f"  {fme_base}_0..{fme_base}_{N_VCOARSEN_LAYERS-1}: {tag}  "
                  f"max|diff|={max_err_all:.2e}  max|rel|={max_rel_all:.2e}")

            # Generate difference map for last layer (skip if grids differ)
            if HAS_MPL and HAS_XR and grids_match:
                last_k = N_VCOARSEN_LAYERS - 1
                vname = f"{fme_base}_{last_k}"
                fme_arr = get_var(ds_fme, vname)
                if fme_arr is not None:
                    fme_1t = fme_arr[tidx].astype(np.float64)
                    off_1t = offline_vc[:, last_k]
                    diff_map = fme_1t - off_1t

                    # Get coordinates
                    lon_name = next((d for d in ["lon", "ncol"] if d in get_dims(ds_fme)), None)
                    lat_var = get_var(ds_fme, "lat")
                    lon_var = get_var(ds_fme, "lon")
                    if lat_var is not None and lon_var is not None:
                        if lat_var.ndim == 1 and lon_var.ndim == 1:
                            lons, lats = np.meshgrid(lon_var, lat_var)
                        else:
                            lons, lats = lon_var, lat_var
                        # Side-by-side: online vs offline
                        vmin_f = float(np.nanpercentile(valid_data(fme_1t) or [0], 2))
                        vmax_f = float(np.nanpercentile(valid_data(fme_1t) or [0], 98))
                        p = side_by_side_comparison(
                            fme_1t, lons, lats, f"Online {vname}",
                            off_1t, lons, lats, f"Offline vcoarsen({'+'.join(legacy_sources)})",
                            f"{fme_base}: Online vs Offline (layer {last_k})",
                            xv_outdir, f"xv_vcoarsen_{fme_base}_{last_k}.png",
                            cmap="RdYlBu_r", vmin=vmin_f, vmax=vmax_f)
                        if p:
                            xv_plots.append(p)

    # -- 3. VCOARSEN LINEARITY: STW_k == Q_k + CLDICE_k + ... ----------------
    print("\n--- Test 3: Vcoarsen Linearity (STW_k = sum of components) ---")
    # If we have offline vcoarsen of individual components (Q, CLDICE, CLDLIQ,
    # RAINQM), their sum should equal the online STW_k (since vcoarsen is linear).
    if hyai is not None and hybi is not None and leg_ps is not None:
        tidx = -1
        ps_1t = leg_ps[tidx]
        pint = compute_pint(ps_1t, hyai, hybi)

        # Offline vcoarsen each component
        components_vc = {}
        all_found = True
        for comp in ["Q", "CLDICE", "CLDLIQ", "RAINQM"]:
            arr = get_var(ds_leg, comp)
            if arr is None or arr.ndim < 3:
                all_found = False
                break
            comp_1t = arr[tidx].astype(np.float64)
            if comp_1t.shape[0] < comp_1t.shape[1]:
                comp_1t = comp_1t.T  # (lev, ncol) -> (ncol, lev)
            components_vc[comp] = offline_vcoarsen_avg(comp_1t, pint)

        if all_found:
            sum_vc = sum(components_vc.values())  # (ncol, 8)
            max_lin_err = 0.0
            max_lin_rel = 0.0

            for k in range(N_VCOARSEN_LAYERS):
                stw_k = get_var(ds_fme, f"STW_{k}")
                if stw_k is None:
                    continue
                stw_1t = stw_k[tidx].astype(np.float64).ravel()
                sum_1t = sum_vc[:, k]

                # Different grids: compare global means
                if stw_1t.size != sum_1t.size:
                    sv = valid_data(stw_1t)
                    ov = valid_data(sum_1t)
                    if sv is not None and ov is not None:
                        rel = abs(sv.mean() - ov.mean()) / max(abs(ov.mean()), 1e-30)
                        max_lin_rel = max(max_lin_rel, rel)
                        max_lin_err = max(max_lin_err, abs(sv.mean() - ov.mean()))
                    continue

                valid = (np.abs(stw_1t) < 1e10) & np.isfinite(stw_1t)
                if not valid.any():
                    continue

                diff = np.abs(stw_1t[valid] - sum_1t[valid])
                scale = np.maximum(np.abs(sum_1t[valid]), 1e-30)
                max_lin_err = max(max_lin_err, float(diff.max()))
                max_lin_rel = max(max_lin_rel, float((diff / scale).max()))

            if max_lin_rel < 1e-5:
                status = "PASS"
            else:
                status = "FAIL"
                issues.append(f"  LINEARITY STW: max|rel| = {max_lin_rel:.2e}")

            results.append({"test": "LINEARITY", "field": "STW_k",
                            "status": status,
                            "detail": f"STW_k vs sum(Q_k+CLDICE_k+CLDLIQ_k+RAINQM_k): "
                                      f"max|rel|={max_lin_rel:.2e}"})
            print(f"  STW_k linearity: {status}  max|rel|={max_lin_rel:.2e}")
        else:
            print("  SKIP linearity: component fields not all found in legacy")

    close_ds(ds_fme)
    close_ds(ds_leg)

    # -- 4. TIMING & STORAGE COMPARISON ---------------------------------------
    print("\n--- Timing & Storage ---")

    # Storage: compare total NetCDF file sizes
    def dir_nc_size(d):
        total = 0
        count = 0
        for f in glob.glob(os.path.join(d, "*.nc")):
            sz = os.path.getsize(f)
            total += sz
            count += 1
        return total, count

    fme_bytes, fme_nfiles = dir_nc_size(fme_rundir)
    leg_bytes, leg_nfiles = dir_nc_size(legacy_rundir)
    fme_gb = fme_bytes / 1e9
    leg_gb = leg_bytes / 1e9
    ratio = fme_gb / leg_gb if leg_gb > 0 else float('inf')
    print(f"  FME output:    {fme_nfiles} files, {fme_gb:.2f} GB")
    print(f"  Legacy output: {leg_nfiles} files, {leg_gb:.2f} GB")
    print(f"  Storage ratio: {ratio:.2f}x {'(smaller)' if ratio < 1 else '(larger)'}")

    results.append({"test": "STORAGE", "field": "total NC files",
                     "status": "INFO",
                     "detail": f"FME: {fme_nfiles} files / {fme_gb:.2f} GB, "
                               f"Legacy: {leg_nfiles} files / {leg_gb:.2f} GB, "
                               f"ratio: {ratio:.2f}x"})

    # Timing: parse model_timing_stats from both runs
    fme_timing = read_timing_summary(fme_rundir)
    leg_timing = read_timing_summary(legacy_rundir)

    # Try to extract total model time from timing files
    def parse_model_time(timing_lines):
        """Extract total model run time in seconds from timing summary."""
        if not timing_lines:
            return None
        for line in timing_lines:
            low = line.lower()
            # Look for lines with 'total' or 'model' and a number
            if ('total' in low or 'model' in low) and ('run' in low or 'time' in low):
                parts = line.split()
                for p in parts:
                    try:
                        val = float(p)
                        if val > 1:  # skip small numbers (likely ratios)
                            return val
                    except ValueError:
                        continue
        return None

    fme_time = parse_model_time(fme_timing)
    leg_time = parse_model_time(leg_timing)
    if fme_time and leg_time:
        overhead = ((fme_time - leg_time) / leg_time) * 100
        print(f"  FME model time:    {fme_time:.1f} s")
        print(f"  Legacy model time: {leg_time:.1f} s")
        print(f"  FME overhead: {overhead:+.1f}%")
        results.append({"test": "TIMING", "field": "model run time",
                         "status": "INFO",
                         "detail": f"FME: {fme_time:.1f}s, Legacy: {leg_time:.1f}s, "
                                   f"overhead: {overhead:+.1f}%"})
    elif fme_timing or leg_timing:
        print("  Timing data found in only one run -- cannot compare")
    else:
        print("  No timing data found in either run")

    # -- Summary --------------------------------------------------------------
    print("\n" + "-" * 50)
    n_pass = sum(1 for r in results if r["status"] in ("BFB", "PASS", "CLOSE"))
    n_fail = sum(1 for r in results if r["status"] in ("DIFF", "FAIL"))
    n_skip = sum(1 for r in results if r["status"] == "SKIP")
    n_info = sum(1 for r in results if r["status"] == "INFO")

    print(f"  CROSS-VERIFY TOTAL: {n_pass} pass, {n_fail} fail, "
          f"{n_skip} skip, {n_info} info")
    if not issues:
        print("  ALL CROSS-VERIFICATION PASSED")
    else:
        for iss in issues:
            print(iss)

    return issues, results, xv_plots


def write_cross_verify_html(outdir, results, xv_plots):
    """Write the cross-verification section of the HTML dashboard."""
    if not results:
        return ""

    # Group by test type
    tests_by_type = {}
    for r in results:
        tests_by_type.setdefault(r["test"], []).append(r)

    n_total_pass = sum(1 for r in results if r["status"] in ("BFB", "PASS", "CLOSE"))
    n_total_fail = sum(1 for r in results if r["status"] in ("DIFF", "FAIL"))
    n_total_info = sum(1 for r in results if r["status"] == "INFO")

    html = '<h2 id="Cross_Verify">Cross-Verification: FME vs Legacy</h2>\n'
    html += '<p>Both cases run identical physics from the same ICs. '
    html += 'FME diagnostics are output-only and do not modify model state. '
    html += f'<strong>{n_total_pass} pass, {n_total_fail} fail, {n_total_info} info</strong></p>\n'

    # Tabbed interface for test categories
    tab_ids = list(tests_by_type.keys())
    if xv_plots:
        tab_ids.append("PLOTS")

    html += '<div class="tabs">\n'
    for i, tid in enumerate(tab_ids):
        labels = {
            "BFB": "BFB Identity", "GMEAN": "Global Means",
            "VCOARSEN": "Vcoarsen", "LINEARITY": "Linearity",
            "STORAGE": "Storage", "TIMING": "Timing", "PLOTS": "Figures",
        }
        active = " active" if i == 0 else ""
        n_p = sum(1 for r in tests_by_type.get(tid, []) if r["status"] in ("BFB", "PASS", "CLOSE"))
        n_t = len(tests_by_type.get(tid, []))
        badge = f' <span class="badge">{n_p}/{n_t}</span>' if tid != "PLOTS" else ""
        html += f'<button class="tab{active}" onclick="showTab(\'{tid}\')">{labels.get(tid, tid)}{badge}</button>\n'
    html += '</div>\n'

    for tid, test_results in tests_by_type.items():
        n_pass = sum(1 for r in test_results if r["status"] in ("BFB", "PASS", "CLOSE"))
        n_tot = len(test_results)
        status_cls = "pass" if n_pass == n_tot else ("fail" if n_pass < n_tot else "")

        labels = {
            "BFB": "BFB Identity (shared fields match bit-for-bit)",
            "GMEAN": "Area-Weighted Global-Mean Comparison (cos-lat weighted, &lt;2% tolerance)",
            "VCOARSEN": "Online vs Offline Vcoarsen (area-weighted global mean comparison)",
            "LINEARITY": "Vcoarsen Linearity (STW_k = sum of component_k)",
            "STORAGE": "Storage Comparison", "TIMING": "Timing Comparison",
        }
        vis = "" if tid == tab_ids[0] else ' style="display:none"'
        html += f'<div class="tab-content" id="tab_{tid}"{vis}>\n'
        html += f'<h3>{labels.get(tid, tid)} '
        html += f'<span class="{status_cls}">({n_pass}/{n_tot})</span></h3>\n'

        # Sortable table
        html += '<table class="sortable"><thead><tr>'
        html += '<th onclick="sortTable(this,0)">Field</th>'
        html += '<th onclick="sortTable(this,1)">Status</th>'
        html += '<th onclick="sortTable(this,2)">Detail</th></tr></thead><tbody>\n'

        for r in test_results:
            cls = "pass" if r["status"] in ("BFB", "PASS", "CLOSE") else \
                  "fail" if r["status"] in ("DIFF", "FAIL") else ""
            html += (f'<tr><td>{r["field"]}</td>'
                     f'<td class="{cls}">{r["status"]}</td>'
                     f'<td style="font-family:monospace;font-size:0.85em">{r["detail"]}</td></tr>\n')
        html += '</tbody></table>\n'
        html += '</div>\n'

    # Figures tab
    if xv_plots:
        vis = "" if "PLOTS" == tab_ids[0] else ' style="display:none"'
        html += f'<div class="tab-content" id="tab_PLOTS"{vis}>\n'
        html += '<h3>Native vs Remapped Comparison Figures</h3>\n'
        html += '<p>Left: native ne30pg2 (tripcolor). Right: remapped Gaussian lat-lon (pcolormesh).</p>\n'
        html += '<div class="gallery">\n'
        for p in xv_plots:
            if p and os.path.exists(p):
                rel = os.path.relpath(p, outdir)
                html += (f'<div class="fig wide"><a href="{rel}">'
                         f'<img src="{rel}"/></a>'
                         f'<span>{os.path.basename(p)}</span></div>\n')
        html += '</div>\n</div>\n'

    return html


# -----------------------------------------------------------------------------
# File inventory (EAM only)
# -----------------------------------------------------------------------------

def collect_file_inventory(rundir):
    """Collect EAM file inventory data and print summary.

    Returns list of (label, count, file_list) tuples for HTML rendering.
    """
    print("\n=== File Inventory ===")
    categories = [
        ("EAM h0", "*.eam.h0.*.nc", None),
    ]
    total = 0
    inventory_data = []
    for label, pattern, exclude in categories:
        hits = find_files(rundir, pattern, exclude=exclude)
        total += len(hits)
        status = f"{len(hits)} file(s)" if hits else "NONE"
        print(f"  {label:38s} {status}")
        inventory_data.append((label, len(hits), hits))
    print(f"  {'TOTAL':38s} {total} EAM-related file(s)")
    return inventory_data


# -----------------------------------------------------------------------------
# HTML index
# -----------------------------------------------------------------------------

def write_html_index(outdir, all_plots_by_comp, all_issues, file_inventory_data,
                     all_fill_reports, timing_summary=None, extra_html=""):
    index = os.path.join(outdir, "verify_eam.html")
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

    # Group plots by component
    fig_sections = ""
    for comp, plots in all_plots_by_comp.items():
        if not plots:
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
    if extra_html:
        nav_items.append("Cross_Verify")
    for comp in all_plots_by_comp:
        if all_plots_by_comp[comp]:
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
    <title>EAM FME Output Verification</title>
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
      /* Tabs */
      .tabs {{ display:flex; flex-wrap:wrap; gap:4px; margin:16px 0 0; }}
      .tab {{ padding:8px 18px; border:none; background:#e0e4ea; color:#1a2744;
              cursor:pointer; font-weight:600; font-size:0.9em; border-radius:6px 6px 0 0;
              transition: all 0.15s; }}
      .tab:hover {{ background:#c8d0dc; }}
      .tab.active {{ background:#1a2744; color:#fff; }}
      .badge {{ background:rgba(255,255,255,0.25); padding:2px 7px; border-radius:10px;
                font-size:0.8em; margin-left:4px; }}
      .tab-content {{ background:#fff; padding:18px; border-radius:0 6px 6px 6px;
                      box-shadow:0 1px 3px rgba(0,0,0,0.08); margin-bottom:16px; }}
      /* Sortable table header */
      .sortable th {{ cursor:pointer; user-select:none; position:relative; padding-right:20px; }}
      .sortable th:after {{ content:''; position:absolute; right:6px; top:50%;
                            transform:translateY(-50%); opacity:0.3; }}
      .sortable th:hover {{ background:#2a3d5c; }}
    </style>
    <script>
    function showTab(id) {{
      document.querySelectorAll('.tab-content').forEach(function(el) {{ el.style.display='none'; }});
      document.querySelectorAll('.tab').forEach(function(el) {{ el.classList.remove('active'); }});
      var tab = document.getElementById('tab_' + id);
      if (tab) tab.style.display = '';
      // Find the button for this tab
      document.querySelectorAll('.tab').forEach(function(btn) {{
        if (btn.textContent.indexOf(id) >= 0 || btn.getAttribute('onclick').indexOf(id) >= 0)
          btn.classList.add('active');
      }});
    }}
    function sortTable(th, col) {{
      var table = th.closest('table');
      var tbody = table.querySelector('tbody');
      var rows = Array.from(tbody.querySelectorAll('tr'));
      var asc = th.dataset.sort !== 'asc';
      th.dataset.sort = asc ? 'asc' : 'desc';
      rows.sort(function(a, b) {{
        var va = a.cells[col].textContent.trim();
        var vb = b.cells[col].textContent.trim();
        var na = parseFloat(va), nb = parseFloat(vb);
        if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
        return asc ? va.localeCompare(vb) : vb.localeCompare(va);
      }});
      rows.forEach(function(r) {{ tbody.appendChild(r); }});
    }}
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
      <h1>EAM FME Online Output Verification</h1>
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

    {extra_html}

    <h2>Diagnostic Figures</h2>
    {fig_sections if fig_sections else '<p>No figures generated.</p>'}

    {timing_html}
    </div>
    <footer>Generated by verify_eam.py &mdash; EAM FME Online Output Verification for E3SM &mdash; {gen_timestamp}</footer>
    </body></html>
    """)
    with open(index, "w") as fh:
        fh.write(html)
    print(f"\nIndex written: {index}")
    return index


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify EAM FME output (vcoarsen, derived fields) and "
                    "produce diagnostic figures with optional cross-verification.")
    parser.add_argument("--rundir", required=True,
                        help="CIME RUNDIR for the FME case (fme_output testmod)")
    parser.add_argument("--outdir",
                        default="/pscratch/sd/m/mahf708/aifigs_eam_fme",
                        help="Output directory for figures and HTML index")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--legacy-rundir",
                        help="CIME RUNDIR for the legacy case (fme_legacy_output testmod) "
                             "for cross-verification")
    args = parser.parse_args()

    if not os.path.isdir(args.rundir):
        sys.exit(f"ERROR: rundir not found: {args.rundir}")

    os.makedirs(args.outdir, exist_ok=True)
    print(f"EAM FME Verification  |  rundir: {args.rundir}")
    if args.legacy_rundir:
        print(f"Legacy RUNDIR         :  {args.legacy_rundir}")
    print(f"Output directory      :  {args.outdir}")
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
        comp_issues, plots, fill_reports = result
        all_issues[name] = comp_issues
        all_plots_by_comp[name] = plots
        all_fill_reports[name] = fill_reports

    run_check("EAM", check_eam, "eam",
              args.rundir, args.verbose)

    # EAM comparison figures (multi-panel layer views, zonal means)
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
    xv_html = ""
    if args.legacy_rundir:
        if not os.path.isdir(args.legacy_rundir):
            print(f"WARNING: legacy-rundir not found: {args.legacy_rundir}")
        else:
            xv_issues, xv_results, xv_plots = cross_verify(
                args.rundir, args.legacy_rundir, args.outdir, args.verbose)
            all_issues["Cross-Verify"] = xv_issues
            if xv_plots:
                all_plots_by_comp["Cross-Verify"] = xv_plots
            xv_html = write_cross_verify_html(args.outdir, xv_results, xv_plots)

    write_html_index(args.outdir, all_plots_by_comp, all_issues,
                     file_inventory_data, all_fill_reports,
                     timing_summary=timing_summary,
                     extra_html=xv_html)

    n_total = sum(len(v) for v in all_issues.values())
    print("\n" + "=" * 70)
    print(f"RESULT: {'ALL CHECKS PASSED' if n_total == 0 else f'{n_total} ISSUES FOUND'}")
    sys.exit(0 if n_total == 0 else 1)


if __name__ == "__main__":
    main()
