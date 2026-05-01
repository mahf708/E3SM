#!/usr/bin/env python3
"""
Benchmark NetCDF4 vs Zarr read performance for the io_chunking testmod.

The io_chunking testmod produces two parallel EAM history streams that
share an identical variable list (mix of 2D and 3D fields):

    *.eam.h1.*.nc   - per-variable chunked   (timelevels:1, lev:8)
    *.eam.h2.*.nc   - contiguous (unchunked) baseline

After the run completes you (the user) convert each of these to a Zarr
store offline, preserving the original chunk layout, e.g.

    ds = xr.open_mfdataset("case.eam.h1.*.nc", combine="by_coords")
    ds.chunk({"time": 1, "lev": 8}).to_zarr("case.eam.h1.zarr", ...)

Then point this script at the four resulting stores and it will time a
representative set of xarray read patterns and print a side-by-side
comparison so you can answer "for the same chunking, is netcdf4 more
efficient than zarr in xarray?".

Usage:
    python benchmark_io.py \
        --nc-chunked   /path/to/case.eam.h1.zarr-source-dir-or-glob \
        --nc-unchunked /path/to/case.eam.h2.zarr-source-dir-or-glob \
        --zarr-chunked   /path/to/case.eam.h1.zarr \
        --zarr-unchunked /path/to/case.eam.h2.zarr \
        --repeats 3 \
        --report report.csv

Any of the four inputs may be omitted; only the supplied stores are
benchmarked. NetCDF inputs may be a single file, a directory, or a glob.
"""

from __future__ import annotations

import argparse
import csv
import gc
import glob
import os
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import xarray as xr


# 2D and 3D variable subsets from the io_chunking testmod. Anything not
# present in the dataset is silently skipped, so the script also works
# on partial outputs.
VARS_2D = ["PS", "PSL", "TS", "TREFHT", "TMQ", "PRECT", "FLUT", "FSNT",
           "LHFLX", "SHFLX", "CLDTOT", "U10", "QREFHT"]
VARS_3D = ["T", "Q", "U", "V", "OMEGA", "Z3", "RELHUM",
           "CLOUD", "CLDLIQ", "CLDICE"]


@dataclass
class StoreSpec:
    label: str
    backend: str          # "netcdf4" or "zarr"
    path: str             # file, directory, glob, or .zarr path


@dataclass
class Result:
    store: str
    backend: str
    benchmark: str
    samples_s: list = field(default_factory=list)
    bytes_read: int = 0

    @property
    def best(self) -> float:
        return min(self.samples_s) if self.samples_s else float("nan")

    @property
    def median(self) -> float:
        return statistics.median(self.samples_s) if self.samples_s else float("nan")

    @property
    def throughput_mibs(self) -> float:
        if not self.samples_s or self.bytes_read == 0:
            return float("nan")
        return (self.bytes_read / (1024 * 1024)) / self.best


@contextmanager
def timed():
    gc.collect()
    t0 = time.perf_counter()
    yield lambda: time.perf_counter() - t0
    # nothing to clean up


def open_store(spec: StoreSpec) -> xr.Dataset:
    if spec.backend == "zarr":
        return xr.open_zarr(spec.path, consolidated=None, decode_times=False)
    # netcdf4
    if os.path.isdir(spec.path):
        pattern = os.path.join(spec.path, "*.nc")
    else:
        pattern = spec.path
    files = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") \
            else [pattern]
    if not files:
        raise FileNotFoundError(f"No files matched {spec.path!r}")
    if len(files) == 1:
        return xr.open_dataset(files[0], engine="netcdf4", decode_times=False,
                               chunks={})
    return xr.open_mfdataset(files, engine="netcdf4", combine="by_coords",
                             decode_times=False, parallel=False)


def common_vars(ds: xr.Dataset) -> tuple[list[str], list[str]]:
    v2 = [v for v in VARS_2D if v in ds.variables]
    v3 = [v for v in VARS_3D if v in ds.variables]
    return v2, v3


def da_nbytes(da: xr.DataArray) -> int:
    return int(np.prod(da.shape)) * da.dtype.itemsize


# ---------- benchmarks ----------------------------------------------------

def bench_open(spec: StoreSpec, repeats: int) -> Result:
    r = Result(spec.label, spec.backend, "open_dataset")
    for _ in range(repeats):
        with timed() as elapsed:
            ds = open_store(spec)
            ds.close()
        r.samples_s.append(elapsed())
    return r


def bench_load_full(spec: StoreSpec, var: str, repeats: int, kind: str) -> Result:
    r = Result(spec.label, spec.backend, f"load_full[{kind}:{var}]")
    for _ in range(repeats):
        ds = open_store(spec)
        if var not in ds.variables:
            ds.close()
            return r
        with timed() as elapsed:
            arr = ds[var].load()
        r.samples_s.append(elapsed())
        r.bytes_read = da_nbytes(arr)
        ds.close()
    return r


def bench_time_slice(spec: StoreSpec, vars_: list[str], repeats: int,
                     kind: str) -> Result:
    r = Result(spec.label, spec.backend, f"first_time_slice[{kind}]")
    for _ in range(repeats):
        ds = open_store(spec)
        present = [v for v in vars_ if v in ds.variables]
        if not present:
            ds.close()
            return r
        with timed() as elapsed:
            sub = ds[present].isel(time=0).load()
        r.samples_s.append(elapsed())
        r.bytes_read = sum(da_nbytes(sub[v]) for v in present)
        ds.close()
    return r


def bench_single_level(spec: StoreSpec, var: str, repeats: int) -> Result:
    """Read one vertical level across all times (3D only)."""
    r = Result(spec.label, spec.backend, f"single_level_alltime[{var}]")
    for _ in range(repeats):
        ds = open_store(spec)
        if var not in ds.variables or "lev" not in ds[var].dims:
            ds.close()
            return r
        mid = ds.sizes["lev"] // 2
        with timed() as elapsed:
            arr = ds[var].isel(lev=mid).load()
        r.samples_s.append(elapsed())
        r.bytes_read = da_nbytes(arr)
        ds.close()
    return r


def bench_column_subset(spec: StoreSpec, var: str, repeats: int,
                        ncol_frac: float = 0.05) -> Result:
    """Read a contiguous ncol-subset across all times and levels."""
    r = Result(spec.label, spec.backend, f"ncol_subset[{var}]")
    for _ in range(repeats):
        ds = open_store(spec)
        if var not in ds.variables or "ncol" not in ds[var].dims:
            ds.close()
            return r
        n = ds.sizes["ncol"]
        k = max(1, int(n * ncol_frac))
        with timed() as elapsed:
            arr = ds[var].isel(ncol=slice(0, k)).load()
        r.samples_s.append(elapsed())
        r.bytes_read = da_nbytes(arr)
        ds.close()
    return r


# ---------- driver --------------------------------------------------------

def run_all(specs: list[StoreSpec], repeats: int) -> list[Result]:
    results: list[Result] = []
    for spec in specs:
        print(f"\n=== {spec.label}  [{spec.backend}]  {spec.path}")
        ds = open_store(spec)
        v2, v3 = common_vars(ds)
        print(f"    2D vars present: {v2}")
        print(f"    3D vars present: {v3}")
        sizes = {d: ds.sizes[d] for d in ("time", "lev", "ncol") if d in ds.sizes}
        print(f"    dim sizes: {sizes}")
        ds.close()

        results.append(bench_open(spec, repeats))
        if v2:
            results.append(bench_load_full(spec, v2[0], repeats, "2D"))
            results.append(bench_time_slice(spec, v2, repeats, "2D"))
        if v3:
            results.append(bench_load_full(spec, v3[0], repeats, "3D"))
            results.append(bench_time_slice(spec, v3, repeats, "3D"))
            results.append(bench_single_level(spec, v3[0], repeats))
            results.append(bench_column_subset(spec, v3[0], repeats))

        for r in results[-7:]:
            if r.samples_s:
                print(f"      {r.benchmark:35s} best={r.best:7.3f}s "
                      f"med={r.median:7.3f}s  {r.throughput_mibs:8.1f} MiB/s")
    return results


def write_csv(results: list[Result], path: str) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["store", "backend", "benchmark",
                    "best_s", "median_s", "bytes_read", "throughput_MiB_s",
                    "samples_s"])
        for r in results:
            if not r.samples_s:
                continue
            w.writerow([r.store, r.backend, r.benchmark,
                        f"{r.best:.6f}", f"{r.median:.6f}",
                        r.bytes_read, f"{r.throughput_mibs:.3f}",
                        ";".join(f"{s:.6f}" for s in r.samples_s)])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nc-chunked",     help="NetCDF4 store: chunked (h1)")
    p.add_argument("--nc-unchunked",   help="NetCDF4 store: unchunked (h2)")
    p.add_argument("--zarr-chunked",   help="Zarr store: chunked (h1)")
    p.add_argument("--zarr-unchunked", help="Zarr store: unchunked (h2)")
    p.add_argument("--repeats", type=int, default=3,
                   help="Repetitions per benchmark (default 3, best-of reported)")
    p.add_argument("--report", default=None, help="Optional CSV output path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    specs: list[StoreSpec] = []
    for label, backend, path in (
        ("nc4_chunked",     "netcdf4", args.nc_chunked),
        ("nc4_unchunked",   "netcdf4", args.nc_unchunked),
        ("zarr_chunked",    "zarr",    args.zarr_chunked),
        ("zarr_unchunked",  "zarr",    args.zarr_unchunked),
    ):
        if path:
            specs.append(StoreSpec(label, backend, path))
    if not specs:
        print("No stores supplied. Pass at least one of --nc-chunked, "
              "--nc-unchunked, --zarr-chunked, --zarr-unchunked.",
              file=sys.stderr)
        return 2

    results = run_all(specs, args.repeats)

    print("\n\n========= SUMMARY (best of N) =========")
    print(f"{'store':16s} {'backend':8s} {'benchmark':38s} "
          f"{'best_s':>8s} {'MiB/s':>10s}")
    for r in results:
        if not r.samples_s:
            continue
        print(f"{r.store:16s} {r.backend:8s} {r.benchmark:38s} "
              f"{r.best:8.3f} {r.throughput_mibs:10.1f}")

    if args.report:
        write_csv(results, args.report)
        print(f"\nWrote CSV report to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
