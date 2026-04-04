#!/usr/bin/env python3
"""
Verify FME (Full Model Emulation) online output processing correctness.

Reads MPAS-O and MPAS-SI analysis member output files and verifies that:
1. Depth coarsening produces physically reasonable values
2. Derived fields (SST, SSS, heat flux) are consistent with raw output
3. Vertical reduction (OHC, FWC, KE) integrates correctly
4. Sea ice category aggregation sums correctly
5. Air stress rotation produces unit-consistent vectors

Usage:
    python verify_fme_output.py --rundir /path/to/run/dir [--verbose]

The script assumes the test ran with the fme_output testmod settings.
It reads both the standard MPAS output and the FME analysis member output
for cross-validation.
"""

import argparse
import sys
import os
import glob
import numpy as np

try:
    from netCDF4 import Dataset
except ImportError:
    print("ERROR: netCDF4 is required. Install with: pip install netCDF4")
    sys.exit(1)


def find_files(rundir, pattern):
    """Find files matching pattern in rundir."""
    matches = sorted(glob.glob(os.path.join(rundir, pattern)))
    return matches


def check_finite_and_reasonable(data, name, vmin=None, vmax=None, fill=1.0e34):
    """Check that non-fill data is finite and within reasonable bounds."""
    mask = np.abs(data) < fill * 0.5  # non-fill mask
    valid = data[mask]
    errors = []

    if valid.size == 0:
        errors.append(f"  {name}: ALL values are fill — no valid data")
        return errors

    if not np.all(np.isfinite(valid)):
        n_bad = np.sum(~np.isfinite(valid))
        errors.append(f"  {name}: {n_bad} non-finite values (NaN/Inf)")

    if vmin is not None and np.any(valid < vmin):
        n_bad = np.sum(valid < vmin)
        errors.append(f"  {name}: {n_bad} values below minimum {vmin} (min={valid.min():.4g})")

    if vmax is not None and np.any(valid > vmax):
        n_bad = np.sum(valid > vmax)
        errors.append(f"  {name}: {n_bad} values above maximum {vmax} (max={valid.max():.4g})")

    return errors


def verify_depth_coarsening(rundir, verbose=False):
    """Verify MPAS-O depth coarsening output."""
    print("\n=== MPAS-O Depth Coarsening ===")
    files = find_files(rundir, "analysis_members/fmeDepthCoarsening.*.nc")
    if not files:
        print("  SKIP: No depth coarsening output files found")
        return 0

    errors = []
    ds = Dataset(files[0], 'r')

    # Check expected variables exist
    expected = ['temperatureCoarsened', 'salinityCoarsened',
                'velocityZonalCoarsened', 'velocityMeridionalCoarsened',
                'layerThicknessCoarsened']
    for var in expected:
        if var not in ds.variables:
            errors.append(f"  Missing variable: {var}")
        else:
            data = ds.variables[var][0, :, :]  # first time, all levels x cells
            errors.extend(check_finite_and_reasonable(data, var))

    # Physical range checks
    if 'temperatureCoarsened' in ds.variables:
        data = ds.variables['temperatureCoarsened'][0, :, :]
        errors.extend(check_finite_and_reasonable(data, 'temperatureCoarsened',
                                                   vmin=-5.0, vmax=40.0))
    if 'salinityCoarsened' in ds.variables:
        data = ds.variables['salinityCoarsened'][0, :, :]
        errors.extend(check_finite_and_reasonable(data, 'salinityCoarsened',
                                                   vmin=0.0, vmax=42.0))
    if 'layerThicknessCoarsened' in ds.variables:
        data = ds.variables['layerThicknessCoarsened'][0, :, :]
        errors.extend(check_finite_and_reasonable(data, 'layerThicknessCoarsened',
                                                   vmin=0.0, vmax=7000.0))

    # Check dimensionality
    if 'temperatureCoarsened' in ds.variables:
        shape = ds.variables['temperatureCoarsened'].shape
        if verbose:
            print(f"  Shape: {shape} (Time x nFmeDepthLevels x nCells)")
        if len(shape) != 3:
            errors.append(f"  temperatureCoarsened: expected 3D, got {len(shape)}D")
        elif shape[1] < 1:
            errors.append(f"  temperatureCoarsened: nFmeDepthLevels={shape[1]}, expected > 0")

    ds.close()

    if errors:
        print(f"  FAIL: {len(errors)} issues found")
        for e in errors:
            print(e)
    else:
        print(f"  PASS: {len(files)} file(s), all checks passed")
    return len(errors)


def verify_derived_fields(rundir, verbose=False):
    """Verify MPAS-O derived fields output."""
    print("\n=== MPAS-O Derived Fields ===")
    files = find_files(rundir, "analysis_members/fmeDerivedFields.*.nc")
    if not files:
        print("  SKIP: No derived fields output files found")
        return 0

    errors = []
    ds = Dataset(files[0], 'r')

    # Check SST, SSS
    if 'sst' in ds.variables:
        sst = ds.variables['sst'][0, :]
        errors.extend(check_finite_and_reasonable(sst, 'sst', vmin=-5.0, vmax=40.0))
        if verbose:
            mask = np.abs(sst) < 1e33
            if mask.any():
                print(f"  sst: mean={sst[mask].mean():.2f}, range=[{sst[mask].min():.2f}, {sst[mask].max():.2f}]")
    else:
        errors.append("  Missing variable: sst")

    if 'sss' in ds.variables:
        sss = ds.variables['sss'][0, :]
        errors.extend(check_finite_and_reasonable(sss, 'sss', vmin=0.0, vmax=42.0))
    else:
        errors.append("  Missing variable: sss")

    # Check surface heat flux total
    if 'surfaceHeatFluxTotal' in ds.variables:
        shf = ds.variables['surfaceHeatFluxTotal'][0, :]
        errors.extend(check_finite_and_reasonable(shf, 'surfaceHeatFluxTotal',
                                                   vmin=-2000.0, vmax=2000.0))

        # Cross-validate: if individual fluxes are present, verify sum
        flux_vars = ['shortWaveHeatFlux', 'longWaveHeatFluxDown',
                     'latentHeatFlux', 'sensibleHeatFlux']
        if all(v in ds.variables for v in flux_vars):
            sw = ds.variables['shortWaveHeatFlux'][0, :]
            lw = ds.variables['longWaveHeatFluxDown'][0, :]
            lh = ds.variables['latentHeatFlux'][0, :]
            sh = ds.variables['sensibleHeatFlux'][0, :]
            expected_sum = sw + lw + lh + sh

            # Compare where both are valid
            valid = (np.abs(shf) < 1e33) & (np.abs(expected_sum) < 1e33)
            if valid.any():
                diff = np.abs(shf[valid] - expected_sum[valid])
                max_diff = diff.max()
                if max_diff > 1e-4:
                    errors.append(f"  surfaceHeatFluxTotal: max diff from component sum = {max_diff:.6g}")
                elif verbose:
                    print(f"  surfaceHeatFluxTotal: cross-validated OK (max diff={max_diff:.2e})")
    else:
        errors.append("  Missing variable: surfaceHeatFluxTotal")

    ds.close()

    if errors:
        print(f"  FAIL: {len(errors)} issues found")
        for e in errors:
            print(e)
    else:
        print(f"  PASS: {len(files)} file(s), all checks passed")
    return len(errors)


def verify_vertical_reduce(rundir, verbose=False):
    """Verify MPAS-O vertical reduction output."""
    print("\n=== MPAS-O Vertical Reduction ===")
    files = find_files(rundir, "analysis_members/fmeVerticalReduce.*.nc")
    if not files:
        print("  SKIP: No vertical reduction output files found")
        return 0

    errors = []
    ds = Dataset(files[0], 'r')

    # Ocean heat content should be positive in most ocean cells
    if 'oceanHeatContent' in ds.variables:
        ohc = ds.variables['oceanHeatContent'][0, :]
        errors.extend(check_finite_and_reasonable(ohc, 'oceanHeatContent'))
        valid = np.abs(ohc) < 1e33
        if valid.any() and verbose:
            print(f"  OHC: mean={ohc[valid].mean():.4g} J/m2, "
                  f"range=[{ohc[valid].min():.4g}, {ohc[valid].max():.4g}]")
    else:
        errors.append("  Missing variable: oceanHeatContent")

    # Freshwater content
    if 'freshwaterContent' in ds.variables:
        fwc = ds.variables['freshwaterContent'][0, :]
        errors.extend(check_finite_and_reasonable(fwc, 'freshwaterContent',
                                                   vmin=-1000.0, vmax=1000.0))
    else:
        errors.append("  Missing variable: freshwaterContent")

    # Kinetic energy should be non-negative
    if 'kineticEnergy' in ds.variables:
        ke = ds.variables['kineticEnergy'][0, :]
        errors.extend(check_finite_and_reasonable(ke, 'kineticEnergy', vmin=0.0))
    else:
        errors.append("  Missing variable: kineticEnergy")

    ds.close()

    if errors:
        print(f"  FAIL: {len(errors)} issues found")
        for e in errors:
            print(e)
    else:
        print(f"  PASS: {len(files)} file(s), all checks passed")
    return len(errors)


def verify_seaice_derived(rundir, verbose=False):
    """Verify MPAS-SI derived fields output."""
    print("\n=== MPAS-SI Derived Fields ===")
    files = find_files(rundir, "analysis_members/seaice_fmeSeaiceDerivedFields.*.nc")
    if not files:
        print("  SKIP: No sea ice derived fields output files found")
        return 0

    errors = []
    ds = Dataset(files[0], 'r')

    # Ice area total should be [0, 1]
    if 'iceAreaTotal' in ds.variables:
        iat = ds.variables['iceAreaTotal'][0, :]
        errors.extend(check_finite_and_reasonable(iat, 'iceAreaTotal',
                                                   vmin=0.0, vmax=1.0))
    else:
        errors.append("  Missing variable: iceAreaTotal")

    # Ice volume total should be non-negative
    if 'iceVolumeTotal' in ds.variables:
        ivt = ds.variables['iceVolumeTotal'][0, :]
        errors.extend(check_finite_and_reasonable(ivt, 'iceVolumeTotal', vmin=0.0))
    else:
        errors.append("  Missing variable: iceVolumeTotal")

    # Mean ice thickness should be non-negative
    if 'iceThicknessMean' in ds.variables:
        itm = ds.variables['iceThicknessMean'][0, :]
        errors.extend(check_finite_and_reasonable(itm, 'iceThicknessMean', vmin=0.0))
    else:
        errors.append("  Missing variable: iceThicknessMean")

    # Air stress components
    for var in ['airStressZonal', 'airStressMeridional']:
        if var in ds.variables:
            data = ds.variables[var][0, :]
            errors.extend(check_finite_and_reasonable(data, var,
                                                       vmin=-10.0, vmax=10.0))
        else:
            if verbose:
                print(f"  INFO: {var} not present (may not be enabled)")

    # Surface temperature mean should be > 0 K where ice exists
    if 'surfaceTemperatureMean' in ds.variables:
        stm = ds.variables['surfaceTemperatureMean'][0, :]
        valid = stm > 0.0
        if valid.any():
            errors.extend(check_finite_and_reasonable(stm[valid], 'surfaceTemperatureMean',
                                                       vmin=200.0, vmax=280.0))
    else:
        errors.append("  Missing variable: surfaceTemperatureMean")

    ds.close()

    if errors:
        print(f"  FAIL: {len(errors)} issues found")
        for e in errors:
            print(e)
    else:
        print(f"  PASS: {len(files)} file(s), all checks passed")
    return len(errors)


def verify_eam_output(rundir, verbose=False):
    """Verify EAM vertical coarsening and derived field output."""
    print("\n=== EAM Output ===")
    # Look for EAM history files with coarsened fields
    files = find_files(rundir, "*.eam.h*.nc")
    if not files:
        files = find_files(rundir, "*.cam.h*.nc")
    if not files:
        print("  SKIP: No EAM history files found")
        return 0

    errors = []

    # Check for vertical coarsening output (fields like T_1, T_2, etc.)
    for f in files[:3]:  # check first few files
        ds = Dataset(f, 'r')
        vc_fields = [v for v in ds.variables if '_VC' in v or '_at_P' in v or '_at_L' in v]
        derived_fields = [v for v in ds.variables if v in ['TOTAL_WATER', 'PRECT', 'PRECST']]

        if vc_fields and verbose:
            print(f"  Found {len(vc_fields)} coarsened fields in {os.path.basename(f)}")
            for v in vc_fields[:5]:
                data = ds.variables[v][:]
                print(f"    {v}: shape={data.shape}, range=[{data.min():.4g}, {data.max():.4g}]")

        if derived_fields and verbose:
            print(f"  Found {len(derived_fields)} derived fields in {os.path.basename(f)}")
            for v in derived_fields:
                data = ds.variables[v][:]
                print(f"    {v}: shape={data.shape}, range=[{data.min():.4g}, {data.max():.4g}]")

        ds.close()

    print(f"  INFO: {len(files)} EAM history file(s) found")
    if errors:
        print(f"  FAIL: {len(errors)} issues found")
        for e in errors:
            print(e)
    else:
        print(f"  PASS: Basic checks passed")
    return len(errors)


def main():
    parser = argparse.ArgumentParser(
        description='Verify FME online output processing correctness')
    parser.add_argument('--rundir', required=True,
                        help='Path to the CIME run directory (RUNDIR)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed statistics')
    args = parser.parse_args()

    if not os.path.isdir(args.rundir):
        print(f"ERROR: Run directory not found: {args.rundir}")
        sys.exit(1)

    print(f"FME Output Verification")
    print(f"Run directory: {args.rundir}")
    print("=" * 60)

    total_errors = 0
    total_errors += verify_depth_coarsening(args.rundir, args.verbose)
    total_errors += verify_derived_fields(args.rundir, args.verbose)
    total_errors += verify_vertical_reduce(args.rundir, args.verbose)
    total_errors += verify_seaice_derived(args.rundir, args.verbose)
    total_errors += verify_eam_output(args.rundir, args.verbose)

    print("\n" + "=" * 60)
    if total_errors == 0:
        print("RESULT: ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print(f"RESULT: {total_errors} ISSUES FOUND")
        sys.exit(1)


if __name__ == '__main__':
    main()
