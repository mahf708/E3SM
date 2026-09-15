# Atmosphere emulator foundation

This branch extracts the atmosphere path from `c6a1234d8107e52c14d95b309f49c79741a6f31b`
onto fork master `80e57e9c2d083a047d05faaa64b1a454f5ae4ad7`.
It retains the shared component machinery, ACE atmosphere operators, the MCT
cap, and both inference backends. Ocean and ice emulator assemblies and their
case configuration are excluded. The second atmosphere channel schema is retained.

`DESIGN.md` is the engineering specification. The first reading path is
`driver/src/atmosphere_main.cpp`, `common/src/emulator_component.cpp`,
`common/src/model/emulated_model.cpp`, then `emulatoratm/src/ace_operators.cpp`.

## Standalone build

Required: C++20 compiler, CMake, MPI C++, yaml-cpp and netCDF-C. Python inference
also needs CPython development files and NumPy in that interpreter. LibTorch is
optional. Standalone does not require CIME, Fortran, MCT or E3SM submodules.

```sh
cmake -S components/emulators -B build-atm \
  -DEMULATOR_ENABLE_MPI=ON -DEMULATOR_ENABLE_NETCDF=ON \
  -DEMULATOR_ENABLE_PYTHON=ON -DPython3_EXECUTABLE="$(command -v python3)" \
  -DBUILD_EMULATOR_TESTS=ON -DEMULATOR_TEST_MPI_RANKS=2
cmake --build build-atm -j2
ctest --test-dir build-atm --output-on-failure --timeout 120
```

For an offline build, provide installed yaml-cpp through `CMAKE_PREFIX_PATH`, or
set `FETCHCONTENT_SOURCE_DIR_YAML-CPP` to the pinned source checkout. Set
`CATCH2_INCLUDE_DIR` to a directory containing `catch2/catch.hpp` (Catch2 v2).
Set `NETCDF_ROOT` if netCDF cannot be discovered. The CMake fallbacks fetch
missing yaml-cpp/Catch2 sources and therefore need network access.

Enable the native checkpoint backend with `-DEMULATOR_ENABLE_LIBTORCH=ON`
and `-DCMAKE_PREFIX_PATH=/path/to/libtorch`. A Python Torch installation can
supply the matching C++ package: `python3 -c 'import torch; print(torch.utils.cmake_prefix_path)'`.
Match its C++ ABI to the compiler and to the rest of the E3SM executable.

## Self-contained software demonstration

With NumPy, SciPy and PyYAML in Python:

```sh
python3 components/emulators/examples/make_fixture.py demo
mpiexec -n 2 build-atm/driver/emulatoratm_driver demo/python_fixture_full.yaml
mpiexec -n 2 build-atm/driver/emulatoratm_driver demo/python_fixture_segment.yaml
mpiexec -n 1 build-atm/driver/emulatoratm_driver demo/python_fixture_resume.yaml
python3 components/emulators/examples/check_results.py demo python_fixture
```

The generated eight-cell grid and deterministic network use the ACE2 channel
contract and the same atmosphere operators as a real model. They test software
behavior, not climate fidelity. The segment stops seven coupler steps into a
six-hour interval. The comparison checks the restart at one rank against the
continuous run at two ranks, including every exported field's global mean.
The existing restart/gather tests also check local field values.

Add `--torch` to the generator to create `model.pt` and `libtorch_*` and
`python_torch_*` configurations. Run full, segment and resume for each backend,
then pass all three backend names to `check_results.py`. The native build must
have LibTorch enabled. CPU agreement allows float32 roundoff. This does not
establish cross-device bitwise equivalence.

The GitHub workflow runs both builds and these demonstrations. Tests that need
private ACE/GPU inputs are a separate validation tier and may skip when those
assets are absent. A green fixture test does not establish a real-model or
hybrid-case result.

## Standalone with an atmosphere checkpoint

Create an atmosphere input file with absolute paths, or paths relative to that
file:

```yaml
spec: /checkout/components/emulators/specs/ace2-eamv3.yaml
coupler_dt: 1800
grid: {file: /data/atmosphere.scrip.nc, domain: full}
initial_condition: /data/atmosphere_ic.nc
inference:
  backend: python
  python_module: e3sm_emulator.torchscript
  model_path: /data/atmosphere.pt
  device: cpu
  seed: 2026
```

For the native runtime set `backend: libtorch`. Both load a TorchScript export,
not an arbitrary training checkpoint. Channel order, normalization, correction
operators, dtype and grid order must match that export. A trace with baked-in
CUDA devices needs a compatible GPU or a fresh CPU export.

The driver configuration is separate:

```yaml
component: atm_in.yaml
start_ymd: 19710101
start_tod: 0
steps: 48
surface:
  file: /data/prescribed_surface.nc
  variables:
    Sf_lfrac: LANDFRAC
    Sf_ofrac: OCNFRAC
    Sf_ifrac: ICEFRAC
    Sx_t: TS
output: atmosphere_means.csv
restart_out: atmosphere.restart
```

The prescribed surface remains fixed during this run. Fractions must follow
the atmosphere surface contract, and `Sx_t` must be the merged surface
contribution expected by `ace.surface_inputs`. All variables must share the
model's exact grid ordering and have shape `[lat,lon]` or `[1,lat,lon]`. The
CSV contains area-weighted means of exports, including the initial state.
Use the component history configuration for spatial diagnostics.

To continue, set `restart_in`, the matching `start_ymd/start_tod`, a new output
name and the number of additional steps. Only the NO_LEAP calendar is supported.
This driver is a prescribed-boundary application. Time-varying forcing streams
are a subsequent driver extension.

## Hybrid E3SM configuration

Hybrid means emulator atmosphere with conventional surface components. The
existing `atm_comp_mct` cap uses the same `EmulatorComponent` as the standalone
driver. The E3SM build compiles Fortran only for this integration.

Use a supported E3SM machine and a case whose long compset selects
`EMULATORATM` and whose atmosphere grid matches the checkpoint. Choose the other
components and calendar consistently with that experiment. A conventional
atmosphere grid alias does not regrid a neural network. The case needs matching
SCRIP geometry, land/ocean domains and conservative flux maps to its surface
components. This branch deliberately carries no personal scratch paths or
unverified Gaussian-to-surface mapping files.

After creating that case, before building namelists:

```sh
./xmlchange EMULATORATM_GRID_FILE=/data/atmosphere.scrip.nc
./xmlchange EMULATORATM_IC_FILE=/data/atmosphere_ic.nc
./xmlchange EMULATORATM_MODEL_FILE=/data/atmosphere.pt
./xmlchange EMULATORATM_SPEC=ace2-eamv3.yaml
./xmlchange EMULATORATM_BACKEND=python,EMULATORATM_DEVICE=cpu
./xmlchange ATM_NCPL=48,CALENDAR=NO_LEAP
```

Enable the selected backend in the integrated CMake configuration using the
machine's supported CMake arguments mechanism. For Python it must receive
`-DEMULATOR_ENABLE_PYTHON=ON` and the intended `Python3_EXECUTABLE`. For LibTorch,
use `-DEMULATOR_ENABLE_LIBTORCH=ON` and its package path (`Torch_ROOT` also enables
it through the existing build helper). Configure compatible compiler, MPI and
ML libraries together.

`user_nl_emulatoratm` supports dotted YAML overrides, including
`inference.python_module: e3sm_emulator.torchscript`, `inference.seed: 2026`, and
`history.fields: [state.PS, exports.Sa_tbot]`. Files and device/backend choices
are runtime settings. Compiling a backend into the executable is a build setting.
Run `case.setup`, `case.build` and the site's normal submission procedure.

Hybrid acceptance requires a smoke run and an exact restart test on that
machine, plus checks of flux signs, mapped coverage and interface budgets.
The standalone fixture cannot certify those properties. No hybrid E3SM case
has been executed as part of this extraction.
