# Emulated components

Machine-learned emulators as E3SM components. This tree builds the framework
(`common/`), the ACE atmosphere on it (`emulatoratm/`), and the CIME glue that
runs it alone or coupled to E3SM's own components.

## Build and test standalone

Only a C++20 compiler, CMake ≥ 3.16 and MPI are required. Everything else is
optional and switched on explicitly:

| Option | Needs | Enables |
|---|---|---|
| `EMULATOR_ENABLE_NETCDF` | serial netCDF-C (`-DNETCDF_ROOT=...`) | grid, initial-condition, restart and history files |
| `EMULATOR_ENABLE_LIBTORCH` | libtorch (`-DCMAKE_PREFIX_PATH=<libtorch>`) | the TorchScript backend |
| `EMULATOR_ENABLE_PYTHON` | Python dev headers, numpy | the embedded-Python backend |

```sh
cmake -S components/emulators -B build -DBUILD_EMULATOR_TESTS=ON \
      -DEMULATOR_ENABLE_NETCDF=ON -DNETCDF_ROOT=/path/to/netcdf \
      -DEMULATOR_ENABLE_LIBTORCH=ON -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build build -j
ctest --test-dir build
```

Tests that need data (a grid file, a checkpoint, a GPU) say so and skip when
it is missing, so the suite runs on a laptop. `EMULATOR_TEST_MPI_RANKS=4`
also runs the MPI tests on four ranks.

## Run in CIME

```sh
cd cime/scripts
./create_newcase --case emu-ace2 --compset EMU2000-ACE2 \
                 --res gauss180x360_gauss180x360 --mach pm-gpu --compiler gnugpu
./create_newcase --case gmpas-ace2 --compset GMPAS-EMU-ACE2 \
                 --res gauss180x360_IcoswISC30E3r5 --mach pm-gpu --compiler gnugpu
```

The atmosphere runs on 4 ranks with inference on the root rank's GPU
(`NTASKS_ATM=4`). Its settings (`EMULATORATM_SPEC`, `_MODEL_FILE`, `_IC_FILE`,
`_DEVICE`, `_GRID_FILE`) are `env_run.xml` variables; anything the component
reads can be overridden in `user_nl_emulatoratm` as `dotted.key: value`, for
example `inference.seed: 2027` or `history.interval: 1d`.

## Layout

```
specs/          one YAML per emulator: the checkpoint's tensor contract, its
                stepping, its operators, and the coupler fields it exchanges
common/src/     the framework: config, fields, grid, coupling, model,
                inference, physics, and the C / Fortran API with the MCT cap
emulatoratm/    the ACE atmosphere: its operators and physics, its cap
driver/         emulator_create(kind), for tools and tests
cime_config/    input-file writer shared by the components' buildnml
```

See `DESIGN.md` for the design.
