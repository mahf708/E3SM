# Emulated components: design

This directory builds machine-learned emulators as E3SM components. The Fortran
emulators in `components/emulator_comps` (EATM, EOCN, EICE on
`mahf708/eocn/add-samudra`) are the knowledge base: what they measured decides
what is built here and how it is tested. Their code is not translated.

## Layers

```
 MCT driver
   │  x2c / c2x attribute vectors, EClock, infodata
 ┌─┴──────────────────────────────────────────────────────────────┐
 │ Fortran cap          emulatoratm/src/atm_comp_mct.F90           │  thin: gsMap, domain, buffers, time
 ├────────────────────────────────────────────────────────────────┤
 │ C API                common/src/emulator_c_api.cpp              │  opaque handles; every call guarded
 ├────────────────────────────────────────────────────────────────┤
 │ Emulator base        common/src/include/emulator.hpp            │  lifecycle, domain, coupler exchange
 │ Components           emulatoratm/src/atm.cpp (EmulatorAtm)      │  configuration, owns the model below
 │                      emulatorocn/src/ocn.cpp (EmulatorOcn)      │  (ocean and ice: no caps yet)
 │                      emulatorice/src/ice.cpp (EmulatorIce)      │
 │ Model logic          emulatoratm/src/ace_atmosphere.cpp         │  independent of MCT
 │                      emulatorocn/src/samudra_ocean.cpp          │
 │                      emulatorice/src/sea_ice_surface.cpp        │
 ├────────────────────────────────────────────────────────────────┤
 │ fields/    FieldList, FieldSet, CouplerBinding, MaskSet,         │
 │            ChannelLayout                                         │
 │ grid/      HorizontalGrid, Decomposition, Domain, read_scrip,    │
 │            GlobalGather, read_grid_fields                         │
 │ coupling/  LongStepClock, IntervalMean, BracketedState,          │
 │            RestartStore, NetworkStepper, Exchange, SharedDomain,  │
 │            julian_day_noleap                                      │
 │ inference/ Tensor, InferenceBackend: stub, python, libtorch       │
 └────────────────────────────────────────────────────────────────┘
```

Dependencies point down only. Nothing under `common/` knows about a
checkpoint, and no component knows how a backend runs a network.

## Rules the code holds itself to

**A component never touches a coupler buffer.** It declares the fields it reads
and writes (`Emulator::coupling_fields`). The base class binds them by name in
`setup_coupling`, pulls imports before `run_impl`, and pushes exports after
`init_impl` and after every `run_impl`. Names are checked once, at bind time.
A required import the coupler does not send, or an export it does not carry, is
an error that names the near misses. Coupler fields a component does not produce
are written as zero on every push and listed, rather than left stale.

**MCT's layout is written down once.** `rAttr(nflds, lsize)` arrives in C
point-major: field `f` at point `p` is `data[p * nflds + f]`. `AttrVectView` is
the only code that does that arithmetic. It also refuses a field list shorter
than `nRattr`, which is what the cap's old `len=256` buffers produced.

**Storage is spans.** A `FieldSet` field is a `std::span<double>`, which is also
what the C++ coupler API on `emulators/coupler-infrastructure` carries in its
`FieldBuffer`. An adapter over the same fields was written and tested there
(`RegistryAdapter`, commit 7804eaf455 on `mahf708/emulators/coupled-emulators`)
and can come here when that API merges; no component storage changes.

**Time is the driver's.** `emulator_run_at(handle, dt, ymd, tod)` passes the
clock through. `LongStepClock` takes one call per coupler step, and a second call
at the same model time changes nothing: it neither advances nor counts. Its
completed-step count is the index a stochastic backend seeds from.

**Every lesson is a test that fails without it.** Where a mutation check was
run, the commit says how many cases fail.

## What emulator_comps taught, and where it lives

| Lesson (measured in emulator_comps) | Here | Test |
|---|---|---|
| Reseeding once at init is not restart-safe (1.07 → 0.003 K) | `InferenceBackend::set_step`, LibTorch `seed` | fresh backend at step 6 draws what the continuous run drew, CPU and CUDA |
| TorchScript re-optimizes after its first calls; float32 output changes (found here, 7.7 Pa PS/day) | LibTorch `jit_optimize` off by default | real ACE2: second run identical; mid-interval restart exact at 29/29 steps |
| ML kernels raise benign FPEs | `FpeGuard`, compiled unconditionally | traps restored, flags cleared, guards nest |
| A mask belongs to a channel (ice on the ocean mask froze tropics; 6.96 → 2.28 K) | `MaskSet`, `FieldSpec::mask` | ice channel reaches the coupler only inside its mask |
| Ocean mask and frac must be binary for `seq_domain_mct` | `grid::Domain::masked` | a continuous fraction is refused |
| The skeleton reported latitude 0 everywhere | `EmulatorAtm` reads `grid_file` | 4 ranks match the Gaussian SCRIP file cell for cell |
| Calendar arithmetic can't express a 5-day step; the driver repeats run | `LongStepClock` | 240 steps → one advance; repeats neither count nor advance |
| Flux channels are interval means; one sample is not | `IntervalMean::add(FieldSet)` | a missing channel is an error and leaves the sums untouched |
| Mean channels must be held, not interpolated (−42 W/m²) | `BracketedState::Temporal` | a mean channel is held at the upper bracket |
| SOLIN is a window mean (330 W/m² RMS from instantaneous, same global mean; 14 W/m²) | `atm::Insolation::window_mean` | same global mean, RMS > 200; 48 sub-steps within 0.1 W/m² |
| Stub land: `Sf_lfrac` 0, `Sx_t` 0 K over land (150 W/m²) | `atm::compute_surface_inputs` | stub-land cell gets LANDFRAC 0.3 and a real TS |
| Near-surface state, humidity cap, frozen-precip units, diurnal shortwave | `atm::compute_surface_exports` | hand-computed cases per formula |
| A fill value of 9.97e36 passes a finiteness check | `grid::read_grid_fields` counts fill-like values | NaN and _FillValue counted separately |
| One NaN in a global network spreads everywhere | `NetworkStepper` checks every output; verdict broadcast | NaN step raises on all 8 ranks, channel and cell named |
| Feed the raw prediction back, not the blended export | `NetworkStepper` step 5 | prognostic inputs equal the prediction |
| Coupler ocean fluxes are open-water weighted; unweighting kept FSDS/FLDS within 3% (−22%/−28% without) | `ocn::coupler_forcing_sample` | hand-computed unweighting, 1% floor, signs |
| Ocean forcing is the mean over the 5-day window that just closed | `SamudraOcean` + `IntervalMean` | window close at step 240; restart mid-window exact over 200 steps |
| The atmosphere emulator's own fluxes drive the ocean (SamudrACE); its SST feeds back | `coupling::Exchange` | real SamudrACE atmosphere + ocean, 10 coupled days, identical on 1 and 4 ranks |
| The coupler needs ice the ocean already predicts; with no ice component the polar ocean is open water (EICE) | `EmulatorIce` reports `ocn.sea_ice_fraction` | real ocean + ice through MCT buffers: ice reports the ocean's previous-step fraction exactly, 48/48 steps, 1 and 4 ranks |
| The ice grid must be the ocean's, and a mismatched decomposition must fail, not mis-index | `coupling::publish_domain` / `shared_domain`; collective check in `EmulatorIce::create_instance` | no ocean → error on every rank; too few ranks → "18 of the ocean's 36" |
| `Si_t` blended by `ifrac` is weighted twice by the merge | `ice::prescribed_skin_temperature`, unblended | 1% and 95% ice cells report one skin; mutation fails 2 of 9 cases |
| At init `x2i` is zero; the bulk scheme then makes NaN, which survives the merge and killed EAM's first step | `ice::bulk_fluxes_defined`; zero, not `spval`, where skipped | zero state → finite exports, `Si_tref = Si_t` |
| The ocean emulator's step contains its ice's melt; handing it over again double-counts | `Fioi_melth/meltw/salt/swpen` zero; `Fioi_taux/y` = atmosphere-ice stress | melt zero, stress passed through; mutation fails 1 case |
| dice's atmosphere-ice bulk formulae | `ice::atm_ice_fluxes` | EICE's own Fortran on five cells, to 1e-12 |

## Deliberate differences from the Fortran

- No pre-review "legacy" surface and no ablation switches. The measured choice
  is the code; an ablation belongs in a branch or a test.
- The near-surface layer without its four channels is an error, not a silent
  fallback to the lowest level.
- Surface-fraction repairs beyond 0.05 stop the run: they mean `Sx_t` was merged
  from different fractions.
- Every rank reads the grid and runs its columns; EOCN ran with `lsize = gsize`.
- Restart state is saved by each primitive (`save_to` / `load_from`), so a piece
  of state cannot be left out.

## Building and testing

The framework is C++20. On Perlmutter the default `/usr/bin/c++` is GCC 7.5, so
use the Cray wrappers:

```bash
export CC=cc CXX=CC FC=ftn
./test                # bare build
./test --python       # plus the embedded-Python backend
```

Fuller configurations, on Perlmutter:

```bash
cmake -S . -B build-full -DCMAKE_BUILD_TYPE=Release -DBUILD_EMULATOR_TESTS=ON \
  -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DCMAKE_Fortran_COMPILER=ftn \
  -DEMULATOR_ENABLE_PYTHON=ON -DEMULATOR_ENABLE_NETCDF=ON \
  -DNETCDF_ROOT=/opt/cray/pe/netcdf/4.9.2.3/gnu/12.3 \
  -DEMULATOR_ENABLE_LIBTORCH=ON -DCMAKE_PREFIX_PATH=<libtorch> \
  -DEMULATOR_TORCH_PYTHON=/pscratch/sd/m/mahf708/ace/.venv/bin/python
```

- CUDA libtorch: point `CMAKE_PREFIX_PATH` at the ACE venv's
  `torch/share/cmake` and add `-DCUDAToolkit_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`.
  Run with the GTL shim on `LD_LIBRARY_PATH` and `MPICH_GPU_SUPPORT_ENABLED=0`.
- MPI tests run on one rank under `ctest`. `-DEMULATOR_TEST_MPI_RANKS=4` adds
  launcher runs, which need an allocation.
- The `[real]` tests use the ACE2-EAMv3 checkpoint and initial condition under
  `/global/cfs/cdirs/e3sm/anolan/ACE2-E3SMv3` and skip with a warning elsewhere.
- Multi-rank tests: every rank must make every collective. A mean computed
  inside `if (rank == 0)` hangs the job, with the root in `Allreduce` and the
  others in `Gatherv`.

## Not done yet

- The atmosphere cap is edited but not compiled; the first CIME case is its test.
  The cap should pass the driver's orbital parameters; they come from `atm_in`
  for now.
- No netCDF/SCORPIO `RestartStore` and no rpointer handling; restarts are tested
  in memory.
- The ocean and sea ice have components but no MCT caps or CIME files yet.
  The ocean's first prediction matches an independent Python run of the real
  checkpoint. (`emulator_comps/eocn/VERIFICATION.md` §1's table does not
  reproduce with the published files and is not used.)
- `EmulatorAtm` does not yet configure the exchange (`surface_from_ocean`,
  `publish_ocean_forcing`) from `atm_in`; the in-process SamudrACE test sets
  them on `AceAtmosphere` directly.
- Then a coupled run against the one-year reference in `SamudrACE-E3SMv3/ref1yr`.
- The C++ coupler API adapter waits for `emulators/coupler-infrastructure` to
  merge.
