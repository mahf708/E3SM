# EATM review

Review of `jonbob/add-eatm` at `23dd0c1b97` (39 commits off `master`, 5521 lines
added), plus the changes made on top of it in response.

EATM is a new atmosphere component that replaces EAM with a traced ACE-family
neural emulator called through FTorch. It runs serially on one rank, drives the
emulator on its own 6-hourly timestep, and hands the coupler a linear
interpolation between the two bracketing emulator states. It works: a 2-year
`GMPAS-EATM` run completed on `pm-gpu` at 5.58 SYPD, with the emulator itself
costing 1271 s out of 30967 s (MPAS-Ocean and MPAS-Seaice dominate).

The findings below are ordered by consequence. Everything marked **[fixed]** has
been addressed in this branch; everything marked **[open]** has not.

---

## Blockers for merging upstream

### 1. FTorch was a hard dependency of every E3SM build **[fixed]**

`components/cmake/find_dep_packages.cmake` called `find_package(FTorch REQUIRED)`
unconditionally, so every configuration on every machine — an ELM developer on a
laptop, the GitHub Actions runners, any machine without an FTorch install —
would fail at configure time. The `TODO (AN): Make optinal with USE_FTORCH` in
the source acknowledged it.

Now guarded on `COMP_ATM STREQUAL "eatm"`, which is available as a cache
variable at that point (`COMP_ATM:UNINITIALIZED=eatm` in the case's
`CMakeCache.txt`).

### 2. Absolute personal paths compiled into the source **[fixed]**

Two `character(len=*), parameter` paths pointed into individual users' scratch
and CFS directories:

| where | path |
|---|---|
| `ace_comp_mod.F90:66` | `/pscratch/sd/m/mahf708/test_ace_repo/test_trace_cuda.pt` |
| `eatm_comp_mod.F90:76` | `/global/cfs/cdirs/e3sm/anolan/ACE2-E3SMv3/initial_conditions/1971010100_time_1.nc` |

The first sits on `$PSCRATCH` and is subject to the NERSC purge, so the
component would stop building-and-running with no code change at all. Both are
now namelist variables (`eatm_model_file`, `eatm_ic_file`) with defaults in
`namelist_defaults_eatm.xml`, overridable from `user_nl_eatm`.

The defaults still point at `/global/cfs/cdirs/e3sm/anolan/ACE2-E3SMv3/`, which
is a personal directory — a blessed copy under `$DIN_LOC_ROOT` is still needed
before this goes upstream. That is flagged with a `TODO` in the defaults file.

### 3. The device was hardwired to CUDA **[fixed]**

`torch_model_load(ace_model, torchscript_file, torch_kCUDA)` meant a `pm-cpu`
build — which the machine file explicitly supports, with its own `FTorch_ROOT`
pointing at a CPU FTorch — would load a CUDA model and fail. Now
`eatm_model_device = 'cpu' | 'gpu'`.

### 4. Machine-wide side effects on `pm-cpu` **[open]**

Three changes in `cime_config/machines/` affect every `pm-cpu` user, not only
EATM builds:

- `config_machines.xml` loads `pytorch/2.8.0` in the **`gnu` module block**, so
  every `pm-cpu gnu` build now has a PyTorch module in its environment (it also
  brings its own Python, which can shadow the one CIME picks).
- `LD_LIBRARY_PATH` unconditionally prepends
  `/global/common/software/nersc9/pytorch/2.8.0/.../nvidia/nccl/lib` for all
  `pm-cpu` configurations.
- `gnu_pm-cpu.cmake` adds that same directory to `CMAKE_EXE_LINKER_FLAGS` for
  any component matching `cpl`.

None of these are gated on the atmosphere being EATM. Left alone because the
CPU EATM path presumably needs them and narrowing them is a judgement call about
that path — but they should be conditioned on the compset or on `COMP_ATM`
before this merges.

### 5. `FTorch_ROOT` for `gnu` points at a personal directory **[open]**

`config_machines.xml` sets `FTorch_ROOT` to
`/global/cfs/cdirs/e3sm/anolan/FTorch_v1.0.0-pytorch_v2.8.0` for `pm-cpu gnu`,
while the `gnugpu` entry correctly uses the shared
`/global/cfs/cdirs/e3sm/software/FTorch/FTorch_v1.1.0+libtorch_2.10.0+cuda_12.9`.
The CPU build should get a shared install too.

### 6. No test coverage **[open]**

There is no EATM entry in `cime_config/tests.py` and no test mod directory. Given
the restart machinery is non-trivial (two time levels, an interpolation
reconstructed from `curr_tod`), an `ERS` test is the one that matters — `SMS`
would not exercise it. This is worth adding even though it can only run on a
machine with FTorch, since the same is already true of the EAMxx CUDA suites.

Note that with `SamudrACE-E3SMv3` an `ERS` test cannot pass: the emulator is
stochastic (see finding 13). A deterministic emulator such as ACE2-EAMv3 is the
one to test restarts with.

---

## Correctness

### 7. `Sa_shum` was the saturation specific humidity **[fixed]**

```fortran
e = datm_shr_esat(tbot(i, j), tbot(i, j))
shum(i, j) = (0.622_R8 * e)/(pbot(i, j) - 0.378_R8 * e)
```

This is `q_sat(T_bot, p_bot)` — the lowest model layer is reported to the
coupler as exactly saturated, everywhere, at every timestep. `Sa_shum` is the
atmospheric humidity in the bulk formula that `flux_atmocn` and `flux_atmice`
use to compute evaporation and latent heat flux.

**How much this actually mattered** (measured, not predicted — an earlier draft
of this review overstated it). Comparing the 2-year `GMPAS-EATM-test4naser` run
against jonbob's `GMPAS-JRA1p5-2023-test` data-atmosphere baseline, both from
MPAS-Analysis at years 0001-0002:

| | global mean `evaporationFlux` | min |
|---|---|---|
| EATM (saturated `Sa_shum`) | -3.828e-05 kg/m2/s | -3.27e-04 |
| JRA data atmosphere | -3.977e-05 kg/m2/s | -1.36e-04 |

So the global mean is only **3.7% low**, not collapsed: the bulk formula's
sensitivity to `Sa_shum` is partly offset by the compensating errors in `Sa_z`
and the resulting stability and exchange coefficients. The peak evaporation is
2.4x the baseline's, so the spatial structure is worse than the mean suggests,
but this is a systematic near-surface moist bias rather than a catastrophic
one.

The fix is still unambiguously right — using the humidity the emulator predicts
instead of assuming saturation — but do not expect it to transform the run, and
do expect it to interact with the `Sa_z` change below rather than act alone.

The emulator predicts the field: `specific_total_water_7` (ACE2-EAMv3) /
`STW_7` (SamudrACE-E3SMv3) is the lowest-layer specific total water. That is now
what is exported. Using total water slightly overstates vapour where there is
condensate, which is a much smaller error than assuming saturation.

Scale of the input error, from the published initial conditions: the global
mean of `STW_7` is 8.3e-3 kg/kg (and `Qat2m` 9.1e-3), whereas `q_sat` at the
lowest layer's temperature and pressure is around 1.2e-2 — roughly 50% too
moist in the mean, and worse in dry subsidence regions where the real RH is low
and the evaporative demand is largest. That the flux error is far smaller than
the humidity error is what the table above records.

Set `eatm_legacy_surface = .true.` to get the old behaviour back.

### 8. The autoregressive loop was fed an interpolated state **[fixed]**

`ace_comp_run` called `ace_eatm_import()` at the top of an emulator step, and
`ace_eatm_import` read the emulator's previous state out of `net_outputs`. But
at that instant `net_outputs` still holds what the coupler was handed at the end
of the *previous coupler step*, which is the interpolation evaluated at
`t_frac = (21600 - dt_cpl)/21600`. With `ATM_NCPL=48` that is 0.9167, so every
6-hourly emulator step was fed a state short of its own prediction by ~8% of the
previous 6-hour tendency — an off-model state the network was never trained to
consume, applied 1460 times a year.

`ace_eatm_import` now takes the state explicitly and is called with
`eatm_intrp%t_ip1`, the emulator's actual prediction for the current time.

### 9. `Sa_z` was a pressure altitude above sea level **[fixed]**

```fortran
pbot(i, j) = (ak_7 + bk_7 * net_outputs(1, 1, i, j))
zbot(i, j) = 44307.694_R8 * ( 1.0_R8 - (pbot(i, j) / SHR_CONST_PSTD)**0.190284_R8 )
```

Two problems. First, `ak_7`/`bk_7` are the coefficients of the *interface* above
the lowest layer (the emulator's `ak`/`bk` have 9 entries for 8 layers, and
interface 8 is `ak=0, bk=1`, i.e. the surface). The layer-mean fields `T_7`,
`U_7`, `V_7`, `STW_7` belong at the layer midpoint, not at its top: over the
ocean the midpoint is ~450 m up, the interface ~940 m.

Second, `Sa_z` is defined as a height *above the surface*, and the standard
atmosphere pressure altitude is a height above mean sea level. Over the ocean
those coincide; over the Tibetan Plateau the formula returns ~5300 m where the
true depth of the layer above ground is a few hundred metres. `Sa_z` sets the
reference height in the surface-flux bulk formulae, so this inflates
`log(z/z0)` and depresses the exchange coefficients over all high topography.

Now the level is the layer midpoint and its height comes from the hypsometric
relation between `PS` and that midpoint:

```fortran
pbot = 0.5*(PS + (ak_bot + bk_bot*PS))
tv   = tbot*(1 + 0.608*shum)
zbot = (Rd*tv/g) * log(PS/pbot)
```

`Sa_pbot` and `Sa_dens` follow the same level, so they are now consistent with
the temperature and wind reported at it.

### 10. Precipitation phase was an all-or-nothing temperature threshold **[fixed where possible]**

`rainl`/`snowl` were assigned by `tbot < 273.15`, with `rainc`/`snowc` forced to
zero. Every grid point flips its entire precipitation between liquid and frozen
as the lowest-layer temperature crosses freezing, which makes the phase field
seen by MPAS-Seaice very noisy near the ice edge.

SamudrACE-E3SMv3 predicts `frozen_precipitation_rate` directly, so with that
emulator the split now comes from the model. ACE2-EAMv3 has no such channel and
still uses the threshold.

A units trap worth recording, because it would have been a 1000x error. The
checkpoint's `variable_metadata` declares **both** precipitation channels as
`m/s`, inherited from EAM's `PRECT`/`PRECS`. The data are actually `kg/m2/s`.
Global means in the published initial condition:

| channel | global mean | 3 mm/day would be |
|---|---|---|
| `surface_precipitation_rate` | 3.33e-5 | 3.5e-5 kg/m2/s, or 3.5e-8 m/s |
| `frozen_precipitation_rate` | 2.25e-6 | 7% of total by mass — plausible |

Read as m/s, the frozen rate would be 67x the *total* precipitation. The traced
model's moisture-budget corrector also compares the precipitation channel
directly against `LHFLX / L_v`, which is kg/m2/s. So `eatm_frzprec_units`
defaults to `kg/m2/s` and no conversion is applied; `'m/s'` remains available
in case a future checkpoint really does use it.

### 11. `Faxa_swnet` carried the downwelling flux **[fixed]**

`swnet` was set to `FSDS`. This is less serious than it looks: in the MCT driver
`Faxa_swnet` is explicitly a diagnostic (`seq_flds_mod.F90:955`), and the net
flux the ocean actually receives (`Foxx_swnet`) is rebuilt in `prep_ocn_mod`
from the four downwelling bands and the surface albedos. So the effect was
confined to the coupler's global energy budget diagnostics — but they were
wrong by the reflected fraction. Where the emulator predicts the upward flux
(`FSUS` / `surface_upward_shortwave_flux`), `swnet = FSDS - FSUS` is now used.
In the published initial condition the global means are `FSDS` 195 W/m2 and
`FSUS` 33 W/m2, so the diagnostic was ~33 W/m2 (17%) too high.

The commented-out latitude-dependent `avg_alb` fudge in the original has been
dropped along with the unused `yc` allocation that supported it.

### 12. Fixed shortwave band split **[open]**

`swvdr/swndr/swvdf/swndf` are fixed fractions of `FSDS` (0.28/0.31/0.24/0.17,
summing to 1). This matches what `datm` does, so it is a defensible convention
and the total downwelling flux is conserved. It does mean the visible:near-IR
ratio and the direct:diffuse ratio are constant in space and time. Over snow and
sea ice, where visible albedo (~0.8) and near-IR albedo (~0.4) differ sharply,
that is a real bias in absorbed shortwave, and it is exactly where the run's
sea-ice response lives. A zenith-angle- and cloud-dependent split would be
better; not attempted here.

### 13. The SamudrACE atmosphere is stochastic **[open — inherent]**

`NoiseConditionedSFNO` draws isotropic spherical noise inside `forward` on every
call (`fme/ace/registry/stochastic_sfno.py`). In `fme` that draw can be seeded
and carried across restarts through `RandomState`; through a traced TorchScript
graph called from FTorch it cannot — the traced graph keeps `aten::randn`, so
the model stays stochastic, but it draws from the global libtorch RNG, which
EATM has no way to seed or checkpoint.

Consequences:

- two identical EATM runs will diverge, and
- a restart will not reproduce a continuous run, so `ERS` cannot pass.

This is intrinsic to the model rather than a defect in EATM, and it is arguably
the point (the emulator is an ensemble emulator). It does need to be stated
explicitly wherever EATM results are compared.

**It can be seeded, though — with a small amount of work.** Checked rather than
assumed:

- The traced graph keeps `aten::randn` with no generator argument, so at replay
  it draws from libtorch's global default CUDA generator. That generator is
  seedable through `torch::manual_seed`.
- FTorch does not expose it: no seed symbol is exported from `libftorch.so` and
  nothing in its `.mod` files mentions one.
- It cannot be pushed into the graph either — `torch.manual_seed` is not
  TorchScript-scriptable, so a scripted wrapper around the traced core will not
  work.

The workable route is a C++ shim in `src/` calling `torch::manual_seed`, bound
from Fortran with `iso_c_binding`, driven by a namelist seed.
`gather_sources` in `components/cmake/cmake_util.cmake` already globs `*.cpp`,
so the file is picked up with no build change; the one addition needed is
propagating Torch's include directories and libraries to the atm target, since
`FTorch::ftorch` exports only its own module directory
(`FTorchConfig.cmake` does `find_dependency(Torch)`, so the `Torch` package is
already in scope where `find_package(FTorch)` runs).

Reseed on every emulator step from a seed derived from the **model date**
rather than a step counter, so a restart lands on the same draw without having
to checkpoint the RNG state.

Caveat worth stating up front: that pins the noise realization, which makes a
rerun follow the same trajectory and makes restart-versus-continuous differ
only by roundoff. It does not by itself guarantee bit-for-bit reproducibility,
which additionally needs deterministic GPU kernels.

Not implemented.

### 13a. The ACE tracing script had drifted away from `fme`, silently **[fixed in the driver]**

Two independent failures, both of which produce a *working* traced model that
is quietly missing physics. Found by tracing the SamudrACE atmosphere and
noticing `corrector_flags: {any_active: false}` in the metadata.

1. The script resolves the corrector configuration from `corrector._config`.
   Current `fme` correctors have no such attribute — the config lives at
   `step.config.corrector` — and a checkpoint with `corrector_disabled_epochs`
   (SamudrACE has one) wraps the corrector in an `EpochScheduledCorrector`
   besides. The lookup returns `None`, the script reports "no correctors
   configured" and traces the bare network: no dry-air conservation, no
   moisture-budget closure, and — because `force_positive_names` comes from the
   same dead attribute — no clamping of the 16 channels that must not go
   negative (all eight `STW_` levels, both precipitation rates, and all six
   radiative fluxes).

   This is not specific to SamudrACE: `corrector._config` is `None` for
   ACE2-EAMv3 too. The ACE2 model currently in use was traced in March against
   an older `fme` and does have its correctors (`any_active: true` in its
   metadata); **re-tracing it today would silently lose them.**

2. The script inlines its own copy of `ATMOSPHERE_FIELD_NAME_PREFIXES` "so the
   traced .pt has zero runtime dependency on fme". That copy predates the
   SamudrACE naming: it knows `specific_total_water_` but not `STW_`, and
   `tendency_of_total_water_path_due_to_advection` but not `DTENDTTW`. With the
   correctors correctly enabled but the channel lookup stale, the water-channel
   list comes back empty and the dry-air corrector dies inside
   `torch.jit.trace` with `stack expects a non-empty TensorList`.

`trace_eatm_model.py` fixes both without touching the tracing script: it
refreshes the prefix map from `fme.core.atmosphere_data`, resolves the
corrector config from `step.config.corrector`, rebuilds the force-positive
indices, and refuses to write a model whose correctors came out inactive or
whose corrector channels did not resolve.

The underlying fix belongs in the ACE repository's copy of the script.

### 14. The traced model drops the total-energy corrector **[open]**

The SamudrACE atmosphere configures
`total_energy_budget_correction: {method: constant_temperature,
constant_unaccounted_heating: 0.09}`. The ACE tracing script logs
`"Total energy budget correction is not yet implemented in the traced model.
Skipping."` and carries on. So the traced model EATM runs is not identical to
what `fme` inference would produce. `trace_eatm_model.py` now warns about this
at trace time rather than leaving it buried in the log.

### 15. Coupler fields that are never written **[partly fixed]**

`a2x` is zeroed once at init, and `atm_export_mct` writes 19 of the 53 fields
the driver defines. The rest stay at zero forever:

- `Sa_topo` — **[fixed]**, now exported as `PHIS/g`. It was zero, which matters
  for the `GPMPAS-EATM` compset where ELM does elevation-class downscaling
  against it. `index_a2x_Sa_topo` was also never looked up in
  `atm_cpl_indices_set`, so the assignment would have written to index 0 had the
  commented-out line been enabled.
- `Sa_wsresp`, `Sa_tau_est`, `Sa_ugust`, `Sa_uovern` — **[open]**, zero. These
  are optional in E3SM and their consumers handle absence, but `Sa_uovern` is
  looked up unconditionally.
- All 14 aerosol deposition fluxes (`Faxa_bcph*`, `Faxa_ocph*`, `Faxa_dst*`) —
  **[open]**, zero. There is no black carbon or dust deposition on snow and sea
  ice. For a 5-year run this is a real (if second-order) forcing omission that
  should be stated in any comparison against a full E3SM run.

### 15a. The coupler's merged surface temperature was weighted twice **[fixed]**

`ace_eatm_import` blended the surface temperature the coupler sends with the
emulator's own as

```fortran
net_inputs(TS) = (1 - lndfrac)*ts + lndfrac*ace_TS
```

But `Sx_t` is *already* fraction-weighted. `prep_atm_mod.F90:655-700` merges it
as `lfrac*Sl_t + ifrac*Si_t + ofrac*So_t`, and in `GMPAS-EATM` the land is a
stub, so the `lfrac` term contributes nothing:

```
ts = ofrac*So_t + ifrac*Si_t        (and exactly 0 over pure land)
```

Multiplying that by `(1 - lndfrac)` scales the ocean and ice contributions by
their fractions a *second* time. A cell that is half land and half ocean gets
`0.25*So_t + 0.5*ace_TS` instead of `0.5*So_t + 0.5*ace_TS` — a cold bias on
every coastline, worst where the ocean is warm relative to the emulator's land.

Completing the merge means *adding*, not re-weighting:

```fortran
net_inputs(TS) = ts + lndfrac*ace_TS      ! stub land
net_inputs(TS) = ts                        ! land model running: already complete
```

The branch is now selected on `lnd_present` from the coupler's infodata, so
`GPMPAS-EATM` (with ELM) takes the coupler's merged value unchanged.

Size of the correction, for a coastal cell that is half land over a 280 K
ocean, so the coupler sends `ts = 0.5*280 = 140`:

| | formula | result with `ace_TS` = 250 K |
|---|---|---|
| before | `(1-0.5)*140 + 0.5*250` | 195 K |
| after | `140 + 0.5*250` | 265 K |

70 K at half-land cells, tapering to zero at both pure land and pure ocean. It
is therefore a coastline-only correction, invisible in a global minimum or
maximum but large where it acts.

Measured: a 1-month rerun tracks the original bit-identically for the first
~9 coupler steps and then diverges, which is this correction propagating.

This is **not** what causes the cold pole below — see #15b.

### 15b. The emulator was fed 0 K over every land point, and no land at all **[fixed]**

The most serious defect found. It is the cause of the cold pool that shows up
in `atm.log` as `tbot` falling from 232.5 K to ~178 K.

The coupler computes the land fraction correctly and then fails to deliver it.
`prep_atm_mod.F90` chooses which field to ship as `Sf_lfrac`:

```fortran
if (samegrid_al) then
   klf = mct_aVect_indexRA(fractions_a,"lfrac")     ! the land fraction
else
   klf = mct_aVect_indexRA(fractions_a,"lfrin")     ! the land *input* fraction
endif
x2a_a%rAttr(index_x2a_Sf_lfrac,n) = fractions_a%Rattr(klf,n)
```

`SLND` makes CIME set `LND_GRID=null`, so the atmosphere and land grids differ,
so `samegrid_al` is false, so the driver ships `lfrin` — which only a real land
model ever populates and with a stub stays identically zero. Measured in the
day-31 coupler history of an actual run:

```
fraca_lfrac    min=0.0000 max=1.0000 mean=0.3426    seq_frac computes it correctly
fraca_lfrin    min=0.0000 max=0.0000 mean=0.0000    never populated: no land model
x2a_Sf_lfrac   min=0.0000 max=0.0000 mean=0.0000    what the atmosphere receives
fraca sum = 1.0000 exactly       x2a sum = 0.6574
```

`ofrac` and `ifrac` arrive intact (0.4996 and 0.1578 in both bundles); only
`lfrac` is lost. Since `prep_atm_mod` merges the surface temperature as
`lfrac*Sl_t + ifrac*Si_t + ofrac*So_t`, and over land `ofrac` and `ifrac` are
also zero, the coupler's `Sx_t` is exactly **0 K** over 26.1% of global area.

This is arguably a driver defect — `prep_atm_mod` should fall back to `lfrac`
when no land model is present — and it would affect any active atmosphere run
with a stub land, not just EATM. `datm` does not notice because it prescribes
its own state and never consumes `Sf_lfrac`. EATM notices because it feeds both
`LANDFRAC` and `TS` straight into a neural network.

EATM passed both straight through:

- `net_inputs(LANDFRAC) = lndfrac` → the emulator was told the planet has **no
  land anywhere**, when `LANDFRAC` is a time-invariant boundary condition it
  was trained with.
- `net_inputs(TS)` → **0 K** over every land point. Both the original blend
  `(1-lndfrac)*ts + lndfrac*ace_TS` and the #15a form `ts + lndfrac*ace_TS`
  reduce to `0` when `lndfrac` is zero, so the "use the emulator's own TS over
  land" logic never executed. It was dead code in this compset.

The result, at day 31, splitting the grid by whether the coupler covered it:

| | mean `T_7` |
|---|---|
| cells where the fractions sum to ~1 | 274.2 K |
| cells where the fractions sum to ~0 | **197.1 K** |

12.0% of global area below 200 K, 23.0% below 210 K, spanning both hemispheres.
`FLDS` and the lowest-layer humidity clamp to exactly zero in that region — the
traced model's force-positive corrector firing, i.e. the network predicting
negative downwelling longwave and negative water, the signature of being
evaluated far outside its training manifold. The initial condition's *global*
minimum `T_7` is 233.15 K, so this is a 55 K excursion below anything in the
initial state, at a point receiving 530 W/m2 of insolation (87.25S, 104E, 3086 m
elevation, sunlit in January).

The fix keys off the fraction deficit rather than `lndfrac`, which is correct
with or without a land model:

```fortran
covered = ofrac + ifrac + lfrac
deficit = 1 - covered
LANDFRAC = lfrac + deficit
TS       = ts + deficit * ace_TS
```

With a land model running, `lfrin` is populated, `Sf_lfrac` arrives intact, the
fractions sum to one, the deficit is zero, and `Sx_t` passes through unchanged.

Verified against ground truth from the same run: reconstructing the land
fraction as `lfrac + (1 - ofrac - ifrac - lfrac)` reproduces
`fractions_a(lfrac)` with a mean absolute error of 5.3e-07, with only 77 of
64800 cells differing by more than 1e-6 — and those by ~1e-3, which is exactly
`eps_fraclim`, the limit below which `seq_frac_mct` snaps a small land fraction
to zero. So the reconstruction is exact to within the driver's own rounding.

`ace_eatm_import` now also logs the coupler fractions and the `LANDFRAC` and
`TS` actually handed to the emulator, so this class of defect is visible in
`atm.log` rather than needing a coupler-history analysis to find.

**Not yet re-run.** The fix is reasoned and the diagnosis is measured, but no
simulation has been done with it. That is the next thing to do.

### 16. `Sa_pslv` is the surface pressure, not sea-level pressure **[open]**

`pslv = PS`. Over the ocean, where `PHIS = 0`, these are identical, so nothing
downstream in this compset is affected. It would be wrong over land if a land
model started using it.

---

## Structure and maintainability

### 17. Channel indices were magic numbers **[fixed]**

`ace_eatm_import` was 40 lines of `net_inputs(1, 24, i, j) = net_outputs(1, 19, i, j)`
with the channel name in a trailing comment, `ace_eatm_export` addressed
outputs as `net_outputs(1, 41, ...)`, and `eatm_restart_mod` had two hand-written
`if/elseif` ladders over `do c = 1, 44` and `do c = 1, 39` reconstructing channel
names from arithmetic on the loop index. Swapping in a checkpoint with a
different channel layout meant editing all of it consistently, with a silent
field mix-up as the failure mode.

Replaced by `eatm_channels_mod.F90`, which holds one table per emulator (channel
names, sizes, resolved named indices, and a prognostic-feedback map built by
matching input names against output names). Adding an emulator is one
`set_table_*` routine plus one branch in `eatm_channels_init`; nothing else
changes. `trace_eatm_model.py` cross-checks a checkpoint's actual channel list
against the same table before writing the traced model out.

### 18. Parameterized derived type **[fixed]**

`t_eatm_interpolator(kind)` was a Fortran 2003 PDT. PDT support is uneven across
compilers E3SM has to build with (Intel and NVHPC in particular have a long
history of PDT bugs). Replaced with a plain type — the kind was only ever
instantiated as `R4`.

### 19. `buildnml` wrote RTM namelist variables **[fixed]**

For `RUN_TYPE=hybrid` or `branch`, `buildnml` appended `finidat_rtm = ...` /
`nrevsn_rtm = ...` to the namelist infile. Those are MOSART variables; EATM's
namelist definition has no such entries, so `build-namelist`'s `validate` would
have rejected them and any branch or hybrid start would have failed at setup.
EATM finds its restart through `rpointer.atm`, so no namelist entry is needed —
the staging check is kept, the bogus variables are gone.

### 20. Dead code **[fixed]**

- `eatm_comp_run` declared `tbot`, `pbot`, `swndr`, `swndf`, `swvdr`, `swvdf`
  and eight more unused locals that **shadowed the module arrays of the same
  name** from `eatmMod`. Nothing wrote to them, but anyone adding a line to that
  routine would have been writing to a local scalar instead of the export array.
- `normalize`/`denormalize` and their types in `eatmMod`, plus
  `init_normalizer`/`finalize_normalizer` in `ace_comp_mod`, were unreachable —
  every call site was commented out, because the traced model does its own
  normalization. Removed, with a pointer to the git history.
- `build-namelist` computed the coupling interval into `$val` and never used it.
  It now computes it and checks that it divides the 6-hour emulator timestep,
  which is a real precondition: if it does not, `mod(CurrentTOD, eatm_model_dt)`
  never hits zero and the emulator silently never advances.
- `user_nl_eatm`'s header was copied verbatim from RTM (`ROF_NCPL`, `ROF GRID`,
  `do_rtm`). Rewritten.
- Unused locals in `atm_cpl_indices_set` (`tot_mon_in_year`, `imon`, `ier`,
  `monstr`) and the unused `yc`/`klat` lookup in `ace_eatm_export`.

### 21. `eatmIO.F90` is a 2075-line copy of `ncdio_pio.F90` **[open]**

Copied from MOSART (the comments still say "called from rtm_comp"). Its `ncd_io`
generic is the reason the restart code works on `real(R4)` arrays, so it is not
gratuitous, but it is 2000 lines of duplicated I/O infrastructure that will drift
from the original. Sharing one copy under `share/` would be better.

### 22. Serial-only, and more than the `npes>1` abort stands in the way **[open]**

`eatmSpmdInit` aborts if `npes > 1`. Removing that abort is not sufficient:
`atm_read_eatm` sets `lsize = gsize` unconditionally, so every rank would claim
the whole grid, while `atm_SetGSMap_mct` builds a genuine `npes`-way
decomposition — and restart/IC I/O is `masterproc`-only. The serial design is
fine for now (the emulator is 4% of runtime) but the inconsistency should be
noted rather than discovered later.

### 23. Import sign conventions are inconsistent, and mostly unused **[open]**

`atm_import_mct` negates `Faxx_sen`, `Faxx_evap`, `Faxx_taux`, `Faxx_tauy` and
`Faxx_lwup`, but not `Faxx_lat`. Of the 21 imported arrays only `ts`, `icefrac`,
`ocnfrac` and `lndfrac` are read by anything. Left alone because the SamudrACE
*ocean* component consumes `LHFLX`, `SHFLX`, `TAUX` and `TAUY`, so these will
matter if the ocean emulator is ever wired in — but the sign conventions need
settling before that.

### 24. `gauss180x360` domain files do not exist **[open, currently harmless]**

`config_grids.xml` declares

```xml
<file grid="atm|lnd" mask="IcoswISC30E3r5">$DIN_LOC_ROOT/share/domains/domain.lnd.gauss180x360_IcoswISC30E3r5.20260128.nc</file>
<file grid="ice|ocn" mask="IcoswISC30E3r5">$DIN_LOC_ROOT/share/domains/domain.ocn.gauss180x360_IcoswISC30E3r5.20260128.nc</file>
```

Neither file is in `/global/cfs/cdirs/e3sm/inputdata/share/domains/`. It does not
break anything today: EATM builds its own domain from the SCRIP mesh
(`filename_eatm`), and `ATM_DOMAIN_FILE` never reaches
`cpl.input_data_list`, so `check_input_data` never looks for it. It will bite the
moment anything starts reading `ATM_DOMAIN_FILE`, and the mapping files that
*are* checked all exist, so this is an easy thing to leave broken unnoticed.

The `gauss180x360_r05_IcoswISC30E3r5` grid alias is in the same position, and
its `map_gauss180x360_to_r05_*` / `map_r05_to_gauss180x360_*` mapping files are
also absent from inputdata — so the `GPMPAS-EATM` (ELM + MOSART) compset cannot
currently run.

---

## Reference runs

What to compare against, and where it is. Everything below predates the changes
on this branch, so it is the "before" side of any A/B.

### Completed runs with output still on disk

| run | what | where |
|---|---|---|
| `GMPAS-EATM-test4naser` | 2 years, June 2026, the branch as reviewed. **The direct baseline.** 363 GB intact: 486 MPAS history files, 73 `cpl.hi.*` (10-daily) | `/pscratch/sd/a/anolan/e3sm_scratch/pm-gpu/GMPAS-EATM-test4naser/run` |
| `GMPAS-EATM-gnugpu` | 2 years, June 2026, earlier EATM code. 363 GB intact, same layout | `/pscratch/sd/a/anolan/e3sm_scratch/pm-gpu/GMPAS-EATM-gnugpu/run` |
| `GMPAS-JRA` | 1-day test only, rundir purged — not usable | — |

The `cpl.hi.*` files are the useful ones for reviewing *this* component: they
carry the `a2x` fields (`Sa_shum`, `Sa_z`, `Sa_pbot`, `Sa_dens`, `Faxa_*`)
exactly as EATM produced them. A case configured with
`HIST_OPTION=ndays, HIST_N=10` and `RUN_STARTDATE=0001-01-01` lines up
file-for-file with them, which makes a field-by-field diff of the fixes
straightforward.

### Published MPAS-Analysis

`/global/cfs/cdirs/e3sm/www/anolan/` has a run per fix, which doubles as a
history of what has already been tried:

```
GMPAS-EATM.IcoswISC30E3r5                     baseline (Apr)
GMPAS-EATM.IcoswISC30E3r5.nextsw_cday         nextsw_cday fix
GMPAS-EATM.IcoswISC30E3r5.time-varying-SOLIN  time-varying SOLIN
GMPAS-EATM.IcoswISC30E3r5.cpl-solin-test      coupler SOLIN test
GMPAS-EATM.IcoswISC30E3r5.pressure-altitude   the zbot pressure-altitude commit
GMPAS-EATM.IcoswISC30E3r5.GMPAS-EATM-gnugpu   Jun
GMPAS-EATM.IcoswISC30E3r5.GMPAS-EATM-test4naser  Jun, current branch
GMPAS-JRA1p5-2023                             JRA control
```

and `/global/cfs/cdirs/e3sm/www/jonbob/GMPAS-JRA1p5-2023-test/` is the
data-model baseline for how the same ocean and sea ice behave with a real
forcing dataset — the right yardstick for "is the emulated atmosphere driving a
sane ocean", as opposed to EATM-vs-EATM.

The configs to reproduce the analysis on a new run are in
`/global/cfs/cdirs/e3sm/anolan/GMPAS-EATM-Analysis/` (`GMPAS-EATM.pmgpu.cfg`,
`GMPAS-JRA1p5-2023.pmcpu.cfg`).

### Measured effect of the fixes

`compare_cpl_hi.py` on the day-11 file, `GMPAS-EATM-test4naser` (original code)
against this branch, both `ACE2-EAMv3` from `0001-01-01`, area-weighted global
means:

| field | original | this branch | change |
|---|---|---|---|
| `a2x_Sa_z` | 1165.5 m | 407.5 m | **-65%** (max 5575 -> 475 m) |
| `a2x_Sa_topo` | 0 | 231.5 m | now exported (max 5087 m) |
| `a2x_Sa_shum` | 7.731e-03 | 6.240e-03 | **-19.3%** (max 2.51e-2 -> 2.04e-2) |
| `a2x_Sa_pbot` | 88224 Pa | 93345 Pa | +5.8% (layer top -> midpoint) |
| `a2x_Sa_dens` | 1.2024 | 1.2743 | +6.0% |
| `a2x_Sa_ptem` | 267.92 K | 263.34 K | -1.7% |
| `a2x_Faxa_swnet` | 170.11 W/m2 | 146.37 W/m2 | **-14.0%** (= global surface albedo) |
| `a2x_Sa_tbot` | 259.63 K | 259.35 K | -0.1% (min 178.45 vs 178.44) |
| `a2x_Faxa_lwdn` | 253.05 | 251.41 | -0.6% |
| `a2x_Faxa_swvdr` | 47.63 | 47.52 | -0.2% |
| `a2x_Faxa_rainl` | 4.653e-05 | 4.665e-05 | +0.3% |
| `a2x_Faxa_snowl` | 2.706e-06 | 2.715e-06 | +0.3% |

The last four are untouched by any fix; their differences are the two
trajectories diverging, which sets the noise floor for reading this table. That
floor grows: under 0.6% at day 11, but `lwdn` -1.1%, `rainl` -0.6% and `snowl`
+1.8% by day 21. Anything at the 1-2% level in a day-21 or later comparison is
not distinguishable from chaos.

The fixes themselves are stable rather than drifting, which is how you tell
them apart from divergence — day 11 vs day 21: `Sa_z` -65.0% / -65.0%,
`Sa_pbot` +5.8% / +5.8%, `Sa_dens` +6.0% / +6.0%, `Faxa_swnet` -14.0% / -13.8%,
`Sa_shum` -19.3% / -20.7%. `Sa_topo` is bit-identical between the two dates, as
a time-invariant boundary field should be.
Ignore `a2x_Sa_u` and `a2x_Sa_v`: their global means are near zero
(-0.157 -> -0.089 m/s), so the percentage change is division noise while the
extremes are unchanged.

`Sa_z` is the largest effect by far. A 65% reduction in the mean and a 12x
reduction in the maximum means the surface-flux reference height was wrong
everywhere there is topography, and every exchange coefficient derived from it
with it.

### What each fix should show up as

| fix | expected signature |
|---|---|
| `Sa_shum` from predicted total water | `Sa_shum` down (measured: -19.3%); `evaporationFlux` toward the JRA baseline (it starts only 3.7% low in the global mean, so look at the spatial structure and the 2.4x-too-large extremes, not the mean) |
| `Sa_z` at the layer midpoint | `Sa_z` roughly halves over ocean, drops by the surface elevation over land (measured: -65% mean, -91% max); changes exchange coefficients, so it interacts with the humidity fix |
| `Faxa_swnet` net rather than downwelling | coupler energy budget diagnostics only (measured: -14%); no change to ocean or ice forcing |
| frozen precipitation from the model | `Faxa_snowl` structure smooths near the ice edge (SamudrACE only; ACE2-EAMv3 has no such channel, hence the +0.3% noise above) |
| `Sa_topo` exported | nonzero over land (measured: 231.5 m mean); only matters with ELM |
| coupler `Sx_t` completed not re-weighted | coastal cells warm toward the ocean temperature; does **not** move the cold pole (#15b) |

## Verified, not a defect

A few things that look wrong and are not:

- **SOLIN is computed for `T + 6 h`, not `T`.** `ace_compute_solin` advances the
  Julian day by one emulator step before calling `shr_orb_decl`. This is
  correct: both checkpoints declare `next_step_forcing_names: ['SOLIN']`, which
  in `fme` means the SOLIN *input* channel carries the next step's value. The
  state exported at coupler time `T` was produced at `T - 6 h` using `SOLIN(T)`,
  so insolation and state are consistent at the time the coupler sees them.

- **The extra `SOLIN_next_step` channel is redundant.** The tracing script
  appends the declared next-step forcing channels after the state block, and the
  traced graph slices them off with `inputs[:, n_in:]`. For a model traced
  without `--add-ocean` nothing consumes them, which is why passing only
  `n_in` channels works. `eatm_pass_forcing` now makes this explicit.

- **Restarts write both time levels.** `t_im1` and `t_ip1` both go to the
  restart and the interpolation is rebuilt from `curr_tod` at init, so for a
  deterministic emulator the restart reproduces the interpolated state exactly.
  `PHIS` and the last `SOLIN` are persisted too; the surface fractions are not,
  but they are re-imported from the coupler before they are next read.

- **Two inferences run at startup.** `atm_init_mct` is called twice by the
  driver; the second call runs `eatm_comp_run` at `TOD = 0`, which is an
  emulator-step boundary, so the emulator advances once during initialization on
  top of the inference in `ace_comp_init`. The net effect is that the run starts
  one emulator step into the trajectory rather than at the initial condition —
  a 6-hour offset, self-consistent thereafter. Not worth changing, but worth
  knowing when comparing against a reference `fme` rollout, which starts at
  step 0.

---

## What changed on this branch

New files:

| file | purpose |
|---|---|
| `src/eatm_channels_mod.F90` | per-emulator channel tables, named indices, prognostic feedback map |
| `tools/trace_eatm_model.py` | trace a checkpoint for EATM; `check_trace=False` for stochastic models; validates the channel layout |
| `tools/make_eatm_ic.py` | build an EATM initial condition from an ACE initial-condition dataset |
| `tools/trace_samudrace_atmosphere.sh` | end-to-end SamudrACE-E3SMv3 atmosphere preparation |
| `tools/run_gmpas_eatm_pm-gpu.sh` | create/build/submit a GMPAS-EATM case on `pm-gpu` |

New namelist variables in `eatm_inparm`: `eatm_emulator`, `eatm_model_file`,
`eatm_ic_file`, `eatm_model_device`, `eatm_pass_forcing`, `eatm_legacy_surface`,
`eatm_frzprec_units`, `eatm_iradsw`. New xml variable `EATM_EMULATOR`.

Answer-changing by default (all revertible with `eatm_legacy_surface = .true.`,
except the autoregressive-input fix which is unconditional):

1. `Sa_shum` from the predicted lowest-layer total water instead of saturation
2. `Sa_z` / `Sa_pbot` / `Sa_dens` at the layer midpoint, height above surface
3. precipitation phase from the predicted frozen rate where available
4. `Faxa_swnet` net rather than downwelling
5. `Sa_topo` exported instead of left at zero
6. the emulator is fed its own prediction rather than an interpolated state
7. the coupler's merged surface temperature is completed rather than re-weighted

Validated on `pm-gpu`, `GMPAS-EATM` at `gauss180x360_IcoswISC30E3r5`:

- Builds clean; the channel table resolves 39 in / 44 out / 33 prognostic
  feedbacks for `ACE2-EAMv3`, and the traced model loads on the GPU.
- Exports are physically sensible: `zbot` 319-485 m (a height above the
  surface, where the pressure-altitude formula gave ~940 m over ocean and
  >5000 m over plateaus), `shum` down to 1.9e-5 kg/kg (a saturation value
  cannot get near that), `pbot` 51-99 kPa, no NaNs over a month.
- A 1-month run reproduces the fixes against the original branch at three
  independent dates -- see "Measured effect of the fixes".
- The restart **write** path is confirmed: 46 variables (44 output channels
  plus `PHIS` and `SOLIN`), `time=2` for the two emulator states, `PHIS` and
  `SOLIN` correctly without a time dimension, names identical to the original
  code so existing restarts stay readable, and `PHIS/g` matching the exported
  `Sa_topo` range exactly as an independent cross-check.
- The restart **read** path is **not** yet tested. It needs a
  `CONTINUE_RUN=TRUE` continuation, and it is what a production run with
  `RESUBMIT` depends on at every segment boundary.

### Timing, measured at 8 nodes

`NTASKS=-7` plus a node for the serial atm; 5.17 SYPD.

| | |
|---|---|
| init | 134 s |
| integration | 45.2-45.8 s per model day (OCN 31.7, ICE 10.6, CPL 3.1, ATM 1.8) |
| end-of-run restart write | **>= 235 s** and not complete when the job was killed |

The restart write is not negligible, which is easy to miss: a 1-month debug run
budgeted at 31 x 45.8 s + init fits inside 1800 s on paper and still hit the
wall. A 1-year segment carries twelve monthly restarts on top of ~4.7 h of
integration, so budget ~5.5 h against the 8 h wallclock, or set
`REST_OPTION=nyears` to trade recovery points for margin.

Beware taking a rate from a very short run: the 3-day run reported 45.8 s/day
but averaging over its startup ramp gives 54 s/day, and neither is the whole
story once restart writes are included.

---

## Session of 2026-08-15 (evening): measured results

Everything below is measured on `pm-gpu`, `GMPAS-EATM` at
`gauss180x360_IcoswISC30E3r5`, 8 nodes, starting `0001-01-01`.

### 25. The land-fraction fix, validated **[fixed, confirmed]**

Finding #15b was reasoned but unrun when it was written. It has now been run.
`eatm-buildcheck-08151753`, 20 days, identical namelist and checkpoint either
side of commit `67d71a2a09` (the pre-fix month is preserved in
`run/baseline_57052155/`):

| | before | after | expected |
|---|---|---|---|
| area below 240 K | **30.2 %** | 0.02 % | ~0 |
| area below 200 K | **12.0 %** | 0.00 % | 0 |
| area with `Faxa_lwdn == 0` | **28.5 %** | 0.00 % | 0 |
| `Sa_tbot` global mean | 259.4 K | **282.3 K** | 280-288 K |
| `Sa_tbot` min | 178.4 K | 238.6 K | ~230 K |
| `Sa_shum` global mean | 6.24e-3 | 8.37e-3 | 6-8e-3 |
| `Sa_lwdn` global mean | 251 W/m2 | 331 W/m2 | ~340 |

Stable, not drifting: day 11 vs day 21 gives `Sa_tbot` 282.29 / 282.31 K and
`Sa_shum` 8.368e-3 / 8.368e-3.

It also repaired the general circulation, which nothing predicted. Ocean-only
zonal-mean `Sa_u`, day 21:

| | SH jet | 50S | 60S | equator | 20S | NH jet |
|---|---|---|---|---|---|---|
| before | 5.8 @ 39S | +4.6 | **-3.1** | **+1.8** | -0.8 | 9.4 @ 54N |
| after | 14.2 @ **52S** | +14.0 | **+9.5** | **-4.9** | -6.9 | 7.1 @ 38N |
| expected | 8-10 @ 45-55S | ~+8 | ~+6 | -3 to -5 | -5 to -6 | 8-12 @ 40-50N |

Collapsed Southern Ocean westerlies, spurious 60S easterlies, wrong-sign
equatorial winds and 4x-weak SH trades all fixed. The SH jet is now somewhat
too strong and the NH jet slightly weak and equatorward; those are the
remaining wind biases.

The coupler-side symptom is unchanged and that is correct: `x2a_Sf_lfrac` is
still 0.0 globally and `x2a_Sx_t` still exactly 0 K over 25.93 % of area, in
both runs. EATM reconstructs the deficit on import rather than repairing what
the driver ships. Fixing the coupler's own fields is a `prep_atm_mod` change
and would matter for `GPMPAS-EATM` with ELM, or for anyone reading `cpl.hi`
naively.

Throughput improved too, 48.01 -> 45.69 s/model-day, and the pre-fix run's
+6.4 %/month secular slowdown disappeared: it was MPAS-SI working harder as the
cold pool grew ice, not a cost.

### 26. `cpl.hi` is instantaneous, not a time mean **[not a defect -- a trap]**

`seq_hist_write` writes `.cpl.hi.` (`driver-mct/main/seq_hist_mod.F90:231`);
the time-averaged file is `.cpl.ha.` from `seq_hist_writeavg` (`:839`), written
only when `AVGHIST_OPTION` is set, which it is not by default.

So every `cpl.hi` field is a snapshot at the file's timestamp -- and with
`HIST_OPTION=ndays`, always at 00:00 UTC, where the sub-solar longitude is
180 E. An analysis that assumes these are means concludes the model has no
diurnal cycle and a frozen sun at 180 E. It does not; three snapshots were
taken at the same phase.

Consequences for anyone using these files:

- Maxima are instantaneous. `Faxa_swnet` reaching 1039 W/m2 is fine for a
  snapshot and impossible for a 10-day mean.
- Ocean-masked global means of shortwave are biased, because the ocean is not
  distributed uniformly in longitude. **Do not build a surface energy budget
  out of `cpl.hi`.** An earlier attempt in this session gave +37 W/m2 for the
  ocean when the ocean's own heat content said -33 W/m2 -- wrong sign.
- `a2x_Faxa_swnet` is doubly wrong for that purpose: the coupler discards it
  and rebuilds the ocean's net shortwave from the four bands and the ocean's
  own albedo (`prep_ocn_mod`).

Set `AVGHIST_OPTION` / `AVGHIST_N` if you want means. Use MPAS's own
`globalStats` for anything budget-like.

### 27. The authoritative energy metric, and what the two emulators score

Global net surface heat flux from the ocean's volume-mean temperature drift
(`mpaso.hist.am.globalStats`, `temperatureAvg`), rho=1026, cp=3996,
V=1.3307e18 m3, A=3.6195e14 m2:

| configuration | days | net surface heat flux |
|---|---|---|
| pre-land-fix (31 d) | 30 | **-160 W/m2** |
| ACE2-EAMv3, post-land-fix | 19 | **-33.3 W/m2** |
| SamudrACE-E3SMv3, `lowest_level` | 19 | **-74.4 W/m2** |
| SamudrACE-E3SMv3, `near_surface` | 16 (running) | **-61.6 W/m2** |

A usable 5-year coupled run wants this inside about +/-10 W/m2. Nothing is
there yet.

Why SamudrACE is worse than ACE2 (day 21, ocean-only area-weighted):

| | ACE2-EAMv3 | SamudrACE | target |
|---|---|---|---|
| `Sa_tbot` | 285.42 K | 283.75 K | ~288 |
| `Sa_shum` | 9.57e-3 | 8.35e-3 | ~9e-3 |
| tropical `Sx_t - Sa_ptem` | 0.78 K | 2.30 K | ~1.0 |
| `Faxx_sen` | -9.70 | -22.99 | ~-11 |
| `Faxx_lat` | -129.3 | -159.5 | ~-105 |

ACE2's disequilibrium and sensible heat are already about right; its residual
is almost entirely latent, i.e. a humidity problem, and `Sa_shum` being
specific *total* water (19.3 % of ocean cells supersaturated at the bottom
level, RH to 2.82) is the thread to pull there. SamudrACE's is temperature.
**The two emulators fail differently and want different fixes.**

### 28. `eatm_surface_layer`: right change, wrong diagnosis **[fixed; does not fix the imbalance]**

Commit `7470599244` exports the state at 10 m from SamudrACE's predicted
`Tat2m`/`Qat2m`/`Uat10m`/`Vat10m` with `pbot = pslv`, matching what `datm`
hands this same ocean under JRA (`datm_comp_mod.F90:1029`, `:1031`, the
`IAF_JRA_1p5` datamode). ACE2-EAMv3 has no near-surface channels and falls
back to `lowest_level` automatically.

It does what it says: `Sa_z` 449 -> 10.0 m, `Sa_tbot` 284.00 -> 287.55 K,
`Sa_shum` 8.54e-3 -> 9.43e-3, `|V|` 9.59 -> 7.38 m/s -- every one moving to
its target value.

**But the premise was wrong.** Measured at day 11, tropical open ocean:

| | SST | `Sa_tbot` | `Sa_ptem` | SST - ptem |
|---|---|---|---|---|
| `lowest_level` | 299.300 | 292.399 (450 m) | **296.926** | 2.37 K |
| `near_surface` | 299.438 | 296.687 (2 m) | **296.687** | 2.75 K |

The emulator's own `Tat2m` is 296.687 K; the dry-adiabatic reduction of its
450 m temperature gives 296.926 K. **They agree to 0.24 K.** The 450 m
sampling was never injecting a large temperature error -- an earlier estimate
of ~1.5 K of spurious cold, repeated several times in this session, is wrong
by a factor of six.

The real bias is that **SST - T_2m is 2.75 K where it should be ~1.0 K**:
SamudrACE's near-surface air is genuinely ~1.75 K too cold for the sea surface
it is given. No reference-height work touches that.

This also explains why sensible heat got *worse*, -22.07 -> -29.78 W/m2, while
wind fell and the disequilibrium barely moved. The exchange coefficient goes
roughly as 1/ln(z/z0)^2: `ln(10/1e-4) = 11.5` against `ln(450/1e-4) = 15.3`,
so it nearly doubles at 10 m. The old configuration had two compensating
errors -- too much wind against too small a coefficient -- and removing the
geometry error exposed the underlying cold bias at full strength. Net still
improves ~7-12 W/m2 because latent gains more than sensible loses.

Keep the change: it is the physically correct state to hand a similarity-theory
scheme, and it is revertible with `eatm_surface_layer = 'lowest_level'`. But it
is not the fix for the energy imbalance.

### 29. The total energy budget corrector is the leading remaining suspect **[open]**

Checked rather than assumed, from the traced metadata and
`trace_eatm_model.py:222`, which sets the flag from
`config.total_energy_budget_correction is not None`:

| corrector | ACE2-EAMv3 | SamudrACE-E3SMv3 | in the traced graph |
|---|---|---|---|
| `conserve_dry_air` | true | true | yes |
| `moisture_budget_correction` | advection_and_precipitation | same | yes |
| `zero_global_mean_moisture_advection` | false | false | not configured |
| `force_positive` | active | active, 16 channels | yes |
| `total_energy_budget_correction` | **not configured** | **configured** | **no** |

So for ACE2 every configured corrector is active and there is nothing to
enable. For SamudrACE one is configured and dropped.

Why it is skipped: `trace.py:508-512` in the ACE tracing script logs
"not yet implemented in the traced model. Skipping" and returns. It is a TODO,
not a limitation. The reason it was harder than the two that *were*
implemented is that it is the only one taking `forcing_data` --
`atmosphere.py:439-442` wants `DSWRFtoa` (TOA down SW) and `HGTsfc` (surface
height), and the traced graph's signature is a single input tensor. **Both are
already in EATM's input block**: `SOLIN` is an input channel and `HGTsfc` is
`PHIS/g`. Everything else it needs -- the vertical coordinate, an
area-weighted global mean -- is already used by the correctors that were
implemented.

What it does: desired global-mean energy path = input path +
(net energy flux into atmosphere + `constant_unaccounted_heating`) * dt, with
the shortfall applied as a **spatially uniform temperature increment** to every
vertical level. That is an atmospheric closure, not a surface-flux one, so it
would not directly fix the coupler's bulk formula. But a systematic
near-surface cold bias is exactly the signature of an atmosphere whose energy
budget is not being closed, and the correlation is now suggestive: the emulator
that configures it and does not get it runs at twice the imbalance of the one
that never configured it.

### 30. `force_positive` is per-channel, not relational **[open, benign]**

The emulator clamps each flux non-negative -- 16 channels including `FSDS`,
`FSUS`, `FLDS`, `FLUS`, `STW_0..7` -- and it works: EATM's own clamp counters
read `shum=0 precip=0 snow=0 fsds=0` on essentially every step. Nothing
enforces `FSUS <= FSDS`, so the emulator's implied surface albedo can exceed 1.

Measured at day 11 rather than inferred from the counts:

| | SamudrACE | ACE2-EAMv3 |
|---|---|---|
| cells with `swnet` floored | 47.5 % | 45.3 % |
| of those, FSDS > 1 W/m2 | 1692 (3.1 % of area) | 1166 (2.1 %) |
| of those, FSDS > 20 W/m2 | **17 (0.02 %)** | 14 (0.02 %) |
| daytime cells floored | 17 of 31444 (**0.1 %**) | 14 of 31586 (0.0 %) |

Confined to the night side and terminator where FSDS is under 1 W/m2, and the
same in both emulators. Benign.

### 31. Exact restart: EATM passes, the coupled system does not **[EATM fixed; system open]**

Hand-rolled ERS in `eatm-ers-4n` (4 nodes, ACE2-EAMv3 because it is
deterministic), three phases in one case: A startup 2 days -> restart at
`0001-01-03`; B `CONTINUE_RUN` 2 days -> `0001-01-05`; C startup 4 days
straight. Driven with `./case.submit --no-batch` inside an interactive
allocation; output stashed per phase under `run/ers/`.

**Control passes exactly.** A vs C over days 2 and 3: **365 of 365** coupler
fields bit-identical. The GPU inference is reproducible; nothing here is
non-deterministic.

**EATM's restart is exact**, three ways:

- its restart file at `0001-01-03` is bit-identical between the run that wrote
  it at end-of-run and the run that wrote it mid-run -- all 46 variables;
- its first export after restart matches the continuous run at the same model
  time to the last digit, all ten logged fields;
- step phasing is right: B's first advance is step 108 at `tod 21600`,
  matching C's step 108. It correctly does *not* re-run the emulator at
  `tod 0`, and `nextsw_cday` follows EAM's exact-restart pattern
  (`atm_comp_mct.F90:201-204` skips `curr_cday` on restart, `:245` recomputes
  `getNextRadCDay` in the second init call).

MPAS-Seaice and MPAS-Ocean restart writes are exact too (57 and 51 variables
bit-identical at `0001-01-03`).

**But B vs C differ** -- 172 of 365 fields at day 4. The seed is roundoff:

| field | day 4 rms(diff)/rms | day 5 | day-4 max abs |
|---|---|---|---|
| `a2x_Sa_tbot` | 7.8e-06 | 1.3e-05 | 0.014 K |
| `x2a_Sx_t` | 3.2e-06 | 8.9e-06 | 0.10 K |
| `o2x_So_t` | 1.8e-06 | 1.1e-05 | 0.036 K |
| `a2x_Faxa_swnet` | 3.4e-04 | 7.5e-04 | 2.2 W/m2 |

Single-precision roundoff amplifying chaotically, roughly doubling per day.
An `ERS` test would formally fail and it is worth filing, but every component's
restart is exact and a segment boundary perturbs at roundoff level. **This does
not block a 5-year `RESUBMIT` chain.** (`i2x_Si_anidf` showing max = NaN is a
masking artifact; ice albedo is undefined where there is no ice.)

### 32. A non-finite initial condition kills the run at step 0 **[fixed]**

The published SamudrACE-E3SMv3 initial conditions carry NaN in `ICEFRAC` over
every cell without sea ice -- 38877 of 64800, 60 % of the globe. `LANDFRAC`
and `OCNFRAC` are clean.

Fatal rather than local: the spherical harmonic transform inside an SFNO is
global, so one NaN returns all 51 output channels as NaN, and the emulator is
autoregressive. `ace_comp_init` runs an inference before the first coupler
import, so the run is dead at step 0 -- observed as `zbot/tbot/pbot/ubot/vbot`
all NaN on the first export, with `shum` and `swnet` reading 0.0 because the
export's own `max(...,0)` floors swallowed the NaN.

`eatm_sanitize_inputs` (commit `54d3a751d2`) runs on the initial condition
before the first inference, replaces non-finite values with zero and reports
the count per channel by name. `make_eatm_ic.py` does the same at build time.

### 33. Nothing in this configuration measures conservation **[open]**

`BUDGETS=FALSE` in `env_run.xml` (`do_budgets = .false.` in `drv_in`),
MPAS-Ocean's `conservationCheck` AM off, MPAS-Seaice's off. Zero matches for
`NET HEAT BUDGET` in any `cpl.log`. So the coupler cannot tell you whether an
ML atmosphere closes energy, and it has no particular reason to.

`./xmlchange BUDGETS=TRUE` gets the surface budget (`seq_diag_mct.F90:2200`).
The coupler has **no TOA term at all** -- but the emulator predicts `FLUT` and
`FSUTOA` and neither is ever read, so `SOLIN - FSUTOA - FLUT` as a logged
global mean is a small change and the highest-value diagnostic available.

### 34. Two production traps found by running into them **[document]**

- **`STOP_N` shorter than `REST_OPTION` writes no restart at all.** A 20-day
  run with `REST_OPTION=nmonths` never fires the restart alarm; the coupler
  warns `Stop time too short ... restarts won't be written` and you get
  nothing to continue from. Use `REST_OPTION=ndays` with `REST_N <= STOP_N`
  for any test you intend to restart.
- **A wall-clock kill during the restart write leaves a corrupt file.** The
  31-day job died 6 s after `mpaso.rst` finished, leaving
  `mpassi.rst.am.timeSeriesStatsMonthly` at 0 bytes; the next run then failed
  to open it with `NetCDF: Unknown file format`. Budget the restart write
  (~150 s for the ~11.9 GB end-of-month set) and clean the rundir between
  runs. MPAS `highFrequencyOutput` also has `clobber_mode: append`, so a
  re-run silently mixes trajectories into the previous file.

### Where this session's data lives

| what | where |
|---|---|
| pre-land-fix month (the "before") | `.../eatm-buildcheck-08151753/run/baseline_57052155/` |
| post-land-fix 20 days | `.../eatm-buildcheck-08151753/run/` |
| SamudrACE `lowest_level` 20 days | `.../GMPAS-EATM-SamudrACE-5yr/run/lowest_level_57056960/` |
| SamudrACE `near_surface` 20 days | `.../GMPAS-EATM-SamudrACE-5yr/run/` |
| ERS phases A/B/C + day-3 restarts | `.../eatm-ers-4n/run/ers/` |

All under `/pscratch/sd/m/mahf708/e3sm_scratch/pm-gpu/`.

### What I would do next, in order

1. **Implement the total energy budget corrector in the tracing script.** Both
   forcing fields it needs are already in EATM's input block. It is the only
   configured corrector being dropped, and the emulator that drops it runs at
   twice the imbalance of the one that never had it.
2. **`./xmlchange BUDGETS=TRUE`** and add the TOA imbalance diagnostic, so the
   next run produces a conservation record instead of one inferred from ocean
   heat content.
3. **Chase ACE2's latent-heat residual separately** -- `Sa_shum` is total water
   including condensate, and 19.3 % of ocean cells are supersaturated.
4. **Start from a spun-up G-case restart.** `config_initial_condition_type =
   'cice_default'` cold-starts with 2.5 m ice at 100 % concentration everywhere
   south of 60S and north of 70N; SH ice is 3-4x observed and contributes an
   unknown share of the residual imbalance in *both* emulators.

## Session of 2026-08-15 (late): the forcing was the wrong field

This session started from an independent review that flagged the startup
clocking as inconsistent. Chasing that led to a larger defect in the same
area: for every step of every run so far, the emulator has been handed a
top-of-atmosphere insolation pattern unlike anything in its training set.

### 35. SOLIN was the instantaneous value where the emulator wants a 6 h mean **[fixed]**

`ace_compute_solin` evaluated `S0 * eccf * max(0, cosz)` at a single instant,
the prediction target time T+dt.

In the E3SMv3 6-hourly stream both emulators were trained on, SOLIN is written
with `cell_methods = "time: mean"`. Measured directly from the training data,
`/global/cfs/cdirs/e3smdata/simulations/v3.LR.historical_0101.aigo/run/*.eam.h0.*.nc`
(`time_period_freq = hour_6`, `time_bnds[0] = [32850.0, 32850.25]`, so the
timestamp is the *end* of the window):

```
cell_methods = "time: mean"   FLDS FLUS FLUT FSDS FSUS FSUTOA SOLIN
                              LHFLX SHFLX PRECT PRECST DTENDTTW TAUX TAUY
cell_methods = "time: point"  PS TS T_0..7 STW_0..7 U_0..7 V_0..7
                              Tat2m Qat2m Uat10m Vat10m LANDFRAC OCNFRAC ICEFRAC PHIS
```

So the SOLIN channel is the mean insolation over the 6 h *leading up to* its
timestamp -- exactly the interval (T, T+dt] the model steps across. The
next-step-forcing convention is consistent with that and was already handled
correctly: fme feeds such a channel from time index `step+1`
(`fme/ace/stepper/single_module.py:1139-1145`).

**This is not a small error.** The instantaneous field and the 6 h mean field
have the *same global mean* -- 342.05 W/m2 either way, because the lit
hemisphere is always the same fraction of the globe -- so no global budget can
see it. Point by point they are different fields: the instantaneous field is a
cosine bullseye at the subsolar point, the 6-hourly mean is a band smeared
across 90 degrees of longitude. Measured on the 180x360 grid at solar
declination -0.4014 rad:

| | value |
|---|---|
| global mean, instantaneous | 342.05 W/m2 |
| global mean, 6 h window mean | 342.05 W/m2 |
| **area-weighted RMS difference** | **329.93 W/m2** |
| max absolute difference | 821.3 W/m2 |

The RMS difference is 96 % of the field's own global mean, and it is invariant
to the phase of the window. That this passed unnoticed for so long is a direct
consequence of the global mean being exactly right.

Fixed by evaluating the window mean with the midpoint rule on 48 sub-intervals
of 7.5 min. Convergence against a 2400-point reference:

| sub-steps | interval | RMS error | max error |
|---|---|---|---|
| 4 | 90 min | 4.24 W/m2 | 15.3 |
| 12 | 30 min | 0.469 | 1.68 |
| 24 | 15 min | 0.117 | 0.42 |
| **48** | **7.5 min** | **0.029** | **0.107** |

48 costs 48 `shr_orb_cosz` evaluations per cell per emulator step, which is
nothing next to one SFNO forward pass, and it puts the quadrature error three
orders of magnitude below the error it replaces.

### 36. Startup advanced the emulator twice at the same model time **[fixed]**

The independent review was right, and the mechanism is now pinned down.

For a prognostic atmosphere the MCT driver calls `atm_init_mct` **twice**, and
does not advance the clock between them:

| # | call site | `EClock_a` curr_tod / stepno |
|---|---|---|
| 1 | `driver-mct/main/cime_comp_mod.F90:1532` (phase 1) | 0 / 0 |
| 2 | `driver-mct/main/cime_comp_mod.F90:2446` (phase 2, gated on `atm_prognostic`) | 0 / 0 |
| - | `cime_comp_mod.F90:2826` `clockAdvance` -- the only one in the driver | -> 1800 / 1 |
| 3 | `cime_comp_mod.F90:3263` first `atm_run_mct` | 1800 / 1 |

`atm_comp_mct` branches on a saved `first_time` flag rather than `atm_phase`,
so the phase-2 call runs the full run method. `ace_comp_run`'s only guard was
`mod(curr_tod, eatm_model_dt) == 0`, which is true at tod = 0. Startup
therefore ran inference twice: once in `ace_comp_init` (IC(T0) -> T0+6h) and
again in the phase-2 run call (T0+6h -> T0+12h). The emulator state ended up
permanently one emulator step ahead of the coupler clock, and because the
restart writes both brackets verbatim the offset survived every restart.

Two fixes, both in `ace_comp_mod.F90`:

- **The advance is now idempotent in model time.** `last_adv_ymd/tod` record
  when the emulator last stepped; an advance is skipped if it has already
  happened at this model time. This does not depend on counting driver phases,
  which is the property that actually matters.
- **The lower bracket is seeded from the initial condition.** Startup used to
  set `t_im1 = t_ip1 = ` the first prediction, throwing away the one state
  whose valid time is known exactly. It now seeds `t_im1` from the IC via the
  new `out_from_in` channel map, so the first interval interpolates T0 -> T0+6h
  properly. The flux channels have no IC counterpart and are held at the first
  prediction, which is the best available.

With both in place the emulator state's valid time equals the coupler clock
from T0 onward, and `julday + dt` in `ace_compute_solin` is then exactly the
prediction target -- the two errors are fixed independently rather than being
left to cancel.

### 37. Flux channels are interval means and were interpolated as snapshots **[fixed]**

Given the `cell_methods` split in #35, the two kinds of output channel have to
reach the coupler differently, and both used to be linearly interpolated
between brackets.

For a snapshot channel that is right. For an interval mean it is not: `t_ip1`
*is* the mean over the interval being stepped across, so it applies unchanged
across the whole window. Interpolating from the previous window's mean means
the applied flux only reaches the correct value at the very end, and averaged
over the interval it is `(mean_previous + mean_current)/2` -- a half-step lag.
At a 6 h emulator step that is a **3 h lag on all surface radiation,
turbulent fluxes, stresses and precipitation**, i.e. 45 degrees of diurnal
phase, plus an equivalent smoothing.

fme relies on the same split internally -- its time coarsener takes snapshot
variables from the end of a window and averages the rest
(`fme/core/dataset/time_coarsen.py:79-85`), and the moisture and energy
correctors multiply a flux by the timestep to get a change in a path quantity,
which only closes for an interval mean.

Fixed with `eatm_channel_is_interval_mean` and a resolved-once `out_is_mean`
table, applied in the new `ace_bracket_blend`. Note the classification is a
property of the training data, not a tunable.

**Deferred, and worth doing next:** the shortwave now reaches the ocean as a
6-hourly step function. That is honest -- there is no sub-6-hour information in
a 6-hourly mean -- but it is not the best available. datm disaggregates an
interval-mean shortwave using the cosine zenith angle (`tintalgo = 'coszen'`),
which preserves the interval mean exactly while restoring the correct diurnal
shape. EATM already computes the window-mean insolation, so the scaling
`FSDS_now = FSDS_mean * SOLIN_inst(t) / SOLIN_windowmean` needs only an
instantaneous `cosz` per coupler step. It should be applied to FSDS/FSUS/swnet
only -- there is no comparable proxy for longwave, precipitation or the
turbulent fluxes. Left out of this session deliberately so that the A/B below
measures a coherent set of changes.

### 38. Non-finite inputs were zero-filled in every channel **[fixed]**

`eatm_sanitize_inputs` replaced every non-finite input with zero and warned.
Zero is the physically correct reading only for a surface fraction that means
"none here" -- which is the case it was written for, the published SamudrACE
ICs carrying NaN in ICEFRAC over 60 % of the globe. For PS, TS, PHIS, a
temperature, a wind or a water channel it substitutes an impossible state that
the emulator integrates forward happily, converting one detectable failure into
an undetectable bias.

Now: zero-fill for `LANDFRAC`/`OCNFRAC`/`ICEFRAC` (`eatm_channel_zero_is_valid`),
abort naming the channel for everything else.

### 39. Nothing checked what came back out of the graph **[fixed]**

Two new checks in `ace_validate_outputs`, called after every inference.

- **Non-finite output is fatal, every step.** The emulator is autoregressive
  and an SFNO's spherical harmonic transform is global, so one NaN in one
  channel becomes every channel on the next step and stays that way. Nothing
  downstream caught it: the export clamps compare against zero, and `NaN < 0`
  is false, so a NaN passed through `max()` untouched and reached the ocean as
  NaN forcing.
- **The channel contract is range-checked once, on the first inference.** A
  traced model is an opaque graph; nothing at load time ties its channel order
  to the compiled table. If `eatm_emulator` names the wrong table, or a
  checkpoint is re-traced with a different layout, every index silently reads
  the wrong field. The ranges (`eatm_channel_range`) are loose enough that no
  plausible atmospheric state trips them and tight enough that reading a
  humidity where a pressure was expected cannot pass.

### 40. The exported humidity was not capped at saturation **[fixed]**

`shr_flux_atmOcn` treats `Sa_shum` as the vapour mixing ratio and drives the
latent flux with `(q_sat(SST) - Sa_shum)`, so a supersaturated value suppresses
or reverses evaporation. In the `lowest_level` configuration the exported field
is the emulator's specific *total* water, condensate included: 19.3 % of ocean
cells arrived supersaturated, with relative humidities up to 2.82 (#27).

No emulator channel separates vapour from condensate, so capping at saturation
is the closest estimate of the vapour part the model's own output supports.
New namelist `eatm_cap_shum`, default `.true.`; the count and the maximum RH
seen before capping are logged each emulator step.

### 41. Surface fractions were bounded only in sum **[fixed]**

`ace_eatm_import` clipped `ofrac + ifrac + lfrac` to [0,1] but never the
individual fields, so a negative ICEFRAC compensated by an ocean fraction above
one still produced a plausible total. Each is now bounded on its own and the
triple renormalised if it exceeds one.

### 42. `EATM_MODE=NULL` was read and ignored **[fixed]**

`bld/build-namelist:299` sets `do_eatm = .false.` for `EATM_MODE=NULL`. The
flag was read, broadcast, printed -- and never tested. Initialization went on
to read a mesh, load a traced graph and run inference regardless.

There is nothing here for a null mode to switch off, and honouring the flag by
silently skipping all of it would hand the coupler an atmosphere exporting
zeros. EATM now aborts with a message pointing at SATM, which is the component
that actually implements a stub atmosphere.

### 43. The atmosphere and the ocean use different surface fluxes **[open, scoped, now measured]**

The independent review's headline finding, and it is real. The emulator
predicts LHFLX, SHFLX and (SamudrACE) TAUX/TAUY, and evolves its atmosphere
consistently with them. The coupler ignores those channels and rebuilds the
turbulent fluxes from the exported state and the SST with `shr_flux_atmOcn`;
that is what the ocean integrates. The difference is energy entering the ocean
without leaving the atmosphere.

**Not fixed this session, deliberately.** Scoping it out produced two reasons
to be careful rather than quick:

1. **The emulator's fluxes are grid-cell means over land + ice + ocean, while
   `Faox_*` is the open-ocean-only flux that `prep_ocn_mod.F90:1261-1267` then
   weights by `afrac`.** Substituting one for the other double-counts in every
   mixed cell. This is the main correctness hazard and it is not cosmetic.
2. **Sea ice would not follow.** There is no `seq_flux_atmice` in the MCT
   driver; MPAS-SI computes its own atm/ice turbulent fluxes internally through
   Icepack (`mpas_seaice_icepack.F:5394` -> `icepack_atmo.F90:829`). An
   "atmosphere supplies the flux" option would cover the open-ocean fraction
   only, leaving an asymmetry that has to be argued for rather than assumed.

Also: `Faox_lwup` is currently built from the ocean's own SST, and `Faox_evap`
must stay exactly `lat / latvap` or the coupler's water and energy budgets
diverge. Neither survives a naive override.

The mechanical route, if it is taken later, is to **overwrite `Faox_*` inside
`seq_flux_atmocn_mct` rather than route new fields through the ocean merge** --
`prep_ocn_mod.F90:922-926` aborts outright if an x2o field is matched by both
a2x and xao, and a hypothetical `Faxa_taux` shares its `itemc` with the
existing `Faox_taux`. Keep the existing `shr_flux_atmocn` call (the ocean and
atmosphere merges still need `tref`, `qref`, `ustar`, `re`, `ssq`, `duu10n`,
`u10res`) and override `sen`/`lat`/`taux`/`tauy`/`evap` immediately before the
store loop at `seq_flux_mct.F90:1685-1724`. `ocn_surface_flux_scheme` has no
`<valid_values>` guard, so a new value flows through the infodata plumbing
untouched and is the cheapest switch.

**What this session did instead: made it measurable in the run.** The new
`ace_flux_budget_report` logs, once per emulator step, area-weighted global
means of the emulator's predicted latent, sensible, stress and radiative terms
against the coupler's applied ones, with the differences and the implied net
surface and TOA budgets. Until the pathway question is settled the number needs
to be visible in the log of every production run rather than reconstructed
afterwards from history files that do not carry the emulator's flux channels at
all.

### What is *not* wrong, checked and cleared

- **The unfilled a2x fields are harmless in this compset.** EATM writes 21 of
  them and leaves the aerosol and dust deposition fluxes at zero. MPAS-O never
  reads `Faxa_bcph*`/`Faxa_ocph*`/`Faxa_dst*` at all (its ocean dust and iron
  come from `ecosysMonthlyClimatology`), and MPAS-SI reads them only under
  `config_use_aerosols` / `config_use_zaerosols`, both default `.false.`.
  `Sa_uovern` is consumed only by ELM's Froude-number precipitation
  downscaling, and EAM itself writes 0.0 on that path. `Sa_co2prog`/`Sa_co2diag`
  are not in the field list for this compset at all -- the `CCSM_BGC=CO2A`
  override keys on `_EAM`, which does not substring-match `_EATM`. **This
  becomes harmful the moment BGC is enabled**: 0 ppm atmospheric pCO2 would
  make the ocean outgas continuously.
- **`Sa_pslv`, which MPAS-O does use unconditionally** (`ocn_comp_mct.F:2554`
  -> `surfacePressure`), is written.
- **The `npes > 1` abort already exists** (`eatmSpmdMod.F90:64`), so the
  single-task assumption cannot be violated silently.

### 44. The obvious way to write the flux comparison is wrong **[trap, avoided]**

Worth recording because the first version of `ace_flux_budget_report` got it
wrong and the numbers looked *reassuring*.

The coupler's `Faxx_*` are **merged** fluxes, `sum over surface types of
frac_s * F_s`. In `GMPAS-EATM` the land model is a stub, so `lfrac` is zero and
over the ~34 % of the globe no surface model covers, the merged flux is not
small -- it is structurally absent. Take a plain area mean of that and compare
it against the emulator's full-cell flux and the coupler is understated by the
uncovered fraction, which happens to make a large disagreement look like a
small one. The first version reported an emulator/coupler latent difference of
+3.8 W/m2 where the covered-area comparison gives roughly three times that.

Both columns are now reported per unit *covered* area: the merged coupler flux
divided by the mean covered fraction, the emulator's flux weighted by that same
fraction. The covered fraction itself is printed on the header line so the
basis is never in doubt.

The same routine also had two sign errors, fixed by pinning the convention to
EAM rather than to intuition: `cam_in%wsx = -Faxx_taux`
(`eam/src/cpl/atm_comp_mct.F90:1801`) and EAM's `TAUX` history field *is*
`cam_in%wsx` (`cam_diagnostics.F90:2145`), so the emulator's `TAUX` and `FLUS`
channels compare against EATM's imported `wsx` and `lwup` with no flip at all.
`Faxx_lat` is not negated on import while `Faxx_sen` is, which is easy to miss.

### 35a. Verifying the SOLIN fix without trusting anything

EATM writes the last SOLIN it computed into its restart file, so a restart is a
direct record of what the emulator was fed. Two signatures separate the
instantaneous field from the window mean, and neither needs orbital parameters,
a calendar, or the model grid -- which matters, because getting those slightly
wrong is exactly how one talks oneself into the wrong answer here:

- **dark fraction.** An instantaneous insolation field is zero over exactly the
  unlit hemisphere: half the globe, to machine precision. A 6 h mean is zero
  only where the sun stays down for the whole window, and the terminator sweeps
  90 degrees of longitude in 6 h, so it lands near a quarter plus polar night.
- **peak value.** The instantaneous field peaks at `S0*eccf` at the subsolar
  point; averaging over a window in which the sun moves 45 degrees either side
  knocks that down by roughly a quarter.

Measured on the two restarts:

| restart | dark fraction | peak (W/m2) | global mean |
|---|---|---|---|
| `near_surface_57058963_baseline`, pre-fix, `0001-01-21` | **0.5000** | 1412.9 | 353.48 |
| `smoke_1day_fixes`, with fix, `0001-01-02` | **0.2767** | 1296.6 | 353.96 |
| expected, instantaneous | 0.5000 | ~1365 | |
| expected, 6 h window mean | ~0.27 | ~1050 | |

A dark fraction of exactly 0.5000 is the smoking gun: every step, the emulator
was told half the planet was in darkness, when the field it was trained on has
the sun somewhere in the window over roughly three quarters of the globe. The
global means agree to 0.5 W/m2, which is why nothing caught this earlier.

`tools/check_eatm_solin.py` runs the test on any EATM restart and exits
non-zero if it finds the pre-fix signature.

### 40a. How much the humidity cap is actually worth: almost nothing

Measured from the day-1 restart of the fixed SamudrACE run, computing `q_sat`
the same way the export does:

| configuration | supersaturated area | max RH | effect of the cap on global mean q |
|---|---|---|---|
| `near_surface` (`Tat2m`/`Qat2m`, `pbot = PS`) | 9.1 % | 1.98 | **-0.07 %** |
| `lowest_level` (`T_7`/`STW_7`, layer midpoint) | 17.4 % | 1.82 | **-0.26 %** |

The `lowest_level` figure reproduces the 19.3 % measured in #27 from a
different run. But the supersaturation is concentrated where there is hardly
any water to remove -- in `lowest_level`, 13.0 of the 17.4 percentage points
sit in the 240-273 K band -- so capping it changes the humidity the bulk
formula sees by a quarter of a percent.

The log line reporting `max RH before cap 4.353` on some steps is the same
effect at its extreme: at 220 K, `q_sat` is around 6e-6 kg/kg, so an absolute
error far too small to matter shows up as a relative humidity of four.

**So the cap is a correctness fix, not a bias fix.** The interface was invalid
-- `shr_flux_atmOcn` is entitled to a vapour mixing ratio and was being handed
total water -- and it should stay fixed. It cannot be worth more than a
fraction of a W/m2, and the direction is *unhelpful* anyway: capping lowers
`Sa_shum`, which raises `(q_sat(SST) - Sa_shum)` and so increases evaporation
and ocean cooling, in a run whose problem is already too much ocean cooling.

### 43a. The split-flux problem is smaller than it looks, and fixable here

One fact changes what finding 43 means, and it was worth checking rather than
assuming.

**The emulator's `LHFLX` and `SHFLX` are not an independent physics package's
opinion. They are the coupler's own fluxes.** In E3SM the atmosphere does not
compute its turbulent surface fluxes; `seq_flux_atmocn_mct` does, and EAM
imports the result: `cam_in%lhf = -x2a(Faxx_lat)`
(`eam/src/cpl/atm_comp_mct.F90:1794`, and `atm_import_export.F90:89`). EAM's
`LHFLX` history field is exactly that imported value
(`cam_diagnostics.F90:2142`). The `aigo` h0 stream the emulators were trained
on therefore contains, in its `LHFLX` channel, `shr_flux_atmOcn` evaluated on
EAMv3's lowest model level and the SST -- merged over surface types, so a
full-cell mean, which is also how EATM's budget report reads it.

So the emulator is trained to predict *what this very coupler would compute*.
The two columns in the budget report are not two irreconcilable
parameterisations. They are the same bulk formula evaluated on two different
atmospheric states:

- the coupler's column: `shr_flux_atmOcn` on the state **EATM exports**;
- the emulator's column: `shr_flux_atmOcn` on the state **EAMv3 had**, learned.

**The ~20-29 W/m2 gap is therefore a measure of how far EATM's exported state
is from the one the emulator's flux prediction assumes** -- not evidence that
the atmosphere and ocean are running different physics. That reframes it from
a driver-level design question into a state-export question, answerable inside
this component, with no change to `seq_flux_mct.F90` and none of the hazards
in 43 (mixed-cell double counting, the Icepack asymmetry, `Faox_lwup` and
`Faox_evap` consistency).

It also explains why the `eatm_surface_layer` change helped but did not close
it (#28). EAMv3's lowest model level sits around 60 m; the emulator's coarsened
lowest layer is at ~450 m and its near-surface diagnostics are at 2 m and 10 m.
Neither is the height the learned flux corresponds to, and the `near_surface`
export is not even internally consistent -- `Tat2m` and `Qat2m` at 2 m,
`Uat10m`/`Vat10m` at 10 m, all declared at a single `Sa_z = 10 m`.

**The experiment this suggests**, now that the disagreement is instrumented and
prints every emulator step: treat the budget report's `turbulent total` column
as the objective, and try export variants against it -- `lowest_level`,
`near_surface`, and a Monin-Obukhov-consistent state at an intermediate height
built from both. A configuration that drives the difference toward zero is one
in which the ocean receives the flux the atmosphere believes it lost, which is
the conservation property the coupled system currently lacks. This is cheap to
test: the mismatch is visible within the first emulator step of a run, so each
variant costs minutes, not a 20-day integration.

### 35b. What the window mean costs, and the optimisation not taken

From the E3SM timers, `a:eatm_datamode` (the whole emulator advance, inference
included) on the SamudrACE case:

| run | coupler steps | total | per step |
|---|---|---|---|
| pre-fix 20-day (`57058963`) | 960 | 27.62 s | 28.8 ms |
| with window mean, 1-day smoke (`57060952`) | 48 | 2.82 s | 58.7 ms |

So the emulator advance roughly doubles, from about 1.8 to 3.4 s per model day
of `ATM Run Time`. In the coupled system that is noise -- the ocean is ~32 s
per model day and the atmosphere runs on its own node -- but it is 48 x 64800
`shr_orb_cosz` calls per emulator step, and it does not have to cost anything.

The inner loop currently recomputes `lat*degtorad`, `lon*degtorad` and a full
`sin`/`cos` pair per cell per sub-interval. Cache `sin(lat)`, `cos(lat)`,
`sin(lon)`, `cos(lon)` per cell once in `ace_cache_areas`, then expand

```
cosz = sin(lat)sin(delta) - cos(lat)cos(delta)cos(a + lon)
     = sin(lat)sin(delta) - cos(lat)cos(delta)[cos(a)cos(lon) - sin(a)sin(lon)]
```

with `a = 2*pi*(jday - floor(jday))`, so `sin(a)`/`cos(a)`/`sin(delta)`/
`cos(delta)` are computed once per sub-interval and the per-cell work is a
handful of multiplies with no transcendentals at all.

**Deliberately not done in this session.** It is algebraically exact but not
bit-exact, and applying it midway through the A/B runs below would have meant
reporting numbers from a build that no longer matched the source. It is a
contained change for whoever picks this up next.

## Session of 2026-08-15 (late): measured results

### 45. What the fixes did to the ocean imbalance: not what one would hope

SamudrACE-E3SMv3, `near_surface`, 20 days from `0001-01-01`, net surface heat
flux from the ocean's volume-mean temperature drift (#27's metric):

| run | dT (K) | fit | endpoint |
|---|---|---|---|
| pre-fix, job `57058963`, 448 ocean tasks | -0.006793 | -63.37 | **-62.37 W/m2** |
| with the fixes, 192 ocean tasks | -0.007459 | -70.09 | **-68.49 W/m2** |

**The imbalance got about 6 W/m2 worse, not better.**

Three reasons that number cannot yet be attributed to the fixes, in order of
how much they worry me:

1. **SamudrACE is stochastic and its RNG is unseeded** (#13). Two runs of the
   *same* code differ by an unmeasured amount, and nobody has ever quantified
   the spread. A 6 W/m2 difference over 20 days may be entirely within it.
2. **The decompositions differ** -- the pre-fix baseline ran on 448 ocean
   tasks, this one on 192, because it had to share a 4-node interactive
   allocation. Round-off diverges from step one.
3. **The humidity cap pushes this way by construction.** Capping lowers
   `Sa_shum`, which raises `(q_sat(SST) - Sa_shum)` and so increases
   evaporation and ocean cooling (#40a). It is the correct thing to do and its
   sign is unhelpful.

This is why finding 46 exists: the same comparison on the *deterministic*
emulator, with the decomposition held fixed.

### 46. The in-run surface exchange, over a full 20 days

The number the new budget report is for, averaged over all 80 emulator steps of
the run above, per unit covered area:

| | emulator | coupler | difference |
|---|---|---|---|
| latent + sensible | 150.10 | 172.05 | **+21.95 W/m2** |

Stable step to step (the last five steps read 23.7, 22.7, 22.6, 23.7, 23.3) and
in close agreement with the 23.2 W/m2 an independent review estimated offline
from day-21 history files. Two routes to the same number, one of them now
printed by the model itself.

Also from the same report, and new: **the emulator's own top-of-atmosphere
balance, `SOLIN - FSUTOA - FLUT`, averages +16.26 W/m2** over the run. That is
an order of magnitude larger than a balanced climate should show, and it is a
direct measurement of the thing `total_energy_budget_correction` exists to
control -- the corrector the SamudrACE checkpoint configures and the tracing
script silently drops (#29). Finding 29 was a suspicion; this is a number.

Together these say the surface loses ~22 W/m2 more than the atmosphere believes
it gave up, while the top of the atmosphere gains ~16 W/m2 that nothing removes.
Neither is a forcing-field problem, which is consistent with the fixes above
being correctness fixes that leave the energy bias where it was.

### 47. Seeding the RNG is the top *methodological* priority **[open, scoped]**

This session ran into #13 as a practical wall rather than a theoretical one.
The SamudrACE A/B in #45 produced a 6 W/m2 difference that **cannot be
interpreted**, because two runs of identical code differ by an unknown amount
and no one has ever measured the spread. Every future science comparison on
this emulator has the same problem. That makes seeding worth more than any
single physics fix on the list.

Two things found while scoping it, both of which make it easier than #13
implies:

- **The build already compiles C++ in `eatm/src`.** The source glob in
  `components/cmake/cmake_util.cmake:15` matches `*.cpp` in every Filepath
  directory, and `build_model.cmake:354` already links `FTorch::ftorch` (and
  so libtorch) into the eatm target. A `.cpp` dropped next to the Fortran is
  compiled and linked with no build-system change at all.
- **What is missing is only the Torch *header* path.** `find_dep_packages.cmake`
  finds FTorch, whose installed `include/` carries `ctorch.h` and the Fortran
  `.mod` files but not `<torch/torch.h>`. `torch::manual_seed` needs the
  LibTorch C++ headers, so this needs a `find_package(Torch)` (or an explicit
  include directory from the same install prefix FTorch was built against)
  before the shim will compile.

Sketch: a C-linkage `void eatm_torch_manual_seed(long)` calling
`torch::manual_seed`, which in current LibTorch seeds the CPU generator and
every CUDA device generator; an `iso_c_binding` interface in `ace_comp_mod`;
and an `eatm_rng_seed` namelist entry called once in `ace_comp_init` after
`torch_model_load`. To make restarts reproducible the generator state, not just
the seed, has to be carried across -- but a fixed seed alone already makes
same-length A/B runs comparable, which is what is needed now.

**Not attempted here**: it needs a build-system change plus a rebuild and a
validation run, and doing that in the last hour of a session with the nodes
busy is how a working branch gets left broken.

### 48. A control run: the decomposition is not a confounder **[measured]**

Before reading anything into #45, the obvious alternative explanation had to go.
The same pre-fix code (`7db0a0e848`), same compset, same start date, ACE2-EAMv3,
10 days, run on two different task counts:

| run | dT (K) | fit | endpoint |
|---|---|---|---|
| 448 ocean tasks, job `57053850` (first 10 days of the 20-day run) | -0.001710 | -33.22 | -33.15 W/m2 |
| 192 ocean tasks, tonight | -0.001710 | -33.22 | -33.16 W/m2 |

**They agree to 0.01 W/m2, with the temperature drift identical to six digits.**
Halving the ocean decomposition does not move this metric at all.

Two consequences:

- The 6 W/m2 in #45 is *not* the decomposition. It is either the stochastic
  spread of an unseeded SamudrACE (#47) or a real response to the corrected
  forcing -- and only #47 being fixed can tell those apart.
- Anything the deterministic ACE2 comparison shows is attributable to the code
  change alone.

It also re-derives the -33.3 W/m2 that #27 recorded for ACE2, from a different
run on a different node count, which is a useful check on the metric itself.

### 49. The deterministic answer: the fixes are worth about 1 W/m2 **[measured]**

ACE2-EAMv3 is deterministic, and #48 showed the decomposition contributes
nothing, so this comparison is attributable to the code change alone. Same
compset, same start, same 192 ocean tasks, 10 days:

| run | dT (K) | fit | endpoint |
|---|---|---|---|
| pre-fix `7db0a0e848` | -0.001710 | -33.22 | **-33.16 W/m2** |
| with the fixes | -0.001760 | -34.03 | **-34.12 W/m2** |

**-0.96 W/m2.** That is the honest size of this session's physics changes on the
energy bias, and it is close to what #40a's arithmetic predicted for the
humidity cap alone in the `lowest_level` configuration ACE2 falls back to
(0.26% of mean humidity, worth of order 1 W/m2, in the cooling direction).

So the fixes in 35-37 are correctness fixes. They put the emulator on the
forcing field it was trained on and align it with the coupler clock; they do
not move the ocean imbalance, and nothing about them ever should have been
expected to -- the SOLIN correction is exactly global-mean-neutral by
construction, and the interval-mean correction is mean-neutral to a boundary
term (see the note in 37).

**A useful by-product: a first bound on SamudrACE's stochastic spread.** The
same set of changes measured -6.12 W/m2 on stochastic SamudrACE (#45) and
-0.96 W/m2 on deterministic ACE2. Allowing for the two emulators differing, the
gap implies a run-to-run spread of order 5 W/m2 over 20 days -- comfortably
larger than any effect this session set out to measure, which is finding 47's
whole point.

### 50. Both emulators disagree with the coupler by the same ~25 W/m2

The in-run report on the ACE2 run above, over all 40 emulator steps:

| | emulator | coupler | difference |
|---|---|---|---|
| latent + sensible, per covered area | 118.48 | 143.46 | **+24.98 W/m2** |
| TOA net, global | | | **+12.84 W/m2** |

Against SamudrACE's +21.95 and +16.26 (#46). **Two different emulators, two
different architectures, one deterministic and one stochastic, trained on
different streams -- and the surface exchange disagrees by 22-25 W/m2 in both,
with a TOA imbalance of 13-16 W/m2 in both.**

That the number is nearly emulator-independent is the strongest evidence yet
that this is structural to the interface rather than a property of either
checkpoint, and it is consistent with 43a: both emulators learned to predict
`shr_flux_atmOcn` evaluated on EAMv3's lowest model level, and EATM is handing
the same `shr_flux_atmOcn` a state from a different height.

Note also that the two orderings do not match -- ACE2 has the *larger* flux
mismatch (25.0 vs 22.0) but by far the *smaller* ocean imbalance (-33 vs -68).
So the surface exchange gap is not the whole story either, and the radiative
terms differ between the two emulators as well. It is the largest single term
that is common to both.

### 51. The export state moves the mismatch by 10 W/m2 -- 43a is testable and true **[measured]**

The experiment 43a proposed, run: SamudrACE-E3SMv3, one model day, everything
identical but `eatm_surface_layer`, comparing the first four emulator steps.

| export | emulator | coupler | mismatch |
|---|---|---|---|
| `near_surface` (2 m / 10 m diagnostics at `Sa_z = 10 m`) | 128.21 | 147.33 | **+19.12 W/m2** |
| `lowest_level` (layer mean at ~450 m) | 127.87 | 157.50 | **+29.63 W/m2** |

**The emulator column barely moves (128.21 vs 127.87) and the coupler column
moves by 10 W/m2.** That is exactly the signature 43a predicts and it is a
strong internal check on the diagnostic itself: how EATM chooses to *describe*
its atmosphere to the coupler cannot change what the emulator predicted, and it
does not. All of the difference is in what `shr_flux_atmOcn` makes of the state
it is handed.

Broken down, it is entirely the latent flux -- `+13.86` -> `+31.12` W/m2 --
while sensible moves the other way, `+5.26` -> `-1.49`.

Three things follow:

1. **`near_surface` is confirmed the better export**, independently of #28's
   route to the same conclusion, and now with a direct measure of *why*: it
   removes 10.5 W/m2 of spurious evaporation, not because it changes the
   atmosphere but because it stops misrepresenting it.
2. **The surface exchange gap is a state-export problem, as 43a argued.** A
   third of it is addressable by changing nothing but the height at which the
   state is declared. No driver change was needed to demonstrate this.
3. **The remaining ~19 W/m2 has an obvious next suspect.** The `near_surface`
   export is not internally consistent: `Tat2m` and `Qat2m` are 2 m values,
   `Uat10m`/`Vat10m` are 10 m values, and all four are declared at a single
   `Sa_z = 10 m`. A Monin-Obukhov-consistent state -- all four fields at one
   height, or the state transferred to the ~60 m level EAMv3's learned flux
   actually corresponds to -- is the experiment to run next.

**This costs one model day per variant.** The mismatch is converged within four
emulator steps, so the objective is readable in about seven minutes of wall
clock on four nodes. There is no reason for the next iteration of this to be
slow.

### 47a. Seeding, scoped down to one CMake line and a 15-line shim

Chased further, and it is smaller than 47 makes it sound. Everything needed is
already installed on Perlmutter; nothing has to be built or staged.

- `FTorch::ftorch` exports only its own include directory and links only
  `stdc++` (`FTorchTargets.cmake:62-63`), so it does **not** propagate the Torch
  headers -- which is why this looked like a build problem.
- But `FTorchConfig.cmake:31` already does
  `find_dependency(Torch REQUIRED PATHS
  "/global/cfs/cdirs/e3sm/software/libtorch/libtorch-shared-with-deps-2.10.0+cu128")`.
  So by the time `find_dep_packages.cmake` has found FTorch, the `Torch`
  package is found in the same scope and **`TORCH_INCLUDE_DIRS` is already
  set** (`TorchConfig.cmake:57-61`).
- The headers are there: `include/torch/csrc/api/include/torch/torch.h` exists,
  and `torch/utils.h:75` is `using at::manual_seed;`.

So the whole change is:

1. **One line** next to the existing FTorch link at
   `components/cmake/build_model.cmake:354`:
   `target_include_directories(${TARGET_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})`.
2. **A `.cpp` dropped into `eatm/src`**, which the source glob at
   `components/cmake/cmake_util.cmake:15` already compiles:
   `extern "C" void eatm_torch_manual_seed(long s) { torch::manual_seed(s); }`.
3. An `iso_c_binding` interface and an `eatm_rng_seed` namelist entry, seeded in
   `ace_comp_init` after `torch_model_load`.

**One thing to check rather than assume**: `at::manual_seed` seeds the CPU
generator, and whether it also reaches the CUDA generators the traced model
actually draws from depends on the LibTorch version. If not,
`at::cuda::manual_seed_all` is the one that matters here, since the model runs
on the GPU. Verify by seeding, running two identical single-day runs, and
diffing the exported fields -- if they are not identical the wrong generator
was seeded.

Not attempted tonight only because it needs a rebuild, and a rebuild would have
invalidated the restart test running on the same executable.

### 52. The clock fix survives a restart **[verified]**

The changes in #36 touch the advance guard, which is exactly the sort of thing
that breaks a restart quietly, and #31 records exact EATM restart as a property
of this branch worth keeping. So it was re-checked rather than reasoned about.

ACE2-EAMv3, deterministic, three phases: **A** startup 1 day writing a restart
at `0001-01-02`; **B** `CONTINUE_RUN` 1 day to `0001-01-03`; **C** a continuous
2-day control to the same date.

The emulator step cadence, straight out of the logs:

```
A   step 12 tod 21600   step 24 tod 43200   step 36 tod 64800   step 48 tod 0 (day 2)
                              -- restart --
B   step 60 tod 21600   step 72 tod 43200   step 84 tod 64800   step 96 tod 0 (day 3)
```

Three things this shows at once:

- **No advance at step 0.** The double-advance at `tod = 0` that #36 fixed is
  gone from a startup run.
- **No advance at the restart time either.** B resumes at `0001-01-02` `tod = 0`,
  which *is* an emulator boundary, and correctly does not step there -- the
  driver's first run call lands at `tod = 1800`, and the brackets read from the
  restart already cover that interval.
- **The cadence is unbroken across the restart**: 12, 24, 36, 48, 60, 72, 84, 96
  -- a regular 12-coupler-step interval with neither a duplicate nor a gap. A
  restart is invisible to the emulator's clock, which is the property that
  matters.

And the values, not just the cadence. Phase C's own step list is identical to
A's followed by B's (12, 24, 36, 48, 60, 72, 84, 96), and comparing restart
files:

| comparison | result |
|---|---|
| A's `eatm.r.0001-01-02` vs C's `eatm.r.0001-01-02` | **46 of 46 variables bit-identical** |
| B's `eatm.r.0001-01-03` vs C's `eatm.r.0001-01-03` | 2 identical, 44 differing, worst 3.4e-3 relative |

The first is the one that tests this session's changes, and it passes exactly:
two independent startup runs of different lengths produce a bit-identical
emulator state a day in, so the startup path is deterministic and the restart
write is exact.

The second is #31's already-documented coupled-system behaviour, not a
regression. MPAS's restart is not bit-for-bit, the emulator is handed marginally
different surface fields on the step after a restart, and an autoregressive
model spreads that through every channel within a day. It was true before these
changes and the magnitude is unchanged.

**So `RESUBMIT` remains safe, and #31's claim still holds with the new clock
handling.**

### Where this session's data lives

All under `/pscratch/sd/m/mahf708/e3sm_scratch/pm-gpu/`, and every case's
executable was rebuilt from the current branch tip at the end, so no stale
pre-fix binary is left anywhere.

| what | where |
|---|---|
| SamudrACE 20-day, pre-fix (the baseline) | `GMPAS-EATM-SamudrACE-5yr/run/near_surface_57058963_baseline/` |
| SamudrACE 20-day, with the fixes | `GMPAS-EATM-SamudrACE-5yr/run/near_surface_20day_WITHFIXES/` |
| SamudrACE 1-day, `lowest_level` export variant (#51) | `GMPAS-EATM-SamudrACE-5yr/run/exportvar_lowest_level_1day/` |
| SamudrACE 1-day smoke (flawed first budget report, see #44) | `GMPAS-EATM-SamudrACE-5yr/run/smoke_1day_fixes/` |
| ACE2 10-day, pre-fix `7db0a0e848` (#48, #49) | `eatm-ers-4n/run/` |
| ACE2 10-day, with the fixes (#49, #50) | `eatm-buildcheck-08151753/run/ace2_10day_WITHFIXES/` |
| ACE2 restart test phases A / B / C (#52) | `eatm-buildcheck-08151753/run/ers_{A,B,C}/` |
| ACE2 20-day pre-fix at 448 tasks (the older baseline) | `eatm-buildcheck-08151753/run/postfix_20day_prev/` |

### What to do next, in order

1. **Seed the RNG** (47, 47a). One CMake line and a 15-line shim, everything
   already installed. Until this is done no SamudrACE result below about
   5 W/m2 means anything, which is most of them.
2. **Scan the export state against the budget report** (43a, 51). The objective
   is already printed each emulator step and converges within four of them, so
   a variant costs about seven minutes. Start with a Monin-Obukhov-consistent
   `near_surface` -- all four fields at one height instead of 2 m temperature
   and humidity beside 10 m winds, all labelled 10 m.
3. **Implement `total_energy_budget_correction` in the tracing script** (29).
   The TOA imbalance is now measured at +12.8 (ACE2) and +16.3 W/m2
   (SamudrACE), so this is no longer a guess about whether it matters.
4. **Disaggregate the shortwave** with a cosine-zenith weight, as datm does
   (37). It preserves the interval mean exactly and restores the diurnal shape
   the 6-hourly means cannot carry.
5. **Optimise the SOLIN window mean** (35b) if the ~1.6 s per model day ever
   matters; the recipe removes the transcendentals from the inner loop.

## Session of 2026-08-16: where this sits against the references

### 53. Measured against anolan's EATM runs and jonbob's JRA baseline **[measured]**

The question this branch exists to answer is whether an emulated atmosphere can
drive a G-case as well as reanalysis forcing does. `jonbob`'s
`GMPAS-JRA1p5-2023` is that target. Ocean net surface heat flux from the
volume-mean temperature drift, computed identically for all of them:

| run | code | d0-10 | d0-20 | d0-90 | d0-365 | **2 years** |
|---|---|---|---|---|---|---|
| **jonbob `GMPAS-JRA1p5-2023` (target)** | datm/JRA | -9.19 | -4.20 | 1.75 | -4.82 | **-3.92** |
| anolan `GMPAS-EATM-gnugpu` | `7fce378e5f` | -95.37 | -87.93 | -48.03 | -27.17 | -14.99 |
| anolan `GMPAS-EATM-test4naser` | `23dd0c1b97` | -57.29 | -55.10 | -46.97 | -44.17 | -39.84 |
| mine, post-land-fix | `67d71a2a09`+ | **-33.15** | -33.26 | | | |
| mine, + this session's fixes | HEAD | **-34.12** | | | | |

All ACE2-EAMv3 except the target. Three things worth extracting.

**The direction is right.** At matched window the ACE2 bias has gone
-95 -> -57 -> -33, a 40-65% reduction against both of anolan's references.
Nearly all of that was the land-fraction reconstruction (#25); this session's
changes held it flat at about -1 W/m2 (#49), which is what they should have
done.

**The remaining gap is the size of the mismatch we measured.** -34.12 against
the target's -9.19 is **24.9 W/m2**. The emulator-versus-coupler turbulent
disagreement measured in-run for ACE2 is **24.98 W/m2** (#50). The denominators
are not identical -- ocean area against covered area -- so some of that
agreement is luck, but the coincidence of magnitude says the term identified in
43a and 51 is quantitatively the whole remaining distance to a JRA-quality run.
That is the strongest argument yet for spending the next effort there rather
than anywhere else on the list.

**Short windows do not predict the equilibrium.** `gnugpu` runs -95 at ten days
and -15 over two years; `test4naser` -57 and -40. So a 10-20 day A/B detects a
bias reliably but says little about where the run settles, and the two-year
production runs are the only test that does. Note the target has no such
transient at all (-9.19 at ten days, -3.92 at two years), so the large early
drift in every EATM run is genuine bias rather than cold-start adjustment --
which is what makes short A/B runs legitimate for detection in the first place.

**Caveat on the two references**: they are different commits, `gnugpu` being
the older, and they disagree with each other by more than this session's entire
effect. Neither is a "correct" EATM baseline; they bracket where the branch was.

### 54. What the humidity cap is really worth: 0.06 W/m2 **[measured]**

#49 attributed most of its -0.96 W/m2 to the humidity cap, reasoning from the
0.26% change in mean humidity. That was wrong, and the direct test says so.
ACE2, 10 days, identical except `eatm_cap_shum`:

| run | W/m2 |
|---|---|
| pre-fix | -33.16 |
| fixes, `eatm_cap_shum = .true.` | -34.12 |
| fixes, `eatm_cap_shum = .false.` | -34.06 |

The cap accounts for **0.06** of the 0.96; the other 0.90 is the emulator
responding to the corrected SOLIN pattern, the aligned clock and the
interval-mean flux handling. Those are individually mean-neutral by
construction, but the forcing *pattern* changed enormously (#35), and the
emulator's response to it is not required to be neutral at all.

### 55. The surface budget, properly accumulated **[measured]**

#46 and #50 read the coupler's fluxes once per emulator step, which compared a
6 h emulator mean against a single 30 min coupler sample. Fixed: the coupler
fields now accumulate every coupling step and are reported at the boundary.
ACE2-EAMv3, 10 days, 40 emulator steps, per unit covered area:

| | emulator | coupler | difference |
|---|---|---|---|
| turbulent (latent + sensible) | 118.48 | 142.73 | **+24.25** |
| net shortwave absorbed | 191.00 | 172.69 | **-18.31** |
| surface longwave up | 406.50 | 407.01 | +0.52 |
| **net surface, downward** | 20.78 | -22.29 | **-43.07** |
| TOA net, global | | | +12.84 |

Three results, one of them a correction to this review's own numbers.

**The turbulent figure survives**: 24.98 sampled, **24.25** accumulated. The
sampling error was 0.7 W/m2, because four evenly spaced samples a day integrate
a diurnal or semi-diurnal signal without bias. Worth fixing, but #50's
conclusion stands unchanged.

**The longwave is fine**: 0.52 W/m2 apart. A single accumulated step had shown
-29.6, which was itself a sampling artifact -- a good illustration of why the
fix was worth making, and a warning against reading any single step of this
report.

**The shortwave was never being measured at all.** The old report used the
emulator's own `FSDS - FSUS` for *both* columns, so the one term where EATM and
the surface models genuinely disagree was invisible. The coupler value is now
rebuilt as the surface models compute it, `sum over bands of
band * (covered_fraction - merged_albedo)` -- a fraction-weighted sum, not a
mean, confirmed at `prep_atm_mod.F90` where the merge reads
`x2a = ... + l2x*fracl`.

### 56. The shortwave gap is a surface-state disagreement, not a band split **[measured]**

Worth stating plainly because the obvious reading is wrong.

The emulator's own implied surface albedo is `FSUS/FSDS` = **0.136** (ACE2;
0.151 for SamudrACE). The coupled surface reflects like **~0.25**. Repartitioning
four bands moves absorbed shortwave by a few percent; it cannot bridge an albedo
gap of 0.11, and tuning the split to close it would hide a state error behind a
spectral knob while corrupting exactly the snow and ice response the split
exists to represent.

**About half is excess sea ice.** `mpassi` regional statistics for the ACE2
10-day run give a total ice area of 32.5 + 13.9 = **46.4 million km2** against
roughly 17-19 observed for January -- about 2.5x too much, from the
`cice_default` cold start (#27's last recommendation). Swapping ocean albedo
(~0.06) for ice (~0.65) over the excess ~5.6% of the globe raises the
covered-area albedo by ~0.05, so roughly half the 0.11 gap. **This is the
directly actionable part**: start from a spun-up G-case restart.

**The rest is structural and should not be "fixed" in the export path.** The
emulator has no albedo input channel; it predicts FSUS from its own learned
surface state and cannot know what MPAS-SI's ice looks like. EATM could rescale
the exported bands until the coupler's absorbed shortwave matched the emulator's
net, which would close the atmosphere's budget -- and would hand the ocean *too
much* sunlight, because the real ice is genuinely brighter than the emulator
believes.

**The ocean is currently getting the physically right shortwave.** The surface
models apply their own albedos to real downwelling bands, which is correct. The
inconsistency lives in the atmosphere, and it is the same object as the +12.8
(ACE2) and +16.3 (SamudrACE) W/m2 TOA imbalance: `FSUTOA` carries the same
albedo assumption that `FSUS` does. A checkpoint taking surface albedo as an
input forcing is the real fix, and it is not an EATM change.

### 57. SamudrACE is reproducible now **[fixed, verified]**

`eatm_rng_seed` (default 0) seeds libtorch through a small C++ shim. Two
1-day SamudrACE runs, identical configuration, same seed:

| compared | result |
|---|---|
| `eatm.r` restart, all channels | **53 of 53 bit-identical** |
| `mpaso` globalStats, all fields | **244 of 244 bit-identical** |

So the stochastic emulator now reproduces exactly, and #45's uninterpretable
6 W/m2 would not happen again. This was the top item on #47's list for a
reason: it is the difference between measuring a change and guessing at one.

Two things it does *not* do, both deliberate. It does not make a restart
bit-reproducible against a continuous run -- the generator state is not carried
in the restart file, only the seed is reapplied. And the seed applied is
`eatm_rng_seed + stepno`, not `eatm_rng_seed`: a resubmitted run re-initializes
each segment, and seeding identically there would replay one noise realization
per segment, which for 1-year segments is a spurious annual periodicity in the
model's own stochasticity.

Build notes for anyone touching this: `FTorchConfig.cmake` already does
`find_dependency(Torch)`, so `TORCH_INCLUDE_DIRS` and `TORCH_LIBRARIES` are in
scope at `build_model.cmake:354`; but `FTorch::ftorch` links only `stdc++`, so
the shim's references into `c10` have to be linked explicitly or the executable
fails with "DSO missing from command line".
