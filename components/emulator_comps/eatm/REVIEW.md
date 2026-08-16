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
use to compute evaporation and latent heat flux, so a saturated bottom layer
suppresses evaporation almost everywhere and can reverse its sign wherever
`T_bot > SST`. For a run whose whole point is a prognostic ocean, this is the
most consequential error in the coupling.

The emulator predicts the field: `specific_total_water_7` (ACE2-EAMv3) /
`STW_7` (SamudrACE-E3SMv3) is the lowest-layer specific total water. That is now
what is exported. Using total water slightly overstates vapour where there is
condensate, which is a much smaller error than assuming saturation.

Scale of the error, from the published initial conditions: the global mean of
`STW_7` is 8.3e-3 kg/kg (and `Qat2m` 9.1e-3), whereas `q_sat` at the lowest
layer's temperature and pressure is around 1.2e-2 — roughly 50% too moist in
the mean, and much worse in dry subsidence regions where the real RH is low and
the evaporative demand is largest.

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
explicitly wherever EATM results are compared. Making it reproducible would
mean exposing `torch::manual_seed` through FTorch and saving/restoring the RNG
state in the EATM restart.

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
