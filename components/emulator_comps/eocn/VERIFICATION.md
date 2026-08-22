# What was run, and what it showed

All on Perlmutter GPU, `pm-gpu` / `gnugpu`, 2026-08-22, one MPI task per
component.  Cases live in `$PSCRATCH/e3sm-repo/eocn-*`.

## 1. The traced model, before any Fortran

`tools/trace_eocn_model.py` checks the checkpoint's channel layout against the
table in `eocn_channels_mod.F90` and refuses to write a model whose corrector
came out inactive.  Driving the resulting `.pt` directly from Python on the
published initial condition gives, over the emulator's ocean mask:

| channel | min | max | mean |
|---|---:|---:|---:|
| `sst` (K) | 269.13 | 309.14 | 286.76 |
| `ssh` (m) | -1.25 | 0.97 | -0.04 |
| `salinityCoarsened_0` (g/kg) | 0.00 | 49.10 | 33.39 |
| `temperatureCoarsened_0` (degC) | -4.01 | 35.92 | 13.59 |
| `velocityZonalCoarsened_0` (m/s) | -1.01 | 1.08 | 0.00 |
| `ocean_sea_ice_fraction` | 0.00 | 1.00 | 0.27 |
| `iceVolumeTotal` (m) | 0.00 | 50.24 | 0.59 |

Without the input masking described in the README the same model returns
`sst` spanning 282-547 K with a 440 K mean, and 160 degC surface water.  That
is the single largest thing this component had to get right.

## 2. `E2000-EATM-EOCN`, 11 days

`gauss180x360_gauss180x360`, both emulators, everything else stubbed,
`ATM_NCPL = OCN_NCPL = 48`.  The run completes.  EOCN advances every 5 days as
it should — the log shows a flux window closing after exactly 240 coupling
steps — and the forcing means it hands the emulator are the right size and
sign (`TAUX` -0.014 N/m2, `FSDS` 127, `FLDS` 224, `FLUS` 271, `FSUS` 7.6,
`LHFLX` 95, `SHFLX` 31 W/m2, all as unweighted global means of the coupler's
ocean-side exchange).

Ocean-fraction-weighted global means from the coupler history:

| day | net (W/m2) | So_t (K) | So_s (g/kg) | Sa_tbot (K) |
|---|---:|---:|---:|---:|
| 1 | -26.8 | 286.615 | 33.359 | 283.341 |
| 5 | -31.9 | 286.830 | 33.383 | 283.938 |
| 8 | -58.9 | 287.315 | 33.425 | 284.135 |
| 11 | -80.9 | 288.115 | 33.489 | 284.585 |

**This is a drift, and it is worth taking seriously.**  The sea surface warms
1.5 K in 11 days while the net surface heat flux the ocean is being given goes
from -27 to -81 W/m2.  The emulated ocean is not responding to the flux it is
handed; it is following its own trajectory, and the atmosphere is responding to
that by losing more and more heat.  Two candidate causes, in the order I would
check them:

1. **No sea ice reaches the atmosphere.**  SamudrACE's ocean-to-atmosphere
   exchange is exactly `[ocean_sea_ice_fraction, sst]`.  With a stub ice
   component the coupler hands EATM `ICEFRAC = 0` everywhere, while Samudra
   internally predicts a global mean ice fraction near 0.2.  The polar ocean is
   presented to the atmosphere as open water at the freezing point, which is
   the wrong albedo and the wrong turbulent exchange over roughly a tenth of
   the globe.  This is the most likely driver and the most tractable fix.
2. **The five-day forcing lag.**  Each emulator step is driven by the mean over
   the window that just closed rather than the one it is predicting (see the
   README).  That is unavoidable online, but it is a real difference from the
   reference inference the checkpoint was validated against.

Neither is a defect in this component's plumbing; both are properties of
running the two halves of a coupled emulator through a coupler that does not
carry the channel they were trained to exchange.

## 2b. The same case, continued to 110 days

Continuing the run makes the shape of the problem clear.  The warming
**saturates and turns over**; the salinity does not.

| day | net (W/m2) | So_t (K) | So_s (g/kg) | Sa_tbot (K) | E (mm/d) | P (mm/d) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | -26.8 | 286.615 | 33.359 | 283.341 | 3.61 | 3.23 |
| 11 | -83.1 | 288.115 | 33.489 | 284.554 | 5.31 | 4.16 |
| 31 | -110.2 | 289.606 | 33.733 | 285.781 | 6.07 | 4.56 |
| 61 | -129.7 | 289.982 | 34.016 | 286.320 | 6.54 | 4.70 |
| 81 | -154.8 | 289.960 | 34.181 | 286.164 | 6.82 | 5.03 |
| 110 | -119.3 | 289.941 | 34.360 | 286.582 | 6.15 | 4.64 |

The sea surface settles about **+3.3 K** above its initial state by day 60 and
then stops: the coupled emulator finds a quasi-equilibrium, it is just the
wrong one.  Salinity, by contrast, rises monotonically by **1.0 g/kg in
110 days** and shows no sign of turning.

The freshwater budget says why.  Evaporation over the ocean goes from
3.6 to 6.2-6.8 mm/day — a latent heat flux around 180 W/m2 against an observed
global ocean mean near 90 — while precipitation rises only from 3.2 to
4.6 mm/day.  `E - P` therefore grows from +0.4 to about +1.7 mm/day, and with
no river runoff (`SROF` is a stub, worth perhaps 0.3 mm/day) there is nothing
to balance it.

That one number, the doubled evaporation, drives both drifts: it is most of the
-120 W/m2 net heat loss and all of the salinity rise.  It is a surface-flux
problem on the atmosphere side of the coupler, not something EOCN can fix.

So the ordering of things to fix, on this evidence, is: the missing sea ice
first (it is what exposes the polar ocean and inflates the turbulent fluxes),
then the surface-flux formulation, and only then anything about EOCN's own
5-day forcing lag.

## 3. Restart

Restarting the same case from the day-6 restart and running to day 12
reproduces days 7-11 to **3.05e-05 K**, which is one float32 ULP at 288 K — the
round trip of the emulator's `real(R4)` state through the `double` restart
file, and nothing else.  The global mean differs by 1e-8 K.

At day 12, the first history file written after an emulator advance that used
post-restart atmospheric fluxes, the difference grows to 0.11 K locally and
2.2e-04 K in the global mean.  That is the stochastic SamudrACE atmosphere:
its libtorch RNG is reseeded at the start of each segment, so the noise
realization after a restart is not the one a continuous run would have drawn.
EOCN's own bookkeeping — both bracketing states, the ten flux accumulators,
the accumulated-step count and the elapsed-time counter — round-trips exactly.

## 4. `F2010-ELM-EOCN` — builds and couples, then the atmosphere blows up

Prognostic EAM on ne30pg2 over the emulated ocean, with ELM, everything else
stubbed.  It builds, and it gets a long way:

* CIME resolves the grid, the maps and the domains;
* EAM and ELM initialise;
* the coupler's grid checks pass, and so does `seq_domain_check`'s fraction
  check — land plus ocean sums to one everywhere;
* EOCN initialises and hands the coupler the same physical ocean it does in the
  emulated case (`sst` 269-309 K, ocean-mean 286.57 K, ice fraction 0.19);
* EAM takes a dynamics step.

Then P3 aborts on the **first physics step** at column 19743, 88.94 N, 45 E —
the polar cap — with a temperature of -3.6e31 K through the whole lower half of
the column.

I read this as the missing sea ice, in its sharpest form: with a stub ice
component the coupler tells EAM that the Arctic Ocean is open water at the
freezing point under a -40 C atmosphere, which is an enormous exchange
concentrated in the smallest cells on the grid.

**That reading was wrong.**  It is recorded here rather than quietly deleted
because it is the kind of wrong that costs a day.  The story fit the symptom
exactly — polar cap, first physics step, sea ice obviously absent — and every
clause in it was individually true except the causal link.  Section 7 has the
actual cause: a hole in the atmosphere-to-ocean mapping weights at the poles,
which has nothing to do with sea ice.  Building EICE (section 6) did not fix
this crash and could not have.

The measurement that broke the story: `So_ssq`, `So_re` and `So_ustar` were
already at 4.6e40 in the coupler diagnostics.  Those are pure `xao` fields —
the atmosphere/ocean flux kernel writes them and no ice component ever touches
them — so whatever was wrong sat upstream of the ice, where an ice component
cannot reach.

### What it took to get CIME this far

Four changes, all on this branch: `eocn` as a valid `-ocn` value and an
`ieflx_opt` default for EAM; `gauss180x360` as a valid mask for ELM; naming the
atmosphere side of the gridmap `ne30np4.pg2` rather than the `ne30pg2` alias
(without which every map silently fell back to `idmap`); and — the one that
took longest — making EOCN's domain **fraction** binary.

That last one is worth recording.  When the atmosphere and ocean grids differ,
`seq_domain_mct.F90:301` builds the ocean fraction on the atmosphere grid by
mapping the ocean domain's *mask*, not its *fraction*, and then requires it to
equal one minus the land model's fraction.  EOCN originally reported the
emulator's continuous sea surface fraction as its domain frac, which disagrees
with a binary mask on every coastal cell and fails that check.  Reporting the
mask itself fixes it and is what the coupler's bookkeeping assumes throughout.

## 5. What Samudra's sea ice is worth, and why the atmosphere-side fix fails

Samudra predicts sea ice: `ocean_sea_ice_fraction` and `iceVolumeTotal` are two
of its 80 output channels, and they are a credible field -- NH 13.9e6 km2,
SH 7.0e6 km2 in early January, mean thickness 1.9 m, fraction above 0.95
poleward of 80N tapering to zero by 50 degrees.

ACE's own coupler (`fme/coupled/stepper.py`, `CoupledOceanFractionConfig` ->
`OceanData.ocean_fraction`) uses it as

    ICEFRAC = ocean_sea_ice_fraction * (1 - LANDFRAC)
    OCNFRAC = max(1 - LANDFRAC - ICEFRAC, 0)

That identity holds to float32 in the published initial conditions, and it is
the same identity E3SM writes as lfrac + ifrac + ofrac = 1.  The difference is
only that E3SM fills ifrac from a sea ice *component*, so with `SICE` the
atmosphere is told the polar ocean is open water.

`eatm_icefrac_from_ocn` (default off) applies ACE's split verbatim, using a
side channel from EOCN (`shr_emul_ice_mod`).  Over 11 days against the same
case with the flag off:

| day | net (no ice) | net (ICEFRAC on) | delta | So_t delta |
|---|---:|---:|---:|---:|
| 2 | -26.8 | -35.0 | -8.1 | 0.000 |
| 6 | -34.2 | -50.6 | -16.4 | 0.000 |
| 9 | -62.6 | -82.1 | -19.6 | +0.002 |
| 12 | -83.1 | -91.6 | -8.5 | +0.008 |

**It gets worse, not better** -- and the latitude breakdown says why (day 6):

| band | Sa_tbot delta | net flux delta |
|---|---:|---:|
| NH polar | -3.52 K | -69.1 W/m2 |
| mid / tropics | -0.17 K | -6.6 W/m2 |
| SH polar | -0.39 K | +1.4 W/m2 |

The atmosphere does the right thing: told there is ice, ACE cools the Arctic
near-surface air by 3.5 K.  The coupler does not.  `domo_frac` is unchanged,
`fractions_a(ifrac)` is still zero, and the merge and the bulk flux scheme
still hand the whole polar ocean to that colder atmosphere as open water at
271 K.  A larger air-sea gradient over an unchanged exposed area is 69 W/m2 of
extra heat loss.

Two things follow.  The ice fraction is a **first-order** term, not a detail.
And a half-closed loop is worse than none: the fraction has to enter the
coupler's own bookkeeping, so the ice-covered share is removed from the ocean's
exposure and given ice albedo and ice fluxes.  That means a sea ice component
reporting `Si_ifrac`, not an atmosphere-side substitution.  `shr_emul_ice_mod`
and `eatm_icefrac_from_ocn` exist to establish this number and should be
deleted once that component exists.

### A trap worth recording

The first attempt keyed the split off `fl`, the coupler's `Sf_lfrac`.  With a
stub land that is zero everywhere -- the real land fraction reaches EATM as the
*deficit* -- so the deficit collapsed to zero, `LANDFRAC` went to zero globally
and `TS` to 0 K over every continent.  It silently reverted the
`eatm_land_deficit` fix and cost 34 W/m2 that had nothing to do with sea ice.
Anything reconstructing fractions inside EATM must key off `fl + deficit`.

## 6. Closing the loop: EICE

Section 5 ended with a prediction -- that the ice fraction has to enter the
coupler's own bookkeeping rather than be substituted inside the atmosphere.
`components/emulator_comps/eice/` does that.  It reports `Si_ifrac` and lets
`seq_frac_mct.F90:651` compute the fractions, which is SamudrACE's identity
written in E3SM's variables.  Everything else it reports -- albedos, surface
temperature, atm/ice turbulent fluxes -- is `dice` in prescribed mode.

Run: `E2000-EATM-EOCN-EICE`, `gauss180x360_gauss180x360`, 11 days, 1 task,
against the two references from section 5.

### The coupler now carries the ice

Every `seq_domain_check` difference between the ice and ocean domains is
exactly zero, because EICE takes its mesh from EOCN rather than reading one:

```
(seq_domain_check_grid) maximum difference for mask 0.00000000000000
(seq_domain_check_grid) maximum difference for lat  0.00000000000000
(seq_domain_check_grid) maximum difference for lon  0.00000000000000
(seq_domain_check_grid) maximum difference for area 0.00000000000000
```

At day 6, weighted per unit sea surface, with `eatm_icefrac_from_ocn = .false.`
so nothing is being substituted inside the atmosphere:

| | ice fraction | `Sx_t` handed to the atm |
|---|---:|---:|
| | *global / NH polar* | *global / NH polar* |
| no ice | 0.000 / 0.000 | 291.7 K / 272.7 K |
| ice to the atmosphere only | 0.000 / 0.000 | 291.7 K / 272.7 K |
| EICE | 0.174 / 0.743 | 287.1 K / 260.4 K |

An Arctic ice fraction of 0.74 in January is credible, and the surface
temperature the coupler merges for the atmosphere falls 12.3 K in that band.
The middle row is the section 5 result restated: substituting the fraction
inside EATM leaves the coupler's own fractions at zero, which is why it made
things worse.

### What the ice costs the ocean's forcing, and why it is divided back out

Adding an ice component changes what the ocean receives.
`prep_ocn_mod.F90:1218` builds every `Foxx_*` as
`afrac*<atm flux> + ifrac*<ice flux>`, with `afrac` and `ifrac` renormalised by
their sum -- on this grid exactly `1 - ocean_sea_ice_fraction`.

Samudra was not trained on that.  In SamudrACE the ocean receives the
atmosphere's whole-cell fluxes and accounts for its own ice internally;
`ocean_sea_ice_fraction` is one of its prognostic outputs, not a boundary
condition it is handed.  Letting the coupler scale the forcing down applies
that insulation a second time, in precisely the cells where it is largest.
`eocn_flux_ifrac_unweight` (default `.true.`) divides it back out.

Global means of the two channels Samudra actually consumes, over its two 5-day
flux windows:

| | FSDS window 1 | FSDS window 2 | FLDS window 1 | FLDS window 2 |
|---|---:|---:|---:|---:|
| no ice (reference) | 123.1 | 126.9 | 224.4 | 222.8 |
| EICE, unweight `.true.` | 121.3 | 123.8 | 204.7 | 197.8 |
| EICE, unweight `.false.` | 107.3 | **98.2** | 183.1 | **160.0** |

With the un-weighting off, the shortwave Samudra sees falls 22 % below the
no-ice reference by the second window and the longwave 28 %, and the gap
*widens* between windows -- the double-counted cooling feeds back through more
ice and more attenuation.  With it on, the shortwave stays within 3 %.  The
residual longwave decrease is not a weighting artefact: it is the atmosphere
genuinely emitting less downward longwave now that it is colder over the ice,
which is the response the whole exercise was after.

Set `eocn_flux_ifrac_unweight = .false.` for an ocean model that does *not*
carry its own ice, where the coupler's open-water split is the correct physics.

### Still open

`Si_t` is fabricated from a seasonal cycle (`dice_comp_mod.F90:670`) because
Samudra carries a fraction and an ice volume but no surface energy balance.
That is the weakest number in the component and the first thing to improve if
the polar atmosphere still looks wrong.  `iceVolumeTotal` is predicted and
currently unused; a thickness-dependent conduction term is the obvious next
step.

`shr_emul_ice_mod` now carries the grid handshake as well as the diagnostic
fraction, so it is no longer purely dead weight -- but `eatm_icefrac_from_ocn`
and `shr_emul_ice_get`/`_put` are, and should go once nobody needs to
reproduce section 5.

## 7. Why `F2010-ELM-EOCN` really blew up: 552 unmapped cells at the poles

The crash in section 4 is not about sea ice.  It is about four cells.

### The evidence

P3 reports the columns it is unhappy with.  Across the whole run there are
exactly three of them — gcol 19740, 19743 and 19858 — and all three sit at
latitude 88.9394 N, longitudes 315, 45 and 225.  Those are three of the four
ne30pg2 cells in the northernmost row.  Nothing else on the globe is wrong.
The temperature is garbage through every one of the 72 levels, from the
surface to the model top, on the first physics step.

A failure that is confined to the polar cap and total within it is a geometry
problem, not a physics problem.

### The cause

`ATM2OCN_SMAPNAME` and `ATM2OCN_VMAPNAME` — the maps that carry the
atmosphere's *state* to the ocean grid — were bilinear.  ne30pg2 cell centres
stop at |lat| = 88.94.  The Gaussian 180x360 grid has two rows beyond that in
each hemisphere, at |lat| = 89.2366.  Bilinear interpolation cannot
extrapolate, so those 552 destination cells get no weights at all, and
`ESMF_RegridWeightGen` was invoked with `--ignore_unmapped`, which turns that
into silence rather than an error:

```
map_ne30pg2_to_gauss180x360_trbilin.20260822.nc
  frac_b: 552 cells == 0, all at lat = ±89.2366
```

So `a2x_ox` arrives at the polar Gaussian rows as exact zeros — zero bottom
height, zero air density, zero temperature.  `seq_flux_atmocn_mct` then runs
there, because those cells are ocean and its only guard is the ocean mask
(`seq_flux_mct.F90:1553`).  It takes `log(zbot/zref)` and divides by `rbot`.
Both are undefined at zero.

`xao_ox` is now poisoned on the polar ocean cells.  The ocean-to-atmosphere
map is conservative and covers those cells completely (`frac_b = 1.0` at all
four polar ne30pg2 cells — checked), so it delivers the poison, undiluted, to
exactly the four columns P3 named.

The whole chain is four steps and every step is silent.

One more thing makes the attribution airtight rather than merely plausible.
The hole exists in *both* hemispheres — 552 cells, 276 per pole — but Samudra's
ocean mask is not symmetric:

```
lat =  89.2366   360 cells   360 ocean     (Arctic)
lat = -89.2366   360 cells     0 ocean     (Antarctica)
```

`seq_flux_atmocn_mct` only evaluates where the ocean mask is set, so the
southern hole is never touched and the northern one always is.  If the
unmapped weights were the cause, the crash had to be northern-only.  Every
failing column was northern.  A story that merely fit the symptom would not
have predicted that asymmetry in advance.

### Why the emulated pair never saw it

Because it never regrids.  `E2000-EATM-EOCN[-EICE]` puts the atmosphere and
the ocean on the *same* grid:

```
a%gauss180x360_l%null_oi%gauss180x360 ...
  ocn2atm_fmapname: "idmap"
  atm2ocn_smapname: "idmap"
```

Every map in that compset is the identity, so neither this bug nor the land
contamination of section 4 can exist there at all.  Sections 2, 2b, 3, 5 and 6
are unaffected and their numbers stand as measured.

This is worth stating plainly because the opposite is easy to assume: the
emulated pair is not a weaker test that happened to miss these bugs, it is a
configuration in which they are structurally impossible.  Everything in
section 7 is the price of putting a prognostic atmosphere on its own native
grid, and it is paid the moment the two grids differ.  The Python reference
implementation in the `ace` repository shares that immunity for the same
reason — both halves are tensors on one grid — so it cannot be used to catch
this class of problem either.

The second-order reason, which would have mattered had the grids differed:
EATM does not read `Faxx_*`.  It diagnoses its own surface fluxes from its own
state, so poisoned flux fields would pass through it unnoticed.  EAM reads
them.

This is the third time on this branch that the same shape of bug has appeared:
a routine evaluated on cells where its inputs were never written.  EICE had it
at initialisation (section 6), the ocn-to-atm map had it over land, and the
atm-to-ocn map has it at the poles.  Emulator components make it worse than
usual, because an emulator that ignores a field cannot warn you that the field
is wrong.

### The fix

Regenerate the state/vector map with nearest-neighbour extrapolation, and drop
`--ignore_unmapped` so a future hole is an error:

```
ESMF_RegridWeightGen -s ne30pg2_scrip -d gaussian_180x360_latlon.scrip \
  -m bilinear --src_type SCRIP --dst_type SCRIP \
  --extrap_method neareststod \
  -w map_ne30pg2_to_gauss180x360_trbilin.20260822b.nc --netcdf4
```

`frac_b` is now 1.0 for all 64800 destination cells.  `config_grids.xml` points
`ATM2OCN_SMAPNAME` and `ATM2OCN_VMAPNAME` at the new map; the ice maps follow
the ocean ones because the two components share a grid.

### What to check when adding a new emulator grid

Any lat-lon emulator grid whose outermost row lies poleward of the atmosphere's
outermost cell centres has this problem, and every lat-lon grid does — a
Gaussian or regular latitude grid always places a row nearer the pole than a
cubed-sphere grid's outermost centres.  Before running anything prognostic:

```python
m = nc.Dataset(mapfile); fb = m.variables['frac_b'][:]
print(mapfile, (fb == 0).sum())
```

on every map in `seq_maps.rc` that is not `idmap`.  It takes seconds and would
have saved all of section 4.

Read the answer with the mask in mind, though — a zero is only a hole if the
destination cell expects coverage.  For this grid pair:

| map | zeros | verdict |
| --- | --- | --- |
| `map_ne30pg2_to_gauss180x360_traave` | 0 | correct |
| `map_ne30pg2_to_gauss180x360_trbilin.20260822b` | 0 | correct, after the fix |
| `map_gauss180x360ocn_to_ne30pg2_traave` | 4801 | **correct** |

The 4801 are ne30pg2 cells lying entirely over land.  They *should* receive no
ocean, and masking the source grid to produce exactly that is what section 4's
other fix was for.  What confirms they are right rather than merely tolerated
is `seq_domain_check`: `ofrac` on the atmosphere grid agrees with one minus
ELM's land fraction to 8.9e-15.  A hole that is real shows up there as a
fraction mismatch; a hole that is correct does not.

## 8. The restart write only worked because the test used serial netcdf

With the map fixed, `F2010-ELM-EOCN-EICE` integrated all 96 steps of its two
days, conserving energy, with no scanner hits from either emulator.  Then it
aborted:

```
PIO: FATAL ERROR: Attached buffer is too small.
  (file = eocn-f2010ice-4n.eocn.r.0001-01-03-00000.nc)
```

EOCN is serial, so it writes whole global fields with `pio_put_var` rather
than a distributed `pio_write_darray`.  Under pnetcdf those become buffered
puts that are not flushed until the file closes, and this restart carries both
bracketing emulator states — 89.7 MB — which is over the default limit.

Section 3's restart test did not catch this because the emulated case runs
`PIO_TYPENAME = netcdf`, whose serial path has no attached buffer, while the
F2010 case inherits `pnetcdf`.  The restart code was never wrong; it had
simply never been executed on the path that fails.  Worth remembering when a
component is only ever exercised in one compset: the compset carries settings
that are part of what is being tested, whether or not anyone chose them.

`eocn_restart_file_write` now raises the limit for the write and restores it
afterwards, following `homme/src/common_io_mod.F90:123`, rather than leaving
it raised for every other component sharing the library.
