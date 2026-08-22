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

I read this as the missing sea ice, in its sharpest form.  With a stub ice
component the coupler tells EAM that the Arctic Ocean is **open water** at the
freezing point under a -40 C atmosphere.  That is a physically enormous
turbulent and radiative exchange, concentrated in the smallest cells on the
grid, and a 30-minute physics step does not survive it.  The emulated
atmosphere does not blow up in the same configuration only because it does not
integrate anything — it drifts instead (section 2b).

So this compset is blocked on the same gap as the drift, and closing that gap
is the prerequisite for a prognostic atmosphere over this ocean, not an
optimisation.  Two things I would check before anything else:

1. rerun with a sea ice component (or the pass-through described in the README)
   so `ICEFRAC` is not zero at the pole;
2. failing that, a shorter atmosphere physics step, to confirm the blow-up is
   the surface exchange rather than something structurally wrong in the merge.

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
