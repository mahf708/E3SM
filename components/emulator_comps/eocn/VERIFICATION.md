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

| date | swnet | lwdn | lwup | latent | sensible | net | So_t (K) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 01-02 | 185.9 | 331.6 | -387.6 | -104.3 | -43.9 | -18.3 | 286.855 |
| 01-05 | 188.8 | 327.2 | -388.8 | -118.2 | -35.2 | -26.2 | 287.073 |
| 01-08 | 194.8 | 325.0 | -391.7 | -138.2 | -41.1 | -51.2 | 287.603 |
| 01-12 | 196.6 | 325.9 | -396.5 | -157.1 | -49.7 | -80.7 | 288.482 |

**This is a drift, and it is worth taking seriously.**  The sea surface warms
1.6 K in 11 days while the net surface heat flux the ocean is being given goes
from -18 to -81 W/m2.  The emulated ocean is not responding to the flux it is
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

## 2b. The same case, continued to 111 days

Continuing the run makes the shape of the problem clear.  The warming
**saturates and turns over**; the salinity does not.

| day | net (W/m2) | So_t (K) | So_s (g/kg) | Sa_tbot (K) |
|---|---:|---:|---:|---:|
| 1 | -18.3 | 286.855 | 33.527 | 283.642 |
| 11 | -82.7 | 288.482 | 33.667 | 284.746 |
| 31 | -121.3 | 290.148 | 33.939 | 286.017 |
| 61 | -129.3 | 290.651 | 34.252 | 286.535 |
| 81 | -157.3 | 290.658 | 34.432 | 286.497 |
| 111 | -131.4 | 290.511 | 34.653 | 286.865 |

The sea surface settles about **+3.8 K** above its initial state by day 70 and
then eases back very slightly: the coupled emulator finds a quasi-equilibrium,
it is just the wrong one.  Salinity, by contrast, rises monotonically by
**1.1 g/kg in 111 days** and shows no sign of turning.

The freshwater budget says why:

| | evaporation | precipitation | E - P |
|---|---:|---:|---:|
| day 1 | 3.60 mm/day | 3.27 mm/day | +0.33 |
| day 111 | 6.67 mm/day | 4.88 mm/day | +1.78 |

Evaporation over the ocean nearly doubles, ending at a latent heat flux of
193 W/m2 against an observed global ocean mean near 90.  That single number
drives both drifts — it is most of the -130 W/m2 net heat loss and all of the
salinity rise — and it is a surface-flux problem on the atmosphere side of the
coupler, not something EOCN can fix.  The absent river runoff (`SROF`) accounts
for perhaps 0.3 mm/day of the freshwater imbalance; the rest is evaporation.

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
post-restart atmospheric fluxes, the difference grows to 0.12 K locally and
1.4e-04 K in the global mean.  That is the stochastic SamudrACE atmosphere:
its libtorch RNG is reseeded at the start of each segment, so the noise
realization after a restart is not the one a continuous run would have drawn.
EOCN's own bookkeeping — both bracketing states, the ten flux accumulators,
the accumulated-step count and the elapsed-time counter — round-trips exactly.

## 4. `F2010-ELM-EOCN`

Prognostic EAM on ne30pg2 over the emulated ocean, with ELM, everything else
stubbed.  Getting CIME to accept the configuration needed four changes, all in
this branch: `eocn` as a valid `-ocn` value and an `ieflx_opt` default for EAM,
`gauss180x360` as a valid mask for ELM, and naming the atmosphere side of the
gridmap `ne30np4.pg2` rather than the `ne30pg2` alias (without which every map
silently fell back to `idmap`).  Maps and domains were generated for this
branch; see the README.
