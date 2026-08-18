# EATM coupling: state of play, open problems, and what to do next

Handoff note. Self-contained: you should be able to act on this without reading
`REVIEW.md`, which is a 2900-line chronological log. Findings referenced as
`#NN` are sections in `REVIEW.md`; read those for the measurements behind any
claim here.

Written 2026-08-18, after two complete 2-year coupled runs and a full
MPAS-Analysis comparison against a JRA1p5-forced control.

---

## TL;DR

The coupling **works**: a 2-year `GMPAS-EATM` run with the ACE2-EAMv3 emulator
tracks a JRA-forced control to 0.01 K/yr of global SST drift and closes its
ocean heat budget to within 1.7 W/m2 of that control.

It is **not structurally correct**. One design flaw, one upstream bug, and a
set of emulator-skill problems that are not coupling issues at all. Only the
first is getting worse with integration time.

Priority order:

1. **Flux-anomaly coupling** (design change) -- the only *growing* error.
2. **`prep_atm_mod` land-fraction fix** (small, upstream) -- currently worked
   around inside EATM.
3. Emulator skill (winds, precip partitioning, tropical shortwave) -- belongs
   to whoever trains the next checkpoint, not to the coupling.

---

## 1. The structural flaw: fluxes are computed twice

**What happens now.** The emulator predicts surface turbulent fluxes directly --
it was trained to. EATM exports an atmospheric *state* (`Sa_tbot`, `Sa_shum`,
`Sa_u`, `Sa_v`, `Sa_z`, ...). The coupler then **discards the emulator's
fluxes** and re-derives its own from that state using `shr_flux_atmOcn`
(`share/util/shr_flux_mod.F90`). The ocean and ice see the coupler's numbers.

**Why it cannot be tuned away.** For the two calculations to agree, the exported
state would have to be simultaneously consistent with the emulator's latent
*and* sensible flux under the coupler's bulk formula. It cannot be, because
scheme 0 gives heat and moisture different, stability-dependent transfer
coefficients (`shr_flux_mod.F90:384-386`):

```fortran
rhn = (1.0_R8-stable) * 0.0327_R8 + stable * 0.018_R8   ! heat,     Stanton
ren = 0.0346_R8                                          ! moisture, Dalton
```

C_H/C_E is 0.945 unstable and 0.520 stable, so no single `Sa_z` nulls both
(#65). This was tested exhaustively:

- The mismatch nulls for latent near 29 m and never nulls for sensible (#63).
- A reference height of 44 m zeroes the *global* mean but only by regional
  cancellation, spanning -13.2 to +12.6 W/m2 across latitude bands, and the
  cancellation degrades with integration time (#66, #69).
- `eatm_ref_height` is therefore left at **10 m**, the honest value, with the
  mismatch reported as a known cost (#69).

**Magnitude and persistence.** ~+21 W/m2 turbulent mismatch in-run (#67);
-15 W/m2 latent bias against the JRA control, holding across both model years
(#72). It does not spin up and it does not anneal.

**Why it now matters more than we thought.** ACE2's latent bias by latitude,
years 1-2, W/m2 against control:

| band | latent | sensible | shortwave |
|---|---|---|---|
| 0-30S | **-19.6** | +5.3 | +6.9 |
| 0-30N | **-20.5** | +4.2 | +16.6 |
| 30-60N | -13.1 | +4.1 | +6.2 |
| 60-90N/S | ~-0.7 | ~+1.6 | ~-1.8 |

The excess evaporative loss peaks in exactly the two bands that are drifting
cold. The tropics are already 1.3 K cold and cooling at 0.39 K/yr (#72), and a
*colder* ocean should evaporate *less* -- so 20 W/m2 of excess latent loss out
of a too-cold ocean is the formulation forcing it, not the ocean state. The
tropical net flux is now near zero, but only because SST fell far enough for
the excess evaporation to balance an offsetting +16.6 W/m2 shortwave excess.
**That is a cold equilibrium held by two large compensating errors, and it is
still moving.**

### Proposed fix: flux-anomaly coupling

Pass the emulator's own fluxes, corrected by a bulk-formula term for the
difference between the SST the emulator implies and the SST MPAS-O actually
has.

- Preserves the SST feedback. This is why re-deriving in the coupler was the
  right instinct in the first place: passing raw emulator fluxes would hand the
  ocean fluxes computed for a *different* SST, breaking the coupled feedback
  loop. Do not do that.
- Respects what the emulator was trained to produce. There is direct evidence
  the emulator's fluxes are the trustworthy half: it reproduces the E3SMv3
  training flux-state relationship closely, implied height 10.4 m against
  12.4 m (#63-64). The error is introduced by the re-derivation.

**Not prototyped.** This is a non-trivial change to the export and merge path
and deserves a design discussion before code. Open questions: where the
correction is applied (EATM export vs `prep_*_mod`); how to treat ice-covered
cells; whether the same treatment is needed for momentum.

---

## 2. Upstream bug: the coupler's land fraction

`x2a_Sf_lfrac` arrives **0.0 globally** and `x2a_Sx_t` is **exactly 0 K over
25.9% of area**. EATM reconstructs the deficit on import (#25, commit
`67d71a2a09`), which is why the runs work -- but the driver still ships the
wrong fields.

Before that reconstruction the emulator was being fed a state where 30% of the
globe was below 240 K; fixing it moved `Sa_tbot` 259.4 -> 282.3 K and
`Sa_lwdn` 251 -> 331 W/m2, and repaired the general circulation. It is by a wide
margin the single highest-value change on this branch.

**Still to do:** fix `prep_atm_mod` so the coupler delivers correct fields.
Matters for any ELM configuration, and for anyone reading `cpl.hi` naively.

---

## 3. Not coupling problems

Do not attempt to fix these in the coupling layer; they are emulator skill and
need a different checkpoint.

| symptom | metric | note |
|---|---|---|
| wind stress curl | R/sd **0.83** | drives `ssh`, which is *degrading*: bias -13.3 -> -25.0 over year 1 -> 2. Barotropic streamfunction still matches at 0.08, so the wind error has begun moving mass but has not yet reorganised the gyres. Watch this on longer runs (#70, #72). |
| snow flux | R/sd **1.70** | worst-reproduced field; 1.96 in the Arctic. Difference exceeds the control's own structure. Commit `88c14c5790` already fixed a precip channel unit error (kg/m2/s vs m/s); the rain/snow partitioning deserves the same scrutiny. |
| tropical shortwave | **+16.6 W/m2** (0-30N) | half of the compensating pair holding the cold tropics in place. |

`R/sd` = area-weighted RMS difference against the JRA control, normalised by the
control field's own spatial standard deviation. Under 0.3, the pattern is
essentially reproduced; near 1, the disagreement is as large as the real
structure.

---

## 4. Recurring trap: global agreement hides regional error

This has now bitten four separate times. Do not accept a global mean as
evidence of correctness:

- A 44 m reference height zeroes the global turbulent mismatch by regional
  cancellation spanning 26 W/m2 (#66).
- ACE2's LWdn bias is +1.7 W/m2 globally with RMSE 14.8 (#70).
- ACE2's `mld` bias is -0.01 m globally with RMSE 12.5 (#72).
- ACE2's global SST drift is 0.01 K/yr while its tropics cool 0.39 K/yr,
  cancelled by polar warming (#72).

`deltaOHC_0-700m` makes the same point structurally: R/sd **0.88** while the
global heat budget matches to under 2 W/m2. Always check the latitude-band
breakdown -- `ace_flux_budget_report` prints one in-run, and the MPAS-Analysis
climatologies support one offline.

---

## 5. Reference runs and how to reproduce the comparison

| run | what it is | drift vs control |
|---|---|---|
| `GMPAS-EATM-ACE2-EAMv3-2yr` | ours, present-day (~2010), AMIP-trained emulator | **-0.010 K/yr** |
| `GMPAS-EATM-SamudrACE-E3SMv3-2yr` | ours, **piControl 1850**, coupled-trained emulator | -0.759 K/yr |
| jonbob `GMPAS-JRA1p5-2023` | JRA-forced control, present-day | -- |
| anolan `GMPAS-EATM-gnugpu` | Apr 2026 code, pre-land-fix | -0.766 K/yr |
| anolan `GMPAS-EATM-test4naser` | **exactly this branch's parent commit** | +0.505 K/yr |

**Do not compare SamudrACE to the control naively.** It is piControl (1850)
against a present-day control, so part of its cold SST, low LWdn and high LWup
is the *correct* answer for its forcing epoch, not error. The tell is that its
latent and sensible biases carry the *wrong sign* for a colder world -- those
are real. Its latent bias also shrinks (-19.5 -> -13.6) as it cools, the
signature of equilibration rather than a worsening defect (#70, #72). An earlier
claim that its LWdn deficit was "a specific actionable bug" was retracted for
this reason.

`4naser` is the direct before/after for this branch's 49 commits: it fails
bipolar, holding ~2x the control's Arctic ice (21.8 vs 11.3, 1e12 m2) while its
Southern Ocean ice collapses to 8.3 vs 11.8.

Analysis configs, driver and submit script: `/pscratch/sd/m/mahf708/mpas_analysis/cfgs/`.
Published sites: `https://portal.nersc.gov/cfs/e3sm/mahf708/<casename>/`.
`REVIEW.md` #71 documents the MPAS-Analysis setup and several traps, including
two that fail *silently* on single-year passes.

---

## 6. What was already fixed (do not re-litigate)

- Land-fraction reconstruction on import (#25) -- the highest-value change.
- Shortwave diurnal disaggregation, `eatm_sw_diurnal` -- the emulator emits an
  interval-mean shortwave while surface albedo is instantaneous, so 13.2% of
  ocean shortwave was landing on night cells and being multiplied by zero.
  Worth +12.8 W/m2 (#35 as repaired).
- Surface-layer export, `eatm_surface_layer = 'near_surface'` -- hand the
  coupler a 2 m/10 m state rather than the lowest model level (~450 m). Worth
  +10.1 W/m2 (#51, #58).
- SOLIN as the interval mean the emulator was trained on (#35).
- RNG seeding via `eatm_rng_seed` so stochastic emulators are reproducible and
  A/B tests mean something (#57).
- `eatm_ref_height` stays at **10 m**. This was settled with a full month of
  runs at both 10 m and 44 m; see #63-#66 and #69 before reopening it.
