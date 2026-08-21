# The hypothesis, and how to falsify it

**H.** The improvement of `mahf708/jonbob/add-eatm` over `23dd0c1b97` is carried
by exactly two edits pointing in opposite directions — the land-fraction
reconstruction (+144.01 W/m²) and the surface-diagnostics package
(−126.02 W/m²) — plus the diurnal shortwave (+12.11). Every other code commit on
the branch is worth ≤1 W/m² on the ocean's heat budget in this configuration, or
is inert in it entirely. The baseline's −57.8 W/m² was the partial cancellation
of the first two.

Five predictions, ordered by how much they would cost me if wrong.

### P1 — the branch reproduces the existing production run
A multi-year run of `mahf708/eatm/measured-fixes` matches
`GMPAS-EATM-ACE2-EAMv3-2yr` to run-to-run noise: year-2 ocean heat drift
≈ −1.4 W/m², SST bias ≈ −0.30 K, SST R/sd ≈ 0.12, barotropic streamfunction
R/sd ≈ 0.08. **Falsified if any of those moves materially** — the branch is
code-identical to what produced that run, so a difference means the replay lost
something the byte-comparison did not catch.

### P2 — the cold pool is persistent, not transient
`eatm_land_deficit = .false.` over a model year holds >25% of area below 240 K
and a TOA imbalance >50 W/m², rather than annealing. Confirmed at 10 days;
this predicts persistence. **Falsified if the pool shrinks below ~10% by month
three.**

### P3 — the two fixes stay a pair at multi-year scale
`base_plus_land` and `base_plus_surface`, run a year each, both end further from
the JRA control than `ctrl` does on ocean heat drift *and* on latent heat flux
bias. **Falsified if either single-fix configuration beats the pair.** This is
the prediction the handoff advice rests on.

### P4 — the risky one: 10-day ranking does not survive
On the 10-day metric `base_plus_land` (+86.72) looks worse than `all_off`
(−57.29). I predict the *opposite* at two years: `base_plus_land` equilibrates
closer to the control, because its TOA imbalance is already repaired (14.6
against 53.8) and the archive shows exactly this inversion between `gnugpu`
(−94.05 at 10 days, −2.84 at year 2) and `test4naser` (−57.82, −35.52).
**Falsified if `base_plus_land` is still the worse of the two at year 2.**
This is the prediction most likely to be wrong, and the cheapest way to learn
whether the 10-day instrument can be trusted for ranking at all.

### P5 — a fixed 10 m export height helps ACE2
`gnugpu` exported a flat 10 m and reached −2.84 W/m² in year 2; HEAD exports
~428 m for ACE2 because `eatm_surface_layer='near_surface'` silently falls back
to `lowest_level` without `Tat2m`/`Qat2m`. Allowing a fixed height for emulators
without near-surface channels cuts the year-2 latent heat flux bias (−15.1 W/m²
against the control) by more than half. **Falsified if it moves latent bias by
less than 5 W/m².** Requires one namelist flag and a rebuild; untested.

---

## What would make the whole hypothesis wrong

The instrument is a 10-day ocean heat drift, and the archive proves it can
mis-rank configurations relative to their equilibrium. The decomposition is
therefore a statement about **bias at ten days**, cross-checked against
independent diagnostics that do not share that failure mode — the TOA imbalance,
the in-run turbulent and shortwave interface budget, the exported surface
temperature field, and sea-ice area. If a multi-year run shows the +144/−126
pair does not hold at equilibrium, the *mechanism* still stands (the cold pool
and the collapsed exchange coefficients are directly observed, not inferred) but
the *magnitudes* do not, and the budget above should be re-derived from year-2
means rather than 10-day drift.
