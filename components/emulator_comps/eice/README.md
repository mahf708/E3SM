# EICE -- the sea ice half of the emulated ocean

EICE is not a sea ice model.  It is the piece of bookkeeping that lets E3SM's
coupler see the sea ice the *ocean* emulator already predicts.

Samudra outputs `ocean_sea_ice_fraction` alongside its SST, salinity and sea
surface height.  In SamudrACE that fraction is what the coupler hands the
atmosphere:

```
ICEFRAC = ocean_sea_ice_fraction * (1 - LANDFRAC)
OCNFRAC = max(1 - LANDFRAC - ICEFRAC, 0)
```

E3SM writes the same identity as `lfrac + ifrac + ofrac = 1`, but it fills
`ifrac` from a sea ice *component*.  With `SICE` there is none, so the coupler
reports the polar ocean as open water at the freezing point all year.
`VERIFICATION.md` section 5 measures what that costs and shows that patching it
inside the atmosphere makes things worse.  EICE is the fix.

## What it does

Reports `Si_ifrac = ocean_sea_ice_fraction` and lets `seq_frac_mct` do the
arithmetic.  No conversion is involved: `Si_ifrac` is already defined as a
proportion of the ice domain's fraction, which is the same thing
`ocean_sea_ice_fraction` is a proportion of.

Everything else -- albedos, surface temperature, the atm/ice turbulent fluxes
-- is `dice` in prescribed mode, which is what an AMIP run already uses.
Samudra gives a fraction and an ice volume, not a surface energy balance, so a
slab is the honest ceiling.

`Fioi_melth`, `Fioi_meltw` and `Fioi_salt` are deliberately zero: Samudra
advances its own ice, so the heat and freshwater of a melting pack are already
inside its step and handing them over again would count them twice.
`Fioi_taux/tauy` pass the atm/ice stress straight through, so that after the
coupler weights ice and open water the total stress on the ocean is what the
atmosphere applied.

## Constraints

* Requires `EOCN` as the ocean component.  It has no grid of its own -- EOCN
  publishes the mesh through `shr_emul_ice_mod` at init, which is why the two
  cannot disagree about it.  Pairing EICE with anything else aborts at init.
* `NTASKS_ICE` must equal `NTASKS_OCN`.  EICE rebuilds EOCN's decomposition
  rather than choosing one; a mismatch aborts rather than mis-indexing.
* Carries no state, so there is no restart file and nothing to keep
  bit-for-bit across one.
* Has no namelist.  Everything it reports is either the emulator's fraction or
  a compile-time constant shared with `dice`.

## Compsets

| alias | what it is |
|---|---|
| `E2000-EATM-EOCN-EICE` | both halves of SamudrACE with the ice fraction in the coupler |
| `F2010-ELM-EOCN-EICE` | prognostic EAM and ELM over the emulated ocean and its ice |

`E2000-EATM-EOCN` and `F2010-ELM-EOCN` are the same pairs without EICE.  Keep
them for measuring what the ice is worth; do not use them for science.

## The flux subtlety

Adding EICE changes what the *ocean* receives, because the coupler starts
splitting every surface flux between open water and ice.  Samudra was trained
on whole-cell fluxes and applies its own ice insulation internally, so EOCN
divides that weighting back out by default.  See `eocn_flux_ifrac_unweight` in
`../eocn/` and section 6 of its `VERIFICATION.md`; the measurement showing why
is there.
