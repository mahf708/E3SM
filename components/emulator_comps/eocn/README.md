# EOCN — emulator ocean component

EOCN drives a traced Samudra-family ocean emulator through FTorch and presents
it to the MCT coupler as a prognostic ocean.  It is the ocean counterpart of
EATM, and the two together run the SamudrACE-E3SMv3 coupled emulator inside
E3SM's own coupler rather than inside `fme`.

## What it is

| | |
|---|---|
| emulator | SamudrACE-E3SMv3 ocean (Samudra) |
| grid | 180x360 Gaussian, the same grid EATM uses |
| vertical | 19 coarsened depth levels |
| step | 5 days |
| channels | 92 in (2 static + 10 atmospheric fluxes + 80 state), 80 out |
| device | CUDA (`eocn_model_device`) |

The 80 output channels are all prognostic: `sst`, `ssh`, 19 levels each of
salinity, potential temperature, zonal and meridional velocity, plus
`ocean_sea_ice_fraction` and `iceVolumeTotal`.

## How it is coupled

**Forcing.**  The checkpoint declares its ten atmospheric flux channels as
*next-step* forcing: in reference inference the model is handed the fluxes over
the interval it is about to predict.  Online there is no such thing, because
the atmosphere has not run those five days yet.  EOCN therefore accumulates the
coupler's fluxes every coupling step and drives each emulator step with the
mean over the interval that just closed.  That is a five-day persistence
assumption on the forcing, and it is the only causal choice available.  State
time labels stay exact.

**Units and signs.**  The coupler's surface fluxes are positive *into* the
surface and the emulator's `FLUS`/`FSUS`/`LHFLX`/`SHFLX` are positive upward,
so those change sign; `TAUX`/`TAUY` are the stress on the atmosphere, which is
minus the coupler's stress on the ocean.  Precipitation arrives as kg/m2/s and
the emulator wants m/s of liquid water equivalent.

**Shortwave.**  The coupler sends only the net absorbed shortwave
(`Foxx_swnet`); the emulator wants `FSDS` and `FSUS` separately.  EOCN splits
the net with a constant ocean albedo of 0.06, which keeps their *difference* —
the term that drives the heat budget — exactly right and misstates only how the
pair is apportioned.

**Interpolation.**  The emulator steps five days at a time; the coupler sees a
linear interpolation between the bracketing states, so a five-day step in SST
does not reach the atmosphere as a discontinuity.  Set
`eocn_interp_state = .false.` to hold each state flat for its whole step
instead.

**Sea ice — the largest open gap.**  Samudra carries its own sea ice
internally (`ocean_sea_ice_fraction`, `iceVolumeTotal`), and SamudrACE's
ocean-to-atmosphere exchange is exactly `[ocean_sea_ice_fraction, sst]`.  EOCN
exports the sea surface temperature but has nowhere to put the ice fraction:
with a stub ice component the coupler sets the atmosphere's `ICEFRAC` to zero
everywhere, while the emulator is predicting a global mean near 0.2.  The polar
ocean therefore reaches the atmosphere as open water at the freezing point —
the wrong albedo and the wrong turbulent exchange over roughly a tenth of the
globe.  `VERIFICATION.md` measures what that costs.

Closing it means giving the coupler an ice fraction.  The two routes worth
considering, cheapest first: a minimal pass-through ice component that reports
EOCN's `ocean_sea_ice_fraction` as its own domain fraction and surface state,
or an EATM option to take `ICEFRAC` from the ocean's export rather than from
`Sf_ifrac`.  The first keeps the coupler's fraction bookkeeping honest; the
second is a smaller change but leaves the coupler's merge inconsistent with
what the atmosphere is told.

**Land.**  Samudra's published initial conditions carry NaN over every land
cell, because an ocean state is undefined there.  A convolutional emulator
spreads one NaN outward on every layer and EOCN is autoregressive, so a single
NaN at step zero is a NaN field for the rest of the run.  `make_eocn_input.py`
substitutes zero, which is what `fme`'s data loader does and what the network
was trained against.  On export, land cells are filled with the freezing point
of sea water rather than left at whatever the network produced there.

## Compsets

| alias | longname |
|---|---|
| `E2000-EATM-EOCN` | `2000_EATM_SLND_SICE_EOCN_SROF_SGLC_SWAV` |
| `F2010-EOCN` | `2010_EAM%CMIP6_SLND_SICE_EOCN_SROF_SGLC_SWAV` |
| `F2010-ELM-EOCN` | `2010_EAM%CMIP6_ELM%SPBC_SICE_EOCN_SROF_SGLC_SWAV` |

Grids: `gauss180x360_gauss180x360` (both emulators on one grid, every map the
identity) and `ne30pg2_gauss180x360` (prognostic EAM over the emulated ocean).

`F2010-EOCN` has a known problem: the coupler sets the atmosphere's land
fraction to `1 - ocean fraction`, so with a stub land the merge hands EAM a
surface state weighted only by the ocean's share of each cell and nothing over
the continents.  `F2010-ELM-EOCN` is the same configuration with a land model.

## Preparing the inputs

```bash
# 1. pull the ocean half out of the coupled SamudrACE checkpoint
python $ACE/scripts/coupled/create_decoupled_checkpoint.py --component ocean \
    --input_path SamudrACE-E3SMv3.tar --output_path ocean.tar

# 2. trace it (adds back the sea-ice-fraction constraint the tracing script drops)
python tools/trace_eocn_model.py ocean.tar samudra_ocn_traced --device cuda

# 3. build the initial condition and the domain files
python tools/make_eocn_input.py --ic-out samudra_ocn_ic_0.nc \
    --domain-dir $SOMEWHERE/share/domains

# 4. only for a grid pair that is not the identity
python tools/make_atm_domain.py --map map_gauss180x360_to_ne30pg2_traave.nc \
    --atm-scrip ne30pg2_scrip_20200209.nc \
    --ocn-domain domain.ocn.gauss180x360_gauss180x360.nc \
    --out domain.lnd.ne30pg2_gauss180x360.nc
```

Point `eocn_model_file` and `eocn_ic_file` at the results in `user_nl_eocn`.

See `VERIFICATION.md` for what has actually been run and measured.

## Known gaps

* The SCRIP mesh EATM and EOCN share puts cell centres at integer degrees while
  the ACE and Samudra data are at half-integer degrees — a half-cell (~55 km)
  zonal offset in the absolute geolocation.  Both emulators use the same mesh,
  so the pair is self-consistent; it matters only against a third grid.
* EOCN runs on a single MPI task, like EATM.
* No history output of its own; the ocean state reaches disk only through the
  coupler history files and the EOCN restart.
