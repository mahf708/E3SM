# FME Online Output -- v3 High-Resolution (1950 CMIP7)

`include_user_mods` overlay that runs the `fme_output` online-output
configuration with the **v3.HR.1950-CMIP7** science settings. It inherits the
entire FME machinery (horizontal remap, vertical coarsening, derived fields,
coupler-native merged forcing) from the sibling `fme_output` testmod and adds
only the run-time settings that define the v3 high-resolution 1950 CMIP7
simulation.

Settings were transcribed from the `20260525.v3.HR.1950-CMIP7` production
run_e3sm script (code base: master @ 20260521 `v3.1.0-beta.1-1613-g57e5cd7039`
plus `win/v3hr_1950_cmip7`; see
<https://e3sm.atlassian.net/wiki/spaces/CM/pages/6321340417>).

## Files

- `include_user_mods` -- pulls in `../fme_output` first (the FME output config).
- `shell_commands` -- appends the v3-HR science settings to
  `user_nl_{eam,elm,mpaso,mpassi}`.

## Intended configuration

```
COMPSET = WCYCL1950-CMIP7
RES     = ne120pg2_r025_RRSwISC6to18E3r5
```

Example (chrysalis):

```
./create_test SMS_Ld5.ne120pg2_r025_RRSwISC6to18E3r5.WCYCL1950-CMIP7.chrysalis_intel \
    --testmod allactive-fme_output_v3hr
```

## What was carried over (settings) vs dropped (output)

Only the **science settings** from the run script are included. The production
**output** (EAM `fincl1..6`/`nhtfrq`/`mfilt`, ELM `hist_*`, the MPAS
`streams.ocean` variable lists, COSP, chem/gas-budget history tapes) is
intentionally dropped -- FME online output replaces it.

| Component | Included (settings) | Dropped (output) |
|-----------|--------------------|------------------|
| EAM   | `clubb_c8=5.2`, `nucleate_ice_subgrid=1.40` | `cosp_lite`, all `fincl*`/`nhtfrq`/`mfilt`/`avgflag_pertape`, `history_chemdyg_summary`, `history_gaschmbudget_*`, `is_output_interactive_volc` |
| ELM   | `finidat` (1950 spun-up IC), `check_finidat_*_consistency=.false.`, CMIP7 `*_popdens` fire forcing (fixed 1950) | `hist_dov2xy`, `hist_fexcl1`, `hist_fincl*`, `hist_mfilt`, `hist_nhtfrq`, `hist_avgflag_pertape` |
| MPAS-O | `config_compute_active_tracer_budgets=.false.`, CVMix Langmuir off + latitude-dependent background diffusion (`3.0e-5`, max-lat `-50`) | `streams.ocean` output-stream edits |
| MPAS-SI | `config_use_sealevel_meltponds=true`, `config_use_level_meltponds=false`, `config_initial_condition_type='cice_default'` | `config_am_timeseriesstatsdaily_enable` (already off in `fme_output`) |

ELM input-data paths are anchored to `$DIN_LOC_ROOT` (portable) rather than the
run script's absolute `/lcrc/...` paths. `is_output_interactive_volc` reads as a
MAM5 diagnostic-output flag (not a scheme switch), so it is treated as output
and dropped; re-add it if it turns out to gate interactive volcanic emissions.

## Prerequisites / known gaps (read before first build)

1. **`WCYCL1950-CMIP7` compset is not in this branch.** It is defined by
   `win/v3hr_1950_cmip7`, which `fme_output`'s lineage does not include (here
   `WCYCL1950` is `EAM%CMIP6`, MAM4). Merge that branch -- it also brings the
   CMIP7 forcing / input-data definitions -- before using the intended compset.
   To smoke-test the FME + science overlay *now*, pair with the closest
   available compset (`WCYCL1850_chemUCI-Linozv3-mam5` for the chemUCI+MAM5
   chemistry, or `WCYCL1950` for the 1950 fixed forcing); forcing era and/or
   chemistry will differ from production.

2. **HR remap maps must be generated and staged.** `fme_output` auto-detects
   the grids and looks, under `$DIN_LOC_ROOT/fme` (override with
   `FME_MAPS_DIR`), for:
   - `map_ne120pg2_to_gaussian_180by360_shifted_trintbilin.nc`
   - `map_RRSwISC6to18E3r5_to_gaussian_180by360_shifted_trintbilin.nc`

   Generate them with the `ncremap` recipe in the header of
   `../fme_output/shell_commands`, using the ne120pg2 and RRSwISC6to18E3r5 SCRIP
   grids. NOTE: the 180x360 (1-degree) target throws away most of the HR
   resolution -- revisit the target grid with the ACE/Samudra team if a finer
   HR tape is wanted.

3. **HR/1950 input data must be staged** under `$DIN_LOC_ROOT`: the ELM
   `finidat` (`elmi.CNPRDCTCBCTOP.r025_RRSwISC6to18E3r5.1950-01-01...nc`) and the
   CMIP7 `popden_cmip7_...simyr1950...nc` fire-forcing stream. These are not
   auto-registered for download; stage them if absent.

4. **Ocean initial condition is NOT overridden.** The run script points
   `streams.ocean` at an EN4-1950 IC on a personal scratch path
   (`.../ac.vanroekel/en4_rrs_test/.../mpaso.RRSwISC6to18E3r5.20260522.nc`), which
   is not appropriate to hard-code into a committed testmod, so the
   compset-default RRSwISC6to18E3r5 IC is used. To reproduce the exact v3-HR
   spin-up, drop a patched `streams.ocean` into `SourceMods/src.mpaso/` after
   `case.setup` (the run script's `patch_mpas_streams` shows the diff).

5. **Vertical levels.** `fme_output`'s `vcoarsen_level_bounds = 0..80` assume the
   v3 L80 atmosphere, which ne120pg2 v3 also uses -- no change needed. The
   `do_aerocom_ind3`-gated fields (cdnc/lwp/ccn) and other `fincl1` entries are
   inherited from `fme_output`; confirm availability under the chemUCI+MAM5
   atmosphere at first build.

6. **PE layout** is not part of the testmod (it is chosen at create_test /
   create_newcase time via `--pecount` or the test PE spec). The run script's
   `custom-150` layout is a machine/run concern, not a science setting.

## Why a new testmod (not a modified `fme_output`)?

`fme_output` is validated and tuned for **v3 LR piControl** SamudrACE
(ne30pg2_r05_IcoswISC30E3r5, WCYCL1850). v3-HR is a different resolution,
compset (WCYCL1950-CMIP7), chemistry (chemUCI+MAM5), and uses HR/1950-specific
input data. Keeping the two as siblings (mirroring the
`wcprod`/`wcprod_1850`/`wcprodrrm` family) preserves the validated LR tape,
keeps HR-only absolute-path settings out of the LR config, and avoids
resolution conditionals in the shared FME script. The shared FME output logic
is reused verbatim via `include_user_mods`, so there is no duplication to
drift.
