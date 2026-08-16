# EATM — emulated atmosphere

EATM replaces EAM with a traced neural emulator from the ACE family, called
through [FTorch](https://github.com/Cambridge-ICCS/FTorch). It presents the
normal MCT atmosphere interface, so the surface components (MPAS-Ocean,
MPAS-Seaice, ELM) do not know the atmosphere is emulated.

```
    coupler                        EATM                      TorchScript
  ------------                 ------------                 -------------
  Sx_t, Sf_*frac  ------->  net_inputs (state)  ------->  normalize
                            + SOLIN (orbital)             SFNO / NoiseConditionedSFNO
                            + PHIS  (persisted)           correctors
                                                          denormalize
  Sa_*, Faxa_*    <-------  net_outputs                <-------
                            (interpolated between
                             two 6-hourly states)
```

The emulator steps every 6 hours; the coupler runs every 30 minutes
(`ATM_NCPL=48`). Inference happens only on emulator-step boundaries, and the
coupler is handed a linear interpolation between the two bracketing emulator
states. Both states go to the restart file, so a restart reproduces that
interpolation exactly (for a deterministic emulator).

## Supported emulators

| `EATM_EMULATOR` | architecture | channels | notes |
|---|---|---|---|
| `ACE2-EAMv3` | SFNO, deterministic | 39 in / 44 out | atmosphere only, prescribed SST |
| `SamudrACE-E3SMv3` | NoiseConditionedSFNO, **stochastic** | 43 in / 51 out | atmosphere half of the coupled emulator |

Channel layouts live in `src/eatm_channels_mod.F90`. Adding an emulator is one
`set_table_*` routine there plus one entry in the `EXPECTED` dict in
`tools/trace_eatm_model.py`; nothing else in EATM addresses channels directly.

`SamudrACE-E3SMv3` draws fresh noise on every forward pass and EATM cannot seed
libtorch's RNG through FTorch, so **runs with it are not reproducible and a
restart will not reproduce a continuous run**. Use `ACE2-EAMv3` for anything
that needs bit-for-bit behaviour, including `ERS` tests.

## Compsets and grid

| compset | components |
|---|---|
| `GMPAS-EATM` | EATM + MPAS-Seaice + MPAS-Ocean + data runoff, stub land |
| `GPMPAS-EATM` | as above plus ELM and MOSART (see REVIEW.md #24 — mapping files are missing from inputdata) |

Grid: `gauss180x360_IcoswISC30E3r5` — a 1° Gaussian atmosphere grid matching the
emulator's output, with the standard ~30 km E3SM ocean/sea-ice mesh.

## Running

### 1. Prepare the emulator (SamudrACE-E3SMv3 only)

The checkpoint has to be traced to TorchScript for the device it will run on.
On a GPU node:

```bash
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --account=e3sm_g
components/emulator_comps/eatm/tools/trace_samudrace_atmosphere.sh
```

That extracts the atmosphere stepper from the coupled checkpoint, traces it
(with `check_trace=False`, which a stochastic model requires), verifies its
channel order against `eatm_channels_mod.F90`, and builds a matching initial
condition from the published ICs.

`ACE2-EAMv3` needs none of this — its traced model and initial condition are the
namelist defaults.

### 2. Create, build, submit

```bash
components/emulator_comps/eatm/tools/run_gmpas_eatm_pm-gpu.sh
```

Defaults to a 5-year `GMPAS-EATM` run on `pm-gpu`, submitted as five 1-year
segments. Override with environment variables:

```bash
EMULATOR=ACE2-EAMv3 CASE_NAME=my-run STOP_N=1 RESUBMIT=4 SUBMIT=false \
  components/emulator_comps/eatm/tools/run_gmpas_eatm_pm-gpu.sh
```

### PE layout

EATM is serial and runs the emulator on a GPU, so it gets a node to itself at
global rank 0 while everything else starts at rank 64:

```
./xmlchange MAX_MPITASKS_PER_NODE=64
./xmlchange NTASKS=-10, NTASKS_ATM=1, NTASKS_ESP=1, NTASKS_IAC=1
./xmlchange ROOTPE=64, ROOTPE_ATM=0, ROOTPE_WAV=1, ROOTPE_GLC=1
./xmlchange PSTRID_ATM=16, EXCL_STRIDE_ATM=16
```

11 nodes total. Measured throughput 5.58 SYPD; the emulator is ~4% of runtime,
MPAS-Ocean (60%) and MPAS-Seaice (33%) dominate.

## Namelist

Everything is in `eatm_inparm` (`user_nl_eatm`); see
`bld/namelist_files/namelist_definition_eatm.xml` for the authoritative list.

| variable | default | |
|---|---|---|
| `eatm_model_file` | per emulator | traced TorchScript model FTorch loads |
| `eatm_ic_file` | per emulator | one 2D field per input channel, for a startup run |
| `eatm_model_device` | `gpu` | must match how the model was traced |
| `eatm_pass_forcing` | `.false.` | append next-step forcing channels (needed only for a model traced with `--add-ocean`) |
| `eatm_legacy_surface` | `.false.` | restore the pre-review surface diagnostics |
| `eatm_frzprec_units` | `m/s` | units of the frozen precipitation channel |
| `eatm_iradsw` | `1` | radiation interval, in coupler steps |

Which emulator to drive is an xml variable, not a namelist one:
`./xmlchange EATM_EMULATOR=SamudrACE-E3SMv3`.

## Files

```
bld/build-namelist                  namelist generation and consistency checks
bld/namelist_files/                 definitions and per-emulator defaults
cime_config/                        CIME component plumbing
src/atm_comp_mct.F90                MCT interface, namelist, import/export
src/atm_cpl_indices.F90             coupler field indices
src/ace_comp_mod.F90                FTorch inference, time interpolation, diagnostics
src/eatm_channels_mod.F90           per-emulator channel tables
src/eatm_comp_mod.F90               allocation, restart driving, orchestration
src/eatm_restart_mod.F90            restart and initial-condition I/O
src/eatmIO.F90                      PIO wrappers (copied from MOSART's ncdio_pio)
src/eatmMod.F90                     shared state
src/eatmSpmdMod.F90                 (serial) MPI setup
tools/                              tracing, initial conditions, run script
REVIEW.md                           code review findings and known limitations
```

Read `REVIEW.md` before trusting EATM output for science — it lists what is
approximated, what is exported as zero, and what is known to be wrong.
