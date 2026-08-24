# Multi-instance support in cpl7/moab

`cpl7/mct` (`driver-mct`) supports CIME's multi-instance capability: setting
`NINST_<COMP> > 1` gives a case `N` copies of a component, each with its own
namelist and its own coupling state. This is what the EAMxx
RCS / NBFB workflow (`components/eamxx/docs/user/multi-instance-rcs.md`)
is built on.

`cpl7/moab` (`driver-moab`) does not support it. This page records why, what a
port would involve, and what to do in the meantime.

## Short version

`driver-moab` is a fork of `driver-mct` in which the coupler-side field storage
was moved out of MCT attribute vectors and into MOAB mesh tags. In `driver-mct`
that storage is reached through `component_type_mod`'s `atm(:)`, `lnd(:)`, ...
arrays, which are already dimensioned by `num_inst_*`, so a second instance gets
a second set of attribute vectors for free. In `driver-moab` it is reached
through module-level *scalar* iMOAB application handles in
`driver-moab/shr/seq_comm_mct.F90` — `mbaxid`, `mboxid`, `mblxid`, `mbixid`,
`mbrxid` and friends. There is exactly one coupler-side mesh, and one set of
`a2x_*`/`x2a_*` tags on it, per component *type* — not per instance.

Everything else follows from that.

## What specifically breaks

Run `driver-moab/tools/audit_multi_instance.py` for the current numbers. As of
this writing:

| # | Assumption | Where |
| --- | --- | --- |
| 1 | 25 iMOAB application handles are module-level scalars, referenced ~1330 times | `driver-moab/shr/seq_comm_mct.F90`, `driver-moab/main/*.F90` |
| 2 | The coupler-side MOAB mesh is built for instance 1 only | `component_mod.F90`, `if (eci == 1) call cplcomp_moab_Init(infodata, comp(1))` |
| 3 | 12 run-loop MOAB exchanges are hard-coded to instance 1 | `cime_comp_mod.F90`, `call component_exch_moab(atm(1), mbaxid, mphaid, ...)` etc. |
| 4 | Area correction runs on instance 1 only | `component_mod.F90`, `component_init_areacor_moab` |
| 5 | Merges assume one instance and never average | `prep_ocn_mod.F90`, `prep_ocn_mrg_moab`: *"no averages, just one ocn instance"* |
| 6 | Two component dycores register their iMOAB app with instance 1's id | `components/eam/src/dynamics/se/dyn_comp.F90`, `components/eamxx/src/dynamics/homme/interface/phys_grid_mod.F90`: `ATM_ID1 = ATMID(1)` |

Because of (2) and (3), a case built with `COMP_INTERFACE=moab` and
`NINST_ATM=2` would *build and start*, but instances 2..N would never be given a
coupler-side mesh and would never exchange fields. That is a silent wrong
answer, not a crash, so `cime_pre_init1` now aborts on `num_inst_max > 1` with a
message pointing here.

Note what is *not* broken. The per-instance MPI communicators and component ids
(`ATMID(:)`, `CPLATMID(:)`, `comp%compid`, `comp%cplcompid`) are inherited
unchanged from `driver-mct` and are already per-instance, and
`component_exch_moab` derives its iMOAB context ids from them. The MCT-era
per-instance scaffolding in the `prep_*` modules (`a2x_ox(num_inst_atm)`, ...)
is also still in place. The missing piece is genuinely the MOAB-side storage,
not the surrounding bookkeeping.

The CIME-side XML (`NINST`, `NINST_LAYOUT`, `MULTI_DRIVER`) is identical between
the two drivers, so nothing on the case-control side needs to change.

## Two ways to add support

### Option 1: `MULTI_DRIVER=TRUE` — cheap, and probably sufficient

With `MULTI_DRIVER=TRUE`, CIME gives each ensemble member its own driver/coupler
instance. Two facts make this a good fit for MOAB:

* `share/build/buildlib.csm_share` sets `NUM_COMP_INST_<COMP>=1` when
  `MULTI_DRIVER` is true, so every driver copy is compiled with
  `num_inst_* == 1`. None of the six problems above are reachable.
* `cime_cpl_init` splits the driver communicator with `mpi_comm_split`, so
  driver instances run on **disjoint** PE sets. A module-level scalar such as
  `mbaxid` is therefore already private per process — the thing that makes
  in-driver multi-instance hard is exactly what makes multi-driver easy.

Most of the supporting plumbing is already threaded: `cpl_inst_tag` reaches
`cpl_modelio.nml`, coupler history and restart file names, and the timing
output; `shr_pio_init1` is passed `driver_comm` rather than the global
communicator; and every `iMOAB_WriteMesh` call in `driver-moab/main` that uses a
fixed filename is inside `#ifdef MOABDEBUG`, so production runs will not have
driver instances clobbering each other's mesh dumps.

`driver-moab` nevertheless aborts on `num_inst_driver > 1` today
(`cime_comp_mod.F90`), and that abort predates this analysis — it is a
"nobody has tried it" guard, not a known failure. Work needed:

1. Relax the abort behind a switch and run `ERS_Vmoab` at `ninst_driver=2`.
2. Confirm the online map-generation path (`allactive-onlinemaps`) does not write
   shared intermediate files across driver instances.
3. Confirm MOAB's own global state (`iMOAB_Initialize` is called once per
   process from `seq_comm_init`) behaves with two independent app sets per job.

Cost: memory scales as `N` × (coupler mesh + remap weights), which is the same
as MCT's `MULTI_DRIVER`. Effort is days of testing rather than a refactor. This
is also the mode the RCS documentation already recommends, to avoid running all
ensemble members through one coupler and running out of memory.

### Option 2: true `NINST > 1` inside one driver — a real project

This is the `MULTI_DRIVER=FALSE` path, where one coupler serves `N` instances.
It requires:

* Promoting the 25 scalar handles to `(num_inst_*)` arrays and threading an
  instance index through ~1330 references across ~15 files.
* Deciding what to do about intersection meshes and remap weights. These are the
  expensive part of coupler init and they are grid properties, not instance
  properties, so sharing them across instances is very desirable. With
  `NINST_LAYOUT=sequential` the instances share PEs and share a decomposition,
  so one intersection app with instance-suffixed tag names would work. With the
  default `NINST_LAYOUT=concurrent` the instances live on different PE sets, the
  covering mesh built inside the intersection app differs per instance, and the
  weights end up duplicated `N` times anyway.
* Adding instance loops and the MCT-style `num_inst_ocn == 1 < num_inst_max`
  averaging to `prep_*_mrg_moab`.
* Giving fractions, atm/ocn fluxes, budget diagnostics, history and restart an
  instance dimension.
* Fixing `ATM_ID1 = ATMID(1)` in the EAM and EAMxx dycores, and — for
  `NINST_LAYOUT=sequential`, where all instances share a process — making the
  component-side handles (`mhid`, `mhpgid`, `mphaid`, `mpoid`, `mlnid`,
  `mrofid`, `mpsiid`) per-instance too.

Cost: a large refactor touching most of `driver-moab/main`, plus the same `N`×
memory for meshes and tags that Option 1 pays. The only thing it buys over
Option 1 is the ability to share remap weights, and only in the sequential
layout.

## Recommendation

Pursue Option 1. It gets multi-instance and RCS/NBFB testing working under
`COMP_INTERFACE=moab` for a fraction of the effort, and it is the configuration
users are told to prefer anyway. Treat Option 2 as a separate project, justified
only if sharing remap weights across ensemble members turns out to matter.

## Until then

Multi-instance cases must use `COMP_INTERFACE=mct`. In the E3SM test suites this
is already the case — `cime_config/tests.py` pins the multi-instance tests to
the MCT driver explicitly:

```text
NCK_Vmct.ne4pg2_oQU480_rx1.A                 (e3sm_developer)
NCK_Vmct.ne4pg2_oQU480.WCYCL1850NS           (e3sm_integration)
ERS_Vmct.hcru_hcru.IELM.elm-multi_inst       (e3sm_land_developer)
```

The MOAB suites (`e3sm_moab_ers`, `e3sm_moab_pem`, `e3sm_moab_dev`) contain no
`NCK` test. Once Option 1 lands, `NCK_Vmoab` with `MULTI_DRIVER=TRUE` is the
natural test to add.

## Auditing

```bash
./driver-moab/tools/audit_multi_instance.py -v
```

The script is a static grep-based inventory of the assumptions in the table
above; it does not build or run the model. Use it to track progress on a port.
