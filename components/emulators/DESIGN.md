# Design

## An emulator is a spec; the infrastructure is code

Everything that is a fact about a checkpoint or a coupling is YAML in
`specs/`, written by hand and reviewed like a namelist. Everything that is
numerics is shared, tested C++. A new emulator is a new spec, plus an operator
only if it needs physics nobody has written.

```yaml
name: ace2-atmosphere
network:                      # the checkpoint's tensor contract
  timestep: 21600
  inputs: [LANDFRAC, OCNFRAC, ICEFRAC, PHIS, SOLIN, PS, TS, "T_{0..7}", ...]
  outputs: [PS, TS, "T_{0..7}", ..., LHFLX, SHFLX, FSDS, ...]
  interval_mean_outputs: [LHFLX, SHFLX, FSDS, ...]
  coupled_inputs: [LANDFRAC, OCNFRAC, ICEFRAC, TS]
  boundary_inputs: [PHIS]
  forcing_inputs: [SOLIN]
stepping: interpolate         # or window_close
operators:                    # applied in order; fields named by reference
  - operator: ace.surface_inputs
    coupler: {lfrac: imports.Sf_lfrac, ofrac: imports.Sf_ofrac, ...}
    to: {landfrac: inputs.LANDFRAC, ...}
  - operator: insolation
    channel: SOLIN
  - operator: ace.surface_exports
    from: {ps: state.PS, ...}
    to: {tbot: exports.Sa_tbot, ...}
coupler:
  imports: [{name: Sf_lfrac, units: "1"}, ...]
  exports: [{name: Sa_tbot, units: K}, ...]
```

The component's input file (`atm_in`, YAML, written by `buildnml`) names the
spec and the case's paths: coupler step, grid file, initial condition, and
the inference backend.

## Layers

```
MCT driver
  │  attribute vectors, clock, infodata
Fortran cap        emulatoratm/src/atm_comp_mct.F90 + common/src/emulator_mct_cap.F90
  │  gsMap, domain, buffers, time: written once, shared by every cap
C API              common/src/emulator_c_api.cpp        opaque handle; each call guarded
  │
Emulator           common/src/emulator.hpp             lifecycle, domain, coupler binding
EmulatorComponent  common/src/emulator_component.cpp   input file, grid, backend, history, restart
EmulatedModel      common/src/model/emulated_model.cpp spec: stepping, initial condition, operators
  │
Operators          model/common_operators.cpp          insolation, exchange.publish, exchange.window_mean
                   emulatoratm/src/ace_operators.cpp   ace.surface_inputs, ace.surface_exports
Physics            plain functions with their own tests (ace_surface, insolation)
  │
Primitives         config/    Section: YAML whose errors name the key; `extends`
                   fields/    FieldSet (spans), FieldList, MaskSet, ChannelLayout, CouplerBinding
                   grid/      HorizontalGrid, Decomposition, Domain, SCRIP reader, GlobalGather
                   coupling/  LongStepClock, IntervalMean, BracketedState, NetworkStepper,
                              RestartStore/File, Exchange
                   inference/ Tensor, InferenceBackend: stub, python, libtorch
```

Dependencies point down only. Nothing under `common/` knows a checkpoint; an
operator knows physics but not which fields it is applied to until the spec
says.

## The step

Per coupler step the model calls, in spec order: `sample` (first call at a
model time), `before_step` and `after_step` (around a network step), and
`exports` (every call). `initialize` runs once after the initial condition.
Fields are named by reference: `imports.`, `exports.`, `inputs.`, `state.`
(blended or held outputs), `upper.` (the latest prediction), `aux.` (shared
between operators, restart state), `statics.` (read once from the initial
condition), `exchange.` (fields published in-process by another component).
A spec whose coupled or forcing input no operator sets is refused before the
network is built.

Two steppings. `interpolate`: the network steps at the start of each interval
and the coupler sees snapshots blended between the brackets and means held.
`window_close`: the network steps when a window of coupler steps closes, from
the state at its start and forcing averaged over it.

## Rules

- **A component never touches a coupler buffer.** It declares fields by name
  and unit; the base class binds them once and moves data before and after
  each `run`. A missing required field is an error naming the near misses.
- **MCT's layout is written in one place.** `AttrVectView` is the only code
  that indexes point-major attribute vectors.
- **Storage is spans.** A field is a `std::span<double>` over memory the
  framework owns, so a different coupler API changes the binding, not the
  components.
- **Time is the driver's.** The cap passes the clock through; a repeated call
  at one model time neither advances nor counts. The completed-step count is
  what a stochastic backend seeds from, so restarts draw the same noise.
- **Every lesson is a test that fails without it.** A rule stated here has a
  test that pins it.
- **Refactor bit for bit.** A change in structure is checked against the
  previous output before a change in behaviour is made.

## Restarts and history

The driver's restart alarm makes the cap write `$CASE.emulatoratm.r.<date>.nc`
and `rpointer.atm`; a continue run reads them back. The file holds every
primitive's state (`save_to` / `load_from`), gathered to the whole grid, so a
restarted run may use a different rank count. `history:` in the input file
writes interval means of any referenced field on the (lat, lon) grid.

## Portability

The framework is a standalone CMake project with three optional dependencies
(netCDF, libtorch, Python); MPI is required. Tests that need data or a GPU
skip with a message. Inside E3SM the same tree is added by
`components/cmake/build_emulator_comps.cmake`; the machine file provides
`Torch_ROOT`. Specs are copied into the build directory so a built case reads
the specs it was built with.
