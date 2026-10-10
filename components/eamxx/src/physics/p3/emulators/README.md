# Emulating P3 process rates

Any subset of P3's process rates can be computed by an emulator instead of P3,
from one rate to all of them, warm or ice, with any number of emulators, each
on its own inference backend. Which rates are emulated is only configuration:
adding a new emulator needs no C++ change.

## How it works

`p3_main_part2` computes P3's process rates, maps them to cell averages
(`back_to_cell_average`), then applies them (conservation checks, update of the
state, diagnostics). With emulators, it runs as two kernels, with the emulators
in between, outside of any kernel:

```
part1 → part2 (Rates): compute and store the rates → emulators → part2 (Apply) → sedimentation …
```

- The 37 stored quantities are named in `p3_process_rates.hpp`: 29 process
  rates, 6 heterogeneous-freezing rates, the density of new rime and the
  wet-growth flag. They are **cell averages**, in P3's units and signs, i.e. as
  P3's conservation routines see them. Emulated rates go through the same
  conservation checks as stock ones.
- The emulators see P3's state as it was when the rates were computed, by name
  (`P3ProcessState`, built in `disp/p3_main_impl_disp.cpp`): prognostics (`qc`,
  `nc`, `qr`, `nr`, `qi`, `ni`, `qm`, `bm`, `qv`, `th_atm`), diagnostic inputs
  (`pres`, `dpres`, `dz`, `cld_frac_l/i/r`, `inv_qc_relvar`, ...) and P3's
  temporaries (`T_atm`, `rho`, `qv_sat_l`, in-cloud `*_incld`, size-distribution
  parameters `mu_c`, `lamc`, `mu_r`, `lamr`, ...). They can also take any
  process rate as input, as P3 computed it.
- Without emulators, part2 runs as one kernel, as before (BFB). Emulators need
  `SCREAM_P3_SMALL_KERNELS=ON`, since the monolithic kernel cannot be split.

The emulators themselves (`share/emulation/eamxx_process_emulator.hpp`) know
nothing about P3: they map named inputs to named outputs, so any
parameterization that exposes its rates the same way can use them. They run
through the inference backends of `components/emulators`:

| backend | model | what it gets |
|---|---|---|
| `python` | a python module with `create_emulator(config)`, returning an object with `infer(inputs, outputs)` | dicts of numpy arrays, by name, each `(ncol, nlev)`; outputs are written in place |
| `libtorch` | a TorchScript file | the inputs as positional arguments of `forward()`, which returns a tuple of the outputs, in the order of the configuration |
| `stub` | none | leaves the outputs at zero |

Inputs are gathered on device from P3's packed views into contiguous
`(ncol, nlev)` buffers, copied to host for the backend (a no-op on CPU), and the
outputs are merged into the packed rates on device.

## Configuration (p3 parameters)

```yaml
p3:
  process_emulators: [warm, melt]   # run in this order
  process_emulators_limit_self_collection: true
  warm:
    backend: python                 # stub | python | libtorch
    model_path: /path/to/model
    inputs:  [qc, nc, qr, nr, rho]  # state names, or rate names
    outputs: [qc2qr_autoconv_tend, ncautr, nc2nr_autoconv_tend, valid]
    masks:                          # optional: use the rates only where the mask > 0.5
      valid: [qc2qr_autoconv_tend, ncautr, nc2nr_autoconv_tend]
    fallback_scale:                 # optional: scale stock P3 where the mask is 0
      qc2qr_autoconv_tend: 0.5
    mode: replace                   # replace | add (add the outputs to P3's rates)
    options:                        # backend options, see components/emulators
      python_module: my_warm_emulator
      python_path: /path/to/module
  melt:
    backend: libtorch
    model_path: /path/to/melt.pt
    inputs:  [T_atm, qi_incld, ni_incld, qi2qr_melt_tend]
    outputs: [qi2qr_melt_tend, ni2nr_melt_tend]
    options: {device: cpu, dtype: float64}
```

Each output must be a process rate or a mask. Rates that no mask gates are
always used. With `mode: add` and P3's own rates as inputs, a model can learn a
correction of P3 rather than a replacement.

`process_emulators_limit_self_collection` limits emulated `nc_selfcollect_tend`
and `nr_selfcollect_tend`, so that they cannot remove more number than there is
in one step: P3's `nc_conservation` uses `nc + nc_selfcollect_tend*dt` as the
source for the other nc sinks, and `nr_conservation` counts
`nc2nr_autoconv_tend`, not `ncautr`, as the nr source.

## Build

```
-DSCREAM_P3_SMALL_KERNELS=ON
-DEAMXX_ENABLE_PROCESS_EMULATORS=ON
-DEMULATOR_ENABLE_PYTHON=ON      # python backend
-DEMULATOR_ENABLE_LIBTORCH=ON    # libtorch backend, with Torch_DIR or CMAKE_PREFIX_PATH
```

## Example

`sdm_warm_rain/` emulates the four warm-rain collision processes with the
SDM-trained emulator of the SDM_emulator bundle, through either backend (one
torch module, used eagerly by the python backend, and exported to TorchScript
by `export_torchscript.py` for the libtorch one). See `sdm_warm_rain.yaml`.

## Tests

- `share/emulation/tests`: the emulator semantics (masks, fallback, add,
  rates as inputs, padding, errors), with a C++ backend.
- `physics/p3/tests/p3_process_rates_tests.cpp`: the registry, and the split of
  part2 (a hook that changes nothing changes nothing; turning the warm-rain
  rates off by name leaves no rain).
- `tests/single-process/p3_process_emulators`: P3 with emulators that return
  P3's own rates must be BFB with stock P3, for all 37 quantities through the
  python backend, and for two emulators in a chain (python, then libtorch).
