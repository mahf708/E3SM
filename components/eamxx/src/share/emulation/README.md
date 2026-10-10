# Emulators in EAMxx

An emulator computes some named quantities from others, through one of the
inference backends of `components/emulators` (`stub`, `python`, `libtorch`).
The same machinery (`ProcessEmulator`, `eamxx_process_emulator.hpp`) emulates
quantities at three depths, all chosen in the input yaml, with no C++ change
for a new emulator:

| what | where | parameters | needs |
|---|---|---|---|
| a whole process, or group of processes | any `AtmosphereProcess` | `field_emulators`, `field_emulators_mode` | |
| any subset of P3's 37 process rates | between the two kernels of P3's part2 | `p3: process_emulators` | `SCREAM_P3_SMALL_KERNELS` |
| P3's sedimentation tendencies, surface precipitation | after P3's sedimentation | `p3: sedimentation_emulators` | `SCREAM_P3_SMALL_KERNELS` |
| SHOC's eddy diffusivities (`tk`, `tkh`), TKE, isotropy | after `shoc_tke`, before the implicit solver | `shoc: eddy_diffusivity_emulators` | `SCREAM_SHOC_SMALL_KERNELS` |

All need `-DEAMXX_ENABLE_PROCESS_EMULATORS=ON`, and `-DEMULATOR_ENABLE_PYTHON=ON`
or `-DEMULATOR_ENABLE_LIBTORCH=ON` (with `Torch_DIR`) for those backends.
Without emulators in the yaml, nothing changes (BFB): the hooks are inactive,
and P3 and SHOC run their usual kernels.

## An emulator

```yaml
my_emulator:
  backend: python                 # stub | python | libtorch
  model_path: /path/to/model      # for libtorch, the TorchScript file
  inputs:  [qc, nc, qr, nr, rho]  # by name
  outputs: [qc2qr_autoconv_tend, ncautr, nc2nr_autoconv_tend, valid]
  masks:                          # optional: change the gated targets only where the mask > 0.5
    valid: [qc2qr_autoconv_tend, ncautr, nc2nr_autoconv_tend]
  fallback_scale:                 # optional: scale the gated target where the mask is <= 0.5
    qc2qr_autoconv_tend: 0.5
  mode: replace                   # replace | add (add the outputs to the targets)
  options:                        # backend options (see the backend headers in components/emulators)
    python_module: my_module
    python_path: /path/to/module
```

Each input and output is one tensor, `(ncol, nlev)`, or `(ncol)` for per-column
quantities. The python model is a module with `create_emulator(config)`,
returning an object with `infer(inputs, outputs)` (dicts by name; outputs are
written in place). A TorchScript model gets the inputs as positional
arguments of `forward()`, in the configured order, and returns the outputs
(a tensor, or a tuple) in the configured order.

## No copies

The caller hands the emulator named strided views (packed, padded, slices of
a bigger view, per column) where they live. They go to the backend **in
place**, with their strides, when the backend accepts their memory space and
`Real` is double:

- libtorch wraps host or CUDA memory with `from_blob` (with strides) and copies
  outputs into the targets on their device;
- python gets numpy arrays on host memory (with strides), or, with option
  `device_arrays: true`, objects with a `__cuda_array_interface__` on device
  memory (cupy, torch, numba, jax can wrap them with no copy);
- the stub accepts anything.

Otherwise (single precision, or a backend that only reads host memory on a
GPU build) the arrays are gathered into contiguous double buffers **on the
device**, mirrored to the host only if the backend needs it. A target is
written in place by the backend when nothing else has to happen to it (no
mask, `mode: replace`, no overlap with an input); otherwise the backend writes
into a buffer, merged into the target by a device kernel.

## Whole processes: field emulators

Any process (or group) can run emulators on its fields:

```yaml
p3:
  field_emulators: [whole_p3]
  field_emulators_mode: replace   # replace: instead of P3; after: after P3 (e.g. to correct it)
  whole_p3:
    backend: libtorch
    model_path: p3.pt
    inputs:  [T_mid, qv, qc, nc, qr, nr, p_mid, pseudo_density]
    outputs: [T_mid, qv, qc, nc, qr, nr, precip_liq_surf_mass]
```

Inputs are the fields the process requires or computes (outputs as the process,
or the emulators before, left them), and, in `after` mode, `X_before` for an
output `X`. Targets are the fields the process computes or updates, including
those of its groups (e.g. tracers). Vector fields `(COL, CMP, LEV)` are one
array per component, `<field>_<index>`. See `eamxx_field_emulators.hpp`.

## Tests

- `share/emulation/tests`: the emulator semantics (masks, fallback, add,
  padding, aliasing of inputs and targets, what goes in place), with a C++ backend.
- `components/emulators/common/tests`: strided and device tensors, the python
  and libtorch backends in place.
- `tests/single-process/field_emulators`: P3 with field emulators. An emulator
  returning P3's outputs after P3 is BFB with stock P3; one that restores the
  outputs as they were before P3 (`X_before`) is BFB with one that replaces P3
  and returns its inputs.
- P3 and SHOC: see `physics/p3/emulators/README.md`, `physics/p3/tests/p3_process_rates_tests.cpp`,
  `tests/single-process/p3_process_emulators`, `physics/shoc/tests/shoc_hooks_tests.cpp`.

GPU paths (device tensors, CUDA `from_blob`, `__cuda_array_interface__`) compile
on CPU builds but have only been exercised on CPU so far.
