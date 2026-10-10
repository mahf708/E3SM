# P3 warm-rain emulator

P3 computes its four warm-rain collision rates (autoconversion, droplet
self-collection, accretion, rain self-collection) in their own stage of
`p3_main` (`impl/p3_warm_rain_impl.hpp`), between the size-distribution step
and the rest of the process loop (`p3_main_part2`). With small kernels
(`SCREAM_P3_SMALL_KERNELS=ON`) each step is its own `(column, level)` kernel,
and `p3_main` calls a host hook (`WarmRainHook`) between the warm-rain kernel
and part2, outside of any kernel. The rates the hook leaves in
`P3Temporaries::warm_rain` are the ones part2 uses.

`P3Microphysics` uses that hook to run an emulator
(`eamxx_p3_warm_rain_emulator.cpp`), with one of two backends
(`warm_rain_emulator_backend`):

- `python` (default): through EAMxx's python interface, see below. Any model
  that implements the contract below can be used.
- `kokkos`: the MLP is evaluated inside a device kernel, one pack of levels
  at a time (`p3_warm_rain_mlp.hpp`), immediately followed by the merge in
  the same kernel. No python and no host-device copies are needed. The model
  file is a text file written from the `.pt` file by `export_kokkos_mlp.py`.

With the python backend:

1. a device kernel gathers the state P3 sees at the warm-rain stage
   (`qc`, `nc`, `qr`, `nr` after the size-distribution clipping, dry `rho`);
2. the python module's `forward()` runs on host copies of those fields;
3. a device kernel (`warm_rain_emulator_merge`) puts the emulated rates in
   place where the module's masks allow, and keeps stock P3 elsewhere.

The exchanged fields are padded like P3's packed views, so the device kernels
work on packs, while python sees `(ncol, nlev)` strided arrays.

## Python module contract

```python
init(model_file)
check_timestep(dt)
forward(qc, nc, qr, nr, rho,                       # inputs, (ncol, nlev)
        qc2qr_autoconv_tend, qc2qr_accret_tend,    # outputs, written in place
        ncautr, nc2nr_autoconv_tend, nc_accret_tend,
        nc_selfcollect_tend, nr_selfcollect_tend,
        use_cloud, use_rain)
```

- Inputs: dry mixing ratios (kg/kg, #/kg) and dry density (kg/m3).
- Outputs: **grid-mean** rates in P3's sign conventions
  (`nc_selfcollect_tend <= 0`), and 0/1 masks: `use_cloud` for the first six
  rates, `use_rain` for `nr_selfcollect_tend`.

`p3_warm_rain_emulator.py` implements this contract for the SDM-trained MLP
(ERF super-droplet LES); it needs numpy and torch at run time. The model file
holds the weights, normalization, gates, training envelope and number-rate
constants, so a retrained model is a file swap.

## Runtime options (p3 parameter list)

| Option | Default | Meaning |
|---|---|---|
| `use_warm_rain_emulator` | false | enable the emulator |
| `warm_rain_emulator_backend` | python | `python` or `kokkos` |
| `warm_rain_emulator_file` | none | model file (`.pt` for python, `.txt` for kokkos) |
| `warm_rain_emulator_kk_factor` | 1.0 | factor on P3's autoconversion where `use_cloud = 0` (the SDM bundle suggests 0.36) |
| `warm_rain_emulator_cloud_self_collection` | true | use the emulated cloud self-collection (zero in stock P3) |
| `py_module_name`, `py_module_path` | `p3_warm_rain_emulator`, this directory | python module to load |

Both backends require `SCREAM_P3_SMALL_KERNELS=ON`; the python backend also
requires `EAMXX_ENABLE_PYTHON=ON`.
