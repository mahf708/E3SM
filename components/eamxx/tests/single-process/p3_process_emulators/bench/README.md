# Emulator cost benchmark

Times P3 with and without emulators at each cut point (see
`share/emulation/README.md`, "What it costs"). Not part of the test suite.

```
python3 make_models.py OUTDIR                 # TorchScript models (needs torch)
python3 make_inputs.py NCOL NLEV NSTEPS OUTDIR  # input_<case>.yaml, in the current directory
OMP_NUM_THREADS=4 <build>/tests/single-process/p3_process_emulators/p3_process_emulators \
    --args -ifile=input_<case>.yaml
grep "^EAMxx::p3::run " eamxx_timing.txt
```

Use a Release build with `SCREAM_P3_SMALL_KERNELS=ON`,
`EAMXX_ENABLE_PROCESS_EMULATORS=ON` and `EMULATOR_ENABLE_LIBTORCH=ON`.
