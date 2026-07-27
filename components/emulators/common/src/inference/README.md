# Emulator inference layer

Machinery for an emulated component to evaluate a trained model: a data
container, a backend interface, a dependency-free stub, and an embedded-Python
bridge. Nothing here knows about atmospheres, land, grids or coupling. A
component packs its fields into tensors, calls `infer`, and reads the results
back — usually into memory it already owns.

```
              component (EmulatorAtm, ...)
                          |
        InferenceConfig   |   InferenceContext  <- from the MCT layer:
        (atm_in settings) |   (component comm + this rank's columns)
                          v
                   InferenceBackend
                          |
              +-----------+-----------+
              |                       |
            stub                   python
        (no dependency)     (embedded CPython, numpy views)
                                      |
                          e3sm_emulator.bridge
                                      |
                          +-----------+-----------+
                          |                       |
                       generic                   ace
                  (column network,        (a globe per sample:
                   pure data parallel)     single / spatial / ensemble)
```

## Files

| File | What it is |
| --- | --- |
| `tensor.hpp/.cpp` | `Tensor` and `TensorMap`: named, shaped, owning-or-viewing buffers |
| `inference_config.hpp/.cpp` | `InferenceConfig`: named settings, free-form options, namelist parsing |
| `inference_context.hpp/.cpp` | `InferenceContext`: ranks, node placement, rendezvous, this rank's columns |
| `inference_backend.hpp/.cpp` | The backend interface and its lifecycle |
| `create_inference_backend.hpp/.cpp` | `create_backend(config, context)` |
| `stub_inference_backend.hpp/.cpp` | No-op backend, no ML dependency |
| `python_interpreter.hpp/.cpp` | Embedded CPython session, GIL/FPE guards, `PyRef` |
| `python_inference_backend.hpp/.cpp` | Calls a Python emulator in-process |
| `inference_error.hpp` | `InferenceError` and the check macro |

The Python side is `../../python/e3sm_emulator/`; tests are in `../../tests`.

## The data container

`Tensor` is a named, shaped buffer that **either owns its memory or views
memory owned by somebody else**. The second case is the one that matters: a
component hands its own field or coupling arrays to a backend with no copy.

```cpp
TensorMap inputs;
inputs.wrap("T", static_cast<const double*>(m_T.data()), {ncol, nlev});
TensorMap outputs;
outputs.wrap("dT", m_dT.data(), {ncol, nlev});
backend->infer(inputs, outputs);   // writes straight into m_dT
```

- **Move-only, `clone()` for copies**, so an accidental deep copy of a
  multi-megabyte field cannot hide inside a function signature.
- **Views over `const` memory are read-only.** `data()` asks for write access
  and throws for such a view; `cdata()` reads without asking. Input fields are
  const views all the way through to numpy, so a model cannot corrupt state.
- **Row-major, leading dimension is the batch.** A Fortran `a(nlev,ncol)`
  corresponds to dims `{ncol, nlev}`.
- **Elements are always `double`.** Every E3SM field is `real(r8)`, and most
  models want float32 — but that conversion is one call on the Python side,
  where the model's precision is actually known. Carrying a dtype system and
  staging buffers through the whole bridge would buy nothing.

## Adding a model to a component

```cpp
// In init: settings live in the component's own namelist, prefixed.
auto config = InferenceConfig::from_file(m_input_file, "inference.");

// The resources the coupler gave us: the *component* communicator, and the
// columns this rank owns.
auto context = make_context(m_comm);
context.set_grid(m_nx, m_ny, m_num_global_cols, m_col_gids.data(),
                 m_lat.data(), m_lon.data(), m_num_local_cols);

m_inference = create_backend(config, context);   // built and ready

// Each step:
m_inference->infer(inputs, outputs);

// In finalize:
m_inference->finalize();
```

with, in `atm_in`:

```
nx: 90
ny: 45
inference.backend:    python
inference.emulator:   ace
inference.model_path: /path/to/ace_ckpt.tar
inference.input:      air_temperature_0
inference.output:     air_temperature_0
```

Settings the C++ layer does not recognise become options and are passed
through to the model, so a Python emulator can grow a setting without anyone
touching this directory.

For code that already thinks in flat arrays there is
`infer(const double* in, double* out, int batch_size)`, which wraps both
buffers as `[batch_size, channels]` using `input_channels` /
`output_channels`.

## The context, and why it is the interesting type

`InferenceContext` carries two things, both from the MCT coupling layer:

1. **The component communicator's** rank, size, per-node rank, and a
   rendezvous (rank 0's hostname and a port it has just confirmed free).
2. **This rank's columns** — global ids, latitudes, longitudes — plus the
   global grid shape.

The first exists because of a specific failure. ACE's `TorchDistributed` and
PhysicsNeMo's `DistributedManager` both discover their rank from
`SLURM_PROCID` / `SLURM_NTASKS` when nothing better is available, and in a
coupled run those describe the **whole job**. A process group built from them
waits forever for ocean and land ranks that will never call in. The Python
bridge publishes the component's own numbers into `RANK`, `WORLD_SIZE`,
`LOCAL_RANK`, `MASTER_ADDR` and `MASTER_PORT` — the variables those libraries
read *first* — so an unmodified upstream model initializes over exactly our
ranks. No upstream change is needed, though an explicit initializer in ACE
(`rank`, `world_size`, `local_rank`, `master_addr`, `master_port`, `h`, `w`)
would be cleaner than environment archaeology, and is worth proposing.

The second exists because the two decompositions never line up. E3SM's is
chosen for load balance across the coupled job; a global model's is the
geometry of its own transforms. Reconciling them is a permutation, and
`e3sm_emulator.decomposition` computes it once.

## The python bridge

The model stays in Python; only tensor *metadata* crosses the boundary:

```
Tensor (C++/Fortran memory)
  -> PyMemoryView_FromMemory(ptr, nbytes, PyBUF_READ|PyBUF_WRITE)
  -> numpy.frombuffer(view, dtype=float64).reshape(dims)     # still a view
```

This goes through the ordinary Python API rather than numpy's C API on
purpose: numpy is then a **runtime** dependency only, so building E3SM never
needs numpy headers, and the arrays are still zero-copy. (pybind11 would be
the other way to write this, and is what EAMxx's
`share/core/eamxx_pysession.hpp` uses; the plain C API keeps the build
dependency down to the Python development headers, and the Python-side
protocol is identical either way.)

At initialization the backend imports a module — `e3sm_emulator.bridge` by
default — and calls `create_emulator(config)` with every setting plus a
`context` dict. Once per step it calls `infer(inputs, outputs)`:

```python
def infer(self, inputs, outputs):
    outputs["dT"][:] = self.model(inputs["T"])
```

Inputs are read-only views, outputs are writable views of the component's own
arrays. **Zero-copy ends at the numpy view**: a model that moves the array
onto a GPU, or converts it to float32, pays for that copy like anybody else,
and every GPU execution stages host→device and back because the container is
host memory.

One interpreter per process, holding the GIL for the duration of each `infer`.
Separate MPI ranks are separate processes with separate interpreters, so ranks
do not contend for a GIL and independent model replicas run fine side by side.

## Running across ranks

**The backend runs a model in one process.** How that relates to the other
ranks is the model's business, and the Python side is where it is decided —
which is exactly why the Python bridge came first. It is the only backend
where a model can contain a collective at all; neither ONNX nor TorchScript
can express one portably.

Two shapes of model, and they scale completely differently:

**Column-local** (`y_i = f(x_i)`): pointwise parameterizations, column MLPs,
per-column vertical networks. `e3sm_emulator.generic` handles these. Every
rank loads its own weights, runs its own columns, communicates nothing.
Adding ranks adds throughput until the per-rank batch is too small to fill a
device — and note where E3SM layouts actually sit: ne30pg2 is 21,600 columns,
so ~1,350 ranks leaves 16 columns per rank, a batch too small to be worth a
GPU kernel launch.

**Globe-per-sample**: ACE and anything else built on spherical transforms.
`Stepper.step` takes `{name: tensor[n_batch, n_lat, n_lon]}`, so one sample is
an entire atmosphere and E3SM's local columns are *not* the batch dimension.
Splitting the grid across ranks splits one sample spatially, which a model
with a global receptive field will absorb and answer plausibly rather than
reject. `e3sm_emulator.ace` handles these, three ways:

| `ace_mode` | ACE backend | What happens |
| --- | --- | --- |
| `single` | `NonDistributed` | One rank assembles the globe and runs an unmodified checkpoint; the others take part only in the exchange. **Reference behavior** — validate the others against it. |
| `spatial` | `ModelTorchDistributed` | Ranks form an `h x w` mesh, each owning a rectangle; ACE's distributed transforms carry the coupling between them. |
| `ensemble` | `TorchDistributed` | Every rank holds the whole globe as its own batch member; the members are averaged on the way out. |

`auto` picks `single` unless `ace_h`/`ace_w` are declared and multiply to the
rank count. It will not invent a mesh: which factorization is right depends on
the model's transforms and on the machine, and guessing is how a run ends up
slow or wrong.

### The decomposition

`e3sm_emulator.decomposition` turns

```
local column -> global id -> (j, i) -> (owning rank, offset in tile)
```

into a plan computed once at initialization, so each step costs **one
all-to-all of exactly the values that have to move** — no rank assembles a
field it does not need, and nothing is broadcast. The reverse is the same plan
read backwards, so a round trip is exact by construction rather than by
agreement between two pieces of index arithmetic. All the fields go in one
exchange, because message count is what hurts at scale.

`single` is not a special case: it is a `1 x 1` tiling on N ranks, where the
all-to-all degenerates into a gather and a scatter. Ranks past the mesh own an
empty tile, and **do not load the checkpoint** — 64 atmosphere ranks should
not hold 64 copies of the weights.

`ensemble` is the one case that is not a permutation (every value goes
everywhere), so it uses an all-gather and holds a full globe per rank. That is
the price of data parallelism over a model whose sample is the globe, and it
is only worth paying when the ranks are doing genuinely different work with
it.

The plan **validates itself**: if the columns do not cover this rank's tile
exactly once, initialization fails with the counts. This is the one place
where being loud matters most — a mismatched grid produces a field with holes
in it, and a model will consume that and return something that looks fine.

`infer()` is **collective** whenever the model is distributed: every rank of
the component communicator must call it the same number of times in the same
order. That is why the component calls it unconditionally rather than behind
a rank test.

### Notes on ACE specifically

Read from `mahf708/ace` at `75d8de6` in July 2026; it is a moving target, so
re-check before relying on any of this.

- **Call the `Stepper`, not `modules[0]`.** Packing, normalization, residual
  prediction, correctors, prescribed SST, derived forcings and output
  unpacking all live in the Stepper. Bypassing it silently omits part of the
  learned timestep.
- **Initialize distributed and the device before loading the checkpoint.**
  `fme/core/step/single_module.py` does `module.to(get_device())` then
  `Distributed.get_instance()` then `wrap_module` while loading, so the order
  is: establish rank/device/process group, *then* `load_stepper`.
- **`spatial` is gated on upstream work.** ACE's `ModelTorchDistributed`, its
  distributed spherical transforms and `get_local_slices` all exist, but the
  two registered builders a deterministic ACE2 checkpoint instantiates —
  `SphericalFourierNeuralOperatorNet` and `SFNO-v0.1.0` — call
  `th.RealSHT`/`th.InverseRealSHT` **unconditionally**
  (`models/modulus/sfnonet.py:500`, `models/makani/sfnonet.py:479`). Only
  `NoiseConditionedSFNO` routes through `dist.get_sht()`. So `spatial` will
  build the mesh and hand each rank a shard to a module that still performs
  global transforms. Treat it as infrastructure under integration, not a mode
  to switch on today.
- **A global model can already run without any of this**, by giving the
  component few ranks (`NTASKS_ATM=1`). The coupler regrids and redistributes
  between components, so it performs the gather — correctly, and with no idle
  ranks, because the component never had them. `single` mode is the version of
  that which does not force the whole component down to one rank.
- **Have the component declare ACE's own grid to the coupler**, so the coupler
  does the interpolation. That is its job, already conservative and validated,
  and it leaves the exchange here a pure index permutation.
- **ACE is autoregressive** and holds prognostic state between calls. The
  Python object persisting across steps handles that naturally; writing and
  reading that state at an E3SM restart boundary is **unsolved**.

## Building

Everything except the stub is opt-in, and `emulator_common` builds and its
tests pass with no ML dependency installed at all.

```bash
cd components/emulators

./test                # core only
./test --python       # plus the embedded-Python backend

# or directly
cmake -S . -B build -DBUILD_EMULATOR_TESTS=ON -DSTANDALONE_MODE=ON \
      -DEMULATOR_ENABLE_PYTHON=ON
cmake --build build --parallel && (cd build && ctest --output-on-failure)
```

- `EMULATOR_ENABLE_PYTHON` needs the Python development headers
  (`Python3::Python`), plus numpy in that interpreter at run time. The
  emulator package directory is baked in as a `sys.path` entry, so a run does
  not have to set `PYTHONPATH`; `option.python_path` still wins.
- `EMULATOR_ENABLE_MPI` (`AUTO`, `ON`, `OFF`; default `AUTO`) controls only
  `InferenceContext`. Without it the context reports rank 0 of 1 and
  everything still builds, so the unit tests run with no launcher.
- The decomposition tests are plain `unittest` and need only numpy. They run
  N logical ranks as threads over the same `Comm` interface the real
  implementation satisfies, so a multi-rank exchange is tested in CI without
  MPI, torch or a launcher.
- The option defaults to `OFF`, so a CIME build is unaffected until a machine
  turns it on.

## Not done yet

Roughly in the order they are worth doing:

- **Wiring the coupling fields.** `EmulatorAtm` builds the backend and calls
  it every step, but `prepare_inputs`/`process_outputs` are still stubs: they
  need `init_coupling_indices` to resolve the MCT field lists first.
- **Restart.** ACE's prognostic state lives in a Python object across steps
  and is dropped at finalize.
- **Device-resident tensors.** The container is host memory only, so every GPU
  run pays a host round trip in both directions. This probably matters more to
  total GPU cost than the choice of runtime. It wants a memory space, a device
  ordinal and a stream on `Tensor`, then DLPack for Python — none of which
  changes the backend interface, which is the point of tensors being the only
  thing that crosses it.
- **Distribution semantics in the tensor metadata.** `{ncol, nlev}` says a
  local shape and nothing about how it relates to the global grid.
- **Ragged tiles.** `split_bounds` refuses a mesh finer than the grid rather
  than handing some rank an empty tile, because a model with a spatial
  receptive field cannot work with one.
- **ONNX Runtime and LibTorch backends**, if a model that exports cleanly ever
  wants them. The interface is ready; neither can express a collective, so
  neither helps the global-model case.
