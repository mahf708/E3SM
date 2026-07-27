# Emulator inference layer

This directory holds the machinery an emulated component uses to evaluate a
trained model: a memory-managed data container, a backend-agnostic interface,
and the backends that implement it (a dependency-free stub, an embedded-Python
bridge, LibTorch/TorchScript, and ONNX Runtime).

Nothing here knows about atmospheres, land, grids or coupling. A component
packs its fields into tensors, calls `infer`, and reads the results back —
usually into memory it already owns.

**The backends are local inference engines.** Each one loads a model in one
process and evaluates it on the tensors that process hands it. How that relates
to the other ranks of a component is a separate question, answered by an
`InferenceExecutor` and its execution policy — today only *one replica per
rank*, which is correct for column-local models and nothing else. Read
"Running across ranks" below before assuming anything about scaling.

```
                     component (EmulatorAtm, EmulatorLnd, ...)
                                      |
                     InferenceContext (comm, rank, node, device)
                                      v
                     InferenceExecutor   <- execution policy
                        (local_replica)     (how ranks cooperate)
                                      |
                        TensorMap in / TensorMap out
                                      v
   InferenceConfig  ->  InferenceBackend  <-  BackendRegistry ("stub", "onnx", ...)
                          (local engine)
                                      |
             +------------+-----------+-----------+--------------+
             |            |                       |              |
           stub        python                  libtorch        onnx
                  (embedded CPython)        (TorchScript)   (ONNX Runtime)
```

## Files

| File | What it is |
| --- | --- |
| `tensor.hpp/.cpp` | `Tensor` (the data container), `TensorMap`, `TensorSpec`, `DType` |
| `inference_config.hpp/.cpp` | `InferenceConfig`: core settings, free-form options, text parsing |
| `inference_context.hpp/.cpp` | `InferenceContext`: rank, node, device — MPI-free |
| `inference_context_mpi.hpp/.cpp` | Building a context from a communicator (the only TU that includes `<mpi.h>`) |
| `inference_backend.hpp/.cpp` | The `InferenceBackend` interface (a **local** engine), lifecycle, validation, flat-array path |
| `inference_executor.hpp/.cpp` | `InferenceExecutor`: how ranks cooperate; `local_replica` |
| `inference_backend_registry.hpp/.cpp` | String-keyed factory registry |
| `create_inference_backend.hpp/.cpp` | `create_backend(...)` conveniences |
| `stub_inference_backend.hpp/.cpp` | Synthetic backend, no ML dependency |
| `python_interpreter.hpp/.cpp` | Embedded CPython session, GIL/FPE guards, `PyRef` |
| `python_inference_backend.hpp/.cpp` | Calls a Python emulator in-process |
| `torch_inference_backend.hpp/.cpp` | TorchScript via LibTorch |
| `onnx_inference_backend.hpp/.cpp` | ONNX Runtime |
| `inference_error.hpp` | `InferenceError` and the check macro |

Tests are in `../../tests` (`test_inference_*.cpp`), the python-side fixtures
and the model generators in `../../tests/fixtures`, and a command line harness
in `../../tools/inference_demo.cpp`.

## The data container

`Tensor` is a named, typed, shaped buffer that **either owns its memory or
views memory owned by somebody else**. The second case is the important one:
a component hands its own field or coupling arrays to a backend with no copy.

```cpp
// Owning: 64-byte-aligned, zero-initialized.
Tensor t("dT", {ncol, nlev}, DType::FLOAT32);

// Viewing a component's own array (nothing allocated, nothing copied).
TensorMap inputs;
inputs.wrap("T", static_cast<const double*>(m_T.data()), {ncol, nlev});
TensorMap outputs;
outputs.wrap("dT", m_dT.data(), {ncol, nlev});
backend->infer(inputs, outputs);   // writes straight into m_dT
```

Design points, and why:

- **Move-only, `clone()` for copies.** An accidental deep copy of a
  multi-megabyte field cannot hide inside a function signature.
- **Views over `const` memory are read-only.** `data()` asks for write access
  and throws for such a view; `cdata()`/`cflat()` read without asking. Input
  fields handed to a model are const views, so a model cannot corrupt state.
- **Element-type conversion is a first-class operation.** E3SM fields are
  `real(r8)`; most trained models are float32. `copy_from` converts, and the
  ONNX and LibTorch backends convert through a scratch buffer that is
  allocated once and reused (`resize` only reallocates when it must grow, so
  per-step batch changes do not cause malloc traffic).
- **Row-major, leading dimension is the batch.** Column-major Fortran arrays
  correspond to reversed dims: `a(nlev,ncol)` is `{ncol, nlev}`.
- **`TensorMap` is ordered *and* name-addressable**, because ONNX and Python
  are name-based while TorchScript's `forward(a, b)` is positional.
- **`TensorSpec`** describes what a model wants — `T[-1,72]:float32`, where
  `-1` is the dynamic (batch) extent. Specs come from configuration or, for
  ONNX, from the model itself, and mismatches are reported with both the
  expected and the actual shape.

## Adding a model to a component

```cpp
#include "inference_executor.hpp"
#include "inference_context_mpi.hpp"   // when built with MPI

// In the component's init: read the inference settings straight out of the
// component namelist, where they are prefixed with `inference.`
auto config = InferenceConfig::from_file_with_prefix(m_input_file, "inference.");

// Tell the inference layer where it is running.  m_comm is the *component*
// communicator, which is what the component already holds.
auto context = make_context_from_comm(m_comm);
// On a machine with accelerators, say which one this rank owns:
// assign_device_round_robin(context, devices_per_node);

m_executor = create_executor(config, context);
m_executor->initialize();

// Each step: wrap what the model needs and run.  Shapes are this rank's.
TensorMap inputs;
inputs.wrap("T",  static_cast<const double*>(m_T.data()),  {ncol, nlev});
inputs.wrap("ps", static_cast<const double*>(m_ps.data()), {ncol, 1});
TensorMap outputs;
outputs.wrap("dT", m_dT.data(), {ncol, nlev});
if (!m_executor->infer(inputs, outputs)) { /* handle a step failure */ }

// In the component's finalize:
m_executor->finalize();
```

Going through an executor rather than calling `create_backend` directly costs
nothing under `local_replica` — `infer()` forwards straight to the backend —
and it is what lets the parallel strategy change from a namelist later without
touching the component. `create_backend` remains available for serial code,
tools and tests.

with, in `atm_in` (or any file the component already reads):

```
nx: 90
ny: 45
inference.backend:    onnx
inference.model_path: /path/to/atm_emulator.onnx
inference.intra_op_threads: 4
```

For code that already thinks in flat arrays there is a convenience overload,
`infer(const double* in, double* out, int batch_size)`, which wraps both
buffers as `[batch_size, channels]` tensors using `input_channels` /
`output_channels`.

## Configuration

Line-oriented `key: value`, deliberately close to the parsing the emulator
components already do on their `*_in` files. `#` and `!` start comments.

```
backend:         onnx           # registry key: which local engine
execution_policy: local_replica # how the ranks cooperate (default)
model_path:      atm.onnx
input:           T[-1,72]:float32    # repeatable, ordered
output:          dT[-1,72]:float32
input_channels:  73             # only for the flat-array path
output_channels: 72
verbose:         true
device:          cuda           # unknown keys become backend options
option.intra_op_threads: 4      # `option.` prefix is optional
```

Read it with `InferenceConfig::from_file`, or from a component namelist with
`from_file_with_prefix(path, "inference.")`. Settings can also be applied one
at a time with `config.apply(key, value)` — that is what the demo's `--set`
does — or set programmatically (`config.backend = "stub"`,
`config.set("mode", "copy")`).

A YAML front end (ekat, as EAMxx uses) can be layered on later: it would
produce an `InferenceConfig` and nothing else would change.

## Running across ranks

Each backend runs a model **in one process**. An `InferenceExecutor` says what
the other ranks are doing, and the only policy implemented today is:

| Policy | What it does | Status |
| --- | --- | --- |
| `local_replica` | Every rank loads its own copy of the model and infers on its own local columns. No communication. | **implemented**, the default |
| `gpu_group` | Ranks sharing an accelerator gather their batches onto one owner, infer once, scatter back. | not implemented |
| `spatial_distributed` | Every rank evaluates part of one global model, with collectives inside the model. | not implemented |

`local_replica` is correct **only** when a column's output depends on that
column alone — `y_i = f(x_i)`. Pointwise parameterizations, column MLPs and
per-column vertical networks qualify. Models that need neighbouring columns, a
global receptive field (spherical transforms, global attention, graph networks
over the whole grid), or a shard of one global input **do not**, and running
them under `local_replica` produces wrong answers rather than an error: the
layer cannot tell from a model file what its receptive field is.

Two practical notes for the cases that are not implemented yet:

- A **global model can run today** without any of this, by giving the
  component few ranks (`NTASKS_ATM=1` or a handful). The coupler already
  regrids and redistributes between components, so it performs the gather —
  correctly, and with no idle ranks, because the component never had them.
- **Several ranks per GPU is refused, not guessed.** With `device: cuda` and
  more than one rank of the component on a node, `create_executor` throws
  unless an ordinal is assigned, because the alternative is every rank quietly
  loading the model onto device 0. Use `assign_device_round_robin()`, set
  `device_id`, or give the component one rank per device.

`InferenceExecutor::infer()` is documented as **collective**: for any policy
but `local_replica`, every rank of the context's communicator must call it the
same number of times in the same order. That is why it is a different type
from the local `InferenceBackend::infer()` — one is an ordinary call, the other
can hang the run if a rank skips it.

### Thread counts

`intra_op_threads` and `inter_op_threads` **default to 1** in the ONNX and
Torch backends. Left to themselves both runtimes open roughly one thread per
core, which in an MPI run means every rank on the node claims every core.
Raise them deliberately, together with a PE layout that leaves the threads
somewhere to run; `auto` hands the decision back to the runtime. LibTorch's
thread counts are process-wide, so the first backend to ask wins and later
disagreements are reported rather than obeyed.

## Choosing a backend

| | stub | python | onnx | torch |
| --- | --- | --- | --- | --- |
| Extra dependency | none | Python + numpy | libonnxruntime (~15 MB) | LibTorch (~2 GB) |
| Model format | none | whatever Python loads | `.onnx` | TorchScript `.pt` |
| Knows its own signature | no | no | **yes** | no (positional) |
| Zero-copy inputs (CPU) | n/a | **yes** (numpy views) | yes (matching dtype) | yes (matching dtype) |
| Zero-copy inputs (GPU) | n/a | no | no | no |
| Custom layers / control flow | n/a | **anything** | export-limited | good |
| In-process Python | no | yes | no | no |

**Zero-copy is a CPU-only claim.** The container is host memory today, so every
GPU execution stages host→device on the way in and device→host on the way out,
in all three backends. For a component whose fields already live on the device
that round trip can cost more than the model.

Recommendation: **ONNX Runtime for production**, because the model file
carries its own input/output names, shapes and precision — so a shape or
precision mismatch is a clear error instead of silently wrong results — and
because it adds one modest shared library and no interpreter. **The Python
bridge for development and for models that will not export**, which in
practice means most research emulators, and the ones with data-dependent
control flow. **The stub for CI and for bringing up a new component**, since
it exercises every code path around the model with nothing installed.

LibTorch is worth it when a model must stay in TorchScript, but it is a heavy
and version-sensitive dependency, and its lack of an introspectable signature
means the argument order and precision have to be written down by hand in the
configuration.

### The python bridge, in more detail

The model stays in Python; only tensor *metadata* crosses the boundary. Arrays
are handed over as numpy views of E3SM memory:

```
Tensor (C++/Fortran memory)
  -> PyMemoryView_FromMemory(ptr, nbytes, PyBUF_READ|PyBUF_WRITE)
  -> numpy.frombuffer(view, dtype=...).reshape(dims)     # still a view
```

This goes through the ordinary Python API rather than numpy's C API on
purpose: numpy is then a **runtime** dependency only, so building E3SM never
needs numpy headers, and the arrays are still zero-copy. (pybind11 would be
the other way to write this backend — it is what EAMxx's
`share/core/eamxx_pysession.hpp` uses. The plain C API was chosen here to keep
the build dependency down to the Python development headers; the Python-side
protocol is identical either way, so switching later would not affect users.)

A Python emulator provides either style, detected from its signature:

```python
# in-place: writes into views of E3SM memory, no copies at all
def infer(self, inputs, outputs):
    outputs["dT"][:] = self.model(inputs["T"])

# return: more convenient, one copy per output
def infer(self, inputs):
    return {"dT": self.model(inputs["T"])}
```

See `../../tests/fixtures/emulator_*.py` for complete working examples.

**What this backend does and does not give you.** One interpreter per process,
holding the GIL for the duration of each `infer`. Separate MPI ranks are
separate processes with separate interpreters, so ranks do not contend for a
GIL and independent model replicas run fine side by side. What it does *not*
provide is a distributed inference facility: the backend does not accept,
duplicate or validate a communicator, does not initialize `torch.distributed`,
and does not propagate a collective failure safely. A model that wants to be
MPI-parallel can be handed a communicator handle through the option map and
rebuild it with `mpi4py` — that is a **convention available to model code**,
and the model owns everything that follows from it.

The Python bridge is nevertheless the most promising route to genuinely
distributed inference, because it is the only one where the model can contain
collectives at all: neither ONNX nor TorchScript can express them portably.

**Zero-copy ends at the numpy view.** A Python model that moves the array onto
a GPU, or converts it to a framework tensor with a different layout, pays for
that copy like anybody else.

## Adding a backend

Implement three methods and register a factory. Nothing else needs to change —
no enum, no switch, no edits to this directory:

```cpp
class MyBackend : public InferenceBackend {
public:
  explicit MyBackend(const InferenceConfig& c) : InferenceBackend(c) {}
  std::string name() const override { return "MyRuntime"; }
protected:
  void init_impl() override { /* load the model */ }
  bool infer_impl(const TensorMap& in, TensorMap& out) override { /* run */ }
  void final_impl() override { /* release */ }
};

BackendRegistry::instance().register_backend(
    "my_runtime",
    [](const InferenceConfig& c) { return std::make_shared<MyBackend>(c); });
```

The built-in backends are registered explicitly (in
`inference_backend_registry.cpp`) rather than by static initializers in each
translation unit: `emulator_common` is a static library, and a linker may drop
an object file whose symbols are never referenced, which would silently
un-register a backend.

If a backend can interrogate its model, override `input_specs()` /
`output_specs()` so callers and error messages get the real signature. If it
stages data through its own buffers and can therefore accept a different
precision than the model's, override `converts_element_types()`.

## Building

Everything except the stub is opt-in, and `emulator_common` builds and its
tests pass with no ML dependency installed at all.

```bash
cd components/emulators

# Core only (no ML dependency)
./test

# With backends
cmake -S . -B build -DBUILD_EMULATOR_TESTS=ON -DSTANDALONE_MODE=ON \
      -DEMULATOR_ENABLE_PYTHON=ON \
      -DEMULATOR_ENABLE_ONNXRUNTIME=ON -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
      -DEMULATOR_ENABLE_TORCH=ON -DTORCH_ROOT=/path/to/libtorch
cmake --build build --parallel && (cd build && ctest --output-on-failure)
```

- `EMULATOR_ENABLE_MPI` (`AUTO`, `ON`, `OFF`; default `AUTO`) controls only the
  context helper `make_context_from_comm`. The engines and their unit tests
  stay MPI-free either way, so the suite runs without a launcher; with MPI on,
  one additional test runs under `mpiexec` on two ranks.
- `EMULATOR_ENABLE_PYTHON` needs the Python development headers
  (`Python3::Python`), plus numpy in that interpreter at run time.
- `EMULATOR_ENABLE_ONNXRUNTIME` uses `cmake/FindONNXRuntime.cmake`; point
  `ONNXRUNTIME_ROOT` at an unpacked
  [release](https://github.com/microsoft/onnxruntime/releases).
- `EMULATOR_ENABLE_TORCH` uses `cmake/FindLibTorch.cmake`, which by default
  goes through PyTorch's own `TorchConfig.cmake`. If that aborts because a
  CUDA-enabled wheel cannot find a CUDA toolkit, add
  `-DLIBTORCH_USE_CMAKE_PACKAGE=OFF` to search for the headers and libraries
  directly (CPU TorchScript needs only `c10` and `torch_cpu`).

The backend model fixtures used by the tests are generated at configure time
by `tests/fixtures/make_onnx_model.py` (needs the `onnx` package) and
`make_torch_model.py` (needs `torch`); a test whose fixture cannot be built is
skipped with a message rather than failing the build.

## The demo harness

`emulator_inference_demo` builds a backend, reports what the model expects,
feeds it synthetic columns and times the steps — useful for checking that a
model loads before wiring it into a component, and for comparing backends on
the same model:

```bash
./build/common/tools/emulator_inference_demo --list
./build/common/tools/emulator_inference_demo \
    --backend onnx --model atm.onnx --columns 4608 --steps 50
./build/common/tools/emulator_inference_demo \
    --config atm_in --prefix inference. --columns 384
```

## Notes from bringing this up

Findings worth keeping, since they shaped the code:

- **The embedded interpreter cannot be restarted.** Finalizing CPython and
  starting it again fails as soon as a C extension is involved: numpy 2.x
  says "cannot load module more than once per process", and PyTorch behaves
  the same way. `PyInterpreter::finalize()` therefore drops a customer but
  leaves the interpreter running; what actually holds memory (the model and
  its weights) is released when the backend drops its Python references.
  `shutdown()` exists for the rare caller that really wants Py_Finalize.
- **Importing numpy raises benign FPEs**, which kills a build that traps them
  (an E3SM debug build). Imports are wrapped in an `FpeGuard`, the same thing
  EAMxx's `PySession::safe_import` does.
- **ONNX Runtime's `GetInputTypeInfo(i).GetTensorTypeAndShapeInfo()` dangles.**
  The shape info borrows from the `TypeInfo` temporary; reading element types
  through it yields garbage. The `TypeInfo` has to be kept in a named
  variable.
- **A failing `TorchConfig.cmake` raises a CMake `FATAL_ERROR`**, which
  `find_package(... QUIET)` cannot suppress — so `FindLibTorch.cmake` cannot
  try the config package and fall back automatically. The choice is an
  explicit option instead.
- **Plumbing overhead is not the problem — in one process.** With a trivial
  model at 4608 columns × 3 levels, single process, one CPU core, per-step cost
  was ≈0.02 ms for the Python bridge, ≈0.03 ms for ONNX Runtime and ≈0.06 ms
  for LibTorch. That measures the wrapper, not a model and not a parallel run:
  it says the bridge is cheap and that the Python bridge is *not* inherently
  slower than the compiled runtimes once arrays are views instead of copies.
  It says nothing about scaling. Note also where E3SM layouts actually sit —
  ne30pg2 is 21,600 columns, so ~1,350 ranks leaves **16 columns per rank**,
  a batch too small to be worth a GPU kernel launch. Real numbers need strong
  and weak scaling, ranks per node, ranks per GPU, initialization separated
  from stepping, and maximum rather than mean rank latency.

## Not done yet

Roughly in the order they are worth doing:

- **Device-resident tensors.** The container is host memory only, so every GPU
  run pays a host round trip in both directions. This probably matters more to
  total GPU cost than the choice of runtime. It wants a memory space (host,
  pinned, device), a device ordinal and a stream on `Tensor`, then ONNX
  `IOBinding`, `torch::from_blob` on device pointers, and DLPack for Python.
  None of that changes the `InferenceBackend` interface — which is the point of
  tensors being the only thing that crosses it.
- **The `gpu_group` policy**, for layouts with several ranks per accelerator:
  split the communicator by node and device, elect an owner, gather ragged
  local batches, infer once, scatter back, reusing the buffers.
- **Distribution semantics in the tensor metadata.** `{-1, 72}` says a local
  shape and nothing about how it relates to the global grid. A spatial model
  needs replicated-versus-sharded, global IDs, global shape and halo width.
- **`spatial_distributed`**, once a concrete emulator needs it, prototyped
  through the Python backend with a model that already speaks
  `torch.distributed`.
- Wiring into `EmulatorAtm` — see "Adding a model to a component" for the
  pattern; the hooks in `atm.cpp` (`prepare_inputs`, `process_outputs`) are
  where it goes.
- Fortran/C entry points for the inference layer (only the C++ API exists;
  the emulator components that use it are C++).
- ONNX Runtime outputs are copied out of runtime-owned buffers. Binding
  caller memory as pre-allocated outputs would remove that copy for models
  with fully static output shapes.
- ONNX Runtime global thread pools (an `Env`-level threading option plus
  `DisablePerSessionThreads`), which matter once one process holds several
  sessions.
- No batching or queuing across components: each `infer` call is synchronous.
