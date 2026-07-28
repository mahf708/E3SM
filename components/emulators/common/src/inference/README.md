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
                   pure data parallel)      single / spatial)
```

| File | What it is |
| --- | --- |
| `tensor.*` | `Tensor`/`TensorMap`: named, shaped, owning-or-viewing buffers |
| `inference_config.*` | Named settings, free-form options, namelist parsing |
| `inference_context.*` | Ranks, rendezvous, this rank's columns, `agree_or_throw` |
| `inference_backend.*` | The backend interface and its lifecycle |
| `create_inference_backend.*` | `create_backend(config, context)` |
| `stub_inference_backend.*` | No-op backend, no ML dependency |
| `python_interpreter.*` | Embedded CPython session, GIL/FPE guards, `PyRef` |
| `python_inference_backend.*` | Calls a Python emulator in-process |
| `inference_error.hpp` | `InferenceError` and the check macro |

The Python side is `../../python/e3sm_emulator/`; tests are in `../../tests`.
Each file states its own reasoning; this page covers what spans several.

## Status

**Working:** the Python bridge, the coupling path through `EmulatorAtm`, a
column-local (`generic`) backend, and single-rank ACE as a numerical
reference.

What the tests actually exercise, since "covered" is easy to overclaim: the
bridge end to end against a numpy fixture, including its failure and rollback
paths; `EmulatorAtm` from `x2a` through a model and back into `a2x` on one
rank; the decomposition and the collective-failure semantics across N logical
ranks (threads, not processes); `generic`'s packing and output splitting
against a faked torch; and ACE's mode selection, mesh-ordering and
output-shape checks. What they do **not** exercise: loading a real TorchScript
module or checkpoint, any ACE stepper call, torch itself, and anything at all
across real MPI processes. All of them run with no MPI, torch or checkpoint
installed, which is what makes them worth having in CI and also exactly what
bounds what they prove.

**Refused on purpose:** ACE's `TorchDistributed` (it splits a batch of globes
and a coupled run supplies one), and `ace_mode=spatial`, which is implemented
but gated — see `e3sm_emulator/ace.py`.

**Never yet run:** nothing here has touched a real ACE checkpoint. There is no
test involving several MPI processes, `torch.distributed`, a component
communicator inside an MPMD job, GPU binding, a checkpoint load, a real ACE
timestep, or a restart. The passing test suite is evidence about local logic,
not about distributed integration. The decisive next step is one pinned ACE
checkpoint on two MPI ranks compared against one, then a restart comparison —
not another unit test.

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

// Each step. Tensors view the component's own memory; nothing is copied.
TensorMap inputs;
inputs.wrap("T", static_cast<const double*>(m_T.data()), {ncol, nlev});
TensorMap outputs;
outputs.wrap("dT", m_dT.data(), {ncol, nlev});
m_inference->infer(inputs, outputs);             // writes into m_dT

m_inference->finalize();                         // in finalize
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

`inference.input`/`output` name **coupling fields**, resolved once against the
`x2a` and `a2x` field lists MCT hands the component. Every mismatch is fatal
at initialization, because each one otherwise shows up as a partly filled
field — plausible numbers over part of the globe, zeros or stale values over
the rest, which nothing downstream would catch. The sharpest is **an input the
coupler does not carry**: it has no other source, so tolerating one is
permission to run the model on zeros, and
`inference.unsafe_allow_zero_filled_inputs` is the honest name for wanting
that anyway. An unmatched *output* is fine — a model may produce diagnostics
the coupler does not consume. See `EmulatorAtm::validate_coupling`.

## Why the context is the interesting type

`InferenceContext` carries the **component communicator's** rank, size,
per-node rank and a rendezvous, and **this rank's columns**.

The first exists because of a specific failure. ACE's `TorchDistributed` and
PhysicsNeMo's `DistributedManager` both discover their rank from
`SLURM_PROCID` / `SLURM_NTASKS` when nothing better is available, and in a
coupled run those describe the **whole job**. A process group built from them
waits forever for ocean and land ranks that will never call in. The Python
bridge publishes the component's own numbers into `RANK`, `WORLD_SIZE`,
`LOCAL_RANK`, `MASTER_ADDR` and `MASTER_PORT` — the variables those libraries
read *first* — so an unmodified upstream model initializes over exactly our
ranks. An explicit initializer in ACE would be cleaner than environment
archaeology, and is worth proposing.

The second exists because the two decompositions never line up. E3SM's is
chosen for load balance across the coupled job; a global model's is the
geometry of its own transforms. Reconciling them is a permutation, and
`e3sm_emulator.decomposition` computes it once.

## The python bridge

Only tensor *metadata* crosses the boundary:

```
Tensor (C++/Fortran memory)
  -> PyMemoryView_FromMemory(ptr, nbytes, PyBUF_READ|PyBUF_WRITE)
  -> numpy.frombuffer(view, dtype=float64).reshape(dims)     # still a view
```

through the ordinary Python API rather than numpy's C API on purpose: numpy is
then a **runtime** dependency only, so building E3SM never needs numpy
headers, and the arrays are still zero-copy. (pybind11 is the other way to
write this, and is what EAMxx's `eamxx_pysession.hpp` uses; the plain C API
keeps the build dependency down to the Python development headers, and the
Python-side protocol is identical either way.)

The backend imports a module — `e3sm_emulator.bridge` by default — calls
`create_emulator(config)` once, then `infer(inputs, outputs)` per step:

```python
def infer(self, inputs, outputs):
    outputs["dT"][:] = self.model(inputs["T"])
```

Inputs are read-only views, outputs are writable views of the component's own
arrays. **Zero-copy ends at the numpy view**: a model that moves the array
onto a GPU, or converts it to float32, pays for that copy like anybody else.
One interpreter per process, holding the GIL for each `infer`; separate MPI
ranks are separate processes, so ranks do not contend for it.

## Running across ranks

**The backend runs a model in one process.** How that relates to the other
ranks is the model's business, and Python is where it is decided — which is
why the Python bridge came first. It is the only backend where a model can
contain a collective at all; neither ONNX nor TorchScript can express one.

**Column-local** models (`y_i = f(x_i)`) are the easy case: `generic` gives
every rank its own weights and its own columns, and communicates nothing.
Adding ranks adds throughput until the per-rank batch stops filling a device —
note where E3SM layouts sit: ne30pg2 is 21,600 columns, so ~1,350 ranks leaves
16 columns per rank.

**Globe-per-sample** models are the hard case, and `e3sm_emulator.ace`
explains the two modes and why one of them is gated.

### The decomposition

`e3sm_emulator.decomposition` turns

```
local column -> global id -> (j, i) -> (owning rank, offset in tile)
```

into a plan computed once, so each step costs **one all-to-all of exactly the
values that have to move** — no rank assembles a field it does not need, and
nothing is broadcast. All the fields travel in one exchange, because message
count is what hurts at scale. The reverse is the same plan read backwards, so
a round trip is exact by construction rather than by agreement between two
pieces of index arithmetic.

`single` is not a special case: it is a `1 x 1` tiling on N ranks, where the
all-to-all degenerates into a gather and a scatter. Ranks past the mesh own an
empty tile and **do not load the checkpoint** — 64 atmosphere ranks should not
hold 64 copies of the weights.

`TorchComm` implements the exchange with one `isend`/`irecv` per peer, so a
rank posts up to `P - 1` messages. `single` mode is a gather and cheap; a
blocked E3SM decomposition talks to few peers, a round-robin one to all of
them. The scalable version is `MPI_Alltoallv` — see below.

`infer()` is **collective** whenever the model is distributed: every rank must
call it the same number of times in the same order, which is why the component
calls it unconditionally rather than behind a rank test.

### Failing together

A component holds inconsistent state the moment one rank fails alone. In
`single` mode only one rank loads the checkpoint, so a bad path raises *there*
while the others finish construction happily, and the next collective hangs
instead of reporting the real error.

`Comm.agree()` turns that into an error on every rank, carrying the failing
rank's message so a healthy rank is not sent to the wrong log; the rank that
actually failed re-raises its own exception, type and traceback intact.
`run_where()` wraps the work only some ranks do, and `agree_or_throw` is the
same idea in C++ for the pre-flight checks.

The boundary has to be drawn one operation wider than feels necessary — the
shape check in `stack_output_tiles`, the row-count checks inside
`PermutationExchange`, the packing that immediately precedes an exchange. Each
of those can fail on one rank alone, and each sits next to the communication
it protects.

**What a healthy run pays.** `infer()` reaches four agreement points per
timestep, multiplied by every step of a run. `agree()` therefore starts with
`any_true()` — one `all_reduce` of a single integer — and returns there when
nothing is wrong; the text is gathered only once somebody has some. One
`Comm.allgather` is *two* torch collectives, which is why the test pinning
this counts at the torch level against a recording stub.

## Building

Everything except the stub is opt-in, and `emulator_common` builds and its
tests pass with no ML dependency installed at all.

```bash
cd components/emulators
./test                # core only
./test --python       # plus the embedded-Python backend
```

- `EMULATOR_ENABLE_PYTHON` (default `OFF`) needs the Python development
  headers, plus numpy in that interpreter at run time. The emulator package
  directory is baked in as a `sys.path` entry, so a run does not have to set
  `PYTHONPATH`; `inference.python_path` still wins. Off by default, so a CIME
  build is unaffected until a machine turns it on.
- `EMULATOR_ENABLE_MPI` (`AUTO`, `ON`, `OFF`; default `AUTO`) controls only
  `InferenceContext`. Without it the context reports rank 0 of 1 and
  everything still builds, so the unit tests run with no launcher.
- The python tests are plain `unittest` and need only numpy. The multi-rank
  ones run N logical ranks as threads over the same `Comm` interface the real
  implementation satisfies, so a multi-rank exchange is tested in CI without
  MPI, torch or a launcher. The fake cluster **reports deadlocks rather than
  rescuing them**: a rank that leaves while the others wait at a collective
  makes them time out, and the harness raises `RankDivergence` naming who was
  stuck. A forgiving harness would be worse than none — an earlier version of
  this released the waiting ranks, which made the owner-failure tests pass
  against deliberately broken code.

## Not done yet

Roughly in order. The first three stand between this and a run you would trust.

- **Restart.** `AceEmulator.state_for_restart()` returns the prognostic state
  on columns; nothing writes or reads it. Until it does, a restarted run
  continues a different atmosphere.
- **An end-to-end test on real ranks.** `EmulatorAtm` is driven through import
  → infer → export on one rank, but nothing has run several MPI processes
  against a checkpoint.
- **A coexistence test with another embedded-Python host.** The interpreter is
  reference counted and never finalized, and a failed initialization now
  unwinds under the GIL, both so that EAMxx's `PySession` and this backend can
  share a process. Neither property has been exercised with EAMxx actually
  present, and the failure it guards against — releasing Python references on
  a thread that does not hold the GIL — is silent until it is not.
- **Verify a distributed ACE numerically** before anybody trusts `spatial`:
  identical one-step results at 1, 2, 4 and 8 ranks against `ace_mode=single`,
  then multistep trajectories, then restarts — with communication, H2D/D2H and
  model time measured separately.
- **`MPI_Alltoallv` for the redistribution.** Duplicate the component
  communicator in C++, expose the Fortran handle, rebuild it with
  `mpi4py.MPI.Comm.f2py`. Replaces `P - 1` point-to-point operations per rank
  with the machine's tuned collective and drops the extra gloo group. PyTorch
  would still need its own group for ACE's *internal* collectives.
- **Have the component declare ACE's own grid to the coupler**, so the coupler
  does the interpolation — its job, already conservative and validated — and
  the exchange here stays a pure index permutation.
- **An ensemble contract**, which is what would make data parallelism over a
  globe-per-sample model meaningful, and what `TorchDistributed` needs.
- **Device-resident tensors.** The container is host memory only, so every GPU
  run pays a host round trip in both directions. This probably matters more to
  total GPU cost than the choice of runtime. It wants a memory space, a device
  ordinal and a stream on `Tensor`, then DLPack for Python — none of which
  changes the backend interface, which is the point of tensors being the only
  thing that crosses it.
- **Distribution semantics in the tensor metadata.** `{ncol, nlev}` says a
  local shape and nothing about how it relates to the global grid.
- **Ragged tiles.** `split_bounds` refuses a mesh finer than the grid rather
  than handing some rank an empty tile.
- **ONNX Runtime and LibTorch backends**, if a model that exports cleanly ever
  wants them. The interface is ready; neither can express a collective, so
  neither helps the global-model case.
