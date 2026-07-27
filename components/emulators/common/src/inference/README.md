# Emulator inference layer

Machinery for an emulated component to evaluate a trained model: a data
container, a backend interface, a dependency-free stub, and an embedded-Python
bridge. Nothing here knows about atmospheres, land, grids or coupling. A
component packs its fields into tensors, calls `infer`, and reads the results
back — usually into memory it already owns.

## What this is, and what it is not

**Working:** the Python bridge, the coupling path through `EmulatorAtm`, a
column-local (`generic`) backend, and single-rank ACE as a numerical
reference. All of it is covered by tests that run without MPI, torch or a
checkpoint.

**Not working, and deliberately so:** this is *not* distributed-ACE support.
`TorchDistributed` was asked for and has been **removed** — it splits a batch
of globes across ranks and a coupled run supplies one, so it cannot accelerate
a single trajectory (details below). `ModelTorchDistributed` is **implemented
but gated off**, because the builders a deterministic ACE2 checkpoint
instantiates still call non-distributed spherical transforms, and running
anyway produces plausible wrong fields rather than an error.

**Never yet run:** nothing here has touched a real ACE checkpoint. There is no
test involving several MPI processes, `torch.distributed`, a component
communicator inside an MPMD job, several ranks per node, GPU binding, a
checkpoint load, a real ACE timestep, or a restart. The passing test suite is
evidence about local logic, not about distributed integration. The decisive
next step is one pinned ACE checkpoint on two MPI ranks, compared against one,
followed by a restart comparison — not another unit test.

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

`inference.input` and `inference.output` name **coupling fields**: they are
resolved once against the `x2a` and `a2x` field lists MCT hands the component,
and each step `prepare_inputs`/`process_outputs` gather and scatter between
`rAttr(nflds, lsize)` and the field-contiguous layout a model wants.

Mismatches are fatal at initialization, because every one of them otherwise
shows up as a partly filled field — plausible numbers over part of the globe,
zeros or stale values over the rest, which nothing downstream would catch:

- the coupler's `field_size` disagreeing with this rank's column count (the
  earlier code took the shorter of the two, which is how that failure hides);
- a field list whose length disagrees with its attribute vector, which makes
  every resolved row index wrong by an unknown amount;
- a resolved row outside the buffer;
- **a declared input the coupler does not carry.** An unmatched input has no
  other source, so tolerating one is permission to run the model on zeros.
  `inference.unsafe_allow_zero_filled_inputs: true` says a zero-filled run is
  genuinely wanted — the honest name, since there is no second source for
  those buffers today. A declared *output* is different — a model may legitimately
  produce diagnostics the coupler does not consume — so that is reported and
  not fatal. If internal-state or computed inputs are wanted later, the honest
  form is an explicit source per field (`inference.input.<name>.source:
  coupling | computed | state`) rather than a blanket permission.

No coupling buffers at all is accepted only for the stub backend, which runs
no model and is exactly the case where unfed inputs are the point. A real
backend with declared inputs and nothing feeding them is refused for the same
reason as any other zero-filled input.

All of these checks are **agreed across the component communicator** before
any rank builds a backend. A mismatch usually holds on one rank only, and a
lone rank throwing while the others are already inside collective model
initialization hangs the run rather than failing it. Validation therefore
collects its complaints, reduces them over `m_comm`, and either every rank
throws or none does — which is also why validation now runs *before*
`create_backend`, not after it.

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
reject. `e3sm_emulator.ace` handles these, two ways:

| `ace_mode` | ACE backend | Status |
| --- | --- | --- |
| `single` | `NonDistributed` | **Supported.** One rank assembles the globe and runs an unmodified checkpoint; the others take part only in the exchange. The numerical reference. |
| `spatial` | `ModelTorchDistributed` | **Gated.** Ranks form an `h x w` mesh, each owning a rectangle. Refused unless `ace_unsafe_allow_unverified_spatial` is set — see below. |

`auto` picks `single` unless `ace_h`/`ace_w` are declared and multiply to the
rank count. It will not invent a mesh: which factorization is right depends on
the model's transforms and on the machine, and guessing is how a run ends up
slow or wrong.

**`TorchDistributed` is deliberately not offered.** It replicates the model
and splits a *batch* across ranks, so it only helps when the component
supplies several independent globes. A coupled E3SM trajectory has one: every
rank would receive the same globe and the same deterministic weights and
compute the same answer N times. Reducing those to a mean and storing it as
the autoregressive state would also collapse any ensemble that did exist,
after a single step. It becomes worth wiring when the coupling contract grows
an explicit ensemble — separate initial states, random states, prescribed
forcings, autoregressive states and outputs per member — and
`ReplicaExchange` is the (tested, currently unreachable) piece that will serve
it. Do not average unless the component asks for an ensemble mean.

**Why `spatial` is gated.** ACE's `ModelTorchDistributed` will build the mesh
and hand each rank a rectangle, but the two builders a deterministic ACE2
checkpoint instantiates — `SphericalFourierNeuralOperatorNet`
(`registry/sfno.py`) and `SFNO-v0.1.0` — construct `torch_harmonics.RealSHT`
and `InverseRealSHT` **unconditionally** (`models/modulus/sfnonet.py:500`,
`models/makani/sfnonet.py:479`) rather than through `Distributed.get_sht()`.
Only `NoiseConditionedSFNO` is wired through the distributed constructors. A
sharded input therefore reaches a module that still performs global
transforms, and the result is plausible numbers rather than an error. So the
mode raises at initialization with that explanation attached, and
`ace_unsafe_allow_unverified_spatial` is the acknowledgement — named so that
nobody leaves it in a production namelist without noticing what it turns off.
Inspecting the loaded checkpoint's builder and permitting only a known-good
set would be better protection than an opt-in flag; that is the right fix once
one builder is verified. The order of work before setting
it: route every global operator in the builder through the distributed
constructors, then check one-step *and* multistep output against
`ace_mode=single` on the same checkpoint at 1, 2, 4 and 8 ranks.

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

**In spatial mode the partition is ACE's, not ours.** ACE slices its spatial
dimensions with `torch_harmonics.distributed.compute_split_shapes`, and our
columns have to land in exactly those rows. So each rank asks ACE for the
slice it was given (`Distributed.get_local_slices`), the answers are gathered,
and the routing plan is built from those — one partitioning algorithm rather
than two that can drift. `split_bounds` reproduces
`compute_split_shapes` (remainder to the **low**-numbered ranks, which is also
`numpy.array_split`'s convention) and is pinned by a unit test, but it is the
reference and the fallback, not the authority. Getting that convention
backwards is invisible on an even split and silently wrong on every uneven
one.

The plan **validates itself**: if the columns do not cover this rank's tile
exactly once, initialization fails with the counts, and the reconstructed mesh
is cross-checked against the slice ACE reported for this rank. This is the one
place where being loud matters most — a mismatched grid produces a field with
holes in it, and a model will consume that and return something that looks
fine.

**The exchange is a genuine all-to-all**, and `TorchComm` implements it with
one `isend`/`irecv` per peer, so a rank posts up to `P - 1` messages. Packing
every field into one exchange keeps the message *count* independent of the
number of variables, but not of `P`. `single` mode is cheap (only rank 0
receives, so it is a gather), and a blocked E3SM decomposition talks to few
peers; a round-robin one talks to all of them. The scalable version is to use
the communicator this layer already holds: duplicate the component `MPI_Comm`
in C++, expose its Fortran handle, rebuild it with `mpi4py.MPI.Comm.f2py` and
call `MPI_Alltoallv`, which gets the machine's tuned algorithm and drops the
extra Gloo group from the redistribution path entirely. PyTorch would still
need its own group for ACE's *internal* collectives; E3SM-to-model
redistribution does not. Not done — see "Not done yet".

`infer()` is **collective** whenever the model is distributed: every rank of
the component communicator must call it the same number of times in the same
order. That is why the component calls it unconditionally rather than behind
a rank test.

### Notes on ACE specifically

**Pinned to `mahf708/ace` at `75d8de6bcb0a30192720a16fc99f4eca0f54dbd2`**, in
July 2026. This is not decoration: ACE's stepper and distributed APIs both
move, and a review of this code against a different revision reported that
`load_stepper` did not exist and that `step()` returned a `StepOutput` —
neither of which is true at the pinned commit, where `load_stepper` is
`fme/ace/stepper/single_module.py:1837` and `step()` returns a `TensorDict`.
Both readings can be right about their own revision, which is the whole
argument for pinning. `PINNED_ACE_COMMIT` in `ace.py` records it, and the
adapter checks the two API points it depends on and names the pin when they
are missing rather than failing somewhere deep inside a load.

- **Call the `Stepper`, not `modules[0]`.** Packing, normalization, residual
  prediction, correctors, prescribed SST, derived forcings and output
  unpacking all live in the Stepper. Bypassing it silently omits part of the
  learned timestep.
- **Initialize distributed and the device before loading the checkpoint.**
  `fme/core/step/single_module.py` does `module.to(get_device())` then
  `Distributed.get_instance()` then `wrap_module` while loading, so the order
  is: establish rank/device/process group, *then* `load_stepper`.
- **`Distributed.get_instance()` refuses a multi-rank instance outside
  `Distributed.context()`** (`distributed.py:125`), and that context also owns
  the shutdown. A component's init/finalize bracket is exactly that lifetime,
  so `AceEmulator` enters the context in its constructor via an `ExitStack`
  and closes it in `finalize()`. One consequence: ACE does not support nesting
  the context, so two emulator components in one process cannot both drive
  ACE today.
- **Device ownership is not inferred.** `local_rank % device_count` looks
  reasonable and quietly puts two components' rank 0 on device 0 — MCT hands
  us a communicator and a decomposition, not a GPU ownership map, and our
  per-component local rank says nothing about what the ocean and land ranks on
  this node already hold. The supported contract is one visible device per
  rank (`--gpus-per-task=1 --gpu-bind=closest`, or an equivalent
  `CUDA_VISIBLE_DEVICES`); anything else has to be stated with
  `inference.device_id`. An ambiguous binding raises rather than guessing.
- **`LOCAL_RANK` is a device ordinal, not a rank.** Every consumer of it in
  ACE and PhysicsNeMo feeds it straight to `torch.cuda.set_device`
  (`torch_distributed.py:49`, `model_torch_distributed.py:144`,
  `pnd_manager.py:481`), so it indexes the devices *this process can see*.
  Publishing the component-local rank there breaks under exactly the
  one-GPU-per-rank binding recommended above: `device_count() == 1`, and the
  fourth rank on a node would ask for device 3. `Context.device_ordinal()`
  resolves it and `export()` publishes that; the adapter also calls
  `torch.cuda.set_device` itself, before entering the context, so the current
  device is right by the time a checkpoint is loaded.
- **Process groups the adapter creates are the adapter's to destroy.** In
  multi-rank `single` mode ACE runs `NonDistributed`, whose `shutdown()` is
  literally `return` (`non_distributed.py:136`), so nothing upstream releases
  the gloo groups the exchange needs — and exiting `Distributed.context()`
  does not either. Every group created here is registered with the same
  `ExitStack` that holds the ACE context, so it is released in reverse order
  by `finalize()` *and* by the constructor's rollback. A group ACE made stays
  ACE's. (ACE also skips its own shutdown when the context exits by exception,
  `distributed.py:85-87` — deliberate for a training script that is about to
  exit, less so for an embedded component. Not fixable from here.)
- **Entering the context can itself poison the process.** `context()` sets
  `_entered = True` and *then* calls `get_instance()` outside its own
  try/finally (`distributed.py:78-79`), so a failure while building the mesh,
  the process group or the PhysicsNeMo manager leaves the flag set for the
  life of the process, and every later attempt dies as "Nested
  Distributed.context() is not supported" — masking the original error.
  Since `enter_context()` never returned, there is nothing registered to undo
  it, so the adapter puts the flag back by hand. That repairs the guard but
  does **not** undo whatever partial process-group or singleton state was
  built before `get_instance()` gave up — it is a workaround for the pinned
  revision, not proof that retrying in the same process is safe. The real fix
  is upstream: move `instance = cls.get_instance()` inside the `try`, with
  exception-safe cleanup around it.
- **A root-only failure has to become everybody's failure.** In `single` mode
  only one rank loads the checkpoint, so a bad path or a corrupt file raises
  *there* while every other rank finishes construction happily — and the
  component then holds inconsistent state, so the next collective or the
  teardown hangs instead of reporting the real error. `Comm.agree()` closes
  that: the failure is caught, exchanged over the whole communicator, and
  re-raised everywhere, with the failing rank's message attached so a healthy
  rank does not send you to the wrong log. The rank that actually failed
  re-raises its own exception, type and traceback intact.

  The same treatment covers the two rank-local checks in the decomposition —
  `cell_indices()` (this rank's global ids) and the tile validation — because
  both can fail on one rank alone, and one of them runs *before* the
  exchange's collectives. `agree()` is itself a collective, so it has to be
  reached exactly once per rank on every path; that constraint is why
  validation collects complaints instead of throwing them.
- **A failed constructor must not leak the context.** `Distributed.context()`
  is entered in `AceEmulator.__init__` and closed in `finalize()`, so anything
  that throws in between — a missing checkpoint, a grid mismatch, an API that
  moved — would leave `Distributed._entered` true and the process group alive,
  and the next attempt in that process would fail as a nested context or hang.
  Integration is exactly when those failures are routine, so the initialization
  after the context is entered closes the stack and re-raises.
- **A global model can already run without any of this**, by giving the
  component few ranks (`NTASKS_ATM=1`). The coupler regrids and redistributes
  between components, so it performs the gather — correctly, and with no idle
  ranks, because the component never had them. `single` mode is the version of
  that which does not force the whole component down to one rank.
- **Have the component declare ACE's own grid to the coupler**, so the coupler
  does the interpolation. That is its job, already conservative and validated,
  and it leaves the exchange here a pure index permutation.
- **ACE is autoregressive** and holds prognostic state between calls. The
  Python object persisting across steps handles that naturally.
  `AceEmulator.state_for_restart()` returns that state on *columns* rather
  than on tiles, which is what makes it writable through the component's
  existing restart path and reloadable under a different rank count — but
  nothing calls it yet, because the component has no restart plumbing to hand
  it to. Until then, a run that stops and restarts restarts a different
  atmosphere.

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

Roughly in the order they are worth doing. The first three are what stand
between this and a run you would trust.

- **Restart.** `AceEmulator.state_for_restart()` exists and returns the
  prognostic state on columns; nothing writes or reads it. Until it does, a
  restarted run continues a different atmosphere.
- **An end-to-end test through the component.** The pieces are covered
  separately — the bridge against a numpy fixture, the decomposition across
  fake ranks, the field packing by inspection — but nothing yet drives
  `EmulatorAtm` through import → infer → export with a real coupling buffer.
- **`MPI_Alltoallv` for the redistribution.** Duplicate the component
  communicator in C++, expose the Fortran handle, rebuild it with
  `mpi4py.MPI.Comm.f2py`. Replaces `P - 1` point-to-point operations per rank
  with the machine's tuned collective, and removes the extra Gloo group.
- **Verify a distributed ACE numerically** before anybody trusts `spatial`:
  identical one-step results at 1, 2, 4 and 8 ranks against `ace_mode=single`,
  then multistep trajectories, then restarts — with communication, H2D/D2H and
  model time measured separately.
- **Hybrid data-and-spatial meshes.** ACE's `ModelTorchDistributed` builds a
  `(data, h, w)` mesh; this requires `h * w == world_size`, so `P_data = 1`.
  Right for one deterministic trajectory, and the thing to relax for a coupled
  ensemble.
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
