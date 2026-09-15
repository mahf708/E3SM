# Atmosphere emulator design specification

## Purpose and first delivery

Make a trained atmosphere usable as an E3SM component and as a separate
application, with the same stepping code and explicit scientific assumptions.
The component-level replacement is the first deliverable. Process-level
replacement, coupled ocean/ice emulators and distributed neural operators
follow after this boundary is stable.

The implementation baseline is the atmosphere subset of
`mahf708/emulators/coupled-emulators-master` at `c6a1234d`. The clean branch is
based on fork master `80e57e9c`. The code preserves working pieces rather than
introducing a new general plugin framework. The requirements below distinguish
current behavior from proposed changes to the coupler interface.

## Responsibilities

| Unit | Owns | Receives from its caller |
| --- | --- | --- |
| Application driver | Calendar, run length, MPI lifetime, diagnostics policy | Component configuration and external boundary data |
| Coupler adapter | Lifecycle translation, field binding, transport and remapping | Component communicator, driver time, domain and field buffers |
| Atmosphere component | Initialization, import/export field sets, checkpoint lifecycle | Configuration, boundary fields and time |
| Emulated model | Network cadence, brackets, prognostic feedback, operator execution | Model specification, geometry and inference backend |
| Atmosphere operators | Surface-input construction and export diagnostics | Named fields with scientific meaning |
| Inference backend | Model loading, framework tensors, device transfers, model call | Ordered numerical tensors and the saved model-step index |

The MCT adapter and standalone driver assemble the same component. No inference
backend may inspect E3SM field names to choose physical formulas. No coupler
adapter may depend on Torch classes. Runtime libraries stay behind the tensor
interface. YAML describes a model's channels and scientific operators. It does
not become a general programming language.

## Contracts the coupler API must represent

This is a proposed API contract, not a claim that the current C ABI carries all
of this metadata. The current implementation validates names and buffer counts
and stores units in field specifications. It does not negotiate units or temporal
meaning with MCT.

1. **Field identity:** name, quantity, units, sign convention, grid identity,
   staggering, dtype, dimensions, validity/mask and required/optional status.
2. **Time meaning:** a snapshot at a timestamp, a mean over a specified interval,
   or an integral over that interval. Interval endpoints and calendar are explicit.
3. **Memory ownership:** the owner keeps buffers alive until the call finishes.
   Inputs are borrowed read-only views. The receiver writes only declared outputs.
   The API declares host/device memory and completion semantics before permitting
   asynchronous access.
4. **Execution resources:** a component communicator, decomposition/global IDs and
   a device assignment. A backend never infers its process group from job-wide
   environment variables.
5. **Lifecycle:** configure, bind, initialize or restore, advance to a specified
   time, checkpoint and finalize. Initialization publishes valid initial exports.
6. **Failure:** a status and diagnostic reach all affected ranks before the next
   collective. C++ exceptions never cross the Fortran boundary.

A future API can express this without inheriting the implementation's class
hierarchy. A narrow port plus host-specific adapters is sufficient. The data
contract matters more than whether the host implements it in C, Fortran or C++.

## Time integration

For ACE2, the neural network advances 21600 seconds and the example coupler
advances 1800 seconds. At initialization the model predicts the first upper
bracket. Snapshot exports interpolate between brackets. Interval-mean exports
retain their declared mean semantics instead of being treated as snapshots.
At the interval boundary, the model samples coupling inputs and predicts the
next bracket. This is an explicit coupling approximation. It cannot use future
surface feedback that has not yet been computed.

Repeated calls at the same timestamp do not advance the network twice. The
current clock requires an integer ratio of model and coupler steps and assumes
successive distinct calls arrive at the specified cadence. A future host API
should reject skipped/backward timestamps and negotiate time intervals explicitly.
Support for arbitrary calendars and noninteger cadence ratios is outside the
first delivery.

The atmosphere export operators contain scientific approximations, including
near-surface diagnostics and shortwave partitioning. Engineers preserve their
interfaces and tests. The scientific owner decides whether those approximations
and the selected checkpoint define a defensible coupled experiment.

## Restart semantics

Restart state includes the model-step index, interval position, last call time,
both output brackets and operator auxiliary/accumulated state. Prognostic
network inputs are reconstructed from the raw upper prediction. The initial
condition supplies static/boundary fields on restart, never a replacement for
prognostic state. A continuation without a restart file must fail.

A seeded stochastic backend derives each call's seed from the saved model-step
index, not from the number of calls since process startup. The Python adapter
uses the same seed mixing as LibTorch. Exactness is a property to test for a
particular export/runtime/device combination. Different GPU kernels or runtime
versions can change floating-point results. Arbitrary stateful Python objects
and hidden recurrent state are not checkpointed by this interface.

The current restart format supports redistribution across component rank counts.
Promotion criteria include uninterrupted versus interrupted trajectories at a
mid-interval stop. Future restart metadata must also identify model/spec hashes,
grid ordering and runtime configuration so incompatible restarts fail early.

## Backend and portability policy

| Choice | Benefit | Cost and constraint |
| --- | --- | --- |
| Embedded Python | Fast model integration and a natural path to Python-native model code | Interpreter/library compatibility, GIL, package environment, explicit copies to framework tensors |
| LibTorch | Direct C++ execution of the exported module | C++ ABI/toolchain agreement, runtime packaging, exported-graph restrictions |
| Host float64 tensors | Small common contract compatible with current coupler buffers | Packing, dtype conversion and device transfer are visible costs |
| Root-rank global inference | Correct whole-globe input order with a simple reference implementation | Global memory and gather/scatter bottleneck. More MPI ranks do not distribute the network |

The first supported reference path is CPU. GPU support requires a compatible
export and runtime. A CUDA trace can retain device constants, so changing a
configuration string alone does not establish CPU portability. Python does not
by itself guarantee AMD/Intel accelerator portability either.

The source uses a narrow inference strategy interface, RAII ownership, explicit
factories and a registry for model operators. The factory is explicit because
static-library dead stripping can make registration through constructors fragile.
No additional inheritance layer is needed merely to name a design pattern.

## MPI and future decomposition

The application owns MPI initialization/finalization. Standalone duplicates its
communicator. E3SM supplies its component communicator. Every rank participates
in gather/scatter, including ranks with zero local cells. The global backend
runs only on component rank zero. A Python object must not call collectives on
the full component communicator when only rank zero enters it.

A decomposed inference backend needs a different execution capability: all
participating ranks enter inference, local tensors have explicit partition
metadata, and the model owns its halo exchanges or global transforms. Those
operations must derive from the supplied component resources. The present
root-inference implementation is a reference path, not evidence that ACE can
already run with a decomposed neural network.

## Acceptance and ownership

| Delivery | Responsible role | Acceptance evidence |
| --- | --- | --- |
| Component contract and MCT adapter | Coupler/software owner | Same component runs with both hosts, field/sign/time contracts agree |
| Backend packaging | Runtime/portability owner | Python and LibTorch CPU fixtures, errors and resource lifetime tests |
| Checkpoint and scientific operators | Scientific/model owner | Channel contract, preprocessing, export reference and physical budgets |
| Standalone/restart regression | Test owner | Whole-field comparisons, repeated-call tests, rank-count changes and mid-interval restarts |
| Hybrid case | Joint science/software owners | Supported-machine build, smoke and exact-restart results with approved domains/maps |

Assign one named owner per delivery and review the resulting evidence together.
The software team can choose internal class organization. The scientific owner
approves coupling semantics, field definitions and model validity. Changes to
those semantics require a recorded design decision and a regression reference.
Code comments explain invariants or non-obvious constraints. Usage and design
rationale belong in documentation. Small PRs should preserve an executable path.

The first review should settle field/time metadata, synchronous buffer lifetime,
communicator ownership and error propagation. Then establish the two-host
regression. Defer device-resident asynchronous exchange and distributed inference
until their performance evidence justifies expanding the contract.
