"""Running an ACE checkpoint as an E3SM component.

ACE is the hard case, and worth being precise about why.  Its samples are
*globes*: ``Stepper.step`` takes ``{name: tensor[n_batch, n_lat, n_lon]}``, so
one sample is an entire atmosphere.  E3SM's local columns are therefore not
ACE's batch dimension — splitting the grid across ranks splits one sample
spatially, and a model with a global receptive field (spherical transforms,
global attention) gives *wrong answers rather than errors* if you pretend
otherwise.  Everything below exists to make that impossible to do by accident.

Two ways to spread one globe over the ranks the coupler gave us:

======================  =========================  =========================
``inference.ace_mode``  ACE backend                Status
======================  =========================  =========================
``single``              ``NonDistributed``         **Supported.**  One rank
                                                   assembles the globe and
                                                   runs an unmodified
                                                   checkpoint; the others take
                                                   part only in the exchange.
                                                   The numerical reference.
``spatial``             ``ModelTorchDistributed``  **Gated.**  Ranks form an
                                                   ``h x w`` mesh, each owning
                                                   a rectangle.  Refused
                                                   unless
                                                   :data:`SPATIAL_OVERRIDE`
                                                   is set, because the
                                                   builders a deterministic
                                                   ACE2 checkpoint
                                                   instantiates still call
                                                   non-distributed spherical
                                                   transforms — see
                                                   :data:`_SPATIAL_WARNING`.
======================  =========================  =========================

``TorchDistributed`` is deliberately *not* offered.  It replicates the model
and splits a *batch* across ranks, so it can only help when the component
supplies several independent globes.  A single coupled E3SM trajectory has one
globe: handing every rank the same globe and the same deterministic weights
would compute the same answer N times, and reducing those to a mean and
storing it as the autoregressive state would collapse any ensemble that did
exist after a single step.  It becomes worth wiring when the coupling contract
grows an explicit ensemble — separate initial states, random states,
prescribed forcings, autoregressive states and outputs per member — and
:class:`~e3sm_emulator.decomposition.ReplicaExchange` is the piece that will
serve it.

Two things this module gets right that a naive integration does not:

*Rank discovery.*  ACE's ``TorchDistributed`` and PhysicsNeMo's
``DistributedManager`` both fall back to ``SLURM_PROCID`` / ``SLURM_NTASKS``,
which in a coupled run describe the entire job.  A process group built from
those blocks forever waiting for ocean and land ranks.  :meth:`Context.export`
publishes the *component* communicator's rank, size and rendezvous into the
variables they read first, so an unmodified upstream ACE initializes over
exactly our ranks.

*Order of operations.*  ``fme/core/step/single_module.py`` moves modules onto
the device and wraps them for distribution while the checkpoint is being
loaded, so rank, device and process group all have to exist first.  That is
the order below, and it is not negotiable.

Settings (``inference.*`` in the component namelist)::

    emulator:           ace
    model_path:         /path/to/ace_ckpt.tar
    ace_mode:           auto        # auto | single | spatial
    ace_h:              4           # spatial only: latitude ranks
    ace_w:              8           # spatial only: longitude ranks
    ace_unsafe_allow_unverified_spatial: false   # read the above first
    input:              air_temperature_0    # ACE's own variable names
    output:             air_temperature_0
    lon_fastest:        true        # how the coupler numbers its columns
    device_id:          0           # only if the launcher did not bind one

An input named ``<name>_next`` is routed to the stepper's
``next_step_input_data`` as ``<name>``; that is how prescribed SSTs and
forcings valid at the *end* of the step get in.
"""

from __future__ import annotations

import contextlib
import os

import numpy as np

from .comm import SerialComm, TorchComm, run_where
from .context import Context
from .decomposition import PermutationExchange, Tiling, split_bounds

#: The revision of https://github.com/mahf708/ace these calls were read from
#: and are known to match.  ACE is a moving target and its distributed and
#: stepper APIs have both changed shape recently, so the adapter states what
#: it was written against and checks for it rather than failing somewhere
#: deep inside a load.  Bump this together with the code, not before it.
PINNED_ACE_COMMIT = "75d8de6bcb0a30192720a16fc99f4eca0f54dbd2"

#: The ``fme.__version__`` that commit reports.  Coarse — a release string,
#: not a SHA — so it catches a wholesale version change and nothing finer.
#: **This is a compatibility declaration, not dependency pinning.**  The real
#: pin belongs in whatever builds the environment: a lockfile, a CIME machine
#: definition, or an install script that names the SHA.
PINNED_ACE_VERSION = "2026.4.0"

#: ace_mode -> the value ACE's Distributed selector reads.
_FME_BACKEND = {"single": "none", "spatial": "model"}

#: Suffix marking an input that belongs to the *next* step.
_NEXT_SUFFIX = "_next"

#: The flag that lets ``spatial`` run anyway.  Named so that nobody sets it
#: by accident or leaves it in a production namelist without noticing: what it
#: turns off is a check against a *known* wrong-answer condition, not a
#: conservative default.
SPATIAL_OVERRIDE = "ace_unsafe_allow_unverified_spatial"

_SPATIAL_WARNING = (
    "ace_mode=spatial builds ACE's (h, w) process mesh and hands each rank a "
    "rectangle of the globe, but the two builders a deterministic ACE2 "
    "checkpoint instantiates -- SphericalFourierNeuralOperatorNet "
    "(registry/sfno.py) and SFNO-v0.1.0 -- construct torch_harmonics RealSHT "
    "and InverseRealSHT unconditionally rather than through "
    "Distributed.get_sht(). A sharded input therefore reaches a module that "
    "still performs global transforms, which produces plausible numbers "
    "rather than an error. Only NoiseConditionedSFNO is wired through the "
    "distributed constructors.\n\n"
    "Before setting " + SPATIAL_OVERRIDE + ": route every global operator in "
    "the builder you are using through the distributed constructors, then "
    "check one-step and multistep output against ace_mode=single on the same "
    "checkpoint at 1, 2, 4 and 8 ranks. Setting the flag does not make the "
    "mode correct; it only records that you have taken responsibility for "
    "checking that it is."
)


def build(config: dict, context: Context) -> "AceEmulator":
    return AceEmulator(config, context)


def _flag(config: dict, key: str, default: bool = False) -> bool:
    value = config.get(key)
    if value is None or value == "":
        return default
    return str(value).lower() in ("true", "1", "yes", "on", ".true.")


def resolve_mode(config: dict, context: Context) -> tuple[str, int, int]:
    """Pick a mode and an ``h x w`` mesh, or explain why we cannot.

    Kept as a free function so the decision is testable without torch, ACE or
    a checkpoint — it is the part most likely to be got wrong.
    """
    mode = str(config.get("ace_mode", "auto")).lower()
    h = int(config.get("ace_h", 0) or 0)
    w = int(config.get("ace_w", 0) or 0)
    world = context.world_size

    if mode == "auto":
        # Never guess a mesh.  Which factorization of the ranks is right
        # depends on the model's transforms and on the machine, and picking
        # one silently is how a run ends up slow or wrong.
        mode = "spatial" if (h * w == world and h * w > 1) else "single"

    if mode not in _FME_BACKEND:
        raise ValueError(
            f"Unknown ace_mode '{mode}'. Use one of: "
            f"{', '.join(sorted(_FME_BACKEND))}. TorchDistributed "
            "(data-parallel over a batch of globes) is not offered: a coupled "
            "run supplies one globe, so there is nothing to distribute."
        )

    if mode == "spatial":
        if h <= 0 or w <= 0:
            raise ValueError(
                "ace_mode=spatial needs `inference.ace_h` and "
                "`inference.ace_w`, the latitude and longitude extents of the "
                "process mesh."
            )
        if h * w != world:
            raise ValueError(
                f"ace_mode=spatial needs ace_h * ace_w to be the component's "
                f"rank count, got {h} * {w} = {h * w} for {world} ranks. "
                "Hybrid data-and-spatial meshes (P_data * h * w) are not "
                "supported yet; every rank must own a rectangle."
            )
        if not _flag(config, SPATIAL_OVERRIDE):
            raise ValueError(_SPATIAL_WARNING)
    else:
        h = w = 1

    return mode, h, w


class AceEmulator:
    """One ACE checkpoint, stepped once per E3SM timestep."""

    def __init__(self, config: dict, context: Context):
        self.context = context
        self.verbose = bool(config.get("verbose", False))
        self.mode, self.h, self.w = resolve_mode(config, context)
        self.lon_fastest = _flag(config, "lon_fastest", default=True)
        self._exit_stack = contextlib.ExitStack()

        ny, nx = context.ny, context.nx
        if ny <= 0 or nx <= 0:
            raise ValueError(
                "The ACE emulator needs a structured latitude-longitude grid; "
                f"the coupler reported nx={nx}, ny={ny}. Declare the model's "
                "own grid in the component namelist and let the coupler "
                "regrid onto it."
            )

        self.input_names = list(config.get("inputs") or [])
        self.output_names = list(config.get("outputs") or [])
        if not self.output_names:
            raise ValueError(
                "The ACE emulator needs `inference.output` entries naming the "
                "variables to hand back to the coupler."
            )

        # 1. Rank, device and process group, before anything reads a weight.
        #    The device ordinal has to be settled first: LOCAL_RANK *is* the
        #    device ordinal as far as ACE and PhysicsNeMo are concerned, and
        #    they call torch.cuda.set_device with it while loading.
        device_id = config.get("device_id")
        self.device_ordinal = context.device_ordinal(
            None if device_id in (None, "") else int(device_id)
        )
        _set_cuda_device(self.device_ordinal)
        context.export(device_ordinal=self.device_ordinal)
        os.environ["FME_DISTRIBUTED_BACKEND"] = _FME_BACKEND[self.mode]
        if self.mode == "spatial":
            os.environ["FME_DISTRIBUTED_H"] = str(self.h)
            os.environ["FME_DISTRIBUTED_W"] = str(self.w)

        _check_ace_revision(enabled=context.is_root)

        from fme.core.distributed import Distributed

        # Distributed.get_instance() refuses to hand out a multi-rank instance
        # outside Distributed.context(), and the context also owns the
        # shutdown.  A component's init/finalize bracket is exactly that
        # lifetime, so the context is entered here and closed in finalize()
        # rather than wrapped around a single call.
        #
        # Entering it can itself fail, and at the pinned revision that leaves
        # the process unusable: context() sets `_entered = True` and *then*
        # calls get_instance() outside its own try/finally
        # (distributed.py:78-79), so a failure while building the mesh, the
        # process group or the PhysicsNeMo manager leaves the flag set
        # forever, and every later attempt dies as "Nested
        # Distributed.context() is not supported" — masking the real error.
        # Since enter_context() never returned, the ExitStack has nothing
        # registered to undo, so put the flag back by hand.  The proper fix is
        # upstream: move `instance = cls.get_instance()` inside the try.
        try:
            self._exit_stack.enter_context(Distributed.context())
        except BaseException:
            Distributed._entered = False
            raise

        # Everything from here on can fail — a missing checkpoint, a grid
        # mismatch, an API that moved — and if it does, finalize() is never
        # called because the object never comes into existence.  The context
        # would then stay entered and the process group alive, so the next
        # attempt in the same process fails as a nested context or hangs.
        # Integration is exactly when those failures are routine.
        try:
            self.dist = Distributed.get_instance()

            # `single` runs ACE with NonDistributed, so nothing upstream
            # builds a process group — but the exchange still needs one.  In
            # spatial mode ACE has already built its own, and initializing
            # again would fail.
            self.comm = self._make_comm()

            # 2. The decomposition.  Built before the checkpoint so a grid
            #    mismatch is reported in milliseconds rather than after a
            #    multi-gigabyte load.
            tiling = self._make_tiling(ny, nx)
            self.exchange = PermutationExchange(
                self.comm, context.col_gids, tiling, self.lon_fastest
            )
            self.tile_shape = self.exchange.tile_shape

            # 3. The checkpoint — only where it will actually be evaluated.
            #    In `single` mode that is one rank, which is the point: 64
            #    atmosphere ranks should not hold 64 copies of the weights.
            #
            #    That asymmetry is precisely why the load cannot simply throw.
            #    Only rank 0 opens the checkpoint, so a bad path or a corrupt
            #    file raises *there* while every other rank finishes
            #    construction happily, and the component then holds
            #    inconsistent state across ranks: the next collective — or
            #    teardown — hangs instead of reporting the real error. So the
            #    failure is caught, agreed over the whole communicator, and
            #    re-raised everywhere with rank 0's message attached.
            self.owns_model = self.tile_shape[0] > 0 and self.tile_shape[1] > 0
            self.stepper = None
            problem, error = "", None
            try:
                if self.owns_model:
                    model_path = config.get("model_path") or ""
                    if not model_path:
                        raise ValueError(
                            "The ACE emulator needs `inference.model_path`, "
                            "pointing at an ACE checkpoint."
                        )
                    self.stepper = _load_stepper(model_path)
            except Exception as exc:  # noqa: BLE001 - re-raised by agree()
                problem, error = (
                    f"rank {self.comm.rank} could not load the ACE "
                    f"checkpoint: {type(exc).__name__}: {exc}",
                    exc,
                )
            self.comm.agree(problem, error)
        except BaseException:
            self._exit_stack.close()
            raise

        #: Prognostic fields carried between timesteps.  ACE is
        #: autoregressive, so this is real model state; see
        #: :meth:`state_for_restart`.
        self.state: dict = {}

        if self.verbose and context.is_root:
            print(
                f"[e3sm_emulator.ace] mode={self.mode} "
                f"mesh={self.h}x{self.w} tile={self.tile_shape} "
                f"device={self.device_ordinal} "
                f"backend={_FME_BACKEND[self.mode]}",
                flush=True,
            )

    # -- setup helpers ------------------------------------------------------

    def _make_comm(self):
        """Build the exchange's communicator, owning whatever we create.

        Ownership matters because in multi-rank `single` mode ACE runs
        `NonDistributed`, whose `shutdown()` is literally `return` — so
        nothing upstream destroys a group we made, and the exit from
        `Distributed.context()` does not either.  Every group created here is
        registered with the same ExitStack that holds the ACE context, so it
        is released both by :meth:`finalize` and by the constructor's rollback,
        in reverse order of creation.
        """
        if self.context.world_size == 1:
            return SerialComm()

        import torch.distributed as dist

        if not dist.is_initialized():
            # Only reachable in `single` mode: the other modes have already
            # had ACE build the default group, and it belongs to ACE.
            dist.init_process_group(backend="gloo", init_method="env://")
            self._exit_stack.callback(dist.destroy_process_group)

        comm = TorchComm()
        self._exit_stack.callback(comm.close)
        return comm

    def _make_tiling(self, ny: int, nx: int) -> Tiling:
        """The partition of the globe, taken from ACE where ACE owns it.

        In spatial mode the tiling is not ours to choose: ACE slices its
        tensors with ``torch_harmonics.compute_split_shapes`` and our columns
        have to land in exactly those rows.  So we ask each rank what slice
        ACE gave it, gather the answers, and build the routing plan from
        those — one partitioning algorithm rather than two that can drift.
        """
        if self.mode != "spatial":
            return Tiling(ny, nx, 1, 1)

        # Asking ACE for this rank's slice is rank-local and precedes the
        # allgather below, so a failure here would strand the other ranks in
        # it.
        def local_slice():
            j_slice, i_slice = self.dist.get_local_slices((1, ny, nx))[-2:]
            return np.array(
                [
                    j_slice.start or 0,
                    j_slice.stop or ny,
                    i_slice.start or 0,
                    i_slice.stop or nx,
                ],
                dtype=np.int64,
            ).reshape(1, 4)

        mine = run_where(self.comm, True, local_slice)
        everyone = self.comm.allgather(mine)

        # Rank r owns tile (r // w, r % w), so the h boundaries are the row
        # starts of the first column of ranks and vice versa.
        j_bounds = [int(everyone[r * self.w][0]) for r in range(self.h)] + [ny]
        i_bounds = [int(everyone[r][2]) for r in range(self.w)] + [nx]

        tiling = Tiling.from_bounds(ny, nx, self.h, self.w, j_bounds, i_bounds)
        if not tiling.agrees_with_even_split():
            # Not fatal — ACE's partition is authoritative — but it means the
            # reference implementation in split_bounds has drifted from
            # torch_harmonics, and every test that pins it is now testing the
            # wrong thing.
            print(
                "[e3sm_emulator.ace] warning: ACE's spatial partition "
                f"({split_bounds(ny, self.h).tolist()} expected, "
                f"{j_bounds} reported) differs from split_bounds(). Using "
                "ACE's. Update split_bounds to match torch_harmonics.",
                flush=True,
            )
        _check_mesh_ordering(everyone, tiling, ny, nx, self.h, self.w)
        return tiling

    # -- per-step -----------------------------------------------------------

    def infer(self, inputs: dict, outputs: dict) -> None:
        import torch

        from fme.core.device import get_device
        from fme.core.step.args import StepArgs

        # Columns -> this rank's share of the globe.  One all-to-all for all
        # the fields at once: message count is what hurts at scale, not bytes.
        names = self.input_names or sorted(inputs)
        gathered = self._to_tiles(inputs, names)

        # Everything the owning ranks do alone goes inside run_where: the
        # tensor conversions, the stepper call, the missing-output check and
        # the trip back to host memory.  Any of them can fail on the owner
        # while every other rank walks on into _to_columns() and blocks in the
        # redistribution — a hang at a point with nothing to do with the
        # cause.  Guarding only stepper.step() would leave that hole open for
        # a malformed input or an absent output; the whole block belongs here.
        def run_model():
            device = get_device()
            now, nxt = {}, {}
            for name, tile in gathered.items():
                # [n_batch=1, n_lat, n_lon]; this rank contributes one globe,
                # whether that is the whole thing or a rectangle of it.
                tensor = torch.from_numpy(tile).to(device=device).float().unsqueeze(0)
                if name.endswith(_NEXT_SUFFIX):
                    nxt[name[: -len(_NEXT_SUFFIX)]] = tensor
                else:
                    now[name] = tensor

            # Prognostic state from the previous step wins over whatever the
            # coupler sent: ACE integrates its own atmosphere forward.
            now.update(self.state)
            if not nxt:
                # With a single time level from the coupler there is nothing
                # better to prescribe at the end of the step.
                nxt = dict(now)

            # Call the Stepper, not modules[0]: packing, normalization,
            # residual prediction, correctors, prescribed SST and derived
            # forcings all live here, and bypassing it silently drops part of
            # the learned timestep.
            result = self.stepper.step(StepArgs(input=now, next_step_input_data=nxt))
            result = _as_field_mapping(result)

            self.state = {
                name: result[name]
                for name in self.stepper.prognostic_names
                if name in result
            }
            missing = [n for n in self.output_names if n not in result]
            if missing:
                raise ValueError(
                    f"The ACE stepper did not produce {missing}. It produces: "
                    f"{sorted(result)}."
                )
            return {
                name: result[name].detach().to("cpu", torch.float64).numpy()[0]
                for name in self.output_names
            }

        produced = run_where(self.comm, self.owns_model, run_model)
        if produced is None:
            # This rank owns no part of the globe. It still has to take part
            # in the exchange — infer() is collective.
            produced = {
                name: np.empty((0, 0), dtype=np.float64) for name in self.output_names
            }

        self._to_columns(produced, outputs)

    def _to_tiles(self, inputs: dict, names) -> dict:
        """Move every named input field onto this rank's tile, in one go.

        The packing is rank-local and sits immediately before a collective, so
        a field missing or misshapen on one rank alone would leave the others
        waiting in the all-to-all.  Settle it first.
        """
        columns = run_where(
            self.comm,
            True,
            lambda: np.stack(
                [
                    np.asarray(inputs[name], dtype=np.float64).reshape(-1)
                    for name in names
                ],
                axis=1,
            ),
        )
        tile = self.exchange.to_tile(columns)
        return {name: np.ascontiguousarray(tile[..., k]) for k, name in enumerate(names)}

    def _to_columns(self, produced: dict, outputs: dict) -> None:
        """Move the model's fields back onto the coupler's columns."""
        nj, ni = self.tile_shape
        stacked = np.stack(
            [
                produced[name].reshape(nj, ni)
                if produced[name].size
                else np.empty((nj, ni), dtype=np.float64)
                for name in self.output_names
            ],
            axis=-1,
        )
        columns = self.exchange.to_columns(stacked)
        for k, name in enumerate(self.output_names):
            outputs[name].reshape(-1)[:] = columns[:, k]

    # -- restart ------------------------------------------------------------

    def state_for_restart(self) -> dict:
        """This rank's prognostic state, as numpy arrays on its own columns.

        ACE is autoregressive, so a run that stops and restarts without this
        continues a *different* atmosphere.  Returning it on *columns* rather
        than on tiles is what makes it writable through the component's
        existing restart path, and reloadable under a different rank count.

        **Collective: every rank must call it, the same number of times.**
        The state lives only on the ranks that own the model, so the field
        names are agreed first — otherwise the ranks that hold nothing would
        return early and leave the owner alone in the redistribution, which
        is a deadlock rather than an empty result.

        Nothing calls this yet: the component has no restart plumbing to hand
        it to. It is written collectively so that wiring it up is safe when
        that arrives.
        """
        import torch

        # Owners announce what they hold; everyone learns the same name list.
        announced = self.comm.allgather_text(
            "\n".join(sorted(self.state)) if self.owns_model else ""
        )
        names = sorted({n for block in announced for n in block.split("\n") if n})
        if not names:
            return {}

        nj, ni = self.tile_shape

        def tiles():
            return np.stack(
                [
                    self.state[n].detach().to("cpu", torch.float64).numpy().reshape(
                        nj, ni
                    )
                    if self.owns_model and n in self.state
                    else np.empty((nj, ni), dtype=np.float64)
                    for n in names
                ],
                axis=-1,
            )

        stacked = run_where(self.comm, True, tiles)
        columns = self.exchange.to_columns(stacked)
        return {name: columns[:, k] for k, name in enumerate(names)}

    def finalize(self) -> None:
        # Dropping these is what frees the weights and the carried
        # atmosphere.  Closing the stack then releases, in reverse order of
        # creation, every process group this adapter made and finally exits
        # Distributed.context().
        #
        # It does *not* release a group ACE made: in spatial mode the default
        # group is ACE's, and its own shutdown runs on the way out of the
        # context.  Note that ACE skips that shutdown when the context exits
        # by exception (distributed.py:85-87, deliberately — a training script
        # is about to exit anyway), so a failed run leaves ACE's group up.
        # That one is not ours to fix from here.
        self.state = {}
        self.stepper = None
        self._exit_stack.close()


def _check_mesh_ordering(reported, tiling: Tiling, ny: int, nx: int, h: int, w: int):
    """Check every rank's rectangle, not just its size.

    Comparing tile *shapes* is nearly worthless as a check: on an evenly
    divisible mesh every tile has the same shape, so a permuted rank order
    passes while each rank works on somebody else's piece of the planet — and
    a global model fed a rotated globe returns a plausible field, not an
    error.  So compare the actual coordinates, for all ranks, and confirm the
    rectangles tile the sphere.

    Args:
        reported: ``(size, 4)`` of ``[j_start, j_stop, i_start, i_stop]``,
            gathered in rank order, as ACE reported them.
        tiling: what the assumed ``(r // w, r % w)`` ordering reconstructs.
    """
    covered = np.zeros((ny, nx), dtype=np.int32)
    for rank in range(h * w):
        j0, j1, i0, i1 = (int(v) for v in reported[rank])
        origin = tiling.tile_origin(rank)
        shape = tiling.tile_shape(rank)
        expected = (origin[0], origin[0] + shape[0], origin[1], origin[1] + shape[1])
        if (j0, j1, i0, i1) != expected:
            raise ValueError(
                f"Rank {rank}: ACE owns rows {j0}:{j1} and columns {i0}:{i1}, "
                f"but the mesh ordering assumed here puts it at "
                f"{expected[0]}:{expected[1]}, {expected[2]}:{expected[3]}. "
                "The (h, w) rank ordering assumed here does not match ACE's "
                "DeviceMesh, and routing columns on this assumption would give "
                "every rank the wrong part of the globe."
            )
        covered[j0:j1, i0:i1] += 1

    if not np.array_equal(covered, np.ones((ny, nx), dtype=np.int32)):
        gaps = int((covered == 0).sum())
        overlaps = int((covered > 1).sum())
        raise ValueError(
            f"The reported rectangles do not tile the {ny}x{nx} grid: "
            f"{gaps} cell(s) belong to nobody and {overlaps} to more than one."
        )


def _set_cuda_device(ordinal: int) -> None:
    """Claim this rank's device before ACE looks at it.

    ACE reads ``get_device()`` (which is ``torch.cuda.current_device()``)
    while loading a checkpoint, so the current device has to be right by then.
    Setting it here also means the value is correct even in `single` mode,
    where no ACE distributed backend runs to set it for us.
    """
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.set_device(int(ordinal))


def _check_ace_revision(enabled: bool) -> None:
    """Say something when the installed ACE is not the one this targets.

    Best effort, and not a substitute for pinning: ``fme`` reports a coarse
    release version rather than a commit, so this catches a wholesale version
    change and nothing finer.  The real pin belongs in whatever builds the
    environment — a lockfile, a CIME machine definition, or an install script
    that names the SHA. This is a compatibility declaration that complains
    when it is obviously violated.

    Runs on the component root whatever the verbosity: it costs an attribute
    lookup and says nothing unless something is actually wrong, and a run that
    is not in verbose mode is exactly the one where a silent version drift
    would go unnoticed.
    """
    if not enabled:
        return
    try:
        import fme

        version = getattr(fme, "__version__", None)
    except ImportError:
        return
    if version is not None and version != PINNED_ACE_VERSION:
        print(
            f"[e3sm_emulator.ace] warning: fme {version} is installed; this "
            f"adapter was written against {PINNED_ACE_VERSION} "
            f"(mahf708/ace {PINNED_ACE_COMMIT[:12]}). Pin the revision in the "
            "environment if anything here misbehaves.",
            flush=True,
        )


def _load_stepper(model_path: str):
    """Load a checkpoint, saying plainly when ACE has moved under us."""
    try:
        from fme.ace.stepper.single_module import load_stepper
    except ImportError as exc:
        raise ImportError(
            "Could not import load_stepper from fme.ace.stepper.single_module. "
            f"This adapter was written against mahf708/ace {PINNED_ACE_COMMIT}, "
            "where it lives there; ACE's stepper and distributed APIs both "
            "move. Pin that revision, or update this adapter to the "
            "checkpoint-loading API of the revision you are running."
        ) from exc
    return load_stepper(model_path)


def _as_field_mapping(result):
    """Read a step result as ``{name: tensor}``.

    At the pinned revision ``Stepper.step`` returns a ``TensorDict``, which is
    a mapping.  Later revisions have wrapped it; unwrap the obvious shapes and
    otherwise say exactly what came back, because silently indexing the wrong
    object is how a wrong field reaches the coupler.
    """
    from collections.abc import Mapping

    if isinstance(result, Mapping):
        return result
    for attribute in ("data", "prediction", "output"):
        candidate = getattr(result, attribute, None)
        if isinstance(candidate, Mapping):
            return candidate
    raise TypeError(
        f"Stepper.step returned {type(result).__name__}, which is not a "
        "mapping of field name to tensor and exposes no .data/.prediction/"
        f".output that is. This adapter targets mahf708/ace "
        f"{PINNED_ACE_COMMIT}."
    )
