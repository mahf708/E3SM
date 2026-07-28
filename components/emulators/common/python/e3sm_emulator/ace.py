"""Running an ACE checkpoint as an E3SM component.

ACE's samples are *globes*: ``Stepper.step`` takes
``{name: tensor[n_batch, n_lat, n_lon]}``, so one sample is an entire
atmosphere and E3SM's local columns are **not** ACE's batch dimension.
Splitting the grid across ranks splits one sample spatially, and a model with
a global receptive field answers that plausibly rather than rejecting it.
Everything here exists to make that impossible to do by accident.

Two ways to spread one globe over the ranks the coupler gave us:

``single``   ``NonDistributed``. One rank assembles the globe and runs an
             unmodified checkpoint; the others take part only in the exchange.
             Supported, and the numerical reference.
``spatial``  ``ModelTorchDistributed`` over an ``h x w`` mesh. Implemented and
             refused unless :data:`SPATIAL_OVERRIDE` is set -- see
             :data:`_SPATIAL_WARNING`.

``TorchDistributed`` is not offered: it splits a *batch* of globes, a coupled
trajectory supplies one, and averaging N identical answers into the
autoregressive state would collapse any ensemble that did exist.

See ``../../src/inference/README.md`` for the rest of the argument, including
why ``Context.export`` must publish the component's rank rather than the job's,
and why device, rank and process group all have to exist before the checkpoint
is loaded.

Settings (``inference.*`` in the component namelist)::

    emulator:           ace
    model_path:         /path/to/ace_ckpt.tar
    ace_mode:           auto        # auto | single | spatial
    ace_h:              4           # spatial only: latitude ranks
    ace_w:              8           # spatial only: longitude ranks
    ace_unsafe_allow_unverified_spatial: false   # read _SPATIAL_WARNING first
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

#: The revision of https://github.com/mahf708/ace these calls were read from.
#: ACE's stepper and distributed APIs both move, so the adapter states what it
#: was written against and checks the two API points it depends on rather than
#: failing deep inside a load.  Bump this with the code.
PINNED_ACE_COMMIT = "75d8de6bcb0a30192720a16fc99f4eca0f54dbd2"

#: The ``fme.__version__`` that commit reports.  Coarse -- a release string,
#: not a SHA -- and a compatibility declaration rather than dependency
#: pinning: the real pin belongs in whatever builds the environment.
PINNED_ACE_VERSION = "2026.4.0"

#: ace_mode -> the value ACE's Distributed selector reads.
_FME_BACKEND = {"single": "none", "spatial": "model"}

#: Suffix marking an input that belongs to the *next* step.
_NEXT_SUFFIX = "_next"

#: The flag that lets ``spatial`` run anyway.  Named so nobody leaves it in a
#: production namelist without noticing: what it turns off is a check against
#: a *known* wrong-answer condition, not a conservative default.
SPATIAL_OVERRIDE = "ace_unsafe_allow_unverified_spatial"

_SPATIAL_WARNING = (
    "ace_mode=spatial hands each rank a rectangle of the globe, but the two "
    "builders a deterministic ACE2 checkpoint instantiates -- "
    "SphericalFourierNeuralOperatorNet (registry/sfno.py) and SFNO-v0.1.0 -- "
    "construct torch_harmonics RealSHT and InverseRealSHT unconditionally "
    "rather than through Distributed.get_sht(). A sharded input therefore "
    "reaches a module that still performs global transforms, which produces "
    "plausible numbers rather than an error. Only NoiseConditionedSFNO is "
    "wired through the distributed constructors.\n\n"
    "Before setting " + SPATIAL_OVERRIDE + ": route every global operator in "
    "your builder through the distributed constructors, then check one-step "
    "and multistep output against ace_mode=single on the same checkpoint at "
    "1, 2, 4 and 8 ranks. The flag does not make the mode correct; it records "
    "that you have taken responsibility for checking that it is."
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

    A free function so the decision is testable without torch, ACE or a
    checkpoint -- it is the part most likely to be got wrong.
    """
    mode = str(config.get("ace_mode", "auto")).lower()
    h = int(config.get("ace_h", 0) or 0)
    w = int(config.get("ace_w", 0) or 0)
    world = context.world_size

    if mode == "auto":
        # Never guess a mesh: which factorization is right depends on the
        # model's transforms and on the machine, and picking one silently is
        # how a run ends up slow or wrong.
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
                "`inference.ace_w`, the extents of the process mesh."
            )
        if h * w != world:
            raise ValueError(
                f"ace_mode=spatial needs ace_h * ace_w to be the component's "
                f"rank count, got {h} * {w} = {h * w} for {world} ranks. "
                "Hybrid data-and-spatial meshes are not supported yet; every "
                "rank must own a rectangle."
            )
        if not _flag(config, SPATIAL_OVERRIDE):
            raise ValueError(_SPATIAL_WARNING)
    else:
        h = w = 1

    return mode, h, w


def stack_output_tiles(produced: dict, names, tile_shape) -> np.ndarray:
    """Lay the model's named fields out as one ``(nj, ni, nfields)`` array.

    The last thing before the scatter, and the last thing that can fail on one
    rank alone -- which is why it is called from inside ``run_where``.  The
    shape check is the point: a stepper can return a correctly *named* field
    of the wrong size, and ``.numpy()[0]`` will not notice.
    """
    nj, ni = int(tile_shape[0]), int(tile_shape[1])
    tiles = []
    for name in names:
        if name not in produced:
            raise ValueError(
                f"No output named '{name}' to scatter; got {sorted(produced)}."
            )
        tile = np.asarray(produced[name], dtype=np.float64)
        if tile.shape != (nj, ni):
            raise ValueError(
                f"Output '{name}' has shape {tile.shape}, but this rank owns a "
                f"{nj}x{ni} tile. A model that returns the wrong size here "
                "would fail alone, in the scatter, with every other rank "
                "already waiting."
            )
        tiles.append(tile)
    if not tiles:
        return np.empty((nj, ni, 0), dtype=np.float64)
    return np.stack(tiles, axis=-1)


def empty_output_tiles(names, tile_shape) -> np.ndarray:
    """The contribution of a rank that owns no part of the globe."""
    return np.empty(
        (int(tile_shape[0]), int(tile_shape[1]), len(names)), dtype=np.float64
    )


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
                "own grid in the namelist and let the coupler regrid onto it."
            )

        self.input_names = list(config.get("inputs") or [])
        self.output_names = list(config.get("outputs") or [])
        if not self.output_names:
            raise ValueError(
                "The ACE emulator needs `inference.output` entries naming the "
                "variables to hand back to the coupler."
            )

        # 1. Rank, device and process group, before anything reads a weight.
        #    LOCAL_RANK *is* the device ordinal as far as ACE and PhysicsNeMo
        #    are concerned, and they call torch.cuda.set_device with it while
        #    loading, so it has to be settled first.
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

        # get_instance() refuses a multi-rank instance outside
        # Distributed.context(), and that context owns the shutdown, so a
        # component's init/finalize bracket is its lifetime.  Entering it can
        # fail *after* it sets `_entered = True` (get_instance() is called
        # outside its own try/finally), which makes every later attempt die as
        # "Nested Distributed.context() is not supported" and masks the real
        # error.  enter_context() never returned, so nothing is registered to
        # undo it; put the flag back by hand.  The proper fix is upstream.
        try:
            self._exit_stack.enter_context(Distributed.context())
        except BaseException:
            Distributed._entered = False
            raise

        # Anything that throws from here on means finalize() is never called,
        # because the object never comes into existence -- so the context would
        # stay entered and the process group alive, and the next attempt in
        # this process would hang or report a nested context.
        try:
            self.dist = Distributed.get_instance()

            # `single` runs ACE with NonDistributed, so nothing upstream builds
            # a group -- but the exchange still needs one.  In spatial mode ACE
            # has already built its own.
            self.comm = self._make_comm()

            # 2. The decomposition, before the checkpoint, so a grid mismatch
            #    costs milliseconds rather than a multi-gigabyte load.
            self.exchange = PermutationExchange(
                self.comm, context.col_gids, self._make_tiling(ny, nx),
                self.lon_fastest,
            )
            self.tile_shape = self.exchange.tile_shape

            # 3. The checkpoint, only where it will be evaluated: 64 atmosphere
            #    ranks should not hold 64 copies of the weights.  That
            #    asymmetry is why the load cannot simply throw -- a bad path
            #    would raise on one rank while the others finished construction
            #    happily, and the next collective would hang.
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
                    f"rank {self.comm.rank} could not load the ACE checkpoint: "
                    f"{type(exc).__name__}: {exc}",
                    exc,
                )
            self.comm.agree(problem, error)
        except BaseException:
            self._exit_stack.close()
            raise

        #: Prognostic fields carried between timesteps.  ACE is
        #: autoregressive, so this is real model state.
        self.state: dict = {}

        if self.verbose and context.is_root:
            print(
                f"[e3sm_emulator.ace] mode={self.mode} mesh={self.h}x{self.w} "
                f"tile={self.tile_shape} device={self.device_ordinal}",
                flush=True,
            )

    # -- setup helpers ------------------------------------------------------

    def _make_comm(self):
        """Build the exchange's communicator, owning whatever we create.

        In multi-rank `single` mode ACE runs NonDistributed, whose shutdown()
        is literally `return`, so nothing upstream destroys a group we made and
        leaving Distributed.context() does not either.  Everything created here
        is registered with the stack that holds that context, so it is released
        by finalize() and by the constructor's rollback alike.  A group ACE
        made stays ACE's.
        """
        if self.context.world_size == 1:
            return SerialComm()

        import torch.distributed as dist

        if not dist.is_initialized():
            # Only reachable in `single` mode: the other modes have already had
            # ACE build the default group, and it belongs to ACE.
            dist.init_process_group(backend="gloo", init_method="env://")
            self._exit_stack.callback(dist.destroy_process_group)

        comm = TorchComm()
        self._exit_stack.callback(comm.close)
        return comm

    def _make_tiling(self, ny: int, nx: int) -> Tiling:
        """The partition of the globe, taken from ACE where ACE owns it.

        In spatial mode the tiling is not ours to choose: ACE slices its
        tensors with ``torch_harmonics.compute_split_shapes`` and our columns
        have to land in exactly those rows.  Building the plan from what ACE
        reports means one partitioning algorithm rather than two that can
        drift.
        """
        if self.mode != "spatial":
            return Tiling(ny, nx, 1, 1)

        # Rank-local, and it precedes the allgather, so a failure here would
        # strand the other ranks in it.
        def local_slice():
            j, i = self.dist.get_local_slices((1, ny, nx))[-2:]
            return np.array(
                [j.start or 0, j.stop or ny, i.start or 0, i.stop or nx],
                dtype=np.int64,
            ).reshape(1, 4)

        reported = self.comm.allgather(run_where(self.comm, True, local_slice))

        # Rank r owns tile (r // w, r % w), so the h boundaries are the row
        # starts of the first column of ranks and vice versa.
        j_bounds = [int(reported[r * self.w][0]) for r in range(self.h)] + [ny]
        i_bounds = [int(reported[r][2]) for r in range(self.w)] + [nx]

        tiling = Tiling.from_bounds(ny, nx, self.h, self.w, j_bounds, i_bounds)
        if not tiling.agrees_with_even_split():
            # Not fatal -- ACE's partition is authoritative -- but it means
            # split_bounds has drifted from torch_harmonics, and every test
            # that pins it is now testing the wrong thing.
            print(
                "[e3sm_emulator.ace] warning: ACE's spatial partition "
                f"({split_bounds(ny, self.h).tolist()} expected, {j_bounds} "
                "reported) differs from split_bounds(). Using ACE's.",
                flush=True,
            )
        check_mesh_ordering(reported, tiling, self.h, self.w)
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

        # Everything the owning ranks do alone goes inside run_where -- the
        # conversions, the stepper call, the missing-output check, the trip
        # back to host memory.  Any of them can fail on the owner while every
        # other rank walks into the scatter and blocks there.  Guarding only
        # stepper.step() would leave that hole open.
        def run_model():
            device = get_device()
            now, nxt = {}, {}
            for name, tile in gathered.items():
                # [n_batch=1, n_lat, n_lon]: this rank contributes one globe,
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
                nxt = dict(now)  # one time level; nothing better to prescribe

            # The Stepper, not modules[0]: packing, normalization, residual
            # prediction, correctors, prescribed SST and derived forcings all
            # live here, and bypassing it drops part of the learned timestep.
            result = _as_field_mapping(
                self.stepper.step(StepArgs(input=now, next_step_input_data=nxt))
            )

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
            return stack_output_tiles(
                {
                    name: result[name].detach().to("cpu", torch.float64).numpy()[0]
                    for name in self.output_names
                },
                self.output_names,
                self.tile_shape,
            )

        stacked = run_where(self.comm, self.owns_model, run_model)
        if stacked is None:
            # This rank owns no part of the globe, but infer() is collective,
            # so it still takes part in the exchange.
            stacked = empty_output_tiles(self.output_names, self.tile_shape)

        columns = self.exchange.to_columns(stacked)
        for k, name in enumerate(self.output_names):
            outputs[name].reshape(-1)[:] = columns[:, k]

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
                [np.asarray(inputs[n], dtype=np.float64).reshape(-1) for n in names],
                axis=1,
            ),
        )
        tile = self.exchange.to_tile(columns)
        return {
            name: np.ascontiguousarray(tile[..., k]) for k, name in enumerate(names)
        }

    # -- restart ------------------------------------------------------------

    def state_for_restart(self) -> dict:
        """This rank's prognostic state, as numpy arrays on its own columns.

        ACE is autoregressive, so a run that stops and restarts without this
        continues a *different* atmosphere.  Returning it on *columns* rather
        than on tiles is what makes it writable through a component's existing
        restart path, and reloadable under a different rank count.

        **Collective: every rank must call it, the same number of times.** The
        state lives only on the ranks that own the model, so the field names
        are agreed first -- otherwise the ranks holding nothing would return
        early and leave the owner alone in the redistribution, which is a
        deadlock rather than an empty result.

        Nothing calls this yet; it is written collectively so that wiring it up
        is safe when a component grows restart plumbing.
        """
        import torch

        announced = {
            rank: frozenset(n for n in block.split("\n") if n)
            for rank, block in enumerate(
                self.comm.allgather_text(
                    "\n".join(sorted(self.state)) if self.owns_model else ""
                )
            )
        }
        owners = {r: names for r, names in announced.items() if names}
        if not owners:
            return {}
        # Owners must hold the *same* fields: a union across them would give
        # whichever tile lacks one uninitialized values, and the restart would
        # carry garbage over part of the globe.
        distinct = set(owners.values())
        if len(distinct) > 1:
            raise ValueError(
                "The ranks holding model state disagree about which prognostic "
                "fields they hold: "
                + "; ".join(f"rank {r}: {sorted(n)}" for r, n in owners.items())
                + ". A union across them would write uninitialized values into "
                "the restart for whichever tile lacks a field."
            )
        names = sorted(next(iter(distinct)))
        nj, ni = self.tile_shape

        def tiles():
            return np.stack(
                [
                    self.state[n].detach().to("cpu", torch.float64).numpy().reshape(
                        nj, ni
                    )
                    if self.owns_model
                    else np.empty((nj, ni), dtype=np.float64)
                    for n in names
                ],
                axis=-1,
            )

        columns = self.exchange.to_columns(run_where(self.comm, True, tiles))
        return {name: columns[:, k] for k, name in enumerate(names)}

    def finalize(self) -> None:
        # Dropping these frees the weights and the carried atmosphere; closing
        # the stack then releases every group this adapter made, in reverse
        # order, and exits Distributed.context().
        self.state = {}
        self.stepper = None
        self._exit_stack.close()


def check_mesh_ordering(reported, tiling: Tiling, h: int, w: int) -> None:
    """Check every rank's rectangle by coordinates, not by size.

    Comparing tile *shapes* is nearly worthless: on an evenly divisible mesh
    every tile has the same shape, so a permuted rank order would pass while
    each rank worked on somebody else's piece of the planet -- and a global
    model fed a rotated globe returns a plausible field, not an error.

    That the rectangles tile the sphere follows: ``tiling`` was built from
    boundaries :class:`~e3sm_emulator.decomposition.Tiling` already checked
    cover the grid exactly once, so matching it rank for rank is the whole
    property.

    Args:
        reported: ``(size, 4)`` of ``[j_start, j_stop, i_start, i_stop]``,
            gathered in rank order, as ACE reported them.
        tiling: what the assumed ``(r // w, r % w)`` ordering reconstructs.
    """
    for rank in range(h * w):
        j0, j1, i0, i1 = (int(v) for v in reported[rank])
        (ej0, ei0), (nj, ni) = tiling.tile_origin(rank), tiling.tile_shape(rank)
        if (j0, j1, i0, i1) != (ej0, ej0 + nj, ei0, ei0 + ni):
            raise ValueError(
                f"Rank {rank}: ACE owns rows {j0}:{j1} and columns {i0}:{i1}, "
                f"but the mesh ordering assumed here puts it at "
                f"{ej0}:{ej0 + nj}, {ei0}:{ei0 + ni}. Routing columns on this "
                "assumption would give every rank the wrong part of the globe."
            )


def _set_cuda_device(ordinal: int) -> None:
    """Claim this rank's device before ACE looks at it.

    ACE reads ``get_device()`` while loading a checkpoint, so the current
    device has to be right by then -- including in `single` mode, where no ACE
    distributed backend runs to set it for us.
    """
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.set_device(int(ordinal))


def _check_ace_revision(enabled: bool) -> None:
    """Say something when the installed ACE is not the one this targets.

    Best effort, not a substitute for pinning.  Runs whatever the verbosity:
    it costs an attribute lookup and says nothing unless something is wrong,
    and a quiet run is exactly the one where drift would go unnoticed.
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
            f"adapter was written against {PINNED_ACE_VERSION} (mahf708/ace "
            f"{PINNED_ACE_COMMIT[:12]}). Pin the revision if anything here "
            "misbehaves.",
            flush=True,
        )


def _load_stepper(model_path: str):
    """Load a checkpoint, saying plainly when ACE has moved under us."""
    try:
        from fme.ace.stepper.single_module import load_stepper
    except ImportError as exc:
        raise ImportError(
            "Could not import load_stepper from fme.ace.stepper.single_module, "
            f"where it lives at mahf708/ace {PINNED_ACE_COMMIT}. Pin that "
            "revision, or update this adapter to the checkpoint-loading API of "
            "the revision you are running."
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
        f"Stepper.step returned {type(result).__name__}, which is not a mapping "
        "of field name to tensor and exposes no .data/.prediction/.output that "
        f"is. This adapter targets mahf708/ace {PINNED_ACE_COMMIT}."
    )
