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
                                                   unless ``ace_allow_spatial``
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
    ace_allow_spatial:  false       # see above before setting this
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

from .comm import SerialComm, TorchComm
from .context import Context
from .decomposition import PermutationExchange, Tiling, split_bounds

#: The revision of https://github.com/mahf708/ace these calls were read from
#: and are known to match.  ACE is a moving target and its distributed and
#: stepper APIs have both changed shape recently, so the adapter states what
#: it was written against and checks for it rather than failing somewhere
#: deep inside a load.  Bump this together with the code, not before it.
PINNED_ACE_COMMIT = "75d8de6bcb0a30192720a16fc99f4eca0f54dbd2"

#: ace_mode -> the value ACE's Distributed selector reads.
_FME_BACKEND = {"single": "none", "spatial": "model"}

#: Suffix marking an input that belongs to the *next* step.
_NEXT_SUFFIX = "_next"

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
    "Before setting ace_allow_spatial: route every global operator in the "
    "builder you are using through the distributed constructors, then check "
    "one-step and multistep output against ace_mode=single on the same "
    "checkpoint at 1, 2, 4 and 8 ranks."
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
        if not _flag(config, "ace_allow_spatial"):
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
        context.export()
        os.environ["FME_DISTRIBUTED_BACKEND"] = _FME_BACKEND[self.mode]
        if self.mode == "spatial":
            os.environ["FME_DISTRIBUTED_H"] = str(self.h)
            os.environ["FME_DISTRIBUTED_W"] = str(self.w)

        from fme.core.distributed import Distributed

        # Distributed.get_instance() refuses to hand out a multi-rank instance
        # outside Distributed.context(), and the context also owns the
        # shutdown.  A component's init/finalize bracket is exactly that
        # lifetime, so the context is entered here and closed in finalize()
        # rather than wrapped around a single call.
        self._exit_stack.enter_context(Distributed.context())
        self.dist = Distributed.get_instance()

        # `single` runs ACE with NonDistributed, so nothing upstream builds a
        # process group — but the exchange still needs one.  In spatial mode
        # ACE has already built its own, and initializing again would fail.
        self.comm = self._make_comm()

        # 2. The decomposition.  Built before the checkpoint so a grid
        #    mismatch is reported in milliseconds rather than after a
        #    multi-gigabyte load.
        tiling = self._make_tiling(ny, nx)
        self.exchange = PermutationExchange(
            self.comm, context.col_gids, tiling, self.lon_fastest
        )
        self.tile_shape = self.exchange.tile_shape

        # 3. The checkpoint — only where it will actually be evaluated.  In
        #    `single` mode that is one rank, which is the point: 64 atmosphere
        #    ranks should not hold 64 copies of the weights.
        self.owns_model = self.tile_shape[0] > 0 and self.tile_shape[1] > 0
        self.stepper = None
        if self.owns_model:
            model_path = config.get("model_path") or ""
            if not model_path:
                raise ValueError(
                    "The ACE emulator needs `inference.model_path`, pointing "
                    "at an ACE checkpoint."
                )
            self.stepper = _load_stepper(model_path)

        #: Prognostic fields carried between timesteps.  ACE is
        #: autoregressive, so this is real model state; see
        #: :meth:`state_for_restart`.
        self.state: dict = {}

        if self.verbose and context.is_root:
            print(
                f"[e3sm_emulator.ace] mode={self.mode} "
                f"mesh={self.h}x{self.w} tile={self.tile_shape} "
                f"backend={_FME_BACKEND[self.mode]}",
                flush=True,
            )

    # -- setup helpers ------------------------------------------------------

    def _make_comm(self):
        if self.context.world_size == 1:
            return SerialComm()

        import torch.distributed as dist

        if not dist.is_initialized():
            # Only reachable in `single` mode. env:// picks up exactly what
            # Context.export() published.
            dist.init_process_group(backend="gloo", init_method="env://")
        return TorchComm()

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

        j_slice, i_slice = self.dist.get_local_slices((1, ny, nx))[-2:]
        mine = np.array(
            [j_slice.start or 0, j_slice.stop or ny, i_slice.start or 0,
             i_slice.stop or nx],
            dtype=np.int64,
        ).reshape(1, 4)
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
        # Cross-check that the reported slice really is this rank's tile.
        expected = tiling.tile_shape(self.comm.rank)
        actual = (int(mine[0, 1] - mine[0, 0]), int(mine[0, 3] - mine[0, 2]))
        if expected != actual:
            raise ValueError(
                f"Rank {self.comm.rank}: ACE reports a {actual} slice but the "
                f"reconstructed mesh says {expected}. The (h, w) rank ordering "
                "assumed here does not match ACE's DeviceMesh."
            )
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

        if self.owns_model:
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
            produced = {
                name: result[name].detach().to("cpu", torch.float64).numpy()[0]
                for name in self.output_names
            }
        else:
            # This rank owns no part of the globe. It still has to take part
            # in the exchange — infer() is collective.
            produced = {
                name: np.empty((0, 0), dtype=np.float64) for name in self.output_names
            }

        self._to_columns(produced, outputs)

    def _to_tiles(self, inputs: dict, names) -> dict:
        """Move every named input field onto this rank's tile, in one go."""
        columns = np.stack(
            [np.asarray(inputs[name], dtype=np.float64).reshape(-1) for name in names],
            axis=1,
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
        restarts a different atmosphere.  Returning it on *columns* rather
        than on tiles is what makes it writable through the component's
        existing restart path, and reloadable under a different rank count.

        Not yet called from anywhere: the component has no restart plumbing
        to hand it to.
        """
        if not self.state:
            return {}
        names = sorted(self.state)
        nj, ni = self.tile_shape
        stacked = np.stack(
            [
                self.state[n].detach().to("cpu").numpy().reshape(nj, ni)
                if self.owns_model
                else np.empty((nj, ni), dtype=np.float64)
                for n in names
            ],
            axis=-1,
        )
        columns = self.exchange.to_columns(stacked)
        return {name: columns[:, k] for k, name in enumerate(names)}

    def finalize(self) -> None:
        # Dropping these is what frees the weights and the carried
        # atmosphere.  Closing the stack exits Distributed.context(), which
        # is what shuts the process group down — the same object that opened
        # it closes it.
        self.state = {}
        self.stepper = None
        self._exit_stack.close()


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
