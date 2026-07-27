"""Running an ACE checkpoint as an E3SM component.

ACE is the hard case, and worth being precise about why.  Its samples are
*globes*: ``Stepper.step`` takes ``{name: tensor[n_batch, n_lat, n_lon]}``, so
one sample is an entire atmosphere.  E3SM's local columns are therefore not
ACE's batch dimension — splitting the grid across ranks splits one sample
spatially, and a model with a global receptive field (spherical transforms,
global attention) gives *wrong answers rather than errors* if you pretend
otherwise.  Everything below exists to make that impossible to do by accident.

Three ways to spread one globe over the ranks the coupler gave us, each
mapping onto one of ACE's own distributed backends:

======================  ======================  =============================
``inference.ace_mode``  ACE backend             What happens
======================  ======================  =============================
``single``              ``NonDistributed``      One rank assembles the globe
                                                and runs an unmodified
                                                checkpoint; the others take
                                                part only in the exchange.
                                                Reference behavior — validate
                                                everything else against it.
``spatial``             ``ModelTorchDistributed``  Ranks form an ``h x w``
                                                mesh, each owning a rectangle;
                                                ACE's distributed transforms
                                                carry the coupling between
                                                them.
``ensemble``            ``TorchDistributed``    Every rank holds the whole
                                                globe as its own batch member
                                                and the members are averaged
                                                on the way out.
======================  ======================  =============================

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

    emulator:    ace
    model_path:  /path/to/ace_ckpt.tar
    ace_mode:    auto          # auto | single | spatial | ensemble
    ace_h:       4             # spatial only: latitude ranks
    ace_w:       8             # spatial only: longitude ranks
    input:       air_temperature_0     # ACE's own variable names
    output:      air_temperature_0
    lon_fastest: true          # how the coupler numbers its columns

An input named ``<name>_next`` is routed to the stepper's
``next_step_input_data`` as ``<name>``; that is how prescribed SSTs and
forcings valid at the *end* of the step get in.
"""

from __future__ import annotations

import os

import numpy as np

from .comm import SerialComm, TorchComm
from .context import Context
from .decomposition import PermutationExchange, ReplicaExchange, Tiling

#: ace_mode -> the value ACE's Distributed selector reads.
_FME_BACKEND = {"single": "none", "spatial": "model", "ensemble": "torch"}

#: Suffix marking an input that belongs to the *next* step.
_NEXT_SUFFIX = "_next"


def build(config: dict, context: Context) -> "AceEmulator":
    return AceEmulator(config, context)


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
        if world == 1:
            mode = "single"
        elif h * w == world and h * w > 1:
            mode = "spatial"
        else:
            # Refuse to guess a mesh.  Which factorization of the ranks is
            # right depends on the model's transforms and on the machine, and
            # picking one silently is how a run ends up slow or wrong.
            mode = "single"

    if mode not in _FME_BACKEND:
        raise ValueError(
            f"Unknown ace_mode '{mode}'. Use one of: "
            f"{', '.join(sorted(_FME_BACKEND))}."
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
                "Every rank must own a rectangle."
            )
    else:
        h = w = 1

    return mode, h, w


class AceEmulator:
    """One ACE checkpoint, stepped once per E3SM timestep."""

    def __init__(self, config: dict, context: Context):
        self.context = context
        self.verbose = bool(config.get("verbose", False))
        self.mode, self.h, self.w = resolve_mode(config, context)
        self.lon_fastest = str(config.get("lon_fastest", "true")).lower() not in (
            "false",
            "0",
            "no",
            "off",
        )

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

        self.dist = Distributed.get_instance()

        # `single` runs ACE with NonDistributed, so nothing upstream builds a
        # process group — but the exchange still needs one.  The other two
        # modes have already built theirs, and initializing again would fail.
        self.comm = self._make_comm()

        # 2. The decomposition.  Built before the checkpoint so a grid
        #    mismatch is reported in milliseconds rather than after a
        #    multi-gigabyte load.
        if self.mode == "ensemble":
            self.exchange = ReplicaExchange(
                self.comm, context.col_gids, ny, nx, self.lon_fastest
            )
            self.tile_shape = (ny, nx)
        else:
            tiling = Tiling(ny, nx, self.h, self.w)
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
            from fme.ace.stepper.single_module import load_stepper

            self.stepper = load_stepper(model_path)

        #: Prognostic fields carried between timesteps.  ACE is
        #: autoregressive, so this is real model state — writing it at an
        #: E3SM restart boundary is not solved here.
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

            if self.mode == "ensemble":
                # Each rank ran a different member; the coupler gets the mean.
                result = {
                    name: self.dist.reduce_mean(value.clone())
                    for name, value in result.items()
                }

            self.state = {
                name: result[name]
                for name in self.stepper.prognostic_names
                if name in result
            }
            produced = {
                name: result[name].detach().to("cpu", torch.float64).numpy()[0]
                for name in self.output_names
                if name in result
            }
            missing = [n for n in self.output_names if n not in result]
            if missing:
                raise ValueError(
                    f"The ACE stepper did not produce {missing}. It produces: "
                    f"{sorted(result)}."
                )
        else:
            # This rank owns no part of the globe this time. It still has to
            # take part in the exchange — infer() is collective.
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
        if self.mode == "ensemble":
            grid = self.exchange.to_grid(columns)
        else:
            grid = self.exchange.to_tile(columns)
        return {name: np.ascontiguousarray(grid[..., k]) for k, name in enumerate(names)}

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

    def finalize(self) -> None:
        # Dropping these is what actually frees the weights and the carried
        # atmosphere.  The process group is deliberately left up: the
        # interpreter cannot be restarted anyway (numpy and torch both refuse
        # to load twice in one process), and tearing a group down while
        # another emulator in the same run still holds it would be worse than
        # leaking it until the process exits.
        self.state = {}
        self.stepper = None
