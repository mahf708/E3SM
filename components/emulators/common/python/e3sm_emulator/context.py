"""What the coupler told us: which ranks we have, and which columns we own.

The C++ backend builds this dict from :cpp:class:`InferenceContext`, which in
turn is built from the *component* MPI communicator that MCT handed the
emulator.  Everything downstream — the process group a distributed model
builds, the tiling of the global grid, device affinity — is derived from here
and from nothing else.

That is the point.  ACE's ``TorchDistributed`` and PhysicsNeMo's
``DistributedManager`` both discover their rank from ``SLURM_PROCID`` /
``SLURM_NTASKS`` when nothing better is available, and in a coupled run those
describe the *whole job*.  A process group built from them blocks forever
waiting for ocean and land ranks that will never call in.  :meth:`export` puts
the component's own numbers into the variables those libraries read, so an
unmodified upstream model initializes over exactly our ranks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

#: Environment variables that torch.distributed and PhysicsNeMo read to
#: discover the job.  We overwrite all of them.
_TORCH_ENV = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")


@dataclass
class Context:
    """Ranks, node placement and this rank's share of the grid."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    local_size: int = 1
    node_name: str = ""
    master_addr: str = "127.0.0.1"
    master_port: int = 0

    nx: int = 0
    ny: int = 0
    num_global_cols: int = 0
    col_gids: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))
    lat: np.ndarray = field(default_factory=lambda: np.empty(0, np.float64))
    lon: np.ndarray = field(default_factory=lambda: np.empty(0, np.float64))

    @classmethod
    def from_dict(cls, data: dict) -> "Context":
        """Build from the dict the C++ backend passes to the factory."""
        return cls(
            rank=int(data.get("rank", 0)),
            world_size=int(data.get("world_size", 1)),
            local_rank=int(data.get("local_rank", 0)),
            local_size=int(data.get("local_size", 1)),
            node_name=str(data.get("node_name", "")),
            master_addr=str(data.get("master_addr", "127.0.0.1")),
            master_port=int(data.get("master_port", 0)),
            nx=int(data.get("nx", 0)),
            ny=int(data.get("ny", 0)),
            num_global_cols=int(data.get("num_global_cols", 0)),
            col_gids=np.asarray(data.get("col_gids", []), dtype=np.int64),
            lat=np.asarray(data.get("lat", []), dtype=np.float64),
            lon=np.asarray(data.get("lon", []), dtype=np.float64),
        )

    @property
    def num_local_cols(self) -> int:
        return int(self.col_gids.size)

    @property
    def is_root(self) -> bool:
        return self.rank == 0

    def export(self, device_ordinal: int = 0) -> None:
        """Publish this component's rank and rendezvous to the environment.

        Call this *before* importing anything that builds a process group, and
        certainly before loading a checkpoint: ACE moves modules onto the
        device and wraps them for distribution as part of loading, so the
        device and the group have to exist first.

        ``LOCAL_RANK`` is **a device ordinal, not a rank.**  Every consumer of
        it in ACE and PhysicsNeMo feeds it straight to
        ``torch.cuda.set_device`` (``torch_distributed.py:49``,
        ``model_torch_distributed.py:144``, ``pnd_manager.py:481``), so it has
        to be an index into the devices *this process can see*.  Publishing
        the component-local rank there is wrong under exactly the binding this
        module recommends elsewhere: with one visible GPU per rank,
        ``device_count() == 1`` and the fourth rank on a node would ask for
        device 3.  Resolve it with :meth:`device_ordinal` and pass it here.
        """
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["LOCAL_RANK"] = str(int(device_ordinal))
        os.environ["MASTER_ADDR"] = self.master_addr or "127.0.0.1"
        os.environ["MASTER_PORT"] = str(self.master_port or 29500)
        # Force the env:// discovery path.  With FME_USE_SRUN=1 ACE would go
        # back to reading SLURM_PROCID/SLURM_NTASKS, which in a coupled run
        # count every rank in the job rather than every rank of this
        # component.
        os.environ["FME_USE_SRUN"] = "0"

    def device_ordinal(
        self, device_id: int | None = None, visible_devices: int | None = None
    ) -> int:
        """Which accelerator this rank owns, in the *process-visible* namespace.

        Three different numbers get confused here, so they are named apart:

        - ``rank`` and ``local_rank`` describe position among *this
          component's* processes;
        - ``visible_devices`` is how many GPUs this process can see, which the
          launcher decides via ``CUDA_VISIBLE_DEVICES``;
        - the return value indexes that visible set, and is what
          ``torch.cuda.set_device`` and ``LOCAL_RANK`` mean.

        There is no guessing, deliberately.  MCT gives us a communicator and a
        field decomposition; it does *not* give us a GPU ownership map, and our
        per-component ``local_rank`` says nothing about what the ocean and land
        ranks sharing this node have already claimed.  ``local_rank %
        device_count`` looks reasonable and quietly puts two components' rank 0
        on device 0.

        So: one visible device per rank is the supported contract — what
        ``--gpus-per-task=1 --gpu-bind=closest`` or a per-rank
        ``CUDA_VISIBLE_DEVICES`` already produces — and anything else has to be
        stated with ``inference.device_id``.

        ``visible_devices`` is injectable so this is testable without a GPU.
        """
        if visible_devices is None:
            visible_devices = _cuda_device_count()

        if visible_devices <= 0:
            return 0  # CPU run; the value is unused but must be well-formed
        if device_id is not None:
            if not 0 <= device_id < visible_devices:
                raise ValueError(
                    f"device_id={device_id} is out of range; this rank can see "
                    f"{visible_devices} device(s). Note that device_id indexes "
                    "the devices this process can see, which is what "
                    "CUDA_VISIBLE_DEVICES leaves it, not the node's devices."
                )
            return int(device_id)
        if visible_devices == 1 or self.local_size == 1:
            return 0
        raise ValueError(
            f"Rank {self.rank} can see {visible_devices} GPUs and shares this "
            f"node with {self.local_size - 1} other rank(s) of this component, "
            "so which device it owns is not ours to decide — another "
            "component's ranks may already hold some of them. Bind one device "
            "per rank in the job launcher (for example --gpus-per-task=1 "
            "--gpu-bind=closest), or state it with `inference.device_id`."
        )

    def torch_device(self, device_id: int | None = None):
        """The accelerator this rank owns, as a ``torch.device``.

        Requires torch; imported lazily so a torch-free build can still use
        the rest of this module.
        """
        import torch

        if not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda", self.device_ordinal(device_id))

    def describe(self) -> str:
        return (
            f"rank {self.rank}/{self.world_size} "
            f"(local {self.local_rank}/{self.local_size}) on "
            f"{self.node_name or '?'}, rendezvous "
            f"{self.master_addr}:{self.master_port}, "
            f"{self.num_local_cols} of {self.num_global_cols} columns "
            f"on a {self.nx}x{self.ny} grid"
        )


def _cuda_device_count() -> int:
    """Visible CUDA devices, or 0 when torch is absent or has no CUDA."""
    try:
        import torch
    except ImportError:
        return 0
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())


def torch_env_snapshot() -> dict:
    """The distributed environment as it stands, for logging and tests."""
    return {key: os.environ.get(key) for key in _TORCH_ENV}
