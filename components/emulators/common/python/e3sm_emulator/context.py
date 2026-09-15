"""What the coupler told us: which ranks we have, and which columns we own.

Built only from the component's own MPI communicator, never from
``SLURM_PROCID``/``SLURM_NTASKS``, which would see the whole coupled job.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Context:
    """Ranks and this rank's share of the grid."""

    rank: int = 0
    world_size: int = 1
    node_name: str = ""
    #: The component communicator's Fortran handle, or -1.
    fortran_comm: int = -1
    #: True when ``infer`` sees the whole grid gathered on this rank (a global
    #: network stepped by the component), so the grid describes every column.
    gathered: bool = False

    nx: int = 0
    ny: int = 0
    num_global_cols: int = 0
    col_gids: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    lat: np.ndarray = field(default_factory=lambda: np.empty(0, np.float64))
    lon: np.ndarray = field(default_factory=lambda: np.empty(0, np.float64))

    @classmethod
    def from_dict(cls, data: dict) -> "Context":
        """Build from the dict the C++ backend passes to the factory."""
        return cls(
            rank=int(data.get("rank", 0)),
            world_size=int(data.get("world_size", 1)),
            node_name=str(data.get("node_name", "")),
            fortran_comm=int(data.get("fortran_comm", -1)),
            gathered=bool(data.get("gathered", False)),
            nx=int(data.get("nx", 0)),
            ny=int(data.get("ny", 0)),
            num_global_cols=int(data.get("num_global_cols", 0)),
            col_gids=np.asarray(data.get("col_gids", []), dtype=np.int64),
            lat=np.asarray(data.get("lat", []), dtype=np.float64),
            lon=np.asarray(data.get("lon", []), dtype=np.float64),
        )

    def mpi_comm(self):
        """The component communicator as an mpi4py communicator.

        Raises ImportError without mpi4py, and ValueError when the context
        came from no communicator.
        """
        if self.fortran_comm < 0:
            raise ValueError("this context carries no communicator")
        from mpi4py import MPI

        return MPI.Comm.f2py(self.fortran_comm)

    @property
    def num_local_cols(self) -> int:
        return int(self.col_gids.size)

    @property
    def is_root(self) -> bool:
        return self.rank == 0

    def describe(self) -> str:
        return (
            f"rank {self.rank}/{self.world_size} on {self.node_name or '?'}, "
            f"{self.num_local_cols} of {self.num_global_cols} columns "
            f"on a {self.nx}x{self.ny} grid"
        )
