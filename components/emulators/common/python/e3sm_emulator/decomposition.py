"""Getting from E3SM's columns to a global model's grid, and back.

E3SM hands each rank an arbitrary set of column global ids.  A global model
(ACE and anything else built on spherical transforms) wants a rectangular
patch of a ``ny x nx`` latitude-longitude grid, because that is what its
distributed transforms slice.  The two decompositions do not line up, and
never will: the coupler's is chosen for load balance across the whole coupled
job, the model's for the geometry of its own operators.

So the job here is a *permutation*, computed once:

    local column  ->  global id  ->  (j, i)  ->  (owning rank, offset in tile)

and its transpose on the way out.  With the plan in hand each step costs one
all-to-all of exactly the values that have to move, which is the cheapest this
can be — no rank assembles a field it does not need, and nothing is broadcast.

The tiling covers the three ways a model can be spread over the ranks:

===============  ==========================================================
``1 x 1``        one rank assembles the globe, the rest idle through the
                 model call.  The all-to-all degenerates into a gather.
``h x w``        every rank owns a rectangle and the model's own collectives
                 do the rest.  The permutation is the whole cost.
replicated       every rank wants the entire globe (an ensemble member
                 each).  Not a permutation; see :class:`ReplicaExchange`.
===============  ==========================================================

Nothing here knows what a model is, and nothing here imports torch.
"""

from __future__ import annotations

import numpy as np

from .comm import Comm


def _as_2d(values: np.ndarray, expected_rows: int, what: str) -> np.ndarray:
    """View an array as ``(expected_rows, k)``, accepting any trailing shape.

    Written out rather than done with ``reshape(n, -1)`` because numpy cannot
    infer the trailing extent of an empty array, and a rank owning no columns
    is perfectly normal on a large layout.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    elif values.ndim == 3:
        # (nj, ni, k) is how a tile arrives; the leading pair is the rows.
        values = values.reshape(values.shape[0] * values.shape[1], values.shape[2])
    elif values.ndim != 2:
        raise ValueError(f"Expected 1-, 2- or 3-D {what}, got shape {values.shape}.")
    if values.shape[0] != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows of {what}, got {values.shape[0]}."
        )
    return values


def split_bounds(n: int, parts: int) -> np.ndarray:
    """Start indices of ``parts`` near-equal contiguous chunks of ``n``.

    Returns ``parts + 1`` boundaries, so chunk ``p`` is
    ``[bounds[p], bounds[p + 1])``.

    **This must agree with torch_harmonics exactly.**  ACE slices its spatial
    dimensions with ``torch_harmonics.distributed.compute_split_shapes``,
    which is ``[base + 1] * remainder + [base] * (parts - remainder)`` — the
    remainder goes to the *low*-numbered ranks.  A partition that merely looks
    balanced is not good enough: on any grid where ``n % parts != 0`` a
    different convention puts our values in the wrong rows of the model's
    tensor, which is a wrong answer rather than an error.

    Where it matters — spatial mode — the plan is built from the slices ACE
    itself reports rather than from this function, and the two are
    cross-checked (see :meth:`Tiling.from_bounds` and
    ``e3sm_emulator.ace``).  This stays the reference implementation, and the
    thing the unit tests pin.
    """
    if parts <= 0:
        raise ValueError(f"Need at least one part, got {parts}.")
    if n < parts:
        raise ValueError(
            f"Cannot split {n} points over {parts} ranks: some rank would own "
            "nothing, and a model with a spatial receptive field cannot work "
            "with an empty tile."
        )
    base, remainder = divmod(int(n), int(parts))
    sizes = [base + 1] * remainder + [base] * (parts - remainder)
    return np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)


class Tiling:
    """A rectangular partition of a ``ny x nx`` grid over ``h x w`` ranks.

    ``h`` splits latitude and ``w`` splits longitude, matching the ``(h, w)``
    process mesh ACE's ``ModelTorchDistributed`` builds.  Rank ``r`` in the
    spatial group owns tile ``(r // w, r % w)`` — row-major, which is how a
    ``DeviceMesh`` of shape ``(data, h, w)`` orders its ranks when there is a
    single data replica.

    A communicator may be larger than ``h * w``.  Ranks past the mesh own an
    empty tile: they hand their columns over, wait, and take the answers
    back.  ``1 x 1`` on many ranks is exactly that case, and is how one rank
    comes to hold the whole globe.
    """

    def __init__(self, ny: int, nx: int, h: int = 1, w: int = 1, bounds=None):
        if ny <= 0 or nx <= 0:
            raise ValueError(f"Need a positive grid shape, got {ny}x{nx}.")
        self.ny, self.nx = int(ny), int(nx)
        self.h, self.w = int(h), int(w)
        self.size = self.h * self.w
        if bounds is None:
            self._j_bounds = split_bounds(self.ny, self.h)
            self._i_bounds = split_bounds(self.nx, self.w)
        else:
            self._j_bounds, self._i_bounds = bounds
            if self._j_bounds[-1] != self.ny or self._i_bounds[-1] != self.nx:
                raise ValueError(
                    f"The supplied bounds cover "
                    f"{self._j_bounds[-1]}x{self._i_bounds[-1]}, not the "
                    f"{self.ny}x{self.nx} grid."
                )
        # Lookup tables rather than searchsorted per cell: the grid has at
        # most a few thousand points per axis, and this makes the hot path a
        # pair of fancy-index reads.
        self._j_owner = np.repeat(np.arange(self.h), np.diff(self._j_bounds))
        self._i_owner = np.repeat(np.arange(self.w), np.diff(self._i_bounds))

    @classmethod
    def from_bounds(cls, ny: int, nx: int, h: int, w: int, j_bounds, i_bounds):
        """Build a tiling from a partition somebody else chose.

        Used in spatial mode, where the partition has to be ACE's rather than
        ours: the model slices its tensors with
        ``torch_harmonics.compute_split_shapes`` and we have to land our
        columns in exactly those rows and columns.  Deriving the plan from
        what ACE reports removes the possibility of two partitioning
        algorithms drifting apart.
        """
        tiling = cls(
            ny,
            nx,
            h,
            w,
            bounds=(np.asarray(j_bounds, np.int64), np.asarray(i_bounds, np.int64)),
        )
        return tiling

    def agrees_with_even_split(self) -> bool:
        """True if this partition is the one :func:`split_bounds` would pick."""
        return np.array_equal(
            self._j_bounds, split_bounds(self.ny, self.h)
        ) and np.array_equal(self._i_bounds, split_bounds(self.nx, self.w))

    def tile_shape(self, rank: int) -> tuple[int, int]:
        """``(nj, ni)`` of the tile owned by ``rank``; ``(0, 0)`` past the mesh."""
        if int(rank) >= self.size:
            return (0, 0)
        hr, wr = divmod(int(rank), self.w)
        return (
            int(self._j_bounds[hr + 1] - self._j_bounds[hr]),
            int(self._i_bounds[wr + 1] - self._i_bounds[wr]),
        )

    def tile_origin(self, rank: int) -> tuple[int, int]:
        """``(j0, i0)`` of the tile owned by ``rank``."""
        hr, wr = divmod(int(rank), self.w)
        return int(self._j_bounds[hr]), int(self._i_bounds[wr])

    def locate(self, j: np.ndarray, i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Owning rank and offset within that rank's tile, for each cell."""
        hr = self._j_owner[j]
        wr = self._i_owner[i]
        rank = hr * self.w + wr
        ni = np.diff(self._i_bounds)[wr]
        offset = (j - self._j_bounds[hr]) * ni + (i - self._i_bounds[wr])
        return rank.astype(np.int64), offset.astype(np.int64)


def cell_indices(
    col_gids: np.ndarray, ny: int, nx: int, lon_fastest: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """``(j, i)`` grid indices of 1-based column global ids.

    E3SM numbers the columns of a structured grid in a single sweep.
    ``lon_fastest`` says longitude varies fastest, which is the usual
    ordering; set it False for a grid numbered down the columns instead.
    """
    gids = np.asarray(col_gids, dtype=np.int64)
    if gids.size and (gids.min() < 1 or gids.max() > ny * nx):
        raise ValueError(
            f"Column global ids run {gids.min()}..{gids.max()}, which does not "
            f"fit a {ny}x{nx} grid holding {ny * nx} columns. The emulator's "
            "grid and the coupler's do not agree."
        )
    zero_based = gids - 1
    if lon_fastest:
        return zero_based // nx, zero_based % nx
    return zero_based % ny, zero_based // ny


class PermutationExchange:
    """Moves values between local columns and a tile of the global grid.

    Both directions are the same plan read forwards and backwards, so a
    round trip is exact by construction rather than by agreement between two
    pieces of index arithmetic.
    """

    def __init__(
        self,
        comm: Comm,
        col_gids: np.ndarray,
        tiling: Tiling,
        lon_fastest: bool = True,
    ):
        if tiling.size > comm.size:
            raise ValueError(
                f"The tiling wants {tiling.size} ranks but the communicator "
                f"has {comm.size}. Give the component more ranks, or a "
                "smaller mesh."
            )
        self.comm = comm
        self.tiling = tiling
        self.num_local_cols = int(np.asarray(col_gids).size)
        self.tile_shape = tiling.tile_shape(comm.rank)
        self.tile_size = self.tile_shape[0] * self.tile_shape[1]

        # cell_indices() inspects *this rank's* global ids, so it can fail on
        # one rank alone — and it runs before the exchanges below.  A rank
        # that raised here while the others entered exchange_counts() would
        # hang them, so settle it collectively first.  Every rank reaches
        # exactly one agree() before any exchange, and one after.
        try:
            j, i = cell_indices(col_gids, tiling.ny, tiling.nx, lon_fastest)
            dest_rank, dest_offset = tiling.locate(j, i)
            problem, error = "", None
        except Exception as exc:  # noqa: BLE001 - re-raised by agree()
            j = i = dest_rank = dest_offset = None
            problem, error = f"rank {comm.rank}: {exc}", exc
        comm.agree(problem, error)

        # Group the local columns by destination, so each rank's share is one
        # contiguous slice of the send buffer.
        self._send_order = np.lexsort((dest_offset, dest_rank))
        self._send_counts = np.bincount(dest_rank, minlength=comm.size)
        self._recv_counts = comm.exchange_counts(self._send_counts)

        # Learn, once, where each value we are about to receive belongs.
        ordered_offsets = dest_offset[self._send_order].reshape(-1, 1)
        self._recv_offsets = comm.alltoall(
            ordered_offsets, self._send_counts, self._recv_counts
        ).reshape(-1)

        # _validate() checks *this rank's* tile, so it too can fail alone.
        try:
            self._validate()
            problem, error = "", None
        except Exception as exc:  # noqa: BLE001 - re-raised by agree()
            problem, error = f"rank {comm.rank}: {exc}", exc
        comm.agree(problem, error)

    def _validate(self) -> None:
        """Refuse a decomposition that does not cover this rank's tile.

        Worth doing eagerly: a mismatched grid produces a field with holes in
        it, and a model will happily consume that and return something
        plausible.  A loud failure at initialization is the only way this
        gets noticed.
        """
        received = int(self._recv_offsets.size)
        if received != self.tile_size:
            raise ValueError(
                f"Rank {self.comm.rank} owns a {self.tile_shape[0]}x"
                f"{self.tile_shape[1]} tile ({self.tile_size} cells) but the "
                f"columns route {received} values to it. The coupler's grid "
                "and the model's do not describe the same globe."
            )
        seen = np.zeros(self.tile_size, dtype=bool)
        seen[self._recv_offsets] = True
        if not seen.all():
            # The count already matched, so anything uncovered means some
            # other cell is covered twice.
            missing = int((~seen).sum())
            raise ValueError(
                f"Rank {self.comm.rank} would get {missing} of its "
                f"{self.tile_size} cells not at all, and as many others "
                "twice. Check the column global ids and `lon_fastest`."
            )

    def to_tile(self, values: np.ndarray) -> np.ndarray:
        """``(ncol, k)`` on the coupler's columns -> ``(nj, ni, k)``."""
        values = _as_2d(values, self.num_local_cols, "column values")
        recv = self.comm.alltoall(
            values[self._send_order], self._send_counts, self._recv_counts
        )
        width = values.shape[1]
        tile = np.empty((self.tile_size, width), dtype=np.float64)
        tile[self._recv_offsets] = recv
        # Spell out the trailing extent: a rank that owns no tile has a
        # zero-size array, and numpy cannot infer a -1 through one.
        return tile.reshape(self.tile_shape[0], self.tile_shape[1], width)

    def to_columns(self, tile: np.ndarray) -> np.ndarray:
        """``(nj, ni, k)`` -> ``(ncol, k)`` on the coupler's columns."""
        tile = _as_2d(tile, self.tile_size, "tile values")
        recv = self.comm.alltoall(
            tile[self._recv_offsets], self._recv_counts, self._send_counts
        )
        columns = np.empty((self.num_local_cols, tile.shape[1]), dtype=np.float64)
        columns[self._send_order] = recv
        return columns


class ReplicaExchange:
    """Gives every rank the whole globe, for one ensemble member each.

    Not a permutation — every value goes everywhere — so it costs an
    all-gather of the entire field per step and holds a full globe per rank.
    That is the price of data parallelism over a model whose sample *is* the
    globe.  The return trip needs no communication at all: each rank already
    holds the answer for its own columns.

    **Nothing currently uses this.**  It is the piece an ensemble contract
    would need, and it is kept (and tested) because that contract is a
    plausible next step — but data parallelism over a *single* coupled
    trajectory is not one: every rank would run the same deterministic model
    on the same globe.  Wiring it up means the component first supplying
    genuinely independent members, each with its own initial state, random
    state, prescribed forcings, autoregressive state and outputs.  See
    ``e3sm_emulator.ace`` for why ``TorchDistributed`` is not offered today.
    """

    def __init__(
        self,
        comm: Comm,
        col_gids: np.ndarray,
        ny: int,
        nx: int,
        lon_fastest: bool = True,
    ):
        self.comm = comm
        self.ny, self.nx = int(ny), int(nx)
        self.num_local_cols = int(np.asarray(col_gids).size)

        j, i = cell_indices(col_gids, ny, nx, lon_fastest)
        self._local_flat = (j * nx + i).astype(np.int64)
        self._all_flat = comm.allgather(self._local_flat.reshape(-1, 1)).reshape(-1)

        if self._all_flat.size != ny * nx:
            raise ValueError(
                f"The ranks together hold {self._all_flat.size} columns but "
                f"the {ny}x{nx} grid has {ny * nx}. The coupler's grid and "
                "the model's do not describe the same globe."
            )
        seen = np.zeros(ny * nx, dtype=bool)
        seen[self._all_flat] = True
        if not seen.all():
            raise ValueError(
                "The columns do not cover the global grid exactly once. "
                "Check the column global ids and `lon_fastest`."
            )

    def to_grid(self, values: np.ndarray) -> np.ndarray:
        """``(ncol, k)`` -> the full ``(ny, nx, k)`` grid, on every rank."""
        values = _as_2d(values, self.num_local_cols, "column values")
        width = values.shape[1]
        gathered = self.comm.allgather(values)
        grid = np.empty((self.ny * self.nx, width), dtype=np.float64)
        grid[self._all_flat] = gathered
        return grid.reshape(self.ny, self.nx, width)

    def to_columns(self, grid: np.ndarray) -> np.ndarray:
        """``(ny, nx, k)`` -> this rank's columns.  Purely local."""
        grid = _as_2d(grid, self.ny * self.nx, "grid values")
        return grid[self._local_flat]
