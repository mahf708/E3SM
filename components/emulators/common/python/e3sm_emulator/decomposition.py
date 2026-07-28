"""Getting from E3SM's columns to a global model's grid, and back.

E3SM hands each rank an arbitrary set of column global ids; a model built on
spherical transforms wants a rectangle of a ``ny x nx`` grid.  The two
decompositions never line up -- the coupler's is chosen for load balance
across the coupled job, the model's for the geometry of its own operators --
so the reconciliation is a *permutation*, computed once::

    local column  ->  global id  ->  (j, i)  ->  (owning rank, offset in tile)

and its transpose on the way out.  With the plan in hand each step costs one
all-to-all of exactly the values that have to move.  ``1 x 1`` is not a
special case: it is a tiling on N ranks where one rank owns everything and the
all-to-all degenerates into a gather and a scatter.

Nothing here knows what a model is, and nothing here imports torch.
"""

from __future__ import annotations

import numpy as np

from .comm import Comm, run_where


def split_bounds(n: int, parts: int) -> np.ndarray:
    """Start indices of ``parts`` near-equal contiguous chunks of ``n``.

    Returns ``parts + 1`` boundaries, so chunk ``p`` is
    ``[bounds[p], bounds[p + 1])``.

    **This must agree with torch_harmonics exactly.**  ACE slices its spatial
    dimensions with ``compute_split_shapes``, which gives the remainder to the
    *low*-numbered ranks.  A partition that merely looks balanced is not good
    enough: wherever ``n % parts != 0`` a different convention puts our values
    in the wrong rows of the model's tensor, which is a wrong answer rather
    than an error.  In spatial mode the plan is built from the slices ACE
    itself reports and cross-checked against this; this stays the reference
    implementation, and the thing the unit tests pin.
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
            "grid and the coupler's do not describe the same globe."
        )
    zero_based = gids - 1
    if lon_fastest:
        return zero_based // nx, zero_based % nx
    return zero_based % ny, zero_based // ny


class Tiling:
    """A rectangular partition of a ``ny x nx`` grid over ``h x w`` ranks.

    ``h`` splits latitude and ``w`` splits longitude, matching the ``(h, w)``
    process mesh ACE's ``ModelTorchDistributed`` builds.  Rank ``r`` owns tile
    ``(r // w, r % w)`` -- row-major, which is how a ``DeviceMesh`` of shape
    ``(data, h, w)`` orders its ranks with a single data replica.  A
    communicator may be larger than ``h * w``; ranks past the mesh own an
    empty tile, hand their columns over, and take the answers back.
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
            self._j_bounds = _checked_bounds(bounds[0], self.ny, self.h, "rows")
            self._i_bounds = _checked_bounds(bounds[1], self.nx, self.w, "columns")
        # Lookup tables rather than a search per cell: the grid has at most a
        # few thousand points per axis, and this makes the hot path a pair of
        # fancy-index reads.
        self._j_owner = np.repeat(np.arange(self.h), np.diff(self._j_bounds))
        self._i_owner = np.repeat(np.arange(self.w), np.diff(self._i_bounds))

    @classmethod
    def from_bounds(cls, ny: int, nx: int, h: int, w: int, j_bounds, i_bounds):
        """Build a tiling from a partition somebody else chose.

        Used in spatial mode, where the partition has to be ACE's rather than
        ours: deriving the plan from what ACE reports removes the possibility
        of two partitioning algorithms drifting apart.
        """
        return cls(ny, nx, h, w, bounds=(j_bounds, i_bounds))

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
        ni = np.diff(self._i_bounds)[wr]
        offset = (j - self._j_bounds[hr]) * ni + (i - self._i_bounds[wr])
        return (hr * self.w + wr).astype(np.int64), offset.astype(np.int64)


class PermutationExchange:
    """Moves values between local columns and a tile of the global grid.

    Both directions are the same plan read forwards and backwards, so a round
    trip is exact by construction rather than by agreement between two pieces
    of index arithmetic.  All the fields go in one exchange, because message
    count is what hurts at scale.
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
                f"has {comm.size}. Give the component more ranks, or a smaller "
                "mesh."
            )
        self.comm = comm
        self.tiling = tiling
        self.num_local_cols = int(np.asarray(col_gids).size)
        self.tile_shape = tiling.tile_shape(comm.rank)
        self.tile_size = self.tile_shape[0] * self.tile_shape[1]

        # cell_indices() inspects *this rank's* global ids, so it can fail on
        # one rank alone -- and it runs before the exchanges below, so that
        # rank would hang the others.  Settle it collectively first: every
        # rank reaches exactly one agreement before any exchange, and one
        # after.
        dest_rank, dest_offset = _agreed(
            comm,
            lambda: tiling.locate(
                *cell_indices(col_gids, tiling.ny, tiling.nx, lon_fastest)
            ),
        )

        # Group the local columns by destination, so each rank's share is one
        # contiguous slice of the send buffer.
        self._send_order = np.lexsort((dest_offset, dest_rank))
        self._send_counts = np.bincount(dest_rank, minlength=comm.size)
        self._recv_counts = comm.exchange_counts(self._send_counts)

        # Learn, once, where each value we are about to receive belongs.
        self._recv_offsets = comm.alltoall(
            dest_offset[self._send_order].reshape(-1, 1),
            self._send_counts,
            self._recv_counts,
        ).reshape(-1)

        _agreed(comm, self._validate)

    def _validate(self) -> None:
        """Refuse a decomposition that does not cover this rank's tile.

        Worth doing eagerly: a mismatched grid produces a field with holes in
        it, and a model will happily consume that and return something
        plausible.  A loud failure at initialization is the only way this gets
        noticed.
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
                f"{self.tile_size} cells not at all, and as many others twice. "
                "Check the column global ids and `lon_fastest`."
            )

    def to_tile(self, values: np.ndarray) -> np.ndarray:
        """``(ncol, k)`` on the coupler's columns -> ``(nj, ni, k)``.

        **Collective.**  The shape check is settled across the ranks first: it
        inspects only this rank's array, so a caller that supplied the wrong
        column count on one rank would otherwise raise there while everybody
        else entered the all-to-all.  Keeping it here rather than in the caller
        puts the invariant next to the communication it protects, and covers
        callers not yet written.
        """
        values = _agreed(
            self.comm, lambda: _as_2d(values, self.num_local_cols, "column values")
        )
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
        """``(nj, ni, k)`` -> ``(ncol, k)``.  Collective; see :meth:`to_tile`."""
        tile = _agreed(self.comm, lambda: _as_2d(tile, self.tile_size, "tile values"))
        recv = self.comm.alltoall(
            tile[self._recv_offsets], self._recv_counts, self._send_counts
        )
        columns = np.empty((self.num_local_cols, tile.shape[1]), dtype=np.float64)
        columns[self._send_order] = recv
        return columns


def _checked_bounds(bounds, extent: int, parts: int, what: str) -> np.ndarray:
    """Validate a partition somebody else chose.

    A tiling built from these must still cover the grid exactly once, so the
    invariant is checked here rather than by whoever consumes the tiles: a
    partition that starts late, ends early or runs backwards would otherwise
    show up as an index error deep in the exchange, or as a field with holes
    in it that a model consumes happily.
    """
    bounds = np.asarray(bounds, np.int64)
    if bounds.size != parts + 1:
        raise ValueError(
            f"Need {parts + 1} boundaries for {parts} parts of {what}, got "
            f"{bounds.size}."
        )
    if bounds[0] != 0 or bounds[-1] != extent:
        raise ValueError(
            f"The supplied {what} bounds run {bounds[0]}..{bounds[-1]}, not "
            f"0..{extent}."
        )
    if np.any(np.diff(bounds) <= 0):
        raise ValueError(
            f"The supplied {what} bounds {bounds.tolist()} are not increasing, "
            "so some part would own nothing or overlap its neighbour."
        )
    return bounds


def _agreed(comm: Comm, work):
    """Run rank-local work that precedes a collective, agreeing on failure.

    ``run_where(comm, True, work)`` with a name that says why it is there: the
    work runs on every rank, but it can fail on *one*, and that rank must not
    raise while the others walk into the exchange.
    """
    return run_where(comm, True, work)


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
