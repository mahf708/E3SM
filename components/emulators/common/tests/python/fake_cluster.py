"""A whole cluster in one process, so the index arithmetic can be tested.

The decomposition is the part of this design most likely to be silently
wrong, and it is also the part that normally needs a job launcher to
exercise.  :class:`FakeComm` closes that gap: N threads share a rendezvous
and step through the collectives together, held in lockstep by a barrier.
No MPI, no torch, no launcher — but the same
:class:`~e3sm_emulator.comm.Comm` interface the real implementation
satisfies, so what passes here is the code that runs on the machine.
"""

from __future__ import annotations

import threading

import numpy as np

from e3sm_emulator.comm import Comm


class Cluster:
    """Rendezvous shared by the fake ranks."""

    def __init__(self, size: int):
        self.size = size
        self.barrier = threading.Barrier(size)
        self.alltoall_inbox = [[None] * size for _ in range(size)]
        self.allgather_inbox = [None] * size


class FakeComm(Comm):
    """One rank's view of a :class:`Cluster`.

    Every collective is "publish my contribution, barrier, read everybody's",
    which is exactly the ordering a real collective enforces.
    """

    def __init__(self, cluster: Cluster, rank: int):
        self._cluster = cluster
        self.rank = rank
        self.size = cluster.size

    def alltoall(self, send, send_counts, recv_counts):
        send = np.atleast_2d(np.asarray(send))
        width = send.shape[1] if send.ndim > 1 else 1
        starts = np.concatenate([[0], np.cumsum([int(c) for c in send_counts])])
        for r in range(self.size):
            self._cluster.alltoall_inbox[r][self.rank] = np.array(
                send[starts[r] : starts[r + 1]], copy=True
            )
        self._cluster.barrier.wait()
        blocks = [np.asarray(b) for b in self._cluster.alltoall_inbox[self.rank]]
        for source, (block, expected) in enumerate(zip(blocks, recv_counts)):
            if block.shape[0] != int(expected):
                raise AssertionError(
                    f"rank {self.rank} expected {expected} items from "
                    f"{source}, got {block.shape[0]}"
                )
        self._cluster.barrier.wait()
        if not blocks:
            return np.empty((0, width), dtype=send.dtype)
        return np.concatenate(blocks, axis=0)

    def allgather(self, block):
        block = np.atleast_2d(np.asarray(block))
        self._cluster.allgather_inbox[self.rank] = np.array(block, copy=True)
        self._cluster.barrier.wait()
        gathered = np.concatenate(
            [np.asarray(b) for b in self._cluster.allgather_inbox], axis=0
        )
        self._cluster.barrier.wait()
        return gathered


def run_ranks(size: int, body):
    """Run ``body(comm, rank)`` on ``size`` fake ranks and collect the results.

    An exception on any rank is re-raised here, so a test failure inside a
    rank is a test failure and not a hang.
    """
    cluster = Cluster(size)
    results: list = [None] * size
    errors: list = [None] * size

    def target(rank: int) -> None:
        try:
            results[rank] = body(FakeComm(cluster, rank), rank)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors[rank] = exc
            # Let the other ranks out of whatever barrier they are waiting on
            # rather than deadlocking the whole test run.
            cluster.barrier.abort()

    threads = [threading.Thread(target=target, args=(r,)) for r in range(size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # A rank that aborted the barrier takes the others down with it, so
    # report a real failure ahead of the collateral BrokenBarrierErrors.
    raised = [e for e in errors if e is not None]
    for error in raised:
        if not isinstance(error, threading.BrokenBarrierError):
            raise error
    if raised:
        raise raised[0]
    return results
