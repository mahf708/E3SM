"""The two collectives the decomposition needs, and nothing else.

Keeping them behind a tiny interface has two payoffs: the index arithmetic in
:mod:`e3sm_emulator.decomposition` is testable on one process with a fake
cluster, and the real implementation can sit on ``torch.distributed`` without
the rest of the package importing torch.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

#: How much of a failure message :meth:`Comm.agree` will carry.  Truncating a
#: diagnostic is acceptable; :meth:`Comm.allgather_text` itself is exact,
#: because it also carries the restart field schema.
_MESSAGE_BYTES = 2048


class Comm:
    """Collectives over the emulator's ranks.

    Buffers are 2-D ``(n, k)`` numpy arrays: ``n`` items, ``k`` values each.
    Counts are always in *items*, never in elements or bytes.
    """

    rank: int = 0
    size: int = 1

    def alltoall(
        self,
        send: np.ndarray,
        send_counts: Sequence[int],
        recv_counts: Sequence[int],
    ) -> np.ndarray:
        """Send ``send_counts[r]`` items to rank ``r``, in rank order."""
        raise NotImplementedError

    def allgather(self, block: np.ndarray) -> np.ndarray:
        """Concatenate every rank's block, in rank order, on every rank."""
        raise NotImplementedError

    def exchange_counts(self, send_counts: Sequence[int]) -> np.ndarray:
        """Learn how many items each rank is about to send us.

        An all-to-all of exactly one integer per rank, expressed with the
        primitive above so an implementation only ever has to write one.
        """
        send = np.asarray(send_counts, dtype=np.int64).reshape(self.size, 1)
        ones = [1] * self.size
        return self.alltoall(send, ones, ones).reshape(self.size)

    def any_true(self, flag: bool) -> bool:
        """One bit per rank, reduced with OR.  The cheapest collective there is.

        Exists so that :meth:`agree` costs a single small reduction when
        nothing is wrong, which is every timestep of a healthy run.  Gathering
        the diagnostic text is then paid for only when there is text to
        gather.

        The default goes through :meth:`allgather`; implementations with a
        real reduction should override it with one.
        """
        gathered = self.allgather(
            np.array([[1 if flag else 0]], dtype=np.int64)
        )
        return bool(gathered.any())

    def allgather_text(self, text: str) -> list[str]:
        """One string per rank, in rank order.  Exact, never truncated.

        Two all-gathers: the byte lengths, then blocks padded to the longest.
        A fixed width would have been one call, but this also carries the
        restart field schema, and silently dropping a state variable because
        the name list ran past a buffer is not a trade worth making. Callers
        that only want a diagnostic can truncate before calling.
        """
        if self.size == 1:
            return [text]

        raw = text.encode("utf-8")
        lengths = self.allgather(
            np.array([[len(raw)]], dtype=np.int64)
        ).reshape(self.size)
        width = int(lengths.max())
        if width == 0:
            return [""] * self.size

        block = np.zeros(width, dtype=np.uint8)
        block[: len(raw)] = np.frombuffer(raw, dtype=np.uint8)
        gathered = self.allgather(block.reshape(-1, 1)).reshape(self.size, width)
        return [
            bytes(gathered[r][: int(lengths[r])]).decode("utf-8", "replace")
            for r in range(self.size)
        ]

    def agree(self, problem: str = "", error: BaseException | None = None) -> None:
        """Raise on every rank, or on none.

        The failure this exists to prevent: a model that only *some* ranks
        load, or a check that only fails on the rank whose data is wrong, so
        one rank raises while the others sail on into the next collective and
        the run hangs instead of failing.  A hang tells you nothing; this
        turns it into an error that names the cause on every rank.

        Every rank's message is exchanged, not just a flag, so a rank that was
        fine still reports the real reason the run is stopping rather than a
        bare "some other rank failed".

        **This is itself a collective.**  It must be reached exactly once, in
        the same order, on every rank — which means it belongs *after* a
        collective section, never inside one branch of an `if`.

        Args:
            problem: This rank's complaint, or "" if it has none.
            error: The exception ``problem`` describes, when there was one.
                It is re-raised unchanged on the rank that hit it, so the
                caller still sees the real type and the original traceback;
                the other ranks get a RuntimeError quoting it.
        """
        if self.size == 1:
            if problem:
                raise error if error is not None else RuntimeError(problem)
            return

        # The healthy path — every step of a run that is working — pays for
        # one small reduction and stops here.  Only a real failure is worth
        # the cost of moving everybody's message around.
        if not self.any_true(bool(problem)):
            return

        # A diagnostic may be truncated; that is a fair trade for a bounded
        # exchange, and unlike a field schema nothing downstream depends on
        # its exact contents.
        problems = {
            rank: text
            for rank, text in enumerate(
                self.allgather_text(problem[:_MESSAGE_BYTES])
            )
            if text
        }
        if not problems:
            return
        if problem:
            raise error if error is not None else RuntimeError(problem)
        rank, text = sorted(problems.items())[0]
        raise RuntimeError(
            f"This rank completed this phase cleanly, but rank {rank} did "
            f"not, so the run is stopping on every rank rather than hanging. "
            f"Rank {rank} said:\n{text}"
        )



class SerialComm(Comm):
    """One rank. Every collective is a no-op or a copy."""

    rank = 0
    size = 1

    def alltoall(self, send, send_counts, recv_counts):
        send = np.atleast_2d(np.asarray(send))
        if int(send_counts[0]) != int(recv_counts[0]):
            raise ValueError(
                f"Serial all-to-all must send what it receives, got "
                f"{list(send_counts)} and {list(recv_counts)}."
            )
        return np.array(send[: int(send_counts[0])], copy=True)

    def allgather(self, block):
        return np.array(np.atleast_2d(np.asarray(block)), copy=True)

    def any_true(self, flag):
        return bool(flag)


class TorchComm(Comm):
    """Collectives over ``torch.distributed``.

    Always on a gloo group, even when the model runs on GPUs: what moves here
    is host memory that came from, and goes back to, E3SM's own arrays, and
    staging it through the device to use NCCL would cost two extra copies to
    save nothing.

    The all-to-all is written with point-to-point sends rather than
    ``all_to_all_single`` because gloo does not implement the latter, and a
    CPU exchange is exactly the case that has to work everywhere.
    """

    def __init__(self, group=None):
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError(
                "TorchComm needs an initialized torch.distributed. Call "
                "Context.export() and let the model build its process group "
                "first."
            )
        self._dist = dist
        # A dedicated gloo group: the model's own group may be NCCL, and the
        # buffers here live on the host.  Whoever creates a group owns it, so
        # a caller-supplied one is never destroyed here.
        self._owns_group = group is None
        self._group = group if group is not None else dist.new_group(backend="gloo")
        # Anything after new_group() must not be able to lose it: the caller
        # only learns the object exists — and only then registers its
        # cleanup — once __init__ returns.
        try:
            self.rank = dist.get_rank(group=self._group)
            self.size = dist.get_world_size(group=self._group)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Destroy the group this object created.  Idempotent.

        A process group is a real resource — file descriptors, a rendezvous
        entry, a background thread — and an embedded component may be built
        and torn down inside a process that keeps running.  Leaving it behind
        is not free the way it is for a training script that is about to exit.
        """
        if self._group is not None and self._owns_group:
            self._dist.destroy_process_group(self._group)
        self._group = None

    def any_true(self, flag: bool) -> bool:
        """A single all_reduce, rather than the base class's all-gather."""
        import torch

        value = torch.tensor([1 if flag else 0], dtype=torch.int64)
        self._dist.all_reduce(
            value, op=self._dist.ReduceOp.MAX, group=self._group
        )
        return bool(value.item())

    def _to_tensor(self, array):
        import torch

        return torch.from_numpy(np.ascontiguousarray(array))

    def alltoall(self, send, send_counts, recv_counts):
        import torch

        send = np.ascontiguousarray(send)
        k = send.shape[1] if send.ndim > 1 else 1
        send = send.reshape(-1, k)
        send_counts = [int(c) for c in send_counts]
        recv_counts = [int(c) for c in recv_counts]

        send_starts = np.concatenate([[0], np.cumsum(send_counts)])
        recv_starts = np.concatenate([[0], np.cumsum(recv_counts)])
        recv = np.empty((int(recv_starts[-1]), k), dtype=send.dtype)
        torch_dtype = torch.from_numpy(np.empty(0, dtype=send.dtype)).dtype

        # Post every receive before any send, so no rank can block on a send
        # whose partner has not reached its matching receive.
        requests = []
        buffers = {}
        for r in range(self.size):
            if r == self.rank or recv_counts[r] == 0:
                continue
            buf = torch.empty((recv_counts[r], k), dtype=torch_dtype)
            buffers[r] = buf
            requests.append(self._dist.irecv(buf, src=r, group=self._group))
        for r in range(self.size):
            if r == self.rank or send_counts[r] == 0:
                continue
            block = send[send_starts[r] : send_starts[r + 1]]
            requests.append(
                self._dist.isend(self._to_tensor(block), dst=r, group=self._group)
            )

        # Our own share never goes on the wire.
        if send_counts[self.rank] > 0:
            recv[recv_starts[self.rank] : recv_starts[self.rank + 1]] = send[
                send_starts[self.rank] : send_starts[self.rank + 1]
            ]

        for request in requests:
            request.wait()
        for r, buf in buffers.items():
            recv[recv_starts[r] : recv_starts[r + 1]] = buf.numpy()
        return recv

    def allgather(self, block):
        import torch

        block = np.ascontiguousarray(block)
        k = block.shape[1] if block.ndim > 1 else 1
        block = block.reshape(-1, k)

        counts = torch.zeros(self.size, dtype=torch.int64)
        counts[self.rank] = block.shape[0]
        self._dist.all_reduce(counts, group=self._group)
        counts = counts.tolist()

        # all_gather wants equal shapes, so pad to the largest block and trim.
        widest = max(counts) if counts else 0
        padded = np.zeros((widest, k), dtype=block.dtype)
        padded[: block.shape[0]] = block
        send = self._to_tensor(padded)
        recv = [torch.empty_like(send) for _ in range(self.size)]
        self._dist.all_gather(recv, send, group=self._group)
        return np.concatenate(
            [recv[r].numpy()[: counts[r]] for r in range(self.size)], axis=0
        )


def run_where(comm: Comm, participating: bool, work):
    """Run ``work()`` on the ranks that own it, and fail everywhere or nowhere.

    The shape this exists for: only some ranks hold the model, so only they
    call it, and an exception there leaves every other rank walking into the
    next collective alone.  A hang is the result, at a point unrelated to the
    cause.  Wrapping the owner-only work makes the failure collective:

    .. code-block:: python

        result = run_where(comm, self.owns_model, lambda: stepper.step(args))
        # every rank is here, or none is
        redistribute(result)

    The whole owner-only section belongs inside ``work``, not just the risky
    line — a conversion, a lookup or a missing output diverges exactly as
    badly as the model call itself.

    Args:
        comm: The communicator every rank shares.
        participating: Whether *this* rank runs the work.
        work: Zero-argument callable, run only where ``participating``.

    Returns:
        What ``work()`` returned, or None on ranks that did not run it.
    """
    problem = ""
    error: BaseException | None = None
    result = None
    if participating:
        try:
            result = work()
        except Exception as exc:  # noqa: BLE001 - re-raised by agree()
            problem = f"rank {comm.rank}: {type(exc).__name__}: {exc}"
            error = exc
    comm.agree(problem, error)
    return result
