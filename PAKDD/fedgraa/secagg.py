"""Secure-aggregation channel (W2.1, FEDGRAA_V2_MASTER_PLAN.md F4).

Models the *ideal-functionality* abstraction of secure aggregation (Bonawitz
et al., "Practical Secure Aggregation for Privacy-Preserving Machine
Learning," CCS 2017): each client submits a contribution, and the only thing
the server can ever learn is the **sum** (or mean) over all contributions
that survived to the end of the round. This module does not implement the
cryptographic protocol itself (Shamir secret sharing + pairwise masking) --
it implements the interface contract FedGRAA-v2's protocol needs, with the
security property that matters for F4 enforced *structurally*: an individual
client's contribution is folded into a running total the instant it is
submitted and is never retained anywhere, under any name, on the channel
object. There is therefore no method -- public, private, or reachable via
introspection -- that can return what any one client contributed.

Why this is safe to model this way (F4). ``fedgraa/aggregation.py``'s
``_grad_wrt_weights`` needs only the pooled sum -- never an individual
client's gradient, loss, or histogram -- so a channel that can only ever
produce a sum is not a lossy stand-in for a richer real channel; it is
exactly what the consuming algorithm needs and no more.

Dropout (T-7, T-13). If fewer than the reconstruction threshold ``t`` survive
a round, the real Shamir-secret-sharing scheme cannot remove the pairwise
masks and the sum is not recoverable at all -- modeled here as
:class:`SecAggDropoutError`, raised by :meth:`SecAggChannel.aggregate` rather
than silently returning a wrong (under-counted) answer.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

__all__ = ["SecAggChannel", "SecAggDropoutError"]


class SecAggDropoutError(RuntimeError):
    """Raised when fewer clients survived a round than the Shamir
    reconstruction threshold requires -- the sum is not recoverable at all,
    not merely computed over a smaller population."""


class SecAggChannel:
    """One round of secure aggregation over a fixed-shape list of arrays.

    Usage::

        ch = SecAggChannel(n_clients=14, threshold=7)
        for k in participating_clients:
            ch.submit(client_k_contribution)   # e.g. [n_h, n_m, n_n] as 0-d arrays
        total = ch.aggregate()                 # sum, or raises SecAggDropoutError
        mean  = ch.aggregate_mean()             # sum / n_submitted

    Each element of ``contribution`` may be any shape (a scalar wrapped as a
    0-d array, a full gradient array, a length-``C`` histogram, ...); the
    channel only ever needs elementwise ``+=``, so it is shape-agnostic.

    **Structural guarantee (F4, T-6).** After construction, this object's
    only stateful attributes are a running elementwise total, a submission
    counter, and immutable round parameters -- there is no attribute, method,
    or property anywhere on this class that returns what an individual
    ``submit()`` call passed in. Once :meth:`submit` returns, the caller's
    ``contribution`` argument is not referenced by ``self`` in any form.
    """

    def __init__(self, n_clients: int, threshold: int):
        if n_clients < 1:
            raise ValueError(f"n_clients must be >= 1, got {n_clients}")
        if not 1 <= threshold <= n_clients:
            raise ValueError(
                f"threshold must be in [1, {n_clients}], got {threshold}"
            )
        self._n_clients = n_clients
        self._threshold = threshold
        self._total: List[np.ndarray] | None = None
        self._n_submitted = 0
        self._closed = False

    @property
    def n_clients(self) -> int:
        return self._n_clients

    @property
    def threshold(self) -> int:
        return self._threshold

    @property
    def n_submitted(self) -> int:
        """How many clients have submitted so far -- a count, never an identity
        or a value; this is the only per-round progress information exposed."""
        return self._n_submitted

    def submit(self, contribution: Sequence[np.ndarray]) -> None:
        """Fold one client's contribution into the running total.

        Raises if called after :meth:`aggregate`/:meth:`aggregate_mean` has
        already closed the round -- each channel instance is single-use
        (mirrors a real SecAgg round, which cannot accept late submissions
        after reconstruction).
        """
        if self._closed:
            raise RuntimeError(
                "this SecAggChannel has already aggregated its round; "
                "construct a new channel for the next round"
            )
        if self._n_submitted >= self._n_clients:
            raise RuntimeError(
                f"all {self._n_clients} expected clients have already submitted"
            )
        arrs = [np.asarray(a) for a in contribution]
        if self._total is None:
            self._total = [np.zeros_like(a) for a in arrs]
        elif len(arrs) != len(self._total):
            raise ValueError(
                f"contribution has {len(arrs)} arrays, expected {len(self._total)}"
            )
        for i, a in enumerate(arrs):
            self._total[i] = self._total[i] + a
        self._n_submitted += 1
        # ``arrs``/``contribution`` fall out of scope here -- nothing above
        # retains a reference to the per-client value itself, only the
        # elementwise running total and a plain integer counter.

    def aggregate(self) -> List[np.ndarray]:
        """The sum over every contribution submitted this round.

        Raises :class:`SecAggDropoutError` if fewer than ``threshold`` clients
        submitted -- the real protocol cannot reconstruct the sum at all in
        that case, so this never silently returns a partial/wrong total.
        Raises ``RuntimeError`` if this round was already aggregated (each
        channel instance is single-use, matching :meth:`submit`'s own
        already-closed check).
        """
        if self._closed:
            raise RuntimeError(
                "this SecAggChannel has already aggregated its round; "
                "construct a new channel for the next round"
            )
        if self._n_submitted < self._threshold:
            raise SecAggDropoutError(
                f"only {self._n_submitted}/{self._n_clients} clients submitted "
                f"this round; Shamir reconstruction threshold is "
                f"{self._threshold} -- the sum cannot be recovered"
            )
        self._closed = True
        assert self._total is not None
        return [t.copy() for t in self._total]

    def aggregate_mean(self) -> List[np.ndarray]:
        """``aggregate() / n_submitted`` -- the renormalized mean over
        surviving clients (T-7's "valid renormalized vector when survivors
        >= threshold")."""
        total = self.aggregate()
        n = self._n_submitted
        return [t / n for t in total]
