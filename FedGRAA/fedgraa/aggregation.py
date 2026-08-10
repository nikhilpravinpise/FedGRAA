"""Aggregation rules: FedAvg, Static Reparameterization, and FedGRAA.

* :func:`fedavg_weights` — data-size weighting ``|D_k| / |D|`` (eq. 5).
* :func:`static_reparam_weights` — inverse-data-size weighting (eq. 7), also
  used to initialize FedGRAA's weights.
* :class:`FedGRAA` — the Algorithm-2 weight learner that, each round, refines the
  aggregation weights by gradient descent on the composite validation loss
  ``L' = (1-gamma) * min(L_h, L_m) + gamma * L_u`` (eq. 8-10).
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .model import MLP, Adam, Params, aggregate
from .partition import ValidationSplit
from .secagg import SecAggChannel


# --------------------------------------------------------------------------- #
# Static weighting rules
# --------------------------------------------------------------------------- #
def fedavg_weights(client_sizes: Sequence[int]) -> np.ndarray:
    """FedAvg: weight proportional to dataset size (eq. 5)."""
    sizes = np.asarray(client_sizes, dtype=float)
    total = sizes.sum()
    if total == 0:
        raise ValueError("Total dataset size is zero")
    return sizes / total


def static_reparam_weights(client_sizes: Sequence[int]) -> np.ndarray:
    """Static Reparameterization: weight proportional to ``1/|D_k|`` (eq. 7)."""
    sizes = np.asarray(client_sizes, dtype=float)
    if np.any(sizes <= 0):
        raise ValueError("All client sizes must be positive")
    inv = 1.0 / sizes
    return inv / inv.sum()


# --------------------------------------------------------------------------- #
# FedGRAA weight learner (Algorithm 2)
# --------------------------------------------------------------------------- #
def _loss_and_grad_P(
    model: MLP, params: Params, X: np.ndarray, y: np.ndarray
) -> Tuple[float, Params]:
    """Loss and gradient w.r.t. the (aggregated) parameters on a subset.

    Returns ``(inf, zero-grads)`` for an empty subset so that ``min`` and the
    composite loss degrade gracefully.
    """
    if len(y) == 0:
        return float("inf"), [np.zeros_like(p) for p in params]
    return model.loss_and_grad(params, X, y)


def _composite_min_arm(L_h: float, L_m: float) -> Tuple[float, Optional[str]]:
    """NaN-symmetric selection of the ``min(L_h, L_m)`` arm (eq. 8-9).

    Plain ``min()``/``<=`` comparisons against NaN are not symmetric in which
    operand they silently drop, so the same arm-selection logic is shared
    between :meth:`FedGRAA.composite_loss` and
    :meth:`FedGRAA._composite_grad_P` to guarantee they agree on which
    non-finite case they catch. A non-finite value in exactly one arm is
    treated as a numerical blow-up in that arm and excluded in favour of the
    finite one. Returns ``(value, source)`` where ``source`` is ``"h"``,
    ``"m"``, or ``None`` if both arms are non-finite.
    """
    h_ok, m_ok = np.isfinite(L_h), np.isfinite(L_m)
    if h_ok and m_ok:
        return (L_h, "h") if L_h <= L_m else (L_m, "m")
    if h_ok:
        return L_h, "h"
    if m_ok:
        return L_m, "m"
    return float("nan"), None


def _warn_on_nonfinite_arm(
    L_h: float, L_m: float, source: Optional[str], L_u: float, fallback_desc: str
) -> None:
    """Emit a warning that distinguishes an empty split from a blow-up.

    ``source is None`` means both ``L_h`` and ``L_m`` are non-finite, which is
    the genuine "D_V^h and D_V^m both empty" case the original message named.
    A non-``None`` source with only one of the two finite means the *other*
    arm hit a numerical blow-up, not an empty split -- a distinct failure mode
    that the original single warning text conflated.
    """
    if source is None:
        if np.isfinite(L_u):
            warnings.warn(
                "D_V^h and D_V^m are both empty or both non-finite "
                f"(numerical blow-up); {fallback_desc}.",
                RuntimeWarning,
                stacklevel=3,
            )
    elif not (np.isfinite(L_h) and np.isfinite(L_m)):
        bad = "L_m" if source == "h" else "L_h"
        warnings.warn(
            f"{bad} is non-finite (numerical blow-up, not an empty split); "
            "L_a falls back to the finite arm.",
            RuntimeWarning,
            stacklevel=3,
        )


def _grad_wrt_weights(
    grad_P: Sequence[np.ndarray], client_params: Sequence[Params]
) -> np.ndarray:
    """``dL'/dw_k = <grad_P, P_k>`` for every client (Section 2.2 of plan)."""
    grads = np.empty(len(client_params), dtype=float)
    for k, P_k in enumerate(client_params):
        acc = 0.0
        for g_arr, p_arr in zip(grad_P, P_k):
            acc += float(np.sum(g_arr * p_arr))
        grads[k] = acc
    return grads


def project_to_simplex(w: np.ndarray) -> np.ndarray:
    """Euclidean projection of ``w`` onto ``{x : x >= 0, sum(x) = 1}``.

    Standard algorithm (Duchi et al., 2008, "Efficient Projections onto the
    L1-Ball for Learning in High Dimensions", Fig. 1). Promoted here (F1,
    FEDGRAA_V2_MASTER_PLAN.md W0.2) from the diagnostic-only copy in
    ``scripts/diagnose_fedgraa.py``, which now imports this implementation
    instead of maintaining its own.
    """
    w = np.asarray(w, dtype=np.float64)
    n = len(w)
    u = np.sort(w)[::-1]
    css = np.cumsum(u) - 1.0
    idx = np.arange(1, n + 1)
    cond = u - css / idx > 0
    if not np.any(cond):
        return np.full(n, 1.0 / n, dtype=np.float32)
    rho = idx[cond][-1]
    theta = css[cond][-1] / rho
    return np.maximum(w - theta, 0.0).astype(np.float32)


class FedGRAA:
    """Gradient-Refined Adaptive Averaging weight learner (Algorithm 2).

    Holds its own Adam optimizer state for the weight vector ``w`` and the
    personalization coefficient ``gamma``. Call :meth:`update` once per round
    with the current client models and the previous weights.
    """

    def __init__(
        self,
        gamma: float,
        eta_prime: float = 0.1,
        weight_epochs: int = 5,
        layer_sizes: Optional[Sequence[int]] = None,
    ):
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        self.gamma = float(gamma)
        self.eta_prime = float(eta_prime)
        self.weight_epochs = int(weight_epochs)
        self.optimizer = Adam(lr=eta_prime)
        self._model: Optional[MLP] = None
        if layer_sizes is not None:
            self._model = MLP(layer_sizes, seed=0)

    def _get_model(self, params: Params) -> MLP:
        """Return the held MLP, or build one from params if not provided."""
        if self._model is not None:
            return self._model
        from .model import _infer_layer_sizes
        n_features = params[0].shape[0]
        return MLP(_infer_layer_sizes(params, n_features), seed=0)

    def composite_loss(self, params: Params, val: ValidationSplit) -> float:
        """``L' = (1-gamma) * min(L_h, L_m) + gamma * L_u`` (eq. 8-9)."""
        model = self._get_model(params)
        L_h, _ = _loss_and_grad_P(model, params, val.X_h, val.y_h)
        L_m, _ = _loss_and_grad_P(model, params, val.X_m, val.y_m)
        L_u, _ = _loss_and_grad_P(model, params, val.X_n, val.y_n)
        L_a, source = _composite_min_arm(L_h, L_m)
        _warn_on_nonfinite_arm(L_h, L_m, source, L_u, "L_a falls back to 0")
        if source is None:
            L_a = 0.0
        if not np.isfinite(L_u):
            L_u = 0.0
        return (1.0 - self.gamma) * L_a + self.gamma * L_u

    def _composite_grad_P(
        self, params: Params, val: ValidationSplit
    ) -> Tuple[float, Params]:
        """Composite loss and its gradient w.r.t. the aggregated parameters."""
        model = self._get_model(params)
        L_h, g_h = _loss_and_grad_P(model, params, val.X_h, val.y_h)
        L_m, g_m = _loss_and_grad_P(model, params, val.X_m, val.y_m)
        L_u, g_u = _loss_and_grad_P(model, params, val.X_n, val.y_n)

        # Standard sub-gradient of L_a = min(L_h, L_m): gradient of the
        # smaller branch. This is the only reading used for strict reproduction.
        # Uses the same NaN-symmetric arm selection as composite_loss() so the
        # two "parallel" computations of L_a can never disagree on which
        # non-finite case they catch.
        L_a, source = _composite_min_arm(L_h, L_m)
        _warn_on_nonfinite_arm(L_h, L_m, source, L_u, "gradient of L_a zeroed")
        if source == "h":
            g_a = g_h
        elif source == "m":
            g_a = g_m
        else:
            L_a, g_a = 0.0, [np.zeros_like(p) for p in params]
        if not np.isfinite(L_u):
            L_u, g_u = 0.0, [np.zeros_like(p) for p in params]

        one_minus_g = 1.0 - self.gamma
        grad_P = [one_minus_g * a + self.gamma * u for a, u in zip(g_a, g_u)]
        loss = one_minus_g * L_a + self.gamma * L_u
        return loss, grad_P

    def update(
        self,
        client_params: List[Params],
        weights_prev: np.ndarray,
        val: ValidationSplit,
    ) -> np.ndarray:
        """Refine the aggregation weights for ``E'`` epochs (Algorithm 2).

        Implements Algorithm 2 exactly as printed: the weight vector is updated
        by Adam for ``E'`` epochs with the gradient recomputed at the current
        ``w`` each epoch (the literal reading of lines 2-6). No simplex
        projection is applied afterwards, because Algorithm 2 does not contain
        such a step.

        Gap 2.1 fix: Adam state is reset at the start of every round so that
        each call is an independent E'-epoch sub-problem (consistent with the
        natural reading of Algorithm 2 and with the paper's per-round framing).

        Returns the updated weight vector ``w^t``.
        """
        self.optimizer.reset()  # fresh Adam state each round (gap 2.1)
        w = np.array(weights_prev, dtype=np.float32).copy()

        # Iterative reading: recompute gradient at the current w each epoch.
        for _ in range(self.weight_epochs):
            P = aggregate(client_params, w)              # theta'_g = sum_k w_k theta_k
            _, grad_P = self._composite_grad_P(P, val)   # dL'/dP
            grad_w = _grad_wrt_weights(grad_P, client_params)  # dL'/dw
            self.optimizer.step([w], [grad_w])           # Adam step on w (eq. 10)

        return w


class ProjectedFedGRAA(FedGRAA):
    """Non-default, non-strict-reproduction weight learner (F1, W0.2).

    Identical to :class:`FedGRAA` except that the weight vector is
    Euclidean-projected back onto the probability simplex (:func:`project_to_simplex`)
    after every ``E'``-epoch Adam update. Promoted here from the
    diagnostic-only ablation in ``scripts/diagnose_fedgraa.py`` so that
    FedGRAA-v2 (F1) can subclass a first-class, importable, tested variant
    instead of a scratch script.

    **This is not the strict-reproduction default.** ``FederatedTrainer(method=
    "fedgraa")`` and every other strict-reproduction entry point must keep
    constructing the unprojected :class:`FedGRAA` (see
    ``test_strict_reproduction_default_unchanged``, T-19). Only FedGRAA-v2 and
    diagnostic tooling should ever instantiate this class directly.
    """

    def update(
        self,
        client_params: List[Params],
        weights_prev: np.ndarray,
        val: ValidationSplit,
    ) -> np.ndarray:
        w = super().update(client_params, weights_prev, val)
        return project_to_simplex(w)


# --------------------------------------------------------------------------- #
# FedGRAA-v2: the two-phase, count-weighted, SecAgg-mediated exact protocol
# (W2.3a, F12) -- corrects both obstructions the naive "SecAgg-sum of
# per-client gradients" protocol has (measured wrong by 160-557%,
# scripts/verify_v2_exactness.py).
# --------------------------------------------------------------------------- #
def two_phase_composite_grad(
    model: MLP,
    params: Params,
    client_vals: Sequence[ValidationSplit],
    gamma: float,
    n_clients_total: Optional[int] = None,
    threshold: Optional[int] = None,
) -> Tuple[Params, dict]:
    """Reproduce the pooled ``grad_P L'`` exactly from per-client validation
    splits, via SecAgg, without ever exposing an individual client's loss or
    gradient to the server (F4, F12; FEDGRAA_V2_MASTER_PLAN.md §3).

    Each element of ``client_vals`` is one client's own :class:`ValidationSplit`
    -- produced locally by that client applying the server-broadcast strata
    tags (:func:`fedgraa.partition.apply_tags`) to its own rows. Two
    SecAgg-mediated phases follow the plan's corrected pseudocode:

    **Phase 1 (global arm selection, obstruction B).** Each client returns
    three *unnormalized* loss sums (``sum_i loss_i`` over its own h/m/n rows)
    and the three corresponding row counts -- six scalars, SecAgg-summed in
    one channel. The server divides to recover the pooled ``L_h``, ``L_m``
    (mapping a zero count to ``+inf`` *before* dividing, never ``0/0``) and
    applies the **unmodified** :func:`_composite_min_arm` -- the same
    function the pooled (v1) computation uses -- so the arm choice is
    identical to what a server holding the real pooled ``D_V`` would pick.

    **Phase 2 (count-weighted sum-form gradient, obstruction A).** The server
    broadcasts ``arm`` and the two SecAgg-summed counts ``n_arm``, ``n_n``
    alongside it -- safe to reveal (like ``arm`` itself, both are *pooled*
    aggregates the server already learns from phase 1, not any individual
    client's data). Each client can then pre-combine its own contribution
    into a **single** model-shaped array using the *global* normalizers:

    ``c_k = (1-gamma)/n_arm * n_{arm,k} * grad L_{arm,k}(theta')
          +    gamma/n_n   * n_{n,k}   * grad L_{n,k}(theta')``

    SecAgg-summing ``c_k`` over clients gives ``grad_P L'`` exactly in one
    pass (no further division needed): ``sum_k c_k = (1-gamma)/n_arm * sum_k
    n_{arm,k} grad L_{arm,k} + gamma/n_n * sum_k n_{n,k} grad L_{n,k} =
    (1-gamma) * grad_a_pooled + gamma * grad_u_pooled``. This sends **one**
    model-sized array per client per refresh (not two), which is what makes
    the measured bytes match FEDGRAA_V2_MASTER_PLAN.md §4.1's
    ``(1 + ceil(E'/m)) x baseline`` formula exactly (T-12) -- a naive
    per-stratum-then-combine variant that instead returns two separate
    arrays (``n_{arm,k} grad L_{arm,k}`` and ``n_{n,k} grad L_{n,k}``,
    combined server-side) is equally exact but costs ~1.5x more bytes per
    refresh, which is *not* what the plan's byte table describes.

    Returns ``(grad_P, stats)``; ``stats`` records the per-call communication
    shape (T-12: bytes and *rounds* must be billed separately) for the caller
    (:class:`fedgraa.trainer.FederatedTrainerV2`) to instrument.
    """
    n = len(client_vals)
    if n == 0:
        raise ValueError("two_phase_composite_grad requires at least one client")
    n_clients_total = n_clients_total if n_clients_total is not None else n
    threshold = threshold if threshold is not None else n_clients_total

    # ---- Phase 1: unnormalized scalar sums -> pooled L_h/L_m -> global arm --
    ch1 = SecAggChannel(n_clients=n_clients_total, threshold=threshold)
    for v in client_vals:
        s_h = float(model.loss(params, v.X_h, v.y_h)) * len(v.y_h) if len(v.y_h) else 0.0
        s_m = float(model.loss(params, v.X_m, v.y_m)) * len(v.y_m) if len(v.y_m) else 0.0
        s_n = float(model.loss(params, v.X_n, v.y_n)) * len(v.y_n) if len(v.y_n) else 0.0
        ch1.submit([
            np.array(s_h), np.array(s_m), np.array(s_n),
            np.array(len(v.y_h)), np.array(len(v.y_m)), np.array(len(v.y_n)),
        ])
    s_h_tot, s_m_tot, s_n_tot, n_h_arr, n_m_arr, n_n_arr = ch1.aggregate()
    n_h, n_m, n_n = int(n_h_arr), int(n_m_arr), int(n_n_arr)
    L_h = float(s_h_tot) / n_h if n_h > 0 else float("inf")
    L_m = float(s_m_tot) / n_m if n_m > 0 else float("inf")
    _, arm = _composite_min_arm(L_h, L_m)
    n_arm = n_h if arm == "h" else n_m if arm == "m" else 0

    # ---- Phase 2: count-weighted, client-pre-combined, single-array sum ----
    zero_like = [np.zeros_like(p) for p in params]
    coef_arm = (1.0 - gamma) / n_arm if n_arm > 0 else 0.0
    coef_u = gamma / n_n if n_n > 0 else 0.0
    ch2 = SecAggChannel(n_clients=n_clients_total, threshold=threshold)
    for v in client_vals:
        contribution = list(zero_like)
        if arm == "h":
            Xs, ys = v.X_h, v.y_h
        elif arm == "m":
            Xs, ys = v.X_m, v.y_m
        else:
            Xs, ys = v.X_h[:0], v.y_h[:0]
        if coef_arm != 0.0 and len(ys):
            _, g = _loss_and_grad_P(model, params, Xs, ys)
            scale = coef_arm * len(ys)
            contribution = [c + scale * gi for c, gi in zip(contribution, g)]
        if coef_u != 0.0 and len(v.y_n):
            _, g = _loss_and_grad_P(model, params, v.X_n, v.y_n)
            scale = coef_u * len(v.y_n)
            contribution = [c + scale * gi for c, gi in zip(contribution, g)]
        ch2.submit(contribution)

    grad_P = ch2.aggregate()  # already grad_P L' exactly -- no further division

    stats = {
        "n_h": n_h, "n_m": n_m, "n_n": n_n, "arm": arm,
        "n_participating": n,
        "n_clients_total": n_clients_total,
        "phase1_scalars_per_client": 6,
        "phase1_broadcast_scalars": 3,   # arm (1 byte) + n_arm + n_n
        "phase2_arrays_per_client": len(params),
        "rounds": 2,
    }
    return grad_P, stats


class FedGRAAv2(ProjectedFedGRAA):
    """FedGRAA-v2 weight learner (W2.3, F5).

    Identical to :class:`ProjectedFedGRAA` -- Adam on ``w`` for ``E'`` epochs,
    then a simplex projection -- except that ``grad_P L'`` is supplied by an
    externally-injected callback rather than computed from a pooled
    :class:`~fedgraa.partition.ValidationSplit`, because in the federated
    setting no such pooled split exists on the server; F12's whole point is
    that ``grad_P L'`` must instead be reconstructed via
    :func:`two_phase_composite_grad`. Keeping this class protocol-agnostic
    (it only calls ``grad_provider(theta_prime_g) -> grad_P``) lets the
    refresh-interval logic below be unit-tested against a bare fake callback,
    independent of SecAgg/data wiring (T-4a/T-4b test the callback's
    correctness separately).

    **Refresh interval `m` (F5, `grad_refresh_every`).** ``grad_provider`` is
    invoked only every ``m`` inner epochs (``e % m == 0``); the Adam step on
    ``w`` still runs every epoch, reusing the last-refreshed gradient in
    between. The number of ``grad_provider`` calls this method makes per round
    is exactly ``ceil(weight_epochs / m)`` -- the quantity T-12 checks against
    the ``(1 + ceil(E'/m)) x baseline`` byte formula.
    """

    def __init__(
        self,
        gamma: float,
        eta_prime: float = 0.1,
        weight_epochs: int = 5,
        layer_sizes: Optional[Sequence[int]] = None,
        grad_refresh_every: int = 1,
    ):
        super().__init__(
            gamma=gamma, eta_prime=eta_prime,
            weight_epochs=weight_epochs, layer_sizes=layer_sizes,
        )
        if grad_refresh_every < 1:
            raise ValueError(
                f"grad_refresh_every must be >= 1, got {grad_refresh_every}"
            )
        self.grad_refresh_every = int(grad_refresh_every)
        self.last_n_refreshes = 0  # T-12 instrumentation: refreshes in the last update() call

    def update(  # type: ignore[override]
        self,
        client_params: List[Params],
        weights_prev: np.ndarray,
        grad_provider,
    ) -> np.ndarray:
        """``grad_provider(theta_prime_g: Params) -> Params`` returns
        ``grad_P L'`` only (the loss/stats a real caller may also want are the
        caller's own responsibility to record, e.g. via ``two_phase_composite_grad``'s
        ``stats`` return, which this method never sees)."""
        self.optimizer.reset()
        w = np.array(weights_prev, dtype=np.float32).copy()

        grad_P: Optional[Params] = None
        n_refreshes = 0
        for e in range(self.weight_epochs):
            if e % self.grad_refresh_every == 0:
                P = aggregate(client_params, w)
                grad_P = grad_provider(P)
                n_refreshes += 1
            assert grad_P is not None
            grad_w = _grad_wrt_weights(grad_P, client_params)
            self.optimizer.step([w], [grad_w])

        self.last_n_refreshes = n_refreshes
        return project_to_simplex(w)
