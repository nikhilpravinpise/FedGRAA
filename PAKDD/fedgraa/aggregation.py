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
        L_a = min(L_h, L_m)
        # Both D_V^h and D_V^m empty — warn and fall back to L_u only.
        if not np.isfinite(L_a):
            if np.isfinite(L_u):
                warnings.warn(
                    "D_V^h and D_V^m are both empty; L_a falls back to 0. "
                    "The (1-gamma)*L_a term is zeroed regardless of gamma.",
                    RuntimeWarning,
                    stacklevel=2,
                )
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
        if L_h <= L_m:
            L_a, g_a = L_h, g_h
        else:
            L_a, g_a = L_m, g_m

        # Both empty — warn and zero out (same as composite_loss).
        if not np.isfinite(L_a):
            if np.isfinite(L_u):
                warnings.warn(
                    "D_V^h and D_V^m are both empty; gradient of L_a zeroed.",
                    RuntimeWarning,
                    stacklevel=2,
                )
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
