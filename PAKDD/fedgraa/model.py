"""From-scratch dense network matching the paper's Keras model exactly.

Architecture (Section 5): input -> Dense(200, tanh) -> Dense(100, tanh)
-> Dense(50, tanh) -> Dense(n_classes, softmax), trained with sparse categorical
cross-entropy. Parameters are stored as a flat list

    [W1, b1, W2, b2, ..., WL, bL]

of NumPy arrays so that aggregation across clients (``P = sum_k w_k P_k``) is a
simple per-array linear combination, which is what FedGRAA differentiates
through.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

Params = List[np.ndarray]


# --------------------------------------------------------------------------- #
# Initialization
# --------------------------------------------------------------------------- #
def glorot_uniform(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """Keras' default kernel initializer (``glorot_uniform``)."""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


def init_params(layer_sizes: Sequence[int], seed: int = 0) -> Params:
    """Create ``[W1, b1, ...]`` with Glorot kernels and zero biases."""
    rng = np.random.default_rng(seed)
    params: Params = []
    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        params.append(glorot_uniform(fan_in, fan_out, rng))
        params.append(np.zeros(fan_out))
    return params


def clone_params(params: Params) -> Params:
    return [p.copy() for p in params]


def aggregate(client_params: Sequence[Params], weights: np.ndarray) -> Params:
    """Weighted sum of client parameter sets: ``P = sum_k w_k P_k`` (Algo 1)."""
    weights = np.asarray(weights, dtype=float)
    if len(client_params) != len(weights):
        raise ValueError("client_params and weights length mismatch")
    n_arrays = len(client_params[0])
    out: Params = []
    for j in range(n_arrays):
        stacked = np.stack([cp[j] for cp in client_params], axis=0)
        out.append(np.tensordot(weights, stacked, axes=(0, 0)))
    return out


# --------------------------------------------------------------------------- #
# Numerics
# --------------------------------------------------------------------------- #
def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def softmax(logits: np.ndarray) -> np.ndarray:
    return np.exp(_log_softmax(logits))


# --------------------------------------------------------------------------- #
# MLP
# --------------------------------------------------------------------------- #
class MLP:
    """Dense tanh/softmax classifier with manual forward/backward passes."""

    def __init__(self, layer_sizes: Sequence[int], seed: int = 0):
        self.layer_sizes = tuple(layer_sizes)
        self.params: Params = init_params(self.layer_sizes, seed=seed)

    @property
    def n_layers(self) -> int:
        return len(self.layer_sizes) - 1

    def set_params(self, params: Params) -> None:
        self.params = clone_params(params)

    # -- forward ----------------------------------------------------------- #
    @staticmethod
    def forward(params: Params, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Return (probabilities, cache) for the given parameters and inputs."""
        activations = [X]
        a = X
        n_layers = len(params) // 2
        for layer in range(n_layers):
            W = params[2 * layer]
            b = params[2 * layer + 1]
            z = a @ W + b
            if layer < n_layers - 1:
                a = np.tanh(z)            # hidden: tanh
            else:
                probs = softmax(z)        # output: softmax
                a = probs
            activations.append(a)
        cache = {"activations": activations, "n_layers": n_layers}
        return activations[-1], cache

    @staticmethod
    def predict(params: Params, X: np.ndarray) -> np.ndarray:
        probs, _ = MLP.forward(params, X)
        return probs.argmax(axis=1)

    # -- loss -------------------------------------------------------------- #
    @staticmethod
    def loss(params: Params, X: np.ndarray, y: np.ndarray) -> float:
        """Mean sparse categorical cross-entropy."""
        n_layers = len(params) // 2
        a = X
        for layer in range(n_layers):
            z = a @ params[2 * layer] + params[2 * layer + 1]
            a = np.tanh(z) if layer < n_layers - 1 else z
        logp = _log_softmax(a)
        y = np.asarray(y, dtype=int)
        return float(-logp[np.arange(len(y)), y].mean())

    # -- backward ---------------------------------------------------------- #
    @staticmethod
    def loss_and_grad(
        params: Params, X: np.ndarray, y: np.ndarray
    ) -> Tuple[float, Params]:
        """Return (loss, gradients) w.r.t. every parameter array.

        Standard back-propagation for tanh hidden layers and a softmax /
        cross-entropy output. ``grad`` has the same structure as ``params``.
        """
        probs, cache = MLP.forward(params, X)
        activations = cache["activations"]
        n_layers = cache["n_layers"]
        y = np.asarray(y, dtype=int)
        N = len(y)

        loss = float(-np.log(probs[np.arange(N), y] + 1e-12).mean())

        grads: Params = [np.zeros_like(p) for p in params]

        # Output layer: dL/dz = (probs - onehot) / N  (softmax + mean CE)
        delta = probs.copy()
        delta[np.arange(N), y] -= 1.0
        delta /= N

        for layer in reversed(range(n_layers)):
            a_prev = activations[layer]
            grads[2 * layer] = a_prev.T @ delta
            grads[2 * layer + 1] = delta.sum(axis=0)
            if layer > 0:
                W = params[2 * layer]
                da_prev = delta @ W.T
                a_cur = activations[layer]          # tanh output of this layer
                delta = da_prev * (1.0 - a_cur ** 2)  # tanh'(z) = 1 - tanh(z)^2
        return loss, grads


# --------------------------------------------------------------------------- #
# Adam optimizer (same update rule as Keras / Kingma & Ba 2014)
# --------------------------------------------------------------------------- #
class Adam:
    """Adam optimizer operating on a list of parameter arrays."""

    def __init__(
        self,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-7,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._m: List[np.ndarray] | None = None
        self._v: List[np.ndarray] | None = None
        self._t = 0

    def reset(self) -> None:
        self._m = None
        self._v = None
        self._t = 0

    def step(self, params: Params, grads: Sequence[np.ndarray]) -> None:
        """In-place Adam update of ``params`` using ``grads``."""
        if self._m is None:
            self._m = [np.zeros_like(p) for p in params]
            self._v = [np.zeros_like(p) for p in params]
        self._t += 1
        lr_t = self.lr * (
            np.sqrt(1.0 - self.beta2 ** self._t) / (1.0 - self.beta1 ** self._t)
        )
        for i, g in enumerate(grads):
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * (g * g)
            params[i] -= lr_t * self._m[i] / (np.sqrt(self._v[i]) + self.epsilon)
