"""Keras Sequential backend matching the paper's network exactly.

Architecture (Section 5): input -> Dense(200, tanh) -> Dense(100, tanh)
-> Dense(50, tanh) -> Dense(n_classes, softmax), trained with sparse categorical
cross-entropy. Parameters are stored as a flat list

    [W1, b1, W2, b2, ..., WL, bL]

of float32 NumPy arrays so that aggregation across clients
(``P = sum_k w_k P_k``) is a simple per-array linear combination.

The paper explicitly describes the model as a Keras Sequential network; this
module is the only implementation used by the reproduction pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

import tensorflow as tf

Params = List[np.ndarray]


def _set_seed(seed: int) -> None:
    """Set TensorFlow/Keras random seeds for reproducibility."""
    tf.random.set_seed(seed)


def _infer_layer_sizes(params: Sequence[np.ndarray], n_features: int) -> Tuple[int, ...]:
    """Reconstruct ``(n_features, ..., n_classes)`` from a parameter list."""
    sizes = [n_features]
    for i in range(0, len(params), 2):
        sizes.append(int(params[i].shape[1]))
    return tuple(sizes)


# --------------------------------------------------------------------------- #
# Initialization
# --------------------------------------------------------------------------- #
def init_params(layer_sizes: Sequence[int], seed: int = 0) -> Params:
    """Create ``[W1, b1, ...]`` with Keras Glorot-uniform kernels and zero biases."""
    _set_seed(seed)
    params: Params = []
    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        W = tf.keras.initializers.GlorotUniform(seed=seed)(
            shape=(fan_in, fan_out), dtype="float32"
        ).numpy()
        b = np.zeros(fan_out, dtype=np.float32)
        params.append(W)
        params.append(b)
    return params


def clone_params(params: Params) -> Params:
    return [p.copy() for p in params]


def aggregate(client_params: Sequence[Params], weights: np.ndarray) -> Params:
    """Weighted sum of client parameter sets: ``P = sum_k w_k P_k`` (Algo 1)."""
    weights = np.asarray(weights, dtype=np.float32)
    if len(client_params) != len(weights):
        raise ValueError("client_params and weights length mismatch")
    n_arrays = len(client_params[0])
    out: Params = []
    for j in range(n_arrays):
        stacked = np.stack([cp[j].astype(np.float32) for cp in client_params], axis=0)
        out.append(np.tensordot(weights, stacked, axes=(0, 0)))
    return out


# --------------------------------------------------------------------------- #
# MLP
# --------------------------------------------------------------------------- #
class MLP:
    """Dense tanh/softmax classifier backed by a Keras Sequential model."""

    def __init__(self, layer_sizes: Sequence[int], seed: int = 0):
        self.layer_sizes = tuple(layer_sizes)
        self.seed = seed
        self._model = self._build_model()
        self._initialize_weights(seed)

    def _build_model(self) -> tf.keras.Model:
        """Build the Keras Sequential model described in the paper."""
        layers: List[tf.keras.layers.Layer] = [
            tf.keras.layers.Input(shape=(self.layer_sizes[0],), dtype="float32")
        ]
        n_layers = len(self.layer_sizes) - 1
        for i, fan_out in enumerate(self.layer_sizes[1:], start=1):
            layers.append(
                tf.keras.layers.Dense(
                    fan_out,
                    activation="tanh" if i < n_layers else None,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(
                        seed=self.seed
                    ),
                    dtype="float32",
                )
            )
        model = tf.keras.Sequential(layers)
        model.build(input_shape=(None, self.layer_sizes[0]))
        return model

    def _initialize_weights(self, seed: int) -> None:
        """Force Keras Glorot-uniform initialization (deterministic with seed)."""
        _set_seed(seed)
        self._model.set_weights(init_params(self.layer_sizes, seed=seed))

    @property
    def n_layers(self) -> int:
        return len(self.layer_sizes) - 1

    def get_params(self) -> Params:
        """Return current Keras weights as ``[W1, b1, ...]``."""
        return [
            w.numpy().astype(np.float32)
            if hasattr(w, "numpy")
            else np.asarray(w, dtype=np.float32)
            for w in self._model.get_weights()
        ]

    def set_params(self, params: Params) -> None:
        """Load parameters into the Keras model."""
        self._model.set_weights([np.asarray(p, dtype=np.float32) for p in params])

    # -- forward ----------------------------------------------------------- #
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Return (probabilities, cache) for the given inputs.

        The cache is kept for API compatibility with the old NumPy backend;
        downstream code should rely on ``final_logits`` and ``activations``.
        """
        X32 = np.asarray(X, dtype=np.float32)
        logits = self._model(X32, training=False)
        probs = tf.nn.softmax(logits).numpy()
        cache = {
            "final_logits": logits.numpy(),
            "activations": [X32, probs],  # minimal: input + output
            "n_layers": self.n_layers,
        }
        return probs, cache

    def predict(self, params: Params, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        self.set_params(params)
        X32 = np.asarray(X, dtype=np.float32)
        logits = self._model(X32, training=False)
        return tf.argmax(logits, axis=1).numpy().astype(int)

    # -- loss -------------------------------------------------------------- #
    def loss(self, params: Params, X: np.ndarray, y: np.ndarray) -> float:
        """Mean sparse categorical cross-entropy."""
        self.set_params(params)
        X32 = tf.convert_to_tensor(X, dtype=tf.float32)
        y32 = tf.convert_to_tensor(y, dtype=tf.int32)
        logits = self._model(X32, training=False)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y32, logits=logits)
        )
        return float(loss.numpy())

    # -- backward ---------------------------------------------------------- #
    def loss_and_grad(
        self, params: Params, X: np.ndarray, y: np.ndarray
    ) -> Tuple[float, Params]:
        """Return (loss, gradients) w.r.t. every parameter array."""
        self.set_params(params)
        X32 = tf.convert_to_tensor(X, dtype=tf.float32)
        y32 = tf.convert_to_tensor(y, dtype=tf.int32)

        with tf.GradientTape() as tape:
            logits = self._model(X32, training=True)
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=y32, logits=logits
                )
            )

        grads = tape.gradient(loss, self._model.trainable_variables)
        grad_params = [g.numpy().astype(np.float32) for g in grads]
        return float(loss.numpy()), grad_params


# --------------------------------------------------------------------------- #
# Adam optimizer (Keras implementation)
# --------------------------------------------------------------------------- #
class Adam:
    """Adam optimizer operating on a list of parameter arrays.

    Wraps ``tf.keras.optimizers.Adam`` and maintains persistent variables so
    that optimizer state (first/second moments) survives across ``step`` calls.
    This matches the paper's Adam optimizer with default Keras hyperparameters.
    """

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
        self._opt = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=beta1, beta_2=beta2, epsilon=epsilon
        )
        self._vars: List[tf.Variable] | None = None
        self._t = 0

    def reset(self) -> None:
        """Reset Adam state (moments and step counter)."""
        self._opt = tf.keras.optimizers.Adam(
            learning_rate=self.lr,
            beta_1=self.beta1,
            beta_2=self.beta2,
            epsilon=self.epsilon,
        )
        self._vars = None
        self._t = 0

    def step(self, params: Params, grads: Sequence[np.ndarray]) -> None:
        """In-place Adam update of ``params`` using ``grads``."""
        if self._vars is None:
            self._vars = [
                tf.Variable(np.asarray(p, dtype=np.float32), trainable=True)
                for p in params
            ]
            self._opt.build(self._vars)

        for v, p in zip(self._vars, params):
            v.assign(np.asarray(p, dtype=np.float32))

        grad_tensors = [
            tf.convert_to_tensor(np.asarray(g, dtype=np.float32), dtype=tf.float32)
            for g in grads
        ]
        self._opt.apply_gradients(zip(grad_tensors, self._vars))
        self._t += 1

        for p, v in zip(params, self._vars):
            p[...] = v.numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# Fused XLA training step (Phase 2: eliminate numpy↔TF ping-pong)
# --------------------------------------------------------------------------- #
@tf.function(jit_compile=True, reduce_retracing=True)
def _fused_client_step(
    x: tf.Tensor, y: tf.Tensor, model: tf.keras.Model,
    opt: tf.keras.optimizers.Optimizer,
    n_valid: tf.Tensor,
) -> tf.Tensor:
    """One step: forward + backward + optimizer. All XLA-fused, no numpy conversion.

    ``x`` and ``y`` are **always** padded to the full trace-time batch shape.
    ``n_valid`` (scalar tensor) indicates how many of the leading samples are
    real; the remainder are padding with zero gradient contribution via masking.
    This ensures fixed-shape tensors and eliminates ``tf.function`` retracing
    across epochs and clients.
    """
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits
        )
        batch_size = tf.shape(x)[0]
        mask = tf.cast(tf.range(batch_size) < n_valid, dtype=losses.dtype)
        loss = tf.reduce_sum(losses * mask) / tf.cast(n_valid, dtype=losses.dtype)
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss


@tf.function(jit_compile=True, reduce_retracing=True)
def _fused_proximal_step(
    x: tf.Tensor, y: tf.Tensor, model: tf.keras.Model,
    opt: tf.keras.optimizers.Optimizer,
    global_vars: Sequence[tf.Tensor], mu: tf.Tensor,
    n_valid: tf.Tensor,
) -> tf.Tensor:
    """Fused step with FedProx proximal term: ``mu/2 * ||theta - theta_g||^2``."""
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits
        )
        batch_size = tf.shape(x)[0]
        mask = tf.cast(tf.range(batch_size) < n_valid, dtype=losses.dtype)
        loss = tf.reduce_sum(losses * mask) / tf.cast(n_valid, dtype=losses.dtype)
    grads = tape.gradient(loss, model.trainable_variables)
    prox_grads = [
        g + mu * (v - gv)
        for g, v, gv in zip(grads, model.trainable_variables, global_vars)
    ]
    opt.apply_gradients(zip(prox_grads, model.trainable_variables))
    return loss


def build_tf_optimizer(
    lr: float,
    variables: Sequence[tf.Variable],
) -> tf.keras.optimizers.Adam:
    """Create and pre-build a TF Adam optimizer for a set of variables.

    Pre-building ensures the optimizer's slot variables (momentum, velocity)
    exist before ``@tf.function(jit_compile=True)`` tracing, preventing
    re-triggered compilation.
    """
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    opt.apply_gradients([(tf.zeros_like(v), v) for v in variables])
    opt.iterations.assign(0)
    return opt
