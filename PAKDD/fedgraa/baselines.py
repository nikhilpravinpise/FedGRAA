"""Additional federated learning baselines mentioned in the paper.

* :class:`FedProxTrainer` — adds a proximal term to local training [5].
* :class:`FedOptServer` — server-side adaptive optimizer over the aggregated
  client update [7].
* :class:`FedSNDTrainer` — distribution-aware static reweighting baseline [12].

These are provided so the reproduction can compare FedGRAA against the
methods discussed in Section 2 (Related Work), addressing gap m-3.  They are
not part of the base paper's Figures 1-3.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import tensorflow as tf

from .config import FedConfig
from .metrics import accuracy, last_n_class_accuracy
from .model import (
    MLP,
    Adam,
    Params,
    _fused_client_step,
    _fused_proximal_step,
    aggregate,
    build_tf_optimizer,
    init_params,
)
from .partition import ClientShard, FederatedSplit
from .trainer import TrainHistory, client_train_round


# --------------------------------------------------------------------------- #
# FedProx: local proximal term
# --------------------------------------------------------------------------- #
def proximal_client_train_round(
    model: MLP,
    opt: tf.keras.optimizers.Optimizer,
    X: np.ndarray,
    y: np.ndarray,
    cfg: FedConfig,
    global_weights: Params,
    rng: np.random.Generator,
    mu: float = 0.01,
) -> Params:
    """Fused XLA client training with FedProx proximal term.

    One ``set_weights`` per round, then all iterations run inside
    ``_fused_proximal_step`` which adds ``mu * (v - v_global)`` to each gradient.
    """
    model.set_params(global_weights)
    n = len(y)
    if n == 0:
        return model.get_params()

    global_vars = [tf.constant(g, dtype=tf.float32) for g in global_weights]
    mu_tensor = tf.constant(mu, dtype=tf.float32)

    pad_x = np.zeros((cfg.batch_size, X.shape[1]), dtype=np.float32)
    pad_y = np.zeros(cfg.batch_size, dtype=np.int32)

    all_perm = np.array(
        [rng.permutation(n) for _ in range(cfg.local_epochs)],
        dtype=np.int32,
    )
    for epoch in range(cfg.local_epochs):
        perm = all_perm[epoch]
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            n_valid = len(idx)
            if n_valid < cfg.batch_size:
                pad_x[:n_valid] = X[idx]
                pad_y[:n_valid] = y[idx]
                x_batch, y_batch = pad_x, pad_y
            else:
                x_batch, y_batch = X[idx], y[idx]
            _fused_proximal_step(
                tf.constant(x_batch, dtype=tf.float32),
                tf.constant(y_batch, dtype=tf.int32),
                model._model, opt,
                global_vars, mu_tensor,
                tf.constant(n_valid, dtype=tf.int32),
            )
    return model.get_params()


class FedProxTrainer:
    """FedProx baseline: FedAvg aggregation + proximal local training."""

    def __init__(
        self,
        split: FederatedSplit,
        cfg: FedConfig,
        mu: float = 0.01,
    ):
        self.split = split
        self.cfg = cfg
        self.mu = mu
        self.spec = split.spec
        self.rare_class_ids = self.spec.rare_client_ids(cfg.n_rare)
        self.client_sizes = [c.n_train for c in split.clients]
        layer_sizes = self.spec.model_config().layer_sizes
        self._model = MLP(layer_sizes, seed=cfg.seed)
        self._client_tf_optimizers: Dict[int, tf.keras.optimizers.Adam] = {
            c.client_id: build_tf_optimizer(cfg.eta,
                                            self._model._model.trainable_variables)
            for c in split.clients
        }

    def train(
        self,
        eval_every: int = 20,
        verbose: bool = False,
    ) -> TrainHistory:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        global_params: Params = init_params(self._model.layer_sizes, seed=cfg.seed)

        from .aggregation import fedavg_weights

        w = fedavg_weights(self.client_sizes)
        history = TrainHistory(
            method="fedprox",
            gamma=None,
            n_rare=cfg.n_rare,
        )
        history.record(
            0,
            accuracy(self._model, global_params, self.split.X_val, self.split.y_val),
            last_n_class_accuracy(
                self._model,
                global_params,
                self.split.X_val,
                self.split.y_val,
                self.rare_class_ids,
            ),
            w,
        )

        for rnd in range(1, cfg.rounds + 1):
            client_params = [
                proximal_client_train_round(
                    self._model,
                    self._client_tf_optimizers[client.client_id],
                    client.X_train, client.y_train,
                    cfg, global_params, rng, self.mu,
                )
                for client in self.split.clients
            ]
            w = fedavg_weights(self.client_sizes)
            global_params = aggregate(client_params, w)

            if rnd % eval_every == 0 or rnd == cfg.rounds:
                all_a = accuracy(self._model, global_params, self.split.X_val, self.split.y_val)
                last_a = last_n_class_accuracy(
                    self._model,
                    global_params,
                    self.split.X_val,
                    self.split.y_val,
                    self.rare_class_ids,
                )
                history.record(rnd, all_a, last_a, w)
                if verbose:
                    print(
                        f"  [fedprox mu={self.mu}] round {rnd:3d}/{cfg.rounds}"
                        f"  all={all_a:.3f}  last-n={last_a:.3f}",
                        flush=True,
                    )

        self.final_params = global_params
        return history


# --------------------------------------------------------------------------- #
# FedOpt: server-side adaptive optimizer
# --------------------------------------------------------------------------- #
class FedOptServer:
    """FedOpt baseline: server runs Adam on the pseudo-gradient (theta_g - delta).

    Following Reddi et al. [7], the server maintains a global model and updates
    it with an adaptive optimizer using the average client update as a
    pseudo-gradient.
    """

    def __init__(
        self,
        split: FederatedSplit,
        cfg: FedConfig,
        server_lr: float = 0.01,
        tau: float = 1e-3,
    ):
        self.split = split
        self.cfg = cfg
        self.server_lr = server_lr
        self.tau = tau
        self.spec = split.spec
        self.rare_class_ids = self.spec.rare_client_ids(cfg.n_rare)
        self.client_sizes = [c.n_train for c in split.clients]
        layer_sizes = self.spec.model_config().layer_sizes
        self._model = MLP(layer_sizes, seed=cfg.seed)
        self._client_tf_optimizers: Dict[int, tf.keras.optimizers.Adam] = {
            c.client_id: build_tf_optimizer(cfg.eta,
                                            self._model._model.trainable_variables)
            for c in split.clients
        }
        # Server optimizer operates on numpy params — Adam wrapper is fine.
        self.server_optimizer = Adam(lr=server_lr)

    def train(
        self,
        eval_every: int = 20,
        verbose: bool = False,
    ) -> TrainHistory:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        global_params: Params = init_params(self._model.layer_sizes, seed=cfg.seed)

        from .aggregation import fedavg_weights

        w = fedavg_weights(self.client_sizes)
        history = TrainHistory(method="fedopt", gamma=None, n_rare=cfg.n_rare)
        history.record(
            0,
            accuracy(self._model, global_params, self.split.X_val, self.split.y_val),
            last_n_class_accuracy(
                self._model,
                global_params,
                self.split.X_val,
                self.split.y_val,
                self.rare_class_ids,
            ),
            w,
        )

        for rnd in range(1, cfg.rounds + 1):
            client_params = [
                client_train_round(
                    self._model,
                    self._client_tf_optimizers[client.client_id],
                    client.X_train, client.y_train,
                    cfg, global_params, rng,
                )
                for client in self.split.clients
            ]
            # Pseudo-gradient: global - weighted average client update.
            avg_update = aggregate(client_params, w)
            pseudo_grad = [gp - up for gp, up in zip(global_params, avg_update)]
            # Server Adam step: params -= server_lr * m / (sqrt(v) + tau)
            self.server_optimizer.step(global_params, pseudo_grad)

            if rnd % eval_every == 0 or rnd == cfg.rounds:
                all_a = accuracy(self._model, global_params, self.split.X_val, self.split.y_val)
                last_a = last_n_class_accuracy(
                    self._model,
                    global_params,
                    self.split.X_val,
                    self.split.y_val,
                    self.rare_class_ids,
                )
                history.record(rnd, all_a, last_a, w)
                if verbose:
                    print(
                        f"  [fedopt] round {rnd:3d}/{cfg.rounds}"
                        f"  all={all_a:.3f}  last-n={last_a:.3f}",
                        flush=True,
                    )

        self.final_params = global_params
        return history


# --------------------------------------------------------------------------- #
# FedSND: distribution-aware static reweighting
# --------------------------------------------------------------------------- #
class FedSNDTrainer:
    """FedSND baseline: simple distribution-aware static aggregation weights.

    Note: this is Related Work (Section 2) and is not part of the base paper's
    Figures 1-3 reproduction.
    """

    def __init__(self, split: FederatedSplit, cfg: FedConfig):
        self.split = split
        self.cfg = cfg
        self.spec = split.spec
        self.rare_class_ids = self.spec.rare_client_ids(cfg.n_rare)
        self.client_sizes = [c.n_train for c in split.clients]
        layer_sizes = self.spec.model_config().layer_sizes
        self._model = MLP(layer_sizes, seed=cfg.seed)
        self._client_tf_optimizers: Dict[int, tf.keras.optimizers.Adam] = {
            c.client_id: build_tf_optimizer(cfg.eta,
                                            self._model._model.trainable_variables)
            for c in split.clients
        }

    def train(
        self,
        eval_every: int = 20,
        verbose: bool = False,
    ) -> TrainHistory:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        global_params: Params = init_params(self._model.layer_sizes, seed=cfg.seed)

        from .aggregation import static_reparam_weights
        w = static_reparam_weights(self.client_sizes)
        history = TrainHistory(method="fedsnd", gamma=None, n_rare=cfg.n_rare)
        history.record(
            0,
            accuracy(self._model, global_params, self.split.X_val, self.split.y_val),
            last_n_class_accuracy(
                self._model,
                global_params,
                self.split.X_val,
                self.split.y_val,
                self.rare_class_ids,
            ),
            w,
        )

        for rnd in range(1, cfg.rounds + 1):
            client_params = [
                client_train_round(
                    self._model,
                    self._client_tf_optimizers[client.client_id],
                    client.X_train, client.y_train,
                    cfg, global_params, rng,
                )
                for client in self.split.clients
            ]
            w = static_reparam_weights(self.client_sizes)
            global_params = aggregate(client_params, w)

            if rnd % eval_every == 0 or rnd == cfg.rounds:
                all_a = accuracy(self._model, global_params, self.split.X_val, self.split.y_val)
                last_a = last_n_class_accuracy(
                    self._model,
                    global_params,
                    self.split.X_val,
                    self.split.y_val,
                    self.rare_class_ids,
                )
                history.record(rnd, all_a, last_a, w)
                if verbose:
                    print(
                        f"  [fedsnd] round {rnd:3d}/{cfg.rounds}"
                        f"  all={all_a:.3f}  last-n={last_a:.3f}",
                        flush=True,
                    )

        self.final_params = global_params
        return history
