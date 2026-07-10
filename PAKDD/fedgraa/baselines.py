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

from typing import Dict, List

import numpy as np

from .config import FedConfig
from .metrics import accuracy, last_n_class_accuracy
from .model import Adam, MLP, Params, aggregate, init_params
from .partition import ClientShard, FederatedSplit
from .trainer import TrainHistory, _client_update


# --------------------------------------------------------------------------- #
# FedProx: local proximal term
# --------------------------------------------------------------------------- #
def _proximal_client_update(
    global_params: Params,
    client: ClientShard,
    cfg: FedConfig,
    rng: np.random.Generator,
    model: MLP,
    optimizer: Adam,
    mu: float = 0.01,
) -> Params:
    """Local training with FedProx proximal term: mu/2 * ||theta - theta_g||^2."""
    params: Params = [p.copy() for p in global_params]
    X, y = client.X_train, client.y_train
    n = len(y)
    if n == 0:
        return params
    for _ in range(cfg.local_epochs):
        perm = rng.permutation(n)
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            _, grads = model.loss_and_grad(params, X[idx], y[idx])
            # Add proximal gradient: mu * (theta_k - theta_g)
            for j, (g, p, gp) in enumerate(zip(grads, params, global_params)):
                g = g + mu * (p - gp)
                grads[j] = g
            optimizer.step(params, grads)
    return params


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
        self._client_optimizers: Dict[int, Adam] = {
            c.client_id: Adam(lr=cfg.eta) for c in split.clients
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
                _proximal_client_update(
                    global_params, client, cfg, rng, self._model,
                    self._client_optimizers[client.client_id], self.mu,
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
        self._client_optimizers: Dict[int, Adam] = {
            c.client_id: Adam(lr=cfg.eta) for c in split.clients
        }
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
                _client_update(
                    global_params, client, cfg, rng, self._model,
                    self._client_optimizers[client.client_id],
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
        self._client_optimizers: Dict[int, Adam] = {
            c.client_id: Adam(lr=cfg.eta) for c in split.clients
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
                _client_update(
                    global_params, client, cfg, rng, self._model,
                    self._client_optimizers[client.client_id],
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
