"""The Algorithm-1 federated training loop (server + ClientUpdate).

Supports the three methods compared in the paper:

* ``"fedavg"``  — fixed data-size weights every round (eq. 5).
* ``"static"``  — fixed inverse-size weights every round (eq. 7).
* ``"fedgraa"`` — weights initialized by Static Reparam and **re-learned each
  round** by :class:`~fedgraa.aggregation.FedGRAA` (Algorithm 2).

The only line that differs between methods is how ``w^t`` is produced; the local
``ClientUpdate`` (Adam, ``E`` epochs, batch ``B``) and the global aggregation
``theta_g = sum_k w_k theta_k`` are identical across all three.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .aggregation import FedGRAA, fedavg_weights, static_reparam_weights
from .config import FedConfig
from .metrics import accuracy, last_n_class_accuracy
from .model import MLP, Adam, Params, aggregate, init_params
from .partition import ClientShard, FederatedSplit


@dataclass
class TrainHistory:
    """Per-evaluation-round metrics collected during training."""

    method: str
    gamma: Optional[float]
    n_rare: int
    rounds: List[int] = field(default_factory=list)
    all_acc: List[float] = field(default_factory=list)
    last_n_acc: List[float] = field(default_factory=list)
    weights: List[np.ndarray] = field(default_factory=list)

    def record(self, rnd: int, all_a: float, last_a: float, w: np.ndarray) -> None:
        self.rounds.append(rnd)
        self.all_acc.append(all_a)
        self.last_n_acc.append(last_a)
        self.weights.append(np.asarray(w).copy())

    @property
    def final_all_acc(self) -> float:
        return self.all_acc[-1] if self.all_acc else 0.0

    @property
    def final_last_n_acc(self) -> float:
        return self.last_n_acc[-1] if self.last_n_acc else 0.0


def _client_update(
    global_params: Params,
    client: ClientShard,
    cfg: FedConfig,
    rng: np.random.Generator,
) -> Params:
    """Local SGD/Adam training on one client (Algorithm 1, ClientUpdate)."""
    params: Params = [p.copy() for p in global_params]
    optimizer = Adam(lr=cfg.eta)  # fresh local optimizer each round
    X, y = client.X_train, client.y_train
    n = len(y)
    if n == 0:
        return params
    for _ in range(cfg.local_epochs):
        perm = rng.permutation(n)
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            _, grads = MLP.loss_and_grad(params, X[idx], y[idx])
            optimizer.step(params, grads)
    return params


class FederatedTrainer:
    """Runs the federated learning process for a chosen aggregation method."""

    def __init__(
        self,
        split: FederatedSplit,
        cfg: FedConfig,
        method: str = "fedgraa",
    ):
        method = method.lower()
        if method not in {"fedavg", "static", "fedgraa"}:
            raise ValueError(f"Unknown method {method!r}")
        self.split = split
        self.cfg = cfg
        self.method = method
        self.spec = split.spec
        self.rare_class_ids = self.spec.rare_client_ids(cfg.n_rare)

        self.client_sizes = [c.n_train for c in split.clients]
        self.fedgraa: Optional[FedGRAA] = None
        if method == "fedgraa":
            self.fedgraa = FedGRAA(
                gamma=cfg.gamma,
                eta_prime=cfg.eta_prime,
                weight_epochs=cfg.weight_epochs,
            )

    def _initial_weights(self) -> np.ndarray:
        if self.method == "fedavg":
            return fedavg_weights(self.client_sizes)
        # static and fedgraa both initialize via Static Reparameterization (eq. 7)
        return static_reparam_weights(self.client_sizes)

    def _round_weights(
        self, w_prev: np.ndarray, client_params: List[Params]
    ) -> np.ndarray:
        if self.method == "fedavg":
            return fedavg_weights(self.client_sizes)
        if self.method == "static":
            return static_reparam_weights(self.client_sizes)
        assert self.fedgraa is not None
        return self.fedgraa.update(client_params, w_prev, self.split.val_split)

    def train(
        self,
        eval_every: int = 20,
        verbose: bool = False,
    ) -> TrainHistory:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        layer_sizes = self.spec.model_config().layer_sizes
        global_params: Params = init_params(layer_sizes, seed=cfg.seed)

        w = self._initial_weights()
        history = TrainHistory(
            method=self.method,
            gamma=cfg.gamma if self.method == "fedgraa" else None,
            n_rare=cfg.n_rare,
        )

        for rnd in range(1, cfg.rounds + 1):
            # --- clients train locally (in parallel conceptually) ---------- #
            client_params: List[Params] = [
                _client_update(global_params, client, cfg, rng)
                for client in self.split.clients
            ]
            # --- server produces this round's weights --------------------- #
            w = self._round_weights(w, client_params)
            # --- global aggregation: theta_g = sum_k w_k theta_k ---------- #
            global_params = aggregate(client_params, w)

            if rnd % eval_every == 0 or rnd == cfg.rounds:
                all_a = accuracy(global_params, self.split.X_val, self.split.y_val)
                last_a = last_n_class_accuracy(
                    global_params,
                    self.split.X_val,
                    self.split.y_val,
                    self.rare_class_ids,
                )
                history.record(rnd, all_a, last_a, w)
                if verbose:
                    print(
                        f"[{self.method:>7} g={history.gamma} n={cfg.n_rare}] "
                        f"round {rnd:3d}  all={all_a:.3f}  last-n={last_a:.3f}"
                    )

        self.final_params = global_params
        return history
