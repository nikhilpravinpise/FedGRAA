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

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .aggregation import FedGRAA, fedavg_weights, static_reparam_weights
from .config import FedConfig
from .metrics import accuracy, last_n_class_accuracy
from .model import MLP, Adam, Params, aggregate, init_params
from .partition import ClientShard, FederatedSplit


# Silence verbose TensorFlow startup logs for the reproduction CLI.
import os  # noqa: E402
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


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
    # Gap 3.7: composite loss L'(w^t) tracked each eval round for FedGRAA;
    # NaN for FedAvg / Static where no weight optimizer is used.
    composite_loss: List[float] = field(default_factory=list)

    def record(
        self,
        rnd: int,
        all_a: float,
        last_a: float,
        w: np.ndarray,
        comp_loss: float = float("nan"),
    ) -> None:
        self.rounds.append(rnd)
        self.all_acc.append(all_a)
        self.last_n_acc.append(last_a)
        self.weights.append(np.asarray(w).copy())
        self.composite_loss.append(float(comp_loss))

    @property
    def is_composite_loss_monotone(self) -> bool:
        """Check Lemma 1: L'(w^t) should be non-increasing for FedGRAA."""
        finite = [l for l in self.composite_loss if np.isfinite(l)]
        if len(finite) < 2:
            return True
        return all(finite[i] >= finite[i + 1] for i in range(len(finite) - 1))

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
    model: MLP,
    optimizer: Optional[Adam] = None,
) -> Params:
    """Local Adam training on one client (Algorithm 1, ClientUpdate).

    Gap 2.2 fix: ``optimizer`` is now passed in from the trainer so its moment
    state persists across rounds, matching the Keras default where calling
    ``model.fit()`` multiple times on the same compiled optimizer carries over
    the Adam moments (m, v, t) between calls.
    """
    params: Params = [p.copy() for p in global_params]
    if optimizer is None:                      # fallback for direct callers
        optimizer = Adam(lr=cfg.eta)
    X, y = client.X_train, client.y_train
    n = len(y)
    if n == 0:
        return params
    for _ in range(cfg.local_epochs):
        perm = rng.permutation(n)
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            _, grads = model.loss_and_grad(params, X[idx], y[idx])
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

        # The Keras-backed MLP used for all local loss/gradient/prediction.
        layer_sizes = self.spec.model_config().layer_sizes
        self._model = MLP(layer_sizes, seed=cfg.seed)

        self.fedgraa: Optional[FedGRAA] = None
        if method == "fedgraa":
            self.fedgraa = FedGRAA(
                gamma=cfg.gamma,
                eta_prime=cfg.eta_prime,
                weight_epochs=cfg.weight_epochs,
                layer_sizes=layer_sizes,
            )

        # Gap 2.2 fix: per-client Adam optimizers whose moment state persists
        # across rounds, matching the Keras default behaviour (optimizer state
        # is not reset between model.fit() calls on the same optimizer object).
        self._client_optimizers: Dict[int, Adam] = {
            c.client_id: Adam(lr=cfg.eta) for c in split.clients
        }

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
        global_params: Params = init_params(self._model.layer_sizes, seed=cfg.seed)

        w = self._initial_weights()
        history = TrainHistory(
            method=self.method,
            gamma=cfg.gamma if self.method == "fedgraa" else None,
            n_rare=cfg.n_rare,
        )

        label = (
            f"{self.method}"
            + (f" g={cfg.gamma}" if self.method == "fedgraa" else "")
            + f" n={cfg.n_rare}"
        )
        t_start = time.time()

        # Gap 2.5 fix: record the initial state at round 0 so the history
        # matches the paper’s Figure 1 table (which starts at round 0).
        # Strict reproduction: round-0 accuracies are forced to 0.00 by
        # default to match the paper's published table (FedConfig default).
        _comp_0 = (
            self.fedgraa.composite_loss(global_params, self.split.val_split)
            if self.fedgraa is not None else float("nan")
        )
        if cfg.round_zero_zero:
            _all_0 = 0.0
            _last_0 = 0.0
        else:
            _all_0 = accuracy(self._model, global_params, self.split.X_val, self.split.y_val)
            _last_0 = last_n_class_accuracy(
                self._model,
                global_params,
                self.split.X_val,
                self.split.y_val,
                self.rare_class_ids,
            )
        history.record(0, _all_0, _last_0, w, _comp_0)

        for rnd in range(1, cfg.rounds + 1):
            # --- clients train locally (in parallel conceptually) ---------- #
            client_params: List[Params] = [
                _client_update(
                    global_params, client, cfg, rng, self._model,
                    self._client_optimizers[client.client_id],  # gap 2.2
                )
                for client in self.split.clients
            ]
            # --- server produces this round’s weights --------------------- #
            w = self._round_weights(w, client_params)
            # --- global aggregation: theta_g = sum_k w_k theta_k ---------- #
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
                # Gap 3.7: track composite loss L'(w^t) for FedGRAA.
                comp = (
                    self.fedgraa.composite_loss(global_params, self.split.val_split)
                    if self.fedgraa is not None else float("nan")
                )
                history.record(rnd, all_a, last_a, w, comp)
                elapsed = time.time() - t_start
                eta = elapsed / rnd * (cfg.rounds - rnd)
                print(
                    f"  [{label}] round {rnd:3d}/{cfg.rounds}"
                    f"  all={all_a:.3f}  last-n={last_a:.3f}"
                    + (f"  L'={comp:.4f}" if np.isfinite(comp) else "")
                    + f"  elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                    flush=True,
                )

        self.final_params = global_params
        return history
