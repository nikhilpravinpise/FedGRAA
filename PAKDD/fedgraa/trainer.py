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

# Env vars MUST be set BEFORE importing tensorflow (Phase 7 of speedup plan).
import os  # noqa: E402
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")

import hashlib
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import tensorflow as tf

from .aggregation import FedGRAA, fedavg_weights, static_reparam_weights
from .config import FedConfig
from .metrics import accuracy, last_n_class_accuracy
from .model import (
    MLP,
    Params,
    _fused_client_step,
    aggregate,
    build_tf_optimizer,
    init_params,
)
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


# --------------------------------------------------------------------------- #
# Phase 14: Checkpoint / Resume
# --------------------------------------------------------------------------- #

@dataclass
class TrainingCheckpoint:
    round: int
    global_params: Params
    weights: np.ndarray
    optimizer_states: Dict[int, List[np.ndarray]]
    eval_history: TrainHistory
    rng_state: dict


def save_checkpoint(checkpoint: TrainingCheckpoint, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(checkpoint, f)
    os.replace(tmp, path)


def load_checkpoint(path: str) -> Optional[TrainingCheckpoint]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def resume_or_init(
    checkpoint_dir: Optional[str],
    trainer: "FederatedTrainer",
    global_params: Params,
    w: np.ndarray,
    history: TrainHistory,
    rng: np.random.Generator,
    rnd_start: int,
) -> int:
    """Load checkpoint if available, otherwise return ``rnd_start`` unchanged."""
    if checkpoint_dir is None:
        return rnd_start
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(ckpt_dir / f"{trainer.method}.ckpt")
    ckpt = load_checkpoint(ckpt_path)
    if ckpt is None:
        return rnd_start
    # Restore all state from checkpoint (element-wise copy preserves array identity)
    for p, cp in zip(global_params, ckpt.global_params):
        p[...] = cp
    w[:] = ckpt.weights
    history.rounds = list(ckpt.eval_history.rounds)
    history.all_acc = list(ckpt.eval_history.all_acc)
    history.last_n_acc = list(ckpt.eval_history.last_n_acc)
    history.weights = [x.copy() for x in ckpt.eval_history.weights]
    history.composite_loss = list(ckpt.eval_history.composite_loss)
    rng.bit_generator.state = ckpt.rng_state
    for cid, state in ckpt.optimizer_states.items():
        if cid in trainer._client_tf_optimizers:
            trainer._client_tf_optimizers[cid].set_weights(state)
    resumed_round = ckpt.round
    print(f"  [resume] loaded checkpoint at round {resumed_round}", flush=True)
    return resumed_round + 1


def _extract_optimizer_states(
    optimizers: Dict[int, tf.keras.optimizers.Optimizer],
) -> Dict[int, List[np.ndarray]]:
    return {
        cid: [v.numpy() for v in opt.variables]
        for cid, opt in optimizers.items()
    }


def client_train_round(
    model: MLP,
    opt: tf.keras.optimizers.Optimizer,
    X: np.ndarray,
    y: np.ndarray,
    cfg: FedConfig,
    global_weights: Params,
    rng: np.random.Generator,
) -> Params:
    """Train one client with fused XLA — no numpy↔TF ping-pong.

    Operates directly on ``model``'s ``tf.Variable`` s via ``_fused_client_step``.
    One ``set_weights`` per round (not per iteration). Reads params back to
    numpy exactly once at the end.

    Args:
        model: Keras-backed MLP with pre-built variables.
        opt: Pre-built TF Adam optimizer (``build_tf_optimizer``).
        X: Client training features (numpy).
        y: Client training labels (numpy).
        cfg: FedConfig (uses ``local_epochs``, ``batch_size``).
        global_weights: Global model params to start from (numpy arrays).
        rng: NumPy RNG for shuffling.
    """
    # ONE numpy→TF copy per round (instead of per iteration):
    model.set_params(global_weights)

    n = len(y)
    if n == 0:
        return model.get_params()

    # Pre-allocated padding buffers — allocated once, reused every iteration
    # to avoid per-iteration ``np.zeros`` overhead.
    pad_x = np.zeros((cfg.batch_size, X.shape[1]), dtype=np.float32)
    pad_y = np.zeros(cfg.batch_size, dtype=np.int32)

    # Phase 12: pre-compute all epoch shuffles once (saves ~1.2ms per round)
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
            _fused_client_step(
                tf.constant(x_batch, dtype=tf.float32),
                tf.constant(y_batch, dtype=tf.int32),
                model._model, opt,
                tf.constant(n_valid, dtype=tf.int32),
            )
    # ONE TF→numpy copy per round:
    return model.get_params()


def _client_update(
    global_params: Params,
    client: ClientShard,
    cfg: FedConfig,
    rng: np.random.Generator,
    model: MLP,
    optimizer: tf.keras.optimizers.Optimizer,
) -> Params:
    """Legacy shim — delegates to ``client_train_round``.

    Kept so that ``baselines.py`` callers continue to work with the old
    call signature while benefiting from the fused XLA path.
    """
    return client_train_round(
        model, optimizer, client.X_train, client.y_train,
        cfg, global_params, rng,
    )


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

        # Phase 2: pre-built TF Adam optimizers for fused XLA client training.
        # Each optimizer has its slot variables pre-created via
        # ``build_tf_optimizer`` so that ``@tf.function(jit_compile=True)``
        # does not re-trace. The moment state persists across rounds (Gap 2.2).
        self._client_tf_optimizers: Dict[int, tf.keras.optimizers.Adam] = {
            c.client_id: build_tf_optimizer(cfg.eta,
                                            self._model._model.trainable_variables)
            for c in split.clients
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
        checkpoint_dir: Optional[str] = None,
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

        # Phase 14: resume from checkpoint if available
        rnd_start = resume_or_init(
            checkpoint_dir, self, global_params, w, history, rng, 1,
        )
        t_start = time.time()

        # Only record round 0 if we didn't resume (or if history is empty)
        if rnd_start == 1:
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
            # --- clients train locally (fused XLA, no ping-pong) ----------- #
            client_params: List[Params] = [
                client_train_round(
                    self._model,
                    self._client_tf_optimizers[client.client_id],
                    client.X_train, client.y_train,
                    cfg, global_params, rng,
                )
                for client in self.split.clients
            ]
            # --- server produces this round's weights --------------------- #
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
                # Phase 14: save checkpoint after each eval
                if checkpoint_dir is not None:
                    ckpt_dir = Path(checkpoint_dir)
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_path = str(ckpt_dir / f"{self.method}.ckpt")
                    save_checkpoint(
                        TrainingCheckpoint(
                            round=rnd,
                            global_params=[p.copy() for p in global_params],
                            weights=w.copy(),
                            optimizer_states=_extract_optimizer_states(
                                self._client_tf_optimizers
                            ),
                            eval_history=history,
                            rng_state=rng.bit_generator.state,
                        ),
                        ckpt_path,
                    )

        self.final_params = global_params
        return history
