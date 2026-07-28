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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from .aggregation import (
    FedGRAA,
    FedGRAAv2,
    fedavg_weights,
    static_reparam_weights,
    two_phase_composite_grad,
)
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
from .partition import (
    ClientShard,
    FederatedSplit,
    ValidationSplit,
    apply_tags,
    bottom_n_rare_classes,
    local_label_histogram,
    stratum_tags,
)
from .privacy import laplace_mechanism
from .secagg import SecAggChannel, SecAggDropoutError


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
    # W2.5 (zeta_probe): relative-L2 gap between the federated grad_P L' and
    # the oracle-pooled reference, logged each eval round for FedGRAA-v2 runs.
    # NaN for every method that doesn't measure it (v1, FedAvg, Static).
    zeta: List[float] = field(default_factory=list)

    def record(
        self,
        rnd: int,
        all_a: float,
        last_a: float,
        w: np.ndarray,
        comp_loss: float = float("nan"),
        zeta: float = float("nan"),
    ) -> None:
        self.rounds.append(rnd)
        self.all_acc.append(all_a)
        self.last_n_acc.append(last_a)
        self.weights.append(np.asarray(w).copy())
        self.composite_loss.append(float(comp_loss))
        self.zeta.append(float(zeta))

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

        # Communication instrumentation -- purely additive counters, mirroring
        # FederatedTrainerV2's, so the frontier figure's leftmost ("1x") point
        # is a MEASURED byte count rather than an analytically-asserted one
        # (T-12 made the 6x/2x multipliers measurements; this makes the
        # denominator they are multiples of one too). Reads nothing, changes
        # no numeric behaviour, and is never consulted by the weight learner
        # -- the strict-reproduction path (T-19) is untouched.
        self.n_params = sum(int(p.size) for p in init_params(layer_sizes, seed=0))
        self.bytes_sent = 0
        self.comm_rounds = 0

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
            # No communication is billed here: v1's weight learner runs
            # entirely server-side against the pooled ``D_V`` it holds. That
            # is precisely the privacy violation FedGRAA-v2 removes, and
            # precisely why v1 sits at 1x on the frontier's traffic axis
            # while v2 sits at (1 + ceil(E'/m))x.
            w = self._round_weights(w, client_params)
            # --- global aggregation: theta_g = sum_k w_k theta_k ---------- #
            global_params = aggregate(client_params, w)

            # Algorithm-1 base traffic: global params broadcast down, trained
            # params returned up, once per round for every client. Same wire
            # model as FederatedTrainerV2._msg_bytes (float32), so the two
            # counters are directly comparable -- which is the whole point.
            self.bytes_sent += 4 * 2 * self.n_params * len(self.split.clients)
            self.comm_rounds += 1

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


# --------------------------------------------------------------------------- #
# W2.5 -- zeta_probe: measurement-only comparison of the federated grad_P L'
# against the oracle-pooled reference.
# --------------------------------------------------------------------------- #
def zeta_probe(
    model: MLP,
    params: Params,
    oracle_val: ValidationSplit,
    federated_grad_P: Params,
    gamma: float,
) -> float:
    """Relative-L2 gap ``zeta_t`` between a federated ``grad_P L'`` and the
    oracle-pooled reference (W2.5, FEDGRAA_V2_MASTER_PLAN.md §4.2/§6.2).

    **Measurement-only.** Computing the oracle reference requires a pooled
    :class:`~fedgraa.partition.ValidationSplit` -- exactly the server-held
    ``D_V`` a real privacy-compliant deployment must never construct (F12's
    whole point). This function exists so an experiment harness that *does*
    have the pooled ground truth available (every dataset in this repo does,
    since ``partition_non_iid`` builds it for exactly this purpose) can
    measure ``zeta_t`` for the frontier figure's annotation -- it must never
    be called from inside the deployed protocol itself
    (:class:`FederatedTrainerV2` calls it only from its evaluation branch,
    gated by ``measure_zeta=True``, never from the training path that
    produces the ``grad_P`` actually used to update ``w``).

    **Scope disclosure (§6.2 step 3's zeta decomposition).**
    :class:`FederatedTrainerV2` calls this with *both* sides recomputed fresh
    at the current ``global_params`` (post-round), so what it measures is
    ``zeta_secagg`` (float32 summation-order noise -- provably ~0 by T-4) and,
    if ``dp_epsilon`` is set, any downstream effect of DP noise flipping the
    rarity ranking (a different ``D_V^n``/``D_V^h``/``D_V^m`` split than the
    oracle's, hence a genuinely different gradient). It does **not** capture
    ``zeta_stale(m)`` -- the gap between a refresh's ``grad_P`` and the true
    gradient at a *later*, already-updated ``theta'_g`` several inner epochs
    on -- which would require per-epoch instrumentation inside
    :meth:`~fedgraa.aggregation.FedGRAAv2.update`'s loop rather than a single
    post-round measurement. ``test_zeta_decomposition_sources_are_separable``
    (T-18) isolates each source directly instead of relying on this one
    end-of-round call to separate them.
    """
    oracle_fg = FedGRAA(gamma=gamma, layer_sizes=model.layer_sizes)
    _, oracle_grad = oracle_fg._composite_grad_P(params, oracle_val)
    fed = np.concatenate([np.ravel(g) for g in federated_grad_P]).astype(np.float64)
    ora = np.concatenate([np.ravel(g) for g in oracle_grad]).astype(np.float64)
    den = np.linalg.norm(ora)
    diff = np.linalg.norm(fed - ora)
    return float(diff / den) if den else float(diff)


# --------------------------------------------------------------------------- #
# W2.4 -- FederatedTrainerV2: the privacy-compliant federated trainer.
# --------------------------------------------------------------------------- #
class FederatedTrainerV2:
    """FedGRAA-v2 (FEDGRAA_V2_MASTER_PLAN.md §3, W2.4).

    Removes both of FedGRAA-v1's structural privacy assumptions:

    * **No server-side oracle rare-class list.** Each round, the rarity
      ranking is derived from a SecAgg-summed, full-local-shard histogram
      (F3's ``h_k^rank``) via :func:`~fedgraa.partition.bottom_n_rare_classes`
      -- never from :meth:`~fedgraa.config.DatasetSpec.rare_client_ids`'s
      hard-coded ``rare_label_order``.
    * **No server-held labelled validation set.** The ``D_V^h``/``D_V^m``/
      ``D_V^n`` stratification is a length-``C`` tag vector
      (:func:`~fedgraa.partition.stratum_tags`) derived from a *second*
      SecAgg-summed histogram (F3's ``h_k^strat``, over each client's own
      held-out split) and broadcast to clients, who tag their own rows
      locally (F2, :func:`~fedgraa.partition.apply_tags`) -- the server never
      sees an individual row.
    * **The composite-loss gradient is reconstructed, not summed.** Each
      round's ``grad_P L'`` comes from
      :func:`~fedgraa.aggregation.two_phase_composite_grad` (F12), not a
      naive per-client gradient sum -- see that function's docstring for why
      the naive sum is wrong by 160-557% (measured,
      ``scripts/verify_v2_exactness.py``).

    **Dropout (T-7/T-13)** is modeled at the SecAgg-channel level: a client
    dropped for a round (``cfg.p_drop``) does not contribute to that round's
    rarity histograms or composite-gradient channels -- the two things this
    design actually routes through SecAgg (F4) -- but still runs ordinary
    local training and still participates in the plain (non-SecAgg)
    ``theta_g = sum_k w_k theta_k`` aggregation, which this design never
    puts behind SecAgg (Algorithm 1 is unchanged from v1).

    **Aborted rounds.** If fewer clients survive a round than
    ``secagg_threshold``, Shamir reconstruction fails and the server recovers
    *nothing* -- so that round produces no weight update: ``w`` is carried
    forward and training continues, and ``self.failed_secagg_rounds`` is
    incremented. This is the real protocol's own semantics (an unrecoverable
    round is aborted, not fatal), and it is what makes a dropout study
    possible at all under a security-correct majority threshold: at
    ``p_drop=0.3`` with K=9-14, ~9-10% of rounds abort, so propagating the
    exception would give a 200-round run a ~1e-9 chance of finishing. The
    abort *rate* is a measured, reportable property of the (K, threshold,
    p_drop) triple -- read it off ``failed_secagg_rounds`` rather than
    inferring it. ``zeta`` is recorded as NaN for an aborted round, since no
    federated gradient was reconstructed to compare against the oracle.

    **DP placement.** When ``cfg.dp_epsilon`` is set, Laplace noise is added
    by each client *before* SecAgg submission, never by the server after
    aggregation -- see :meth:`_histograms_and_tags`. This is what makes the
    aggregate noise ``(Delta_1/eps)*sqrt(2K)``, the quantity
    :func:`fedgraa.privacy.aggregate_noise_std` models and T-9 asserts.

    **Communication instrumentation (T-12).** ``self.bytes_sent`` and
    ``self.comm_rounds`` accumulate across the whole ``train()`` call, billed
    at the point each message would actually be sent: the base per-round
    local-training broadcast/return (Algorithm 1, identical to v1), plus, for
    every ``grad_refresh_every``-th inner epoch, the ``theta'_g`` broadcast
    and the two-phase protocol's own (now byte-optimal, single-array-return)
    phase-2 message -- see :func:`~fedgraa.aggregation.two_phase_composite_grad`'s
    docstring for why this specific message shape is what makes the measured
    total match FEDGRAA_V2_MASTER_PLAN.md §4.1's ``(1 + ceil(E'/m)) x baseline``
    formula rather than some other multiplier.
    """

    def __init__(
        self,
        split: FederatedSplit,
        cfg: FedConfig,
        secagg_threshold: Optional[int] = None,
        measure_zeta: bool = False,
    ):
        if cfg.grad_refresh_every < 1:
            raise ValueError("cfg.grad_refresh_every must be >= 1")
        self.split = split
        self.cfg = cfg
        self.spec = split.spec
        self.n_clients_total = split.n_clients
        self.secagg_threshold = (
            secagg_threshold if secagg_threshold is not None
            else max(1, self.n_clients_total // 2 + 1)
        )
        self.client_sizes = [c.n_train for c in split.clients]
        self.measure_zeta = measure_zeta

        layer_sizes = self.spec.model_config().layer_sizes
        self._model = MLP(layer_sizes, seed=cfg.seed)
        self.n_params = sum(int(p.size) for p in init_params(layer_sizes, seed=0))

        self.fedgraa_v2 = FedGRAAv2(
            gamma=cfg.gamma,
            eta_prime=cfg.eta_prime,
            weight_epochs=cfg.weight_epochs,
            layer_sizes=layer_sizes,
            grad_refresh_every=cfg.grad_refresh_every,
        )

        self._client_tf_optimizers: Dict[int, tf.keras.optimizers.Adam] = {
            c.client_id: build_tf_optimizer(
                cfg.eta, self._model._model.trainable_variables
            )
            for c in split.clients
        }

        # T-12 instrumentation: bytes/rounds billed across the whole run.
        self.bytes_sent = 0
        self.comm_rounds = 0
        self.last_rare_class_ids: List[int] = []
        self._last_grad_P: Optional[Params] = None
        # T-7/T-13: rounds whose SecAgg reconstruction failed (survivors <
        # threshold) and were therefore aborted rather than aggregated. See
        # :meth:`train` for why this is a counter and not an exception.
        self.failed_secagg_rounds = 0

    def _msg_bytes(self, n_float32: int) -> int:
        return int(n_float32) * 4  # float32 wire format, matching F5's byte model

    def _round_participants(
        self, rng: np.random.Generator, force_full: bool = False
    ) -> List[int]:
        """Client indices that survive this round's SecAgg (T-7/T-13)."""
        if force_full or self.cfg.p_drop <= 0.0:
            return list(range(self.n_clients_total))
        keep = rng.random(self.n_clients_total) >= self.cfg.p_drop
        idx = [i for i, k in enumerate(keep) if k]
        # Never drop to zero participants -- SecAgg's threshold check below is
        # what should reject an under-participation round, not this method.
        return idx if idx else [int(rng.integers(self.n_clients_total))]

    def _histograms_and_tags(
        self,
        rng: np.random.Generator,
        bill_bytes: bool = True,
        force_full: bool = False,
    ) -> Tuple[List[int], np.ndarray, List[int]]:
        """One round's rarity ranking + stratum tags, derived entirely from
        SecAgg-summed client histograms (F3) -- no oracle rare list, no
        server-held D_V. Returns ``(rare_ids, tags, participants)``."""
        n_classes = self.spec.n_classes
        participants = self._round_participants(rng, force_full=force_full)

        ch_rank = SecAggChannel(self.n_clients_total, self.secagg_threshold)
        ch_strat = SecAggChannel(self.n_clients_total, self.secagg_threshold)
        for i in participants:
            shard = self.split.clients[i]
            h_rank = local_label_histogram(shard, n_classes, scope="full")
            if self.cfg.dp_epsilon is not None:
                # DP is applied by EACH CLIENT, BEFORE submission -- never
                # server-side after aggregation. The server is the adversary
                # this design exists to defend against (F4), so noising a sum
                # the server has already computed exactly protects nothing
                # from it. Adding independent Laplace(0, Delta_1/eps) per
                # client also makes the aggregate noise
                # ``(Delta_1/eps) * sqrt(2K)`` -- exactly the quantity
                # :func:`fedgraa.privacy.aggregate_noise_std` models, that
                # the plan's Sec 5.2 table reports, that ``T-9`` asserts, and
                # that ``verify_v2_plan_claims.py``'s BottomN-recovery
                # measurement (F3) was computed under. A single server-side
                # draw would deliver ``sqrt(K)`` times less noise than every
                # one of those artifacts assumes.
                #
                # Deliberately NOT clipped per client: clipping before the
                # sum biases each contribution upward and destroys both the
                # unbiasedness and the closed-form sigma. The aggregate is
                # clipped once below instead, which is DP-safe by
                # post-processing immunity.
                h_rank = laplace_mechanism(h_rank, self.cfg.dp_epsilon, rng=rng)
            ch_rank.submit([h_rank])
            ch_strat.submit([local_label_histogram(shard, n_classes, scope="heldout")])

        if bill_bytes:
            # Two histograms, each length n_classes, SecAgg-summed from every
            # participant, round-trip (submit + broadcast-back of the tags).
            # Billed here, right after submission and before `aggregate()`:
            # in a real SecAgg deployment the masked shares are already on
            # the wire by this point, so an aborted round below (too few
            # survivors to reconstruct) must still count this traffic --
            # under-billing it would understate `bytes_sent` specifically on
            # T4's p_drop>0 cells, where aborts are frequent by design.
            self.bytes_sent += self._msg_bytes(2 * n_classes * len(participants))

        counts_rank = ch_rank.aggregate()[0].astype(np.float64)
        counts_strat = ch_strat.aggregate()[0]

        if self.cfg.dp_epsilon is not None:
            # Post-processing of the DP release -- no additional budget.
            counts_rank = np.clip(counts_rank, 0.0, None)

        rare_ids = bottom_n_rare_classes(
            counts_rank, self.spec.attack_ids, self.cfg.n_rare
        )
        tags = stratum_tags(counts_strat, rare_ids)
        return rare_ids, tags, participants

    def train(self, eval_every: int = 20, verbose: bool = False) -> TrainHistory:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        global_params: Params = init_params(self._model.layer_sizes, seed=cfg.seed)
        w = static_reparam_weights(self.client_sizes)
        history = TrainHistory(method="fedgraa_v2", gamma=cfg.gamma, n_rare=cfg.n_rare)

        # Round-0 rare-class ids, derived the same way (F3) so even round-0
        # reporting never touches the oracle rare_label_order. Not billed as
        # protocol traffic -- it is a reproduction-fidelity labelling step
        # for the round-0 report row, mirroring v1's round-0 convention
        # (round_zero_zero forces the accuracy figures themselves to 0.0).
        # ``force_full``: round 0 is a labelling step, not a protocol round
        # (it is explicitly not billed), so it uses full participation rather
        # than risking a SecAgg abort before training has begun.
        rare_ids, _tags0, _participants0 = self._histograms_and_tags(
            rng, bill_bytes=False, force_full=True
        )
        self.last_rare_class_ids = rare_ids

        if cfg.round_zero_zero:
            _all_0, _last_0 = 0.0, 0.0
        else:
            _all_0 = accuracy(
                self._model, global_params, self.split.X_val, self.split.y_val
            )
            _last_0 = last_n_class_accuracy(
                self._model, global_params, self.split.X_val, self.split.y_val,
                rare_ids,
            )
        history.record(0, _all_0, _last_0, w)

        t_start = time.time()
        for rnd in range(1, cfg.rounds + 1):
            # --- clients train locally (unchanged from v1 / Algorithm 1) --- #
            client_params: List[Params] = [
                client_train_round(
                    self._model,
                    self._client_tf_optimizers[client.client_id],
                    client.X_train, client.y_train,
                    cfg, global_params, rng,
                )
                for client in self.split.clients
            ]

            # --- F3: rarity ranking + stratum tags from SecAgg histograms -- #
            #
            # A round in which fewer clients survive than the Shamir
            # reconstruction threshold is an ABORTED ROUND, not a failed job.
            # Bonawitz et al.'s protocol cannot recover the sum at all in that
            # case, so the server has no w-update to apply -- it carries
            # ``w`` forward and proceeds to the next round, exactly as a real
            # deployment would. Raising out of ``train()`` instead would make
            # any dropout study at a majority threshold unrunnable: at
            # ``p_drop=0.3`` with K=9-14 clients, P(a round aborts) is ~9-10%,
            # so P(a 200-round run completes) is ~1e-9. The abort RATE is
            # itself the quantity a dropout experiment should report, which is
            # why it is counted here rather than swallowed.
            #
            # Local training (above) and the plain theta_g aggregation (below)
            # are unaffected: neither is routed through SecAgg in this design.
            round_aborted = False
            client_vals: List[ValidationSplit] = []
            try:
                rare_ids, tags, participants = self._histograms_and_tags(rng)
                self.last_rare_class_ids = rare_ids
                client_vals = [
                    apply_tags(
                        self.split.clients[i].X_val, self.split.clients[i].y_val,
                        tags, rare_ids,
                    )
                    for i in participants
                ]
                n_part = len(participants)

                def grad_provider(P: Params) -> Params:
                    grad_P, stats = two_phase_composite_grad(
                        self._model, P, client_vals, cfg.gamma,
                        n_clients_total=self.n_clients_total,
                        threshold=self.secagg_threshold,
                    )
                    # theta'_g broadcast (down) + phase-2's single-array return (up).
                    self.bytes_sent += self._msg_bytes(2 * self.n_params * n_part)
                    self.bytes_sent += self._msg_bytes(
                        stats["phase1_scalars_per_client"] * n_part
                        + stats["phase1_broadcast_scalars"]
                    )
                    self.comm_rounds += stats["rounds"]
                    self._last_grad_P = grad_P
                    return grad_P

                # theta'_g = sum_k w_k theta_k is computed over EVERY client
                # (Algorithm 1's aggregation is never SecAgg'd or dropout-gated
                # in this design -- only the histogram/gradient channels are,
                # via `client_vals`/`participants` above); `grad_provider`
                # above is independent of this alignment and only consults the
                # SecAgg channels' own participant list.
                w = self.fedgraa_v2.update(client_params, w, grad_provider)
            except SecAggDropoutError:
                self.failed_secagg_rounds += 1
                round_aborted = True
                rare_ids = self.last_rare_class_ids
                # ``w`` is carried forward unchanged -- no weight update this
                # round, because the server never recovered the gradient.

            # Algorithm-1 base traffic: broadcast global params down, trained
            # params up, once per round for every client (unaffected by
            # SecAgg dropout -- this channel is never SecAgg'd in this design).
            self.bytes_sent += self._msg_bytes(2 * self.n_params * self.n_clients_total)
            self.comm_rounds += 1

            global_params = aggregate(client_params, w)

            if rnd % eval_every == 0 or rnd == cfg.rounds:
                all_a = accuracy(
                    self._model, global_params, self.split.X_val, self.split.y_val
                )
                last_a = last_n_class_accuracy(
                    self._model, global_params, self.split.X_val, self.split.y_val,
                    rare_ids,
                )
                zeta = float("nan")
                # A round whose SecAgg reconstruction failed has no federated
                # gradient to compare against the oracle, so zeta is genuinely
                # undefined there -- recorded as NaN rather than silently
                # reusing a stale participant set from an earlier round.
                if self.measure_zeta and not round_aborted:
                    # W2.5: compute BOTH the federated and the oracle-pooled
                    # grad_P L' fresh, at the SAME point (global_params after
                    # this round's aggregation) -- reusing the mid-round
                    # `_last_grad_P` here would confound the measurement with
                    # "which inner epoch it was last refreshed at" on top of
                    # the actual protocol gap this probe exists to measure.
                    fed_grad_now, _ = two_phase_composite_grad(
                        self._model, global_params, client_vals, cfg.gamma,
                        n_clients_total=self.n_clients_total,
                        threshold=self.secagg_threshold,
                    )
                    zeta = zeta_probe(
                        self._model, global_params, self.split.val_split,
                        fed_grad_now, cfg.gamma,
                    )
                history.record(rnd, all_a, last_a, w, float("nan"), zeta)
                if verbose:
                    elapsed = time.time() - t_start
                    eta = elapsed / rnd * (cfg.rounds - rnd)
                    print(
                        f"  [fedgraa_v2 g={cfg.gamma} m={cfg.grad_refresh_every}]"
                        f" round {rnd:3d}/{cfg.rounds}"
                        f"  all={all_a:.3f}  last-n={last_a:.3f}"
                        + (f"  zeta={zeta:.3e}" if np.isfinite(zeta) else "")
                        + f"  elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                        flush=True,
                    )

        self.final_params = global_params
        return history
