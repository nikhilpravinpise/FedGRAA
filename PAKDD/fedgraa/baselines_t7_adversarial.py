"""T7 defense reproductions (docs/research_c/STAGE_C_REMAINING_WORK_MASTER_PLAN_2026-08-02.md
§2; judgment calls in
docs/research_c/STAGE_C_T7_JUDGMENT_CALLS_2026-08-02.md, IDs JC-T7-1..12,16).

Two external defense reproductions, D2's own scope:

* :class:`SecFedNIDSTrainer` -- Zhao et al., *Future Generation Computer
  Systems* 134:154-169, 2022. Model-level poisoned-client detection: a
  per-client Synaptic-Intelligence-style importance accumulator
  (:func:`fedgraa.model._fused_importance_step`), a server-side consensus
  filter over the top-K most "commonly important" coordinates, and
  Stochastic Outlier Selection (Janssens et al., 2012) over each client's
  weight delta restricted to those coordinates. Reference code confirmed
  public and read in full: `github.com/zhangzhao156/SecFedNIDS`.
* :class:`WeiDetectTrainer` -- Sameera K.M. et al., arXiv:2504.04367
  (*JISA* 2025). Server-side validation-score generation + a Weibull
  distribution fit over those scores, retaining the top-T clients by CDF
  value.

AdaAggRL (D2's third defense arm) is **not** in this module -- it is reused
unmodified from `fedgraa/baselines_t5_adaptive.py::AdaAggRLTrainer`, per
JC-T7-16 (evaluated under D1a/label-flipping only, since its own
per-round client-training call is not a hookable seam without editing that
class, which the master plan says must not happen).

--------------------------------------------------------------------------
CITATION-VERIFICATION DISCLOSURE
--------------------------------------------------------------------------
Every mechanism below was read off primary-source full text or the
reference implementation's actual source code, not an abstract-level
summary -- see the full account and every disclosed judgment call in
`STAGE_C_T7_JUDGMENT_CALLS_2026-08-02.md`. The two load-bearing deviations,
summarized here (full rationale in that table):

* **SOS is implemented directly** (`sos_outlier_probabilities`), not via
  the `pyod` package SecFedNIDS's own reference code uses, following this
  project's own precedent (`fedgraa/privacy.py`) of not adding a new
  third-party dependency for one baseline's sake. Faithful to the
  algorithm's own published definition (Janssens et al., 2012): a
  perplexity-calibrated affinity matrix, then each point's outlier
  probability as the product, over every other point, of "that point did
  not bind me as a neighbor."
* **`contamination`/consensus-threshold/perplexity are all rescaled**
  proportionally to this project's own (much smaller) client count,
  per JC-T7-9 -- SecFedNIDS's reference script hard-codes all three to its
  own 100-client experimental setup.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from .config import FedConfig
from .metrics import accuracy, last_n_class_accuracy
from .model import (
    MLP,
    Params,
    _fused_importance_step,
    aggregate,
    build_tf_optimizer,
    init_params,
)
from .partition import FederatedSplit
from .trainer import TrainHistory, client_train_round

__all__ = [
    "SecFedNIDSTrainer",
    "WeiDetectTrainer",
    "sos_outlier_probabilities",
    "secfednids_consensus_filter",
    "weibull_top_t_selection",
    "validation_f1_scores",
]


# --------------------------------------------------------------------------- #
# Shared: Stochastic Outlier Selection (Janssens et al., 2012)
# --------------------------------------------------------------------------- #
def _pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    sq = np.sum(X * X, axis=1)
    D = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    return np.maximum(D, 0.0)


def _calibrate_sigma(dist_row: np.ndarray, target_perplexity: float,
                      max_iter: int = 50, tol: float = 1e-5) -> float:
    lo, hi = 1e-10, 1e10
    sigma = 1.0
    for _ in range(max_iter):
        sigma = 0.5 * (lo + hi)
        aff = np.exp(-dist_row / (2.0 * sigma ** 2))
        s = aff.sum()
        if s <= 1e-300:
            lo = sigma
            continue
        p = aff / s
        H = -np.sum(p * np.log(p + 1e-300))
        perp = np.exp(H)
        if abs(perp - target_perplexity) < tol:
            break
        if perp > target_perplexity:
            hi = sigma
        else:
            lo = sigma
    return sigma


def sos_outlier_probabilities(X: np.ndarray, perplexity: float = 4.0) -> np.ndarray:
    """Stochastic Outlier Selection (Janssens et al., 2012): for each row of
    ``X``, the probability that *no other row* selects it as a bound
    neighbor. High values indicate outliers.

    ``perplexity`` is capped at ``n-1`` automatically (a client population
    smaller than the perplexity target degenerates the affinity
    calibration otherwise).
    """
    n = X.shape[0]
    if n <= 1:
        return np.zeros(n)
    target_perp = min(perplexity, max(1.0, n - 1))
    D = _pairwise_sq_dist(X)
    A = np.zeros((n, n))
    for i in range(n):
        row = D[i].copy()
        row[i] = np.inf
        sigma = _calibrate_sigma(row, target_perp)
        aff = np.exp(-row / (2.0 * sigma ** 2))
        aff[i] = 0.0
        s = aff.sum()
        A[i] = aff / s if s > 1e-300 else np.full(n, 1.0 / (n - 1))
        A[i, i] = 0.0
    # Binding probability that j binds i is A[j, i]; outlier probability of
    # i is the product, over every j != i, of "j did not bind i."
    not_bound = 1.0 - A
    np.fill_diagonal(not_bound, 1.0)
    outlier_prob = np.prod(not_bound, axis=0)
    return outlier_prob


def _sos_flag_outliers(X: np.ndarray, contamination: float, perplexity: float) -> np.ndarray:
    """Boolean array, True where flagged as an outlier: the
    ``contamination``-fraction of rows with the highest outlier probability."""
    n = X.shape[0]
    scores = sos_outlier_probabilities(X, perplexity=perplexity)
    n_flag = max(0, min(n, int(round(contamination * n))))
    if n_flag == 0:
        return np.zeros(n, dtype=bool)
    threshold_idx = np.argsort(-scores)[:n_flag]
    flags = np.zeros(n, dtype=bool)
    flags[threshold_idx] = True
    return flags


# --------------------------------------------------------------------------- #
# SecFedNIDS (Zhao et al., FGCS 2022)
# --------------------------------------------------------------------------- #
def secfednids_consensus_filter(
    omega_per_client: Sequence[Params],
    delta_per_client: Sequence[Params],
    assumed_malicious_fraction: float = 0.3,
    consensus_frac: float = 0.9,
    top_k_frac: float = 0.1,
    perplexity: float = 4.0,
) -> np.ndarray:
    """SecFedNIDS's model-level detection (`defence_our`, JC-T7-9): each
    client's per-parameter importance (``omega``) selects its own top-K
    "important" coordinates; coordinates selected by more than
    ``consensus_frac * K`` clients form the consensus set; each client's
    weight delta restricted to that consensus set feeds
    :func:`sos_outlier_probabilities`. Returns a boolean array, True where a
    client is flagged malicious (to be excluded from aggregation).
    """
    K = len(omega_per_client)
    if K == 0:
        return np.zeros(0, dtype=bool)
    n_arrays = len(omega_per_client[0])

    # Per-client, per-array top-K% index selection.
    selected_indices: List[List[np.ndarray]] = []
    for omega in omega_per_client:
        per_array = []
        for arr in omega:
            flat = arr.ravel()
            k = max(1, int(round(top_k_frac * flat.size))) if flat.size > 1000 else \
                max(1, int(round(0.1 * flat.size)))
            idx = np.argsort(-np.abs(flat))[:k]
            per_array.append(idx)
        selected_indices.append(per_array)

    # Consensus: coordinates selected by more than consensus_frac*K clients.
    consensus_threshold = consensus_frac * K
    consensus_masks: List[np.ndarray] = []
    for j in range(n_arrays):
        size = omega_per_client[0][j].size
        counts = np.zeros(size, dtype=np.int32)
        for per_array in selected_indices:
            counts[per_array[j]] += 1
        consensus_masks.append(counts > consensus_threshold)

    # Feature vector per client: weight delta restricted to consensus coords.
    feature_dim = int(sum(m.sum() for m in consensus_masks))
    if feature_dim == 0:
        return np.zeros(K, dtype=bool)
    X = np.zeros((K, feature_dim), dtype=np.float64)
    for i, delta in enumerate(delta_per_client):
        col = 0
        for j, m in enumerate(consensus_masks):
            flat = delta[j].ravel()
            n_sel = int(m.sum())
            if n_sel > 0:
                X[i, col:col + n_sel] = flat[m]
            col += n_sel

    if np.ptp(X) == 0.0:
        return np.zeros(K, dtype=bool)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    return _sos_flag_outliers(X, contamination=assumed_malicious_fraction, perplexity=perplexity)


class SecFedNIDSTrainer:
    """SecFedNIDS baseline (D2): FedAvg-style rounds with attacker
    injection + the consensus/SOS defense above. See
    ``STAGE_C_T7_JUDGMENT_CALLS_2026-08-02.md`` JC-T7-9/10/11/12.

    ``attack_fn``/``attacker_fraction`` mirror the T7 harness's own
    attack-injection contract (§2.6 step 5): a fixed set of attacker client
    ids (deterministic given ``cfg.seed``, persistent for the whole run --
    JC-T7-17) has its trained contribution passed through ``attack_fn``
    (the uniform ``(P_current, P_global) -> P_poisoned`` interface,
    ``attacks.py::wrap_model_poison_attack``) before this round's
    defense/aggregation. D1a (label-flipping) is applied upstream, to
    ``split.clients[i].y_train`` directly, before this trainer is even
    constructed -- it needs no attack_fn wiring at all.
    """

    def __init__(
        self,
        split: FederatedSplit,
        cfg: FedConfig,
        attack_fn=None,
        attacker_fraction: float = 0.0,
        assumed_malicious_fraction: float = 0.3,
    ):
        self.split = split
        self.cfg = cfg
        self.attack_fn = attack_fn
        self.attacker_fraction = attacker_fraction
        self.assumed_malicious_fraction = assumed_malicious_fraction
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
        n_attackers = int(round(attacker_fraction * len(split.clients)))
        rng_id = np.random.default_rng(cfg.seed)
        self.attacker_ids = set(
            rng_id.choice(len(split.clients), size=n_attackers, replace=False).tolist()
        )
        self.last_flagged: List[int] = []
        self.attack_success_rate_history: List[float] = []

    def _client_round_with_importance(self, client, global_params, rng):
        model = self._model
        opt = self._client_tf_optimizers[client.client_id]
        model.set_params(global_params)
        omega_vars = [tf.Variable(tf.zeros_like(v), trainable=False)
                      for v in model._model.trainable_variables]
        X, y = client.X_train, client.y_train
        n = len(y)
        if n == 0:
            return model.get_params(), [v.numpy() for v in omega_vars]
        cfg = self.cfg
        pad_x = np.zeros((cfg.batch_size, X.shape[1]), dtype=np.float32)
        pad_y = np.zeros(cfg.batch_size, dtype=np.int32)
        all_perm = np.array(
            [rng.permutation(n) for _ in range(cfg.local_epochs)], dtype=np.int32,
        )
        for epoch in range(cfg.local_epochs):
            perm = all_perm[epoch]
            for start in range(0, n, cfg.batch_size):
                idx = perm[start:start + cfg.batch_size]
                n_valid = len(idx)
                if n_valid < cfg.batch_size:
                    pad_x[:n_valid] = X[idx]
                    pad_y[:n_valid] = y[idx]
                    x_batch, y_batch = pad_x, pad_y
                else:
                    x_batch, y_batch = X[idx], y[idx]
                _fused_importance_step(
                    tf.constant(x_batch, dtype=tf.float32),
                    tf.constant(y_batch, dtype=tf.int32),
                    model._model, opt, omega_vars,
                    tf.constant(n_valid, dtype=tf.int32),
                )
        return model.get_params(), [v.numpy() for v in omega_vars]

    def train(self, eval_every: int = 20, verbose: bool = False) -> TrainHistory:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        global_params: Params = init_params(self._model.layer_sizes, seed=cfg.seed)
        history = TrainHistory(method="secfednids", gamma=None, n_rare=cfg.n_rare)

        n = len(self.split.clients)
        w0 = np.full(n, 1.0 / n, dtype=np.float64)
        history.record(
            0,
            accuracy(self._model, global_params, self.split.X_val, self.split.y_val),
            last_n_class_accuracy(
                self._model, global_params, self.split.X_val, self.split.y_val,
                self.rare_class_ids,
            ),
            w0,
        )

        for rnd in range(1, cfg.rounds + 1):
            client_params: List[Params] = []
            omegas: List[Params] = []
            for i, client in enumerate(self.split.clients):
                p, om = self._client_round_with_importance(client, global_params, rng)
                if i in self.attacker_ids and self.attack_fn is not None:
                    # Uniform (P_current, P_global) -> P_poisoned contract
                    # (attacks.py's wrap_model_poison_attack et al.).
                    p = self.attack_fn(p, global_params)
                client_params.append(p)
                omegas.append(om)

            deltas = [[pj - gj for pj, gj in zip(p, global_params)] for p in client_params]
            flagged = secfednids_consensus_filter(
                omegas, deltas, assumed_malicious_fraction=self.assumed_malicious_fraction,
                consensus_frac=min(0.9, 1.0 - 1.0 / max(n, 2)),
            )
            self.last_flagged = [i for i, f in enumerate(flagged) if f]
            if self.attacker_ids:
                tp = len(set(self.last_flagged) & self.attacker_ids)
                self.attack_success_rate_history.append(1.0 - tp / len(self.attacker_ids))

            keep = [i for i in range(n) if not flagged[i]]
            if not keep:
                keep = list(range(n))
            w = np.zeros(n, dtype=np.float64)
            w[keep] = 1.0 / len(keep)
            global_params = aggregate(client_params, w)

            if rnd % eval_every == 0 or rnd == cfg.rounds:
                all_a = accuracy(self._model, global_params, self.split.X_val, self.split.y_val)
                last_a = last_n_class_accuracy(
                    self._model, global_params, self.split.X_val, self.split.y_val,
                    self.rare_class_ids,
                )
                history.record(rnd, all_a, last_a, w)
                if verbose:
                    print(
                        f"  [secfednids] round {rnd:3d}/{cfg.rounds}"
                        f"  all={all_a:.3f}  last-n={last_a:.3f}"
                        f"  flagged={self.last_flagged}",
                        flush=True,
                    )

        self.final_params = global_params
        return history


# --------------------------------------------------------------------------- #
# WeiDetect (Sameera K.M. et al., arXiv:2504.04367)
# --------------------------------------------------------------------------- #
def validation_f1_scores(
    model: MLP, client_params: Sequence[Params], X_ad: np.ndarray, y_ad: np.ndarray,
) -> np.ndarray:
    """Per-client macro-F1 on the server's auxiliary set ``AD`` (JC-T7-3)."""
    from sklearn.metrics import f1_score
    scores = np.zeros(len(client_params))
    for i, p in enumerate(client_params):
        y_pred = model.predict(p, X_ad)
        scores[i] = f1_score(y_ad, y_pred, average="macro", zero_division=0)
    return scores


def weibull_top_t_selection(
    scores: np.ndarray, assumed_malicious_fraction: float = 0.3,
) -> np.ndarray:
    """WeiDetect's Weibull-CDF top-T selection (Eqs. 11-14, JC-T7-1/2):
    fit a 2-parameter Weibull to ``scores`` (shifted to be strictly
    positive), rank by CDF value, retain the top
    ``T = round((1 - assumed_malicious_fraction) * M)``. Returns a boolean
    keep-mask.
    """
    from scipy import stats as scipy_stats

    M = len(scores)
    T = max(1, min(M, int(round((1.0 - assumed_malicious_fraction) * M))))
    s = np.asarray(scores, dtype=np.float64)
    shifted = s - s.min() + 1e-6  # Weibull support is (0, inf); scores can be 0
    if np.ptp(shifted) < 1e-9:
        # Degenerate: every client scores identically -- no information to
        # rank on, keep all (no basis to exclude anyone).
        return np.ones(M, dtype=bool)
    shape, loc, scale = scipy_stats.weibull_min.fit(shifted, floc=0)
    cdf = scipy_stats.weibull_min.cdf(shifted, shape, loc=loc, scale=scale)
    keep_idx = np.argsort(-cdf)[:T]
    keep = np.zeros(M, dtype=bool)
    keep[keep_idx] = True
    return keep


class WeiDetectTrainer:
    """WeiDetect baseline (D2): FedAvg-style rounds with attacker
    injection + the Weibull-CDF top-T filter above. See
    ``STAGE_C_T7_JUDGMENT_CALLS_2026-08-02.md`` JC-T7-1/2/3/4. ``attack_fn``
    is the same uniform ``(P_current, P_global) -> P_poisoned`` interface
    :class:`SecFedNIDSTrainer` uses.
    """

    def __init__(
        self,
        split: FederatedSplit,
        cfg: FedConfig,
        attack_fn=None,
        attacker_fraction: float = 0.0,
        assumed_malicious_fraction: float = 0.3,
    ):
        self.split = split
        self.cfg = cfg
        self.attack_fn = attack_fn
        self.attacker_fraction = attacker_fraction
        self.assumed_malicious_fraction = assumed_malicious_fraction
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
        # AD (JC-T7-3): server's pooled validation split, confidence>0.9 rows.
        self._X_ad, self._y_ad = self._build_ad()
        n_attackers = int(round(attacker_fraction * len(split.clients)))
        rng_id = np.random.default_rng(cfg.seed)
        self.attacker_ids = set(
            rng_id.choice(len(split.clients), size=n_attackers, replace=False).tolist()
        )
        self.last_flagged: List[int] = []
        self.attack_success_rate_history: List[float] = []

    def _build_ad(self) -> Tuple[np.ndarray, np.ndarray]:
        params0 = init_params(self._model.layer_sizes, seed=self.cfg.seed)
        self._model.set_params(params0)
        probs, _ = self._model.forward(self.split.X_val)
        conf = probs.max(axis=1)
        keep = conf > 0.9
        if keep.sum() == 0:
            return self.split.X_val, self.split.y_val
        return self.split.X_val[keep], self.split.y_val[keep]

    def train(self, eval_every: int = 20, verbose: bool = False) -> TrainHistory:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        global_params: Params = init_params(self._model.layer_sizes, seed=cfg.seed)
        history = TrainHistory(method="weidetect", gamma=None, n_rare=cfg.n_rare)

        n = len(self.split.clients)
        w0 = np.full(n, 1.0 / n, dtype=np.float64)
        history.record(
            0,
            accuracy(self._model, global_params, self.split.X_val, self.split.y_val),
            last_n_class_accuracy(
                self._model, global_params, self.split.X_val, self.split.y_val,
                self.rare_class_ids,
            ),
            w0,
        )

        for rnd in range(1, cfg.rounds + 1):
            client_params: List[Params] = []
            for i, client in enumerate(self.split.clients):
                p = client_train_round(
                    self._model, self._client_tf_optimizers[client.client_id],
                    client.X_train, client.y_train, cfg, global_params, rng,
                )
                if i in self.attacker_ids and self.attack_fn is not None:
                    p = self.attack_fn(p, global_params)
                client_params.append(p)

            scores = validation_f1_scores(self._model, client_params, self._X_ad, self._y_ad)
            keep_mask = weibull_top_t_selection(
                scores, assumed_malicious_fraction=self.assumed_malicious_fraction,
            )
            self.last_flagged = [i for i in range(n) if not keep_mask[i]]
            if self.attacker_ids:
                tp = len(set(self.last_flagged) & self.attacker_ids)
                self.attack_success_rate_history.append(1.0 - tp / len(self.attacker_ids))

            keep = [i for i in range(n) if keep_mask[i]]
            if not keep:
                keep = list(range(n))
            w = np.zeros(n, dtype=np.float64)
            w[keep] = 1.0 / len(keep)
            global_params = aggregate(client_params, w)

            if rnd % eval_every == 0 or rnd == cfg.rounds:
                all_a = accuracy(self._model, global_params, self.split.X_val, self.split.y_val)
                last_a = last_n_class_accuracy(
                    self._model, global_params, self.split.X_val, self.split.y_val,
                    self.rare_class_ids,
                )
                history.record(rnd, all_a, last_a, w)
                if verbose:
                    print(
                        f"  [weidetect] round {rnd:3d}/{cfg.rounds}"
                        f"  all={all_a:.3f}  last-n={last_a:.3f}"
                        f"  flagged={self.last_flagged}",
                        flush=True,
                    )

        self.final_params = global_params
        return history
