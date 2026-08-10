"""Tier-T5 same-domain baselines: the three federated-NIDS / adaptive-aggregation
methods closest to FedGRAA in the literature.

* :class:`FedMADETrainer` — FedMADE (Sun, Sharma, Nwodo, Stavrou, Wang;
  ISC 2024; arXiv:2408.07152). Server-side CPM -> DBSCAN clustering ->
  least-squares cluster weights.
* :class:`SBFedNIDSTrainer` — SBFedNIDS (Zeng, Zheng, Li; ICASSP 2025;
  DOI 10.1109/ICASSP49660.2025.10889171). Data-level self-balancing:
  federated cost-sensitive learning + client SMOTE.
* :class:`FedAPATrainer` — FedAPA (Sun, Sun, Pan, Fu, Guo; IJCAI 2025;
  arXiv:2502.07456). Server-side gradient of the client-parameter change
  with respect to the per-client aggregation-weight row.

Why these three, and why in their own module
--------------------------------------------
``fedgraa/baselines.py`` holds the generic FL baselines (FedProx, FedOpt, a
static distribution-aware control). These three are different in kind: they
are FedGRAA's *same-domain* competitors -- two are federated NIDS papers with
the same class-imbalance motivation, and one shares FedGRAA's core idea of
*learning* the aggregation weights by gradient descent rather than fixing them.

**FedAPA is flagged in this project's literature review as FedGRAA's single
most mechanistically similar competitor**, and a reviewer would read it as the
most important missing comparison in the paper. Both methods learn aggregation
weights on the server by differentiating a scalar objective with respect to the
weight vector. They differ in the objective: FedGRAA differentiates a
*composite validation loss* ``L' = (1-gamma) min(L_h, L_m) + gamma L_u``
(``fedgraa.aggregation.FedGRAA``), evaluated on a held-out stratified
validation split, and produces **one** global weight vector; FedAPA
differentiates a *proxy for each client's own local loss* -- half the squared
norm of that client's parameter change -- using no validation data at all, and
produces **one weight row per client** (a full M x M weight matrix), because
FedAPA is a personalization method and FedGRAA is not. Keeping FedAPA next to
the other two same-domain baselines, rather than mixed in with FedProx/FedOpt,
keeps that comparison legible.

Verification status (per method)
--------------------------------
This codebase's rule is that an unverified mechanism must be labelled as such
in the same breath as it is described (see
``fedgraa.baselines.StaticDistributionAwareTrainer``'s docstring for the
expected tone). Applying that rule here:

**FedMADE — VERIFIED against the arXiv full text (arXiv:2408.07152).** Every
step below is taken from the paper's Section 4.2 and Algorithm 1: the Class
Probability Matrix, DBSCAN over CPMs, the cluster CPM, the least-squares
objective ``min_alpha ||sum_k alpha_k P_{C_k} - I||_2`` with ``alpha_k >= 0``,
the per-client weight ``beta_i = alpha_k / |C_k|``, the final normalization
``beta_i <- beta_i / sum beta``, the ten-samples-per-class server auxiliary
set, and the Adam(lr=0.01) solver for ``alpha``. Three details the paper does
*not* state are marked ``INFERRED`` at their point of use: DBSCAN's ``eps`` and
``min_samples``, the number of Adam steps for ``alpha``, and whether ``||.||_2``
on a matrix means Frobenius or spectral norm.

**SBFedNIDS — NOT VERIFIED beyond the published abstract; the full text is
paywalled.** The paper is real and was located: Yifan Zeng and Peijia Zheng
(Sun Yat-sen University) and Jian Li (Qilu University of Technology /
Shandong Academy of Sciences), "A Federated Learning Network Intrusion
Detection System for Multiple Imbalances", ICASSP 2025, published 2025-03-12,
DOI 10.1109/ICASSP49660.2025.10889171, IEEE Xplore document 10889171. It has
no arXiv preprint and no open-access copy: OpenAlex reports
``oa_status: closed`` with ``any_repository_has_fulltext: false``, IEEE Xplore
returns no body to an unauthenticated fetch, and no author-hosted PDF, code
repository, or slide deck was found. What is verified is therefore exactly the
abstract, quoted here in full so a reader can check this implementation against
it without a subscription:

    "Centralized training of deep learning-based network intrusion detection
    systems (DLNIDS) raises privacy concerns and incurs huge overhead.
    Federated learning (FL), while preserving privacy, confronts challenges
    including size imbalance, heterogeneous and global imbalance. To address
    these issues, this paper proposes a Self-Balancing Federated learning
    Network Intrusion Detection System (SBFedNIDS) for accurate and efficient
    network intrusion detection while preserving privacy and tackling negative
    impact caused by multiple imbalances resulting from FL. SBFedNIDS can
    efficiently mitigate model bias cause by multiple imbalances and addresses
    imbalances at data-level via federated cost sensitive learning (FCSL) and
    client synthetic minority over-sampling technique (CSMOTE), without
    requiring additional training resources, unlike generative models.
    Extensive experiments on a benchmark dataset show that SBFedNIDS
    outperforms baselines in multiple imbalanced scenarios and even surpasses
    centralized training in several settings."

Two consequences, stated plainly:

1. The *names and level* of the two components are verified -- FCSL and CSMOTE,
   both explicitly "at data-level". Their *equations* are not: the abstract
   gives no cost function, no oversampling ratio, no neighbour count. What
   :class:`SBFedNIDSTrainer` implements under those two names is a best-faith
   reconstruction from the standard forms of each technique (inverse-frequency
   cost-sensitive cross-entropy; SMOTE as defined by Chawla et al., 2002),
   parameterized to be inspectable. It is a reconstruction, not a reproduction.
2. This project's prior literature notes described SBFedNIDS as
   "self-balancing local rebalance + **balance-aware aggregation** (balance
   statistics rather than gradient-based)". The first half is corroborated by
   the abstract; **the second half is not.** The abstract says the method
   addresses the imbalances "at data-level" and never mentions a modified
   aggregation rule. This implementation therefore does *not* invent a
   balance-statistics aggregation rule. It aggregates by data size, as FedAvg
   does -- over the *post-rebalancing* client sizes, which is not an extra
   assumption but the arithmetic consequence of CSMOTE changing how many rows
   each client holds. If the paywalled Section 3 turns out to contain a genuine
   balance-aware aggregation rule, this class is wrong in that specific,
   documented way and should be revised rather than defended.

**FedAPA — VERIFIED against the arXiv full text (arXiv:2502.07456) except for
one sign convention and two unstated hyperparameters.** Verified: the
feature-extractor/decision-layer split (only the extractor is uploaded and
aggregated), Eq. 5 ``theta_bar_i = sum_j a_ij theta_j``, Eq. 6
``delta_theta_i = theta_i - theta_bar_i``, Eq. 7's form
``A_i <- A_i -/+ eta (grad_{A_i} theta_bar_i)^T delta_theta_i``, the stated
proxy ("half of the squared norm of the client-parameter delta serves as a
proxy for the local loss"), the initialization ``a_ii = 1``/``a_ij = 0``, and
Algorithm 1 lines 11-13 (clip to ``[0,1]``, set ``a_ii <- 0.5``, normalize).
The sign of Eq. 7 and the definition of "normalize" are discussed at
:func:`fedapa_weight_row_update`, where the choice is made and justified rather
than hidden.

Relation to the tiers
---------------------
These are T3/T5 comparison baselines for the compute campaign. Like everything
in ``fedgraa/baselines.py``, they are Related Work, not part of the base
paper's Figures 1-3 reproduction.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from .aggregation import fedavg_weights
from .config import FedConfig
from .metrics import accuracy, last_n_class_accuracy
from .model import (
    MLP,
    Adam,
    Params,
    aggregate,
    build_tf_optimizer,
    init_params,
)
from .partition import ClientShard, FederatedSplit
from .trainer import TrainHistory, client_train_round


# =========================================================================== #
# FedMADE (arXiv:2408.07152) -- VERIFIED
# =========================================================================== #
def build_server_auxiliary_set(
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    per_class: int = 10,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """FedMADE's server-side auxiliary set: up to ``per_class`` rows per class.

    Verified: "the server ... a small auxiliary dataset of ten samples from
    each class" (Section 4.2), hence ``per_class=10``. The paper does not say
    where those rows come from. INFERRED: they are drawn (without replacement,
    deterministically from ``seed``) from the server-side pooled validation
    pool ``D_V``, which is the only labelled data the server holds in this
    harness. Classes absent from ``D_V`` simply contribute no rows; see
    :func:`class_probability_matrix` for how their CPM row is handled.
    """
    rng = np.random.default_rng(seed)
    idx_parts: List[np.ndarray] = []
    for c in range(n_classes):
        idx_c = np.flatnonzero(y_val == c)
        if len(idx_c) == 0:
            continue
        take = min(per_class, len(idx_c))
        idx_parts.append(rng.choice(idx_c, size=take, replace=False))
    if not idx_parts:
        raise ValueError("server auxiliary set is empty: D_V contains no labels")
    idx = np.concatenate(idx_parts)
    return X_val[idx], y_val[idx]


def class_probability_matrix(
    model: MLP,
    params: Params,
    X_aux: np.ndarray,
    y_aux: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """FedMADE's Class Probability Matrix (Eq. 1) -- VERIFIED.

    Row ``c`` is the mean softmax output over the auxiliary rows of class
    ``c``: ``P_bar_c = (1/n_c) sum_{j=1..n_c} p(x_{c,j})``. The result is
    ``(n_classes, n_classes)``; a perfect classifier yields the identity, which
    is exactly what Eq. 3 optimizes towards.

    A class with no auxiliary rows gets an all-zero row. The paper does not
    consider this case (it assumes ten rows per class exist); a zero row is the
    choice that keeps the Eq. 3 residual well-defined and simply contributes a
    constant ``-I`` term for that class, identically for every cluster, so it
    cannot bias the ``alpha`` comparison between clusters.
    """
    model.set_params(params)
    probs, _ = model.forward(X_aux)
    cpm = np.zeros((n_classes, n_classes), dtype=np.float64)
    for c in range(n_classes):
        mask = y_aux == c
        if np.any(mask):
            cpm[c] = probs[mask].mean(axis=0)
    return cpm


def dbscan_cluster_cpms(
    cpms: Sequence[np.ndarray],
    eps: Optional[float] = None,
    min_samples: int = 1,
) -> np.ndarray:
    """DBSCAN over flattened CPMs -- VERIFIED step, INFERRED parameters.

    Verified: "the server applies the DBSCAN clustering technique to the
    obtained CPMs" (Section 4.2). The paper states neither ``eps`` nor
    ``min_samples``.

    INFERRED choices, both chosen to be defensible rather than tuned:

    * ``min_samples=1`` -- with ``min_samples>1`` DBSCAN labels sparse points
      as noise (``-1``), and FedMADE's Algorithm 1 has no branch for a client
      that belongs to no cluster; ``min_samples=1`` makes every client belong
      to some cluster (a singleton at worst), so the algorithm as printed is
      always well-defined. Under this setting a singleton cluster gets
      ``beta_i = alpha_k`` directly, which is the natural degenerate case of
      Eq. 4.
    * ``eps=None`` -> the **median pairwise Euclidean distance** between the
      flattened CPMs. A fixed absolute ``eps`` is meaningless here because CPM
      scale depends on the class count; the median distance is scale-free and
      makes "clients whose CPMs are unusually close" the thing that merges,
      which is what "cluster devices by their traffic patterns" describes.
      Pass an explicit ``eps`` to override.

    Returns an integer label per client.
    """
    from sklearn.cluster import DBSCAN

    flat = np.stack([np.asarray(c, dtype=np.float64).ravel() for c in cpms], axis=0)
    n = flat.shape[0]
    if n == 1:
        return np.zeros(1, dtype=int)

    if eps is None:
        diff = flat[:, None, :] - flat[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        iu = np.triu_indices(n, k=1)
        eps = float(np.median(dist[iu]))
        if not np.isfinite(eps) or eps <= 0.0:
            # Every CPM identical (e.g. round 0, or a collapsed model): any
            # positive eps puts them all in one cluster, which is correct.
            eps = 1e-12

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(flat).labels_
    # min_samples=1 cannot produce noise, but an explicit min_samples can.
    # Give each noise point its own singleton cluster rather than dropping it.
    labels = np.asarray(labels, dtype=int).copy()
    next_id = labels.max() + 1 if labels.size and labels.max() >= 0 else 0
    for i in np.flatnonzero(labels < 0):
        labels[i] = next_id
        next_id += 1
    return labels


def solve_cluster_weights(
    ccpms: Sequence[np.ndarray],
    n_steps: int = 200,
    lr: float = 0.01,
) -> np.ndarray:
    """FedMADE Eq. 3: ``min_alpha ||sum_k alpha_k P_{C_k} - I||_2``, ``alpha_k >= 0``.

    VERIFIED: the objective, the non-negativity constraint, the absence of a
    sum-to-one constraint at this stage (normalization happens later, on
    ``beta``), and the solver -- "the Adam optimizer with a learning rate of
    0.01" (Section 5.1).

    INFERRED (two points, both stated in the paper only implicitly):

    * ``||.||_2`` on a matrix is read as the **Frobenius** norm. The spectral
      norm is the other reading of that symbol; Frobenius is the standard
      reading of "L2 distance" between matrices, is what makes the objective
      separable over classes, and is what a first-order Adam solver can
      descend without an SVD per step.
    * The **squared** Frobenius norm is minimized. This is not an
      approximation: ``||R||_F`` and ``||R||_F^2`` have the same minimizer over
      the same feasible set; squaring only removes the non-differentiable point
      at ``R = 0``.
    * ``n_steps=200`` -- the paper gives the optimizer and its learning rate
      but not the iteration count.

    Non-negativity is enforced by clipping ``alpha`` at 0 after every Adam step
    (projected gradient), matching the constraint as written.
    """
    P = np.stack([np.asarray(c, dtype=np.float32) for c in ccpms], axis=0)  # (K,C,C)
    K, C, _ = P.shape
    I = np.eye(C, dtype=np.float32)

    alpha = np.full(K, 1.0 / K, dtype=np.float32)
    opt = Adam(lr=lr)
    for _ in range(n_steps):
        R = np.tensordot(alpha, P, axes=(0, 0)) - I          # (C,C) residual
        grad = 2.0 * np.tensordot(P, R, axes=([1, 2], [0, 1]))  # d||R||_F^2/d alpha
        opt.step([alpha], [grad.astype(np.float32)])
        np.clip(alpha, 0.0, None, out=alpha)
    return alpha.astype(np.float64)


def fedmade_weights(
    model: MLP,
    client_params: Sequence[Params],
    X_aux: np.ndarray,
    y_aux: np.ndarray,
    n_classes: int,
    client_sizes: Sequence[int],
    eps: Optional[float] = None,
    min_samples: int = 1,
    alpha_steps: int = 200,
    alpha_lr: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """One round of FedMADE's server-side weighting. Returns ``(beta, labels)``.

    Algorithm 1 lines 6-14, VERIFIED end to end: CPM per client (Eq. 1),
    DBSCAN over CPMs, cluster CPM (Eq. 2), cluster weights (Eq. 3),
    ``beta_i = alpha_k / |C_k|`` (Eq. 4), then ``beta_i <- beta_i / sum beta``
    (line 14).

    Degenerate fallback: if every ``alpha_k`` clips to 0 the normalization in
    line 14 is ``0/0``. The paper does not cover this. We fall back to FedAvg
    weights and say so, rather than emitting NaNs into the aggregation.
    """
    cpms = [
        class_probability_matrix(model, p, X_aux, y_aux, n_classes)
        for p in client_params
    ]
    labels = dbscan_cluster_cpms(cpms, eps=eps, min_samples=min_samples)

    uniq = np.unique(labels)
    ccpms = [np.mean([cpms[i] for i in np.flatnonzero(labels == k)], axis=0)
             for k in uniq]                                     # Eq. 2
    alpha = solve_cluster_weights(ccpms, n_steps=alpha_steps, lr=alpha_lr)  # Eq. 3

    beta = np.zeros(len(client_params), dtype=np.float64)
    for k_idx, k in enumerate(uniq):                            # Eq. 4
        members = np.flatnonzero(labels == k)
        beta[members] = alpha[k_idx] / len(members)

    total = beta.sum()
    if not np.isfinite(total) or total <= 0.0:
        return fedavg_weights(client_sizes).astype(np.float64), labels
    return (beta / total), labels


class FedMADETrainer:
    """FedMADE baseline (Sun, Sharma, Nwodo, Stavrou & Wang, ISC 2024;
    arXiv:2408.07152) -- VERIFIED against the arXiv full text.

    Per round: clients train locally; the server evaluates every returned model
    on a small balanced auxiliary set to build a Class Probability Matrix,
    clusters the CPMs with DBSCAN (this is FedMADE's "cluster devices by their
    traffic patterns" -- note that the clustering acts on *model behaviour*,
    not on raw label counts), averages CPMs within a cluster, solves a
    non-negative least-squares problem pushing the weighted cluster CPMs toward
    the identity matrix, and splits each cluster's weight equally among its
    members before normalizing.

    This makes FedMADE a **measured-contribution** method: a client's weight is
    determined by how close its cluster's *measured* class-conditional
    behaviour is to perfect classification. Contrast SBFedNIDS (weights follow
    from data-level rebalancing, no model evaluation), FedAPA (weights follow
    from parameter-space geometry, no validation data), and FedGRAA (weights
    follow from the gradient of a stratified composite validation loss).

    The paper's headline result -- up to 71.07% improvement in minority attack
    classification accuracy -- is a claim about the original paper's setting,
    not about this reproduction.

    Constructor arguments beyond ``split``/``cfg`` correspond one-to-one to the
    ``INFERRED`` items documented on the helper functions above.
    """

    def __init__(
        self,
        split: FederatedSplit,
        cfg: FedConfig,
        aux_per_class: int = 10,
        dbscan_eps: Optional[float] = None,
        dbscan_min_samples: int = 1,
        alpha_steps: int = 200,
        alpha_lr: float = 0.01,
    ):
        self.split = split
        self.cfg = cfg
        self.spec = split.spec
        self.aux_per_class = aux_per_class
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.alpha_steps = alpha_steps
        self.alpha_lr = alpha_lr
        self.rare_class_ids = self.spec.rare_client_ids(cfg.n_rare)
        self.client_sizes = [c.n_train for c in split.clients]
        layer_sizes = self.spec.model_config().layer_sizes
        self._model = MLP(layer_sizes, seed=cfg.seed)
        self._client_tf_optimizers: Dict[int, tf.keras.optimizers.Adam] = {
            c.client_id: build_tf_optimizer(
                cfg.eta, self._model._model.trainable_variables
            )
            for c in split.clients
        }
        self.X_aux, self.y_aux = build_server_auxiliary_set(
            split.X_val, split.y_val, self.spec.n_classes,
            per_class=aux_per_class, seed=cfg.seed,
        )
        # Instrumentation: DBSCAN labels at each aggregation round.
        self.cluster_labels_history: List[np.ndarray] = []

    def train(self, eval_every: int = 20, verbose: bool = False) -> TrainHistory:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        global_params: Params = init_params(self._model.layer_sizes, seed=cfg.seed)

        w = fedavg_weights(self.client_sizes)
        history = TrainHistory(method="fedmade", gamma=None, n_rare=cfg.n_rare)
        history.record(
            0,
            accuracy(self._model, global_params, self.split.X_val, self.split.y_val),
            last_n_class_accuracy(
                self._model, global_params,
                self.split.X_val, self.split.y_val, self.rare_class_ids,
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
            w, labels = fedmade_weights(
                self._model, client_params,
                self.X_aux, self.y_aux, self.spec.n_classes,
                self.client_sizes,
                eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples,
                alpha_steps=self.alpha_steps,
                alpha_lr=self.alpha_lr,
            )
            self.cluster_labels_history.append(labels)
            global_params = aggregate(client_params, w)

            if rnd % eval_every == 0 or rnd == cfg.rounds:
                all_a = accuracy(
                    self._model, global_params, self.split.X_val, self.split.y_val
                )
                last_a = last_n_class_accuracy(
                    self._model, global_params,
                    self.split.X_val, self.split.y_val, self.rare_class_ids,
                )
                history.record(rnd, all_a, last_a, w)
                if verbose:
                    print(
                        f"  [fedmade] round {rnd:3d}/{cfg.rounds}"
                        f"  all={all_a:.3f}  last-n={last_a:.3f}"
                        f"  clusters={len(np.unique(labels))}",
                        flush=True,
                    )

        self.final_params = global_params
        return history


# =========================================================================== #
# SBFedNIDS (ICASSP 2025, DOI 10.1109/ICASSP49660.2025.10889171)
# RECONSTRUCTION -- full text paywalled, see the module docstring.
# =========================================================================== #
def federated_cost_weights(
    global_counts: np.ndarray, n_classes: int
) -> np.ndarray:
    """FCSL cost weights from the **global** (federated) class histogram.

    RECONSTRUCTION, not a reproduction. What is verified from the abstract is
    only that SBFedNIDS applies "federated cost sensitive learning (FCSL)" at
    the data level, and that one of the three imbalances it names is *global*
    imbalance -- an imbalance no client can see in its own shard. The
    implication we act on is the one thing the name pins down: the cost weights
    must come from the pooled class histogram (a SecAgg-summable statistic),
    not from any client's local counts. That is what makes it *federated* cost
    sensitivity rather than ordinary local cost sensitivity.

    The specific cost function is **not** verified -- the abstract states no
    equation. We use the standard inverse-frequency ("balanced") weighting,
    ``c_k = N / (n_present * count_k)``, rescaled to mean 1 over the classes
    that are present so the effective learning rate is unchanged relative to
    unweighted training. Absent classes get weight 0 (they contribute no rows).
    A different published cost function would change the numbers.
    """
    counts = np.asarray(global_counts, dtype=np.float64)
    if len(counts) < n_classes:
        counts = np.concatenate([counts, np.zeros(n_classes - len(counts))])
    counts = counts[:n_classes]
    present = counts > 0
    w = np.zeros(n_classes, dtype=np.float64)
    if not np.any(present):
        return np.ones(n_classes, dtype=np.float64)
    total = counts.sum()
    w[present] = total / (present.sum() * counts[present])
    w[present] /= w[present].mean()          # mean-1 rescaling
    return w


def client_smote(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    rng: np.random.Generator,
    k_neighbors: int = 5,
    max_expansion: float = 5.0,
    nn_pool_cap: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """CSMOTE: local minority over-sampling on one client's own shard.

    RECONSTRUCTION, not a reproduction. Verified from the abstract: SBFedNIDS
    uses a "client synthetic minority over-sampling technique (CSMOTE)" at the
    data level and explicitly *without* generative models. Not verified: the
    target ratio, the neighbour count, or whether CSMOTE differs from textbook
    SMOTE in any way beyond being run per client. What is implemented is
    textbook SMOTE (Chawla, Bowyer, Hall & Kegelmeyer, JAIR 2002) applied to
    each client's own training rows: for each minority class, repeatedly pick a
    real minority row, pick one of its ``k`` nearest same-class neighbours, and
    emit ``x + lambda (x_nn - x)`` with ``lambda ~ U(0,1)``, until the class
    reaches the client's local majority count.

    Guards that are ours, not the paper's: ``max_expansion`` caps total
    synthesis at that multiple of the client's original row count (a client
    holding one minority row and 10,000 majority rows would otherwise
    synthesize 10,000 near-duplicates of a single point), and ``nn_pool_cap``
    subsamples the neighbour pool for the exact k-NN search so the pairwise
    distance matrix stays bounded. Both are documented rather than silent.

    A class with a single row has no neighbour to interpolate towards; that row
    is duplicated instead, which is what SMOTE degenerates to at ``m = 1``.
    """
    n = len(y)
    if n == 0:
        return X, y
    counts = np.bincount(y, minlength=n_classes)
    target = int(counts.max())
    budget = int(round(max_expansion * n))

    new_X: List[np.ndarray] = []
    new_y: List[np.ndarray] = []
    for c in range(n_classes):
        m = int(counts[c])
        if m == 0 or m >= target or budget <= 0:
            continue
        need = min(target - m, budget)
        budget -= need

        idx_c = np.flatnonzero(y == c)
        if m == 1:
            new_X.append(np.repeat(X[idx_c], need, axis=0))
            new_y.append(np.full(need, c, dtype=y.dtype))
            continue

        pool = idx_c
        if m > nn_pool_cap:
            pool = rng.choice(idx_c, size=nn_pool_cap, replace=False)
        Xp = np.asarray(X[pool], dtype=np.float64)
        d = np.sqrt(
            np.maximum(
                (Xp * Xp).sum(1)[:, None] + (Xp * Xp).sum(1)[None, :]
                - 2.0 * Xp @ Xp.T,
                0.0,
            )
        )
        np.fill_diagonal(d, np.inf)
        k = min(k_neighbors, len(pool) - 1)
        nn = np.argsort(d, axis=1, kind="stable")[:, :k]        # (P, k)

        base = rng.integers(0, len(pool), size=need)
        pick = rng.integers(0, k, size=need)
        lam = rng.random(size=(need, 1))
        x_base = Xp[base]
        x_nn = Xp[nn[base, pick]]
        new_X.append((x_base + lam * (x_nn - x_base)).astype(X.dtype))
        new_y.append(np.full(need, c, dtype=y.dtype))

    if not new_X:
        return X, y
    return (
        np.concatenate([X] + new_X, axis=0),
        np.concatenate([y] + new_y, axis=0),
    )


@tf.function(jit_compile=True, reduce_retracing=True)
def _fused_cost_sensitive_step(
    x: tf.Tensor, y: tf.Tensor, model: tf.keras.Model,
    opt: tf.keras.optimizers.Optimizer,
    class_w: tf.Tensor, n_valid: tf.Tensor,
) -> tf.Tensor:
    """One fused client step with per-class cost weights (FCSL).

    Mirrors ``fedgraa.model._fused_client_step`` exactly -- same padding mask,
    same out-of-range label guard -- except that each row's cross-entropy is
    multiplied by ``class_w[y]`` and the batch loss is the cost-weighted mean
    ``sum(w_i l_i) / sum(w_i)`` rather than the plain mean.
    """
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        n_classes = tf.shape(logits)[-1]
        labels_safe = tf.where(
            tf.logical_or(y < 0, y >= tf.cast(n_classes, y.dtype)),
            tf.zeros_like(y), y,
        )
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels_safe, logits=logits
        )
        batch_size = tf.shape(x)[0]
        mask = tf.cast(tf.range(batch_size) < n_valid, dtype=losses.dtype)
        row_w = tf.gather(class_w, labels_safe) * mask
        denom = tf.maximum(tf.reduce_sum(row_w), tf.constant(1e-12, losses.dtype))
        loss = tf.reduce_sum(losses * row_w) / denom
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def cost_sensitive_client_train_round(
    model: MLP,
    opt: tf.keras.optimizers.Optimizer,
    X: np.ndarray,
    y: np.ndarray,
    cfg: FedConfig,
    global_weights: Params,
    rng: np.random.Generator,
    class_w: np.ndarray,
) -> Params:
    """``client_train_round`` with FCSL's cost-weighted objective."""
    model.set_params(global_weights)
    n = len(y)
    if n == 0:
        return model.get_params()

    cw = tf.constant(np.asarray(class_w, dtype=np.float32), dtype=tf.float32)
    pad_x = np.zeros((cfg.batch_size, X.shape[1]), dtype=np.float32)
    pad_y = np.zeros(cfg.batch_size, dtype=np.int32)

    all_perm = np.array(
        [rng.permutation(n) for _ in range(cfg.local_epochs)], dtype=np.int32
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
            _fused_cost_sensitive_step(
                tf.constant(x_batch, dtype=tf.float32),
                tf.constant(y_batch, dtype=tf.int32),
                model._model, opt, cw,
                tf.constant(n_valid, dtype=tf.int32),
            )
    return model.get_params()


class SBFedNIDSTrainer:
    """SBFedNIDS baseline (Zeng, Zheng & Li, ICASSP 2025,
    DOI 10.1109/ICASSP49660.2025.10889171).

    ⚠️ **Best-faith RECONSTRUCTION from the published abstract, not a verified
    reproduction.** The full text is behind IEEE Xplore's paywall with no
    open-access or preprint copy (see the module docstring, which quotes the
    entire verified abstract and records the search that established this). The
    two component *names* and the fact that both act at the data level are
    verified; their equations are not. Do not describe this class as a faithful
    reproduction in any results table -- describe it as a reconstruction of
    FCSL + CSMOTE.

    Mechanism as implemented, per round:

    1. **FCSL** (once, at construction): the server sums the per-client class
       histograms into a global histogram and derives inverse-frequency cost
       weights (:func:`federated_cost_weights`). Every client then trains with
       a cost-weighted cross-entropy. This is the only component that can see
       *global* imbalance, which no client observes locally.
    2. **CSMOTE** (once, at construction): each client SMOTE-oversamples its own
       minority classes up to its local majority count
       (:func:`client_smote`), addressing *heterogeneous* (local) imbalance.
    3. **Aggregation**: data-size weighting over the **post-CSMOTE** client
       sizes. As the module docstring records, the abstract describes no
       aggregation rule at all, so none is invented here; the weight vector
       nevertheless differs from FedAvg's, because CSMOTE changed the sizes it
       is computed from. That shift *is* the self-balancing effect on *size*
       imbalance -- the third imbalance the abstract names -- and it is a
       consequence of step 2 rather than an extra assumption.

    Note on this harness: ``partition_non_iid`` gives every client one attack
    class plus an equal number of benign rows, so local label distributions are
    close to 50/50 by construction and CSMOTE is nearly inert on the standard
    split -- FCSL carries almost all of the effect there. CSMOTE becomes active
    under ``partition_by_silo`` or any deliberately skewed partition. This is a
    property of the partition, not a defect of the reconstruction, but it means
    a null result on the standard split should not be read as "CSMOTE does not
    work".
    """

    def __init__(
        self,
        split: FederatedSplit,
        cfg: FedConfig,
        k_neighbors: int = 5,
        max_expansion: float = 5.0,
        use_csmote: bool = True,
        use_fcsl: bool = True,
    ):
        self.split = split
        self.cfg = cfg
        self.spec = split.spec
        self.use_csmote = use_csmote
        self.use_fcsl = use_fcsl
        self.rare_class_ids = self.spec.rare_client_ids(cfg.n_rare)
        self.client_sizes = [c.n_train for c in split.clients]
        layer_sizes = self.spec.model_config().layer_sizes
        self._model = MLP(layer_sizes, seed=cfg.seed)
        self._client_tf_optimizers: Dict[int, tf.keras.optimizers.Adam] = {
            c.client_id: build_tf_optimizer(
                cfg.eta, self._model._model.trainable_variables
            )
            for c in split.clients
        }

        C = self.spec.n_classes
        # --- FCSL: global (SecAgg-summable) class histogram ------------------
        self.global_counts = np.zeros(C, dtype=np.int64)
        for client in split.clients:
            self.global_counts += np.bincount(client.y_train, minlength=C)[:C]
        self.class_weights = (
            federated_cost_weights(self.global_counts, C)
            if use_fcsl else np.ones(C, dtype=np.float64)
        )

        # --- CSMOTE: local rebalancing, once, before training ---------------
        rng = np.random.default_rng(cfg.seed)
        self.rebalanced: List[Tuple[np.ndarray, np.ndarray]] = []
        for client in split.clients:
            if use_csmote:
                Xr, yr = client_smote(
                    client.X_train, client.y_train, C, rng,
                    k_neighbors=k_neighbors, max_expansion=max_expansion,
                )
            else:
                Xr, yr = client.X_train, client.y_train
            self.rebalanced.append((Xr, yr))
        self.rebalanced_sizes = [len(yr) for _, yr in self.rebalanced]

    def aggregation_weights(self) -> np.ndarray:
        """Data-size weights over the post-CSMOTE client sizes (step 3)."""
        return fedavg_weights(self.rebalanced_sizes)

    def train(self, eval_every: int = 20, verbose: bool = False) -> TrainHistory:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        global_params: Params = init_params(self._model.layer_sizes, seed=cfg.seed)

        w = self.aggregation_weights()
        history = TrainHistory(method="sbfednids", gamma=None, n_rare=cfg.n_rare)
        history.record(
            0,
            accuracy(self._model, global_params, self.split.X_val, self.split.y_val),
            last_n_class_accuracy(
                self._model, global_params,
                self.split.X_val, self.split.y_val, self.rare_class_ids,
            ),
            w,
        )

        for rnd in range(1, cfg.rounds + 1):
            client_params = [
                cost_sensitive_client_train_round(
                    self._model,
                    self._client_tf_optimizers[client.client_id],
                    Xr, yr, cfg, global_params, rng, self.class_weights,
                )
                for client, (Xr, yr) in zip(self.split.clients, self.rebalanced)
            ]
            w = self.aggregation_weights()
            global_params = aggregate(client_params, w)

            if rnd % eval_every == 0 or rnd == cfg.rounds:
                all_a = accuracy(
                    self._model, global_params, self.split.X_val, self.split.y_val
                )
                last_a = last_n_class_accuracy(
                    self._model, global_params,
                    self.split.X_val, self.split.y_val, self.rare_class_ids,
                )
                history.record(rnd, all_a, last_a, w)
                if verbose:
                    print(
                        f"  [sbfednids] round {rnd:3d}/{cfg.rounds}"
                        f"  all={all_a:.3f}  last-n={last_a:.3f}",
                        flush=True,
                    )

        self.final_params = global_params
        return history


# =========================================================================== #
# FedAPA (arXiv:2502.07456) -- VERIFIED except one sign convention
# =========================================================================== #
def _dot_params(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> float:
    """Flat inner product between two parameter lists."""
    return float(sum(np.sum(ai.astype(np.float64) * bi.astype(np.float64))
                     for ai, bi in zip(a, b)))


def fedapa_weight_row_update(
    a_row: np.ndarray,
    self_index: int,
    client_extractors: Sequence[Params],
    delta: Params,
    eta: float,
    update_sign: float = 1.0,
) -> np.ndarray:
    """FedAPA Eq. 7 plus Algorithm 1 lines 11-13, for one client's weight row.

    VERIFIED pieces (arXiv:2502.07456):

    * Eq. 5 ``theta_bar_i = sum_j a_ij theta_j``, so the Jacobian
      ``d theta_bar_i / d a_ij`` is exactly ``theta_j``. Hence Eq. 7's
      ``(grad_{A_i} theta_bar_i)^T delta_theta_i`` has ``j``-th entry
      ``<theta_j, delta_theta_i>`` -- the inner product of client ``j``'s
      uploaded feature extractor with client ``i``'s parameter change. Note
      that the inner product is against the raw uploaded parameters, not
      against a parameter *difference*; that is what the paper writes.
    * Eq. 6 ``delta_theta_i = theta_i - theta_bar_i``, computed after local
      training.
    * Line 11 ``a_ij <- min(max(a_ij, 0), 1)``, line 12 ``a_ii <- 0.5``,
      line 13 "Normalize A_i", **in that order**.

    ⚠️ **The sign of Eq. 7 is the one place this implementation departs from
    the printed equation, deliberately and visibly.** The paper prints
    ``A_i <- A_i - eta (grad_{A_i} theta_bar_i)^T delta_theta_i``, but it also
    states that "half of the squared norm of the client-parameter delta serves
    as a proxy for the local loss" and that this proxy is *minimized* over
    ``A_i``. Differentiating that stated proxy, ``L = 0.5 ||theta_i -
    theta_bar_i||^2`` with ``theta_i`` held fixed, gives
    ``dL/da_ij = -<theta_j, delta_theta_i>``, so gradient *descent* on the
    paper's own proxy is ``a_ij <- a_ij + eta <theta_j, delta_theta_i>``. The
    chain-rule route reaches the same place: local SGD moved the client from
    ``theta_bar_i`` to ``theta_i``, so ``grad_{theta_bar_i} L_i`` is
    approximately ``-delta_theta_i / lr``, and descending gives the same plus
    sign. The plus sign is also the only one with a sensible reading -- it
    raises ``a_ij`` when client ``j``'s parameters point where client ``i``'s
    local training wants to go. We therefore take the printed minus to be a
    sign convention folded into how the paper writes ``grad``, and default to
    ``update_sign=+1``. Pass ``update_sign=-1.0`` to run the literal
    transcription instead; the flag exists so this judgement is reversible and
    testable rather than buried.

    INFERRED: "Normalize ``A_i``" is read as dividing the row by its sum, which
    (with line 11's clipping) lands the row exactly on the probability simplex.
    The paper does not say whether it means sum-to-one or something that
    preserves line 12's ``a_ii = 0.5`` exactly; sum-to-one is the literal
    reading of line 13 as printed, and is the reading that makes the row a
    valid aggregation weight vector. Under it, line 12 fixes the *ratio* of
    self-weight to peer weight rather than the final self-weight.

    Degenerate case: if the row sums to zero after clipping (possible only if
    ``a_ii`` is somehow zeroed), a uniform row is returned rather than NaNs.
    """
    a = np.asarray(a_row, dtype=np.float64).copy()
    grad = np.array(
        [_dot_params(theta_j, delta) for theta_j in client_extractors],
        dtype=np.float64,
    )
    a = a + update_sign * eta * grad              # Eq. 7
    np.clip(a, 0.0, 1.0, out=a)                   # Algorithm 1, line 11
    a[self_index] = 0.5                           # Algorithm 1, line 12
    total = a.sum()                               # Algorithm 1, line 13
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(a), 1.0 / len(a))
    return a / total


class FedAPATrainer:
    """FedAPA baseline (Sun, Sun, Pan, Fu & Guo, IJCAI 2025; arXiv:2502.07456)
    -- VERIFIED against the arXiv full text, with the one sign convention
    documented at :func:`fedapa_weight_row_update`.

    **This project's literature review flags FedAPA as FedGRAA's single most
    mechanistically similar competitor**, and therefore as the most important
    missing comparison in the paper. Both learn aggregation weights on the
    server by gradient descent. The differences that matter for the comparison:

    ===================  ==============================  =====================
    .                    FedGRAA                         FedAPA
    ===================  ==============================  =====================
    objective            composite validation loss       proxy for the local
                         ``(1-g) min(L_h,L_m) + g L_u``  loss, ``0.5||dtheta||^2``
    data needed          stratified server-side ``D_V``  none (parameter space
                                                         only)
    output               one global weight vector        one weight row per
                                                         client (M x M matrix)
    what is shared       the whole model                 feature extractor only;
                                                         decision layer stays
                                                         private
    goal                 rare-class recall on a global   per-client
                         model                           personalization
    ===================  ==============================  =====================

    Per round (Algorithm 1):

    1. Server forms each client's personalized extractor ``theta_bar_i =
       sum_j a_ij theta_j`` (Eq. 5) and sends it down; the client attaches its
       own retained decision layer.
    2. The client trains the full model locally for ``E`` epochs and uploads
       **only** the extractor ``theta_i``; the decision layer is kept.
    3. Server computes ``delta_theta_i = theta_i - theta_bar_i`` (Eq. 6) and
       updates row ``A_i`` (Eq. 7), then clips / sets ``a_ii = 0.5`` /
       normalizes (lines 11-13).

    Initialization is the paper's: ``a_ii = 1``, ``a_ij = 0`` for ``j != i``.

    **Reporting convention (a judgement call, made explicit).** FedAPA has no
    single global model, so there is nothing to put in ``TrainHistory`` that is
    literally comparable to the other methods' global-model accuracy. What is
    recorded is the **mean over clients of each personalized model's accuracy
    on the shared server validation set** ``D_V`` -- FedAPA's own protocol
    (evaluate the personalized models) applied to this harness's shared
    evaluation set so the number sits on the same axis as every other baseline.
    ``final_params`` is set to the consensus aggregate over the mean weight
    row, so downstream code expecting one parameter set still works, and
    ``consensus_all_acc`` / ``consensus_last_n_acc`` record that consensus
    model's accuracy separately. ``weights`` in the history is the column mean
    of ``A`` (each client's average influence over all clients) -- a valid
    simplex point, since it is a mean of simplex points. The full matrix per
    recorded round is kept in ``weight_matrix_history``.

    The server-side weight learning rate ``eta`` is listed as an input in
    Algorithm 1 with no stated default; ``server_lr`` exposes it (default
    ``0.01``).
    """

    def __init__(
        self,
        split: FederatedSplit,
        cfg: FedConfig,
        server_lr: float = 0.01,
        update_sign: float = 1.0,
        n_head_arrays: int = 2,
    ):
        self.split = split
        self.cfg = cfg
        self.spec = split.spec
        self.server_lr = server_lr
        self.update_sign = update_sign
        # Params are [W1, b1, ..., WL, bL]; the decision layer h(phi_i) is the
        # final Dense layer, i.e. the last two arrays. Everything before it is
        # the feature extractor f(theta_i), the only part FedAPA uploads.
        self.n_head_arrays = n_head_arrays
        self.rare_class_ids = self.spec.rare_client_ids(cfg.n_rare)
        self.client_sizes = [c.n_train for c in split.clients]
        layer_sizes = self.spec.model_config().layer_sizes
        self._model = MLP(layer_sizes, seed=cfg.seed)
        self._client_tf_optimizers: Dict[int, tf.keras.optimizers.Adam] = {
            c.client_id: build_tf_optimizer(
                cfg.eta, self._model._model.trainable_variables
            )
            for c in split.clients
        }
        M = len(split.clients)
        self.A = np.eye(M, dtype=np.float64)   # a_ii = 1, a_ij = 0 (paper's init)
        self.weight_matrix_history: List[np.ndarray] = []
        self.consensus_all_acc: List[float] = []
        self.consensus_last_n_acc: List[float] = []

    # -- extractor / head split --------------------------------------------- #
    def _extractor(self, params: Params) -> Params:
        return list(params[: len(params) - self.n_head_arrays])

    def _head(self, params: Params) -> Params:
        return list(params[len(params) - self.n_head_arrays :])

    def mean_weight_vector(self) -> np.ndarray:
        """Column mean of ``A``: each client's average influence. A simplex point."""
        return self.A.mean(axis=0)

    def _personalized_metrics(
        self, client_full_params: Sequence[Params]
    ) -> Tuple[float, float]:
        accs, rares = [], []
        for p in client_full_params:
            accs.append(
                accuracy(self._model, p, self.split.X_val, self.split.y_val)
            )
            rares.append(
                last_n_class_accuracy(
                    self._model, p,
                    self.split.X_val, self.split.y_val, self.rare_class_ids,
                )
            )
        return float(np.mean(accs)), float(np.mean(rares))

    def train(self, eval_every: int = 20, verbose: bool = False) -> TrainHistory:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        init = init_params(self._model.layer_sizes, seed=cfg.seed)

        M = len(self.split.clients)
        # Uploaded extractors and retained (private) decision layers.
        extractors: List[Params] = [
            [p.copy() for p in self._extractor(init)] for _ in range(M)
        ]
        heads: List[Params] = [
            [p.copy() for p in self._head(init)] for _ in range(M)
        ]

        w = self.mean_weight_vector()
        history = TrainHistory(method="fedapa", gamma=None, n_rare=cfg.n_rare)
        history.record(
            0,
            accuracy(self._model, init, self.split.X_val, self.split.y_val),
            last_n_class_accuracy(
                self._model, init,
                self.split.X_val, self.split.y_val, self.rare_class_ids,
            ),
            w,
        )
        self.weight_matrix_history.append(self.A.copy())
        self.consensus_all_acc.append(history.all_acc[0])
        self.consensus_last_n_acc.append(history.last_n_acc[0])

        client_full: List[Params] = []
        for rnd in range(1, cfg.rounds + 1):
            # --- 1. personalized aggregation (Eq. 5), per client ------------
            bars: List[Params] = [
                aggregate(extractors, self.A[i].astype(np.float32))
                for i in range(M)
            ]

            # --- 2. local training on (theta_bar_i, phi_i) ------------------
            new_extractors: List[Params] = []
            client_full = []
            for i, client in enumerate(self.split.clients):
                start = list(bars[i]) + list(heads[i])
                trained = client_train_round(
                    self._model,
                    self._client_tf_optimizers[client.client_id],
                    client.X_train, client.y_train,
                    cfg, start, rng,
                )
                new_extractors.append([p.copy() for p in self._extractor(trained)])
                heads[i] = [p.copy() for p in self._head(trained)]  # stays private
                client_full.append([p.copy() for p in trained])

            # --- 3. server-side weight-row update (Eq. 6-7, lines 11-13) ----
            A_new = np.empty_like(self.A)
            for i in range(M):
                delta = [
                    ti - bi for ti, bi in zip(new_extractors[i], bars[i])
                ]                                                   # Eq. 6
                A_new[i] = fedapa_weight_row_update(
                    self.A[i], i, new_extractors, delta,
                    eta=self.server_lr, update_sign=self.update_sign,
                )
            self.A = A_new
            extractors = new_extractors
            w = self.mean_weight_vector()

            if rnd % eval_every == 0 or rnd == cfg.rounds:
                all_a, last_a = self._personalized_metrics(client_full)
                history.record(rnd, all_a, last_a, w)
                self.weight_matrix_history.append(self.A.copy())
                consensus = aggregate(
                    [list(e) + list(h) for e, h in zip(extractors, heads)],
                    w.astype(np.float32),
                )
                self.consensus_all_acc.append(
                    accuracy(self._model, consensus,
                             self.split.X_val, self.split.y_val)
                )
                self.consensus_last_n_acc.append(
                    last_n_class_accuracy(
                        self._model, consensus,
                        self.split.X_val, self.split.y_val, self.rare_class_ids,
                    )
                )
                if verbose:
                    print(
                        f"  [fedapa] round {rnd:3d}/{cfg.rounds}"
                        f"  mean-personalized all={all_a:.3f}  last-n={last_a:.3f}",
                        flush=True,
                    )

        self.final_client_params = client_full
        self.final_heads = heads
        self.final_extractors = extractors
        self.final_params = aggregate(
            [list(e) + list(h) for e, h in zip(extractors, heads)],
            self.mean_weight_vector().astype(np.float32),
        )
        return history
