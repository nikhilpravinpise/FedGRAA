"""Tier-3 robust-aggregation baselines: FLTrust, RAGA, and FedTruth.

Three *server-side robust aggregation* baselines. All three replace FedAvg's
data-size weighting with a weight vector derived from the geometry of the
submitted model updates, but -- and this is the whole point of running them
side by side -- they derive it in three structurally different ways, and only
one of them uses a server-held reference direction at all.

Methods and citations
---------------------
* :class:`FLTrustTrainer` -- Xiaoyu Cao, Minghong Fang, Jia Liu, Neil Zhenqiang
  Gong. "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping."
  NDSS 2021. arXiv:2012.13995.
* :class:`RAGATrainer` -- Shiyuan Zuo, Xingrun Yan, Rongfei Fan, Han Hu,
  Hangguan Shan, Tony Q. S. Quek, Puning Zhao. "Federated Learning Resilient to
  Byzantine Attacks and Data Heterogeneity" (RAGA, Robust Average Gradient
  Algorithm). IEEE Transactions on Mobile Computing, 2025. arXiv:2403.13374.
* :class:`FedTruthTrainer` -- Sheldon C. Ebron, Meiying Zhang, Kan Yang.
  "Identifying the Truth of Global Model: A Generic Solution to Defend Against
  Byzantine and Backdoor Attacks in Federated Learning." ACISP 2025.
  arXiv:2311.10248.

Verification status
-------------------
**All three mechanisms below were read off the papers' own full text**, not
inferred from abstracts or from secondary descriptions:

* FLTrust -- full text via the ar5iv HTML render of arXiv:2012.13995.
  Algorithm 1 (``ModelUpdate``), the trust score ``TS_i = ReLU(c_i)``, the
  magnitude normalization, and the aggregation rule were all read verbatim.
* RAGA -- full text via the arXiv HTML render of arXiv:2403.13374v3.
  Eqs. (13)-(16) and Algorithm 1 were read verbatim.
* FedTruth -- full text via the arXiv HTML render of arXiv:2311.10248v3.
  Eq. (1) (the optimization objective) and Sec. 3.3-3.4 (the coordinate-descent
  solution, the coefficient-function requirements, and the two worked examples
  of ``g``) were read verbatim.

Where the papers are genuinely silent, the gap is named explicitly at the point
of the choice -- see "Documented deviations" on each class. Nothing below is
presented as verified that was not actually read.

**Correction to a common mis-grouping.** These three are often lumped together
as "FLTrust and its descendants". That is wrong for two of them, and the code
reflects the papers rather than the folklore:

* Only **FLTrust** uses a server-held root dataset and a cosine-similarity
  trust score against a server reference direction.
* **RAGA** uses *no* reference dataset and *no* similarity score whatsoever.
  It is a dataset-size-weighted **geometric median** of the uploaded gradients,
  solved by Weiszfeld iteration. Its robustness comes from the geometric
  median's breakdown point, not from trust bootstrapping.
* **FedTruth** explicitly *rejects* FLTrust's root-dataset requirement (the
  paper names this as its advantage over FLTrust). It is a **truth-discovery**
  method: an alternating coordinate descent that jointly estimates a
  "ground-truth" update and the weights, with weights driven by each client's
  distance to the current estimate.

Consequently the three weight functions in this module -- :func:`fltrust_trust_scores`,
:func:`raga_weights`, :func:`fedtruth_weights` -- share no code path and are
tested against each other for exactly that reason.

Root-dataset construction (FLTrust only) -- DISCLOSED ADAPTATION
----------------------------------------------------------------
FLTrust assumes the service provider holds a small clean **root dataset**
``D_0`` (the paper's default: **100 training examples**), collected out of band.
No such externally-supplied set exists in this repository's data pipeline, and
none exists in FLTrust's own experimental setup either -- the paper itself
obtains ``D_0`` by *sampling a small labelled set from the task's training
data*. This module therefore does the same, and states it rather than assuming
it:

    :class:`FLTrustTrainer` samples ``root_size`` rows (default 100, matching
    the paper) uniformly at random from the pooled **client training** data,
    allocated proportionally to client size, and **removes those rows from the
    contributing clients' training sets** so the server's root set is disjoint
    from every client's local data.

Two properties of that construction are deliberate:

* It draws from ``X_train``/``y_train``, **never** from ``X_val``/``y_val``.
  The per-client validation pools form this repository's server-side ``D_V``,
  which is used for evaluation and for FedGRAA's rare-class stratification;
  drawing the root set from it would leak the evaluation set into training and
  would silently couple this baseline to FedGRAA's own machinery.
* The sampled rows are *removed* from the clients rather than duplicated, so
  no row is simultaneously "the server's clean reference" and "a client's
  training data". The per-client cap of ``n_k // 2`` guarantees no client is
  emptied. :class:`FLTrustTrainer` therefore trains its clients on marginally
  less data than the other baselines in this repository; that is the honest
  cost of giving the server a genuinely held-out root set, and it is reported
  on the instance as ``self.root_size_actual`` and ``self.client_sizes``.

This is a legitimate and well-precedented experimental choice (it is what every
open FLTrust reproduction does, and what the paper does), **not** a shortcut --
but it is an adaptation, and a reader comparing numbers against the NDSS paper
must know that ``D_0`` here is an in-distribution sample of the benchmark's own
training data rather than an independently collected clean corpus. FLTrust's
own bias study (root data drawn from a skewed distribution) is *not* reproduced
here.

RAGA and FedTruth take no root dataset and no such adaptation: they see exactly
the same client data as every other baseline in this repository.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from .config import FedConfig
from .metrics import accuracy, last_n_class_accuracy
from .model import (
    MLP,
    Params,
    aggregate,
    build_tf_optimizer,
    init_params,
)
from .partition import FederatedSplit
from .trainer import TrainHistory, client_train_round

__all__ = [
    "flatten_params",
    "model_update",
    "fltrust_trust_scores",
    "fltrust_aggregate",
    "raga_weights",
    "fedtruth_weights",
    "FLTrustTrainer",
    "RAGATrainer",
    "FedTruthTrainer",
]


# --------------------------------------------------------------------------- #
# Shared plumbing
# --------------------------------------------------------------------------- #
def flatten_params(params: Sequence[np.ndarray]) -> np.ndarray:
    """Flatten ``[W1, b1, ...]`` into one float64 vector.

    All three methods reason about a model update as a single point in
    ``R^d`` (cosine similarity, Euclidean distance, geometric median are all
    defined on the concatenated vector, not per-layer), so every geometric
    quantity in this module is computed on this flattening. float64 is used
    deliberately: the norms and inner products below are reductions over
    ~10^4-10^5 float32 entries, where float32 accumulation is a real source of
    trust-score noise.
    """
    if len(params) == 0:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate([np.ravel(np.asarray(p, dtype=np.float64)) for p in params])


def model_update(global_params: Params, local_params: Params) -> Params:
    """FLTrust's ``g_i = w_i - w`` (Algorithm 1, line 7).

    Note the sign convention. FLTrust (and RAGA, and FedTruth) define the
    submitted quantity as ``local - global``, and FLTrust's server then applies
    ``w <- w + alpha * g``. Elsewhere in this repository (``FedOptServer``) the
    opposite "pseudo-gradient" convention ``global - local`` is used, with the
    server applying a descent step. The two differ by a global sign, which is
    invisible to cosine similarity and to every distance used below; this
    module follows the *papers'* convention so the aggregation formulas can be
    transcribed literally.
    """
    return [np.asarray(lp, dtype=np.float32) - np.asarray(gp, dtype=np.float32)
            for gp, lp in zip(global_params, local_params)]


def _uniform(k: int) -> np.ndarray:
    return np.full(k, 1.0 / k, dtype=np.float32)


def _normalize_nonneg(v: np.ndarray) -> np.ndarray:
    """Normalize a non-negative vector onto the simplex; uniform if it sums to 0."""
    v = np.asarray(v, dtype=np.float64)
    total = v.sum()
    if not np.isfinite(total) or total <= 0.0:
        return _uniform(len(v))
    return (v / total).astype(np.float32)


# --------------------------------------------------------------------------- #
# FLTrust (Cao et al., NDSS 2021, arXiv:2012.13995)
# --------------------------------------------------------------------------- #
def fltrust_trust_scores(
    client_updates: Sequence[Params], server_update: Params
) -> np.ndarray:
    """FLTrust's **raw** (unnormalized) trust scores ``TS_i = ReLU(c_i)``.

    Verified against arXiv:2012.13995 (full text). With ``g_i`` the client's
    model update and ``g_0`` the server's model update on the root dataset,

        c_i  = <g_i, g_0> / (||g_i|| * ||g_0||)          (cosine similarity)
        TS_i = ReLU(c_i)                                 (clip negatives to 0)

    A client whose update points *away* from the server's reference direction
    is assigned trust exactly zero and is thereby fully excluded from the
    round -- this is the ReLU clipping, and it is the single property that
    makes FLTrust Byzantine-robust.

    Returns the raw scores in ``[0, 1]``, **not** normalized: the paper's
    aggregation divides by ``sum_j TS_j`` only at the final step, and the raw
    magnitudes are what a diagnostic wants to inspect. Use
    :func:`fltrust_aggregate` for the full rule.
    """
    g0 = flatten_params(server_update)
    n0 = float(np.linalg.norm(g0))
    scores = np.zeros(len(client_updates), dtype=np.float64)
    if n0 == 0.0 or not np.isfinite(n0):
        # A zero server update carries no direction, so no client can be
        # scored against it. The paper does not define this case (it cannot
        # arise with a nondegenerate root dataset); returning all-zero trust
        # makes the round a no-op rather than inventing a direction.
        return scores
    for i, upd in enumerate(client_updates):
        gi = flatten_params(upd)
        ni = float(np.linalg.norm(gi))
        if ni == 0.0 or not np.isfinite(ni):
            continue
        c = float(np.dot(gi, g0)) / (ni * n0)
        scores[i] = max(0.0, c)
    return scores


def fltrust_aggregate(
    global_params: Params,
    client_updates: Sequence[Params],
    server_update: Params,
    alpha: float = 1.0,
) -> Tuple[Params, np.ndarray]:
    """One FLTrust server round. Returns ``(new_global_params, weights)``.

    Verified against arXiv:2012.13995 (full text). Two steps beyond the trust
    score, both of which distinguish FLTrust from a plain weighted average:

    1. **Magnitude normalization.** Each client update is rescaled onto the
       server update's hypersphere::

           g_bar_i = (||g_0|| / ||g_i||) * g_i

       so that a client cannot gain influence by simply submitting a larger
       vector. This is why FLTrust cannot be expressed as
       ``aggregate(client_params, w)`` -- the rescaling is not a reweighting
       of the client *parameters*, and this function updates the global model
       directly instead.

    2. **Trust-weighted average, then a server step.**::

           g = (1 / sum_j TS_j) * sum_i TS_i * g_bar_i
           w <- w + alpha * g

    ``alpha`` is the paper's global learning rate. **Default 1.0**, which makes
    the round exactly "replace the global model by the trust-weighted average
    of magnitude-normalized client models" -- the direct analogue of this
    repository's ``theta_g = sum_k w_k theta_k`` aggregation, and the setting
    under which FLTrust is comparable to the other baselines here. The paper
    tunes ``alpha`` jointly with the local learning rate per dataset (e.g.
    3e-4 for MNIST-0.1); those values are tied to its own SGD-with-fixed-
    iteration-count local solver and do not transfer to this repository's
    Adam-with-``E``-epochs clients, so they are not adopted. Deviation
    disclosed; ``alpha`` is exposed so it can be swept.

    **Degenerate all-zero-trust rounds.** If every ``TS_i`` is zero (no client
    agrees with the server's direction), the paper's ``1 / sum_j TS_j`` is
    ``0/0`` and it does not define the case. The convention taken here is the
    only one consistent with the mechanism's meaning -- *no client is trusted,
    so no update is applied* -- and it emits a ``RuntimeWarning``. The weight
    vector returned for that round is uniform, purely so that the recorded
    history stays a valid simplex; it does **not** describe an aggregation that
    happened, because no aggregation happened.
    """
    ts = fltrust_trust_scores(client_updates, server_update)
    total = float(ts.sum())
    if total <= 0.0 or not np.isfinite(total):
        warnings.warn(
            "FLTrust: all trust scores are zero this round (no client update "
            "has positive cosine similarity with the server's root-dataset "
            "update). The paper leaves 1/sum(TS) undefined here; no update is "
            "applied and the recorded weight vector is uniform as a "
            "placeholder only.",
            RuntimeWarning,
            stacklevel=2,
        )
        return [np.array(p, copy=True) for p in global_params], _uniform(len(client_updates))

    w = (ts / total).astype(np.float32)
    n0 = float(np.linalg.norm(flatten_params(server_update)))

    agg: Params = [np.zeros_like(np.asarray(p, dtype=np.float32)) for p in global_params]
    for wi, upd in zip(w, client_updates):
        if wi == 0.0:
            continue
        ni = float(np.linalg.norm(flatten_params(upd)))
        if ni == 0.0 or not np.isfinite(ni):
            continue
        scale = np.float32(wi * (n0 / ni))  # TS_i/sum(TS) * ||g_0||/||g_i||
        for j, u in enumerate(upd):
            agg[j] += scale * np.asarray(u, dtype=np.float32)

    new_global = [
        np.asarray(gp, dtype=np.float32) + np.float32(alpha) * a
        for gp, a in zip(global_params, agg)
    ]
    return new_global, w


# --------------------------------------------------------------------------- #
# RAGA (Zuo et al., IEEE TMC 2025, arXiv:2403.13374)
# --------------------------------------------------------------------------- #
def raga_weights(
    client_updates: Sequence[Params],
    client_sizes: Sequence[int],
    tol: float = 1e-7,
    max_iter: int = 100,
) -> np.ndarray:
    """RAGA's aggregation weights: dataset-size-weighted **geometric median**.

    Verified against arXiv:2403.13374v3 (full text). RAGA uses **no root
    dataset, no reference direction, and no similarity score.** Eq. (15)
    defines the aggregate as

        z* = argmin_y  sum_i beta_i * ||y - z_i||,     beta_i = S_i / sum_j S_j

    where ``z_i`` is client ``i``'s uploaded (averaged) gradient and ``S_i`` is
    the number of samples in client ``i``'s local dataset -- so RAGA is
    simultaneously *size-weighted* (its answer to data heterogeneity) and
    *median-like* (its answer to Byzantine clients). The paper obtains an
    eps-optimal solution "by resorting to the Weiszfeld algorithm" (eq. 16)
    and the global step is ``w^{t+1} = w^t - eta^t * z*`` (eq. 14).

    Robustness here is *not* a trust score: it is the geometric median's
    breakdown point. A client far from the bulk gets a small weight because
    ``1/||z* - z_i||`` is small, no matter which direction it lies in. Contrast
    FLTrust, where a malicious update is excluded only if it is *anti-aligned*
    with the server reference.

    Implementation: standard Weiszfeld iteration,

        y <- sum_i (beta_i / ||y - z_i||) z_i  /  sum_i (beta_i / ||y - z_i||)

    initialized at the beta-weighted mean, iterated to ``tol`` (relative) or
    ``max_iter``. This function returns the **final iteration's normalized
    Weiszfeld coefficients**

        a_i = (beta_i / ||z* - z_i||) / sum_j (beta_j / ||z* - z_j||)

    rather than ``z*`` itself. That is exact, not an approximation: at the
    Weiszfeld fixed point ``z* = sum_i a_i z_i`` by construction, and because
    the geometric median is translation-equivariant, applying the same ``a``
    to the client *parameters* reproduces the paper's global step at
    ``eta^t = 1`` exactly --
    ``sum_i a_i w_i = w - sum_i a_i (w - w_i) = w - z*``.
    Returning ``a`` therefore lets RAGA use this repository's
    ``aggregate(client_params, w)`` path unchanged while remaining the
    geometric median rather than a mean.

    **Documented deviations.** (i) The paper's ``z_i`` is the average of
    ``K^t`` local SGD gradients; this repository's clients run ``E`` Adam
    epochs, so ``z_i`` is taken as the resulting model update ``w_i - w``,
    which is the same quantity up to the local optimizer's step scaling. RAGA's
    "freely selectable local round number ``K^t``" flexibility is therefore
    inherited from ``cfg.local_epochs`` rather than swept. (ii) The paper does
    not print its Weiszfeld iteration count or tolerance; ``tol``/``max_iter``
    here are standard choices, exposed as arguments.

    Degenerate cases: if ``z*`` coincides with some ``z_i`` the inverse
    distance is clamped by a tiny epsilon (the geometric median is genuinely at
    that point, and the clamp puts essentially all mass there, which is the
    correct behaviour); if all updates are identical the weights fall back to
    ``beta`` itself.
    """
    k = len(client_updates)
    if k == 0:
        raise ValueError("raga_weights requires at least one client update")
    if len(client_sizes) != k:
        raise ValueError("client_updates and client_sizes length mismatch")
    if k == 1:
        return np.ones(1, dtype=np.float32)

    Z = np.stack([flatten_params(u) for u in client_updates], axis=0)  # (k, d)
    sizes = np.asarray(client_sizes, dtype=np.float64)
    if np.any(sizes < 0):
        raise ValueError("client_sizes must be non-negative")
    beta = _normalize_nonneg(sizes).astype(np.float64)

    # Scale-aware epsilon: distances live on the scale of the updates.
    scale = float(np.mean(np.linalg.norm(Z, axis=1)))
    eps = max(scale, 1.0) * 1e-12

    y = beta @ Z  # beta-weighted mean initialization
    coeff = beta.copy()
    for _ in range(max_iter):
        dist = np.linalg.norm(Z - y[None, :], axis=1)
        inv = beta / np.maximum(dist, eps)
        s = inv.sum()
        if not np.isfinite(s) or s <= 0.0:
            coeff = beta.copy()
            break
        coeff = inv / s
        y_new = coeff @ Z
        shift = float(np.linalg.norm(y_new - y))
        y = y_new
        if shift <= tol * max(scale, 1.0):
            break

    return _normalize_nonneg(coeff)


# --------------------------------------------------------------------------- #
# FedTruth (Ebron, Zhang & Yang, ACISP 2025, arXiv:2311.10248)
# --------------------------------------------------------------------------- #
def _fedtruth_distance(Z: np.ndarray, truth: np.ndarray, distance: str) -> np.ndarray:
    if distance == "euclidean":
        return np.linalg.norm(Z - truth[None, :], axis=1)
    if distance == "cosine":
        nt = float(np.linalg.norm(truth))
        nz = np.linalg.norm(Z, axis=1)
        denom = nz * nt
        cos = np.zeros(len(Z), dtype=np.float64)
        good = denom > 0.0
        cos[good] = (Z[good] @ truth) / denom[good]
        return 1.0 - cos  # d = 1 - S_c, in [0, 2]
    raise ValueError(f"distance must be 'euclidean' or 'cosine', got {distance!r}")


def fedtruth_weights(
    client_updates: Sequence[Params],
    coefficient: str = "neg_log",
    distance: str = "euclidean",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> np.ndarray:
    """FedTruth's aggregation weights: truth-discovery coordinate descent.

    Verified against arXiv:2311.10248v3 (full text). FedTruth **explicitly
    removes FLTrust's root-dataset requirement** -- the paper names "does not
    require a benign root dataset on the server" as its advantage over FLTrust
    -- and instead *estimates* the ground-truth global update jointly with the
    weights. Eq. (1):

        min_{Delta*, p}  sum_k g(p_k) * d(Delta*, Delta_k)   s.t. sum_k p_k = 1

    where ``d`` is a distance function, ``p_k`` is client ``k``'s "performance"
    and ``g`` is a non-negative, monotonically **decreasing**, differentiable
    coefficient function. Sec. 3.3 solves it by alternating coordinate descent:

        Step 1 (weights):  p_k = d(Delta*, Delta_k) / sum_j d(Delta*, Delta_j)
                           a_k = g(p_k) / sum_j g(p_j)
        Step 2 (truth):    Delta* = sum_k a_k * Delta_k

    repeated to convergence (the paper reports 5-40 iterations per round, with
    a convergence threshold of 1e-6). Because ``sum_k a_k = 1``, ``Delta* =
    sum_k a_k Delta_k`` is equivalent to ``sum_k a_k w_k`` on the client
    parameters, so this function returns ``a`` and the caller uses this
    repository's ``aggregate(client_params, w)`` path.

    How this differs from the other two: FedTruth has **no reference signal at
    all** -- not a root dataset (FLTrust) and not a fixed size prior (RAGA).
    The reference is bootstrapped from the client updates themselves and
    refined; a client is down-weighted for being far from the *consensus it
    helped produce*, and the weights are unrelated to dataset size.

    **Documented deviations -- the paper does not declare experimental
    defaults, and this is the one genuine under-specification of the three:**

    * ``coefficient`` -- the paper requires ``g`` to be non-negative,
      differentiable and decreasing, and gives exactly two worked examples,
      ``g(p) = 1/p`` and ``g(p) = -log(p)``, **without declaring which one the
      experiments use**. The default here is ``"neg_log"`` (``g(p) = -log p``).
      Both are implemented and selectable. The choice is disclosed rather than
      hidden because it is not neutral: with ``g(p) = 1/p`` and Euclidean
      distance the update reduces to ``a_k proportional to 1/d_k``, which is
      *exactly* the unweighted Weiszfeld iteration -- i.e. that configuration
      of FedTruth converges to the plain geometric median and would collapse
      onto :func:`raga_weights` with uniform sizes. That reduction is an
      observation made here, **not** a claim from the paper (the paper does not
      mention the geometric median or Weiszfeld anywhere). ``"neg_log"`` is
      preferred as the default so that this baseline exercises the mechanism
      the paper actually presents as novel rather than silently duplicating
      RAGA; ``"inverse"`` is available for anyone who wants the other reading.
    * ``distance`` -- the paper evaluates Euclidean, cosine, angular and
      Manhattan and reports that Euclidean/Manhattan are best against Byzantine
      attacks while cosine is best against backdoor attacks, again without
      naming a single default. ``"euclidean"`` is the default here (this
      repository's threat setting is not a backdoor setting); ``"cosine"`` is
      implemented and selectable. Angular and Manhattan are not implemented.
    * The paper describes ``Delta*`` as initialized from "the result of FedAvg
      or simple average" without saying which. The simple (uniform) average is
      used here, which is also the ``|D_k|``-agnostic reading consistent with
      FedTruth's stated independence from dataset sizes.
    * ``tol`` defaults to the paper's stated 1e-6; ``max_iter`` (100) bounds a
      loop the paper reports terminating in 5-40 iterations.
    """
    if coefficient not in {"neg_log", "inverse"}:
        raise ValueError(
            f"coefficient must be 'neg_log' or 'inverse', got {coefficient!r}"
        )
    k = len(client_updates)
    if k == 0:
        raise ValueError("fedtruth_weights requires at least one client update")
    if k == 1:
        return np.ones(1, dtype=np.float32)

    Z = np.stack([flatten_params(u) for u in client_updates], axis=0)  # (k, d)
    scale = float(np.mean(np.linalg.norm(Z, axis=1)))
    eps = 1e-12

    a = np.full(k, 1.0 / k, dtype=np.float64)
    truth = a @ Z  # simple average initialization
    for _ in range(max_iter):
        d = _fedtruth_distance(Z, truth, distance)
        d = np.maximum(d, 0.0)
        d_sum = float(d.sum())
        if not np.isfinite(d_sum) or d_sum <= 0.0:
            # Every update coincides with the estimated truth -- the objective
            # is already 0 and every client is equally "correct".
            a = np.full(k, 1.0 / k, dtype=np.float64)
            break
        p = np.maximum(d / d_sum, eps)  # performance metric, sums to ~1
        if coefficient == "inverse":
            gp = 1.0 / p
        else:  # neg_log: p < 1 => -log p > 0
            gp = -np.log(np.minimum(p, 1.0 - eps))
        gp = np.maximum(gp, 0.0)
        g_sum = float(gp.sum())
        if not np.isfinite(g_sum) or g_sum <= 0.0:
            a = np.full(k, 1.0 / k, dtype=np.float64)
            break
        a = gp / g_sum
        truth_new = a @ Z
        shift = float(np.linalg.norm(truth_new - truth))
        truth = truth_new
        if shift <= tol * max(scale, 1.0):
            break

    return _normalize_nonneg(a)


# --------------------------------------------------------------------------- #
# Trainers
# --------------------------------------------------------------------------- #
class _RobustTrainerBase:
    """Shared scaffolding: model, per-client Adam, evaluation, round loop.

    Only :meth:`_round` differs between the three methods -- which is the point
    of grouping them: the local ``ClientUpdate`` (Adam, ``E`` epochs, batch
    ``B``) and the evaluation are identical to every other baseline in this
    repository, so any difference in results is attributable to the server-side
    aggregation rule alone.
    """

    method_name = "robust"

    def __init__(self, split: FederatedSplit, cfg: FedConfig):
        self.split = split
        self.cfg = cfg
        self.spec = split.spec
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

    # -- per-client training data (FLTrust overrides to exclude root rows) --- #
    def _client_data(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        c = self.split.clients[idx]
        return c.X_train, c.y_train

    def _train_clients(
        self, global_params: Params, rng: np.random.Generator
    ) -> List[Params]:
        out: List[Params] = []
        for idx, client in enumerate(self.split.clients):
            X, y = self._client_data(idx)
            out.append(
                client_train_round(
                    self._model,
                    self._client_tf_optimizers[client.client_id],
                    X, y, self.cfg, global_params, rng,
                )
            )
        return out

    def _round(
        self,
        global_params: Params,
        client_params: List[Params],
        rng: np.random.Generator,
    ) -> Tuple[Params, np.ndarray]:
        raise NotImplementedError

    def _initial_weights(self) -> np.ndarray:
        return _uniform(len(self.split.clients))

    def train(self, eval_every: int = 20, verbose: bool = False) -> TrainHistory:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        global_params: Params = init_params(self._model.layer_sizes, seed=cfg.seed)

        w = self._initial_weights()
        history = TrainHistory(
            method=self.method_name, gamma=None, n_rare=cfg.n_rare
        )
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
            client_params = self._train_clients(global_params, rng)
            global_params, w = self._round(global_params, client_params, rng)

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
                        f"  [{self.method_name}] round {rnd:3d}/{cfg.rounds}"
                        f"  all={all_a:.3f}  last-n={last_a:.3f}",
                        flush=True,
                    )

        self.final_params = global_params
        return history


class FLTrustTrainer(_RobustTrainerBase):
    """FLTrust (Cao et al., NDSS 2021, arXiv:2012.13995).

    Server-side trust bootstrapping from a small root dataset. Each round the
    server (a) runs the *same* local training procedure on its root dataset to
    obtain its own reference update ``g_0``, (b) scores every client update by
    ``TS_i = ReLU(cos(g_i, g_0))``, (c) rescales every client update to
    ``||g_0||``, and (d) averages them weighted by the normalized trust scores.
    See :func:`fltrust_trust_scores` and :func:`fltrust_aggregate` for the
    verified equations.

    **Root dataset.** See this module's docstring for the full disclosure: the
    root set is ``root_size`` rows sampled from the pooled *client training*
    data (never validation) and removed from the contributing clients. The
    actual size achieved is on ``self.root_size_actual``; the reduced per-client
    training sizes are on ``self.client_sizes``.

    **Server local training (documented deviation).** FLTrust's server runs
    ``ModelUpdate`` for the same number of SGD *iterations* ``R_l`` as the
    clients. This repository parameterizes local work in *epochs* (``E``) over
    each participant's own data, so a server holding ~100 root rows performs
    far fewer optimizer steps per round than a client holding thousands. The
    default keeps ``E`` identical for server and clients (the literal reading
    of "the same ModelUpdate"); ``server_local_epochs`` is exposed so the step
    counts can be matched instead if desired. The server keeps its own
    persistent Adam state, mirroring the clients.
    """

    method_name = "fltrust"

    def __init__(
        self,
        split: FederatedSplit,
        cfg: FedConfig,
        root_size: int = 100,
        alpha: float = 1.0,
        server_local_epochs: Optional[int] = None,
    ):
        super().__init__(split, cfg)
        if root_size < 1:
            raise ValueError(f"root_size must be >= 1, got {root_size}")
        self.alpha = float(alpha)
        self.root_size_requested = int(root_size)

        # -- carve the root dataset out of the pooled client training data -- #
        root_rng = np.random.default_rng(cfg.seed + 104_729)  # disjoint from training rng
        sizes = np.asarray([c.n_train for c in split.clients], dtype=np.int64)
        total = int(sizes.sum())
        if total == 0:
            raise ValueError("FLTrust: clients have no training data to sample a root set from")

        target = min(int(root_size), total)
        share = sizes * (target / total)
        quota = np.floor(share).astype(np.int64)
        rem = target - int(quota.sum())
        if rem > 0:  # hand out the remainder by largest fractional part
            order = np.argsort(-(share - quota), kind="stable")
            quota[order[:rem]] += 1
        # Never take more than half of any client's rows -- the root set must
        # not hollow out a client (and must never empty one).
        quota = np.minimum(quota, sizes // 2)

        root_X: List[np.ndarray] = []
        root_y: List[np.ndarray] = []
        self._kept: List[Tuple[np.ndarray, np.ndarray]] = []
        for i, client in enumerate(split.clients):
            n_k, q = int(sizes[i]), int(quota[i])
            if q > 0:
                pick = root_rng.choice(n_k, size=q, replace=False)
                mask = np.ones(n_k, dtype=bool)
                mask[pick] = False
                root_X.append(client.X_train[pick])
                root_y.append(client.y_train[pick])
                self._kept.append((client.X_train[mask], client.y_train[mask]))
            else:
                self._kept.append((client.X_train, client.y_train))

        if not root_X:
            raise ValueError(
                "FLTrust: could not sample a non-empty root dataset -- every "
                "client has fewer than 2 training rows. Lower root_size or use "
                "a larger split."
            )
        self.root_X = np.concatenate(root_X, axis=0)
        self.root_y = np.concatenate(root_y, axis=0)
        self.root_size_actual = int(len(self.root_y))
        if self.root_size_actual < self.root_size_requested:
            warnings.warn(
                f"FLTrust: requested root_size={self.root_size_requested} but only "
                f"{self.root_size_actual} rows could be sampled without taking more "
                f"than half of any client's training data. Proceeding with the "
                f"smaller root dataset.",
                RuntimeWarning,
                stacklevel=2,
            )
        # Client sizes reported to the outside world reflect the removal.
        self.client_sizes = [len(y) for _, y in self._kept]

        self._server_cfg = (
            cfg if server_local_epochs is None
            else dataclasses.replace(cfg, local_epochs=int(server_local_epochs))
        )
        self._server_optimizer = build_tf_optimizer(
            cfg.eta, self._model._model.trainable_variables
        )
        # Diagnostics: last round's raw (unnormalized) trust scores.
        self.last_trust_scores: Optional[np.ndarray] = None

    def _client_data(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._kept[idx]

    def server_update(
        self, global_params: Params, rng: np.random.Generator
    ) -> Params:
        """``g_0 = ModelUpdate(w, D_0, b, beta, R_l)`` -- the reference direction."""
        server_params = client_train_round(
            self._model, self._server_optimizer,
            self.root_X, self.root_y,
            self._server_cfg, global_params, rng,
        )
        return model_update(global_params, server_params)

    def _round(self, global_params, client_params, rng):
        g0 = self.server_update(global_params, rng)
        updates = [model_update(global_params, cp) for cp in client_params]
        self.last_trust_scores = fltrust_trust_scores(updates, g0)
        return fltrust_aggregate(global_params, updates, g0, alpha=self.alpha)


class RAGATrainer(_RobustTrainerBase):
    """RAGA (Zuo et al., IEEE TMC 2025, arXiv:2403.13374).

    Robust Average Gradient Algorithm: a dataset-size-weighted **geometric
    median** of the uploaded updates, solved by Weiszfeld iteration. **No root
    dataset, no reference direction, no similarity score** -- see
    :func:`raga_weights` for the verified equations and for why the returned
    Weiszfeld coefficients reproduce eq. (14)'s global step exactly.

    ``weiszfeld_tol`` / ``weiszfeld_max_iter`` control the eps-optimality of
    eq. (16); the paper does not print its own values.
    """

    method_name = "raga"

    def __init__(
        self,
        split: FederatedSplit,
        cfg: FedConfig,
        weiszfeld_tol: float = 1e-7,
        weiszfeld_max_iter: int = 100,
    ):
        super().__init__(split, cfg)
        self.weiszfeld_tol = float(weiszfeld_tol)
        self.weiszfeld_max_iter = int(weiszfeld_max_iter)

    def _round(self, global_params, client_params, rng):
        updates = [model_update(global_params, cp) for cp in client_params]
        w = raga_weights(
            updates, self.client_sizes,
            tol=self.weiszfeld_tol, max_iter=self.weiszfeld_max_iter,
        )
        return aggregate(client_params, w), w


class FedTruthTrainer(_RobustTrainerBase):
    """FedTruth (Ebron, Zhang & Yang, ACISP 2025, arXiv:2311.10248).

    Truth-discovery aggregation: alternating coordinate descent that jointly
    estimates a "ground-truth" global update and the per-client weights, with a
    client down-weighted in proportion to a decreasing function of its distance
    to the current estimate. **No root dataset** -- the paper names removing
    FLTrust's root-dataset requirement as its contribution -- and **no dataset-
    size prior** either. See :func:`fedtruth_weights` for the verified
    equations and for the disclosure of the two experimental defaults the paper
    leaves unspecified (``coefficient``, ``distance``).
    """

    method_name = "fedtruth"

    def __init__(
        self,
        split: FederatedSplit,
        cfg: FedConfig,
        coefficient: str = "neg_log",
        distance: str = "euclidean",
        tol: float = 1e-6,
        max_iter: int = 100,
    ):
        super().__init__(split, cfg)
        if coefficient not in {"neg_log", "inverse"}:
            raise ValueError(
                f"coefficient must be 'neg_log' or 'inverse', got {coefficient!r}"
            )
        if distance not in {"euclidean", "cosine"}:
            raise ValueError(
                f"distance must be 'euclidean' or 'cosine', got {distance!r}"
            )
        self.coefficient = coefficient
        self.distance = distance
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    def _round(self, global_params, client_params, rng):
        updates = [model_update(global_params, cp) for cp in client_params]
        w = fedtruth_weights(
            updates,
            coefficient=self.coefficient,
            distance=self.distance,
            tol=self.tol,
            max_iter=self.max_iter,
        )
        return aggregate(client_params, w), w
