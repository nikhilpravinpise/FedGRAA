"""Tier-5 adaptive-aggregation-weight baselines — FedGRAA's direct competitor set.

Four published methods that, like FedGRAA, decide *how much each client counts*
in the global average rather than changing local training. They are the closest
published neighbours of FedGRAA's Algorithm 2 and therefore the sharpest
comparison the paper can offer.

* :class:`FedAWATrainer` — Shi, Zhao, Zhang, Zhou, Guo, Chang, "FedAWA: Adaptive
  Optimization of Aggregation Weights in Federated Learning Using Client
  Vectors", CVPR 2025, arXiv:2503.15842.
* :class:`DAPFLTrainer` — Yang, Feng, Tong, Wang, Guo, Wang, Ding, Fang, Liao,
  "DA-PFL: Dynamic Affinity Aggregation for Personalized Federated Learning",
  AAAI 2024, arXiv:2403.09284 (extended in IEEE TNNLS 2025).
* :class:`FedAWARETrainer` — Zeng, Xu, Liu, Pan, Wang, Tang, "On the Power of
  Adaptive Weighted Aggregation in Heterogeneous Federated Learning and Beyond",
  arXiv:2310.02702v4 (Dec 2024). "FedAWARE" is the method name used as the
  paper's informal handle; the paper has no other short title.
* :class:`AdaAggRLTrainer` — Wang, Zhang, Wen, Qiu, Guo, "Defending Against
  Sophisticated Poisoning Attacks with RL-based Aggregation in Federated
  Learning", AAAI 2025, arXiv:2406.14217, DOI 10.1609/aaai.v39i24.34733.

These are Related-Work comparisons (tier T5 of the compute campaign). None of
them is part of the base FedGRAA paper's Figures 1-3.


================================================================================
⚠️  FedAWA is NOT FedAWARE. Read this before touching either class.
================================================================================
The two names differ by three letters and both are described in one-line
summaries as "adaptive aggregation weighting", but they are different papers by
different authors computing weights by *opposite* principles. Conflating them
would silently turn two baselines into one:

* **FedAWA (CVPR 2025)** *minimises disagreement.* It learns ``lambda`` by
  gradient descent on an objective whose first term is
  ``sum_k lambda_k * ||tau_k - tau_g||_2`` — the weighted distance of each
  client vector from the *merged* client vector — plus a
  ``1 - cos(sum_k lambda_k theta_k, theta_g)`` regulariser. Minimising it pushes
  mass **toward** clients whose update direction agrees with the consensus
  direction. Global-generalisation-oriented; majority-favouring.

* **FedAWARE (arXiv 2310.02702)** *minimises the aggregate itself.* It solves
  ``lambda = argmin_{lambda in simplex} ||sum_i lambda_i m_i||^2`` — the
  minimum-norm point in the convex hull of the momentum-averaged client updates
  (an MGDA-style common-descent direction), by Frank-Wolfe. The minimum-norm
  point sits *away* from any direction the majority agrees on and **toward**
  clients whose updates conflict with the rest, because cancelling a
  conflicting pair is what shrinks the norm. Minority/consensus-dynamics
  oriented.

Mechanically: FedAWA runs an iterative SGD-with-momentum loop on softmax
logits and never forms a Gram matrix; FedAWARE runs Frank-Wolfe on the
``K x K`` Gram matrix of client momenta and never runs gradient descent on
``lambda``. ``test_fedawa_and_fedaware_differ_on_identical_input`` in
``tests/test_baselines_t5_adaptive.py`` asserts they disagree on identical
input, precisely so this docstring cannot rot into a lie.


================================================================================
Per-method verification status and disclosed deviations
================================================================================
This repository's standing rule (see
:class:`fedgraa.baselines.StaticDistributionAwareTrainer`) is that an
implementation must never present an inferred formula as a verified one. What
follows is what was actually read, and what was actually approximated.

**FedAWA — full-text verified, plus reference-implementation verified.**
Verified against the arXiv HTML full text (Eq. 3, Algorithm 1) *and* the
authors' released ``server_funct.py``
(github.com/ChanglongShi/FedAWA). Verified: client vector
``tau_k = theta_k - theta_g``; merged vector ``tau_g = sum_k lambda_k tau_k``;
Eq. 3's two-term objective; ``theta_g^{t+1} = sum_k lambda_k theta_k``; the
simplex constraint enforced by **softmax over free logits** (not projection);
optimisation by **SGD with momentum 0.9 and weight decay 5e-4** on those
logits, for ``server_epochs`` steps.

  *Deviations, all disclosed:*
  (a) The paper states the regulariser as ``d(sum_k lambda_k theta_k, theta_g)``
      with ``d = 1 - cosine``; the released code instead builds a per-client
      cost matrix and computes ``sum_k lambda_k * d(theta_k, theta_g)``, a
      linearised variant. **We implement the paper's Eq. 3 as printed**, since
      the paper is the citable artifact. Both are gradient-descended over the
      same softmax-parameterised ``lambda``, so the comparison is unaffected in
      kind.
  (b) Neither paper nor code states the *initial* logits. We initialise them at
      ``log(fedavg_weights)`` so round 1 starts from the data-size weights that
      are this repository's FedAvg reference point, and carry the logits across
      rounds (matching the released code's persistent ``global_T_weight``).
      Judgment call, not a paper claim.
  (c) The released code's server learning rate is 0.01 with SGD; the paper's
      Table of hyperparameters lists 0.08 with 0.99 decay for the *client*
      optimiser. We default to the code's 0.01 for the weight optimiser and
      expose it as a constructor argument.
  (d) Only FedAWA (whole-model weights) is implemented, not FedAWA-L (Eq. 4's
      per-layer weights). ``TrainHistory`` records one weight vector per round,
      which a per-layer variant would not fit without changing shared plotting
      and persistence code owned elsewhere.

**FedAWARE — full-text verified.** Verified against the arXiv HTML full text of
v4 (Eqs. 2-3, Algorithm 1). Verified: the momentum recursion
``m_i^t = (1-alpha) m_i^{t-1} + alpha g_i^t`` over uploaded pseudo-gradients
``g_i^t = x^{t,0} - x^{t,K}``; the min-norm-point QP over the simplex; the
explicit statement "We use the Frank-Wolfe algorithm (Jaggi, 2013) to solve
it"; the *server-side* update ``x^{t+1} = x^t - eta_g d^t`` with
``d^t = sum_i lambda_i^t m_i^t`` (note: FedAWARE does **not** aggregate client
parameters directly — it takes a server step along the min-norm direction).

  *Deviations, all disclosed:* the paper grid-searches ``eta_g`` per task and
  sweeps ``alpha`` over ``{0.1, ..., 0.9}``; no single value is canonical, so
  the defaults here (``alpha=0.5``, ``eta_g=1.0``) are chosen, not quoted. At
  ``alpha=1, eta_g=1`` the update reduces exactly to ``sum_i lambda_i theta_i``,
  which is a useful sanity anchor. Partial participation (``S^t``) is not
  modelled because every trainer in this repository trains all clients every
  round; the ``i not in S^t`` branch of Eq. 3 is therefore never taken.

**DA-PFL — full-text verified.** Verified against the arXiv HTML full text of
the AAAI 2024 version (Eqs. 3-10, Algorithm 1). Verified: the overlapping-class
vectors ``E_ij``/``E_ji``; the affinity metric
``c_ij = (|E_ij| / K) * (2 - adjusted_cosine(E_ij, E_ji))``; the dynamic term
``theta_ij = (c_ij / sum_j c_ij) * (1 - exp(-(||w_i - w_j||^2 + eps) / sigma))``;
``alpha_ij = theta_ij / sum_j theta_ij``; the **personalised** aggregate
``w_i^g = sum_{j != i} alpha_ij w_j`` (client ``i``'s own model is excluded);
and the L2-regularised local objective of Eq. 10.

  ⚠️ **Correction to the common one-line summary.** DA-PFL does *not* upweight
  clients with *similar* label distributions. The ``2 - cosine`` form inverts
  the similarity: the paper's own words are that the metric captures the
  "complementary relationship of class distribution between clients", and that
  aggregation prefers clients with "complementary data distribution". Two
  clients with identical label histograms score ``c = |E|/K * 1``; two exactly
  anti-correlated ones score ``c = |E|/K * 3``. The overlap factor ``|E_ij|/K``
  then requires the complementarity to be *within a shared label support*.
  ``test_dapfl_affinity_prefers_complementary_over_identical`` pins this down.

  *Deviations, all disclosed:* (a) ``sigma`` (Eq. 7's temperature) and
  ``lambda`` (Eq. 10's proximal coefficient) are named but never given numeric
  values in the text we could access; the defaults here (1.0 and 0.01) are
  chosen, not quoted. (b) The paper's Eq. 10 is ambiguous about whether the
  client re-initialises at ``w_i^g`` or continues from ``w_i^{t-1}``; we
  initialise at ``w_i^g`` (the standard reading, and the only one that makes
  Algorithm 1's "Send ``w_i^g`` to clients" load-bearing) and use ``w_i^g`` as
  the proximal anchor, reusing
  :func:`fedgraa.baselines.proximal_client_train_round`. (c) Adjusted cosine is
  undefined when the overlap has fewer than two classes or zero variance; we
  set the cosine to 0 there (so ``c = 2|E_ij|/K``), which the paper does not
  specify. (d) **Reporting convention:** DA-PFL is personalised, so there is no
  single global model. ``TrainHistory`` accuracies are the *mean over the K
  personalised models* evaluated on the shared validation set, the recorded
  weight vector is the column-mean of the ``alpha`` matrix (still a valid
  simplex point), and ``final_params`` is the size-weighted mean personalised
  model. Every one of those is a reporting choice made here, not a paper claim;
  the per-client models are kept on ``self.personalized_params``.

**AdaAggRL — full-text verified for the mechanism; the RL stack is
DELIBERATELY SIMPLIFIED, and this is the largest approximation in this file.**
Verified against the arXiv HTML full text: the per-client state is a vector of
similarity metrics ``s_k = (S_{k,R}, S_{k,cl}, S_{k,cg}, S_{k,lg})`` in
``(0, 1]`` — a gradient-inversion reconstruction similarity plus three
MMD-based distribution similarities (current-vs-historical client,
current-client-vs-global, historical-client-vs-global); the action is a
5-dimensional continuous vector (four simplex weights over the state metrics,
plus a threshold ``b in [0,1]``); the aggregation weight is the normalised dot
product ``a . s_k``, thresholded at ``delta = max(w~) * b`` with an
**exponential penalty ``lambda^{h_k}``** applied to clients below threshold;
and the reward is ``r = f(theta^t) - f(theta^{t+1})``, the round-over-round
decrease in training loss. The RL algorithm is TD3 (MlpPolicy, Adam lr 1e-5,
batch 64, gamma 0.99).

  *What is faithful here:* the state metrics ``S_cl``/``S_cg``/``S_lg`` (RBF-MMD
  between the softmax output distributions of the relevant model pair on a
  probe set, mapped through ``exp(-MMD^2)`` into ``(0, 1]``), the action
  semantics, the normalised-dot-product scoring, the ``max(w~) * b`` threshold,
  the exponential penalty with a per-client strike counter, and the reward
  definition.

  *What is simplified, and why:*
  1. **No gradient inversion, so the state is 3-dimensional, not 4.**
     ``S_{k,R}`` is defined as the cosine similarity between the gradient of
     *reconstructed* data (30 optimisation iterations over 16 dummy images per
     client per round, per the paper) and the uploaded gradient. That is an
     image-domain reconstruction attack; this repository's data is tabular
     network-flow records, and the cost is disproportionate to a baseline. We
     **drop** ``S_{k,R}`` rather than substitute a look-alike quantity, and the
     action's weight vector is correspondingly 3-dimensional. This is a
     reduction of the published method, not a reproduction of it.
  2. **The MMD probe set is the server-held validation split, not
     reconstructed data.** AdaAggRL's server computes output distributions on
     inverted data; ours uses ``split.X_val``. This is a *weaker* assumption
     than the paper's (no inversion attack is performed) and it stays inside
     FedGRAA-v1's own threat model, which already grants the server a labelled
     ``D_V``. It would not be admissible under FedGRAA-**v2**'s threat model.
  3. **TD3 is replaced by a bandit-style REINFORCE policy.** The task-level
     licence for this is explicit; what we implement is a state-independent
     Gaussian policy over the 4 action logits (3 metric weights + threshold),
     updated once per round by a single REINFORCE step against a running-mean
     reward baseline. There is **no** replay buffer, **no** twin critics, **no**
     target networks, and **no** delayed policy update — i.e. none of the four
     things that make TD3 "TD3". A federated run here is 200 rounds, which is
     200 environment steps in total; an off-policy actor-critic has no
     opportunity to converge on 200 samples, so a full TD3 stack would add
     large machinery and reproduce noise rather than the paper's behaviour. The
     scoring function, the threshold, and the penalty — the parts that actually
     determine aggregation weights — are unchanged.
  4. **No poisoning attack is simulated.** AdaAggRL is framed as a defence; this
     repository's campaign runs it as an aggregation-weight method on clean
     clients. Its published defensive advantage is therefore *not* exercised,
     and any comparison table must say so rather than implying AdaAggRL
     underperforms at its stated job.

  Consequently ``AdaAggRLTrainer`` must be described in any results artifact as
  "AdaAggRL (simplified RL, 3-metric state)" or equivalent — never as a
  faithful reproduction of arXiv:2406.14217.

================================================================================
Hyperparameters are constructor arguments, not :class:`~fedgraa.config.FedConfig`
fields, mirroring :class:`fedgraa.baselines.FedProxTrainer` (``mu``) and
:class:`fedgraa.baselines.FedOptServer` (``server_lr``, ``tau``). No field was
added to ``FedConfig`` because none is genuinely required.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from .aggregation import fedavg_weights, project_to_simplex
from .baselines import proximal_client_train_round
from .config import FedConfig
from .metrics import accuracy, last_n_class_accuracy
from .model import (
    MLP,
    Params,
    aggregate,
    build_tf_optimizer,
    init_params,
)
from .partition import FederatedSplit, local_label_histogram
from .trainer import TrainHistory, client_train_round


# --------------------------------------------------------------------------- #
# Shared numerical helpers
# --------------------------------------------------------------------------- #
def _flatten(params: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate a ``[W1, b1, ...]`` parameter list into one float64 vector."""
    return np.concatenate([np.ravel(p) for p in params]).astype(np.float64)


def _softmax(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    z = u - np.max(u)
    e = np.exp(z)
    return e / e.sum()


def _pairwise_sqdist(Z: np.ndarray) -> np.ndarray:
    """``||z_i - z_j||^2`` for every pair, via the Gram matrix.

    Uses ``|a-b|^2 = |a|^2 + |b|^2 - 2 a.b`` rather than an ``n x n x d``
    broadcast temporary, which at the paper's model size (d ~ 4e4) would
    allocate tens of megabytes per call.
    """
    Z = np.asarray(Z, dtype=np.float64)
    sq = np.sum(Z * Z, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, 0.0)
    return d2


def _pseudo_gradients(
    global_params: Params, client_params: Sequence[Params]
) -> np.ndarray:
    """``g_k = theta_g - theta_k`` for every client, as a ``K x d`` matrix.

    Same quantity :class:`fedgraa.baselines.FedOptServer` forms at the
    aggregate level (``[gp - up for gp, up in zip(...)]``), computed per client.
    """
    g = _flatten(global_params)
    return np.stack([g - _flatten(cp) for cp in client_params], axis=0)


def _client_vectors(
    global_params: Params, client_params: Sequence[Params]
) -> np.ndarray:
    """FedAWA's client vectors ``tau_k = theta_k - theta_g`` (``K x d``)."""
    return -_pseudo_gradients(global_params, client_params)


# --------------------------------------------------------------------------- #
# FedAWA (CVPR 2025) — learned softmax weights minimising Eq. 3
# --------------------------------------------------------------------------- #
def fedawa_objective(lam: np.ndarray, tau: np.ndarray, theta_g: np.ndarray) -> float:
    """FedAWA Eq. 3 evaluated at ``lam``.

    ``sum_k lam_k * ||tau_k - tau_g||_2 + (1 - cos(sum_k lam_k theta_k, theta_g))``
    where ``tau_g = sum_k lam_k tau_k``. Because ``sum_k lam_k = 1`` and
    ``theta_k = theta_g + tau_k``, the merged model is simply
    ``theta_g + tau_g``.
    """
    lam = np.asarray(lam, dtype=np.float64)
    tau_g = lam @ tau
    dist = float(lam @ np.linalg.norm(tau - tau_g, axis=1))
    merged = theta_g + tau_g
    n_m = float(np.linalg.norm(merged))
    n_g = float(np.linalg.norm(theta_g))
    cos = float(merged @ theta_g / (n_m * n_g)) if n_m > 0 and n_g > 0 else 0.0
    return dist + (1.0 - cos)


def _fedawa_grad_lambda(
    lam: np.ndarray, tau: np.ndarray, theta_g: np.ndarray
) -> np.ndarray:
    """Analytic ``dL/dlambda`` of :func:`fedawa_objective`.

    Term A (``sum_k lam_k ||tau_k - tau_g||``) contributes
    ``||tau_k - tau_g|| - <tau_k, sum_j lam_j u_j>`` with
    ``u_j = (tau_j - tau_g) / ||tau_j - tau_g||``, because
    ``d||tau_j - tau_g|| / dlam_k = -<u_j, tau_k>``.
    Term B (``1 - cos``) contributes ``-<tau_k, dcos/dmerged>``.
    """
    tau_g = lam @ tau
    diff = tau - tau_g
    norms = np.linalg.norm(diff, axis=1)
    unit = diff / np.maximum(norms, 1e-12)[:, None]
    s = lam @ unit
    grad = norms - tau @ s

    merged = theta_g + tau_g
    n_m = float(np.linalg.norm(merged))
    n_g = float(np.linalg.norm(theta_g))
    if n_m > 0 and n_g > 0:
        cos = float(merged @ theta_g / (n_m * n_g))
        d_cos = theta_g / (n_m * n_g) - cos * merged / (n_m * n_m)
        grad = grad - tau @ d_cos
    return grad


def fedawa_weights(
    client_params: Sequence[Params],
    global_params: Params,
    logits: Optional[np.ndarray] = None,
    server_lr: float = 0.01,
    server_epochs: int = 5,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """One round of FedAWA's server-side weight optimisation (Eq. 3, Alg. 1).

    ``lambda = softmax(logits)``; the *logits* are what SGD-with-momentum
    updates, which is how the released reference implementation enforces
    ``||lambda||_1 = 1``. Returns ``(lambda, updated_logits)``; carry the logits
    into the next round (the reference code keeps a persistent
    ``global_T_weight``).
    """
    tau = _client_vectors(global_params, client_params)
    theta_g = _flatten(global_params)
    K = tau.shape[0]
    u = (
        np.zeros(K, dtype=np.float64)
        if logits is None
        else np.asarray(logits, dtype=np.float64).copy()
    )
    buf = np.zeros(K, dtype=np.float64)
    for _ in range(server_epochs):
        lam = _softmax(u)
        g_lam = _fedawa_grad_lambda(lam, tau, theta_g)
        # Chain through the softmax: dL/du = lam * (g - <lam, g>).
        g_u = lam * (g_lam - float(lam @ g_lam)) + weight_decay * u
        buf = momentum * buf + g_u
        u = u - server_lr * buf
    return _softmax(u).astype(np.float32), u


class FedAWATrainer:
    """FedAWA baseline (CVPR 2025, arXiv:2503.15842).

    Each round: clients train locally from ``theta_g``; the server forms the
    client vectors ``tau_k = theta_k - theta_g``, runs ``server_epochs`` steps of
    SGD-with-momentum on softmax logits to minimise Eq. 3, then aggregates
    ``theta_g <- sum_k lambda_k theta_k``.

    See the module docstring for the FedAWA-vs-FedAWARE distinction and for the
    disclosed deviations (Eq. 3 as printed rather than the released code's
    linearised regulariser; chosen logit initialisation; FedAWA-L not
    implemented).
    """

    def __init__(
        self,
        split: FederatedSplit,
        cfg: FedConfig,
        server_lr: float = 0.01,
        server_epochs: int = 5,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
    ):
        self.split = split
        self.cfg = cfg
        self.server_lr = float(server_lr)
        self.server_epochs = int(server_epochs)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
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
        # Disclosed judgment call: logits start at log(FedAvg weights) so round 1
        # begins at this repository's FedAvg reference point. Persistent across
        # rounds, mirroring the released code's ``global_T_weight``.
        self.logits = np.log(np.maximum(fedavg_weights(self.client_sizes), 1e-12))

    def train(self, eval_every: int = 20, verbose: bool = False) -> TrainHistory:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        global_params: Params = init_params(self._model.layer_sizes, seed=cfg.seed)

        w = _softmax(self.logits).astype(np.float32)
        history = TrainHistory(method="fedawa", gamma=None, n_rare=cfg.n_rare)
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
            w, self.logits = fedawa_weights(
                client_params, global_params,
                logits=self.logits,
                server_lr=self.server_lr,
                server_epochs=self.server_epochs,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
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
                        f"  [fedawa] round {rnd:3d}/{cfg.rounds}"
                        f"  all={all_a:.3f}  last-n={last_a:.3f}",
                        flush=True,
                    )

        self.final_params = global_params
        return history


# --------------------------------------------------------------------------- #
# FedAWARE (arXiv:2310.02702v4) — Frank-Wolfe minimum-norm point
# --------------------------------------------------------------------------- #
def frank_wolfe_min_norm(
    gram: np.ndarray, n_iters: int = 100, tol: float = 1e-12
) -> np.ndarray:
    """``argmin_{lambda in simplex} lambda^T G lambda`` by Frank-Wolfe.

    ``gram`` is the ``K x K`` Gram matrix ``M M^T`` of the client update
    vectors, so the objective is exactly ``||sum_i lambda_i m_i||^2``
    (FedAWARE Eq. 2). Frank-Wolfe with exact line search: the linear
    minimisation oracle over the simplex is ``argmin_j (G lambda)_j``, and the
    quadratic admits a closed-form step size. This is the projection-free
    solver the paper names ("We use the Frank-Wolfe algorithm (Jaggi, 2013) to
    solve it"), chosen there because ``d >> K``.
    """
    gram = np.asarray(gram, dtype=np.float64)
    K = gram.shape[0]
    if K == 1:
        return np.ones(1, dtype=np.float32)
    lam = np.full(K, 1.0 / K, dtype=np.float64)
    for _ in range(n_iters):
        grad = gram @ lam                      # (1/2) of the true gradient
        j = int(np.argmin(grad))
        d = -lam.copy()
        d[j] += 1.0                            # d = e_j - lambda
        dGd = float(d @ gram @ d)
        if dGd <= tol:
            break
        gamma = -float(lam @ gram @ d) / dGd
        gamma = float(np.clip(gamma, 0.0, 1.0))
        if gamma <= tol:
            break
        lam = lam + gamma * d
    # Guard against float drift only; ``lam`` is already on the simplex.
    return project_to_simplex(lam)


def fedaware_weights(momenta: np.ndarray) -> np.ndarray:
    """FedAWARE Eq. 2: minimum-norm point in the convex hull of ``momenta``."""
    M = np.asarray(momenta, dtype=np.float64)
    return frank_wolfe_min_norm(M @ M.T)


class FedAWARETrainer:
    """FedAWARE baseline (arXiv:2310.02702v4).

    Each round: clients upload pseudo-gradients ``g_i = theta_g - theta_i``; the
    server maintains momentum ``m_i <- (1-alpha) m_i + alpha g_i`` (Eq. 3),
    solves for the minimum-norm point ``lambda`` in the convex hull of the
    ``m_i`` by Frank-Wolfe (Eq. 2), and takes a server step
    ``theta_g <- theta_g - eta_g * sum_i lambda_i m_i`` (Alg. 1).

    Note this is a *server descent step along a common-descent direction*, not
    a weighted average of client parameters — see the module docstring for why
    that is the opposite philosophy to :class:`FedAWATrainer`.
    """

    def __init__(
        self,
        split: FederatedSplit,
        cfg: FedConfig,
        alpha: float = 0.5,
        eta_g: float = 1.0,
        fw_iters: int = 100,
    ):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.split = split
        self.cfg = cfg
        self.alpha = float(alpha)
        self.eta_g = float(eta_g)
        self.fw_iters = int(fw_iters)
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
        self.momenta: Optional[np.ndarray] = None

    def train(self, eval_every: int = 20, verbose: bool = False) -> TrainHistory:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        global_params: Params = init_params(self._model.layer_sizes, seed=cfg.seed)
        shapes = [p.shape for p in global_params]
        sizes = [int(p.size) for p in global_params]

        K = len(self.split.clients)
        w = np.full(K, 1.0 / K, dtype=np.float32)
        history = TrainHistory(method="fedaware", gamma=None, n_rare=cfg.n_rare)
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
            g = _pseudo_gradients(global_params, client_params)   # K x d
            if self.momenta is None:
                self.momenta = np.zeros_like(g)
            self.momenta = (1.0 - self.alpha) * self.momenta + self.alpha * g

            w = fedaware_weights(self.momenta)
            d_flat = np.asarray(w, dtype=np.float64) @ self.momenta

            # Server step x^{t+1} = x^t - eta_g d^t, unflattened back onto the
            # parameter list. FedAWARE never forms ``sum_k w_k theta_k``.
            new_params: Params = []
            off = 0
            for shape, size, p in zip(shapes, sizes, global_params):
                step = d_flat[off:off + size].reshape(shape)
                new_params.append((p - self.eta_g * step).astype(np.float32))
                off += size
            global_params = new_params

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
                        f"  [fedaware a={self.alpha}] round {rnd:3d}/{cfg.rounds}"
                        f"  all={all_a:.3f}  last-n={last_a:.3f}",
                        flush=True,
                    )

        self.final_params = global_params
        return history


# --------------------------------------------------------------------------- #
# DA-PFL (AAAI 2024) — dynamic affinity aggregation, personalized
# --------------------------------------------------------------------------- #
def dapfl_affinity_matrix(
    label_hists: Sequence[np.ndarray], n_classes: Optional[int] = None
) -> np.ndarray:
    """DA-PFL Eqs. 3-6: the static client-affinity matrix ``c_ij``.

    ``c_ij = (|E_ij| / K) * (2 - adjusted_cosine(E_ij, E_ji))`` where
    ``E_ij``/``E_ji`` are the two clients' counts restricted to the classes both
    hold, and the adjusted cosine subtracts the mean ``s_bar`` of all elements
    of ``E_ij`` and ``E_ji`` combined. ``2 - cos`` **inverts** similarity: the
    metric rewards *complementary* class distributions over identical ones (see
    the module docstring's correction).

    The diagonal is 0 — Eq. 9 aggregates over ``j != i`` only. Where the
    adjusted cosine is undefined (overlap of fewer than two classes, or zero
    variance in either vector) it is taken as 0, which the paper does not
    specify.
    """
    H = np.asarray(label_hists, dtype=np.float64)
    K = H.shape[0]
    n_classes = int(H.shape[1] if n_classes is None else n_classes)
    C = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            mask = (H[i] > 0) & (H[j] > 0)
            m = int(mask.sum())
            if m == 0:
                continue
            e_ij, e_ji = H[i][mask], H[j][mask]
            s_bar = float(np.concatenate([e_ij, e_ji]).mean())
            a, b = e_ij - s_bar, e_ji - s_bar
            den = float(np.linalg.norm(a) * np.linalg.norm(b))
            cos = float(a @ b / den) if den > 1e-12 else 0.0
            C[i, j] = (m / n_classes) * (2.0 - cos)
    return C


def dapfl_alpha_matrix(
    affinity: np.ndarray,
    client_params: Sequence[Params],
    sigma: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """DA-PFL Eqs. 7-8: the per-round personalized aggregation weights.

    ``theta_ij = (c_ij / sum_j c_ij) * (1 - exp(-(||w_i - w_j||^2 + eps)/sigma))``
    then ``alpha_ij = theta_ij / sum_j theta_ij``. Row ``i`` is a simplex point
    over ``j != i`` (``alpha_ii = 0``).
    """
    C = np.asarray(affinity, dtype=np.float64)
    K = C.shape[0]
    flat = np.stack([_flatten(p) for p in client_params], axis=0)
    sq = np.sum((flat[:, None, :] - flat[None, :, :]) ** 2, axis=2)
    dyn = 1.0 - np.exp(-(sq + eps) / float(sigma))

    A = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        row_c = C[i].copy()
        row_c[i] = 0.0
        tot = row_c.sum()
        c_bar = row_c / tot if tot > 1e-300 else np.zeros(K)
        theta = c_bar * dyn[i]
        theta[i] = 0.0
        s = theta.sum()
        if s > 1e-300:
            A[i] = theta / s
        elif tot > 1e-300:
            # Every model is identical (round 1) or sigma swamped the dynamic
            # term: fall back to the affinity-only weights, which is the limit
            # of Eq. 8 as the (constant) dynamic factor cancels.
            A[i] = c_bar
        else:
            A[i] = np.full(K, 1.0 / (K - 1)) if K > 1 else np.ones(1)
            A[i, i] = 0.0
    return A


class DAPFLTrainer:
    """DA-PFL baseline (AAAI 2024, arXiv:2403.09284).

    Personalized FL: every client keeps its own model. Each round the server
    builds ``alpha`` from the static complementarity affinity (Eqs. 3-6) and the
    live pairwise model distances (Eq. 7-8), sends each client its own
    ``w_i^g = sum_{j != i} alpha_ij w_j`` (Eq. 9), and the client trains from
    ``w_i^g`` under an L2 proximal pull back to ``w_i^g`` (Eq. 10).

    Reporting convention (a choice made here, not a paper claim): recorded
    accuracies are the mean over the K personalized models on the shared
    validation split; the recorded weight vector is the column-mean of
    ``alpha``; ``final_params`` is the size-weighted mean personalized model.
    Per-client models remain on ``self.personalized_params``.
    """

    def __init__(
        self,
        split: FederatedSplit,
        cfg: FedConfig,
        sigma: float = 1.0,
        prox_lambda: float = 0.01,
        eps: float = 1e-12,
    ):
        self.split = split
        self.cfg = cfg
        self.sigma = float(sigma)
        self.prox_lambda = float(prox_lambda)
        self.eps = float(eps)
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
        # Eqs. 3-6 run once, before round 1: clients upload their statistical
        # vectors S_i and the server computes the affinity matrix (Alg. 1).
        hists = [
            local_label_histogram(c, self.spec.n_classes, scope="full")
            for c in split.clients
        ]
        self.affinity = dapfl_affinity_matrix(hists, self.spec.n_classes)
        self.personalized_params: List[Params] = []

    def _evaluate(self, params_list: Sequence[Params]) -> Tuple[float, float]:
        all_a = float(np.mean([
            accuracy(self._model, p, self.split.X_val, self.split.y_val)
            for p in params_list
        ]))
        last_a = float(np.mean([
            last_n_class_accuracy(
                self._model, p, self.split.X_val, self.split.y_val,
                self.rare_class_ids,
            )
            for p in params_list
        ]))
        return all_a, last_a

    def _mean_params(self, params_list: Sequence[Params]) -> Params:
        w = fedavg_weights(self.client_sizes)
        return aggregate(list(params_list), w)

    def train(self, eval_every: int = 20, verbose: bool = False) -> TrainHistory:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        K = len(self.split.clients)
        local_params: List[Params] = [
            init_params(self._model.layer_sizes, seed=cfg.seed) for _ in range(K)
        ]

        w = np.full(K, 1.0 / K, dtype=np.float32)
        history = TrainHistory(method="dapfl", gamma=None, n_rare=cfg.n_rare)
        all_a, last_a = self._evaluate(local_params)
        history.record(0, all_a, last_a, w)

        for rnd in range(1, cfg.rounds + 1):
            A = dapfl_alpha_matrix(
                self.affinity, local_params, sigma=self.sigma, eps=self.eps
            )
            personalized_global = [
                aggregate(local_params, A[i].astype(np.float32)) for i in range(K)
            ]
            local_params = [
                proximal_client_train_round(
                    self._model,
                    self._client_tf_optimizers[client.client_id],
                    client.X_train, client.y_train,
                    cfg, personalized_global[i], rng, self.prox_lambda,
                )
                for i, client in enumerate(self.split.clients)
            ]
            # Column-mean of the alpha rows: a convex combination of simplex
            # points, hence itself a simplex point.
            w = A.mean(axis=0).astype(np.float32)

            if rnd % eval_every == 0 or rnd == cfg.rounds:
                all_a, last_a = self._evaluate(local_params)
                history.record(rnd, all_a, last_a, w)
                if verbose:
                    print(
                        f"  [dapfl s={self.sigma}] round {rnd:3d}/{cfg.rounds}"
                        f"  mean-all={all_a:.3f}  mean-last-n={last_a:.3f}",
                        flush=True,
                    )

        self.personalized_params = local_params
        self.final_params = self._mean_params(local_params)
        return history


# --------------------------------------------------------------------------- #
# AdaAggRL (AAAI 2025) — RL-learned aggregation weights (SIMPLIFIED, see header)
# --------------------------------------------------------------------------- #
def _rbf_mmd2(P: np.ndarray, Q: np.ndarray) -> float:
    """Squared RBF-kernel MMD between two sets of output distributions.

    Bandwidth by the median heuristic over the pooled pairwise squared
    distances. Returned value is clipped at 0 (the V-statistic estimator can go
    marginally negative).
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    Z = np.vstack([P, Q])
    sq = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=2)
    pos = sq[sq > 0]
    med = float(np.median(pos)) if pos.size else 1.0
    gamma = 1.0 / max(med, 1e-12)
    Kz = np.exp(-gamma * sq)
    n = len(P)
    val = Kz[:n, :n].mean() + Kz[n:, n:].mean() - 2.0 * Kz[:n, n:].mean()
    return float(max(val, 0.0))


def _mmd_similarity(P: np.ndarray, Q: np.ndarray) -> float:
    """``exp(-MMD^2)``, mapping the MMD into AdaAggRL's ``(0, 1]`` state range."""
    return float(np.exp(-_rbf_mmd2(P, Q)))


def adaaggrl_aggregation_weights(
    states: np.ndarray,
    action_weights: np.ndarray,
    b: float,
    strikes: np.ndarray,
    penalty_base: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """AdaAggRL's scoring + thresholding + exponential penalty.

    ``w~_k = (a . s_k) / sum_j (a . s_j)``; ``delta = max(w~) * b``; clients with
    ``w~_k < delta`` take one more strike and are scaled by
    ``penalty_base ** strikes_k``; clients at or above threshold have their
    strike counter reset. Returns ``(weights, updated_strikes)``.
    """
    S = np.asarray(states, dtype=np.float64)
    a = np.asarray(action_weights, dtype=np.float64)
    raw = S @ a
    tot = raw.sum()
    w_tilde = raw / tot if tot > 1e-300 else np.full(len(raw), 1.0 / len(raw))

    delta = float(w_tilde.max()) * float(b)
    below = w_tilde < delta
    new_strikes = np.where(below, np.asarray(strikes, dtype=np.float64) + 1.0, 0.0)
    penalised = w_tilde * (float(penalty_base) ** new_strikes)

    s = penalised.sum()
    if s <= 1e-300:
        penalised = np.full(len(penalised), 1.0 / len(penalised))
        s = 1.0
    return (penalised / s).astype(np.float32), new_strikes


def reinforce_step(
    logits: np.ndarray,
    noise: np.ndarray,
    advantage: float,
    lr: float,
    sigma: float,
) -> np.ndarray:
    """One REINFORCE step for a Gaussian policy ``a ~ N(logits, sigma^2 I)``.

    ``grad_logits log pi(a | logits) = (a - logits) / sigma^2 = noise / sigma``,
    so ascent on the expected reward is
    ``logits += lr * advantage * noise / sigma``. A positive advantage moves the
    policy mean *toward* the sampled perturbation; a negative one moves it away.

    This replaces AdaAggRL's TD3 actor-critic — see the module docstring for the
    full list of what that omits and why.
    """
    return np.asarray(logits, dtype=np.float64) + (
        float(lr) * float(advantage) / max(float(sigma), 1e-12)
    ) * np.asarray(noise, dtype=np.float64)


class AdaAggRLTrainer:
    """AdaAggRL baseline (AAAI 2025, arXiv:2406.14217) — **simplified RL**.

    ⚠️ Not a faithful reproduction of the published method. The scoring
    function, threshold, exponential penalty, and reward definition are
    reproduced; the state is reduced from four metrics to three (no gradient
    inversion), the MMD probe set is the server's validation split rather than
    reconstructed data, and TD3 is replaced by a bandit-style REINFORCE policy
    with no replay buffer, twin critics, target networks, or delayed updates.
    Read the module docstring before citing or tabulating this class.

    Per round: each client's state is
    ``(S_cl, S_cg, S_lg) = (exp(-MMD^2) of client-now vs client-last,
    client-now vs global, client-last vs global)`` over softmax outputs on a
    probe subsample; the policy samples an action (3 simplex metric weights +
    threshold ``b``); weights follow
    :func:`adaaggrl_aggregation_weights`; the reward is the decrease in probe
    loss across the round; the policy takes one REINFORCE step.
    """

    def __init__(
        self,
        split: FederatedSplit,
        cfg: FedConfig,
        policy_lr: float = 0.05,
        policy_sigma: float = 0.3,
        penalty_base: float = 0.5,
        n_probe: int = 128,
        baseline_decay: float = 0.9,
    ):
        self.split = split
        self.cfg = cfg
        self.policy_lr = float(policy_lr)
        self.policy_sigma = float(policy_sigma)
        self.penalty_base = float(penalty_base)
        self.baseline_decay = float(baseline_decay)
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

        # Policy parameters: 3 logits for the metric weights (softmax) and one
        # logit for the threshold b (sigmoid). State-independent by design.
        self.n_metrics = 3
        self.action_logits = np.zeros(self.n_metrics, dtype=np.float64)
        self.threshold_logit = np.zeros(1, dtype=np.float64)
        self.reward_baseline = 0.0
        self.strikes = np.zeros(len(split.clients), dtype=np.float64)
        self.rewards: List[float] = []

        probe_rng = np.random.default_rng(cfg.seed + 977)
        n_val = len(self.split.y_val)
        k = int(min(n_probe, n_val))
        idx = probe_rng.choice(n_val, size=k, replace=False) if k < n_val else np.arange(n_val)
        self.probe_X = self.split.X_val[idx]
        self.probe_y = self.split.y_val[idx]

    # -- state -------------------------------------------------------------- #
    def _probs(self, params: Params) -> np.ndarray:
        self._model.set_params(params)
        probs, _ = self._model.forward(self.probe_X)
        return np.asarray(probs, dtype=np.float64)

    def _states(
        self,
        client_params: Sequence[Params],
        prev_client_params: Sequence[Params],
        global_params: Params,
    ) -> np.ndarray:
        p_global = self._probs(global_params)
        rows = []
        for cur, prev in zip(client_params, prev_client_params):
            p_cur = self._probs(cur)
            p_prev = self._probs(prev)
            rows.append([
                _mmd_similarity(p_cur, p_prev),    # S_cl: current vs historical
                _mmd_similarity(p_cur, p_global),  # S_cg: current vs global
                _mmd_similarity(p_prev, p_global),  # S_lg: historical vs global
            ])
        return np.asarray(rows, dtype=np.float64)

    # -- policy ------------------------------------------------------------- #
    def _sample_action(
        self, rng: np.random.Generator
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        eps_a = rng.standard_normal(self.n_metrics)
        eps_b = rng.standard_normal(1)
        a = _softmax(self.action_logits + self.policy_sigma * eps_a)
        b = float(1.0 / (1.0 + np.exp(-(self.threshold_logit + self.policy_sigma * eps_b)[0])))
        return a, b, eps_a, eps_b

    def _policy_update(self, reward: float, eps_a: np.ndarray, eps_b: np.ndarray) -> None:
        advantage = reward - self.reward_baseline
        self.action_logits = reinforce_step(
            self.action_logits, eps_a, advantage, self.policy_lr, self.policy_sigma
        )
        self.threshold_logit = reinforce_step(
            self.threshold_logit, eps_b, advantage, self.policy_lr, self.policy_sigma
        )
        self.reward_baseline = (
            self.baseline_decay * self.reward_baseline
            + (1.0 - self.baseline_decay) * reward
        )

    def train(self, eval_every: int = 20, verbose: bool = False) -> TrainHistory:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        global_params: Params = init_params(self._model.layer_sizes, seed=cfg.seed)
        K = len(self.split.clients)
        prev_client_params: List[Params] = [
            [p.copy() for p in global_params] for _ in range(K)
        ]

        w = np.full(K, 1.0 / K, dtype=np.float32)
        history = TrainHistory(method="adaaggrl", gamma=None, n_rare=cfg.n_rare)
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
            loss_before = float(self._model.loss(global_params, self.probe_X, self.probe_y))

            states = self._states(client_params, prev_client_params, global_params)
            action, b, eps_a, eps_b = self._sample_action(rng)
            w, self.strikes = adaaggrl_aggregation_weights(
                states, action, b, self.strikes, self.penalty_base
            )
            global_params = aggregate(client_params, w)

            # r = f(theta^t) - f(theta^{t+1}): the decrease in loss this round.
            loss_after = float(self._model.loss(global_params, self.probe_X, self.probe_y))
            reward = loss_before - loss_after
            if not np.isfinite(reward):
                reward = 0.0
            self.rewards.append(reward)
            self._policy_update(reward, eps_a, eps_b)

            prev_client_params = [[p.copy() for p in cp] for cp in client_params]

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
                        f"  [adaaggrl] round {rnd:3d}/{cfg.rounds}"
                        f"  all={all_a:.3f}  last-n={last_a:.3f}  r={reward:+.4f}",
                        flush=True,
                    )

        self.final_params = global_params
        return history
