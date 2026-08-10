"""Data-free / validation-free contribution-estimation baselines (tier T3/T5).

Three published methods that, like FedGRAA, decide *how much each client's
model counts* in the global average -- but, unlike FedGRAA, do so **without a
server-held labelled validation set**. They are the closest competitors in the
literature and are reproduced here as experimental baselines.

* :class:`CELMTrainer` -- CELM. Ukaye, Tastan, Abdu-Aguye, Nandakumar,
  "Data-Free Client Contribution Estimation via Logit Maximization for
  Federated Learning", arXiv:2605.18892v1 (17 May 2026).
* :class:`SpectralFedTrainer` -- SpectralFed. Ukaye, Abdu-Aguye, Tastan,
  Nandakumar, "Data-Free Contribution Estimation in Federated Learning using
  Gradient von Neumann Entropy", arXiv:2604.22562 (CVPR 2026 FedVision
  Workshop).
* :class:`CoAstTrainer` -- CoAst. Wu, Zhang, Li, Xu, Zhong, "CoAst:
  Validation-Free Contribution Assessment for Federated Learning",
  arXiv:2409.02495.

--------------------------------------------------------------------------
CITATION-VERIFICATION DISCLOSURE
--------------------------------------------------------------------------
Every equation number quoted below was read off the **arXiv HTML full text**
of the paper named, not off the abstract and not from memory. Where the paper
under-specifies something this file has to pin down, the gap is named here and
again at the point of implementation -- it is never silently filled with a
plausible-looking formula.

**CELM -- full-text verified (arxiv.org/html/2605.18892v1).** Eqs. 6-14 and
the aggregation equation ``w_g^(t) = sum_i c_i^(t) w_i^(t)`` were read
verbatim, as were the stated hyperparameters (LM learning rate 0.01, 200
optimization steps per class-client probe, lambda = 0.001, EMA beta = 0.5,
warm-up ``T_w = 0.05 T``). Confirmed against the primary source, and
correcting the one-line summary this reproduction was briefed with: the probe
is **logit maximization by gradient ascent on the synthetic input x**, i.e.
the *mechanism is not gradient-based on the model*, but the probe itself is
solved by gradient ascent in input space -- ``s_c`` is explicitly "the
pre-softmax logit for class c". Weights are frozen after warm-up, as briefed.
*Under-specified and pinned here:* (i) ``c_i^(0)``, the EMA state before the
first round, is set to uniform ``1/N`` (SpectralFed's Algorithm 1 initializes
its own ``c_i^0 <- 1/n``, so uniform is the family's own convention); (ii)
one probe vector per (client, class) is optimized, batched across classes in a
single tensor -- exact, not an approximation, because the objective in Eq. 6
is separable across classes; (iii) the paper is written for image clients and
initializes probes "from random Gaussian noise from the standard normal
distribution" with warm-start reuse across rounds, which is what
:class:`CELMTrainer` does on this repo's tabular feature vectors.

**SpectralFed -- full-text verified (arxiv.org/html/2604.22562v1).** Eqs.
9/11/12 and the Algorithm 1 box were read verbatim, including
``S~^(1) = S^(1)``, ``S~^(t) = mu S~^(t-1) + (1-mu) S^(t)``,
``c_i^(t) = S~_i^(t) / sum_i S~_i^(t)``, ``mu = 0.9``, ``M_i^(t) = Delta
w_{i,L}^(t)`` of shape ``C x d``, ``A = M M^T``, and the fact that **higher
entropy earns a higher weight**. *One genuine ambiguity in the source:* Eq. 11
defines the density matrix by trace normalization (``rho = A / Tr(A)``, whose
eigenvalues sum to 1 and give a textbook von Neumann entropy), while Eq. 12 --
described by the paper as what the implementation does -- normalizes by the
Frobenius norm (``A~ = A / ||A||_F``) and feeds those eigenvalues straight into
``-sum_j lambda_j log lambda_j``; the paper only remarks the two are
"equivalent ... up to a scaling factor", which is not true of the entropy
value itself. Eq. 12 is therefore the **default** here (it is the stated
implementation) and Eq. 11 is available as ``normalization="trace"``. The
resulting entropy is non-negative either way, since ``||A||_F >= lambda_max``.
*Not implemented:* SpectralFuse (the paper's Algorithm 2), which fuses this
entropy with Class-Specific Shapley Values through a rank-adaptive Kalman
filter. Only the entropy branch is reproduced; the Kalman-fusion branch is out
of scope for this baseline and is **not** silently approximated by
:class:`SpectralFedTrainer`.

**CoAst -- full-text verified (arxiv.org/html/2409.02495v1), with one
deliberate, disclosed adaptation.** Algorithm 1 (per-layer argsort of
``|Delta_i|``, keep the top ``r%``, replace by ``alpha * sgn``), Eq. 7's
ternary quantization, Eq. 11's ``U^(t,k) = sum_{e=t+1}^{t+k} U^e``, and Eq. 13's
sign-agreement score were read verbatim, as were ``r = 10%``, ``alpha = 0.02``
and ``k = 2..5``. **The adaptation:** CoAst is explicitly *post hoc* -- the
paper states it "neither affects the local training procedure nor changes the
global models" -- so it defines a contribution *score*, not an aggregation
weight, and gives no formula for turning one into the other. This harness
compares methods by the weight vector they aggregate with, so
:class:`CoAstTrainer` offers both readings: ``weight_mode="fedavg"`` is the
paper-faithful one (FedAvg aggregation, scores recorded but never used), and
the default ``weight_mode="score"`` aggregates with the scores, mapped to the
simplex by clipping at zero and normalizing. That clip-and-normalize map is
**this file's choice, not CoAst's** -- it is chosen to mirror CELM's own
Eqs. 9-12 (``max(0, .)`` then normalize) rather than invented freely, but no
part of it is attributable to the CoAst authors. *Two further ambiguities in
the source:* Algorithm 1 writes the local update as ``Delta_i = Theta^(t-1) -
theta_i^t`` (global minus local) while Eq. 8 then adds ``alpha * Delta~`` to
``Theta^(t-1)``, which together would step *away* from the client; the sign
that makes the method mean what its own Section 4 says it means is
``Delta_i = theta_i^t - Theta^(t-1)``, and that is what is implemented. And
Eq. 13 is written over ``sgn(theta_i^t)`` where the surrounding text says the
quantity compared is the client's (pruned) *update*; the update reading is
implemented. Both are flagged again in :class:`CoAstTrainer`.

--------------------------------------------------------------------------
Like :mod:`fedgraa.baselines`, none of this is part of the base paper's
Figures 1-3; these are Related Work comparisons only. All three classes follow
that module's shape exactly (``__init__(split, cfg, ...)`` +
``train(eval_every, verbose) -> TrainHistory``), and all method-specific
hyperparameters are constructor arguments rather than new
:class:`~fedgraa.config.FedConfig` fields, matching ``FedProxTrainer(mu=...)``
and ``FedOptServer(server_lr=..., tau=...)``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import tensorflow as tf

from .config import FedConfig
from .metrics import accuracy, last_n_class_accuracy
from .model import MLP, Params, aggregate, build_tf_optimizer, init_params
from .partition import FederatedSplit
from .trainer import TrainHistory, client_train_round

__all__ = [
    "CELMTrainer",
    "SpectralFedTrainer",
    "CoAstTrainer",
    "celm_contribution_weights",
    "logit_maximization_probe",
    "von_neumann_entropy",
    "spectralfed_weights",
    "ternary_quantize_update",
    "coast_sign_agreement",
]


def _uniform(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n, dtype=np.float64)


def _normalize_or_uniform(scores: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Non-negative ``scores`` -> simplex, falling back to uniform if degenerate.

    Every one of the three methods normalizes a non-negative score vector by
    its own sum (CELM eq. 12, SpectralFed Algorithm 1 line 14, and this file's
    disclosed CoAst score->weight map). None of the three papers says what to
    do when that sum is zero -- which is reachable in all three (every client's
    debiased evidence clipped to 0; an all-zero round of updates; every
    cumulative CoAst score non-positive). Uniform is the only choice that
    keeps ``sum(w) == 1`` without inventing a preference between clients.
    """
    scores = np.asarray(scores, dtype=np.float64)
    total = float(scores.sum())
    if not np.isfinite(total) or total <= eps:
        return _uniform(len(scores))
    w = scores / total
    if not np.all(np.isfinite(w)):
        return _uniform(len(scores))
    return w


# =========================================================================== #
# CELM -- Data-Free Client Contribution Estimation via Logit Maximization
# arXiv:2605.18892v1
# =========================================================================== #
def logit_maximization_probe(
    model: MLP,
    params: Params,
    n_features: int,
    n_classes: int,
    x_init: np.ndarray,
    steps: int = 200,
    lr: float = 0.01,
    l2: float = 1e-3,
) -> tuple:
    """Solve CELM eq. 6 for every class at once; return ``(scores, x_star)``.

    Eq. 6 (verbatim from arXiv:2605.18892v1)::

        x*_{i,c} = argmax_x  s_c(x; w_i) - lambda ||x||_2^2

    with ``s_c`` the **pre-softmax logit** for class ``c``, solved by Adam
    gradient ascent in input space (paper: LM learning rate 0.01, 200 steps per
    class-client probe, lambda = 0.001, Gaussian init with warm-start reuse
    across rounds). Eq. 7 then reads the raw evidence off the optimum,
    ``q~_{i,c} = s_c(x*_{i,c}; w_i)``.

    All ``K`` class probes are optimized as one ``(K, n_features)`` tensor.
    That is exact rather than an approximation: the objective is a sum of
    per-row terms with no coupling between rows, so the joint gradient step is
    the per-class gradient step, row by row.

    ``x_init`` is the warm start (shape ``(n_classes, n_features)``); the
    returned ``x_star`` is meant to be fed back in next round.
    """
    model.set_params(params)
    x = tf.Variable(np.asarray(x_init, dtype=np.float32))
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    eye = tf.eye(n_classes, dtype=tf.float32)
    lam = tf.constant(float(l2), dtype=tf.float32)

    for _ in range(int(steps)):
        with tf.GradientTape() as tape:
            logits = model._model(x, training=False)
            # s_c(x_c) for each row c: the diagonal of the (K, C) logit matrix.
            own = tf.reduce_sum(logits * eye, axis=1)
            objective = tf.reduce_sum(own - lam * tf.reduce_sum(x * x, axis=1))
            neg = -objective  # Adam descends; eq. 6 ascends.
        grads = tape.gradient(neg, [x])
        opt.apply_gradients(zip(grads, [x]))

    logits = model._model(x, training=False)
    scores = tf.reduce_sum(logits * eye, axis=1).numpy().astype(np.float64)
    return scores, x.numpy()


def celm_contribution_weights(
    evidence: np.ndarray,
    bias: float,
    prev_weights: np.ndarray,
    ema_beta: float = 0.5,
    eps: float = 1e-12,
) -> np.ndarray:
    """CELM eqs. 9-13: raw evidence matrix -> this round's contribution weights.

    ``evidence`` is the cross-client evidence matrix ``q~`` of shape
    ``(N clients, K classes)`` from eq. 7; ``bias`` is ``b^(t-1)`` from eq. 8
    (the mean per-class logit of the *global* model's own probes at the
    previous round). Verbatim from arXiv:2605.18892v1::

        (9)   q_{i,c}   = max(0, q~_{i,c} - b^(t-1))
        (10)  r_{i,c}   = q_{i,c} / (sum_j q_{j,c} + eps)
        (11)  c^_i      = (1/K) sum_c r_{i,c}
        (12)  c-_i      = c^_i / sum_j c^_j
        (13)  c_i^(t)   = beta c_i^(t-1) + (1 - beta) c-_i^(t)

    Eq. 9's clip is what makes "upweight clients providing strong,
    discriminative evidence for underrepresented classes" concrete: a client
    whose logit for class ``c`` does not beat the global model's own baseline
    contributes literally nothing to that class's column.
    """
    q_tilde = np.asarray(evidence, dtype=np.float64)
    if q_tilde.ndim != 2:
        raise ValueError(f"evidence must be 2-D (N, K), got shape {q_tilde.shape}")
    prev = np.asarray(prev_weights, dtype=np.float64)
    if len(prev) != q_tilde.shape[0]:
        raise ValueError("prev_weights length must equal the number of clients")

    q = np.maximum(0.0, q_tilde - float(bias))          # eq. 9
    col = q.sum(axis=0) + eps                            # eq. 10 denominator
    r = q / col[None, :]                                 # eq. 10
    c_hat = r.mean(axis=1)                               # eq. 11
    c_bar = _normalize_or_uniform(c_hat, eps)            # eq. 12
    c = float(ema_beta) * prev + (1.0 - float(ema_beta)) * c_bar   # eq. 13
    # The EMA of two simplex points with beta in [0, 1] is already a simplex
    # point; re-normalizing only guards float32/float64 drift.
    return _normalize_or_uniform(c, eps)


class CELMTrainer:
    """CELM baseline (arXiv:2605.18892v1) -- see this module's disclosure block.

    Each warm-up round the server probes every client model by logit
    maximization (eq. 6) to build the cross-client evidence matrix (eq. 7),
    debiases it against the previous global model's own probes (eqs. 8-9),
    normalizes per class then per client (eqs. 10-12), EMA-smooths (eq. 13),
    and aggregates ``w_g^(t) = sum_i c_i^(t) w_i^(t)``. After
    ``T_w = warmup_frac * T`` rounds the weights are **frozen** at
    ``c^(T_w)`` for the rest of training (eq. 14) -- which is also why this
    baseline is cheap despite the 200-step probes: the probing stops after 5%
    of rounds.

    Hyperparameter defaults are the paper's own (``warmup_frac=0.05``,
    ``probe_steps=200``, ``probe_lr=0.01``, ``probe_l2=1e-3``,
    ``ema_beta=0.5``). ``T_w`` is floored at 1 round so that a short run still
    computes weights at least once rather than degenerating to uniform.
    """

    def __init__(
        self,
        split: FederatedSplit,
        cfg: FedConfig,
        warmup_frac: float = 0.05,
        probe_steps: int = 200,
        probe_lr: float = 0.01,
        probe_l2: float = 1e-3,
        ema_beta: float = 0.5,
        eps: float = 1e-12,
    ):
        if not 0.0 < warmup_frac <= 1.0:
            raise ValueError(f"warmup_frac must be in (0, 1], got {warmup_frac}")
        if not 0.0 <= ema_beta <= 1.0:
            raise ValueError(f"ema_beta must be in [0, 1], got {ema_beta}")
        if probe_steps < 1:
            raise ValueError(f"probe_steps must be >= 1, got {probe_steps}")
        self.split = split
        self.cfg = cfg
        self.warmup_frac = float(warmup_frac)
        self.probe_steps = int(probe_steps)
        self.probe_lr = float(probe_lr)
        self.probe_l2 = float(probe_l2)
        self.ema_beta = float(ema_beta)
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

        self.n_classes = self.spec.n_classes
        self.n_features = self.spec.n_features
        # T_w = 0.05 T (eq. 14); at least one round so weights are ever computed.
        self.warmup_rounds = max(1, int(round(self.warmup_frac * cfg.rounds)))
        # Recorded so a caller (or a test) can check the freeze actually happens.
        self.frozen_weights: Optional[np.ndarray] = None
        self.evidence_history: List[np.ndarray] = []
        # Warm-start probe state (paper: probes are reused across rounds).
        self._probe_x: Dict[int, np.ndarray] = {}
        self._probe_x_global: Optional[np.ndarray] = None

    # -- probing ---------------------------------------------------------- #
    def _init_probe(self, rng: np.random.Generator) -> np.ndarray:
        """Standard-normal probe init (paper: "random Gaussian noise")."""
        return rng.standard_normal((self.n_classes, self.n_features)).astype(np.float32)

    def _evidence_matrix(
        self,
        client_params: Sequence[Params],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Eq. 7's ``q~`` for every (client, class), warm-started per client."""
        rows = []
        for i, params in enumerate(client_params):
            if i not in self._probe_x:
                self._probe_x[i] = self._init_probe(rng)
            scores, x_star = logit_maximization_probe(
                self._model, params, self.n_features, self.n_classes,
                self._probe_x[i], self.probe_steps, self.probe_lr, self.probe_l2,
            )
            self._probe_x[i] = x_star
            rows.append(scores)
        return np.stack(rows, axis=0)

    def _global_bias(self, global_params: Params, rng: np.random.Generator) -> float:
        """Eq. 8: ``b^(t-1) = (1/K) sum_c s_c(x*_{g,c}; w_g^(t-1))``."""
        if self._probe_x_global is None:
            self._probe_x_global = self._init_probe(rng)
        scores, x_star = logit_maximization_probe(
            self._model, global_params, self.n_features, self.n_classes,
            self._probe_x_global, self.probe_steps, self.probe_lr, self.probe_l2,
        )
        self._probe_x_global = x_star
        return float(np.mean(scores))

    # -- training loop ----------------------------------------------------- #
    def train(self, eval_every: int = 20, verbose: bool = False) -> TrainHistory:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        global_params: Params = init_params(self._model.layer_sizes, seed=cfg.seed)

        n_clients = len(self.split.clients)
        self._probe_x = {}
        self._probe_x_global = None
        # c^(0): uniform. The paper does not state the EMA's initial value;
        # SpectralFed (same group, Algorithm 1) initializes c_i^0 <- 1/n.
        w = _uniform(n_clients)

        history = TrainHistory(method="celm", gamma=None, n_rare=cfg.n_rare)
        history.record(
            0,
            accuracy(self._model, global_params, self.split.X_val, self.split.y_val),
            last_n_class_accuracy(
                self._model, global_params, self.split.X_val, self.split.y_val,
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

            if rnd <= self.warmup_rounds:
                # b^(t-1) is probed on the PREVIOUS round's global model, i.e.
                # `global_params` before this round's aggregation (eq. 8).
                bias = self._global_bias(global_params, rng)
                evidence = self._evidence_matrix(client_params, rng)
                self.evidence_history.append(evidence)
                w = celm_contribution_weights(
                    evidence, bias, w, self.ema_beta, self.eps,
                )
                self.frozen_weights = w.copy()
            # else: eq. 14 -- c^(t) = c^(T_w) for all t > T_w. `w` is simply
            # carried forward untouched, and no probing is done at all.

            global_params = aggregate(client_params, w)

            if rnd % eval_every == 0 or rnd == cfg.rounds:
                all_a = accuracy(
                    self._model, global_params, self.split.X_val, self.split.y_val
                )
                last_a = last_n_class_accuracy(
                    self._model, global_params, self.split.X_val, self.split.y_val,
                    self.rare_class_ids,
                )
                history.record(rnd, all_a, last_a, w)
                if verbose:
                    print(
                        f"  [celm Tw={self.warmup_rounds}] round {rnd:3d}/{cfg.rounds}"
                        f"  all={all_a:.3f}  last-n={last_a:.3f}",
                        flush=True,
                    )

        self.final_params = global_params
        return history


# =========================================================================== #
# SpectralFed -- Gradient von Neumann Entropy
# arXiv:2604.22562
# =========================================================================== #
def von_neumann_entropy(
    M: np.ndarray,
    normalization: str = "frobenius",
    eps: float = 1e-12,
) -> float:
    """Von Neumann entropy of the class-space Gram matrix of ``M`` (eqs. 9-12).

    ``M`` is the final-layer update in class-major orientation, shape
    ``(C, d)`` (arXiv:2604.22562: "For the update, M in R^(C x d)"). The Gram
    matrix ``A := M M^T in S^C_+`` is normalized and its eigenvalues are fed to
    ``S = -sum_j lambda_j log lambda_j`` (eq. 9).

    ``normalization``:

    * ``"frobenius"`` (default) -- eq. 12, ``A~ = A / ||A||_F``. This is what
      the paper states the implementation does. The eigenvalues do **not** sum
      to 1, so this is a scaled quantity rather than a textbook von Neumann
      entropy; it is nonetheless non-negative, because ``||A||_F >=
      lambda_max`` puts every eigenvalue in ``[0, 1]``.
    * ``"trace"`` -- eq. 11, ``rho = A / Tr(A)``, the textbook density matrix
      whose eigenvalues do sum to 1.

    The paper calls these "equivalent ... up to a scaling factor", which is
    true of the eigenvalues but not of the entropy value; see this module's
    disclosure block. A zero update (norm below ``eps``) scores 0.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"M must be 2-D (C, d), got shape {M.shape}")
    A = M @ M.T
    if normalization == "frobenius":
        norm = float(np.linalg.norm(A, ord="fro"))
    elif normalization == "trace":
        norm = float(np.trace(A))
    else:
        raise ValueError(
            f"normalization must be 'frobenius' (eq. 12) or 'trace' (eq. 11), "
            f"got {normalization!r}"
        )
    if not np.isfinite(norm) or norm <= eps:
        return 0.0
    lam = np.linalg.eigvalsh(A / norm)
    lam = lam[lam > eps]          # PSD up to round-off; drop the numerical zeros
    if lam.size == 0:
        return 0.0
    return float(-np.sum(lam * np.log(lam)))


def spectralfed_weights(
    entropies: Sequence[float],
    smoothed_prev: Optional[np.ndarray],
    momentum: float = 0.9,
    eps: float = 1e-12,
) -> tuple:
    """Algorithm 1 lines 13-14; returns ``(weights, smoothed_entropy)``.

    Verbatim: ``S~^(1) = S^(1)`` (no smoothing at the first round),
    ``S~^(t) = mu S~^(t-1) + (1 - mu) S^(t)``, then
    ``c_i^(t) = S~_i^(t) / sum_i S~_i^(t)``. Higher entropy -> higher weight.
    Pass ``smoothed_prev=None`` for the first round.
    """
    S = np.asarray(entropies, dtype=np.float64)
    if smoothed_prev is None:
        S_smooth = S.copy()
    else:
        prev = np.asarray(smoothed_prev, dtype=np.float64)
        S_smooth = float(momentum) * prev + (1.0 - float(momentum)) * S
    # Entropy is non-negative by construction, so the sum-normalization of
    # line 14 already lands on the simplex; _normalize_or_uniform only covers
    # the all-zero round the paper does not discuss.
    return _normalize_or_uniform(np.maximum(S_smooth, 0.0), eps), S_smooth


class SpectralFedTrainer:
    """SpectralFed baseline (arXiv:2604.22562, Algorithm 1).

    Each round the server takes every client's **final-layer weight update**
    ``M_i = (theta_i,L - theta_g,L)^T`` (shape ``C x d``), forms the class-space
    Gram matrix ``A = M M^T``, normalizes it and reads its von Neumann entropy
    (eq. 12). Clients whose updates spread evenly across class directions score
    high; a client whose update collapses onto a single class direction scores
    ~0. Entropies are momentum-smoothed (``mu = 0.9``) and normalized into
    aggregation weights.

    Only Algorithm 1 (SpectralFed) is reproduced. Algorithm 2 (SpectralFuse:
    rank-adaptive Kalman fusion of this entropy with Class-Specific Shapley
    Values) is **not** implemented -- see this module's disclosure block.
    """

    def __init__(
        self,
        split: FederatedSplit,
        cfg: FedConfig,
        momentum: float = 0.9,
        normalization: str = "frobenius",
        eps: float = 1e-12,
    ):
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if normalization not in {"frobenius", "trace"}:
            raise ValueError(
                f"normalization must be 'frobenius' or 'trace', got {normalization!r}"
            )
        self.split = split
        self.cfg = cfg
        self.momentum = float(momentum)
        self.normalization = normalization
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
        self.entropy_history: List[np.ndarray] = []

    def _client_entropies(
        self, client_params: Sequence[Params], global_params: Params
    ) -> np.ndarray:
        """``S_i^(t)`` for every client, from the final-layer update only."""
        # Params are [W1, b1, ..., WL, bL]; the final Dense kernel is [-2] and
        # has shape (d, C), so the paper's class-major M is its transpose.
        W_g = np.asarray(global_params[-2], dtype=np.float64)
        out = np.empty(len(client_params), dtype=np.float64)
        for i, p in enumerate(client_params):
            delta = np.asarray(p[-2], dtype=np.float64) - W_g
            out[i] = von_neumann_entropy(delta.T, self.normalization, self.eps)
        return out

    def train(self, eval_every: int = 20, verbose: bool = False) -> TrainHistory:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        global_params: Params = init_params(self._model.layer_sizes, seed=cfg.seed)

        n_clients = len(self.split.clients)
        w = _uniform(n_clients)          # Algorithm 1: c_i^0 <- 1/n
        smoothed: Optional[np.ndarray] = None

        history = TrainHistory(method="spectralfed", gamma=None, n_rare=cfg.n_rare)
        history.record(
            0,
            accuracy(self._model, global_params, self.split.X_val, self.split.y_val),
            last_n_class_accuracy(
                self._model, global_params, self.split.X_val, self.split.y_val,
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
            S = self._client_entropies(client_params, global_params)
            self.entropy_history.append(S)
            w, smoothed = spectralfed_weights(S, smoothed, self.momentum, self.eps)
            global_params = aggregate(client_params, w)

            if rnd % eval_every == 0 or rnd == cfg.rounds:
                all_a = accuracy(
                    self._model, global_params, self.split.X_val, self.split.y_val
                )
                last_a = last_n_class_accuracy(
                    self._model, global_params, self.split.X_val, self.split.y_val,
                    self.rare_class_ids,
                )
                history.record(rnd, all_a, last_a, w)
                if verbose:
                    print(
                        f"  [spectralfed mu={self.momentum}]"
                        f" round {rnd:3d}/{cfg.rounds}"
                        f"  all={all_a:.3f}  last-n={last_a:.3f}",
                        flush=True,
                    )

        self.final_params = global_params
        return history


# =========================================================================== #
# CoAst -- Validation-Free Contribution Assessment
# arXiv:2409.02495
# =========================================================================== #
def ternary_quantize_update(
    delta: Sequence[np.ndarray], prune_ratio: float = 0.10
) -> List[np.ndarray]:
    """CoAst Algorithm 1 / eq. 7: keep the top ``r%`` of each layer, as signs.

    For each parameter array (the paper's "layer") the entries are argsorted by
    ``|Delta|`` in descending order, the top ``r%`` are selected, and each
    selected entry is replaced by its sign; everything else is zero::

        (7)  u~ = +1 if u >  delta_j^r
             u~ = -1 if u < -delta_j^r
             u~ =  0 otherwise

    At least one entry per array is kept so that a very small layer (a bias
    vector, say) does not silently drop out of the score -- the paper's
    percentile phrasing leaves that case open.
    """
    if not 0.0 < prune_ratio <= 1.0:
        raise ValueError(f"prune_ratio must be in (0, 1], got {prune_ratio}")
    out: List[np.ndarray] = []
    for arr in delta:
        a = np.asarray(arr, dtype=np.float64)
        flat = a.ravel()
        k = max(1, int(round(prune_ratio * flat.size)))
        keep = np.argpartition(-np.abs(flat), k - 1)[:k]
        q = np.zeros_like(flat)
        q[keep] = np.sign(flat[keep])
        out.append(q.reshape(a.shape))
    return out


def coast_sign_agreement(
    quantized_update: Sequence[np.ndarray],
    accumulated_global_update: Sequence[np.ndarray],
) -> float:
    """CoAst eq. 13: ``p^_i^t = sum_h sgn(.[h]) sgn(U^(t,k)[h])``.

    Entries the quantization zeroed contribute nothing, so the sum runs
    effectively over the client's own top-``r%`` indices even though eq. 13
    is written over all of ``|Theta|``. A client whose important update
    directions agree with where the global model actually went over the next
    ``k`` rounds scores positively; one that pulled the opposite way scores
    negatively.
    """
    total = 0.0
    for q, u in zip(quantized_update, accumulated_global_update):
        total += float(np.sum(np.sign(np.asarray(q)) * np.sign(np.asarray(u))))
    return total


class CoAstTrainer:
    """CoAst baseline (arXiv:2409.02495) -- see this module's disclosure block.

    Mechanism, as published: each round the server ternary-quantizes every
    client's local update to its top-``r%`` entries (Algorithm 1, eq. 7),
    remembers it, and ``k`` rounds later scores it by sign agreement (eq. 13)
    against ``U^(t,k) = sum_{e=t+1}^{t+k} U^e``, the accumulated global update
    over those ``k`` subsequent rounds (eq. 11). Per-client scores are summed
    across rounds (eq. 5). No validation data is used anywhere.

    ``weight_mode``:

    * ``"score"`` (default) -- aggregate with the cumulative scores, clipped at
      zero and normalized. **This mapping is not in the paper**; CoAst is a
      post-hoc scorer and states it "neither affects the local training
      procedure nor changes the global models". The mode exists because this
      harness compares methods by the weights they aggregate with.
    * ``"fedavg"`` -- the paper-faithful reading: the global model is plain
      FedAvg and the scores are pure measurement. ``self.contribution_scores``
      and ``self.score_weight_history`` are populated identically in both
      modes, so the *mechanism* can be tested without the adaptation.

    Two sign/term ambiguities in the source are resolved as documented in this
    module's disclosure block: the local update is taken as
    ``theta_i^t - Theta^(t-1)``, and eq. 13 is read as comparing the client's
    quantized *update* (not its raw parameters) against ``U^(t,k)``.

    Because a round's score needs the following ``k`` rounds' global updates,
    the first ``k`` rounds are necessarily unscored; weights stay uniform until
    the first score lands. Defaults are the paper's (``r = 10%``, ``k = 2``);
    ``alpha = 0.02`` is carried for completeness but does not affect eq. 13,
    which only reads signs.
    """

    def __init__(
        self,
        split: FederatedSplit,
        cfg: FedConfig,
        prune_ratio: float = 0.10,
        alpha: float = 0.02,
        k: int = 2,
        weight_mode: str = "score",
        eps: float = 1e-12,
    ):
        if weight_mode not in {"score", "fedavg"}:
            raise ValueError(
                f"weight_mode must be 'score' or 'fedavg', got {weight_mode!r}"
            )
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.split = split
        self.cfg = cfg
        self.prune_ratio = float(prune_ratio)
        self.alpha = float(alpha)
        self.k = int(k)
        self.weight_mode = weight_mode
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
        n_clients = len(split.clients)
        self.contribution_scores = np.zeros(n_clients, dtype=np.float64)
        self.score_weight_history: List[np.ndarray] = []

    def train(self, eval_every: int = 20, verbose: bool = False) -> TrainHistory:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        global_params: Params = init_params(self._model.layer_sizes, seed=cfg.seed)

        from .aggregation import fedavg_weights

        n_clients = len(self.split.clients)
        self.contribution_scores = np.zeros(n_clients, dtype=np.float64)
        self.score_weight_history = []
        fedavg_w = fedavg_weights(self.client_sizes)
        w = fedavg_w if self.weight_mode == "fedavg" else _uniform(n_clients)

        # round index -> [quantized update per client]; and the per-round U^e.
        pending: Dict[int, List[List[np.ndarray]]] = {}
        global_updates: Dict[int, List[np.ndarray]] = {}

        history = TrainHistory(method="coast", gamma=None, n_rare=cfg.n_rare)
        history.record(
            0,
            accuracy(self._model, global_params, self.split.X_val, self.split.y_val),
            last_n_class_accuracy(
                self._model, global_params, self.split.X_val, self.split.y_val,
                self.rare_class_ids,
            ),
            w,
        )

        for rnd in range(1, cfg.rounds + 1):
            prev_global = [p.copy() for p in global_params]
            client_params = [
                client_train_round(
                    self._model,
                    self._client_tf_optimizers[client.client_id],
                    client.X_train, client.y_train,
                    cfg, global_params, rng,
                )
                for client in self.split.clients
            ]

            # Algorithm 1: Delta_i = theta_i^t - Theta^(t-1), then top-r% signs.
            pending[rnd] = [
                ternary_quantize_update(
                    [cp - gp for cp, gp in zip(params, prev_global)],
                    self.prune_ratio,
                )
                for params in client_params
            ]

            # The weight vector this round actually aggregates with is the one
            # standing at the end of the previous round: a round's own score
            # cannot exist until its k successor rounds have happened, so
            # there is nothing newer available at this point.
            w_used = w
            global_params = aggregate(client_params, w_used)
            # U^t = Theta^t - Theta^(t-1) (eq. 10 up to the proportionality the
            # paper leaves implicit; eq. 13 reads only signs, so any positive
            # scale factor -- including alpha/N -- is irrelevant to the score).
            global_updates[rnd] = [
                gp - pg for gp, pg in zip(global_params, prev_global)
            ]

            # Score the round that is now exactly k rounds behind (eqs. 11-13).
            t0 = rnd - self.k
            if t0 >= 1 and t0 in pending:
                accum = [np.zeros_like(a) for a in global_updates[rnd]]
                for e in range(t0 + 1, t0 + self.k + 1):
                    for j, u in enumerate(global_updates[e]):
                        accum[j] = accum[j] + u
                for i, q_i in enumerate(pending[t0]):
                    self.contribution_scores[i] += coast_sign_agreement(q_i, accum)
                del pending[t0]
                # global_updates[t0] can never be needed again: any later t0'
                # only reads rounds > t0'.
                global_updates.pop(t0, None)

            score_w = _normalize_or_uniform(
                np.maximum(self.contribution_scores, 0.0), self.eps
            )
            self.score_weight_history.append(score_w)
            w = score_w if self.weight_mode == "score" else fedavg_w

            if rnd % eval_every == 0 or rnd == cfg.rounds:
                all_a = accuracy(
                    self._model, global_params, self.split.X_val, self.split.y_val
                )
                last_a = last_n_class_accuracy(
                    self._model, global_params, self.split.X_val, self.split.y_val,
                    self.rare_class_ids,
                )
                history.record(rnd, all_a, last_a, w_used)
                if verbose:
                    print(
                        f"  [coast k={self.k} r={self.prune_ratio}"
                        f" mode={self.weight_mode}] round {rnd:3d}/{cfg.rounds}"
                        f"  all={all_a:.3f}  last-n={last_a:.3f}",
                        flush=True,
                    )

        self.final_params = global_params
        return history
