"""Tier-5 baseline: FedPGO -- FedGRAA's single closest published competitor.

:class:`FedPGOTrainer` -- Bikila, D. D. & Capek, J., "FedPGO: A personalized
federated learning framework with gradient orthogonalization for IoT
intrusion detection." *Computer Networks* 280:112210, published
2026-03-10. DOI 10.1016/j.comnet.2026.112210.

This project's own literature review (``docs/research_c/Final_Stage_C.md``,
the row citing this paper) flags FedPGO as **"the closest paper in the
corpus; occupies the three-way intersection on FedGRAA's own dataset"** --
same domain (federated NIDS), same primary dataset (CIC-IDS2017), same
mechanism family (gradient-based adaptive reweighting) -- differing from
FedGRAA on personalization (parameter-split, not a single global model), no
gamma-tunable rare-class objective, and an accuracy/robustness evaluation
axis rather than per-minority-class recall. Omitting it from the baseline
grid would be the single most likely fatal miss a domain-literate reviewer
could flag (``Final_Stage_C.md``'s own words).

================================================================================
Primary-source status (2026-07-31)
================================================================================
This is a **faithful implementation from the paper's primary text**, which is
local at ``docs/research_c/FedPGO/FedPGO.md`` (obtained 2026-07-30; the
citation is triple-confirmed by metadata: DBLP ``journals/cn/BikilaC26``,
OpenAlex ``W7134986038``, Semantic Scholar ``CorpusId:286471251``). It
implements the mechanism exactly as the primary text specifies it -- AGO
decomposition (Eqs. 6-8, Algorithm 1), the PGF score (Eq. 9), score-weighted
aggregation (Eq. 10, Algorithm 2), and the cross-round gradient memory
(Algorithm 3, step 23) -- and **replaces** this module's earlier disclosed
proxy (a reconstruction from the literature review's mechanism summary,
committed at ``c5c8ac8``/``30fd064``; the proxy's 20 frontier cells are
archived at ``results/v2_frontier/_archive_fedpgo_proxy_2026-07-30/`` and the
proxy survives only as git history + that archive). The decision to
reimplement and rerun is ``docs/research_c/STAGE_C_FEDPGO_FAITHFUL_PLAN_2026-07-31.md``
(Path 1, D1).

================================================================================
The mechanism, as implemented (all equation numbers from the primary text)
================================================================================

The paper's ``g_i`` is the *gradient* of client ``i``'s local model. Its own
Algorithm 3 is internally ambiguous -- line 10 trains the local model before
line 15 computes ``g_i = grad L_i(w_t)`` -- and it trains with Adam, so a
literal gradient is not recoverable without a second backward pass. This
implementation therefore approximates ``g_i`` by the per-round
shared-parameter **delta** ``tau_k = updated_shared - shared_global``
(repo-wide convention; JC1). All vectors live in delta space.

Client side, per round ``t`` and client ``i``:

1. Train ``E`` local epochs on the client's own ``client.X_train`` (the
   harness already carves a deterministic 90/10 train/val split per client at
   partition time -- ``fedgraa/partition.py`` -- so this is the same
   ``X_train`` every other baseline trains on), starting from
   ``shared_global + personalized[i]``.
2. ``g_i = tau_k = updated_shared - shared_global``.
3. **AGO** (Eqs. 6-7, Alg. 1): ``c_i = (g_i . g_G) / (||g_G||^2 + eps)``,
   ``g_par = c_i * g_G``, ``g_perp = g_i - g_par``, where ``g_G`` is the
   **previous round's** aggregated global gradient (gradient memory, Alg. 3
   step 23; initialized to the zero vector).
4. **Recompose** (Eq. 8, Alg. 1 step 6): ``g'_i = alpha_i * g_par +
   (1 - alpha_i) * g_perp``.
5. **Personalization / Delta-L-val** (Alg. 3 steps 11-14): evaluate
   ``L_before`` on the client's own per-client validation split
   ``client.X_val/y_val`` (the 10% slice carved at partition time), fine-tune
   the personalized head only (``personalize_steps`` gradient steps at
   ``lr = cfg.eta``), evaluate ``L_after``, and record
   ``Delta_L_val = L_before - L_after``. The fine-tuned head persists as the
   client's personalized parameters for the next round.
6. **PGF score** (Eq. 9): ``S_i = lambda1 * cos(g_i, g_G) + lambda2 * ||g_perp||
   + lambda3 * Delta_L_val``, where ``cos(g_i, g_G) = (g_i . g_G) /
   (||g_i|| * ||g_G||)``; at round 1 (``g_G = 0``) the cosine term is defined
   as 0 (JC4).

Server side, per round:

7. **Weights** (Eq. 10, Alg. 2 steps 3-6): ``w_i = S_i / (sum_j S_j + eps)``,
   with scores clipped at 0 before summing (JC3); uniform fallback if every
   score is ~0 (existing degenerate-round repo convention).
8. **Aggregate** (Alg. 2 steps 7-8): ``g'_G = sum_i w_i * g'_i``;
   ``shared_global <- shared_global + server_lr * g'_G``.
9. **Gradient memory** (Alg. 3 step 23): ``g_G <- g'_G`` (persisted across
   rounds; the *unscaled* aggregated delta, since Alg. 3 step 22 applies
   ``eta`` separately -- JC8 maps the sign convention).

================================================================================
Judgment calls (each disclosed here, not claimed as a paper value)
================================================================================

==========  =========================================  =================================================
ID          Question                                  Choice / rationale
==========  =========================================  =================================================
JC1         What is ``g_i``?                          Per-round shared delta ``tau_k``, not a raw
                                                      gradient pass -- repo-wide delta convention; the
                                                      paper's Alg. 3 is internally ambiguous (trains
                                                      before it computes the gradient) and trains with
                                                      Adam; "no additional communication overhead" claim.
JC2         Where does per-client validation data     The harness's existing per-client splits
            come from?                                (``ClientShard.X_val/y_val``, carved at partition
                                                      time by ``_train_val_split``) -- no carve needed;
                                                      every baseline trains on the same 90% ``X_train``
                                                      so zero data-efficiency deviation; the head
                                                      fine-tune uses only the client's own 10% slice;
                                                      reporting stays on the pooled split (the
                                                      harness's standing convention).
JC3         Negative PGF scores?                      Clip at 0 before normalization -- the paper
                                                      asserts ``w_i in [0, 1]`` (Eq. 10 discussion),
                                                      which requires ``S_i >= 0``.
JC4         Round-1 degeneracy (``g_G = 0``)          ``c_i = 0``, ``g_par = 0``, ``g_perp = g_i``,
                                                      ``g'_i = (1-alpha_i) * g_i``; cosine term 0 --
                                                      direct consequence of Eq. 6 with ``g_G = 0``.
JC5         ``alpha_i`` values                        Constructor scalar ``alpha`` default 0.5 (optional
                                                      per-client array), validated in [0, 1] -- the
                                                      paper gives no formula ("may be adapted based on
                                                      data diversity or gradient alignment").
JC6         ``lambda1, lambda2, lambda3``             Constructor defaults all 1.0 -- the paper gives
                                                      no values ("tuned via validation performance").
JC7         Head fine-tune size                       ``personalize_steps`` default 5, lr = ``cfg.eta``,
                                                      head (personalized) parameters only, on the
                                                      client's own ``X_val/y_val`` -- Alg. 3 step 12
                                                      ("Fine-tune personalized head using validation
                                                      data") gives no count; kept cheap (per client per
                                                      round).
JC8         Delta-space sign convention               Server update ``+ server_lr * g'_G`` with
                                                      ``server_lr = 1.0`` -- the paper's
                                                      ``w_{t+1} = w_t - eta * g'_G`` maps onto it with
                                                      the delta sign convention; consistent with the
                                                      repo's T3/T5 baselines.
JC9         Local training loss                       Plain softmax cross-entropy, via this repo's
                                                      shared ``client_train_round`` (``trainer.py``) --
                                                      the primary text's own Sec. 3.3/Eq. 4 names focal
                                                      loss as part of FedPGO's local-training procedure,
                                                      not merely an incidental detail, and this is
                                                      *not* implemented here. Deviation, not merely an
                                                      ambiguity: kept for fair-comparison consistency
                                                      with every other T3/T5/TC baseline, all of which
                                                      share this same generic trainer and loss. Flagged
                                                      2026-08-03 as a real, if disclosed-late, gap --
                                                      FedPGO's measured rare-class recall may differ
                                                      from what the paper's own focal-loss training
                                                      would produce.
JC10        Shared/personalized layer-split boundary  ``personalized_layers=1`` default (last layer
                                                      only) -- a judgment call about which of this
                                                      project's MLP layers correspond to the paper's
                                                      Conv1D-based shared extractor ``f_s`` vs.
                                                      personalized head ``f_p``; exposed as a
                                                      constructor argument, tested
                                                      (``test_fedpgo_personalized_heads_differ_shared_
                                                      layers_aggregated``).
==========  =========================================  =================================================

Constructor surface (all disclosed defaults, none claimed as paper values):
``alpha=0.5``, ``lambda1=lambda2=lambda3=1.0``, ``personalize_steps=5``,
``personalized_layers=1``, ``server_lr=1.0``, ``eps=1e-12``.

**Reporting convention (a choice made here, not a paper claim, matching
DA-PFL's own precedent):** recorded accuracy is the mean over the K
personalized models on the shared validation split; the recorded weight
vector is PGF's own ``w`` simplex; ``final_params`` is the FedAvg-weighted
mean of the personalized models (for any caller that needs one representative
parameter set); per-client models remain on ``self.personalized_params``.

Not part of the base FedGRAA paper's Figures 1-3 -- this is a Related-Work
(tier T5) comparison.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

from .config import FedConfig
from .metrics import accuracy, last_n_class_accuracy
from .model import (
    MLP,
    Params,
    aggregate,
    build_tf_optimizer,
    clone_params,
    init_params,
)
from .partition import FederatedSplit
from .trainer import TrainHistory, client_train_round

__all__ = ["FedPGOTrainer", "fedpgo_decompose", "fedpgo_pgf_score", "fedpgo_weights"]


def _flatten(params: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate a parameter-list into one flat vector (delta space)."""
    return np.concatenate([a.ravel() for a in params])


def fedpgo_decompose(
    g_i: Sequence[np.ndarray],
    g_G: Sequence[np.ndarray],
    alpha: float,
    eps: float = 1e-12,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """AGO decomposition (Eqs. 6-8, Algorithm 1).

    Given client ``i``'s local gradient ``g_i`` (delta space) and the global
    gradient ``g_G``, returns ``(g_par, g_perp, g_prime)``:

    * ``g_par``: the projection of ``g_i`` onto ``g_G`` --
      ``c_i * g_G`` with ``c_i = (g_i . g_G) / (||g_G||^2 + eps)`` (Eq. 6);
    * ``g_perp``: ``g_i - g_par`` (Eq. 7);
    * ``g_prime``: the recomposed update ``alpha * g_par + (1 - alpha) *
      g_perp`` (Eq. 8).

    The projection coefficient is a single scalar taken over the FLATTENED
    concatenation of every array (one coefficient per client, not one per
    layer), since "the global gradient" is a single direction, not a per-layer
    family of them.

    Round-1 degeneracy (``g_G = 0``, JC4): ``c_i = 0`` falls out of the
    formula -- ``g_par = 0``, ``g_perp = g_i``, ``g_prime = (1 - alpha) * g_i``
    -- with ``eps`` protecting the denominator from a literal division by
    zero.
    """
    flat_k = _flatten(g_i)
    flat_g = _flatten(g_G)
    denom = float(np.dot(flat_g, flat_g))
    coeff = float(np.dot(flat_k, flat_g)) / (denom + eps)

    parallel = [coeff * g for g in g_G]
    orthogonal = [k - coeff * g for k, g in zip(g_i, g_G)]
    recomposed = [
        alpha * p + (1.0 - alpha) * o
        for p, o in zip(parallel, orthogonal)
    ]
    return parallel, orthogonal, recomposed


def fedpgo_pgf_score(
    g_i: Sequence[np.ndarray],
    g_G: Sequence[np.ndarray],
    g_perp: Sequence[np.ndarray],
    delta_l_val: float,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    lambda3: float = 1.0,
    eps: float = 1e-12,
) -> float:
    """PGF score (Eq. 9): ``S_i = l1*cos(g_i,g_G) + l2*||g_perp|| + l3*DL_val``.

    ``cos(g_i, g_G) = (g_i . g_G) / (||g_i|| * ||g_G||)``. If either norm is
    ~0 (round 1 with ``g_G = 0``, JC4) the cosine term is 0. ``g_perp`` is the
    orthogonal component of the *same* decomposition used for the recomposed
    update (Eq. 7). ``delta_l_val`` is the validation-loss reduction after
    personalization (Alg. 3 steps 11-14).
    """
    flat_k = _flatten(g_i)
    flat_g = _flatten(g_G)
    norm_k = float(np.linalg.norm(flat_k))
    norm_g = float(np.linalg.norm(flat_g))
    if norm_k < eps or norm_g < eps:
        cos_term = 0.0
    else:
        cos_term = float(np.dot(flat_k, flat_g)) / (norm_k * norm_g)
    orth_norm = float(np.linalg.norm(_flatten(g_perp)))
    return (
        lambda1 * cos_term
        + lambda2 * orth_norm
        + lambda3 * float(delta_l_val)
    )


def fedpgo_weights(scores: Sequence[float], eps: float = 1e-12) -> np.ndarray:
    """Score-weighted aggregation weights (Eq. 10, Algorithm 2 steps 3-6).

    ``w_i = S_i / (sum_j S_j + eps)``. Scores are clipped at 0 before
    summing (JC3) so that weights stay in [0, 1] as the paper asserts; a
    degenerate round (every score ~0) falls back to uniform weights, the
    existing degenerate-round convention ``fedgraa/baselines_t3_trust.py``
    uses.
    """
    scores = np.asarray(scores, dtype=np.float64)
    clipped = np.maximum(scores, 0.0)
    total = float(clipped.sum())
    if total <= eps:
        return np.full(len(scores), 1.0 / len(scores), dtype=np.float32)
    return (clipped / (total + eps)).astype(np.float32)


class FedPGOTrainer:
    """FedPGO baseline -- see this module's docstring for the full
    primary-source implementation note and judgment-call disclosures.
    """

    def __init__(
        self,
        split: FederatedSplit,
        cfg: FedConfig,
        alpha: Union[float, Sequence[float]] = 0.5,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        lambda3: float = 1.0,
        personalize_steps: int = 5,
        personalized_layers: int = 1,
        server_lr: float = 1.0,
        eps: float = 1e-12,
    ):
        self.split = split
        self.cfg = cfg
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.lambda3 = float(lambda3)
        self.personalize_steps = int(personalize_steps)
        self.server_lr = float(server_lr)
        self.eps = float(eps)
        self.spec = split.spec
        self.rare_class_ids = self.spec.rare_client_ids(cfg.n_rare)
        self.client_sizes = [c.n_train for c in split.clients]
        layer_sizes = self.spec.model_config().layer_sizes
        self._model = MLP(layer_sizes, seed=cfg.seed)
        self._client_tf_optimizers: Dict[int, tf.keras.optimizers.Adam] = {
            c.client_id: build_tf_optimizer(cfg.eta, self._model._model.trainable_variables)
            for c in split.clients
        }

        K = split.n_clients
        self._alpha = np.asarray(alpha, dtype=np.float64)
        if self._alpha.ndim == 0:
            self._alpha = np.full(K, float(self._alpha))
        if self._alpha.shape != (K,):
            raise ValueError(
                f"alpha must be a scalar or an array of length K={K}; got shape "
                f"{self._alpha.shape}"
            )
        if not np.all((self._alpha >= 0.0) & (self._alpha <= 1.0)):
            raise ValueError(
                f"alpha values must be in [0, 1]; got {self._alpha.tolist()}"
            )
        self.alpha: np.ndarray = self._alpha

        n_arrays = len(init_params(layer_sizes, seed=0))
        n_personalized_arrays = 2 * int(personalized_layers)  # (W, b) per layer
        if not (0 < n_personalized_arrays < n_arrays):
            raise ValueError(
                f"personalized_layers={personalized_layers} leaves "
                f"{n_arrays - n_personalized_arrays} shared array(s); must be "
                f"strictly between 0 and {n_arrays // 2} layers."
            )
        self._n_shared = n_arrays - n_personalized_arrays
        self.personalized_params: List[Params] = []

    def _split(self, params: Params) -> Tuple[Params, Params]:
        """Split a full parameter list into (shared, personalized head)."""
        return list(params[: self._n_shared]), list(params[self._n_shared:])

    def _initial_state(
        self, rng: np.random.Generator,
    ) -> Tuple[Params, List[Params], Params]:
        """Fresh (shared_global, personalized, g_G) state at round 0.

        ``g_G`` is the Alg. 3 gradient memory, initialized to the zero vector
        (Alg. 3 step 3). Personalized heads are independent per-client copies
        of the initial head (never aggregated -- parameter split).
        """
        init_full = init_params(self._model.layer_sizes, seed=self.cfg.seed)
        shared_global, init_head = self._split(init_full)
        personalized = [
            clone_params(init_head) for _ in range(len(self.split.clients))
        ]
        g_G = [np.zeros_like(a) for a in shared_global]
        return shared_global, personalized, g_G

    def _head_fine_tune(
        self,
        shared_global: Params,
        head_params: Params,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[Params, float]:
        """Alg. 3 steps 11-14: fine-tune the personalized head on validation.

        Returns ``(fine_tuned_head, delta_L_val)`` where
        ``delta_L_val = L_before - L_after``. Only the personalized (head)
        arrays are optimized -- the shared parameters never move here. A
        client with an empty validation slice (only possible on a degenerate
        1-sample shard) is skipped with ``delta_L_val = 0``.
        """
        if len(y_val) == 0:
            return clone_params(head_params), 0.0
        L_before = self._model.loss(
            shared_global + list(head_params), X_val, y_val
        )
        ft = clone_params(head_params)
        for _ in range(self.personalize_steps):
            _, grads = self._model.loss_and_grad(shared_global + ft, X_val, y_val)
            for j in range(len(ft)):
                ft[j] -= self.cfg.eta * grads[self._n_shared + j]
        L_after = self._model.loss(shared_global + ft, X_val, y_val)
        return ft, L_before - L_after

    def _one_round(
        self,
        rnd: int,
        shared_global: Params,
        personalized: List[Params],
        g_G: Params,
        rng: np.random.Generator,
    ) -> Tuple[Params, List[Params], Params, np.ndarray, dict]:
        """One full FedPGO round; the unit ``train()`` loops over.

        Returns ``(shared_global, personalized, g_G, w, info)`` with the state
        advanced by one round and ``info`` carrying the per-client mechanism
        internals (``tau_list``, ``g_prime_list``, ``delta_l``, ``scores``)
        for tests and audit -- discarded by ``train()``.

        The gradient memory ``g_G`` is the stateful part of the mechanism
        (Alg. 3 step 23): it is **not** recomputed from this round's deltas --
        it is the previous round's aggregated ``g'_G`` (zero at round 1).
        """
        cfg = self.cfg
        updated_full = [
            client_train_round(
                self._model,
                self._client_tf_optimizers[client.client_id],
                client.X_train, client.y_train,
                cfg, shared_global + personalized[i], rng,
            )
            for i, client in enumerate(self.split.clients)
        ]
        updated_shared = []
        trained_head = []
        for i, full in enumerate(updated_full):
            shared_i, head_i = self._split(full)
            updated_shared.append(shared_i)
            trained_head.append(head_i)

        tau_list = [
            [u - g for u, g in zip(shared_i, shared_global)]
            for shared_i in updated_shared
        ]

        # Alg. 3 steps 11-14: head-only fine-tune on the client's own
        # validation slice; the fine-tuned head persists as the client's
        # personalized parameters.
        delta_l = []
        for i, client in enumerate(self.split.clients):
            ft, dl = self._head_fine_tune(
                shared_global, trained_head[i], client.X_val, client.y_val,
            )
            personalized[i] = ft
            delta_l.append(dl)

        # Eqs. 6-8 + Eq. 9 + Eq. 10, per the module docstring.
        decomposed = [
            fedpgo_decompose(tau_k, g_G, float(self.alpha[i]), self.eps)
            for i, tau_k in enumerate(tau_list)
        ]
        g_prime_list = [d[2] for d in decomposed]
        g_perp_list = [d[1] for d in decomposed]
        scores = [
            fedpgo_pgf_score(
                tau_k, g_G, g_perp,
                delta_l[i],
                lambda1=self.lambda1, lambda2=self.lambda2,
                lambda3=self.lambda3, eps=self.eps,
            )
            for i, (tau_k, g_perp) in enumerate(zip(tau_list, g_perp_list))
        ]
        w = fedpgo_weights(scores, eps=self.eps)

        # Alg. 2 steps 7-8 (Eq. 10 weights applied to the recomposed g'_i).
        agg_delta = aggregate(g_prime_list, w)
        shared_global = [
            g + self.server_lr * d for g, d in zip(shared_global, agg_delta)
        ]
        # Alg. 3 step 23: g_G <- g'_G (the UNSCALED aggregated delta; the
        # server learning rate applies only to the parameter update above).
        g_G = [d.copy() for d in agg_delta]

        info = {
            "rnd": rnd,
            "tau_list": tau_list,
            "g_prime_list": g_prime_list,
            "delta_l": delta_l,
            "scores": scores,
        }
        return shared_global, personalized, g_G, w, info

    def _evaluate(self, full_params_list: Sequence[Params]) -> Tuple[float, float]:
        all_a = float(np.mean([
            accuracy(self._model, p, self.split.X_val, self.split.y_val)
            for p in full_params_list
        ]))
        last_a = float(np.mean([
            last_n_class_accuracy(
                self._model, p, self.split.X_val, self.split.y_val,
                self.rare_class_ids,
            )
            for p in full_params_list
        ]))
        return all_a, last_a

    def train(self, eval_every: int = 20, verbose: bool = False) -> TrainHistory:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        K = len(self.split.clients)

        shared_global, personalized, g_G = self._initial_state(rng)

        w = np.full(K, 1.0 / K, dtype=np.float32)
        history = TrainHistory(method="fedpgo", gamma=None, n_rare=cfg.n_rare)
        full_params = [shared_global + personalized[i] for i in range(K)]
        all_a, last_a = self._evaluate(full_params)
        history.record(0, all_a, last_a, w)

        for rnd in range(1, cfg.rounds + 1):
            shared_global, personalized, g_G, w, _ = self._one_round(
                rnd, shared_global, personalized, g_G, rng,
            )

            if rnd % eval_every == 0 or rnd == cfg.rounds:
                full_params = [shared_global + personalized[i] for i in range(K)]
                all_a, last_a = self._evaluate(full_params)
                history.record(rnd, all_a, last_a, w)
                if verbose:
                    print(
                        f"  [fedpgo a={float(self.alpha[0]):.2f}] round {rnd:3d}"
                        f"/{cfg.rounds}"
                        f"  mean-all={all_a:.3f}  mean-last-n={last_a:.3f}",
                        flush=True,
                    )

        self.gradient_memory = g_G
        self.personalized_params = [shared_global + personalized[i] for i in range(K)]
        w_fedavg = np.asarray(self.client_sizes, dtype=np.float32)
        w_fedavg = w_fedavg / w_fedavg.sum()
        self.final_params = aggregate(self.personalized_params, w_fedavg)
        return history
