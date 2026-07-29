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
Citation-verification status and a corrected mistake, disclosed in full
================================================================================
**Full text is genuinely inaccessible.** ``Final_Stage_C.md`` records the
paper as access-blocked across six independent channels (Unpaywall,
OpenAIRE, CORE, the author's institutional repository, direct SSRN for the
author's earlier preprint, direct Elsevier), with the citation itself
triple-confirmed by metadata alone: DBLP `journals/cn/BikilaC26`, OpenAlex
`W7134986038`, Semantic Scholar `CorpusId:286471251`.

**This is a reconstruction from that literature review's own mechanism
description, not from the paper's equations**, which nobody working on
this project has read. The description this class implements, verbatim
from ``Final_Stage_C.md``'s baseline-tracking table: *"AGO decomposes
local gradients into components parallel/orthogonal to the global
gradient and adaptively reweights; PGF scores clients on gradient
directionality + orthogonality; parameter-split architecture."* That is
four named pieces -- a decomposition, an adaptive reweighting of the
decomposed pieces, a client-scoring rule built from directionality and
orthogonality, and a personalization split -- and this file implements
exactly those four, in the most literal reading of each phrase, with
every free parameter (how "adaptively" reweights, how the two scores
combine, how many layers are personalized) disclosed below as a
constructor default, not a paper claim.

**A mistake worth recording plainly, since this project's own culture is
built around catching exactly this class of error.** An earlier pass this
session, searching independently for FedPGO, found a search-engine summary
describing an "Adaptive Gradient Orthogonalization (AGO)" / "Personalized
Gradient Fusion (PGF)" mechanism, followed one of the result links to a
*different* paper (Sun, Zhou & Wang, "Robust intrusion detection based on
personalized federated learning for IoT environment," *Computers &
Security*, DOI 10.1016/j.cose.2025.104442), and concluded -- wrongly --
that the AGO/PGF description belonged to that other paper, and therefore
that implementing "FedPGO" from it would be a citation fabrication. On
closer reading of this project's own already-completed, triple-source-
verified literature review (which independently arrived at the identical
AGO/PGF description, attached to the correct citation, with a DOI, a DBLP
key, and a Semantic Scholar corpus ID), that conclusion was corrected: the
mechanism description is genuine and does belong to FedPGO. The error was
failing to verify which underlying source a synthesized search summary
actually drew from before dismissing it -- the opposite failure mode from
fabrication, but a failure of verification discipline all the same, and
recorded here rather than quietly fixed.

================================================================================
The four pieces, and the judgment call behind each
================================================================================

1. **AGO's decomposition.** Each client's shared-parameter update
   ``tau_k = theta_k_shared - theta_g_shared`` is split into a component
   parallel to the round's *global* update direction ``tau_g`` (the
   FedAvg-weighted mean of every ``tau_k``) and a component orthogonal to
   it: ``tau_k_par = (tau_k . tau_g / ||tau_g||^2) * tau_g``,
   ``tau_k_orth = tau_k - tau_k_par``. Standard vector projection; not a
   judgment call.

2. **AGO's "adaptive reweighting."** The description does not say *what*
   is reweighted by *what factor*. This class reweights the orthogonal
   component by a scalar ``orthogonal_weight`` (default 0.5, a constructor
   argument) before it contributes to the client's *effective* update
   used in PGF's scoring and in the final aggregate:
   ``tau_k_eff = tau_k_par + orthogonal_weight * tau_k_orth``. A value
   below 1 reads "AGO" as *damping* directions that disagree with the
   group consensus (matching FedPGO's stated goal of mitigating gradient
   *dominance* -- an orthogonal component large enough to fight the
   consensus direction is exactly what a dominance-mitigating mechanism
   would suppress); a value above 1 would instead amplify disagreement.
   This is a judgment call, disclosed as such, not a quoted formula.

3. **PGF's scoring.** "Scores clients on gradient directionality +
   orthogonality" is implemented as
   ``score_k = relu(cos(tau_k, tau_g)) + orthogonality_weight_score *
   (||tau_k_orth|| / (||tau_k|| + eps))`` -- a directionality term
   (identical in spirit to FedAWA's/FLTrust's ReLU-clipped cosine
   similarity, already precedented in this repository) plus an
   orthogonality-fraction term, then normalized to a simplex via
   ``score_k / sum_j score_j`` (falling back to uniform if every score is
   ~0, the same degenerate-round convention
   ``fedgraa/baselines_t3_trust.py`` already uses). ``orthogonality_weight_score``
   defaults to 1.0 and is a separate constructor argument from AGO's
   ``orthogonal_weight`` -- the two names sound similar but control
   different steps (reweighting the update itself vs. scoring the
   client), and are kept as separate parameters rather than aliased so
   that distinction cannot silently collapse.

4. **Parameter-split personalization.** The trailing ``personalized_layers``
   layers (default 1 -- the output layer) are never aggregated: each
   client keeps and returns its own copy, trained locally every round from
   wherever it left off, exactly matching this repository's existing
   personalization precedent in :class:`fedgraa.baselines_t5_adaptive.DAPFLTrainer`
   (``self.personalized_params``, mean-over-clients reporting). Only the
   leading (shared) layers are aggregated by PGF's weights and folded back
   into the next round's broadcast.

**Reporting convention (a choice made here, not a paper claim, matching
DA-PFL's own precedent):** recorded accuracy is the mean over the K
personalized models on the shared validation split; the recorded weight
vector is PGF's own ``score`` simplex; ``final_params`` is the
FedAvg-weighted mean of the personalized models (for any caller that needs
one representative parameter set); per-client models remain on
``self.personalized_params``.

Not part of the base FedGRAA paper's Figures 1-3 -- this is a Related-Work
(tier T5) comparison.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

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

__all__ = ["FedPGOTrainer", "fedpgo_decompose", "fedpgo_scores"]


def fedpgo_decompose(
    tau_k: Sequence[np.ndarray],
    tau_g: Sequence[np.ndarray],
    orthogonal_weight: float,
    eps: float = 1e-12,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """AGO's parallel/orthogonal split plus the reweighted effective update.

    Returns ``(parallel, orthogonal, effective)``, each a list of per-array
    deltas matching ``tau_k``'s own layer structure. The projection is taken
    over the FLATTENED concatenation of every array (a single scalar
    projection coefficient per client, not one per layer), since "the
    global gradient" in the source description is a single direction, not a
    per-layer family of them.
    """
    flat_k = np.concatenate([a.ravel() for a in tau_k])
    flat_g = np.concatenate([a.ravel() for a in tau_g])
    denom = float(np.dot(flat_g, flat_g))
    coeff = float(np.dot(flat_k, flat_g)) / denom if denom > eps else 0.0

    parallel = [coeff * g for g in tau_g]
    orthogonal = [k - coeff * g for k, g in zip(tau_k, tau_g)]
    effective = [p + orthogonal_weight * o for p, o in zip(parallel, orthogonal)]
    return parallel, orthogonal, effective


def fedpgo_scores(
    tau_list: Sequence[Sequence[np.ndarray]],
    tau_g: Sequence[np.ndarray],
    orthogonality_weight_score: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """PGF's directionality + orthogonality client score, normalized to a
    simplex.

    Degenerate case (every client scores ~0, e.g. every ``tau_k`` is exactly
    zero at round 0 before any local training has happened): falls back to
    uniform weights, matching ``fedgraa/baselines_t3_trust.py``'s existing
    convention for the same degeneracy rather than dividing by ~zero.
    """
    flat_g = np.concatenate([a.ravel() for a in tau_g])
    norm_g = float(np.linalg.norm(flat_g))

    scores = np.zeros(len(tau_list), dtype=np.float64)
    for i, tau_k in enumerate(tau_list):
        flat_k = np.concatenate([a.ravel() for a in tau_k])
        norm_k = float(np.linalg.norm(flat_k))
        if norm_k < eps or norm_g < eps:
            cos_sim = 0.0
        else:
            cos_sim = float(np.dot(flat_k, flat_g)) / (norm_k * norm_g)
        directionality = max(0.0, cos_sim)

        denom = float(np.dot(flat_g, flat_g))
        coeff = float(np.dot(flat_k, flat_g)) / denom if denom > eps else 0.0
        flat_orth = flat_k - coeff * flat_g
        orth_frac = float(np.linalg.norm(flat_orth)) / (norm_k + eps)

        scores[i] = directionality + orthogonality_weight_score * orth_frac

    total = scores.sum()
    if total <= eps:
        return np.full(len(tau_list), 1.0 / len(tau_list), dtype=np.float32)
    return (scores / total).astype(np.float32)


class FedPGOTrainer:
    """FedPGO baseline -- see this module's docstring for the full
    citation-verification and reconstruction disclosure.
    """

    def __init__(
        self,
        split: FederatedSplit,
        cfg: FedConfig,
        orthogonal_weight: float = 0.5,
        orthogonality_weight_score: float = 1.0,
        personalized_layers: int = 1,
        server_lr: float = 1.0,
        eps: float = 1e-12,
    ):
        self.split = split
        self.cfg = cfg
        self.orthogonal_weight = float(orthogonal_weight)
        self.orthogonality_weight_score = float(orthogonality_weight_score)
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
        return list(params[: self._n_shared]), list(params[self._n_shared:])

    def _evaluate(self, full_params_list: Sequence[Params]) -> Tuple[float, float]:
        all_a = float(np.mean([
            accuracy(self._model, p, self.split.X_val, self.split.y_val)
            for p in full_params_list
        ]))
        last_a = float(np.mean([
            last_n_class_accuracy(
                self._model, p, self.split.X_val, self.split.y_val, self.rare_class_ids,
            )
            for p in full_params_list
        ]))
        return all_a, last_a

    def train(self, eval_every: int = 20, verbose: bool = False) -> TrainHistory:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        K = len(self.split.clients)

        init_full = init_params(self._model.layer_sizes, seed=cfg.seed)
        shared_global, _ = self._split(init_full)
        personalized = [self._split(init_full)[1] for _ in range(K)]

        w = np.full(K, 1.0 / K, dtype=np.float32)
        history = TrainHistory(method="fedpgo", gamma=None, n_rare=cfg.n_rare)
        full_params = [shared_global + personalized[i] for i in range(K)]
        all_a, last_a = self._evaluate(full_params)
        history.record(0, all_a, last_a, w)

        for rnd in range(1, cfg.rounds + 1):
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
            for i, full in enumerate(updated_full):
                shared_i, personal_i = self._split(full)
                updated_shared.append(shared_i)
                personalized[i] = personal_i  # parameter-split: never aggregated

            tau_list = [
                [u - g for u, g in zip(shared_i, shared_global)]
                for shared_i in updated_shared
            ]
            fedavg_w = np.asarray(self.client_sizes, dtype=np.float64)
            fedavg_w = fedavg_w / fedavg_w.sum()
            tau_g = [
                sum(fedavg_w[i] * tau_list[i][a] for i in range(K))
                for a in range(len(tau_list[0]))
            ]

            effective_list = [
                fedpgo_decompose(tau_k, tau_g, self.orthogonal_weight, self.eps)[2]
                for tau_k in tau_list
            ]
            w = fedpgo_scores(tau_list, tau_g, self.orthogonality_weight_score, self.eps)

            agg_delta = aggregate(effective_list, w)
            shared_global = [
                g + self.server_lr * d for g, d in zip(shared_global, agg_delta)
            ]

            if rnd % eval_every == 0 or rnd == cfg.rounds:
                full_params = [shared_global + personalized[i] for i in range(K)]
                all_a, last_a = self._evaluate(full_params)
                history.record(rnd, all_a, last_a, w)
                if verbose:
                    print(
                        f"  [fedpgo ow={self.orthogonal_weight}] round {rnd:3d}/{cfg.rounds}"
                        f"  mean-all={all_a:.3f}  mean-last-n={last_a:.3f}",
                        flush=True,
                    )

        self.personalized_params = [shared_global + personalized[i] for i in range(K)]
        w_fedavg = np.asarray(self.client_sizes, dtype=np.float32)
        w_fedavg = w_fedavg / w_fedavg.sum()
        self.final_params = aggregate(self.personalized_params, w_fedavg)
        return history
