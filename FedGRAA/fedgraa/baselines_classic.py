"""Classic (non-personalization, non-NIDS-specific) FL optimizer baselines
(docs/research_c/STAGE_C_REMAINING_WORK_MASTER_PLAN_2026-08-02.md §4).

`Final_Stage_C.md` item 2's own priority list names FedAvg, FedProx,
SCAFFOLD, FedNova, and FedOpt ahead of the 14 T3/T5 external NIDS-specific
baselines. FedAvg/FedProx/FedOpt already exist elsewhere in this codebase
(:class:`fedgraa.trainer.FederatedTrainer` with ``method="fedavg"``,
:class:`fedgraa.baselines.FedProxTrainer`,
:class:`fedgraa.baselines.FedOptServer`) but had never been run on the v2
privacy-compliant harness's arm-dispatch path (``BASELINE_ARM_CLASSES`` in
``scripts/run_v2_frontier_sweep.py``) before this module. This module:

* :class:`FedAvgTrainer` -- a thin 2-arg (``split, cfg``) adapter around
  ``FederatedTrainer(method="fedavg")`` so FedAvg is dispatchable through the
  same ``trainer_cls(split, cfg)`` construction every other
  ``BASELINE_ARM_CLASSES`` entry uses (FedProx/FedOpt already match that
  shape -- ``mu``/``server_lr``/``tau`` all have defaults -- so they need no
  wrapper, only the new arm-name registration in
  ``scripts/run_v2_frontier_sweep.py``).
* :class:`SCAFFOLDTrainer` -- Karimireddy et al., "SCAFFOLD: Stochastic
  Controlled Averaging for Federated Learning", ICML 2020, arXiv:1910.06378.
  Built from scratch; no code for this method existed anywhere in this
  repository before this module.
* :class:`FedNovaTrainer` -- Wang et al., "Tackling the Objective
  Inconsistency Problem in Heterogeneous Federated Optimization", NeurIPS
  2020, arXiv:2007.07481. Also built from scratch.

--------------------------------------------------------------------------
CITATION-VERIFICATION DISCLOSURE
--------------------------------------------------------------------------
Both papers' algorithms were read off their arXiv HTML full text (SCAFFOLD:
arxiv.org/abs/1910.06378 Algorithm 1; FedNova: arxiv.org/abs/2007.07481
Section 3/Algorithm 1), not from memory or an abstract-level summary, per
this project's own standing rule (`STAGE_C_FEDPGO_FAITHFUL_PLAN_2026-07-31.md`
§2). Every place this codebase's shared harness does not map 1:1 onto either
paper's own assumptions is disclosed below rather than silently smoothed
over -- the same discipline every T3/T5 baseline module in this repo already
applies.

**Shared deviation, both methods (JC-TC-1).** Every baseline in this
codebase -- FedAvg, FedProx, FedOpt, and now these two -- trains each client
locally with the same Adam-based fused step
(:func:`fedgraa.model._fused_client_step` /
:func:`fedgraa.model._fused_proximal_step` /
:func:`fedgraa.model._fused_scaffold_step`), never the plain (or momentum)
SGD both SCAFFOLD's and FedNova's own convergence analyses are written
against. This is not a new deviation introduced here -- FedProx already
applies its proximal correction through the same shared Adam optimizer,
disclosed nowhere before this file because no one had occasion to state the
general rule -- it is stated once, here, for both new methods rather than
per-method.

A second-order consequence of JC-TC-1 specific to SCAFFOLD: each client's
Adam optimizer state (first/second moment buffers) is never reset between
communication rounds -- the same pre-existing, already-tracked repo-wide
pattern as Stage A/B's "Gap 2.2" (:mod:`fedgraa.trainer`'s own
``resume_or_init``). For SCAFFOLD specifically this means the momentum
buffers absorb the history of a *per-round-varying* correction term
(``c_global - c_local``, itself drifting round to round as the control
variates update), not merely the raw client gradient. This is a genuine,
if minor, deviation beyond "Adam vs. SGD" alone and was not called out
separately before this note (added 2026-08-03, per the independent Stage C
audit).

**SCAFFOLD-specific (JC-TC-2..5).**

* *(JC-TC-2)* **Option II** control-variate update is used (`c_i^+ <- c_i -
  c + (1/(K eta_l))(x - y_i)`), not Option I (`c_i^+ <- g_i(x)`, an extra
  full-batch gradient evaluation at the pre-round global point) -- Option II
  is the paper's own cheaper alternative and avoids a second forward/backward
  pass per client per round.
* *(JC-TC-3)* **K** (the paper's local-step count) is the *actual* number of
  fused optimizer steps a client takes this round --
  ``cfg.local_epochs * ceil(n_i / cfg.batch_size)`` -- read directly off the
  same loop structure :func:`fedgraa.trainer.client_train_round` uses, not
  assumed constant across clients (it isn't: this codebase's non-IID
  partition gives clients different ``n_i``). **eta_l** is taken as
  ``cfg.eta``, the Adam learning rate every client optimizer is configured
  with -- Option II's formula is written for a literal SGD step size, so
  reusing Adam's configured (but not literally-applied, since Adam's real
  per-parameter step is adaptive) learning rate here is a numerical
  substitution, not an exact match to the paper's own quantity.
* *(JC-TC-4)* **eta_g = 1.0** (the server/global step size). The paper gives
  only a convergence-bound guidance (``eta_g >= sqrt(|S|)``, for the
  partial-participation regime its theorems analyze), not a fixed
  recommended value; ``1.0`` is the standard full-participation default used
  by the reference implementations of this method in the wider literature.
* *(JC-TC-5)* **Full participation.** Every trainer in this codebase
  (FedAvg, FedGRAA, FedGRAA-v2, every T3/T5 baseline) trains all clients
  every round -- there is no client-subsampling path anywhere in this
  repository. SCAFFOLD's Algorithm 1 is written for a sampled subset ``S``;
  with ``S = all clients`` (``|S| = N`` every round), its ``(|S|/N)`` server
  control-variate scaling factor is exactly ``1`` and the sampled-subset
  form specializes to the full-participation form implemented here -- not an
  approximation, since the paper's own algorithm reduces to this when
  sampling is disabled.

**FedNova-specific (JC-TC-6..7).**

* *(JC-TC-6)* The accumulation vector is ``a_i = [1, ..., 1]`` (Section 3's
  base case, "for vanilla SGD"), since -- per JC-TC-1 -- the actual local
  solver is Adam, not vanilla SGD, and the paper only derives closed-form
  ``a_i`` vectors for SGD/proximal/momentum-SGD variants, none of which is
  Adam. ``d_i`` is still computed empirically as the *observed* per-client
  parameter delta divided by ``tau_i`` (the paper's own general definition,
  Eq. in Sec 3.1: ``d_i = (G_i a_i)/||a_i||_1``), which is well-defined
  regardless of which local solver produced the columns of ``G_i``.
* *(JC-TC-7)* ``tau_i`` is computed identically to SCAFFOLD's ``K`` above
  (JC-TC-3) -- ``cfg.local_epochs * ceil(n_i / cfg.batch_size)`` -- and is
  genuinely heterogeneous across clients because of this codebase's non-IID
  client-size partition, so FedNova's normalized averaging is a real,
  non-degenerate reproduction here (it does not collapse to FedAvg's plain
  weighted average, which is exactly the objective-inconsistency gap the
  paper exists to close).

Both classes follow the same shape as every other baseline in this
codebase (``__init__(split, cfg, ...)`` + ``train(eval_every, verbose) ->
TrainHistory``), so all three are directly usable through
``scripts/run_v2_frontier_sweep.py``'s ``BASELINE_ARM_CLASSES`` dispatch,
next to the 14 T3/T5 arms.

Like ``fedgraa/baselines.py``, none of this is part of the base paper's
Figures 1-3 -- these are Related Work comparisons only (§4's classic-FL-
baseline coverage decision).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
import tensorflow as tf

from .config import FedConfig
from .metrics import accuracy, last_n_class_accuracy
from .model import (
    MLP,
    Params,
    _fused_scaffold_step,
    build_tf_optimizer,
    init_params,
)
from .partition import FederatedSplit
from .trainer import FederatedTrainer, TrainHistory, client_train_round

__all__ = [
    "FedAvgTrainer", "SCAFFOLDTrainer", "FedNovaTrainer", "n_local_steps",
    "scaffold_option2_control_variate", "scaffold_server_update",
    "fednova_normalized_update",
]


def n_local_steps(n: int, cfg: FedConfig) -> int:
    """Number of fused local-optimizer steps a client with ``n`` training
    rows actually takes this round, matching
    :func:`fedgraa.trainer.client_train_round`'s own loop structure exactly
    (``local_epochs`` outer passes, ``ceil(n / batch_size)`` mini-batches
    each, including a final partial batch). Zero for an empty shard, mirroring
    ``client_train_round``'s own ``if n == 0: return unchanged`` contract.
    """
    if n <= 0:
        return 0
    return cfg.local_epochs * math.ceil(n / cfg.batch_size)


def _mean_delta(per_client: Sequence[Params], base: Params) -> Params:
    """``mean_i(per_client[i] - base)``, per parameter array -- the
    full-participation server update every trainer in this module needs
    (SCAFFOLD's ``(1/|S|) sum(y_i - x)``, and, via this same helper, the
    server-side control-variate delta ``(1/N) sum(c_i^+ - c_i)``)."""
    n = len(per_client)
    out: Params = []
    for j in range(len(base)):
        stacked = np.stack(
            [per_client[i][j].astype(np.float32) - base[j] for i in range(n)],
            axis=0,
        )
        out.append(stacked.mean(axis=0))
    return out


def scaffold_option2_control_variate(
    c_i_old: Params, c_global: Params, x: Params, y_i: Params,
    K: int, eta_l: float,
) -> Params:
    """SCAFFOLD's Option II control-variate update (Algorithm 1):
    ``c_i^+ <- c_i - c + (1/(K eta_l))(x - y_i)``. Pure function of its
    inputs -- independently testable against a hand-computed value (JC-TC-2).
    """
    Kel = max(K, 1) * eta_l
    return [
        ci - cg + (xj - yj) / Kel
        for ci, cg, xj, yj in zip(c_i_old, c_global, x, y_i)
    ]


def scaffold_server_update(
    global_params: Params, y_list: Sequence[Params], eta_g: float,
) -> Params:
    """SCAFFOLD's full-participation server update:
    ``x <- x + eta_g * (1/|S|) sum(y_i - x)`` (Algorithm 1, JC-TC-4/5)."""
    delta = _mean_delta(y_list, global_params)
    return [gp + eta_g * d for gp, d in zip(global_params, delta)]


def fednova_normalized_update(
    global_params: Params, client_params: Sequence[Params],
    client_sizes: Sequence[int], taus: Sequence[float],
) -> Params:
    """FedNova's normalized-averaging server update (Section 3):
    ``x^(t+1) = x^t + tau_eff * sum_i p_i * (Delta_i / tau_i)``, with
    ``p_i = n_i / sum(n)`` and ``tau_eff = sum_i p_i tau_i``. Clients with
    ``tau_i <= 0`` (empty shard) are excluded from the sum, matching
    :func:`n_local_steps`'s own zero-step contract (JC-TC-7). Pure function
    of its inputs -- independently testable against a hand-computed value.
    """
    sizes = np.asarray(client_sizes, dtype=np.float64)
    p = sizes / sizes.sum()
    taus_arr = np.asarray(taus, dtype=np.float64)
    tau_eff = float(np.sum(p * taus_arr))

    n_arrays = len(global_params)
    out: Params = []
    for j in range(n_arrays):
        acc = np.zeros_like(global_params[j], dtype=np.float32)
        for i in range(len(client_params)):
            if taus_arr[i] <= 0:
                continue
            delta_i = client_params[i][j].astype(np.float32) - global_params[j]
            acc += np.float32(p[i] / taus_arr[i]) * delta_i
        out.append(global_params[j] + np.float32(tau_eff) * acc)
    return out


# --------------------------------------------------------------------------- #
# FedAvg: thin adapter onto the existing v1-era trainer
# --------------------------------------------------------------------------- #
class FedAvgTrainer:
    """2-arg (``split, cfg``) adapter exposing FedAvg through the
    ``BASELINE_ARM_CLASSES`` dispatch shape every T3/T5/classic baseline in
    this codebase uses. Delegates entirely to
    :class:`fedgraa.trainer.FederatedTrainer` (``method="fedavg"``), which
    already implements eq. 5's fixed data-size aggregation weights and was
    already validated against FedGRAA-v1 in Stage B's own ablation -- this
    class adds no new mechanism, only the constructor shape the v2 harness's
    arm dispatch requires.
    """

    def __init__(self, split: FederatedSplit, cfg: FedConfig):
        self._inner = FederatedTrainer(split, cfg, method="fedavg")

    def train(self, eval_every: int = 20, verbose: bool = False) -> TrainHistory:
        history = self._inner.train(eval_every=eval_every, verbose=verbose)
        self.final_params = self._inner.final_params
        return history


# --------------------------------------------------------------------------- #
# SCAFFOLD (Karimireddy et al., ICML 2020)
# --------------------------------------------------------------------------- #
class SCAFFOLDTrainer:
    """SCAFFOLD baseline: control-variate-corrected local training +
    full-participation server/control-variate averaging (Algorithm 1). See
    this module's docstring for JC-TC-1..5, the disclosed judgment calls."""

    def __init__(self, split: FederatedSplit, cfg: FedConfig, eta_g: float = 1.0):
        self.split = split
        self.cfg = cfg
        self.eta_g = eta_g
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
        init = init_params(layer_sizes, seed=cfg.seed)
        # c^0 = 0, c_i^0 = 0 -- Algorithm 1's own stated initialization.
        self._c_global: Params = [np.zeros_like(p) for p in init]
        self._c_local: Dict[int, Params] = {
            c.client_id: [np.zeros_like(p) for p in init] for c in split.clients
        }

    def _client_round(
        self, client, global_params: Params, rng: np.random.Generator,
    ) -> tuple:
        model = self._model
        opt = self._client_tf_optimizers[client.client_id]
        model.set_params(global_params)
        X, y = client.X_train, client.y_train
        n = len(y)
        c_i_old = self._c_local[client.client_id]
        if n == 0:
            # No local data: unchanged params, unchanged control variate --
            # mirrors client_train_round's own n==0 contract.
            return model.get_params(), c_i_old

        cfg = self.cfg
        c_global_t = [tf.constant(c, dtype=tf.float32) for c in self._c_global]
        c_local_t = [tf.constant(c, dtype=tf.float32) for c in c_i_old]

        pad_x = np.zeros((cfg.batch_size, X.shape[1]), dtype=np.float32)
        pad_y = np.zeros(cfg.batch_size, dtype=np.int32)
        all_perm = np.array(
            [rng.permutation(n) for _ in range(cfg.local_epochs)], dtype=np.int32,
        )
        n_steps = 0
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
                _fused_scaffold_step(
                    tf.constant(x_batch, dtype=tf.float32),
                    tf.constant(y_batch, dtype=tf.int32),
                    model._model, opt, c_global_t, c_local_t,
                    tf.constant(n_valid, dtype=tf.int32),
                )
                n_steps += 1

        y_i = model.get_params()
        c_i_new = scaffold_option2_control_variate(
            c_i_old, self._c_global, global_params, y_i, n_steps, cfg.eta,
        )
        return y_i, c_i_new

    def train(self, eval_every: int = 20, verbose: bool = False) -> TrainHistory:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        global_params: Params = init_params(self._model.layer_sizes, seed=cfg.seed)

        from .aggregation import fedavg_weights
        w = fedavg_weights(self.client_sizes)  # reporting only; see below
        history = TrainHistory(method="scaffold", gamma=None, n_rare=cfg.n_rare)
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
            y_list: List[Params] = []
            c_new_list: List[Params] = []
            for client in self.split.clients:
                y_i, c_i_new = self._client_round(client, global_params, rng)
                y_list.append(y_i)
                c_new_list.append(c_i_new)

            # x <- x + eta_g * (1/|S|) * sum(y_i - x) -- full participation.
            global_params = scaffold_server_update(global_params, y_list, self.eta_g)
            # c <- c + (|S|/N) * (1/|S|) sum(c_i^+ - c_i) == c + mean(c_i^+ - c_i)
            # when |S| == N (full participation, JC-TC-5).
            c_old_list = [self._c_local[c.client_id] for c in self.split.clients]
            delta_c = _mean_delta(c_new_list, [np.zeros_like(a) for a in self._c_global])
            delta_c_old = _mean_delta(c_old_list, [np.zeros_like(a) for a in self._c_global])
            self._c_global = [
                cg + (dn - do) for cg, dn, do in zip(self._c_global, delta_c, delta_c_old)
            ]
            for client, c_new in zip(self.split.clients, c_new_list):
                self._c_local[client.client_id] = c_new

            if rnd % eval_every == 0 or rnd == cfg.rounds:
                all_a = accuracy(self._model, global_params, self.split.X_val, self.split.y_val)
                last_a = last_n_class_accuracy(
                    self._model, global_params, self.split.X_val, self.split.y_val,
                    self.rare_class_ids,
                )
                history.record(rnd, all_a, last_a, w)
                if verbose:
                    print(
                        f"  [scaffold] round {rnd:3d}/{cfg.rounds}"
                        f"  all={all_a:.3f}  last-n={last_a:.3f}",
                        flush=True,
                    )

        self.final_params = global_params
        return history


# --------------------------------------------------------------------------- #
# FedNova (Wang et al., NeurIPS 2020)
# --------------------------------------------------------------------------- #
class FedNovaTrainer:
    """FedNova baseline: normalized-averaging aggregation that corrects for
    heterogeneous per-client local-step counts (Section 3). See this
    module's docstring for JC-TC-1, JC-TC-6..7, the disclosed judgment
    calls."""

    def __init__(self, split: FederatedSplit, cfg: FedConfig):
        self.split = split
        self.cfg = cfg
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

    def train(self, eval_every: int = 20, verbose: bool = False) -> TrainHistory:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        global_params: Params = init_params(self._model.layer_sizes, seed=cfg.seed)

        from .aggregation import fedavg_weights
        w_report = fedavg_weights(self.client_sizes)  # TrainHistory.record's w slot
        history = TrainHistory(method="fednova", gamma=None, n_rare=cfg.n_rare)
        history.record(
            0,
            accuracy(self._model, global_params, self.split.X_val, self.split.y_val),
            last_n_class_accuracy(
                self._model, global_params, self.split.X_val, self.split.y_val,
                self.rare_class_ids,
            ),
            w_report,
        )

        for rnd in range(1, cfg.rounds + 1):
            client_params: List[Params] = [
                client_train_round(
                    self._model,
                    self._client_tf_optimizers[client.client_id],
                    client.X_train, client.y_train,
                    cfg, global_params, rng,
                )
                for client in self.split.clients
            ]
            taus = [n_local_steps(c.n_train, cfg) for c in self.split.clients]
            global_params = fednova_normalized_update(
                global_params, client_params, self.client_sizes, taus,
            )

            if rnd % eval_every == 0 or rnd == cfg.rounds:
                all_a = accuracy(self._model, global_params, self.split.X_val, self.split.y_val)
                last_a = last_n_class_accuracy(
                    self._model, global_params, self.split.X_val, self.split.y_val,
                    self.rare_class_ids,
                )
                history.record(rnd, all_a, last_a, w_report)
                if verbose:
                    sizes = np.asarray(self.client_sizes, dtype=np.float64)
                    p = sizes / sizes.sum()
                    tau_eff = float(np.sum(p * np.asarray(taus, dtype=np.float64)))
                    print(
                        f"  [fednova] round {rnd:3d}/{cfg.rounds}"
                        f"  all={all_a:.3f}  last-n={last_a:.3f}"
                        f"  tau_eff={tau_eff:.1f}",
                        flush=True,
                    )

        self.final_params = global_params
        return history
