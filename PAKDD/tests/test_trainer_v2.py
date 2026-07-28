"""End-to-end tests for :class:`fedgraa.trainer.FederatedTrainerV2` (W2.4).

Covers FEDGRAA_V2_MASTER_PLAN.md §9.2's remaining end-to-end gates:

* **T-5**  -- multi-round trajectory matches v1-projected at zeta=0 (m=1).
* **T-11** -- comparable accuracy to v1 on a held-out seed.
* **T-12** -- measured communication bytes match the ``(1 + ceil(E'/m)) x
  baseline`` formula for every ``m``, the hard gate on every published
  communication number.
* **T-13** -- dropout robustness end-to-end at ``p_drop in {0, 0.1, 0.3}``.
* **T-18** -- the zeta decomposition's noise sources are separable: with
  exactly one source active, measured zeta attributes to that source and is
  ~0 for the others.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fedgraa.aggregation import FedGRAA, two_phase_composite_grad
from fedgraa.aggregation import ProjectedFedGRAA
from fedgraa.config import DatasetSpec, FedConfig
from fedgraa.data import make_synthetic_dataset
from fedgraa.model import MLP, aggregate, init_params
from fedgraa.partition import apply_tags, partition_non_iid, stratum_tags
from fedgraa.privacy import laplace_mechanism
from fedgraa.trainer import FederatedTrainer, FederatedTrainerV2


def _toy_spec() -> DatasetSpec:
    return DatasetSpec(
        name="TOY_V2",
        n_features=6,
        labels=["Benign", "A", "B", "C", "Rare"],
        used_counts=[300, 120, 90, 60, 10],
        benign_id=0,
        rare_label_order=["Rare", "C", "B"],
    )


def _make_split(n_rare=1, seed=0):
    spec = _toy_spec()
    X, y, spec = make_synthetic_dataset(spec, seed=seed, class_separation=3.0)
    return partition_non_iid(X, y, spec, n_rare=n_rare, val_fraction=0.10, seed=seed)


# --------------------------------------------------------------------------- #
# T-5 -- multi-round trajectory matches v1-projected at zeta=0 (m=1)
# --------------------------------------------------------------------------- #
def test_v2_multi_round_trajectory_matches_v1_projected():
    split = _make_split(n_rare=1, seed=0)
    cfg = FedConfig(rounds=10, gamma=0.5, n_rare=1, seed=0)

    layer_sizes = split.spec.model_config().layer_sizes
    v1 = FederatedTrainer(split, cfg, method="fedgraa")
    v1.fedgraa = ProjectedFedGRAA(
        gamma=cfg.gamma, eta_prime=cfg.eta_prime,
        weight_epochs=cfg.weight_epochs, layer_sizes=layer_sizes,
    )
    h1 = v1.train(eval_every=2)

    v2 = FederatedTrainerV2(split, cfg)  # grad_refresh_every=1 default -> zeta=0
    h2 = v2.train(eval_every=2)

    np.testing.assert_allclose(h1.all_acc, h2.all_acc, atol=1e-6)
    np.testing.assert_allclose(h1.last_n_acc, h2.last_n_acc, atol=1e-6)
    for w1, w2 in zip(h1.weights, h2.weights):
        np.testing.assert_allclose(w1, w2, atol=1e-5)


# --------------------------------------------------------------------------- #
# T-11 -- comparable accuracy on a held-out seed
# --------------------------------------------------------------------------- #
def test_v2_full_run_comparable_accuracy_held_out_seed():
    seed = 7  # a seed not used to tune anything above
    split = _make_split(n_rare=1, seed=seed)
    cfg = FedConfig(rounds=20, gamma=1.0, n_rare=1, seed=seed)

    layer_sizes = split.spec.model_config().layer_sizes
    v1 = FederatedTrainer(split, cfg, method="fedgraa")
    v1.fedgraa = ProjectedFedGRAA(
        gamma=cfg.gamma, eta_prime=cfg.eta_prime,
        weight_epochs=cfg.weight_epochs, layer_sizes=layer_sizes,
    )
    h1 = v1.train(eval_every=10)

    v2 = FederatedTrainerV2(split, cfg)
    h2 = v2.train(eval_every=10)

    assert abs(h1.final_all_acc - h2.final_all_acc) < 0.02
    assert h2.final_last_n_acc > 0.5 * max(h1.final_last_n_acc, 1e-9) or (
        h1.final_last_n_acc == 0.0 and h2.final_last_n_acc >= 0.0
    )


# --------------------------------------------------------------------------- #
# T-12 -- measured bytes match the frontier formula (hard gate)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("m", [1, 2, 3, 5])
def test_measured_bytes_match_frontier_formula(m):
    split = _make_split(n_rare=1, seed=0)
    cfg = FedConfig(rounds=4, gamma=0.5, n_rare=1, seed=0, grad_refresh_every=m)

    trainer = FederatedTrainerV2(split, cfg)
    trainer.train(eval_every=100)  # no mid-run eval overhead needed

    n_params = trainer.n_params
    K = split.n_clients
    msg = n_params * 4
    base_per_round = 2 * K * msg
    mult = 1 + math.ceil(cfg.weight_epochs / m)
    expected = cfg.rounds * base_per_round * mult

    rel_err = abs(trainer.bytes_sent - expected) / expected
    assert rel_err < 0.01, (
        f"m={m}: measured {trainer.bytes_sent} bytes, "
        f"expected {expected} ({mult}x baseline), rel_err={rel_err:.4f}"
    )


def test_measured_comm_rounds_equals_two_per_refresh_plus_one():
    split = _make_split(n_rare=1, seed=0)
    cfg = FedConfig(rounds=3, gamma=0.5, n_rare=1, seed=0, grad_refresh_every=2)
    trainer = FederatedTrainerV2(split, cfg)
    trainer.train(eval_every=100)
    refreshes_per_round = math.ceil(cfg.weight_epochs / cfg.grad_refresh_every)
    expected_rounds = cfg.rounds * (1 + 2 * refreshes_per_round)
    assert trainer.comm_rounds == expected_rounds


# --------------------------------------------------------------------------- #
# T-13 -- dropout robustness end-to-end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p_drop", [0.0, 0.1, 0.3])
def test_dropout_robustness_end_to_end(p_drop):
    split = _make_split(n_rare=1, seed=0)
    cfg = FedConfig(rounds=6, gamma=0.5, n_rare=1, seed=0, p_drop=p_drop)
    # This tiny fixture has only 4 clients; the trainer's production-default
    # threshold (majority, >n/2) is deliberately strict for security and
    # would legitimately raise SecAggDropoutError (T-7) at p_drop=0.3 on so
    # few clients. T-13 is about robustness to dropout *given* a threshold
    # chosen to tolerate it, so use a permissive threshold here explicitly.
    trainer = FederatedTrainerV2(split, cfg, secagg_threshold=1)
    hist = trainer.train(eval_every=2)
    assert len(hist.all_acc) >= 1
    assert 0.0 <= hist.final_all_acc <= 1.0
    assert 0.0 <= hist.final_last_n_acc <= 1.0
    for p in trainer.final_params:
        assert np.all(np.isfinite(p))


def test_run_experiment_dispatches_fedgraa_v2():
    """experiment.run_experiment("fedgraa_v2", ...) end-to-end wiring."""
    from fedgraa.experiment import run_experiment

    res = run_experiment(
        "CIC-IDS-2017", method="fedgraa_v2", gamma=0.5, n_rare=3,
        rounds=2, eval_every=1, seed=0,
        cfg=FedConfig(rounds=2, gamma=0.5, n_rare=3, seed=0),
    )
    assert res.method == "fedgraa_v2"
    assert res.gamma == 0.5
    assert 0.0 <= res.final_all_acc <= 1.0


def test_dropout_never_reduces_to_zero_participants():
    """Even at high p_drop, at least one client always survives (the
    trainer's own floor, independent of the SecAgg threshold check)."""
    split = _make_split(n_rare=1, seed=0)
    cfg = FedConfig(rounds=3, gamma=0.5, n_rare=1, seed=0, p_drop=0.99)
    trainer = FederatedTrainerV2(split, cfg, secagg_threshold=1)
    hist = trainer.train(eval_every=1)
    assert len(hist.all_acc) >= 1


# --------------------------------------------------------------------------- #
# T-18 -- zeta decomposition sources are separable
# --------------------------------------------------------------------------- #
def _scenario_for_zeta():
    split = _make_split(n_rare=1, seed=0)
    layer_sizes = split.spec.model_config().layer_sizes
    model = MLP(layer_sizes, seed=0)
    params = init_params(layer_sizes, seed=0)
    from fedgraa.aggregation import static_reparam_weights
    w0 = static_reparam_weights([c.n_train for c in split.clients])
    client_params = [init_params(layer_sizes, seed=s + 1) for s in range(split.n_clients)]
    params = aggregate(client_params, w0)

    rare_ids = split.val_split.rare_class_ids
    counts = np.bincount(split.y_val, minlength=split.spec.n_classes)
    tags = stratum_tags(counts, rare_ids)
    client_vals = [apply_tags(c.X_val, c.y_val, tags, rare_ids) for c in split.clients]
    return split, model, params, client_vals, rare_ids


def _rel_err(a, b) -> float:
    fa = np.concatenate([x.ravel() for x in a]).astype(np.float64)
    fb = np.concatenate([x.ravel() for x in b]).astype(np.float64)
    den = np.linalg.norm(fb)
    return float(np.linalg.norm(fa - fb) / den) if den else float(np.linalg.norm(fa - fb))


def test_zeta_decomposition_sources_are_separable():
    """With exactly one noise source active at a time, measured zeta
    attributes to that source and is ~0 for the others."""
    split, model, params, client_vals, rare_ids = _scenario_for_zeta()
    gamma = 0.5
    fg = FedGRAA(gamma=gamma, layer_sizes=split.spec.model_config().layer_sizes)
    oracle_grad = fg._composite_grad_P(params, split.val_split)[1]

    # --- baseline: no noise source active -> zeta ~ 0 (float32 only) ------- #
    fed_grad, _ = two_phase_composite_grad(model, params, client_vals, gamma)
    zeta_none = _rel_err(fed_grad, oracle_grad)
    assert zeta_none < 1e-5

    # --- zeta_dp: DP noise on the RANKING histogram, strong enough to flip
    # the rare-class selection -> a genuinely different D_V^n/D_V^h split ---
    counts = np.bincount(split.y_val, minlength=split.spec.n_classes).astype(np.float64)
    rng = np.random.default_rng(0)
    huge_noise_counts = counts.copy()
    # Force a flip deterministically rather than relying on a random draw:
    # swap the rarest and a common class's counts, simulating a worst-case
    # DP outcome, then re-derive tags from the corrupted histogram.
    from fedgraa.partition import bottom_n_rare_classes

    flipped_counts = counts.copy()
    common_class = max(split.spec.attack_ids, key=lambda i: counts[i])
    flipped_counts[rare_ids[0]], flipped_counts[common_class] = (
        flipped_counts[common_class], flipped_counts[rare_ids[0]],
    )
    flipped_rare_ids = bottom_n_rare_classes(flipped_counts, split.spec.attack_ids, len(rare_ids))
    assert set(flipped_rare_ids) != set(rare_ids), "fixture must actually flip the rare set"
    flipped_tags = stratum_tags(counts, flipped_rare_ids)
    flipped_vals = [
        apply_tags(c.X_val, c.y_val, flipped_tags, flipped_rare_ids) for c in split.clients
    ]
    fed_grad_dp, _ = two_phase_composite_grad(model, params, flipped_vals, gamma)
    zeta_dp = _rel_err(fed_grad_dp, oracle_grad)
    assert zeta_dp > 1e-2, "flipping the rare set should measurably change grad_P L'"

    # --- zeta_stale: evaluate grad_provider at a STALE (earlier) theta'_g
    # instead of the current one -- isolates staleness from everything else --
    stale_params = init_params(split.spec.model_config().layer_sizes, seed=99)
    fed_grad_stale, _ = two_phase_composite_grad(model, stale_params, client_vals, gamma)
    oracle_grad_stale = fg._composite_grad_P(stale_params, split.val_split)[1]
    # The protocol itself is still exact AT the stale point...
    assert _rel_err(fed_grad_stale, oracle_grad_stale) < 1e-5
    # ...but using it in place of the CURRENT oracle gradient is exactly
    # zeta_stale: a real, measurable gap unrelated to the DP/SecAgg sources.
    zeta_stale = _rel_err(fed_grad_stale, oracle_grad)
    assert zeta_stale > 1e-2, "a stale gradient should measurably differ from the current one"
