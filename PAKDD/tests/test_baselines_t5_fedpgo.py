"""Tests for the FedPGO baseline (fedgraa/baselines_t5_fedpgo.py).

The FedPGO mechanism is implemented from the paper's primary text
(``docs/research_c/FedPGO/FedPGO.md``, Eqs. 6-10; Algorithms 1-3) per
``docs/research_c/STAGE_C_FEDPGO_FAITHFUL_PLAN_2026-07-31.md``, and these
tests are the plan's own §5 spec: mechanism-fidelity to Eqs. 6-10 is fully
unit-testable, and the previous 12 proxy-semantics tests were replaced
(signatures changed), not preserved.

Conventions follow ``tests/test_baselines_t5_adaptive.py``: a tiny synthetic
``DatasetSpec`` built with ``make_synthetic_dataset`` + ``partition_non_iid``,
a handful of rounds, no-NaN / valid-simplex checks, plus a property test for
each named piece of the mechanism (AGO decomposition, gradient memory,
recomposition, PGF scoring, Eq. 10 weights, head fine-tune, parameter split).
"""

from __future__ import annotations

import numpy as np
import pytest

from fedgraa.baselines_t5_fedpgo import (
    FedPGOTrainer,
    fedpgo_decompose,
    fedpgo_pgf_score,
    fedpgo_weights,
)
from fedgraa.config import DatasetSpec, FedConfig
from fedgraa.data import make_synthetic_dataset
from fedgraa.model import aggregate, init_params
from fedgraa.partition import partition_non_iid


def _tiny_split(seed: int = 0):
    spec = DatasetSpec(
        name="T5_FEDPGO_TEST",
        n_features=6,
        labels=["Benign", "A", "B", "Rare"],
        used_counts=[300, 120, 90, 24],
        benign_id=0,
        rare_label_order=["Rare"],
    )
    X, y, _ = make_synthetic_dataset(spec, seed=seed)
    return partition_non_iid(X, y, spec, n_rare=1, val_fraction=0.10, seed=seed)


def _cfg(rounds: int = 4) -> FedConfig:
    return FedConfig(rounds=rounds, gamma=0.5, n_rare=1, seed=0)


# --------------------------------------------------------------------------- #
# AGO: exact decomposition against a STORED global gradient (Eqs. 6-7, Alg. 1)
# --------------------------------------------------------------------------- #
def test_fedpgo_decompose_matches_eq6_7_given_stored_gG():
    g_G = [np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)]
    g_i = [np.array([[2.0, 3.0], [0.0, 0.0]], dtype=np.float32)]
    parallel, orthogonal, _ = fedpgo_decompose(g_i, g_G, alpha=0.5)
    # c_i = (2*1 + 3*0) / (||g_G||^2 + eps) = 2/1 = 2  ->  g_par = 2 * g_G (Eq. 6)
    np.testing.assert_allclose(parallel[0], 2.0 * g_G[0])
    # g_perp = g_i - g_par (Eq. 7)
    np.testing.assert_allclose(orthogonal[0], g_i[0] - parallel[0])
    # par + perp reconstructs g_i exactly
    np.testing.assert_allclose(parallel[0] + orthogonal[0], g_i[0], atol=1e-6)


def test_fedpgo_decompose_epsilon_path_guards_the_denominator():
    # Tiny-but-nonzero g_G: the +eps in Eq. 6's denominator keeps the
    # projection coefficient O(dot/norm^2), not O(dot/eps).
    g_G = [np.array([1e-6, 0.0], dtype=np.float32)]
    g_i = [np.array([2.0, 0.0], dtype=np.float32)]
    parallel, orthogonal, _ = fedpgo_decompose(g_i, g_G, alpha=1.0, eps=1e-6)
    # c_i = 2e-6 / (1e-12 + 1e-6) ~= 2.0  ->  g_par = c_i * g_G ~= [2e-6, 0]
    np.testing.assert_allclose(parallel[0], [2e-6, 0.0], atol=1e-9)
    np.testing.assert_allclose(orthogonal[0], [2.0, 0.0], atol=1e-3)
    # Exactly zero g_G: no division by zero, everything finite.
    par0, perp0, prime0 = fedpgo_decompose(
        g_i, [np.zeros(2, dtype=np.float32)], alpha=0.5,
    )
    assert all(np.all(np.isfinite(p)) for p in (par0[0], perp0[0], prime0[0]))


def test_fedpgo_round1_degeneracy_zero_gG():
    """JC4: with g_G = 0 (round 1, Alg. 3 step 3), c_i = 0, g_par = 0,
    g_perp = g_i, g' = (1 - alpha) * g_i -- and the Eq. 9 cosine term is 0."""
    g_i = [np.array([1.0, 2.0], dtype=np.float32)]
    zero = [np.zeros(2, dtype=np.float32)]
    for alpha in (0.0, 0.5, 1.0):
        parallel, orthogonal, prime = fedpgo_decompose(g_i, zero, alpha)
        np.testing.assert_allclose(parallel[0], [0.0, 0.0])
        np.testing.assert_allclose(orthogonal[0], g_i[0])
        np.testing.assert_allclose(prime[0], (1.0 - alpha) * g_i[0])
    score = fedpgo_pgf_score(
        g_i, zero, g_i, 0.0, lambda1=1.0, lambda2=0.0, lambda3=0.0,
    )
    assert score == 0.0


# --------------------------------------------------------------------------- #
# Recompose (Eq. 8)
# --------------------------------------------------------------------------- #
def test_fedpgo_recompose_matches_eq8_across_alpha():
    g_G = [np.array([1.0, 0.0], dtype=np.float32)]
    g_i = [np.array([2.0, 3.0], dtype=np.float32)]
    parallel, orthogonal, _ = fedpgo_decompose(g_i, g_G, alpha=0.5)
    for alpha in (0.0, 0.5, 1.0):
        _, _, prime = fedpgo_decompose(g_i, g_G, alpha)
        expected = [alpha * p + (1.0 - alpha) * o
                    for p, o in zip(parallel, orthogonal)]
        np.testing.assert_allclose(prime[0], expected[0])
    # Endpoints: alpha=1 -> only the global-aligned part; alpha=0 -> only the
    # personalized (orthogonal) part.
    _, _, p1 = fedpgo_decompose(g_i, g_G, 1.0)
    _, _, p0 = fedpgo_decompose(g_i, g_G, 0.0)
    np.testing.assert_allclose(p1[0], parallel[0])
    np.testing.assert_allclose(p0[0], orthogonal[0])


# --------------------------------------------------------------------------- #
# Gradient memory (Alg. 3 step 23) -- the proxy's central bug, pinned
# --------------------------------------------------------------------------- #
def test_fedpgo_gradient_memory_regression():
    """g_G used in round t+1 must equal round t's AGGREGATED g' (Alg. 3 step
    23) -- and must NOT be the same-round FedAvg-weighted mean of local
    deltas, which is exactly what the archived proxy computed."""
    split = _tiny_split(seed=0)
    t = FedPGOTrainer(split, _cfg(rounds=4))
    rng = np.random.default_rng(t.cfg.seed)
    shared, personalized, g_G = t._initial_state(rng)

    shared1, personalized, gG1, w1, info1 = t._one_round(
        1, shared, personalized, g_G, rng,
    )
    # Round 1: memory holds round 1's aggregated g'.
    expected1 = aggregate(info1["g_prime_list"], w1)
    for a, b in zip(gG1, expected1):
        np.testing.assert_allclose(a, b)
    # ... and is NOT the same-round FedAvg mean of the local deltas.
    fedavg_w = np.asarray(t.client_sizes, dtype=np.float64)
    fedavg_w = (fedavg_w / fedavg_w.sum()).astype(np.float32)
    fedavg_mean1 = aggregate(info1["tau_list"], fedavg_w)
    assert not all(
        np.allclose(a, b) for a, b in zip(gG1, fedavg_mean1)
    ), "gradient memory equals the same-round FedAvg mean -- proxy bug"

    shared2, personalized, gG2, w2, info2 = t._one_round(
        2, shared1, personalized, gG1, rng,
    )
    # Round 2 decomposed against round 1's aggregate: recomputing the AGO of
    # round 2's first client against gG1 must reproduce its recomposed g'.
    _, _, prime_recomputed = fedpgo_decompose(
        info2["tau_list"][0], gG1, float(t.alpha[0]), t.eps,
    )
    np.testing.assert_allclose(prime_recomputed[0], info2["g_prime_list"][0][0])
    # ... and gG1 is still not round 2's same-round FedAvg mean.
    fedavg_mean2 = aggregate(info2["tau_list"], fedavg_w)
    assert not all(
        np.allclose(a, b) for a, b in zip(gG1, fedavg_mean2)
    ), "gradient memory equals the same-round FedAvg mean -- proxy bug"

    # The trainer's public post-train state: gradient_memory is the last
    # round's aggregated g', and is non-zero (the mechanism did run).
    t.train(eval_every=2)
    assert any(np.any(a != 0.0) for a in t.gradient_memory)


# --------------------------------------------------------------------------- #
# PGF score (Eq. 9): each term isolatable via the lambda toggles
# --------------------------------------------------------------------------- #
def test_fedpgo_pgf_score_terms_isolatable_via_lambda_toggles():
    g_G = [np.array([1.0, 0.0], dtype=np.float32)]
    aligned = [np.array([1.0, 0.0], dtype=np.float32)]
    anti = [np.array([-1.0, 0.0], dtype=np.float32)]
    mixed = [np.array([1.0, 1.0], dtype=np.float32)]
    zero = [np.zeros(2, dtype=np.float32)]

    # lambda1 only: cos sign distinguishes aligned from anti-aligned.
    s_aligned = fedpgo_pgf_score(
        aligned, g_G, zero, 0.0, lambda1=1.0, lambda2=0.0, lambda3=0.0,
    )
    s_anti = fedpgo_pgf_score(
        anti, g_G, zero, 0.0, lambda1=1.0, lambda2=0.0, lambda3=0.0,
    )
    assert s_aligned == pytest.approx(1.0)
    assert s_anti == pytest.approx(-1.0)

    # lambda2 only: magnitude of the orthogonal (personalization-relevant)
    # component. g_perp of the mixed client has norm 1; the aligned one 0.
    s_mixed = fedpgo_pgf_score(
        mixed, g_G, [np.array([0.0, 1.0], dtype=np.float32)], 0.0,
        lambda1=0.0, lambda2=1.0, lambda3=0.0,
    )
    s_flat = fedpgo_pgf_score(
        aligned, g_G, [np.array([0.0, 0.0], dtype=np.float32)], 0.0,
        lambda1=0.0, lambda2=1.0, lambda3=0.0,
    )
    assert s_mixed == pytest.approx(1.0)
    assert s_flat == pytest.approx(0.0)
    assert s_mixed > s_flat

    # lambda3 only: the validation-loss reduction term.
    s_gain = fedpgo_pgf_score(
        aligned, g_G, zero, 0.4, lambda1=0.0, lambda2=0.0, lambda3=1.0,
    )
    assert s_gain == pytest.approx(0.4)


# --------------------------------------------------------------------------- #
# Delta-L-val (Alg. 3 steps 11-14)
# --------------------------------------------------------------------------- #
def test_fedpgo_head_fine_tune_reduces_validation_loss():
    """The head-only fine-tune on the client's own validation slice must not
    increase validation loss: L_after <= L_before."""
    split = _tiny_split(seed=0)
    t = FedPGOTrainer(split, _cfg(rounds=1), personalize_steps=20)
    rng = np.random.default_rng(t.cfg.seed)
    shared, personalized, _ = t._initial_state(rng)
    client = split.clients[0]
    head_before = personalized[0]
    ft, delta_l = t._head_fine_tune(
        shared, head_before, client.X_val, client.y_val,
    )
    assert delta_l >= -1e-3, f"validation loss increased: {delta_l}"
    # The head actually moved (and the shared params are untouched by design).
    assert not all(
        np.allclose(a, b) for a, b in zip(ft, head_before)
    ), "fine-tune left the head unchanged"


def test_fedpgo_high_validation_gain_out_scores_low_gain_at_lambda3():
    """With lambda1=lambda2=0 the PGF score is the Delta-L-val term alone, so
    a high-validation-gain client must out-score a low-gain one (Eq. 9)."""
    g_G = [np.array([1.0, 0.0], dtype=np.float32)]
    g_i = [np.array([1.0, 0.0], dtype=np.float32)]
    zero = [np.zeros(2, dtype=np.float32)]
    high = fedpgo_pgf_score(
        g_i, g_G, zero, 0.5, lambda1=0.0, lambda2=0.0, lambda3=1.0,
    )
    low = fedpgo_pgf_score(
        g_i, g_G, zero, 0.1, lambda1=0.0, lambda2=0.0, lambda3=1.0,
    )
    w = fedpgo_weights([high, low])
    assert w[0] > w[1]
    np.testing.assert_allclose(w, [5.0 / 6.0, 1.0 / 6.0], atol=1e-6)


# --------------------------------------------------------------------------- #
# Weights (Eq. 10 / Alg. 2): non-negative clip path, +eps denominator,
# uniform fallback on degenerate rounds
# --------------------------------------------------------------------------- #
def test_fedpgo_weights_match_eq10():
    w = fedpgo_weights([2.0, 1.0])
    np.testing.assert_allclose(w, [2.0 / 3.0, 1.0 / 3.0], atol=1e-6)
    # Eq. 10's "+ eps" denominator is real, not decorative.
    w = fedpgo_weights([1.0, 1.0], eps=1.0)
    np.testing.assert_allclose(w, [1.0 / 3.0, 1.0 / 3.0], atol=1e-6)
    # Clip path (JC3): negative scores never produce negative weights.
    w = fedpgo_weights([1.0, -5.0, 1.0])
    assert np.all(w >= 0.0)
    assert abs(w.sum() - 1.0) < 1e-6


def test_fedpgo_weights_degenerate_round_falls_back_to_uniform():
    """Every score ~0 (round 0, or an all-zero round): uniform weights, never
    a division by ~zero (existing degenerate-round repo convention)."""
    w = fedpgo_weights([-1.0, 0.0, -0.5])
    np.testing.assert_allclose(w, np.full(3, 1.0 / 3.0))
    w = fedpgo_weights([0.0, 0.0])
    np.testing.assert_allclose(w, [0.5, 0.5])


def test_fedpgo_anti_aligned_client_downweighted_vs_aligned():
    """Cos term (Eq. 9): an anti-aligned client's score is negative and
    therefore clipped to 0 (JC3); the aligned client takes the weight."""
    g_G = [np.array([1.0, 0.0], dtype=np.float32)]
    zero = [np.zeros(2, dtype=np.float32)]
    aligned = [np.array([1.0, 0.0], dtype=np.float32)]
    anti_aligned = [np.array([-1.0, 0.0], dtype=np.float32)]
    scores = [
        fedpgo_pgf_score(a, g_G, zero, 0.0)
        for a in (aligned, anti_aligned)
    ]
    w = fedpgo_weights(scores)
    assert w[0] > w[1]
    assert w[1] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Parameter-split guarantees (unchanged by the faithful rewrite)
# --------------------------------------------------------------------------- #
def test_fedpgo_personalized_heads_differ_shared_layers_aggregated():
    """The parameter-split's two guarantees: personalized (head) layers are
    never aggregated, so distinct clients end up with distinct heads after
    training on their own (non-identical, non-IID) shards; and the shared
    layers ARE aggregated, so all clients share one identical shared part."""
    split = _tiny_split(seed=2)
    t = FedPGOTrainer(split, _cfg(rounds=4), personalized_layers=1)
    t.train(eval_every=4)
    n_shared = t._n_shared
    models = t.personalized_params
    for i in range(len(models)):
        for j in range(len(models)):
            if i != j:
                for a in range(n_shared):
                    np.testing.assert_allclose(models[i][a], models[j][a])
    heads = [m[n_shared:] for m in models]
    assert not all(
        np.allclose(heads[0][0], h[0]) for h in heads[1:]
    ), "every client's personalized layer is identical -- parameter-split had no effect"


# --------------------------------------------------------------------------- #
# End-to-end: no NaN collapse, valid simplex, determinism, accuracies
# --------------------------------------------------------------------------- #
def test_fedpgo_no_nan_collapse():
    split = _tiny_split(seed=0)
    trainer = FedPGOTrainer(split, _cfg(rounds=4))
    trainer.train(eval_every=2)
    for p in trainer.final_params:
        assert np.all(np.isfinite(p)), (
            f"final params contain non-finite values "
            f"(shape={p.shape}, n_bad={int(np.sum(~np.isfinite(p)))})"
        )
    for model in trainer.personalized_params:
        for p in model:
            assert np.all(np.isfinite(p))


def test_fedpgo_weights_are_valid_simplex_every_recorded_round():
    split = _tiny_split(seed=1)
    trainer = FedPGOTrainer(split, _cfg(rounds=4))
    hist = trainer.train(eval_every=2)
    assert len(hist.weights) >= 2
    K = split.n_clients
    for rnd, w in zip(hist.rounds, hist.weights):
        w = np.asarray(w, dtype=np.float64)
        assert w.shape == (K,), f"round {rnd} weight shape {w.shape} != ({K},)"
        assert np.all(np.isfinite(w)), f"round {rnd} weights non-finite"
        assert np.all(w >= -1e-9), f"round {rnd} has a negative weight: {w}"
        assert abs(w.sum() - 1.0) < 1e-5, f"round {rnd} weights sum to {w.sum()}"


def test_fedpgo_runs_end_to_end():
    split = _tiny_split()
    hist = FedPGOTrainer(split, _cfg(rounds=3)).train(eval_every=3)
    assert len(hist.all_acc) >= 1
    assert 0.0 <= hist.final_all_acc <= 1.0
    assert 0.0 <= hist.final_last_n_acc <= 1.0


def test_fedpgo_deterministic_per_seed():
    split = _tiny_split(seed=3)
    cfg = _cfg(rounds=4)
    h1 = FedPGOTrainer(split, cfg).train(eval_every=2)
    h2 = FedPGOTrainer(split, cfg).train(eval_every=2)
    assert h1.rounds == h2.rounds
    np.testing.assert_allclose(h1.all_acc, h2.all_acc)
    np.testing.assert_allclose(h1.last_n_acc, h2.last_n_acc)
    for w1, w2 in zip(h1.weights, h2.weights):
        np.testing.assert_allclose(w1, w2)


# --------------------------------------------------------------------------- #
# Constructor validation
# --------------------------------------------------------------------------- #
def test_fedpgo_rejects_alpha_out_of_range():
    split = _tiny_split()
    with pytest.raises(ValueError):
        FedPGOTrainer(split, _cfg(), alpha=1.5)
    with pytest.raises(ValueError):
        FedPGOTrainer(split, _cfg(), alpha=-0.1)
    with pytest.raises(ValueError):
        FedPGOTrainer(split, _cfg(), alpha=[0.5, 0.5, 0.5, 0.5, 1.2])


def test_fedpgo_accepts_per_client_alpha_array():
    split = _tiny_split()
    K = split.n_clients
    alphas = [0.1 + 0.05 * i for i in range(K)]
    t = FedPGOTrainer(split, _cfg(), alpha=alphas)
    np.testing.assert_allclose(t.alpha, alphas)
    # Wrong-length arrays are rejected.
    with pytest.raises(ValueError):
        FedPGOTrainer(split, _cfg(), alpha=[0.5] * (K + 1))


def test_fedpgo_rejects_personalized_layers_out_of_range():
    split = _tiny_split()
    with pytest.raises(ValueError):
        FedPGOTrainer(split, _cfg(), personalized_layers=0)
    with pytest.raises(ValueError):
        # 2 layers total for this MLP config; requesting all of them shared
        # away leaves nothing to aggregate.
        n_layers = len(split.spec.model_config().layer_sizes) - 1
        FedPGOTrainer(split, _cfg(), personalized_layers=n_layers)
