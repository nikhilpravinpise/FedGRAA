"""Tests for the T5 adaptive-aggregation-weight baselines.

Conventions follow ``tests/test_reproduction_gaps.py``: a tiny synthetic
``DatasetSpec`` built with ``make_synthetic_dataset`` + ``partition_non_iid``,
and a handful of rounds.

Beyond the shared no-NaN / valid-simplex checks, each method gets one test of
the property that makes it *that* method, and one test asserts FedAWA and
FedAWARE are not the same thing (see the FedAWA-vs-FedAWARE warning in
``fedgraa/baselines_t5_adaptive.py``'s module docstring).
"""

from __future__ import annotations

import numpy as np
import pytest

from fedgraa.baselines_t5_adaptive import (
    AdaAggRLTrainer,
    DAPFLTrainer,
    FedAWARETrainer,
    FedAWATrainer,
    adaaggrl_aggregation_weights,
    dapfl_affinity_matrix,
    dapfl_alpha_matrix,
    fedaware_weights,
    fedawa_objective,
    fedawa_weights,
    frank_wolfe_min_norm,
    reinforce_step,
)
from fedgraa.config import DatasetSpec, FedConfig
from fedgraa.data import make_synthetic_dataset
from fedgraa.partition import partition_non_iid


# --------------------------------------------------------------------------- #
# Shared fixtures / helpers
# --------------------------------------------------------------------------- #
def _tiny_split(seed: int = 0):
    spec = DatasetSpec(
        name="T5_ADAPTIVE_TEST",
        n_features=6,
        labels=["Benign", "A", "B", "Rare"],
        used_counts=[300, 120, 90, 24],
        benign_id=0,
        rare_label_order=["Rare"],
    )
    X, y, _ = make_synthetic_dataset(spec, seed=seed)
    return partition_non_iid(X, y, spec, n_rare=1, val_fraction=0.10, seed=seed)


def _cfg(rounds: int = 4) -> FedConfig:
    return FedConfig(rounds=rounds, gamma=0.5, n_rare=1, seed=0, round_zero_zero=False)


TRAINERS = {
    "fedawa": FedAWATrainer,
    "dapfl": DAPFLTrainer,
    "fedaware": FedAWARETrainer,
    "adaaggrl": AdaAggRLTrainer,
}


def _fake_params(vecs, shape=(2, 3)):
    """Turn a list of length-6 vectors into ``[W]``-style parameter lists."""
    return [[np.asarray(v, dtype=np.float32).reshape(shape)] for v in vecs]


# --------------------------------------------------------------------------- #
# End-to-end: no NaN collapse, valid simplex every recorded round
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(TRAINERS))
def test_t5_adaptive_no_nan_collapse(name):
    """Mirrors ``test_all_methods_no_nan_collapse``: finite params after a run."""
    split = _tiny_split(seed=0)
    trainer = TRAINERS[name](split, _cfg(rounds=4))
    trainer.train(eval_every=2)
    params = trainer.final_params
    assert params is not None, f"{name}: final_params is None"
    for p in params:
        assert np.all(np.isfinite(p)), (
            f"{name}: final params contain non-finite values "
            f"(shape={p.shape}, n_bad={int(np.sum(~np.isfinite(p)))})"
        )


@pytest.mark.parametrize("name", sorted(TRAINERS))
def test_t5_adaptive_weights_are_valid_simplex(name):
    """Every recorded weight vector is non-negative and sums to 1."""
    split = _tiny_split(seed=1)
    trainer = TRAINERS[name](split, _cfg(rounds=4))
    hist = trainer.train(eval_every=2)
    assert len(hist.weights) >= 2
    K = split.n_clients
    for rnd, w in zip(hist.rounds, hist.weights):
        w = np.asarray(w, dtype=np.float64)
        assert w.shape == (K,), f"{name}: round {rnd} weight shape {w.shape} != ({K},)"
        assert np.all(np.isfinite(w)), f"{name}: round {rnd} weights non-finite"
        assert np.all(w >= -1e-9), f"{name}: round {rnd} has a negative weight: {w}"
        assert abs(w.sum() - 1.0) < 1e-5, f"{name}: round {rnd} weights sum to {w.sum()}"


@pytest.mark.parametrize("name", sorted(TRAINERS))
def test_t5_adaptive_trainers_run_end_to_end(name):
    split = _tiny_split()
    hist = TRAINERS[name](split, _cfg(rounds=3)).train(eval_every=3)
    assert len(hist.all_acc) >= 1
    assert 0.0 <= hist.final_all_acc <= 1.0
    assert 0.0 <= hist.final_last_n_acc <= 1.0


# --------------------------------------------------------------------------- #
# FedAWA: alignment with the global update direction
# --------------------------------------------------------------------------- #
def test_fedawa_downweights_anti_aligned_client():
    """A client whose update opposes the consensus direction loses weight.

    Three clients push in a common direction; the fourth pushes in the exact
    opposite one. FedAWA's Eq. 3 penalises ``lambda_k * ||tau_k - tau_g||``, so
    the odd one out must end up below every aligned client.
    """
    d = 6
    common = np.zeros(d, dtype=np.float32)
    common[0] = 1.0
    theta_g = np.zeros(d, dtype=np.float32)

    taus = [common, common * 1.1, common * 0.9, -common]
    client_params = _fake_params([theta_g + t for t in taus])
    global_params = [theta_g.reshape(2, 3)]

    w, _ = fedawa_weights(
        client_params, global_params,
        logits=np.zeros(4), server_lr=0.5, server_epochs=50,
    )
    assert w[3] < w[0], f"anti-aligned client not downweighted: {w}"
    assert w[3] < w[1] and w[3] < w[2], f"anti-aligned client not the minimum: {w}"
    assert w[3] < 1.0 / 4.0, f"anti-aligned client not below uniform: {w}"


def test_fedawa_descends_its_own_objective():
    """The analytic gradient really decreases Eq. 3 (guards the derivation)."""
    rng = np.random.default_rng(3)
    d, K = 8, 5
    theta_g = rng.normal(size=d).astype(np.float32)
    taus = rng.normal(size=(K, d)).astype(np.float32)
    client_params = [[(theta_g + t).reshape(2, 4)] for t in taus]
    global_params = [theta_g.reshape(2, 4)]

    tau_mat = np.asarray(taus, dtype=np.float64)
    theta_flat = np.asarray(theta_g, dtype=np.float64)
    start = np.full(K, 1.0 / K)
    w, _ = fedawa_weights(
        client_params, global_params,
        logits=np.zeros(K), server_lr=0.5, server_epochs=200,
    )
    assert fedawa_objective(w, tau_mat, theta_flat) < fedawa_objective(
        start, tau_mat, theta_flat
    )


# --------------------------------------------------------------------------- #
# FedAWARE: minimum-norm point in the convex hull of client momenta
# --------------------------------------------------------------------------- #
def test_fedaware_finds_minimum_norm_point():
    """``||sum_i lambda_i m_i||`` must beat the uniform average, and beat a
    large random sample of other simplex points."""
    rng = np.random.default_rng(11)
    K, d = 6, 12
    M = rng.normal(size=(K, d))
    M[0] *= 8.0  # one large-magnitude outlier direction

    lam = np.asarray(fedaware_weights(M), dtype=np.float64)
    best = np.linalg.norm(lam @ M)

    uniform = np.linalg.norm(np.full(K, 1.0 / K) @ M)
    assert best <= uniform + 1e-8, f"{best} > uniform {uniform}"

    for _ in range(200):
        cand = rng.dirichlet(np.ones(K))
        assert best <= np.linalg.norm(cand @ M) + 1e-8


def test_fedaware_cancels_exactly_opposing_updates():
    """Two clients with exactly opposite momenta split the weight evenly, and
    the resulting aggregate direction is (numerically) zero."""
    m = np.zeros((2, 5))
    m[0, 0] = 1.0
    m[1, 0] = -1.0
    lam = np.asarray(fedaware_weights(m), dtype=np.float64)
    assert lam == pytest.approx([0.5, 0.5], abs=1e-4)
    assert np.linalg.norm(lam @ m) < 1e-4


def test_frank_wolfe_handles_single_client():
    assert frank_wolfe_min_norm(np.array([[4.0]])) == pytest.approx([1.0])


# --------------------------------------------------------------------------- #
# The FedAWA / FedAWARE conflation guard
# --------------------------------------------------------------------------- #
def test_fedawa_and_fedaware_differ_on_identical_input():
    """Given the identical client updates, the two methods must disagree.

    They are frequently confused because of the near-identical names. FedAWA
    concentrates weight on clients aligned with the consensus; FedAWARE's
    minimum-norm point does the opposite, loading the conflicting direction so
    the aggregate norm shrinks. If this test ever passes trivially (identical
    vectors), one of the two implementations has been overwritten by the other.
    """
    d = 6
    common = np.zeros(d)
    common[0] = 1.0
    taus = np.stack([common, common * 1.1, common * 0.9, -common])
    theta_g = np.zeros(d, dtype=np.float32)

    client_params = _fake_params([theta_g + t for t in taus])
    global_params = [theta_g.reshape(2, 3)]

    w_awa, _ = fedawa_weights(
        client_params, global_params,
        logits=np.zeros(4), server_lr=0.5, server_epochs=50,
    )
    # FedAWARE consumes pseudo-gradients g = theta_g - theta_k = -tau.
    w_aware = fedaware_weights(-taus)

    w_awa = np.asarray(w_awa, dtype=np.float64)
    w_aware = np.asarray(w_aware, dtype=np.float64)
    assert not np.allclose(w_awa, w_aware, atol=1e-3), (
        "FedAWA and FedAWARE produced the same weight vector — the two methods "
        f"have been conflated: {w_awa} vs {w_aware}"
    )
    # And they disagree in the specific, opposite way the papers imply: the
    # anti-aligned client is FedAWA's least-favoured and FedAWARE's most-favoured.
    assert int(np.argmin(w_awa)) == 3, w_awa
    assert int(np.argmax(w_aware)) == 3, w_aware


def test_fedawa_and_fedaware_trainers_differ_end_to_end():
    """The same split + config must not yield identical trained weights."""
    split = _tiny_split(seed=2)
    cfg = _cfg(rounds=3)
    h_awa = FedAWATrainer(split, cfg).train(eval_every=3)
    h_aware = FedAWARETrainer(split, cfg).train(eval_every=3)
    assert not np.allclose(h_awa.weights[-1], h_aware.weights[-1], atol=1e-4)


# --------------------------------------------------------------------------- #
# DA-PFL: complementarity affinity + personalized weighting
# --------------------------------------------------------------------------- #
def test_dapfl_affinity_prefers_complementary_over_identical():
    """DA-PFL's ``2 - cosine`` metric rewards *complementary* label histograms.

    Client 0 and client 1 are exactly anti-correlated over their shared label
    support; client 2 is identical to client 0. The complementary pair must
    score strictly higher — this is the correction to the widespread "similar
    distributions get higher affinity" summary (see the module docstring).
    """
    hists = [
        np.array([10, 1]),   # client 0
        np.array([1, 10]),   # client 1 — complementary to 0
        np.array([10, 1]),   # client 2 — identical to 0
    ]
    C = dapfl_affinity_matrix(hists, n_classes=2)
    assert C[0, 1] > C[0, 2], f"complementary pair not preferred: {C}"
    assert C[0, 1] == pytest.approx(3.0)   # (2/2) * (2 - (-1))
    assert C[0, 2] == pytest.approx(1.0)   # (2/2) * (2 - (+1))
    assert np.all(np.diag(C) == 0.0)


def test_dapfl_affinity_requires_shared_label_support():
    """Zero overlap in class support gives zero affinity (the |E_ij|/K factor)."""
    hists = [np.array([5, 0, 0]), np.array([0, 0, 7]), np.array([4, 0, 1])]
    C = dapfl_affinity_matrix(hists, n_classes=3)
    assert C[0, 1] == 0.0
    assert C[0, 2] > 0.0


def test_dapfl_alpha_rows_are_simplex_over_others():
    """Each personalized row is a simplex point that excludes the client itself,
    and a higher-affinity peer gets more weight at equal model distance."""
    hists = [np.array([10, 1]), np.array([1, 10]), np.array([10, 1])]
    C = dapfl_affinity_matrix(hists, n_classes=2)
    # Equal pairwise model distances so only the affinity term distinguishes.
    base = np.zeros(6, dtype=np.float32)
    params = _fake_params([
        base,
        base + np.eye(6, dtype=np.float32)[0],
        base + np.eye(6, dtype=np.float32)[1],
    ])
    A = dapfl_alpha_matrix(C, params, sigma=1.0)
    for i in range(3):
        assert A[i, i] == 0.0
        assert A[i].sum() == pytest.approx(1.0)
        assert np.all(A[i] >= 0.0)
    # Client 0 must favour its complementary peer (1) over its clone (2).
    assert A[0, 1] > A[0, 2], A


def test_dapfl_trainer_keeps_personalized_models():
    split = _tiny_split(seed=4)
    trainer = DAPFLTrainer(split, _cfg(rounds=3))
    trainer.train(eval_every=3)
    assert len(trainer.personalized_params) == split.n_clients
    flat = [np.concatenate([np.ravel(p) for p in ps])
            for ps in trainer.personalized_params]
    # Personalization means the client models must not all collapse to one model.
    assert not np.allclose(flat[0], flat[1], atol=1e-6)


# --------------------------------------------------------------------------- #
# AdaAggRL: reward-driven policy, thresholding and exponential penalty
# --------------------------------------------------------------------------- #
def test_adaaggrl_policy_moves_toward_rewarded_action():
    """A positive advantage moves the policy mean toward the sampled action;
    a negative advantage moves it away."""
    logits = np.zeros(3)
    noise = np.array([1.0, 0.0, -1.0])

    up = logits
    for _ in range(5):
        up = reinforce_step(up, noise, advantage=+1.0, lr=0.05, sigma=0.3)
    assert up[0] > 0.0 and up[2] < 0.0
    assert up[0] > up[1] > up[2]

    down = logits
    for _ in range(5):
        down = reinforce_step(down, noise, advantage=-1.0, lr=0.05, sigma=0.3)
    assert down[0] < 0.0 and down[2] > 0.0


def test_adaaggrl_weights_track_the_scored_state():
    """Higher state similarity under the action's metric weighting means more
    weight; and the exponential penalty compounds across repeated strikes."""
    states = np.array([
        [0.9, 0.9, 0.9],
        [0.8, 0.8, 0.8],
        [0.1, 0.1, 0.1],   # clear outlier
    ])
    action = np.array([1 / 3, 1 / 3, 1 / 3])
    strikes = np.zeros(3)

    w1, s1 = adaaggrl_aggregation_weights(states, action, b=0.5, strikes=strikes)
    assert w1[0] > w1[1] > w1[2]
    assert s1[2] == 1.0 and s1[0] == 0.0    # only the outlier is below threshold
    assert w1.sum() == pytest.approx(1.0)

    w2, s2 = adaaggrl_aggregation_weights(states, action, b=0.5, strikes=s1)
    assert s2[2] == 2.0
    assert w2[2] < w1[2], "penalty did not compound over repeated strikes"

    # Strikes reset once a client is back above threshold.
    _, s3 = adaaggrl_aggregation_weights(states, action, b=0.0, strikes=s2)
    assert np.all(s3 == 0.0)


def test_adaaggrl_responds_to_validation_reward_signal():
    """Over a few real rounds the policy is actually driven by the reward.

    The trainer records ``r = f(theta^t) - f(theta^{t+1})`` on the probe split
    each round and takes one REINFORCE step. If the policy never moved off its
    initialisation, the reward signal is not wired in.
    """
    split = _tiny_split(seed=5)
    trainer = AdaAggRLTrainer(split, _cfg(rounds=5))
    before = trainer.action_logits.copy()
    trainer.train(eval_every=5)

    assert len(trainer.rewards) == 5
    assert all(np.isfinite(r) for r in trainer.rewards)
    assert not np.allclose(trainer.action_logits, before), (
        "policy logits unchanged after 5 rounds — reward signal is not driving "
        "the policy"
    )
    # Reward drives the update through the advantage term, so a run whose
    # rewards are non-trivial must have moved the baseline off zero too.
    if any(abs(r) > 1e-9 for r in trainer.rewards):
        assert trainer.reward_baseline != 0.0
