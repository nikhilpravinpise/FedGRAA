"""Tests for the DP mechanism and accountants (W2.2, F3/§5.2/§5.3).

T-9: empirical noise std matches the closed-form ``sigma = (Delta_1/eps) *
sqrt(2K)`` within 10%. Plus closed-form checks of both composition
accountants against FEDGRAA_V2_MASTER_PLAN.md §5.3's worked example
(eps_round=0.1, delta=1e-5, T=200 -> eps_total ~= 8.89) and §5.2's DP-recovery
table.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fedgraa.privacy import (
    SENSITIVITY,
    advanced_composition_epsilon,
    aggregate_noise_std,
    composed_epsilon,
    composed_rdp_epsilon,
    laplace_mechanism,
    laplace_rdp,
    rdp_to_dp,
)


def test_sensitivity_constant_is_2():
    assert SENSITIVITY == 2.0


def test_laplace_mechanism_rejects_nonpositive_epsilon():
    with pytest.raises(ValueError):
        laplace_mechanism(np.array([1.0, 2.0]), epsilon=0.0)
    with pytest.raises(ValueError):
        laplace_mechanism(np.array([1.0, 2.0]), epsilon=-1.0)


def test_laplace_mechanism_is_unbiased_in_expectation():
    rng = np.random.default_rng(0)
    counts = np.array([100.0, 50.0, 5.0])
    samples = np.stack([
        laplace_mechanism(counts, epsilon=2.0, rng=rng) for _ in range(20_000)
    ])
    np.testing.assert_allclose(samples.mean(axis=0), counts, atol=0.5)


# --------------------------------------------------------------------------- #
# T-9 -- empirical noise std matches the closed form within 10%.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("epsilon,K", [(1.0, 14), (4.0, 14), (1.0, 9), (10.0, 9)])
def test_aggregate_noise_std_matches_empirical(epsilon, K):
    rng = np.random.default_rng(0)
    b = SENSITIVITY / epsilon
    # SecAgg-summed noise over K clients: sum of K iid Laplace(0, b) draws.
    samples = rng.laplace(0.0, b, size=(20_000, K)).sum(axis=1)
    empirical_std = float(np.std(samples))
    closed_form = aggregate_noise_std(epsilon, K)
    assert abs(empirical_std - closed_form) / closed_form < 0.10, (
        empirical_std, closed_form,
    )


def test_aggregate_noise_std_matches_plan_table_cic():
    """FEDGRAA_V2_MASTER_PLAN.md §5.2 table, CIC-IDS-2017 (K=14)."""
    assert aggregate_noise_std(1.0, 14) == pytest.approx(10.58, abs=0.01)
    assert aggregate_noise_std(4.0, 14) == pytest.approx(2.65, abs=0.01)
    assert aggregate_noise_std(10.0, 14) == pytest.approx(1.06, abs=0.01)


def test_aggregate_noise_std_matches_plan_table_unsw():
    """FEDGRAA_V2_MASTER_PLAN.md §5.2 table, UNSW-NB15 (K=9)."""
    assert aggregate_noise_std(1.0, 9) == pytest.approx(8.49, abs=0.01)
    assert aggregate_noise_std(4.0, 9) == pytest.approx(2.12, abs=0.01)
    assert aggregate_noise_std(10.0, 9) == pytest.approx(0.85, abs=0.01)


def test_aggregate_noise_std_rejects_invalid_args():
    with pytest.raises(ValueError):
        aggregate_noise_std(0.0, 14)
    with pytest.raises(ValueError):
        aggregate_noise_std(1.0, 0)


# --------------------------------------------------------------------------- #
# Advanced composition (§5.3's worked example: eps_total ~= 8.89)
# --------------------------------------------------------------------------- #
def test_advanced_composition_matches_plan_worked_example():
    eps_total, delta_total = advanced_composition_epsilon(
        epsilon_round=0.1, rounds=200, delta_prime=1e-5
    )
    assert eps_total == pytest.approx(8.89, abs=0.01)
    assert delta_total == pytest.approx(1e-5, rel=1e-6)


def test_advanced_composition_monotone_in_rounds():
    e1, _ = advanced_composition_epsilon(0.1, 50, 1e-5)
    e2, _ = advanced_composition_epsilon(0.1, 200, 1e-5)
    assert e2 > e1


def test_advanced_composition_rejects_invalid_args():
    with pytest.raises(ValueError):
        advanced_composition_epsilon(0.1, 0, 1e-5)
    with pytest.raises(ValueError):
        advanced_composition_epsilon(0.1, 10, 0.0)
    with pytest.raises(ValueError):
        advanced_composition_epsilon(0.1, 10, 1.0)


# --------------------------------------------------------------------------- #
# Renyi DP -- the "tighter number" (§5.3)
# --------------------------------------------------------------------------- #
def test_rdp_composition_is_tighter_than_advanced_composition():
    adv_eps, _ = advanced_composition_epsilon(0.1, 200, 1e-5)
    rdp_eps = composed_rdp_epsilon(0.1, 200, 1e-5)
    assert rdp_eps < adv_eps


def test_laplace_rdp_rejects_invalid_alpha_or_epsilon():
    with pytest.raises(ValueError):
        laplace_rdp(alpha=1.0, epsilon0=0.1)
    with pytest.raises(ValueError):
        laplace_rdp(alpha=2.0, epsilon0=0.0)


def test_laplace_rdp_increases_with_epsilon0():
    lo = laplace_rdp(alpha=5.0, epsilon0=0.05)
    hi = laplace_rdp(alpha=5.0, epsilon0=0.5)
    assert hi > lo


def test_rdp_to_dp_is_minimized_over_alpha_grid():
    curve = {2.0: 0.5, 5.0: 0.3, 10.0: 0.4}
    best = rdp_to_dp(curve, delta=1e-5)
    candidates = [
        eps + math.log(1.0 / 1e-5) / (alpha - 1.0) for alpha, eps in curve.items()
    ]
    assert best == pytest.approx(min(candidates))


def test_rdp_to_dp_rejects_invalid_delta():
    with pytest.raises(ValueError):
        rdp_to_dp({2.0: 0.1}, delta=0.0)
    with pytest.raises(ValueError):
        rdp_to_dp({2.0: 0.1}, delta=1.0)


def test_composed_epsilon_reports_both_accountants():
    result = composed_epsilon(epsilon_round=0.1, rounds=200, delta=1e-5)
    assert set(result) == {
        "advanced_composition_epsilon", "advanced_composition_delta",
        "rdp_epsilon", "delta",
    }
    assert result["rdp_epsilon"] < result["advanced_composition_epsilon"]
    assert result["advanced_composition_epsilon"] == pytest.approx(8.89, abs=0.01)


# --------------------------------------------------------------------------- #
# §5.2's BottomN-recovery-under-DP claim, at unit-test scale (full 2000-trial
# version lives in scripts/verify_v2_plan_claims.py; this checks the
# mechanism, not the exact published percentages).
# --------------------------------------------------------------------------- #
def test_dp_noise_rarely_flips_bottom_n_when_gap_is_large():
    """A large gap between the rarest and next-rarest class survives Laplace
    noise at moderate epsilon with high probability -- the mechanism F3's
    recovery-rate table rests on."""
    from fedgraa.partition import bottom_n_rare_classes

    counts = np.array([100.0, 5000.0, 36.0, 652.0, 21.0, 11.0])  # class 5,4,2 rarest
    attack_ids = [1, 2, 3, 4, 5]
    oracle = set(bottom_n_rare_classes(counts, attack_ids, 3))

    rng = np.random.default_rng(0)
    hits = 0
    trials = 500
    for _ in range(trials):
        noised = laplace_mechanism(counts, epsilon=4.0, rng=rng)
        hits += set(bottom_n_rare_classes(noised, attack_ids, 3)) == oracle
    assert hits / trials > 0.95
