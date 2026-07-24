"""Unit tests for ``fedgraa/stats.py`` (stage_b_plan.md T4/§6.1).

Validates every primitive against a SciPy or hand-computed reference, and
covers the zero-variance / identical-differences edge cases the Stage B
ablation table depends on (stage_b_plan.md §4): the combined-fix arm lands
on identical rare-class accuracy across its first three committed seeds, so
a paired test or Cohen's d run against it must not blow up into ``NaN``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats

from fedgraa.stats import (
    bootstrap_ci,
    cohens_d_paired,
    holm_bonferroni,
    paired_ttest,
    wilcoxon_signed_rank,
)


# --------------------------------------------------------------------------- #
# bootstrap_ci
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_matches_seeded_reference_implementation():
    """A bit-exact reproduction of the RNG contract (default_rng(seed).integers
    resampling + percentile), coded independently in the test, must match the
    module's output exactly -- this is a regression guard on the seeding
    scheme itself, since anyone changing it would silently break
    reproducibility of every committed CI in the ablation table."""
    values = np.array([0.60, 0.65, 0.62, 0.58, 0.70, 0.55, 0.64, 0.61, 0.59, 0.66])
    seed = 42

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(10_000, len(values)))
    resampled_means = values[idx].mean(axis=1)
    expected_lo, expected_hi = np.percentile(resampled_means, [2.5, 97.5])

    lo, hi = bootstrap_ci(values, seed=seed)
    assert lo == pytest.approx(expected_lo)
    assert hi == pytest.approx(expected_hi)


def test_bootstrap_ci_contains_true_mean_for_symmetric_data():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=10.0, scale=1.0, size=200)
    lo, hi = bootstrap_ci(values, seed=0)
    assert lo < 10.0 < hi


def test_bootstrap_ci_constant_vector_returns_point_ci_no_nan():
    values = [0.917, 0.917, 0.917]
    lo, hi = bootstrap_ci(values)
    assert lo == pytest.approx(0.917)
    assert hi == pytest.approx(0.917)
    assert not np.isnan(lo) and not np.isnan(hi)


def test_bootstrap_ci_is_deterministic_across_calls():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    r1 = bootstrap_ci(values, seed=7)
    r2 = bootstrap_ci(values, seed=7)
    assert r1 == r2


def test_bootstrap_ci_falls_back_for_statistics_without_axis_kwarg():
    """Custom statistics that don't accept an ``axis=`` kwarg (e.g. a plain
    lambda) must still work, via the per-row fallback loop."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    lo, hi = bootstrap_ci(values, statistic=lambda x: float(np.mean(x)), seed=0)
    lo_ref, hi_ref = bootstrap_ci(values, statistic=np.mean, seed=0)
    assert lo == pytest.approx(lo_ref)
    assert hi == pytest.approx(hi_ref)


def test_bootstrap_ci_empty_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([])


# --------------------------------------------------------------------------- #
# paired_ttest
# --------------------------------------------------------------------------- #
def test_paired_ttest_matches_scipy_for_nondegenerate_data():
    a = np.array([0.80, 0.75, 0.60, 0.90, 0.85, 0.70, 0.65, 0.95, 0.72, 0.68])
    b = np.array([0.60, 0.55, 0.50, 0.65, 0.62, 0.58, 0.52, 0.70, 0.56, 0.54])
    t, p = paired_ttest(a, b)
    t_ref, p_ref = scipy_stats.ttest_rel(a, b)
    assert t == pytest.approx(t_ref)
    assert p == pytest.approx(p_ref)


def test_paired_ttest_identical_diffs_zero_returns_point_result():
    a = np.array([0.5, 0.6, 0.7])
    b = np.array([0.5, 0.6, 0.7])
    t, p = paired_ttest(a, b)
    assert t == 0.0
    assert p == 1.0


def test_paired_ttest_identical_diffs_nonzero_returns_point_result_not_nan():
    """The exact zero-variance case flagged in stage_b_plan.md §4: e.g.
    'both' arm constant at 91.7% across seeds vs. a FedAvg reference that is
    also seed-independent (0.0% every seed, per STAGE_A_MASTER_REPORT.md
    §5.6.2) -- the paired difference is identical (91.7) at every seed."""
    both = np.array([0.917, 0.917, 0.917])
    fedavg = np.array([0.0, 0.0, 0.0])
    t, p = paired_ttest(both, fedavg)
    assert t == 0.0
    assert p == 1.0
    assert not np.isnan(t) and not np.isnan(p)


# --------------------------------------------------------------------------- #
# wilcoxon_signed_rank
# --------------------------------------------------------------------------- #
def test_wilcoxon_matches_scipy_for_nondegenerate_data():
    a = np.array([0.80, 0.75, 0.60, 0.90, 0.85, 0.70, 0.65, 0.95, 0.72, 0.68])
    b = np.array([0.60, 0.55, 0.50, 0.65, 0.62, 0.58, 0.52, 0.70, 0.56, 0.54])
    stat, p = wilcoxon_signed_rank(a, b)
    stat_ref, p_ref = scipy_stats.wilcoxon(a, b)
    assert stat == pytest.approx(stat_ref)
    assert p == pytest.approx(p_ref)


def test_wilcoxon_all_zero_diffs_is_safe_not_an_exception():
    """Raw ``scipy.stats.wilcoxon`` on all-zero paired differences either
    raises ``ValueError`` (older SciPy, all ties dropped under the default
    zero_method) or emits ``NaN`` with a RuntimeWarning (SciPy >=1.18, this
    repo's pinned version) -- neither is a usable result. Our wrapper must
    intercept the degeneracy up front and return a clean, safe point value
    regardless of which behavior the installed SciPy exhibits."""
    a = np.array([0.5, 0.6, 0.7, 0.8])
    b = np.array([0.5, 0.6, 0.7, 0.8])
    stat, p = wilcoxon_signed_rank(a, b)
    assert stat == 0.0
    assert p == 1.0
    assert not np.isnan(stat) and not np.isnan(p)


# --------------------------------------------------------------------------- #
# cohens_d_paired
# --------------------------------------------------------------------------- #
def test_cohens_d_paired_matches_hand_computation():
    a = np.array([2.0, 4.0, 6.0, 8.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    diffs = a - b
    expected = np.mean(diffs) / np.std(diffs, ddof=1)
    d = cohens_d_paired(a, b)
    assert d == pytest.approx(expected)


def test_cohens_d_paired_zero_diff_is_zero_not_nan():
    a = np.array([0.5, 0.5, 0.5])
    b = np.array([0.5, 0.5, 0.5])
    d = cohens_d_paired(a, b)
    assert d == 0.0
    assert not np.isnan(d)


def test_cohens_d_paired_constant_nonzero_diff_is_signed_infinity():
    a = np.array([0.917, 0.917, 0.917])
    b = np.array([0.0, 0.0, 0.0])
    d = cohens_d_paired(a, b)
    assert d == float("inf")

    d_neg = cohens_d_paired(b, a)
    assert d_neg == float("-inf")


# --------------------------------------------------------------------------- #
# holm_bonferroni
# --------------------------------------------------------------------------- #
def test_holm_bonferroni_matches_worked_example():
    # Classic ascending worked example, m=5.
    pvals = [0.01, 0.02, 0.03, 0.04, 0.05]
    adjusted = holm_bonferroni(pvals)
    # i=0: 0.01*5=0.05 | i=1: 0.02*4=0.08 | i=2: 0.03*3=0.09
    # i=3: 0.04*2=0.08 -> monotone max with prior (0.09) -> 0.09
    # i=4: 0.05*1=0.05 -> monotone max with prior (0.09) -> 0.09
    expected = [0.05, 0.08, 0.09, 0.09, 0.09]
    assert adjusted == pytest.approx(expected)


def test_holm_bonferroni_is_order_preserving():
    pvals = [0.03, 0.01, 0.05, 0.02, 0.04]
    adjusted = holm_bonferroni(pvals)
    # Same underlying set as the ascending test, permuted; the i-th output
    # must correspond to the i-th input, not a re-sorted list.
    by_sorted_input = dict(zip(sorted(pvals), holm_bonferroni(sorted(pvals))))
    for orig_p, adj_p in zip(pvals, adjusted):
        assert adj_p == pytest.approx(by_sorted_input[orig_p])


def test_holm_bonferroni_matches_statsmodels_style_reference():
    """Independent reference implementation (textbook Holm step-down),
    compared cell-by-cell against the module's output on a mixed example."""
    pvals = [0.005, 0.011, 0.02, 0.04, 0.13]

    def reference_holm(p):
        p = np.asarray(p, dtype=float)
        m = len(p)
        order = np.argsort(p)
        sorted_p = p[order]
        adj = np.empty(m)
        running_max = 0.0
        for i in range(m):
            val = min(sorted_p[i] * (m - i), 1.0)
            running_max = max(running_max, val)
            adj[i] = running_max
        out = np.empty(m)
        out[order] = adj
        return out.tolist()

    assert holm_bonferroni(pvals) == pytest.approx(reference_holm(pvals))


def test_holm_bonferroni_empty_list():
    assert holm_bonferroni([]) == []


def test_holm_bonferroni_clips_at_one():
    pvals = [0.9, 0.8, 0.95]
    adjusted = holm_bonferroni(pvals)
    assert all(p <= 1.0 for p in adjusted)
