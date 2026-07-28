"""T-14/15 and T-16/17 -- the four test IDs
``FEDGRAA_V2_MASTER_PLAN.md`` §9.2 specifies but that had no implementation.

``docs/audit/PRE_COMPUTE_AUDIT_2026-07-28.md`` **M1** found that
``REMAINING_COMPUTE_WORK.md`` §1's claim -- *"Every one of §9.2's T-4 through
T-19 test IDs has a corresponding, passing test"* -- was false for exactly
these four. They are implemented here rather than withdrawn, because both
properties are cheap to check and both guard something the campaign is about
to rely on.

* **T-14/15 -- pipeline invariants.** "≥10 seeds present; no significance
  claim without p-value + effect size." *Fail loudly.*
* **T-16/17 -- property-based.** "Additivity over random partitions;
  ``Delta_1 = 2`` for neighbouring histograms."

§9.2 nominates ``hypothesis`` for T-16/17. It is deliberately **not** added:
``requirements.txt`` gains no new dependency mid-project (the same stance
``fedgraa/privacy.py`` took toward ``dp-accounting``/``opacus``), and the two
properties are total over a randomized input space that a seeded loop covers
directly. Each case below draws its own partition/neighbour from a fixed RNG,
so failures are reproducible without a shrinking engine.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedgraa.metrics import confusion_matrix, per_class_metrics_from_confusion
from fedgraa.privacy import SENSITIVITY
from fedgraa.stats import bootstrap_ci, cohens_d_paired, paired_ttest
from scripts.run_v2_frontier_sweep import DEFAULT_SEEDS, enumerate_cells

N_PROPERTY_CASES = 200


# =========================================================================== #
# T-14/15 -- pipeline invariants, failing loudly
# =========================================================================== #
def test_t14_campaign_enumerates_at_least_ten_seeds_per_cell():
    """Every published cell must rest on >= 10 seeds. The campaign's own
    default is the only place this is decided, so it is the place to pin it --
    a driver quietly defaulting to 3 seeds would produce a complete-looking
    results tree that no bootstrap CI in this project would be entitled to."""
    assert len(DEFAULT_SEEDS) >= 10

    for tiers in (["T1"], ["T2"], ["T4"], ["T1", "T2", "T4"]):
        cells = enumerate_cells(tiers)
        by_arm: dict = {}
        for c in cells:
            by_arm.setdefault((c.dataset, c.arm, c.gamma, c.config_tag),
                              set()).add(c.seed)
        assert by_arm, f"{tiers} enumerated nothing"
        for key, seeds in by_arm.items():
            assert len(seeds) >= 10, f"{key} has only {len(seeds)} seeds"


def test_t15_a_significance_claim_always_carries_a_p_value_and_effect_size():
    """"No significance claim without p-value + effect size" is only
    enforceable if both are actually computable from the same paired sample
    the claim is made about. This asserts the three pieces travel together and
    stay finite on a realistic 10-seed paired comparison."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.70, 0.05, size=10)
    b = rng.normal(0.62, 0.05, size=10)

    t_stat, p_value = paired_ttest(a, b)
    d = cohens_d_paired(a, b)
    lo, hi = bootstrap_ci(a - b, n_boot=2000, seed=0)

    for name, val in (("t", t_stat), ("p", p_value), ("cohens_d", d),
                      ("ci_lo", lo), ("ci_hi", hi)):
        assert np.isfinite(val), f"{name} is not finite"
    assert 0.0 <= p_value <= 1.0
    assert lo <= hi


def test_t15_a_degenerate_sample_reports_not_significant_never_a_huge_t():
    """"Fail loudly" here means *never emit a spurious significance claim*.

    A zero-variance difference vector is a literal 0/0 (or a division by
    zero), which ``scipy`` would turn into NaN or a platform-dependent huge
    t. ``paired_ttest`` handles this explicitly by returning ``(0.0, 1.0)``
    -- "not significant" -- and this test pins that contract, because the
    failure it prevents is a table cell reading ``p < 0.001`` off a
    single-seed or constant-difference comparison.
    """
    for a, b in (([0.7], [0.6]),                     # one seed
                 ([0.7, 0.8], [0.6, 0.7]),           # constant nonzero diff
                 ([0.7, 0.7, 0.7], [0.7, 0.7, 0.7])):  # all-zero diff
        t_stat, p_value = paired_ttest(a, b)
        assert np.isfinite(t_stat) and np.isfinite(p_value)
        assert (t_stat, p_value) == (0.0, 1.0), (
            f"degenerate sample {a} vs {b} produced t={t_stat}, p={p_value}"
        )


def test_t15_bootstrap_ci_brackets_the_sample_mean():
    rng = np.random.default_rng(1)
    for _ in range(20):
        sample = rng.normal(0.5, 0.1, size=10)
        lo, hi = bootstrap_ci(sample, n_boot=2000, seed=0)
        assert lo <= float(np.mean(sample)) <= hi


# =========================================================================== #
# T-16 -- confusion-matrix additivity over random partitions
# =========================================================================== #
def _random_partition(n: int, n_parts: int, rng) -> list:
    """Split ``range(n)`` into ``n_parts`` disjoint, exhaustive index blocks --
    including empty ones, which are the interesting case."""
    assignment = rng.integers(0, n_parts, size=n)
    return [np.flatnonzero(assignment == p) for p in range(n_parts)]


def _model_fixture(n_features: int, n_classes: int, seed: int):
    """A real ``(params, layer_sizes)`` pair -- ``confusion_matrix`` takes model
    parameters and features, not two label vectors, so the additivity property
    has to be exercised through an actual forward pass to mean anything."""
    from fedgraa.model import init_params

    layer_sizes = [n_features, 8, n_classes]
    return init_params(layer_sizes, seed=seed)


def test_t16_confusion_matrix_is_additive_over_any_random_partition():
    """The pooled confusion matrix must equal the elementwise sum of the
    per-part matrices, for *any* partition of the rows -- including ones that
    leave a part empty or a class entirely absent from a part.

    This is the property the federated evaluation story rests on: it is why
    per-client metrics can be summed rather than re-derived, and why
    ``per_class_metrics`` could be refactored onto
    ``per_class_metrics_from_confusion`` without behaviour change (W1.4).
    Exercised through the real forward pass, so a change to ``predict``'s
    tie-breaking or batching would be caught too.
    """
    rng = np.random.default_rng(0)
    for case in range(N_PROPERTY_CASES):
        n_classes = int(rng.integers(2, 6))
        n_features = int(rng.integers(2, 8))
        n = int(rng.integers(1, 120))

        params = _model_fixture(n_features, n_classes, seed=case)
        X = rng.normal(size=(n, n_features)).astype(np.float32)
        y = rng.integers(0, n_classes, size=n)

        pooled = confusion_matrix(params, X, y, n_classes)
        parts = _random_partition(n, int(rng.integers(1, 5)), rng)
        summed = np.zeros_like(pooled)
        for idx in parts:
            summed += confusion_matrix(params, X[idx], y[idx], n_classes)

        np.testing.assert_array_equal(
            pooled, summed,
            err_msg=f"additivity failed on case {case} "
                    f"(n={n}, C={n_classes}, {len(parts)} parts)",
        )
        assert pooled.sum() == n, "every row must land in exactly one cell"


def test_t16_additivity_holds_when_a_part_is_empty():
    """The degenerate part is the one a naive implementation drops."""
    rng = np.random.default_rng(7)
    params = _model_fixture(4, 3, seed=0)
    X = rng.normal(size=(12, 4)).astype(np.float32)
    y = rng.integers(0, 3, size=12)

    pooled = confusion_matrix(params, X, y, 3)
    summed = (confusion_matrix(params, X, y, 3)
              + confusion_matrix(params, X[:0], y[:0], 3))
    np.testing.assert_array_equal(pooled, summed)


def test_t16_per_class_metrics_from_summed_parts_equal_the_pooled_computation():
    """T-2's "macro-F1 from the pooled matrix" property, re-asserted over
    random partitions rather than a single fixture."""
    rng = np.random.default_rng(2)
    for case in range(40):
        n_classes = int(rng.integers(2, 5))
        n_features = int(rng.integers(2, 6))
        n = int(rng.integers(20, 120))

        params = _model_fixture(n_features, n_classes, seed=case)
        X = rng.normal(size=(n, n_features)).astype(np.float32)
        y = rng.integers(0, n_classes, size=n)

        summed = np.zeros((n_classes, n_classes), dtype=np.int64)
        for idx in _random_partition(n, 3, rng):
            summed += confusion_matrix(params, X[idx], y[idx], n_classes)

        direct = per_class_metrics_from_confusion(
            confusion_matrix(params, X, y, n_classes)
        )
        from_parts = per_class_metrics_from_confusion(summed)
        assert direct.keys() == from_parts.keys()
        for cls in direct:
            for key in direct[cls]:
                assert direct[cls][key] == pytest.approx(
                    from_parts[cls][key], abs=1e-12
                )


# =========================================================================== #
# T-17 -- Delta_1 = 2 for neighbouring label histograms
# =========================================================================== #
def _histogram(labels: np.ndarray, n_classes: int) -> np.ndarray:
    return np.bincount(labels, minlength=n_classes).astype(np.int64)


def test_t17_l1_sensitivity_of_a_label_histogram_is_exactly_two():
    """``SENSITIVITY = 2`` is the constant every epsilon in this project is
    computed from (``privacy.py``), and it is asserted nowhere else. Under
    *change-one-row* neighbouring, one row moves from one class's count to
    another's: one count falls by 1, another rises by 1, so ``||h - h'||_1 =
    2`` exactly -- never more.
    """
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(N_PROPERTY_CASES):
        n_classes = int(rng.integers(2, 12))
        n = int(rng.integers(1, 300))
        labels = rng.integers(0, n_classes, size=n)

        i = int(rng.integers(0, n))
        new_label = int(rng.integers(0, n_classes))
        neighbour = labels.copy()
        neighbour[i] = new_label

        l1 = float(np.abs(
            _histogram(labels, n_classes) - _histogram(neighbour, n_classes)
        ).sum())
        assert l1 <= SENSITIVITY, (
            f"change-one-row L1 distance {l1} exceeds the sensitivity "
            f"{SENSITIVITY} the Laplace scale is derived from"
        )
        worst = max(worst, l1)

    assert worst == SENSITIVITY, (
        "the bound was never attained, so this test would not have detected a "
        "sensitivity that is actually larger than 2"
    )


def test_t17_add_or_remove_one_row_moves_l1_by_exactly_one():
    """Under *add/remove-one-row* neighbouring the distance is 1, so the
    change-one-row bound of 2 dominates -- which is why ``SENSITIVITY = 2`` is
    the conservative, correct choice for a single release."""
    rng = np.random.default_rng(3)
    for _ in range(100):
        n_classes = int(rng.integers(2, 10))
        n = int(rng.integers(2, 200))
        labels = rng.integers(0, n_classes, size=n)
        removed = np.delete(labels, int(rng.integers(0, n)))
        l1 = float(np.abs(
            _histogram(labels, n_classes) - _histogram(removed, n_classes)
        ).sum())
        assert l1 == 1.0
