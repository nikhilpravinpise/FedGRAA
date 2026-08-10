"""Unit tests for the FedGRAA-v2 frontier significance layer
(docs/research_c/STAGE_C_REMAINING_WORK_MASTER_PLAN_2026-08-02.md §3).

Feeds tiny, hand-written frontier-table dicts (the same schema
``build_v2_frontier_table.build_table`` emits) through
``paired_comparison``/``build_significance_table``, mirroring
``tests/test_build_v2_frontier_table.py``'s no-real-campaign-data pattern.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_v2_significance_table import (  # noqa: E402
    build_significance_table,
    paired_comparison,
    render_markdown,
)

DATASET = "TOY-DS"


def _cell(rare_acc_values, seeds=None):
    seeds = seeds if seeds is not None else list(range(len(rare_acc_values)))
    return {
        "n_seeds": len(rare_acc_values),
        "seeds": seeds,
        "rare_acc": {"per_seed": list(rare_acc_values), "mean": sum(rare_acc_values) / len(rare_acc_values),
                     "ci_95": [0.0, 1.0]},
        "all_acc": {"per_seed": [0.5] * len(rare_acc_values), "mean": 0.5, "ci_95": [0.0, 1.0]},
        "bytes_sent": {"per_seed": [1_000_000_000] * len(rare_acc_values), "median_gb": 1.0},
        "median_zeta": None,
    }


# --------------------------------------------------------------------------- #
# paired_comparison
# --------------------------------------------------------------------------- #
def test_paired_comparison_matches_hand_computed_scipy_wilcoxon():
    """Regression test per the plan's §3 step 2: the new function's output
    must match a directly hand-computed scipy.stats.wilcoxon call on a fixed
    synthetic pair, not just "run without crashing"."""
    a_vals = [0.90, 0.92, 0.88, 0.95, 0.91, 0.89, 0.93, 0.94, 0.90, 0.92]
    b_vals = [0.80, 0.82, 0.78, 0.85, 0.81, 0.79, 0.83, 0.84, 0.80, 0.82]
    cell_a = _cell(a_vals)
    cell_b = _cell(b_vals)

    result = paired_comparison(cell_a, cell_b)

    expected_stat, expected_p = scipy_stats.wilcoxon(np.array(a_vals), np.array(b_vals))
    assert result["wilcoxon_statistic"] == pytest.approx(float(expected_stat))
    assert result["p_wilcoxon"] == pytest.approx(float(expected_p))
    assert result["n_paired_seeds"] == 10
    assert result["v2_mean"] == pytest.approx(sum(a_vals) / 10)
    assert result["baseline_mean"] == pytest.approx(sum(b_vals) / 10)
    assert result["mean_diff"] == pytest.approx(result["v2_mean"] - result["baseline_mean"])


def test_paired_comparison_matches_only_common_seeds():
    cell_a = _cell([0.9, 0.8, 0.7, 0.6], seeds=[0, 1, 2, 3])
    cell_b = _cell([0.1, 0.2, 0.3], seeds=[1, 2, 4])
    result = paired_comparison(cell_a, cell_b)
    assert result["n_paired_seeds"] == 2
    assert result["seeds"] == [1, 2]


def test_paired_comparison_no_common_seeds_returns_none():
    cell_a = _cell([0.9], seeds=[0])
    cell_b = _cell([0.1], seeds=[1])
    assert paired_comparison(cell_a, cell_b) is None


def test_paired_comparison_identical_series_is_not_significant():
    cell_a = _cell([0.5] * 10)
    cell_b = _cell([0.5] * 10)
    result = paired_comparison(cell_a, cell_b)
    assert result["p_wilcoxon"] == 1.0


# --------------------------------------------------------------------------- #
# build_significance_table
# --------------------------------------------------------------------------- #
def _toy_frontier_table():
    strong = [0.95, 0.94, 0.96, 0.93, 0.95, 0.97, 0.94, 0.96, 0.95, 0.94]
    weak = [0.50, 0.52, 0.48, 0.51, 0.49, 0.53, 0.50, 0.47, 0.52, 0.49]
    return {
        "datasets": {
            DATASET: {
                "v1_projected": _cell(weak),
                "v2_m1": _cell(strong),
                "v2_m3": _cell(strong),
                "baseline_a": _cell(weak),
                "baseline_b": _cell(weak),
            }
        }
    }


def test_build_significance_table_excludes_v1_projected_as_focal_arm():
    ft = _toy_frontier_table()
    sig = build_significance_table(
        ft, datasets=[DATASET], v2_arms=["v1_projected", "v2_m1", "v2_m3"],
        baseline_arms=["baseline_a", "baseline_b"],
    )
    assert "v1_projected" not in sig["v2_arms"]
    assert set(sig["v2_arms"]) == {"v2_m1", "v2_m3"}


def test_build_significance_table_applies_holm_correction_per_dataset_arm_family():
    ft = _toy_frontier_table()
    sig = build_significance_table(
        ft, datasets=[DATASET], v2_arms=["v2_m1", "v2_m3"],
        baseline_arms=["baseline_a", "baseline_b"],
    )
    for arm in ("v2_m1", "v2_m3"):
        per_baseline = sig["datasets"][DATASET][arm]
        for baseline in ("baseline_a", "baseline_b"):
            assert per_baseline[baseline]["p_holm"] >= per_baseline[baseline]["p_wilcoxon"]
        # A clearly separated strong-vs-weak pair should survive Holm correction.
        assert per_baseline["baseline_a"]["significant"] is True
    assert sig["n_comparisons"] == 4  # 2 v2 arms x 2 baselines


def test_build_significance_table_missing_baseline_data_is_none_not_counted():
    ft = _toy_frontier_table()
    sig = build_significance_table(
        ft, datasets=[DATASET], v2_arms=["v2_m1"],
        baseline_arms=["baseline_a", "nonexistent_baseline"],
    )
    assert sig["datasets"][DATASET]["v2_m1"]["nonexistent_baseline"] is None
    assert sig["n_comparisons"] == 1


def test_build_significance_table_missing_v2_arm_data_yields_empty_family():
    ft = _toy_frontier_table()
    sig = build_significance_table(
        ft, datasets=[DATASET], v2_arms=["v2_m2"],  # no v2_m2 cell exists
        baseline_arms=["baseline_a"],
    )
    assert sig["datasets"][DATASET]["v2_m2"] == {}
    assert sig["n_comparisons"] == 0


# --------------------------------------------------------------------------- #
# render_markdown
# --------------------------------------------------------------------------- #
def test_render_markdown_includes_every_arm_and_baseline():
    ft = _toy_frontier_table()
    sig = build_significance_table(
        ft, datasets=[DATASET], v2_arms=["v2_m1", "v2_m3"],
        baseline_arms=["baseline_a", "baseline_b"],
    )
    md = render_markdown(sig)
    assert DATASET in md
    assert "v2_m1" in md and "v2_m3" in md
    assert "baseline_a" in md and "baseline_b" in md
    assert "significant" in md
