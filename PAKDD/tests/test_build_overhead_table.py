"""Unit tests for the communication-overhead table
(docs/research_c/STAGE_C_REMAINING_WORK_MASTER_PLAN_2026-08-02.md §5).

Feeds tiny, hand-written frontier-table dicts through
``overhead_row``/``build_overhead_table``, mirroring
``tests/test_build_v2_significance_table.py``'s no-real-campaign-data
pattern.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_overhead_table import (  # noqa: E402
    build_overhead_table,
    overhead_row,
    render_markdown,
)

DATASET = "TOY-DS"


def _cell(rare_acc_mean, bytes_gb):
    return {
        "n_seeds": 10,
        "seeds": list(range(10)),
        "rare_acc": {"per_seed": [rare_acc_mean] * 10, "mean": rare_acc_mean, "ci_95": [0.0, 1.0]},
        "all_acc": {"per_seed": [0.5] * 10, "mean": 0.5, "ci_95": [0.0, 1.0]},
        "bytes_sent": {"per_seed": [bytes_gb * 1e9] * 10, "median_gb": bytes_gb},
        "median_zeta": None,
    }


# --------------------------------------------------------------------------- #
# overhead_row
# --------------------------------------------------------------------------- #
def test_overhead_row_computes_exact_6x_ratio():
    v2_cell = _cell(rare_acc_mean=0.88, bytes_gb=6.0)
    approx_cell = _cell(rare_acc_mean=0.10, bytes_gb=1.0)
    row = overhead_row(v2_cell, approx_cell)
    assert row["byte_ratio"] == pytest.approx(6.0)
    assert row["rare_acc_delta"] == pytest.approx(0.78)
    assert row["v2_bytes_gb"] == 6.0
    assert row["approx_bytes_gb"] == 1.0


def test_overhead_row_handles_zero_approx_bytes_as_infinite_ratio():
    v2_cell = _cell(rare_acc_mean=0.9, bytes_gb=6.0)
    approx_cell = _cell(rare_acc_mean=0.1, bytes_gb=0.0)
    row = overhead_row(v2_cell, approx_cell)
    assert row["byte_ratio"] == float("inf")


# --------------------------------------------------------------------------- #
# build_overhead_table
# --------------------------------------------------------------------------- #
def _toy_frontier_table():
    return {
        "datasets": {
            DATASET: {
                "v1_projected": _cell(0.10, 0.5),
                "v2_m1": _cell(0.88, 6.0),
                "v2_m3": _cell(0.90, 6.5),
                "celm": _cell(0.05, 1.0),
                "spectralfed": _cell(0.07, 1.2),
            }
        }
    }


def test_build_overhead_table_excludes_v1_projected_as_focal_arm():
    ft = _toy_frontier_table()
    table = build_overhead_table(
        ft, datasets=[DATASET], v2_arms=["v1_projected", "v2_m1", "v2_m3"],
        approx_baselines=["celm", "spectralfed"],
    )
    assert "v1_projected" not in table["v2_arms"]
    assert set(table["v2_arms"]) == {"v2_m1", "v2_m3"}


def test_build_overhead_table_covers_every_v2_arm_x_approx_baseline_pair():
    ft = _toy_frontier_table()
    table = build_overhead_table(
        ft, datasets=[DATASET], v2_arms=["v2_m1", "v2_m3"],
        approx_baselines=["celm", "spectralfed"],
    )
    for arm in ("v2_m1", "v2_m3"):
        assert set(table["datasets"][DATASET][arm].keys()) == {"celm", "spectralfed"}
        for approx in ("celm", "spectralfed"):
            assert table["datasets"][DATASET][arm][approx] is not None


def test_build_overhead_table_missing_baseline_data_is_none():
    ft = _toy_frontier_table()
    table = build_overhead_table(
        ft, datasets=[DATASET], v2_arms=["v2_m1"],
        approx_baselines=["celm", "nonexistent"],
    )
    assert table["datasets"][DATASET]["v2_m1"]["nonexistent"] is None
    assert table["datasets"][DATASET]["v2_m1"]["celm"] is not None


def test_build_overhead_table_missing_v2_arm_yields_empty_family():
    ft = _toy_frontier_table()
    table = build_overhead_table(
        ft, datasets=[DATASET], v2_arms=["v2_m2"],  # no v2_m2 cell exists
        approx_baselines=["celm"],
    )
    assert table["datasets"][DATASET]["v2_m2"] == {}


# --------------------------------------------------------------------------- #
# render_markdown
# --------------------------------------------------------------------------- #
def test_render_markdown_includes_ratio_and_delta():
    ft = _toy_frontier_table()
    table = build_overhead_table(
        ft, datasets=[DATASET], v2_arms=["v2_m1"], approx_baselines=["celm"],
    )
    md = render_markdown(table)
    assert DATASET in md
    assert "v2_m1" in md
    assert "celm" in md
    assert "6.00x" in md
