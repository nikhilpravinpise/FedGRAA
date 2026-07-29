"""Unit tests for the FedGRAA-v2 frontier-table builder
(FEDGRAA_V2_MASTER_PLAN.md Sec 4.2's "headline figure").

Feeds tiny, hand-written per-cell result JSONs -- written via the same
``CellSpec.path`` naming convention ``run_one_cell`` uses -- through
``build_arm_cell``/``build_table``, mirroring
``tests/test_build_ablation_table.py``'s pattern for Stage B. No real
campaign data or training is needed.

The seed-count-contamination regression this file guards against
(``test_smoke_leftover_cell_is_excluded_from_seed_count``) is not
hypothetical: an earlier version of ``build_v2_frontier_table.py`` counted
11 seeds instead of 10 for exactly this reason, because a real
``--smoke`` (2-round) verification run left a same-arm, same-dataset,
same-seed-0 result JSON sitting in the same output directory as the real
10-seed campaign, and the glob used to collect cells matched both.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_v2_frontier_table import (  # noqa: E402
    ALL_ARMS,
    _is_t1_cell,
    _median_zeta,
    build_arm_cell,
    build_table,
)
from scripts.run_v2_frontier_sweep import CellSpec  # noqa: E402

DATASET = "TOY-DS"


def _write_cell(out_dir: str, **overrides) -> str:
    """Write a minimal result JSON at the exact path ``run_one_cell`` would
    use, so the builder's globbing exercises the real naming convention."""
    defaults = dict(
        dataset=DATASET, arm="v2_m3", gamma=1.0, seed=0, rounds=200,
        out_dir=out_dir,
    )
    defaults.update({k: v for k, v in overrides.items() if k in CellSpec.__dataclass_fields__})
    cell = CellSpec(**defaults)
    os.makedirs(out_dir, exist_ok=True)

    final_last_n_acc = overrides.get("final_last_n_acc", 0.9)
    final_all_acc = overrides.get("final_all_acc", 0.5)
    bytes_sent = overrides.get("bytes_sent", 1_000_000_000)
    zeta = overrides.get("zeta", [1e-7, 2e-7, None])
    p_drop = overrides.get("p_drop", 0.0)
    dp_epsilon = overrides.get("dp_epsilon", None)
    n_rounds = overrides.get("n_rounds", cell.rounds)

    payload = {
        "dataset": cell.dataset, "arm": cell.arm, "gamma": cell.gamma,
        "seed": cell.seed,
        "config": {
            "n_rounds": n_rounds, "p_drop": p_drop, "dp_epsilon": dp_epsilon,
        },
        "final_last_n_acc": final_last_n_acc,
        "final_all_acc": final_all_acc,
        "bytes_sent": bytes_sent,
        "zeta": zeta,
    }
    with open(cell.path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return cell.path


# --------------------------------------------------------------------------- #
# _is_t1_cell
# --------------------------------------------------------------------------- #
def test_is_t1_cell_accepts_canonical_gamma1_cell():
    data = {"gamma": 1.0, "config": {"n_rounds": 200, "p_drop": 0.0, "dp_epsilon": None}}
    assert _is_t1_cell(data)


def test_is_t1_cell_rejects_other_gamma():
    data = {"gamma": 0.5, "config": {"n_rounds": 200, "p_drop": 0.0, "dp_epsilon": None}}
    assert not _is_t1_cell(data)


def test_is_t1_cell_rejects_p_drop_or_dp_epsilon():
    base = {"gamma": 1.0, "config": {"n_rounds": 200, "p_drop": 0.0, "dp_epsilon": None}}
    pdrop = {**base, "config": {**base["config"], "p_drop": 0.1}}
    eps = {**base, "config": {**base["config"], "dp_epsilon": 4.0}}
    assert not _is_t1_cell(pdrop)
    assert not _is_t1_cell(eps)


def test_is_t1_cell_rejects_non_canonical_round_count():
    """The regression this whole file exists to guard: a 2-round smoke-test
    leftover must never be counted as a real frontier seed."""
    data = {"gamma": 1.0, "config": {"n_rounds": 2, "p_drop": 0.0, "dp_epsilon": None}}
    assert not _is_t1_cell(data)


# --------------------------------------------------------------------------- #
# _median_zeta
# --------------------------------------------------------------------------- #
def test_median_zeta_drops_none_entries():
    assert _median_zeta([1.0, None, 3.0, None]) == 2.0


def test_median_zeta_all_none_returns_none():
    assert _median_zeta([None, None]) is None


def test_median_zeta_empty_returns_none():
    assert _median_zeta([]) is None


# --------------------------------------------------------------------------- #
# build_arm_cell / build_table
# --------------------------------------------------------------------------- #
def test_smoke_leftover_cell_is_excluded_from_seed_count(tmp_path):
    out_dir = str(tmp_path)
    # 10 real 200-round seeds.
    for seed in range(10):
        _write_cell(out_dir, arm="v2_m3", seed=seed, final_last_n_acc=0.9)
    # One 2-round smoke leftover for the SAME arm/dataset/seed=0 -- a
    # distinct filename (CellSpec.config_tag appends "_r2"), but the
    # builder's glob is keyed on {dataset}_{arm}_g* and would match it too
    # if _is_t1_cell didn't filter on rounds.
    _write_cell(out_dir, arm="v2_m3", seed=0, rounds=2, n_rounds=2,
                final_last_n_acc=0.0)

    cell = build_arm_cell(DATASET, "v2_m3", results_dir=out_dir)
    assert cell["n_seeds"] == 10
    assert cell["rare_acc"]["mean"] == 0.9  # smoke's 0.0 must not pull this down


def test_build_arm_cell_returns_none_for_missing_arm(tmp_path):
    assert build_arm_cell(DATASET, "v2_m3", results_dir=str(tmp_path)) is None


def test_build_arm_cell_computes_mean_and_ci(tmp_path):
    out_dir = str(tmp_path)
    values = [0.80, 0.85, 0.90, 0.95, 1.00]
    for seed, v in enumerate(values):
        _write_cell(out_dir, arm="v2_m1", seed=seed, final_last_n_acc=v,
                    bytes_sent=2_000_000_000 + seed)

    cell = build_arm_cell(DATASET, "v2_m1", results_dir=out_dir)
    assert cell["n_seeds"] == 5
    assert cell["rare_acc"]["mean"] == sum(values) / len(values)
    lo, hi = cell["rare_acc"]["ci_95"]
    assert lo <= cell["rare_acc"]["mean"] <= hi
    assert cell["bytes_sent"]["median_gb"] > 0


def test_build_arm_cell_median_zeta_only_for_arms_with_zeta_data(tmp_path):
    out_dir = str(tmp_path)
    for seed in range(3):
        _write_cell(out_dir, arm="v2_m2", seed=seed, zeta=[1e-7, 3e-7])
    v2_cell = build_arm_cell(DATASET, "v2_m2", results_dir=out_dir)
    assert v2_cell["median_zeta"] is not None
    assert v2_cell["median_zeta"] > 0

    for seed in range(3):
        _write_cell(out_dir, arm="celm", seed=seed, zeta=[None, None])
    baseline_cell = build_arm_cell(DATASET, "celm", results_dir=out_dir)
    assert baseline_cell["median_zeta"] is None


def test_build_table_covers_every_arm_present_on_disk(tmp_path):
    out_dir = str(tmp_path)
    for seed in range(2):
        _write_cell(out_dir, arm="v1_projected", seed=seed)
        _write_cell(out_dir, arm="fltrust", seed=seed)

    table = build_table(datasets=[DATASET], results_dir=out_dir)
    assert set(table["datasets"][DATASET].keys()) == {"v1_projected", "fltrust"}
    assert set(ALL_ARMS) >= {"v1_projected", "fltrust"}
