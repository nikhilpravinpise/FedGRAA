"""Consolidated FedGRAA-v2 frontier-table builder (FEDGRAA_V2_MASTER_PLAN.md
Sec 4.2's "headline figure").

Reads every per-cell result JSON `scripts/run_v2_frontier_sweep.py` wrote
under `results/v2_frontier/` for the T1 (gamma=1.0, no p_drop/dp_epsilon)
frontier, plus T3/T5's 13 external-baseline arms, and emits
`results/v2_frontier/frontier_table.json`: per (dataset, arm) cell the
per-seed arrays + mean + bootstrap 95% CI for rare-class recall
(`final_last_n_acc`) and all-class accuracy (`final_all_acc`), the median
communication cost (`bytes_sent`, GB), and -- for the v2 arms only, per
FEDGRAA_V2_MASTER_PLAN.md Sec 4.2's own instruction to "annotate each v2
point with its measured median zeta_t" -- the across-seed mean of each
seed's own median-over-rounds zeta.

This mirrors `scripts/build_ablation_table.py`'s JSON schema/CI
conventions (`fedgraa/stats.py::bootstrap_ci`) but reads
`run_one_cell`'s own result-dict shape directly rather than the Stage B
diagnostic-script schema, since Stage C's cells are a different, already-
self-describing format (`config`, `bytes_sent`, `zeta`, etc. all persisted
per cell -- see `run_one_cell`'s own comment on why).

Usage::

    python scripts/build_v2_frontier_table.py
    python scripts/build_v2_frontier_table.py --results-dir results/v2_frontier
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedgraa.stats import bootstrap_ci  # noqa: E402
from scripts.run_v2_frontier_sweep import (  # noqa: E402
    T3_ARM_CLASSES,
    T5_ARM_CLASSES,
)

RESULTS_DIR = "results/v2_frontier"
DATASETS = ["CIC-IDS-2017", "UNSW-NB15"]
V2_FRONTIER_ARMS = ["v1_projected", "v2_m1", "v2_m2", "v2_m3", "v2_m5"]
BASELINE_ARMS = sorted(T3_ARM_CLASSES) + sorted(T5_ARM_CLASSES)
ALL_ARMS = V2_FRONTIER_ARMS + BASELINE_ARMS


CANONICAL_ROUNDS = 200


def _is_t1_cell(data: dict) -> bool:
    """A cell belongs on the frontier's gamma=1.0 anchor with no robustness
    perturbation applied -- excludes T2's gamma sweep and T4's p_drop/
    dp_epsilon cells, which are separate axes, not separate frontier points.

    Also excludes any cell that isn't a full CANONICAL_ROUNDS-round run.
    Several `--smoke` invocations (2 rounds, this same out-dir) were run
    against the real Modal volume this pass, to verify the campaign path
    before committing to the full launch -- see
    docs/research_c/REMAINING_COMPUTE_WORK.md's own account of this. Their
    results never collide with a real cell's filename (`CellSpec.config_tag`
    appends `_r2` whenever `rounds != 200`), but a glob keyed only on
    `{dataset}_{arm}_g*` still matches both -- confirmed the hard way: an
    early version of this function counted 11 seeds instead of 10 for the
    arms/datasets a smoke test happened to touch.
    """
    if data["gamma"] != 1.0:
        return False
    if data["config"]["n_rounds"] != CANONICAL_ROUNDS:
        return False
    cfg = data["config"]
    return not cfg["p_drop"] and cfg["dp_epsilon"] is None


def _median_zeta(zeta_series: List[Optional[float]]) -> Optional[float]:
    """One seed's own median zeta over the rounds it was actually measured
    at. `None` entries (NaN in the original array, or arms with no zeta
    instrumentation at all) are dropped before taking the median; an
    all-`None` series returns `None` rather than raising."""
    vals = [v for v in zeta_series if v is not None]
    if not vals:
        return None
    return statistics.median(vals)


def load_cells(dataset: str, arm: str, results_dir: str = RESULTS_DIR) -> List[dict]:
    ds_tag = dataset.replace("-", "_")
    cells = []
    for path in sorted(glob.glob(os.path.join(results_dir, f"{ds_tag}_{arm}_g*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data["dataset"] != dataset or data["arm"] != arm:
            continue
        if not _is_t1_cell(data):
            continue
        cells.append(data)
    return cells


def build_arm_cell(dataset: str, arm: str, results_dir: str = RESULTS_DIR) -> Optional[dict]:
    cells = load_cells(dataset, arm, results_dir)
    if not cells:
        return None

    rare = [c["final_last_n_acc"] for c in cells]
    all_acc = [c["final_all_acc"] for c in cells]
    bytes_sent = [c["bytes_sent"] for c in cells]
    seeds = [c["seed"] for c in cells]

    per_seed_median_zeta = [_median_zeta(c["zeta"]) for c in cells]
    zeta_vals = [v for v in per_seed_median_zeta if v is not None]

    return {
        "n_seeds": len(cells),
        "seeds": seeds,
        "rare_acc": {
            "per_seed": rare,
            "mean": statistics.mean(rare),
            "ci_95": list(bootstrap_ci(rare)),
        },
        "all_acc": {
            "per_seed": all_acc,
            "mean": statistics.mean(all_acc),
            "ci_95": list(bootstrap_ci(all_acc)),
        },
        "bytes_sent": {
            "per_seed": bytes_sent,
            "median_gb": statistics.median(bytes_sent) / 1e9,
        },
        # Only populated for the v2 arms (measure_zeta=True in run_one_cell);
        # v1_projected and every external baseline have no zeta
        # instrumentation, so this is None for them by construction, not a
        # missing-data gap.
        "median_zeta": statistics.mean(zeta_vals) if zeta_vals else None,
    }


def build_table(datasets: List[str] = DATASETS, results_dir: str = RESULTS_DIR) -> dict:
    table: Dict[str, dict] = {"datasets": {}, "arms": ALL_ARMS,
                               "v2_frontier_arms": V2_FRONTIER_ARMS,
                               "baseline_arms": BASELINE_ARMS}
    for dataset in datasets:
        table["datasets"][dataset] = {}
        for arm in ALL_ARMS:
            cell = build_arm_cell(dataset, arm, results_dir)
            if cell is not None:
                table["datasets"][dataset][arm] = cell
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    args = ap.parse_args()

    table = build_table(args.datasets, args.results_dir)
    out_path = os.path.join(args.results_dir, "frontier_table.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)

    for dataset in args.datasets:
        n_arms = len(table["datasets"].get(dataset, {}))
        print(f"{dataset}: {n_arms}/{len(ALL_ARMS)} arms with data")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
