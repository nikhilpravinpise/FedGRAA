"""Communication-overhead write-up for the FedGRAA-v2 frontier
(docs/research_c/STAGE_C_REMAINING_WORK_MASTER_PLAN_2026-08-02.md §5;
Final_Stage_C.md item 7: "report the ~6x overhead against CELM/SpectralFed's
near-zero-cost approximation, making 'is exact worth 6x?' an experiment
rather than a caveat").

``run_one_cell`` already logs ``bytes_sent`` per cell (consumed by
``build_v2_frontier_table.py`` into each arm's ``bytes_sent.median_gb``).
This script turns that into the specific comparison Final_Stage_C.md item 7
asks for: per dataset, per v2 arm, the communication-cost *ratio* against
CELM and SpectralFed (T3's two "near-zero-cost approximation" baselines),
paired with the rare-class-recall *delta* already computed by
``build_v2_frontier_table.py``'s means -- so "is exact worth N times the
bytes?" has both axes on one row, not split across two tables the reader has
to cross-reference themselves.

Usage::

    python scripts/build_overhead_table.py
    python scripts/build_overhead_table.py --results-dir results/v2_frontier
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_v2_frontier_table import (  # noqa: E402
    DATASETS,
    RESULTS_DIR,
    V2_FRONTIER_ARMS,
    build_table,
)

APPROX_BASELINES = ["celm", "spectralfed"]


def overhead_row(v2_cell: dict, approx_cell: dict) -> dict:
    """One (v2 arm, approx baseline) comparison row: byte ratio and the
    accompanying rare-class-recall delta, both derived from data already on
    disk -- no new compute."""
    v2_gb = v2_cell["bytes_sent"]["median_gb"]
    approx_gb = approx_cell["bytes_sent"]["median_gb"]
    byte_ratio = v2_gb / approx_gb if approx_gb > 0 else float("inf")
    return {
        "v2_bytes_gb": v2_gb,
        "approx_bytes_gb": approx_gb,
        "byte_ratio": byte_ratio,
        "v2_rare_acc_mean": v2_cell["rare_acc"]["mean"],
        "approx_rare_acc_mean": approx_cell["rare_acc"]["mean"],
        "rare_acc_delta": v2_cell["rare_acc"]["mean"] - approx_cell["rare_acc"]["mean"],
    }


def build_overhead_table(
    frontier_table: dict,
    datasets: Sequence[str] = DATASETS,
    v2_arms: Sequence[str] = V2_FRONTIER_ARMS,
    approx_baselines: Sequence[str] = APPROX_BASELINES,
) -> dict:
    out: Dict[str, dict] = {
        "approx_baselines": list(approx_baselines),
        "v2_arms": [a for a in v2_arms if a != "v1_projected"],
        "datasets": {},
    }
    focal_arms = out["v2_arms"]
    for dataset in datasets:
        ds_cells = frontier_table["datasets"].get(dataset, {})
        out["datasets"][dataset] = {}
        for v2_arm in focal_arms:
            v2_cell = ds_cells.get(v2_arm)
            per_approx: Dict[str, Optional[dict]] = {}
            if v2_cell is not None:
                for approx in approx_baselines:
                    approx_cell = ds_cells.get(approx)
                    per_approx[approx] = (
                        overhead_row(v2_cell, approx_cell) if approx_cell is not None else None
                    )
            out["datasets"][dataset][v2_arm] = per_approx
    return out


def render_markdown(overhead_table: dict) -> str:
    lines = [
        "# FedGRAA-v2 communication-overhead table",
        "",
        "Per-cell median `bytes_sent` (SecAgg-instrumented, T-12's byte-"
        "exactness gate) for each FedGRAA-v2 frontier arm, expressed as a "
        "ratio against the two T3 near-zero-cost approximation baselines "
        f"({', '.join(overhead_table['approx_baselines'])}), paired with the "
        "rare-class-recall delta so the \"is exact worth N times the bytes?\" "
        "question has both axes on one row.",
        "",
    ]
    for dataset, arms in overhead_table["datasets"].items():
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| v2 arm | vs. | v2 GB | approx GB | byte ratio | v2 rare-acc | approx rare-acc | rare-acc delta |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for v2_arm, per_approx in arms.items():
            for approx, row in sorted(per_approx.items()):
                if row is None:
                    lines.append(f"| {v2_arm} | {approx} | -- | -- | -- | -- | -- | no data |")
                    continue
                lines.append(
                    f"| {v2_arm} | {approx} | {row['v2_bytes_gb']:.4f} | "
                    f"{row['approx_bytes_gb']:.4f} | {row['byte_ratio']:.2f}x | "
                    f"{row['v2_rare_acc_mean']:.4f} | {row['approx_rare_acc_mean']:.4f} | "
                    f"{row['rare_acc_delta']:+.4f} |"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--frontier-table", default=None,
                     help="Path to an existing frontier_table.json; rebuilt from --results-dir if omitted.")
    args = ap.parse_args()

    if args.frontier_table:
        with open(args.frontier_table, encoding="utf-8") as f:
            frontier_table = json.load(f)
    else:
        frontier_table = build_table(args.datasets, args.results_dir)

    overhead_table = build_overhead_table(frontier_table, datasets=args.datasets)

    json_path = os.path.join(args.results_dir, "overhead_table.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(overhead_table, f, indent=2)

    md_path = os.path.join(args.results_dir, "overhead_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(overhead_table))

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
