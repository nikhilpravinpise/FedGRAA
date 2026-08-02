"""Statistical-significance layer for the FedGRAA-v2 frontier
(docs/research_c/STAGE_C_REMAINING_WORK_MASTER_PLAN_2026-08-02.md §3;
Final_Stage_C.md item 8: "≥10 seeds, paired Wilcoxon, Holm-Bonferroni across
the baseline grid").

`scripts/build_v2_frontier_table.py` computes a bootstrap 95% CI per arm but
never tests FedGRAA-v2 against any of the 14 external baselines directly, and
never corrects for running that test 14 times. This script closes that gap:
for every (dataset, v2 arm) pair it runs a paired Wilcoxon signed-rank test
against each of the 14 external baselines on the seed-matched rare-class
recall (``rare_acc``, i.e. ``final_last_n_acc`` -- this project's metric of
record throughout Stage B/C), then applies a Holm-Bonferroni correction.

**Judgment call (JC-T7-STATS-1), disclosed per this project's own JC-table
convention rather than made silently.** The plan text (§3 step 1) proposes a
family size of "14 x 2 x 5 = 140" comparisons and leaves the correction scope
("per-dataset ... a judgment call to disclose") open. Two corrections to that
arithmetic, made here:

1. The v2 frontier only has **4** v2 arms with data (``v2_m1``, ``v2_m2``,
   ``v2_m3``, ``v2_m5``) -- ``v2_m5`` is included but there is no ``v2_m4``:
   ``run_v2_frontier_sweep.py``'s own ``FRONTIER_MS = [1, 2, 3, 5]`` never
   defines one, by the T1/T2 campaign's original design, not an omission
   introduced here. So the real comparison count is 14 x 2 x 4 = **112**, not
   140.
2. The correction scope chosen is **per (dataset, v2 arm)** -- a family of 14
   comparisons (one v2 arm against each of the 14 baselines, within one
   dataset) -- rather than one global family of 112. This mirrors
   ``build_ablation_table.py``'s own established precedent exactly: that
   script corrects "across the 5 comparisons within each (dataset, gamma)",
   i.e. one focal arm against its multiple references, scoped per dataset.
   Using the same unit of correction here keeps the two significance layers
   in this codebase consistent with each other rather than introducing a
   second, differently-scoped convention.

Usage::

    python scripts/build_v2_significance_table.py
    python scripts/build_v2_significance_table.py --results-dir results/v2_frontier
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedgraa.stats import holm_bonferroni, wilcoxon_signed_rank  # noqa: E402
from scripts.build_v2_frontier_table import (  # noqa: E402
    BASELINE_ARMS,
    DATASETS,
    RESULTS_DIR,
    V2_FRONTIER_ARMS,
    build_table,
)

ALPHA = 0.05
METRIC = "rare_acc"


def paired_comparison(cell_a: dict, cell_b: dict, metric: str = METRIC) -> Optional[dict]:
    """Paired Wilcoxon signed-rank test between two frontier-table cells,
    matched by the intersection of their seed sets (stable, sorted). Mirrors
    ``build_ablation_table.py::paired_comparison``'s seed-matching contract
    exactly, adapted to this table's nested ``{metric: {seeds, per_seed}}``
    schema. Returns ``None`` if the two cells share no common seed."""
    seeds_a = dict(zip(cell_a["seeds"], cell_a[metric]["per_seed"]))
    seeds_b = dict(zip(cell_b["seeds"], cell_b[metric]["per_seed"]))
    common = sorted(set(seeds_a) & set(seeds_b))
    if not common:
        return None
    a = [seeds_a[s] for s in common]
    b = [seeds_b[s] for s in common]
    w, p_w = wilcoxon_signed_rank(a, b)
    return {
        "n_paired_seeds": len(common),
        "seeds": common,
        "v2_mean": float(sum(a) / len(a)),
        "baseline_mean": float(sum(b) / len(b)),
        "mean_diff": float(sum(a) / len(a) - sum(b) / len(b)),
        "wilcoxon_statistic": w,
        "p_wilcoxon": p_w,
    }


def build_significance_table(
    frontier_table: dict,
    datasets: Sequence[str] = DATASETS,
    v2_arms: Sequence[str] = V2_FRONTIER_ARMS,
    baseline_arms: Sequence[str] = BASELINE_ARMS,
    alpha: float = ALPHA,
) -> dict:
    out: Dict[str, dict] = {
        "metric": METRIC,
        "alpha": alpha,
        "correction_scope": "per (dataset, v2_arm): family of len(baseline_arms) comparisons",
        "v2_arms": [a for a in v2_arms if a != "v1_projected"],
        "baseline_arms": list(baseline_arms),
        "datasets": {},
    }
    focal_arms = out["v2_arms"]

    n_total = 0
    for dataset in datasets:
        ds_cells = frontier_table["datasets"].get(dataset, {})
        out["datasets"][dataset] = {}
        for v2_arm in focal_arms:
            v2_cell = ds_cells.get(v2_arm)
            per_baseline: Dict[str, Optional[dict]] = {}
            if v2_cell is not None:
                for baseline in baseline_arms:
                    b_cell = ds_cells.get(baseline)
                    per_baseline[baseline] = (
                        paired_comparison(v2_cell, b_cell) if b_cell is not None else None
                    )
            available = [name for name, res in per_baseline.items() if res is not None]
            if available:
                p_vals = [per_baseline[name]["p_wilcoxon"] for name in available]
                adjusted = holm_bonferroni(p_vals)
                for name, p_holm in zip(available, adjusted):
                    per_baseline[name]["p_holm"] = p_holm
                    per_baseline[name]["significant"] = bool(p_holm < alpha)
                    n_total += 1
            out["datasets"][dataset][v2_arm] = per_baseline

    out["n_comparisons"] = n_total
    return out


def render_markdown(sig_table: dict) -> str:
    lines = [
        "# FedGRAA-v2 frontier significance table",
        "",
        f"Paired Wilcoxon signed-rank test on `{sig_table['metric']}` "
        f"(rare-class recall), 10 matched seeds per comparison, "
        f"Holm-Bonferroni corrected within each (dataset, v2 arm) family "
        f"of {len(sig_table['baseline_arms'])} baselines. "
        f"alpha = {sig_table['alpha']}. {sig_table['n_comparisons']} total comparisons.",
        "",
    ]
    for dataset, arms in sig_table["datasets"].items():
        lines.append(f"## {dataset}")
        lines.append("")
        for v2_arm, per_baseline in arms.items():
            lines.append(f"### {v2_arm}")
            lines.append("")
            lines.append("| baseline | v2 mean | baseline mean | diff | p (Wilcoxon) | p (Holm) | significant |")
            lines.append("|---|---|---|---|---|---|---|")
            for baseline, res in sorted(per_baseline.items()):
                if res is None:
                    lines.append(f"| {baseline} | -- | -- | -- | -- | -- | no data |")
                    continue
                lines.append(
                    f"| {baseline} | {res['v2_mean']:.4f} | {res['baseline_mean']:.4f} | "
                    f"{res['mean_diff']:+.4f} | {res['p_wilcoxon']:.4g} | "
                    f"{res['p_holm']:.4g} | {'yes' if res['significant'] else 'no'} |"
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

    sig_table = build_significance_table(frontier_table, datasets=args.datasets)

    json_path = os.path.join(args.results_dir, "significance_table.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sig_table, f, indent=2)

    md_path = os.path.join(args.results_dir, "significance_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(sig_table))

    print(f"{sig_table['n_comparisons']} significant-tested comparisons across "
          f"{len(args.datasets)} datasets x {len(sig_table['v2_arms'])} v2 arms x "
          f"{len(sig_table['baseline_arms'])} baselines")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
