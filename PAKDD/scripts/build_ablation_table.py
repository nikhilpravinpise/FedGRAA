"""Consolidated ablation-table builder (stage_b_plan.md T5/§6.3).

Reads the per-seed diagnostic JSONs for all four factorial arms --
**literal**, **projection-alone**, **eta-alone (canonical)**, **both** --
plus the FedAvg/Static baseline seed-sweep JSONs, and emits
``results/ablation/{DS}_ablation_table.json``: per (arm, gamma) cell the
per-seed arrays + mean + bootstrap 95% CI for rare-class accuracy,
all-class accuracy, and composite-loss blowup; per (comparison, gamma) the
pre-registered paired significance tests (paired *t*, Wilcoxon, Cohen's *d*,
Holm-Bonferroni-adjusted p) from ``fedgraa/stats.py``; and the 2x2
factorial main-effect/interaction read on the rare-class-accuracy metric of
record (stage_b_plan.md §3/§6).

This mirrors ``scripts/seed_sweep_combined_fix.py``'s JSON-reading
conventions (same per-seed diagnostic file schema, same
``diagnosis.final_last_n_acc`` / ``final_all_acc`` /
``composite_loss_blowup_ratio`` field names) but is written fresh rather
than extended from that script, because ``seed_sweep_combined_fix.py::
aggregate`` is hard-wired to a single arm ("both") and has no CI/p-value
machinery -- Stage A's committed band file and Stage A's completeness check
both depend on that script staying exactly as-is (stage_b_plan.md §6.3).

Usage::

    python scripts/build_ablation_table.py --dataset CIC-IDS-2017
    python scripts/build_ablation_table.py --dataset UNSW-NB15
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedgraa.stats import (  # noqa: E402
    bootstrap_ci,
    cohens_d_paired,
    holm_bonferroni,
    paired_ttest,
    wilcoxon_signed_rank,
)
from scripts.run_ablation_seed_sweep import ALL_ARM_NAMES, output_path  # noqa: E402

DIAG_DIR = "results/diagnostics"
ABLATION_DIR = "results/ablation"
GAMMAS = [0.0, 0.5, 1.0]
ROUNDS = 60

# The 5 pre-registered pairwise comparisons (stage_b_plan.md §3): "both" vs
# every other reference arm, per (dataset, gamma).
COMPARISON_ARMS = ["literal", "projection_alone", "eta_alone", "fedavg", "static"]

DIAGNOSIS_FIELD = {
    "rare_acc": "final_last_n_acc",
    "all_acc": "final_all_acc",
    "blowup": "composite_loss_blowup_ratio",
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_arm_per_seed(
    dataset: str, arm: str, seeds: Sequence[int], rounds: int = ROUNDS, diag_dir: str = DIAG_DIR,
) -> Dict[int, dict]:
    """Load every available per-seed diagnostic JSON for one (dataset, arm).
    Missing seeds are silently skipped (the cell is built from whatever is
    on disk) -- ``check_stage_b_complete.py`` is what enforces "all 10"."""
    per_seed = {}
    for seed in seeds:
        path = output_path(dataset, arm, seed, rounds, diag_dir)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                per_seed[seed] = json.load(f)
    return per_seed


def load_baseline_per_seed(dataset: str, method: str, ablation_dir: str = ABLATION_DIR) -> Optional[dict]:
    """Load a FedAvg/Static seed-sweep summary JSON (as emitted by
    ``run_seed_sweep.py --out results/ablation/seed_sweep_{DS}_{method}_r60.json``)."""
    dataset_tag = dataset.replace("-", "_")
    path = os.path.join(ablation_dir, f"seed_sweep_{dataset_tag}_{method}_r60.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Cell construction
# --------------------------------------------------------------------------- #
def _cell(values: Sequence[float], seeds: Sequence[int]) -> dict:
    values = list(values)
    lo, hi = bootstrap_ci(values) if values else (None, None)
    return {
        "seeds": list(seeds),
        "per_seed": values,
        "mean": float(sum(values) / len(values)) if values else None,
        "ci_95": [lo, hi] if values else None,
    }


def arm_gamma_cell(per_seed: Dict[int, dict], gamma: float) -> Dict[str, dict]:
    """One (arm, gamma) row: rare-acc / all-acc / blowup cells, each with
    per-seed array + mean + bootstrap CI."""
    key = f"g{gamma}"
    seeds = sorted(s for s in per_seed if key in per_seed[s])
    out = {}
    for metric, field in DIAGNOSIS_FIELD.items():
        values = [per_seed[s][key]["diagnosis"][field] for s in seeds]
        out[metric] = _cell(values, seeds)
    return out


def build_arm_table(dataset: str, arm: str, seeds: Sequence[int], rounds: int = ROUNDS,
                     diag_dir: str = DIAG_DIR, gammas: Sequence[float] = GAMMAS) -> Dict[str, dict]:
    per_seed = load_arm_per_seed(dataset, arm, seeds, rounds, diag_dir)
    return {f"g{gamma}": arm_gamma_cell(per_seed, gamma) for gamma in gammas}


def build_baseline_table(dataset: str, method: str, ablation_dir: str = ABLATION_DIR) -> Optional[dict]:
    """FedAvg/Static are gamma-independent (stage_b_plan.md §3) -- a single
    cell, shown against every gamma column by the renderer/comparisons."""
    data = load_baseline_per_seed(dataset, method, ablation_dir)
    if data is None:
        return None
    seeds = data["seeds"]
    return {
        "rare_acc": _cell(data["last_n_acc_per_seed"], seeds),
        "all_acc": _cell(data["all_acc_per_seed"], seeds),
    }


# --------------------------------------------------------------------------- #
# Pairwise significance comparisons
# --------------------------------------------------------------------------- #
def paired_comparison(cell_a: dict, cell_b: dict) -> Optional[dict]:
    """Paired significance test between two cells' rare-acc ``per_seed``
    arrays, matched by the intersection of their seed sets (stable, sorted).
    Returns ``None`` if the two cells share no common seed."""
    seeds_a = {s: v for s, v in zip(cell_a["seeds"], cell_a["per_seed"])}
    seeds_b = {s: v for s, v in zip(cell_b["seeds"], cell_b["per_seed"])}
    common = sorted(set(seeds_a) & set(seeds_b))
    if not common:
        return None
    a = [seeds_a[s] for s in common]
    b = [seeds_b[s] for s in common]
    t, p_t = paired_ttest(a, b)
    w, p_w = wilcoxon_signed_rank(a, b)
    d = cohens_d_paired(a, b)
    return {
        "n_paired_seeds": len(common),
        "seeds": common,
        "t_statistic": t,
        "p_ttest": p_t,
        "wilcoxon_statistic": w,
        "p_wilcoxon": p_w,
        "cohens_d": d,
    }


def build_comparisons_for_gamma(
    both_cell: dict,
    reference_cells: Dict[str, Optional[dict]],
) -> Dict[str, Optional[dict]]:
    """The 5 pre-registered "both vs X" comparisons for one gamma, with
    Holm-Bonferroni correction applied jointly across all 5 (stage_b_plan.md
    §3: "Holm-Bonferroni correction across the 5 comparisons within each
    (dataset, gamma)"). Comparisons whose reference arm has no data yet are
    ``None`` and excluded from the Holm family (so partial data never biases
    the correction factor for the comparisons that do exist)."""
    raw = {}
    for name in COMPARISON_ARMS:
        ref_cell = reference_cells.get(name)
        raw[name] = paired_comparison(both_cell, ref_cell) if ref_cell is not None else None

    available = [name for name, res in raw.items() if res is not None]
    if available:
        p_ttest_vals = [raw[name]["p_ttest"] for name in available]
        adjusted = holm_bonferroni(p_ttest_vals)
        for name, p_holm in zip(available, adjusted):
            raw[name]["p_holm"] = p_holm
    return raw


# --------------------------------------------------------------------------- #
# 2x2 factorial main effects / interaction (rare-acc, the metric of record)
# --------------------------------------------------------------------------- #
def factorial_effects_for_gamma(arm_tables: Dict[str, Dict[str, dict]], gamma: float) -> Optional[dict]:
    """Projection main effect, eta' main effect, and their interaction, on
    rare-class accuracy, matched by the intersection of seeds present in
    all 4 arms at this gamma. Each effect is a per-seed contrast (a linear
    combination of the 4 arm cells), summarized with the same bootstrap-CI
    + paired-vs-zero machinery as the pairwise comparisons -- this is a
    marginal-means/difference-in-differences read, not a full repeated-
    measures ANOVA (stage_b_plan.md §3 flags an ANOVA as a "where cell
    variances permit" complement, not a hard requirement)."""
    key = f"g{gamma}"
    per_arm_seed_val = {}
    for arm in ("literal", "projection_alone", "eta_alone", "both"):
        cell = arm_tables[arm][key]["rare_acc"]
        per_arm_seed_val[arm] = dict(zip(cell["seeds"], cell["per_seed"]))

    common = sorted(set.intersection(*(set(d) for d in per_arm_seed_val.values())))
    if not common:
        return None

    lit = [per_arm_seed_val["literal"][s] for s in common]
    proj = [per_arm_seed_val["projection_alone"][s] for s in common]
    eta = [per_arm_seed_val["eta_alone"][s] for s in common]
    both = [per_arm_seed_val["both"][s] for s in common]

    projection_effect = [0.5 * (b + p) - 0.5 * (l + e) for b, p, l, e in zip(both, proj, lit, eta)]
    eta_effect = [0.5 * (b + e) - 0.5 * (l + p) for b, e, l, p in zip(both, eta, lit, proj)]
    interaction = [(b - p) - (e - l) for b, p, e, l in zip(both, proj, eta, lit)]

    def _summarize(per_seed_effect):
        mean = float(sum(per_seed_effect) / len(per_seed_effect))
        lo, hi = bootstrap_ci(per_seed_effect)
        zeros = [0.0] * len(per_seed_effect)
        t, p_t = paired_ttest(per_seed_effect, zeros)
        return {"n_seeds": len(per_seed_effect), "seeds": common, "mean": mean,
                "ci_95": [lo, hi], "p_vs_zero": p_t}

    return {
        "projection_main_effect": _summarize(projection_effect),
        "eta_main_effect": _summarize(eta_effect),
        "interaction": _summarize(interaction),
    }


# --------------------------------------------------------------------------- #
# Top-level build
# --------------------------------------------------------------------------- #
def build_table(
    dataset: str,
    seeds: Sequence[int] = range(10),
    rounds: int = ROUNDS,
    diag_dir: str = DIAG_DIR,
    ablation_dir: str = ABLATION_DIR,
    gammas: Sequence[float] = GAMMAS,
    git_sha: Optional[str] = None,
) -> dict:
    arm_tables = {
        arm: build_arm_table(dataset, arm, seeds, rounds, diag_dir, gammas)
        for arm in ALL_ARM_NAMES
    }
    baselines = {
        method: build_baseline_table(dataset, method, ablation_dir)
        for method in ("fedavg", "static")
    }

    comparisons = {}
    factorial = {}
    for gamma in gammas:
        key = f"g{gamma}"
        both_cell = arm_tables["both"][key]["rare_acc"]
        reference_cells = {
            "literal": arm_tables["literal"][key]["rare_acc"],
            "projection_alone": arm_tables["projection_alone"][key]["rare_acc"],
            "eta_alone": arm_tables["eta_alone"][key]["rare_acc"],
            "fedavg": baselines["fedavg"]["rare_acc"] if baselines["fedavg"] else None,
            "static": baselines["static"]["rare_acc"] if baselines["static"] else None,
        }
        comparisons[key] = build_comparisons_for_gamma(both_cell, reference_cells)
        factorial[key] = factorial_effects_for_gamma(arm_tables, gamma)

    return {
        "dataset": dataset,
        "rounds": rounds,
        "gammas": list(gammas),
        "requested_seeds": list(seeds),
        "git_sha": git_sha,
        "arms": arm_tables,
        "baselines": baselines,
        "comparisons": comparisons,
        "factorial": factorial,
    }


def _git_sha() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).stdout.strip()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="CIC-IDS-2017")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--diag-dir", default=DIAG_DIR)
    ap.add_argument("--ablation-dir", default=ABLATION_DIR)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.makedirs(args.ablation_dir, exist_ok=True)
    table = build_table(
        args.dataset, seeds=args.seeds, rounds=args.rounds,
        diag_dir=args.diag_dir, ablation_dir=args.ablation_dir, git_sha=_git_sha(),
    )

    dataset_tag = args.dataset.replace("-", "_")
    out_path = args.out or os.path.join(args.ablation_dir, f"{dataset_tag}_ablation_table.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)

    n_seeds_found = {
        arm: len(table["arms"][arm]["g0.0"]["rare_acc"]["seeds"]) for arm in ALL_ARM_NAMES
    }
    print(f"Wrote {out_path}")
    print(f"Seeds found per arm (g=0.0): {n_seeds_found}")
    print(f"Baselines present: {[m for m in ('fedavg', 'static') if table['baselines'][m]]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
