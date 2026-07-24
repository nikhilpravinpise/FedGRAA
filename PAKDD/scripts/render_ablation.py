"""Render the consolidated ablation table as markdown + a figure (stage_b_plan.md T6/§6.4).

Reads ``results/ablation/{DS}_ablation_table.json`` (built by
``scripts/build_ablation_table.py``) and emits:

* ``results/ablation/{DS}_ablation_table.md`` -- a publication-ready
  markdown table: rows = arms/baselines, columns = gamma, cells =
  "rare-acc mean [95% CI] . all-acc . blowup", with significance markers on
  each reference arm keyed to its persisted, Holm-corrected exact p-value
  against the "both" arm (never a bare star with no traceable number).
* ``results/ablation/{DS}_ablation.png`` -- a grouped bar chart of
  rare-class accuracy by arm x gamma, with 95% CI error bars.

Follows ``fedgraa/plot.py``'s optional-matplotlib idiom (``_require_mpl``,
``matplotlib.use("Agg")``) rather than importing that module directly,
since ``fedgraa/plot.py`` is reserved for the strict-reproduction Figure
1/2/3 plots (stage_b_plan.md §6.4).

Usage::

    python scripts/render_ablation.py --dataset CIC-IDS-2017
    python scripts/render_ablation.py --dataset UNSW-NB15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ABLATION_DIR = "results/ablation"

ARM_LABELS = {
    "fedavg": "FedAvg (baseline)",
    "static": "Static Reparam. (baseline)",
    "literal": "Literal (η′=0.1, no proj.)",
    "projection_alone": "Projection-alone (η′=0.1)",
    "eta_alone": "η′-alone (η′=0.01, no proj.)",
    "both": "Both (proj. + η′=0.01) — the fix",
}
ROW_ORDER = ["fedavg", "static", "literal", "projection_alone", "eta_alone", "both"]
BASELINE_NAMES = {"fedavg", "static"}


def _require_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for scripts/render_ablation.py.\n"
            "Install it with:  pip install matplotlib"
        ) from exc
    return plt


# --------------------------------------------------------------------------- #
# Loading + cell lookup
# --------------------------------------------------------------------------- #
def load_table(dataset: str, ablation_dir: str = ABLATION_DIR) -> dict:
    dataset_tag = dataset.replace("-", "_")
    path = os.path.join(ablation_dir, f"{dataset_tag}_ablation_table.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _row_cell(table: dict, row: str, gamma_key: str) -> Optional[dict]:
    if row in BASELINE_NAMES:
        return table["baselines"].get(row)
    return table["arms"].get(row, {}).get(gamma_key)


def _significance_marker(p_holm: Optional[float]) -> str:
    if p_holm is None:
        return ""
    if p_holm < 0.001:
        return "***"
    if p_holm < 0.01:
        return "**"
    if p_holm < 0.05:
        return "*"
    return ""


def _fmt_pct(x: Optional[float]) -> str:
    return f"{x * 100:.1f}%" if x is not None else "—"


def _fmt_ci(cell: Optional[dict]) -> str:
    if cell is None or cell.get("ci_95") is None:
        return "—"
    lo, hi = cell["ci_95"]
    return f"[{lo * 100:.1f}, {hi * 100:.1f}]"


def _fmt_num(x: float) -> str:
    """Compact fixed/scientific rendering for a blowup ratio."""
    return f"{x:.3g}" if (abs(x) >= 1000 or 0 < abs(x) < 0.01) else f"{x:.2f}"


def _fmt_blowup(cell: dict) -> str:
    """Blowup is a heavy-tailed ratio, so report median (max), not the mean.

    On CIC-IDS-2017's literal arm at gamma=0 a single seed blows up by
    1.86e6x while the other nine sit near 3x: the mean (185764x) describes
    no run that happened, the median (3.70x) describes most of them, and the
    max is the striking part worth surfacing explicitly rather than
    averaging away. ``max`` is shown only when it differs materially from
    the median.
    """
    med, mx = cell.get("median"), cell.get("max")
    if med is None:
        return "—"
    if mx is None or mx <= med * 1.05:
        return f"{_fmt_num(med)}× (med)"
    return f"{_fmt_num(med)}× med, {_fmt_num(mx)}× max"


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #
def render_markdown(table: dict) -> str:
    dataset = table["dataset"]
    gammas = table["gammas"]
    gamma_keys = [f"g{g}" for g in gammas]
    comparisons = table.get("comparisons", {})

    lines = [
        f"# {dataset} — Stage B ablation table",
        "",
        f"Rounds = {table['rounds']}. Metric of record = rare-class accuracy "
        "(mean per-class recall over the protected classes), reported as "
        "mean [95% bootstrap CI] over the per-seed values persisted in this "
        "same JSON. All-class accuracy and composite-loss blowup are shown "
        "as secondary/diagnostic columns per cell.",
        "",
        "**Blowup is reported as median (and max), not mean**, because it is "
        "a heavy-tailed ratio rather than a bounded score: a single diverging "
        "seed can exceed the other nine by six orders of magnitude, making a "
        "mean describe no run that actually occurred. Per-seed blowup arrays "
        "and the mean are both retained in the JSON.",
        "",
        "Significance markers on a reference-arm cell denote the paired, "
        "Holm-Bonferroni-corrected comparison of **both** vs. that arm at "
        "the same γ (`* p_holm<.05  ** p_holm<.01  *** p_holm<.001`); exact "
        "p-values and effect sizes are in the JSON's `comparisons` block, "
        "never only a star here.",
        "",
    ]

    header = "| Arm | " + " | ".join(f"γ={g}" for g in gammas) + " |"
    sep = "|---" * (len(gammas) + 1) + "|"
    lines += [header, sep]

    for row in ROW_ORDER:
        cells = []
        for gamma, gkey in zip(gammas, gamma_keys):
            cell = _row_cell(table, row, gkey)
            if cell is None:
                cells.append("(no data yet)")
                continue
            rare = cell.get("rare_acc")
            all_acc = cell.get("all_acc")
            blowup = cell.get("blowup")
            if rare is None or rare.get("mean") is None:
                cells.append("(no data yet)")
                continue

            marker = ""
            if row not in BASELINE_NAMES and row != "both":
                comp = comparisons.get(gkey, {}).get(row)
                if comp is not None:
                    marker = _significance_marker(comp.get("p_holm"))

            parts = [f"{_fmt_pct(rare['mean'])} {_fmt_ci(rare)}{marker}"]
            if all_acc is not None and all_acc.get("mean") is not None:
                parts.append(f"all {_fmt_pct(all_acc['mean'])}")
            if blowup is not None and blowup.get("median") is not None:
                parts.append(f"blowup {_fmt_blowup(blowup)}")
            elif blowup is not None and blowup.get("mean") is not None:
                # pre-median tables (built before median/min/max were added)
                parts.append(f"blowup {blowup['mean']:.2f}×")
            cells.append(" · ".join(parts))
        lines.append(f"| {ARM_LABELS[row]} | " + " | ".join(cells) + " |")

    lines += ["", "## Pre-registered pairwise comparisons (both vs. reference arm)", ""]
    for gamma, gkey in zip(gammas, gamma_keys):
        lines.append(f"**γ={gamma}**")
        lines.append("")
        lines.append("| vs. | p (paired t) | p (Wilcoxon) | Cohen's d | p (Holm) |")
        lines.append("|---|---|---|---|---|")
        gamma_comps = comparisons.get(gkey) or {}
        for ref_name, comp in gamma_comps.items():
            if comp is None:
                lines.append(f"| {ARM_LABELS.get(ref_name, ref_name)} | (no data yet) | | | |")
                continue
            lines.append(
                f"| {ARM_LABELS.get(ref_name, ref_name)} "
                f"| {comp['p_ttest']:.4g} | {comp['p_wilcoxon']:.4g} "
                f"| {comp['cohens_d']:.3g} | {comp['p_holm']:.4g} |"
            )
        lines.append("")

    lines += ["## 2x2 factorial read (rare-class accuracy)", ""]
    lines.append("| γ | Projection main effect | η′ main effect | Interaction |")
    lines.append("|---|---|---|---|")
    factorial = table.get("factorial", {})
    for gamma, gkey in zip(gammas, gamma_keys):
        fx = factorial.get(gkey)
        if fx is None:
            lines.append(f"| {gamma} | (no data yet) | | |")
            continue

        def _fx_str(entry):
            lo, hi = entry["ci_95"]
            return f"{entry['mean'] * 100:+.1f}pp [{lo * 100:.1f}, {hi * 100:.1f}]"

        lines.append(
            f"| {gamma} | {_fx_str(fx['projection_main_effect'])} "
            f"| {_fx_str(fx['eta_main_effect'])} | {_fx_str(fx['interaction'])} |"
        )

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Figure rendering
# --------------------------------------------------------------------------- #
def render_figure(table: dict, save_path: Optional[str] = None):
    plt = _require_mpl()
    import numpy as np

    dataset = table["dataset"]
    gammas = table["gammas"]
    gamma_keys = [f"g{g}" for g in gammas]

    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(ROW_ORDER) * len(gammas)), 5.5))
    n_groups = len(gammas)
    width = 0.8 / n_groups
    x = np.arange(len(ROW_ORDER))
    colors = [f"C{i}" for i in range(n_groups)]

    for i, (gamma, gkey) in enumerate(zip(gammas, gamma_keys)):
        means, err_lo, err_hi = [], [], []
        for row in ROW_ORDER:
            cell = _row_cell(table, row, gkey)
            rare = cell.get("rare_acc") if cell else None
            if rare is None or rare.get("mean") is None:
                means.append(0.0)
                err_lo.append(0.0)
                err_hi.append(0.0)
                continue
            mean = rare["mean"]
            lo, hi = rare["ci_95"]
            means.append(mean * 100)
            err_lo.append(max(0.0, (mean - lo) * 100))
            err_hi.append(max(0.0, (hi - mean) * 100))
        offset = (i - (n_groups - 1) / 2.0) * width
        ax.bar(x + offset, means, width=width, label=f"γ={gamma}", color=colors[i],
               yerr=[err_lo, err_hi], capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS[r] for r in ROW_ORDER], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Rare-class accuracy (%)")
    ax.set_title(f"{dataset} — Stage B ablation: rare-class accuracy by arm × γ (95% CI)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="CIC-IDS-2017")
    ap.add_argument("--ablation-dir", default=ABLATION_DIR)
    args = ap.parse_args()

    table = load_table(args.dataset, args.ablation_dir)
    dataset_tag = args.dataset.replace("-", "_")

    md_path = os.path.join(args.ablation_dir, f"{dataset_tag}_ablation_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(table))
    print(f"Wrote {md_path}")

    png_path = os.path.join(args.ablation_dir, f"{dataset_tag}_ablation.png")
    render_figure(table, save_path=png_path)
    print(f"Wrote {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
