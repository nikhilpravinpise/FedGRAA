"""Render the FedGRAA-v2 frontier figure (FEDGRAA_V2_MASTER_PLAN.md Sec 4.2's
"headline figure" -- "that figure is the paper").

Reads `results/v2_frontier/frontier_table.json` (built by
`scripts/build_v2_frontier_table.py`) and emits:

* `results/v2_frontier/frontier_figure.png` -- two panels (one per dataset),
  x = total training traffic (GB, log scale), y = per-minority-class recall
  (mean over >=10 seeds) with 95% bootstrap CI error bars, per Sec 4.2's own
  spec. The five FedGRAA points (v1-projected, v2 at m in {5,3,2,1}) are
  drawn as filled circles connected by a dashed guide line and annotated
  with each v2 point's measured median zeta_t; the 14 external-baseline
  points (T3's contribution-estimation family, T5's adaptive/NIDS-domain
  family) are drawn as unfilled markers, un-connected, since they are not
  points on FedGRAA's own communication-accuracy tradeoff curve -- they are
  separate methods being compared against it, not more samples of it.
* `results/v2_frontier/frontier_table.md` -- the same data as a markdown
  table (mean/CI per arm per dataset), for citation without opening the PNG.

Follows `scripts/render_ablation.py`'s optional-matplotlib idiom
(`_require_mpl`, `matplotlib.use("Agg")`).

Usage::

    python scripts/build_v2_frontier_table.py   # build the table first
    python scripts/render_v2_frontier.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = "results/v2_frontier"

# v1_projected is FedGRAA's own *privacy-violating* reference point -- Sec
# 4.2: "mark it, don't hide it". The v2 arms are FedGRAA-v2 at each refresh
# interval m; smaller m = more exact = more bytes.
V2_LABELS = {
    "v1_projected": "v1-projected (privacy-violating)",
    "v2_m1": "v2, m=1",
    "v2_m2": "v2, m=2",
    "v2_m3": "v2, m=3",
    "v2_m5": "v2, m=5",
}
V2_ORDER = ["v1_projected", "v2_m5", "v2_m3", "v2_m2", "v2_m1"]  # ascending bytes

BASELINE_LABELS = {
    "celm": "CELM",
    "spectralfed": "SpectralFed",
    "coast": "CoAst",
    "fltrust": "FLTrust",
    "raga": "RAGA",
    "fedtruth": "FedTruth",
    "fedmade": "FedMADE",
    "sbfednids": "SBFedNIDS",
    "fedapa": "FedAPA",
    "fedawa": "FedAWA",
    "dapfl": "DA-PFL",
    "fedaware": "FedAWARE",
    "adaaggrl": "AdaAggRL",
    "fedpgo": "FedPGO",
}
# Distinct marker per baseline so 14 unfilled points stay distinguishable
# without relying on color alone (colorblind- and greyscale-print-safe).
BASELINE_MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*", "h", "<", ">", "d", "p", "8"]


def _require_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for scripts/render_v2_frontier.py.\n"
            "Install it with:  pip install matplotlib"
        ) from exc
    return plt


def load_table(results_dir: str = RESULTS_DIR) -> dict:
    path = os.path.join(results_dir, "frontier_table.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def render_figure(table: dict, save_path: Optional[str] = None):
    plt = _require_mpl()

    datasets = list(table["datasets"].keys())
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.5 * len(datasets), 6),
                              squeeze=False)
    axes = axes[0]

    for ax, dataset in zip(axes, datasets):
        cells = table["datasets"][dataset]

        # --- FedGRAA frontier: v1-projected + v2 at each m, connected ------
        xs, ys, lo_err, hi_err = [], [], [], []
        for arm in V2_ORDER:
            cell = cells.get(arm)
            if cell is None:
                continue
            gb = cell["bytes_sent"]["median_gb"]
            mean = cell["rare_acc"]["mean"]
            lo, hi = cell["rare_acc"]["ci_95"]
            xs.append(gb)
            ys.append(mean * 100)
            lo_err.append(max(0.0, (mean - lo) * 100))
            hi_err.append(max(0.0, (hi - mean) * 100))

            zeta = cell.get("median_zeta")
            marker = "s" if arm == "v1_projected" else "o"
            color = "0.35" if arm == "v1_projected" else "C0"
            ax.errorbar(gb, mean * 100, yerr=[[lo_err[-1]], [hi_err[-1]]],
                        fmt=marker, color=color, markersize=9, capsize=4, zorder=5)
            label = V2_LABELS[arm]
            annot = label if zeta is None else f"{label}\nζ̃={zeta:.1e}"
            ax.annotate(annot, (gb, mean * 100), textcoords="offset points",
                        xytext=(8, 6), fontsize=7.5, zorder=6)

        if len(xs) > 1:
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            ax.plot([xs[i] for i in order], [ys[i] for i in order],
                    linestyle="--", color="C0", alpha=0.5, zorder=1,
                    label="FedGRAA-v2 frontier")

        # --- External baselines: unfilled, unconnected, one marker each ----
        for i, (arm, label) in enumerate(BASELINE_LABELS.items()):
            cell = cells.get(arm)
            if cell is None:
                continue
            gb = cell["bytes_sent"]["median_gb"]
            mean = cell["rare_acc"]["mean"]
            lo, hi = cell["rare_acc"]["ci_95"]
            marker = BASELINE_MARKERS[i % len(BASELINE_MARKERS)]
            ax.errorbar(gb, mean * 100,
                        yerr=[[max(0.0, (mean - lo) * 100)], [max(0.0, (hi - mean) * 100)]],
                        fmt=marker, markerfacecolor="none", markeredgecolor="C3",
                        ecolor="C3", markersize=8, capsize=3, alpha=0.85,
                        label=label, zorder=4)

        ax.set_xscale("log")
        ax.set_xlabel("Total training traffic (GB, log scale)")
        ax.set_ylabel("Per-minority-class recall (%)")
        ax.set_title(dataset)
        ax.grid(True, which="both", alpha=0.25)

    # One shared legend (baselines repeat identically across panels).
    handles, labels = axes[-1].get_legend_handles_labels()
    seen = set()
    dedup_handles, dedup_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            dedup_handles.append(h)
            dedup_labels.append(l)
    fig.legend(dedup_handles, dedup_labels, loc="lower center", ncol=6,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("FedGRAA-v2 frontier: communication cost vs. rare-class recall "
                  "(mean over >=10 seeds, 95% bootstrap CI)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def render_markdown(table: dict) -> str:
    lines = [
        "# FedGRAA-v2 frontier table",
        "",
        "Metric = per-minority-class recall (`final_last_n_acc`), mean [95% "
        "bootstrap CI] over the per-seed values persisted in "
        "`results/v2_frontier/*.json`. `x` = median total training traffic "
        "across seeds, in GB. `median_zeta` is the across-seed mean of each "
        "seed's own median-over-rounds `zeta_t` (v2 arms only -- v1-projected "
        "and every external baseline have no zeta instrumentation).",
        "",
    ]
    for dataset, cells in table["datasets"].items():
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| Arm | GB | Rare recall | 95% CI | All-class acc | median ζ̃ | n seeds |")
        lines.append("|---|---|---|---|---|---|---|")
        for arm in table["arms"]:
            cell = cells.get(arm)
            if cell is None:
                continue
            label = V2_LABELS.get(arm, BASELINE_LABELS.get(arm, arm))
            gb = cell["bytes_sent"]["median_gb"]
            rare = cell["rare_acc"]
            all_acc = cell["all_acc"]
            zeta = cell.get("median_zeta")
            zeta_str = f"{zeta:.2e}" if zeta is not None else "—"
            lines.append(
                f"| {label} | {gb:.3f} | {rare['mean']*100:.2f}% "
                f"| [{rare['ci_95'][0]*100:.2f}, {rare['ci_95'][1]*100:.2f}] "
                f"| {all_acc['mean']*100:.2f}% | {zeta_str} | {cell['n_seeds']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    args = ap.parse_args()

    table = load_table(args.results_dir)

    md_path = os.path.join(args.results_dir, "frontier_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(table))
    print(f"Wrote {md_path}")

    png_path = os.path.join(args.results_dir, "frontier_figure.png")
    render_figure(table, save_path=png_path)
    print(f"Wrote {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
