"""Stage C Master Report figures beyond the frontier plot
(`scripts/render_v2_frontier.py` already covers the v2-vs-14-baselines
frontier). Generates, from the real persisted per-cell JSONs in
`results/v2_frontier/`, matching this project's own established bar-chart
style (`scripts/render_ablation.py`):

1. `classic_fl_comparison.png` — FedGRAA-v2 vs. the 5 classic FL optimizers
   (§4 of the T7 master plan), rare-class recall, both datasets.
2. `overhead_ratio.png` — communication-overhead ratio vs. CELM/SpectralFed
   at each refresh interval `m`, paired with the rare-class-recall it buys.
3. `t7_attacker_weight_trajectory.png` — the mechanism-specific attack's
   measured up-weighting effect over all 200 rounds, both datasets, both
   attacker fractions, against the uniform-share reference line.
4. `t7_defense_asr.png` — attack success rate (1 - detection recall) for
   the SecFedNIDS/WeiDetect defense cross-product, both datasets.

Usage::

    python scripts/render_stage_c_figures.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = "results/v2_frontier"
DATASETS = ["CIC-IDS-2017", "UNSW-NB15"]


def _require_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for scripts/render_stage_c_figures.py.\n"
            "Install it with:  pip install matplotlib"
        ) from exc
    return plt


# --------------------------------------------------------------------------- #
# 1. Classic-FL comparison
# --------------------------------------------------------------------------- #
CLASSIC_ARM_LABELS = {
    "v2_m1": "FedGRAA-v2 (m=1)",
    "v2_m3": "FedGRAA-v2 (m=3)",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fedopt": "FedOpt",
    "scaffold": "SCAFFOLD",
    "fednova": "FedNova",
}
CLASSIC_ARM_ORDER = ["v2_m1", "v2_m3", "fedavg", "fedprox", "fedopt", "scaffold", "fednova"]


def render_classic_fl_comparison(frontier_table: dict, save_path: str):
    plt = _require_mpl()
    import numpy as np

    datasets = list(frontier_table["datasets"].keys())
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(CLASSIC_ARM_ORDER))
    width = 0.35
    colors = ["C0", "C1"]

    for i, dataset in enumerate(datasets):
        cells = frontier_table["datasets"][dataset]
        means, lo_err, hi_err = [], [], []
        for arm in CLASSIC_ARM_ORDER:
            cell = cells.get(arm)
            if cell is None:
                means.append(0.0)
                lo_err.append(0.0)
                hi_err.append(0.0)
                continue
            mean = cell["rare_acc"]["mean"]
            lo, hi = cell["rare_acc"]["ci_95"]
            means.append(mean * 100)
            lo_err.append(max(0.0, (mean - lo) * 100))
            hi_err.append(max(0.0, (hi - mean) * 100))
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width=width, label=dataset, color=colors[i % len(colors)],
               yerr=[lo_err, hi_err], capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([CLASSIC_ARM_LABELS[a] for a in CLASSIC_ARM_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Rare-class recall (%)")
    ax.set_title("FedGRAA-v2 vs. classic FL optimizers (mean over 10 seeds, 95% CI)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2. Overhead ratio
# --------------------------------------------------------------------------- #
def render_overhead_ratio(overhead_table: dict, save_path: str):
    plt = _require_mpl()
    import numpy as np

    datasets = list(overhead_table["datasets"].keys())
    arms = overhead_table["v2_arms"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 5), squeeze=False)
    axes = axes[0]

    for ax, dataset in zip(axes, datasets):
        per_arm = overhead_table["datasets"][dataset]
        ratios = []
        recalls = []
        labels = []
        for arm in arms:
            row = per_arm.get(arm, {}).get("celm")
            if row is None:
                continue
            ratios.append(row["byte_ratio"])
            recalls.append(row["v2_rare_acc_mean"] * 100)
            labels.append(arm.replace("v2_m", "m="))

        x = np.arange(len(labels))
        bars = ax.bar(x, ratios, color="C0", alpha=0.85, label="Byte ratio vs. CELM (left axis)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Communication overhead (×, vs. CELM)")
        ax.set_title(dataset)
        for xi, r in zip(x, ratios):
            ax.annotate(f"{r:.2f}×", (xi, r), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=8)

        ax2 = ax.twinx()
        ax2.plot(x, recalls, color="C3", marker="o", linewidth=2, label="Rare-class recall (right axis)")
        ax2.set_ylabel("Rare-class recall (%)", color="C3")
        ax2.tick_params(axis="y", labelcolor="C3")

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Is exact worth it? Communication overhead vs. rare-class recall, by refresh interval m")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. T7 attacker-weight trajectory
# --------------------------------------------------------------------------- #
def render_attacker_weight_trajectory(results_dir: str, save_path: str):
    plt = _require_mpl()
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), squeeze=False)
    axes = axes[0]

    for ax, dataset in zip(axes, ["CIC_IDS_2017", "UNSW_NB15"]):
        n_clients = 14 if dataset == "CIC_IDS_2017" else 9
        uniform = 100.0 / n_clients
        for frac, color in (("af0.3", "C0"), ("af0.5", "C1")):
            files = sorted(glob.glob(
                os.path.join(results_dir, f"{dataset}_v2_raresig_g1.0_seed*_{frac}.json")))
            if not files:
                continue
            trajectories = []
            for f in files:
                d = json.load(open(f, encoding="utf-8"))
                hist = d.get("attacker_weight_history", [])
                if hist:
                    trajectories.append(hist)
            if not trajectories:
                continue
            min_len = min(len(t) for t in trajectories)
            arr = np.array([t[:min_len] for t in trajectories]) * 100.0
            mean_traj = arr.mean(axis=0)
            rounds = np.arange(1, min_len + 1)
            frac_label = frac.replace("af", "")
            ax.plot(rounds, mean_traj, color=color, linewidth=2,
                    label=f"{int(float(frac_label)*100)}% attackers")

        ax.axhline(uniform, color="0.4", linestyle="--", linewidth=1.5,
                    label=f"Uniform share ({uniform:.1f}%)")
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Attacker's mean aggregation weight (%)")
        ax.set_title(dataset.replace("_", "-"))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("The mechanism-specific attack (D1c): attacker up-weighting vs. round, undefended FedGRAA-v2")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. T7 defense attack-success-rate
# --------------------------------------------------------------------------- #
DEFENSE_ARMS = [
    "secfednids_labelflip", "secfednids_modelpoison", "secfednids_adaptive",
    "weidetect_labelflip", "weidetect_modelpoison", "weidetect_adaptive",
]
DEFENSE_ARM_LABELS = {
    "secfednids_labelflip": "SecFedNIDS\nlabel-flip",
    "secfednids_modelpoison": "SecFedNIDS\nmodel-poison",
    "secfednids_adaptive": "SecFedNIDS\nadaptive",
    "weidetect_labelflip": "WeiDetect\nlabel-flip",
    "weidetect_modelpoison": "WeiDetect\nmodel-poison",
    "weidetect_adaptive": "WeiDetect\nadaptive",
}


def render_t7_defense_asr(results_dir: str, save_path: str):
    plt = _require_mpl()
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(DEFENSE_ARMS))
    width = 0.35
    colors = ["C0", "C1"]

    for i, dataset in enumerate(["CIC_IDS_2017", "UNSW_NB15"]):
        means = []
        for arm in DEFENSE_ARMS:
            files = sorted(glob.glob(
                os.path.join(results_dir, f"{dataset}_{arm}_g1.0_seed*_af0.3.json")))
            vals = []
            for f in files:
                d = json.load(open(f, encoding="utf-8"))
                hist = d.get("attacker_weight_history", [])
                if hist:
                    vals.append(hist[-1])
            means.append(statistics.mean(vals) * 100 if vals else 0.0)
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width=width,
               label=dataset.replace("_", "-"), color=colors[i % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels([DEFENSE_ARM_LABELS[a] for a in DEFENSE_ARMS], fontsize=8)
    ax.set_ylabel("Attack success rate (%, 1 - detection recall)")
    ax.set_title("T7 defense cross-product: attack success rate by (defense, attack) pair, 30% attackers")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    args = ap.parse_args()

    frontier_path = os.path.join(args.results_dir, "frontier_table.json")
    overhead_path = os.path.join(args.results_dir, "overhead_table.json")
    frontier_table = json.load(open(frontier_path, encoding="utf-8"))
    overhead_table = json.load(open(overhead_path, encoding="utf-8"))

    out1 = os.path.join(args.results_dir, "classic_fl_comparison.png")
    render_classic_fl_comparison(frontier_table, out1)
    print(f"Wrote {out1}")

    out2 = os.path.join(args.results_dir, "overhead_ratio.png")
    render_overhead_ratio(overhead_table, out2)
    print(f"Wrote {out2}")

    out3 = os.path.join(args.results_dir, "t7_attacker_weight_trajectory.png")
    render_attacker_weight_trajectory(args.results_dir, out3)
    print(f"Wrote {out3}")

    out4 = os.path.join(args.results_dir, "t7_defense_asr.png")
    render_t7_defense_asr(args.results_dir, out4)
    print(f"Wrote {out4}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
