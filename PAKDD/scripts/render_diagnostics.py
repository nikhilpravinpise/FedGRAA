"""Render plots + a markdown summary table from a diagnose_fedgraa.py JSON file.

Usage::

    python scripts/render_diagnostics.py \\
        --in results/diagnostics/CIC_IDS_2017_literal_literal_r60.json \\
        --out-prefix results/figures/cic_literal
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.diagnose_fedgraa import dict_to_history  # noqa: E402
from fedgraa.plot import plot_composite_loss, plot_weight_evolution  # noqa: E402


_GAMMA_LABEL = {
    "g0.0": "FedGRAA (γ=0)",
    "g0.5": "FedGRAA (γ=0.5)",
    "g1.0": "FedGRAA (γ=1)",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    with open(args.in_path, encoding="utf-8") as f:
        raw = json.load(f)

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    histories = {
        _GAMMA_LABEL.get(key, key): dict_to_history(entry["history"])
        for key, entry in raw.items()
    }

    plot_composite_loss(
        histories,
        title="Composite loss L'(w^t) — Lemma 1 check",
        save_path=f"{args.out_prefix}_composite_loss.png",
    )
    print(f"Saved {args.out_prefix}_composite_loss.png")

    for key, entry in raw.items():
        h = dict_to_history(entry["history"])
        label = _GAMMA_LABEL.get(key, key)
        safe = key.replace(".", "p")
        plot_weight_evolution(
            h,
            title=f"Weight evolution — {label}",
            save_path=f"{args.out_prefix}_weights_{safe}.png",
        )
        print(f"Saved {args.out_prefix}_weights_{safe}.png")

    # Markdown summary table
    lines = [
        "| Config | Monotone (Lemma 1) | First violation round | Blowup ratio | "
        "First simplex violation | Final sum(w) | Final min(w) | All-acc | Rare-acc |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key, entry in raw.items():
        d = entry["diagnosis"]
        blowup = d["composite_loss_blowup_ratio"]
        lines.append(
            f"| {_GAMMA_LABEL.get(key, key)} | {d['monotone_lemma1']} | "
            f"{d['first_monotonicity_violation_round']} | "
            f"{blowup if blowup is None else f'{blowup:.1f}×'} | "
            f"{d['first_simplex_violation_round']} | "
            f"{d['final_weight_sum']:.4f} | {d['final_weight_min']:.4f} | "
            f"{d['final_all_acc']*100:.2f}% | {d['final_last_n_acc']*100:.2f}% |"
        )
    table = "\n".join(lines)
    with open(f"{args.out_prefix}_summary.md", "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"Saved {args.out_prefix}_summary.md")
    print("\n" + table.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
