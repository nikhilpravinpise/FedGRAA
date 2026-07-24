"""Smoke tests for the ablation renderer (stage_b_plan.md T6/§6.4).

Mirrors ``tests/test_plot.py``'s pattern: skip gracefully if matplotlib
isn't installed, use the Agg backend, and verify the public API runs
without error and produces non-empty artifacts -- not a visual inspection.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_ablation_table import build_table  # noqa: E402
from scripts.render_ablation import render_markdown  # noqa: E402
from scripts.run_ablation_seed_sweep import ALL_ARM_NAMES, output_path  # noqa: E402

pytest.importorskip("matplotlib", reason="matplotlib not installed")

DATASET = "TOY-DS"
SEEDS = [0, 1, 2]
GAMMAS = [0.0, 0.5, 1.0]
ROUNDS = 60


def _write_diag_json(diag_dir, arm, seed, rare_by_gamma):
    path = output_path(DATASET, arm, seed, ROUNDS, diag_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        f"g{g}": {"diagnosis": {
            "final_last_n_acc": v, "final_all_acc": v * 0.5 + 0.2,
            "composite_loss_blowup_ratio": 1.0 + v,
        }}
        for g, v in rare_by_gamma.items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


@pytest.fixture()
def toy_table(tmp_path):
    diag_dir = str(tmp_path / "diagnostics")
    ablation_dir = str(tmp_path / "ablation")
    os.makedirs(ablation_dir, exist_ok=True)

    base_vals = {"literal": 0.0, "projection_alone": 0.2, "eta_alone": 0.5, "both": 0.9}
    for arm in ALL_ARM_NAMES:
        for i, seed in enumerate(SEEDS):
            rare_by_gamma = {g: base_vals[arm] + 0.01 * i + 0.02 * g for g in GAMMAS}
            _write_diag_json(diag_dir, arm, seed, rare_by_gamma)

    dataset_tag = DATASET.replace("-", "_")
    for method, val in (("fedavg", 0.0), ("static", 0.85)):
        with open(os.path.join(ablation_dir, f"seed_sweep_{dataset_tag}_{method}_r60.json"), "w") as f:
            json.dump({
                "seeds": SEEDS,
                "all_acc_per_seed": [val * 0.5 + 0.2 + 0.01 * i for i in range(len(SEEDS))],
                "last_n_acc_per_seed": [val + 0.01 * i for i in range(len(SEEDS))],
            }, f)

    return build_table(DATASET, seeds=SEEDS, rounds=ROUNDS, diag_dir=diag_dir,
                        ablation_dir=ablation_dir, gammas=GAMMAS)


def test_render_markdown_produces_nonempty_text_with_all_arms(toy_table):
    md = render_markdown(toy_table)
    assert len(md) > 0
    assert "TOY-DS" in md
    assert "FedAvg" in md
    assert "Both" in md
    assert "factorial" in md.lower()


def test_render_markdown_includes_significance_markers_where_expected(toy_table):
    md = render_markdown(toy_table)
    # "both" (0.9) is far from "literal" (0.0) at every seed with tiny CI ->
    # should be significant, so at least one star must appear in the table.
    assert "*" in md


def test_render_reports_blowup_as_median_not_mean(toy_table):
    """Guards the real audit finding: a mean blowup of 185764x (one diverging
    seed out of ten) misrepresents the cell. The rendered table must show the
    median, and surface the max explicitly rather than averaging it in."""
    from scripts.render_ablation import _fmt_blowup

    heavy = {"median": 3.70, "min": 2.47, "max": 1857580.09, "mean": 185764.05}
    out = _fmt_blowup(heavy)
    assert "3.70" in out
    assert "med" in out
    assert "1.86e+06" in out or "1.86e6" in out or "max" in out
    assert "185764" not in out          # the misleading mean must not appear

    flat = {"median": 1.0, "min": 1.0, "max": 1.0, "mean": 1.0}
    assert _fmt_blowup(flat) == "1.00× (med)"

    md = render_markdown(toy_table)
    assert "median (and max)" in md     # the choice is documented in the table


def test_render_figure_produces_nonempty_png(tmp_path, toy_table):
    import matplotlib
    matplotlib.use("Agg")
    from scripts.render_ablation import render_figure

    out_path = str(tmp_path / "toy_ablation.png")
    fig = render_figure(toy_table, save_path=out_path)
    assert fig is not None
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_full_cli_main_writes_both_artifacts(tmp_path, toy_table, monkeypatch):
    ablation_dir = str(tmp_path / "cli_ablation")
    os.makedirs(ablation_dir, exist_ok=True)
    dataset_tag = DATASET.replace("-", "_")
    with open(os.path.join(ablation_dir, f"{dataset_tag}_ablation_table.json"), "w") as f:
        json.dump(toy_table, f)

    monkeypatch.setattr(sys, "argv", [
        "render_ablation.py", "--dataset", DATASET, "--ablation-dir", ablation_dir,
    ])
    import matplotlib
    matplotlib.use("Agg")
    from scripts.render_ablation import main
    assert main() == 0

    assert os.path.exists(os.path.join(ablation_dir, f"{dataset_tag}_ablation_table.md"))
    assert os.path.getsize(os.path.join(ablation_dir, f"{dataset_tag}_ablation.png")) > 0


def test_render_without_matplotlib(monkeypatch):
    import scripts.render_ablation as render_mod

    def _boom():
        raise ImportError("matplotlib is required for scripts/render_ablation.py.")

    monkeypatch.setattr(render_mod, "_require_mpl", _boom)
    with pytest.raises(ImportError, match="matplotlib"):
        render_mod.render_figure({"dataset": "X", "gammas": [0.0], "arms": {}, "baselines": {}})
