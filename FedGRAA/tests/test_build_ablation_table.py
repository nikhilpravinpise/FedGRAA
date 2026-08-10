"""Unit tests for the ablation-table builder (stage_b_plan.md T5/§6.3).

Feeds tiny, hand-written per-seed diagnostic JSONs (written via the same
``output_path`` naming convention the real driver uses) and a tiny baseline
seed-sweep JSON through ``build_table`` -- no training, no real 10-seed
data needed -- and checks the emitted cells, pairwise comparisons, and
factorial effects against hand computation, mirroring
``tests/test_seed_sweep_combined_fix.py``'s pattern for Stage A.
"""

from __future__ import annotations

import json
import os
import sys

import pytest
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedgraa.stats import holm_bonferroni  # noqa: E402
from scripts.build_ablation_table import COMPARISON_ARMS, build_table  # noqa: E402
from scripts.run_ablation_seed_sweep import output_path  # noqa: E402

DATASET = "TOY-DS"
SEEDS = [0, 1, 2]
GAMMAS = [0.0, 1.0]
ROUNDS = 60

# rare-class accuracy per arm, per gamma, per seed (index-aligned with SEEDS)
RARE_ACC = {
    "literal":           {0.0: [0.00, 0.00, 0.00], 1.0: [0.00, 0.00, 0.00]},
    "projection_alone":  {0.0: [0.10, 0.20, 0.15], 1.0: [0.05, 0.10, 0.08]},
    "eta_alone":         {0.0: [0.50, 0.55, 0.45], 1.0: [0.60, 0.58, 0.62]},
    "both":              {0.0: [0.90, 0.90, 0.90], 1.0: [0.917, 0.917, 0.917]},
}
BASELINE_RARE_ACC = {
    "fedavg": [0.00, 0.00, 0.00],
    "static": [0.85, 0.83, 0.88],
}


def _write_diag_json(diag_dir: str, arm: str, seed: int, per_gamma_rare_acc: dict):
    path = output_path(DATASET, arm, seed, ROUNDS, diag_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {}
    for gamma, val in per_gamma_rare_acc.items():
        payload[f"g{gamma}"] = {
            "diagnosis": {
                "final_last_n_acc": val,
                "final_all_acc": val * 0.6 + 0.1,
                "composite_loss_blowup_ratio": 1.0 + val * 0.01,
            }
        }
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        existing.update(payload)
        payload = existing
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


@pytest.fixture()
def toy_table(tmp_path):
    diag_dir = str(tmp_path / "diagnostics")
    ablation_dir = str(tmp_path / "ablation")
    os.makedirs(ablation_dir, exist_ok=True)

    for arm, per_gamma in RARE_ACC.items():
        for i, seed in enumerate(SEEDS):
            _write_diag_json(diag_dir, arm, seed, {g: per_gamma[g][i] for g in GAMMAS})

    dataset_tag = DATASET.replace("-", "_")
    for method, values in BASELINE_RARE_ACC.items():
        with open(os.path.join(ablation_dir, f"seed_sweep_{dataset_tag}_{method}_r60.json"), "w") as f:
            json.dump({
                "seeds": SEEDS,
                "all_acc_per_seed": [v * 0.6 + 0.1 for v in values],
                "last_n_acc_per_seed": values,
            }, f)

    return build_table(DATASET, seeds=SEEDS, rounds=ROUNDS, diag_dir=diag_dir,
                        ablation_dir=ablation_dir, gammas=GAMMAS)


# --------------------------------------------------------------------------- #
# Arm cells
# --------------------------------------------------------------------------- #
def test_arm_cell_means_match_hand_computation(toy_table):
    for arm, per_gamma in RARE_ACC.items():
        for gamma in GAMMAS:
            cell = toy_table["arms"][arm][f"g{gamma}"]["rare_acc"]
            expected_mean = sum(per_gamma[gamma]) / len(per_gamma[gamma])
            assert cell["mean"] == pytest.approx(expected_mean)
            assert cell["per_seed"] == pytest.approx(per_gamma[gamma])
            assert cell["seeds"] == SEEDS


def test_arm_cell_constant_arm_has_point_ci_not_nan(toy_table):
    cell = toy_table["arms"]["both"]["g1.0"]["rare_acc"]
    lo, hi = cell["ci_95"]
    assert lo == pytest.approx(0.917)
    assert hi == pytest.approx(0.917)


def test_arm_cell_carries_median_min_max(toy_table):
    """Blowup is a heavy-tailed ratio, so every cell must expose median/min/max
    alongside the mean -- reporting only the mean lets one diverging seed
    describe a cell no run actually landed in."""
    for arm in RARE_ACC:
        for gamma in GAMMAS:
            for metric in ("rare_acc", "all_acc", "blowup"):
                cell = toy_table["arms"][arm][f"g{gamma}"][metric]
                values = sorted(cell["per_seed"])
                assert cell["min"] == pytest.approx(values[0])
                assert cell["max"] == pytest.approx(values[-1])
                n = len(values)
                expected_median = (values[n // 2] if n % 2
                                   else (values[n // 2 - 1] + values[n // 2]) / 2)
                assert cell["median"] == pytest.approx(expected_median)


def test_median_is_robust_to_a_single_diverging_seed():
    """The real defect this guards against: CIC-IDS-2017's literal arm at
    gamma=0 has one seed at 1.86e6x and nine near 3x, so mean=185764x while
    median=3.70x. The median must track the nine, not the one."""
    from scripts.build_ablation_table import _cell
    values = [3.78, 7.59, 29.76, 2.58, 3.62, 3.59, 2.47, 3.14, 1857580.09, 3.87]
    cell = _cell(values, list(range(10)))
    assert cell["mean"] > 100_000            # mean is dominated by the outlier
    assert cell["median"] == pytest.approx(3.70, abs=0.01)
    assert cell["max"] == pytest.approx(1857580.09)
    assert cell["min"] == pytest.approx(2.47)


def test_baseline_cells_present_and_correct(toy_table):
    for method, values in BASELINE_RARE_ACC.items():
        cell = toy_table["baselines"][method]["rare_acc"]
        assert cell["per_seed"] == pytest.approx(values)
        assert cell["mean"] == pytest.approx(sum(values) / len(values))


def test_missing_baseline_is_none_not_a_crash(tmp_path):
    diag_dir = str(tmp_path / "diagnostics")
    for arm, per_gamma in RARE_ACC.items():
        for i, seed in enumerate(SEEDS):
            _write_diag_json(diag_dir, arm, seed, {g: per_gamma[g][i] for g in GAMMAS})
    table = build_table(DATASET, seeds=SEEDS, rounds=ROUNDS, diag_dir=diag_dir,
                         ablation_dir=str(tmp_path / "no_baselines_here"), gammas=GAMMAS)
    assert table["baselines"]["fedavg"] is None
    assert table["baselines"]["static"] is None
    assert table["comparisons"]["g0.0"]["fedavg"] is None


# --------------------------------------------------------------------------- #
# Pairwise comparisons + Holm wiring
# --------------------------------------------------------------------------- #
def test_comparison_against_zero_variance_arm_is_degenerate_safe(toy_table):
    """'both' vs 'literal': both is constant 0.90, literal is constant 0.00
    -- diffs are constant 0.90 at every seed, the exact zero-variance
    scenario stage_b_plan.md §4 flags. Must not be NaN."""
    comp = toy_table["comparisons"]["g0.0"]["literal"]
    assert comp["p_ttest"] == 1.0
    assert comp["t_statistic"] == 0.0
    assert comp["cohens_d"] == float("inf")
    assert "nan" not in json.dumps(comp).lower()


def test_comparison_against_varying_arm_matches_scipy(toy_table):
    both = RARE_ACC["both"][0.0]
    proj = RARE_ACC["projection_alone"][0.0]
    t_ref, p_ref = scipy_stats.ttest_rel(both, proj)
    comp = toy_table["comparisons"]["g0.0"]["projection_alone"]
    assert comp["t_statistic"] == pytest.approx(t_ref)
    assert comp["p_ttest"] == pytest.approx(p_ref)


def test_holm_correction_wired_across_all_five_comparisons(toy_table):
    raw_p_in_order = [toy_table["comparisons"]["g0.0"][name]["p_ttest"] for name in COMPARISON_ARMS]
    expected_holm = holm_bonferroni(raw_p_in_order)
    for name, expected in zip(COMPARISON_ARMS, expected_holm):
        assert toy_table["comparisons"]["g0.0"][name]["p_holm"] == pytest.approx(expected)


def test_all_five_comparisons_present_when_all_data_available(toy_table):
    assert set(toy_table["comparisons"]["g0.0"].keys()) == set(COMPARISON_ARMS)
    assert all(v is not None for v in toy_table["comparisons"]["g0.0"].values())


# --------------------------------------------------------------------------- #
# 2x2 factorial effects
# --------------------------------------------------------------------------- #
def test_factorial_projection_main_effect_matches_hand_computation(toy_table):
    both, proj = RARE_ACC["both"][0.0], RARE_ACC["projection_alone"][0.0]
    lit, eta = RARE_ACC["literal"][0.0], RARE_ACC["eta_alone"][0.0]
    expected = [0.5 * (b + p) - 0.5 * (l + e) for b, p, l, e in zip(both, proj, lit, eta)]
    expected_mean = sum(expected) / len(expected)
    assert toy_table["factorial"]["g0.0"]["projection_main_effect"]["mean"] == pytest.approx(expected_mean)


def test_factorial_eta_main_effect_matches_hand_computation(toy_table):
    both, eta = RARE_ACC["both"][0.0], RARE_ACC["eta_alone"][0.0]
    lit, proj = RARE_ACC["literal"][0.0], RARE_ACC["projection_alone"][0.0]
    expected = [0.5 * (b + e) - 0.5 * (l + p) for b, e, l, p in zip(both, eta, lit, proj)]
    expected_mean = sum(expected) / len(expected)
    assert toy_table["factorial"]["g0.0"]["eta_main_effect"]["mean"] == pytest.approx(expected_mean)


def test_factorial_interaction_matches_hand_computation(toy_table):
    both, proj = RARE_ACC["both"][0.0], RARE_ACC["projection_alone"][0.0]
    eta, lit = RARE_ACC["eta_alone"][0.0], RARE_ACC["literal"][0.0]
    expected = [(b - p) - (e - l) for b, p, e, l in zip(both, proj, eta, lit)]
    expected_mean = sum(expected) / len(expected)
    assert toy_table["factorial"]["g0.0"]["interaction"]["mean"] == pytest.approx(expected_mean)


def test_factorial_effects_computed_for_every_gamma(toy_table):
    for gamma in GAMMAS:
        fx = toy_table["factorial"][f"g{gamma}"]
        assert fx is not None
        for key in ("projection_main_effect", "eta_main_effect", "interaction"):
            assert fx[key]["n_seeds"] == len(SEEDS)


# --------------------------------------------------------------------------- #
# Top-level shape
# --------------------------------------------------------------------------- #
def test_table_top_level_keys(toy_table):
    assert toy_table["dataset"] == DATASET
    assert set(toy_table["arms"].keys()) == {"literal", "projection_alone", "eta_alone", "both"}
    assert set(toy_table["baselines"].keys()) == {"fedavg", "static"}
    assert set(toy_table["comparisons"].keys()) == {"g0.0", "g1.0"}
