"""Unit test for the seed-sweep combined-fix aggregator (stage_a_plan.md §6.4).

Feeds hand-written tiny diagnostic dicts through ``aggregate`` (no training
run needed) and checks the emitted mean/std against a NumPy reference, so the
UNSW combined-fix band is trustworthy without re-running 60-round training.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "seed_sweep_combined_fix",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "seed_sweep_combined_fix.py"),
)
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)
aggregate = _mod.aggregate


def _fake_diag(all_acc: float, last_n_acc: float, blowup: float) -> dict:
    return {"diagnosis": {
        "final_all_acc": all_acc,
        "final_last_n_acc": last_n_acc,
        "composite_loss_blowup_ratio": blowup,
    }}


def test_aggregate_mean_std_matches_numpy_reference():
    per_seed = {
        0: {"g0.0": _fake_diag(0.60, 0.80, 1.0), "g1.0": _fake_diag(0.50, 0.90, 1.0)},
        1: {"g0.0": _fake_diag(0.65, 0.75, 1.0), "g1.0": _fake_diag(0.52, 0.90, 1.0)},
        2: {"g0.0": _fake_diag(0.62, 0.60, 1.0), "g1.0": _fake_diag(0.44, 1.00, 1.0)},
    }
    summary = aggregate(per_seed, gammas=[0.0, 1.0])

    assert set(summary.keys()) == {"g0.0", "g1.0"}
    for gamma in (0.0, 1.0):
        entry = summary[f"g{gamma}"]
        all_accs = [per_seed[s][f"g{gamma}"]["diagnosis"]["final_all_acc"] for s in per_seed]
        rare_accs = [per_seed[s][f"g{gamma}"]["diagnosis"]["final_last_n_acc"] for s in per_seed]

        assert entry["all_acc_mean"] == pytest.approx(np.mean(all_accs))
        assert entry["all_acc_std"] == pytest.approx(np.std(all_accs))
        assert entry["last_n_acc_mean"] == pytest.approx(np.mean(rare_accs))
        assert entry["last_n_acc_std"] == pytest.approx(np.std(rare_accs))
        assert entry["all_acc_per_seed"] == all_accs
        assert entry["last_n_acc_per_seed"] == rare_accs
        assert entry["seeds"] == [0, 1, 2]


def test_seed_filename_matches_dataset_and_seed_convention():
    fn0 = _mod._seed_filename("UNSW_NB15", "0.01", 5, 60, 0)
    fn1 = _mod._seed_filename("UNSW_NB15", "0.01", 5, 60, 1)
    assert fn0 == "UNSW_NB15_projected_combined_eta0.01_E5_r60.json"
    assert fn1 == "UNSW_NB15_projected_combined_eta0.01_E5_seed1_r60.json"
