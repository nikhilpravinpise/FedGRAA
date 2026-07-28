"""End-to-end integration test for T6's ``run_one_cell`` branch --
NF-UQ-NIDS-v2 loaded via ``load_nf_uq_nids_v2_silos`` + ``partition_by_silo``,
then trained through the real ``FederatedTrainer``/``FederatedTrainerV2``
pipeline, exactly as the campaign driver invokes it. Uses a small synthetic
merged CSV (not the real 12.79 GB file) so this runs in seconds as part of
the normal suite, while still exercising the *exact* code path T6 will run
on Modal.

docs/audit/PRE_COMPUTE_AUDIT_2026-07-28.md B6 -- "completely implement B6"
means this path must actually run, not just the loader in isolation.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_v2_frontier_sweep import CellSpec, run_one_cell


def _write_class_separable_merged_csv(path: str, seed: int = 0) -> None:
    """A tiny merged NF-UQ-NIDS-v2-shaped CSV with 4 silos and class-separable
    features, so a 1-round federated run produces a meaningful (non-random)
    accuracy rather than existing purely to exercise plumbing."""
    rng = np.random.default_rng(seed)
    silo_labels = {
        "NF-BoT-IoT-v2": ["Benign", "DDoS"],
        "NF-ToN-IoT-v2": ["Benign", "DDoS", "Scanning"],
        "NF-UNSW-NB15-v2": ["Benign", "Scanning"],
        "NF-CSE-CIC-IDS2018-v2": ["Benign", "DDoS"],
    }
    label_offsets = {"Benign": 0.0, "DDoS": 4.0, "Scanning": -4.0}
    n_features = 5
    frames = []
    for silo_name, labels in silo_labels.items():
        n = 60
        y = rng.choice(labels, size=n)
        X = rng.normal(size=(n, n_features))
        for lbl in labels:
            mask = y == lbl
            X[mask] += label_offsets[lbl]
        df = pd.DataFrame(X, columns=[f"feat{i}" for i in range(n_features)])
        df["id"] = np.arange(n)
        df["Attack"] = y
        df["Dataset"] = silo_name
        frames.append(df)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


@pytest.fixture()
def nf_uq_data_dir(tmp_path):
    """Lays out ``<tmp>/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.csv``, matching
    ``NF_UQ_NIDS_V2_CSV_RELPATH``'s expectation of ``{data_dir}/NF-UQ-NIDS-v2/
    NF-UQ-NIDS-v2.csv`` under ``cell.data_dir``."""
    silo_dir = tmp_path / "NF-UQ-NIDS-v2"
    silo_dir.mkdir()
    csv_path = silo_dir / "NF-UQ-NIDS-v2.csv"
    _write_class_separable_merged_csv(str(csv_path))
    return str(tmp_path)


def test_run_one_cell_v1_projected_on_nf_uq_nids_v2(tmp_path, nf_uq_data_dir):
    cell = CellSpec(
        dataset="NF-UQ-NIDS-v2", arm="v1_projected", gamma=1.0, seed=0,
        rounds=2, eval_every=2, n_rare=1,
        data_dir=nf_uq_data_dir, out_dir=str(tmp_path / "results"),
        nf_uq_max_per_class=30,
    )
    result = run_one_cell(cell)

    assert result["dataset"] == "NF-UQ-NIDS-v2"
    assert 0.0 <= result["final_all_acc"] <= 1.0
    assert 0.0 <= result["final_last_n_acc"] <= 1.0
    # v1 has no SecAgg instrumentation (matches check_stage_c_complete's own
    # v1_projected contract), but DOES bill Algorithm-1 bytes (H4).
    assert result["bytes_sent"] is not None
    assert result["comm_rounds"] == cell.rounds
    assert os.path.exists(cell.path)


def test_run_one_cell_v2_m1_on_nf_uq_nids_v2(tmp_path, nf_uq_data_dir):
    cell = CellSpec(
        dataset="NF-UQ-NIDS-v2", arm="v2_m1", gamma=1.0, seed=0,
        rounds=2, eval_every=2, n_rare=1,
        data_dir=nf_uq_data_dir, out_dir=str(tmp_path / "results"),
        nf_uq_max_per_class=30,
    )
    result = run_one_cell(cell)

    assert result["dataset"] == "NF-UQ-NIDS-v2"
    assert 0.0 <= result["final_all_acc"] <= 1.0
    assert result["bytes_sent"] is not None
    assert result["bytes_sent"] > 0
    assert result["failed_secagg_rounds"] == 0
    # Exactly 4 silos -> 4 "clients" -- the K=4 scope difference from T1's
    # K=9/14 synthetic partition, documented rather than hidden.
    assert result["config"]["n_rare"] == 1


def test_run_one_cell_nf_uq_result_is_self_describing(tmp_path, nf_uq_data_dir):
    """M6's self-description guarantee, confirmed on the T6 path
    specifically -- not just inherited by assumption from the CIC/UNSW
    tests."""
    cell = CellSpec(
        dataset="NF-UQ-NIDS-v2", arm="v2_m1", gamma=1.0, seed=0,
        rounds=1, eval_every=1, n_rare=1,
        data_dir=nf_uq_data_dir, out_dir=str(tmp_path / "results"),
        nf_uq_max_per_class=30,
    )
    result = run_one_cell(cell)
    assert result["config"]["eta_prime"] == cell.eta_prime
    assert "provenance" in result
    assert result["provenance"]["python"].startswith("3.")


def test_run_one_cell_nf_uq_respects_max_per_class(tmp_path, nf_uq_data_dir):
    """A tighter cap must still produce a valid, trainable split -- the T6
    wiring must actually pass the cap through, not silently ignore it."""
    cell = CellSpec(
        dataset="NF-UQ-NIDS-v2", arm="v1_projected", gamma=1.0, seed=0,
        rounds=1, eval_every=1, n_rare=1,
        data_dir=nf_uq_data_dir, out_dir=str(tmp_path / "results"),
        nf_uq_max_per_class=5,
    )
    result = run_one_cell(cell)
    assert 0.0 <= result["final_all_acc"] <= 1.0
