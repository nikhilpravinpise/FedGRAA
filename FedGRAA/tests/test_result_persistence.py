"""Tests for result persistence (gap 3.4 / P1: results existed only as prose).

``save_results``/``load_results`` (flat gamma-sweep) had no test coverage at
all before this file. ``save_n_sweep_results``/``load_n_sweep_results`` (nested
n-sweep) are new, added so ``run_experiments.py --results-dir`` can persist
Figures 2-3 the same way ``--results-dir`` already persists Figure 1.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from fedgraa.config import DatasetSpec, FedConfig
from fedgraa.data import make_synthetic_dataset
from fedgraa.experiment import (
    ExperimentResult,
    load_n_sweep_results,
    load_results,
    save_n_sweep_results,
    save_results,
)
from fedgraa.partition import partition_non_iid
from fedgraa.trainer import FederatedTrainer


def _toy_spec():
    return DatasetSpec(
        name="PERSIST_TOY",
        n_features=5,
        labels=["Benign", "A", "B", "Rare"],
        used_counts=[200, 80, 60, 8],
        benign_id=0,
        rare_label_order=["Rare"],
    )


def _toy_split():
    spec = _toy_spec()
    X, y, spec = make_synthetic_dataset(spec, seed=0, class_separation=3.0)
    return partition_non_iid(X, y, spec, n_rare=1, val_fraction=0.1, seed=0)


def _run(method, gamma, split):
    cfg = FedConfig(rounds=6, gamma=gamma or 0.0, n_rare=1, seed=0)
    hist = FederatedTrainer(split, cfg, method=method).train(eval_every=3)
    return ExperimentResult(
        dataset="PERSIST_TOY", method=method, gamma=gamma, n_rare=1, history=hist
    )


@pytest.fixture(scope="module")
def toy_gamma_results():
    split = _toy_split()
    return {
        "fedavg": _run("fedavg", None, split),
        "static": _run("static", None, split),
        "fedgraa_g1.0": _run("fedgraa", 1.0, split),
    }


def test_save_load_results_roundtrip(tmp_path, toy_gamma_results):
    path = str(tmp_path / "gamma_sweep.json")
    save_results(toy_gamma_results, path)
    assert os.path.exists(path)

    reloaded = load_results(path)
    assert set(reloaded.keys()) == set(toy_gamma_results.keys())
    for key, original in toy_gamma_results.items():
        r = reloaded[key]
        assert r.dataset == original.dataset
        assert r.method == original.method
        assert r.gamma == original.gamma
        assert r.n_rare == original.n_rare
        assert r.history.rounds == original.history.rounds
        np.testing.assert_allclose(r.history.all_acc, original.history.all_acc)
        np.testing.assert_allclose(r.history.last_n_acc, original.history.last_n_acc)
        assert r.final_all_acc == pytest.approx(original.final_all_acc)
        assert r.final_last_n_acc == pytest.approx(original.final_last_n_acc)


def test_save_results_serializes_nan_composite_loss_as_null(tmp_path, toy_gamma_results):
    """FedAvg/Static have no composite loss (NaN) — must round-trip as JSON
    null, not as the non-standard-JSON token 'NaN'."""
    path = str(tmp_path / "gamma_sweep.json")
    save_results(toy_gamma_results, path)

    import json
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["fedavg"]["composite_loss"] == [None] * len(raw["fedavg"]["rounds"])

    reloaded = load_results(path)
    assert all(np.isnan(v) for v in reloaded["fedavg"].history.composite_loss)


def test_save_load_n_sweep_results_roundtrip(tmp_path):
    split = _toy_split()
    n_results = {
        1: {
            "fedavg": _run("fedavg", None, split),
            "fedgraa_g0.5": _run("fedgraa", 0.5, split),
        },
    }
    path = str(tmp_path / "n_sweep.json")
    save_n_sweep_results(n_results, path)
    assert os.path.exists(path)

    reloaded = load_n_sweep_results(path)
    assert set(reloaded.keys()) == {1}
    assert set(reloaded[1].keys()) == {"fedavg", "fedgraa_g0.5"}
    for key, original in n_results[1].items():
        r = reloaded[1][key]
        assert r.method == original.method
        assert r.gamma == original.gamma
        assert r.history.rounds == original.history.rounds
        np.testing.assert_allclose(r.history.all_acc, original.history.all_acc)


def test_n_sweep_json_keys_are_strings_but_reload_gives_int_keys(tmp_path):
    split = _toy_split()
    n_results = {3: {"static": _run("static", None, split)}}
    path = str(tmp_path / "n_sweep.json")
    save_n_sweep_results(n_results, path)

    import json
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    assert list(raw.keys()) == ["3"]  # JSON object keys are always strings

    reloaded = load_n_sweep_results(path)
    assert list(reloaded.keys()) == [3]  # but the public API restores int keys
