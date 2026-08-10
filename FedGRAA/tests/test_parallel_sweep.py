"""Tests for Phase 9: Sweep-Level Parallelism.

``test_run_single_config_is_picklable`` and ``test_parallel_sweep_function_exists``
were the only coverage this module had for a long time -- neither actually
*calls* the orchestration logic, so a real bug in argument unpacking, config
reconstruction, or the worker-exception handling would pass silently (flagged
in ``docs/audit/INDEPENDENT_FULL_AUDIT_2026-07-26.md`` SS3.4/SS3.5). The tests
below exercise the real code paths: ``_run_single_config`` is called directly
(fast, monkeypatched ``run_experiment``) to verify argument threading and the
worker-exception-swallowing behaviour, and one real, unmocked, end-to-end run
of ``run_gamma_sweep_parallel`` (real ``ProcessPoolExecutor``, real training on
the synthetic-fallback dataset) is kept as a slower integration test.
"""

import pickle

import pytest

from fedgraa import experiment as experiment_module
from fedgraa.config import FedConfig
from fedgraa.experiment import (
    ExperimentResult,
    _run_single_config,
    run_gamma_sweep_parallel,
)


def test_run_single_config_is_picklable():
    """_run_single_config must be picklable for ProcessPoolExecutor on Windows."""
    pickled = pickle.dumps(_run_single_config)
    unpickled = pickle.loads(pickled)
    assert unpickled is _run_single_config


def _args(key="fedavg", method="fedavg", gamma=0.0, cfg_dict=None, checkpoint_dir=None):
    return (
        key, "CIC-IDS-2017", method, gamma, 3, 1, None,
        1, 0, False, None, cfg_dict, checkpoint_dir,
    )


def test_run_single_config_threads_arguments_into_run_experiment(monkeypatch):
    """The (key, dataset, method, ...) tuple must reach run_experiment unchanged."""
    captured = {}

    def fake_run_experiment(dataset, **kwargs):
        captured["dataset"] = dataset
        captured.update(kwargs)
        return "sentinel-result"

    monkeypatch.setattr(experiment_module, "run_experiment", fake_run_experiment)

    cfg_dict = {
        "rounds": 1, "gamma": 0.5, "n_rare": 3, "seed": 7, "eta": 0.1,
        "local_epochs": 2, "batch_size": 64, "eta_prime": 0.1,
        "weight_epochs": 5, "val_fraction": 0.1, "round_zero_zero": True,
    }
    key, result = _run_single_config(
        _args(key="fedgraa_g0.5", method="fedgraa", gamma=0.5, cfg_dict=cfg_dict)
    )

    assert key == "fedgraa_g0.5"
    assert result == "sentinel-result"
    assert captured["dataset"] == "CIC-IDS-2017"
    assert captured["method"] == "fedgraa"
    assert captured["gamma"] == 0.5
    assert captured["seed"] == 0  # the sweep's seed argument, not cfg_dict's
    assert isinstance(captured["cfg"], FedConfig)
    assert captured["cfg"].local_epochs == 2


def test_run_single_config_catches_worker_exception_instead_of_raising(monkeypatch):
    """A crashed worker must come back as (key, exception), never propagate.

    This is the behaviour run_gamma_sweep_parallel/run_n_sweep_parallel rely on
    to keep one bad config from taking down the whole sweep (see the caller's
    ``isinstance(result_or_exc, Exception)`` check) -- worth pinning directly,
    since the audit noted the crashed config is otherwise silently absent from
    the results dict with no distinguishing signal.
    """

    def failing_run_experiment(dataset, **kwargs):
        raise ValueError("synthetic worker failure")

    monkeypatch.setattr(experiment_module, "run_experiment", failing_run_experiment)

    with pytest.warns(RuntimeWarning, match="synthetic worker failure"):
        key, result = _run_single_config(_args(key="fedavg"))

    assert key == "fedavg"
    assert isinstance(result, ValueError)
    assert str(result) == "synthetic worker failure"


@pytest.mark.slow
def test_parallel_gamma_sweep_runs_real_training_end_to_end():
    """Real ProcessPoolExecutor, real (synthetic-fallback) training, no mocks.

    Slow (~30-60s: process spawn + one real federated round per method) but
    this is the only test in the suite that would catch a genuine regression
    in the multiprocessing orchestration itself -- e.g. a config silently
    dropped, or all_acc/last_n_acc never actually populated by a real client.
    """
    results = run_gamma_sweep_parallel(
        "CIC-IDS-2017",
        gammas=(0.0,),
        n_rare=3,
        rounds=1,
        eval_every=1,
        workers=1,
        data_dir=None,
        cache_dir=None,
    )

    assert set(results.keys()) == {"fedavg", "static", "fedgraa_g0.0"}
    for key, result in results.items():
        assert isinstance(result, ExperimentResult)
        assert 0.0 <= result.final_all_acc <= 1.0
        assert 0.0 <= result.final_last_n_acc <= 1.0


def test_parallel_sweep_function_exists():
    """run_gamma_sweep_parallel must be importable and callable."""
    assert callable(run_gamma_sweep_parallel)
    # Just verify it accepts the standard sweep arguments
    import inspect
    sig = inspect.signature(run_gamma_sweep_parallel)
    assert "dataset" in sig.parameters
    assert "workers" in sig.parameters
