"""Tests for Phase 9: Sweep-Level Parallelism.

These tests verify that the parallel sweep infrastructure is wired
correctly. Full end-to-end parallel execution requires TF and multiple
cores, so the heavy tests are marked ``slow``.
"""

import pickle

from fedgraa.experiment import _run_single_config, run_gamma_sweep_parallel


def test_run_single_config_is_picklable():
    """_run_single_config must be picklable for ProcessPoolExecutor on Windows."""
    pickled = pickle.dumps(_run_single_config)
    unpickled = pickle.loads(pickled)
    assert unpickled is _run_single_config


def test_parallel_sweep_function_exists():
    """run_gamma_sweep_parallel must be importable and callable."""
    assert callable(run_gamma_sweep_parallel)
    # Just verify it accepts the standard sweep arguments
    import inspect
    sig = inspect.signature(run_gamma_sweep_parallel)
    assert "dataset" in sig.parameters
    assert "workers" in sig.parameters
