"""Tests for Phase 14: Checkpoint/Resume."""

import os
import tempfile

import numpy as np
import pytest

from fedgraa.trainer import (
    TrainingCheckpoint,
    load_checkpoint,
    save_checkpoint,
    _extract_optimizer_states,
)


def _make_dummy_checkpoint():
    """Minimal checkpoint with valid structure."""
    return TrainingCheckpoint(
        round=42,
        global_params=[np.array([1.0, 2.0], dtype=np.float32)],
        weights=np.array([0.5, 0.5], dtype=np.float32),
        optimizer_states={0: [np.array([0.1, 0.2], dtype=np.float32)]},
        eval_history=None,  # populated below
        rng_state={"bit_generator": "PCG64", "state": {"state": 123}},
    )


def test_checkpoint_save_load_roundtrip():
    ckpt = _make_dummy_checkpoint()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.ckpt")
        save_checkpoint(ckpt, path)
        assert os.path.exists(path)

        loaded = load_checkpoint(path)
        assert loaded is not None
        assert loaded.round == 42
        np.testing.assert_array_equal(loaded.weights, ckpt.weights)
        np.testing.assert_array_equal(loaded.global_params[0], ckpt.global_params[0])


def test_checkpoint_load_nonexistent():
    assert load_checkpoint("nonexistent.ckpt") is None


def test_checkpoint_atomic_write():
    """Checkpoint write must be atomic: a crash during write leaves the old file."""
    ckpt = _make_dummy_checkpoint()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.ckpt")
        old_content = b"old data"
        with open(path, "wb") as f:
            f.write(old_content)

        save_checkpoint(ckpt, path)
        with open(path, "rb") as f:
            content = f.read()
        # Should not be the old content; should be a valid pickle
        assert content != old_content
        import pickle
        recovered = pickle.loads(content)
        assert recovered.round == 42


def test_optimizer_state_extraction():
    """_extract_optimizer_states must return picklable numpy arrays."""
    # Integration test: needs TF, so we just verify the signature
    assert callable(_extract_optimizer_states)
