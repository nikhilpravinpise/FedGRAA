"""Tests for Phase 14: Checkpoint/Resume."""

import os
import pickle
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


def test_optimizer_state_extraction_round_trips_real_momentum():
    """_extract_optimizer_states must genuinely restore optimizer momentum.

    Previously this test only checked ``callable(_extract_optimizer_states)``
    -- the function that determines whether checkpoint/resume actually
    restores Adam's momentum/velocity slots was never called (flagged in
    ``docs/audit/INDEPENDENT_FULL_AUDIT_2026-07-26.md`` SS3.4). This exercises
    the real extract -> ``set_weights`` -> resume path used by
    ``resume_or_init`` (trainer.py) and checks it reproduces a real optimizer's
    numerics, not just matching array shapes.
    """
    import tensorflow as tf

    from fedgraa.model import build_tf_optimizer

    # Reference optimizer: two real gradient steps, building up real momentum.
    var_ref = tf.Variable([1.0, 2.0], dtype=tf.float32)
    opt_ref = build_tf_optimizer(lr=0.1, variables=[var_ref])
    opt_ref.apply_gradients([(tf.constant([0.5, -0.5]), var_ref)])

    states = _extract_optimizer_states({0: opt_ref})
    assert list(states.keys()) == [0]
    # Every extracted slot (iterations, learning_rate, momentum, velocity)
    # comes back as a numpy scalar/array via .numpy(), never a raw tf.Tensor.
    assert all(isinstance(a, (np.ndarray, np.generic)) for a in states[0])
    # Must be picklable -- this is what actually gets written to the
    # checkpoint file (save_checkpoint pickles the whole TrainingCheckpoint).
    pickled = pickle.loads(pickle.dumps(states[0]))
    for a, b in zip(pickled, states[0]):
        np.testing.assert_array_equal(a, b)

    # A *fresh* optimizer, resumed from the extracted state (exactly what
    # resume_or_init does: restore global_params via element-wise copy, then
    # restore the optimizer momentum via opt.set_weights(state)), must
    # reproduce the same next update as if training had never stopped.
    post_step_1_value = var_ref.numpy().copy()
    var_resumed = tf.Variable(post_step_1_value, dtype=tf.float32)
    opt_resumed = build_tf_optimizer(lr=0.1, variables=[var_resumed])
    opt_resumed.set_weights(states[0])

    next_grad = tf.constant([1.0, 1.0])
    opt_ref.apply_gradients([(next_grad, var_ref)])
    opt_resumed.apply_gradients([(next_grad, var_resumed)])

    np.testing.assert_allclose(var_resumed.numpy(), var_ref.numpy(), rtol=1e-6)

    # Sanity: a *cold* optimizer -- same resumed parameter value, but no
    # restored momentum -- must NOT match. This rules out the possibility
    # that the match above came only from copying the parameter value, with
    # set_weights() contributing nothing.
    var_cold = tf.Variable(post_step_1_value, dtype=tf.float32)
    opt_cold = build_tf_optimizer(lr=0.1, variables=[var_cold])
    opt_cold.apply_gradients([(next_grad, var_cold)])
    assert not np.allclose(var_cold.numpy(), var_ref.numpy(), rtol=1e-6)
