"""Tests for the Keras-backed MLP: shapes, forward pass, and gradients."""

import numpy as np
import pytest
import tensorflow as tf

from fedgraa.model import (
    MLP,
    Adam,
    _fused_client_step,
    aggregate,
    build_tf_optimizer,
    init_params,
)


def test_forward_shapes(tiny_layer_sizes, rng):
    model = MLP(tiny_layer_sizes, seed=0)
    X = rng.normal(size=(10, tiny_layer_sizes[0])).astype(np.float32)
    probs, _ = model.forward(X)
    assert probs.shape == (10, tiny_layer_sizes[-1])
    # softmax rows sum to 1
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(10), atol=1e-6)


def test_loss_matches_manual_cross_entropy(tiny_layer_sizes, rng):
    """loss() and loss_and_grad() must agree and match tf.nn.log_softmax."""
    model = MLP(tiny_layer_sizes, seed=2)
    params = model.get_params()
    X = rng.normal(size=(7, tiny_layer_sizes[0])).astype(np.float32)
    y = rng.integers(0, tiny_layer_sizes[-1], size=7)

    loss_scalar = model.loss(params, X, y)
    loss_from_grad, _ = model.loss_and_grad(params, X, y)
    np.testing.assert_allclose(
        loss_scalar, loss_from_grad, atol=1e-5,
        err_msg="MLP.loss() and MLP.loss_and_grad() disagree",
    )

    logits = model._model(X, training=False).numpy()
    logp = tf.nn.log_softmax(logits).numpy()
    expected = float(-logp[np.arange(7), y].mean())
    np.testing.assert_allclose(loss_scalar, expected, atol=1e-5)


def test_backprop_matches_finite_differences(tiny_layer_sizes, rng):
    """The analytic gradient must match central finite differences."""
    model = MLP(tiny_layer_sizes, seed=3)
    params = model.get_params()
    X = rng.normal(size=(12, tiny_layer_sizes[0])).astype(np.float32)
    y = rng.integers(0, tiny_layer_sizes[-1], size=12)

    _, grads = model.loss_and_grad(params, X, y)

    eps = 1e-5
    for p_idx, p in enumerate(params):
        # check a handful of entries per array
        flat = p.ravel()
        g_flat = grads[p_idx].ravel()
        for j in rng.choice(flat.size, size=min(5, flat.size), replace=False):
            orig = flat[j]
            flat[j] = orig + eps
            lp = model.loss(params, X, y)
            flat[j] = orig - eps
            lm = model.loss(params, X, y)
            flat[j] = orig
            num = (lp - lm) / (2 * eps)
            assert abs(num - g_flat[j]) < 5e-2, (p_idx, j, num, g_flat[j])


def test_aggregate_linear_combination(tiny_layer_sizes, rng):
    p0 = init_params(tiny_layer_sizes, seed=0)
    p1 = init_params(tiny_layer_sizes, seed=1)
    w = np.array([0.3, 0.7])
    agg = aggregate([p0, p1], w)
    for a, b, c in zip(agg, p0, p1):
        np.testing.assert_allclose(a, 0.3 * b + 0.7 * c, atol=1e-7, rtol=1e-6)


def test_forward_caches_final_logits(tiny_layer_sizes, rng):
    """forward() must expose final logits for downstream use."""
    model = MLP(tiny_layer_sizes, seed=9)
    X = rng.normal(size=(5, tiny_layer_sizes[0])).astype(np.float32)
    _, cache = model.forward(X)
    assert "final_logits" in cache, "'final_logits' key missing from forward() cache"
    logits = cache["final_logits"]
    assert logits.shape == (5, tiny_layer_sizes[-1])


def test_fused_xla_step_equivalent_to_eager(tiny_layer_sizes, rng):
    """Fused XLA step must produce the same weight update as eager mode."""
    model = MLP(tiny_layer_sizes, seed=5)
    X = rng.normal(size=(16, tiny_layer_sizes[0])).astype(np.float32)
    y = rng.integers(0, tiny_layer_sizes[-1], size=16)

    init_w = model.get_params()

    # --- Eager path: loss_and_grad + Adam wrapper step (current pattern) ---
    model.set_params(init_w)
    params_eager = [p.copy() for p in init_w]
    opt_eager = Adam(lr=0.01)
    _, grads = model.loss_and_grad(params_eager, X, y)
    opt_eager.step(params_eager, grads)

    # --- Fused XLA path: _fused_client_step directly on model variables ---
    model.set_params(init_w)
    tf_opt = build_tf_optimizer(0.01, model._model.trainable_variables)
    _ = _fused_client_step(
        tf.constant(X, dtype=tf.float32),
        tf.constant(y, dtype=tf.int32),
        model._model, tf_opt,
        tf.constant(len(y), dtype=tf.int32),
    ).numpy()
    params_fused = model.get_params()

    # Both paths start from the same weights and apply one Adam step;
    # the resulting weights must be nearly identical.
    for pa, pf in zip(params_eager, params_fused):
        np.testing.assert_allclose(pa, pf, atol=1e-5, rtol=1e-4,
                                   err_msg="Fused XLA step differs from eager")


def test_fused_xla_multiple_steps_converges(tiny_layer_sizes, rng):
    """Fused XLA training loop over multiple steps must decrease loss."""
    model = MLP(tiny_layer_sizes, seed=5)
    X = rng.normal(size=(64, tiny_layer_sizes[0])).astype(np.float32)
    y = rng.integers(0, tiny_layer_sizes[-1], size=64)

    model.set_params(model.get_params())
    tf_opt = build_tf_optimizer(0.05, model._model.trainable_variables)

    losses = []
    for _ in range(30):
        loss_val = _fused_client_step(
            tf.constant(X, dtype=tf.float32),
            tf.constant(y, dtype=tf.int32),
            model._model, tf_opt,
            tf.constant(len(y), dtype=tf.int32),
        ).numpy()
        losses.append(float(loss_val))

    assert losses[-1] < losses[0], (
        f"Fused XLA loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )


def test_build_tf_optimizer_creates_slot_variables(tiny_layer_sizes, rng):
    """build_tf_optimizer must pre-create Adam slot variables."""
    model = MLP(tiny_layer_sizes, seed=0)
    opt = build_tf_optimizer(0.01, model._model.trainable_variables)
    # After building, the optimizer should have slot variables
    opt_vars = opt.variables  # property, not method
    assert len(opt_vars) > len(model._model.trainable_variables), (
        "Optimizer should have more variables than the model "
        "(weights + momentum + velocity slots)"
    )


def test_fused_xla_step_partial_batch_no_nan(tiny_layer_sizes, rng):
    """_fused_client_step must not produce NaN when n_valid < batch_size
    and the padding region contains out-of-range labels (the root cause
    of the NaN collapse — the tf.where guard must catch it)."""
    model = MLP(tiny_layer_sizes, seed=0)
    batch_size = 16
    n_valid = 7

    X = rng.normal(size=(batch_size, tiny_layer_sizes[0])).astype(np.float32)
    y_valid = rng.integers(0, tiny_layer_sizes[-1], size=n_valid).astype(np.int32)
    y_pad = np.concatenate([y_valid, np.full(batch_size - n_valid, -1, dtype=np.int32)])

    tf_opt = build_tf_optimizer(0.01, model._model.trainable_variables)
    loss_val = _fused_client_step(
        tf.constant(X, dtype=tf.float32),
        tf.constant(y_pad, dtype=tf.int32),
        model._model, tf_opt,
        tf.constant(n_valid, dtype=tf.int32),
    ).numpy()

    assert np.isfinite(loss_val), (
        f"Loss is NaN despite tf.where guard and valid mask: {loss_val}"
    )
    params = model.get_params()
    for p in params:
        assert np.all(np.isfinite(p)), (
            f"NaN parameter after partial-batch fused step with garbage labels"
        )


def test_fused_xla_step_full_batch_no_nan(tiny_layer_sizes, rng):
    """_fused_client_step with n_valid == batch_size must also be NaN-free."""
    model = MLP(tiny_layer_sizes, seed=0)
    batch_size = 16

    X = rng.normal(size=(batch_size, tiny_layer_sizes[0])).astype(np.float32)
    y = rng.integers(0, tiny_layer_sizes[-1], size=batch_size).astype(np.int32)

    tf_opt = build_tf_optimizer(0.01, model._model.trainable_variables)
    loss_val = _fused_client_step(
        tf.constant(X, dtype=tf.float32),
        tf.constant(y, dtype=tf.int32),
        model._model, tf_opt,
        tf.constant(batch_size, dtype=tf.int32),
    ).numpy()

    assert np.isfinite(loss_val), f"Full-batch loss is NaN: {loss_val}"
    params = model.get_params()
    for p in params:
        assert np.all(np.isfinite(p)), (
            f"NaN parameter after full-batch fused step"
        )


def test_adam_decreases_loss(tiny_layer_sizes, rng):
    model = MLP(tiny_layer_sizes, seed=5)
    params = model.get_params()
    X = rng.normal(size=(64, tiny_layer_sizes[0])).astype(np.float32)
    y = rng.integers(0, tiny_layer_sizes[-1], size=64)
    opt = Adam(lr=0.05)
    start = model.loss(params, X, y)
    for _ in range(50):
        _, g = model.loss_and_grad(params, X, y)
        opt.step(params, g)
    assert model.loss(params, X, y) < start

