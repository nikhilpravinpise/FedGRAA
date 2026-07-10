"""Tests for the Keras-backed MLP: shapes, forward pass, and gradients."""

import numpy as np
import pytest
import tensorflow as tf

from fedgraa.model import MLP, Adam, aggregate, init_params


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
        np.testing.assert_allclose(a, 0.3 * b + 0.7 * c)


def test_forward_caches_final_logits(tiny_layer_sizes, rng):
    """forward() must expose final logits for downstream use."""
    model = MLP(tiny_layer_sizes, seed=9)
    X = rng.normal(size=(5, tiny_layer_sizes[0])).astype(np.float32)
    _, cache = model.forward(X)
    assert "final_logits" in cache, "'final_logits' key missing from forward() cache"
    logits = cache["final_logits"]
    assert logits.shape == (5, tiny_layer_sizes[-1])


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

