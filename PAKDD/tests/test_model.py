"""Tests for the MLP: shapes, softmax, and gradient correctness."""

import numpy as np
import pytest

from fedgraa.model import MLP, Adam, aggregate, init_params, softmax


def test_forward_shapes(tiny_layer_sizes, rng):
    params = init_params(tiny_layer_sizes, seed=0)
    X = rng.normal(size=(10, tiny_layer_sizes[0]))
    probs, _ = MLP.forward(params, X)
    assert probs.shape == (10, tiny_layer_sizes[-1])
    # softmax rows sum to 1
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(10), atol=1e-10)


def test_softmax_stable_large_logits():
    logits = np.array([[1000.0, 1000.0, 1000.0]])
    p = softmax(logits)
    np.testing.assert_allclose(p, np.full((1, 3), 1 / 3), atol=1e-10)


def test_loss_matches_manual_cross_entropy(tiny_layer_sizes, rng):
    params = init_params(tiny_layer_sizes, seed=2)
    X = rng.normal(size=(7, tiny_layer_sizes[0]))
    y = rng.integers(0, tiny_layer_sizes[-1], size=7)
    probs, _ = MLP.forward(params, X)
    manual = -np.log(probs[np.arange(7), y] + 1e-12).mean()
    assert abs(MLP.loss(params, X, y) - manual) < 1e-9


def test_backprop_matches_finite_differences(tiny_layer_sizes, rng):
    """The analytic gradient must match central finite differences."""
    params = init_params(tiny_layer_sizes, seed=3)
    X = rng.normal(size=(12, tiny_layer_sizes[0]))
    y = rng.integers(0, tiny_layer_sizes[-1], size=12)

    _, grads = MLP.loss_and_grad(params, X, y)

    eps = 1e-6
    for p_idx, p in enumerate(params):
        # check a handful of entries per array
        flat = p.ravel()
        g_flat = grads[p_idx].ravel()
        for j in rng.choice(flat.size, size=min(5, flat.size), replace=False):
            orig = flat[j]
            flat[j] = orig + eps
            lp = MLP.loss(params, X, y)
            flat[j] = orig - eps
            lm = MLP.loss(params, X, y)
            flat[j] = orig
            num = (lp - lm) / (2 * eps)
            assert abs(num - g_flat[j]) < 1e-4, (p_idx, j, num, g_flat[j])


def test_aggregate_linear_combination(tiny_layer_sizes, rng):
    p0 = init_params(tiny_layer_sizes, seed=0)
    p1 = init_params(tiny_layer_sizes, seed=1)
    w = np.array([0.3, 0.7])
    agg = aggregate([p0, p1], w)
    for a, b, c in zip(agg, p0, p1):
        np.testing.assert_allclose(a, 0.3 * b + 0.7 * c)


def test_adam_decreases_loss(tiny_layer_sizes, rng):
    params = init_params(tiny_layer_sizes, seed=5)
    X = rng.normal(size=(64, tiny_layer_sizes[0]))
    y = rng.integers(0, tiny_layer_sizes[-1], size=64)
    opt = Adam(lr=0.05)
    start = MLP.loss(params, X, y)
    for _ in range(50):
        _, g = MLP.loss_and_grad(params, X, y)
        opt.step(params, g)
    assert MLP.loss(params, X, y) < start
