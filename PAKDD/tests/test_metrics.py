"""Tests for evaluation metrics."""

import numpy as np
import pytest

from fedgraa.metrics import (
    accuracy,
    classification_report,
    last_n_class_accuracy,
    per_class_metrics,
)
from fedgraa.model import init_params


def test_accuracy_perfect_and_zero(monkeypatch):
    from fedgraa import metrics

    X = np.zeros((6, 3))
    y = np.array([0, 1, 2, 0, 1, 2])

    monkeypatch.setattr(metrics.MLP, "predict", staticmethod(lambda p, X: y.copy()))
    params = init_params((3, 4, 3), seed=0)
    assert accuracy(params, X, y) == 1.0

    monkeypatch.setattr(
        metrics.MLP, "predict", staticmethod(lambda p, X: np.zeros(len(X), dtype=int))
    )
    assert accuracy(params, X, y) == pytest.approx(2 / 6)


def test_last_n_class_accuracy_is_mean_recall(monkeypatch):
    from fedgraa import metrics

    # 3 rare classes {2,3,4}; predictions correct for class 2 and 4 only.
    y = np.array([2, 2, 3, 3, 4, 4, 0, 1])
    preds = np.array([2, 2, 0, 0, 4, 4, 0, 1])  # class 3 fully wrong
    X = np.zeros((len(y), 3))

    monkeypatch.setattr(
        metrics.MLP, "predict", staticmethod(lambda p, X: preds.copy())
    )
    params = init_params((3, 4, 5), seed=0)
    acc = last_n_class_accuracy(params, X, y, rare_class_ids=[2, 3, 4])
    # recalls: class2=1.0, class3=0.0, class4=1.0 -> mean = 2/3
    assert acc == pytest.approx(2 / 3)


def test_empty_inputs_return_zero():
    params = init_params((3, 4, 3), seed=0)
    X = np.zeros((0, 3))
    y = np.zeros(0, dtype=int)
    assert accuracy(params, X, y) == 0.0
    assert last_n_class_accuracy(params, X, y, [0]) == 0.0


# --------------------------------------------------------------------------- #
# Gap 3.6: per-class precision / recall / F1
# --------------------------------------------------------------------------- #

def test_per_class_metrics_perfect_predictions(monkeypatch):
    """Perfect predictions should give precision=recall=F1=1 for every class."""
    from fedgraa import metrics

    y = np.array([0, 0, 1, 1, 2, 2])
    X = np.zeros((len(y), 3))
    monkeypatch.setattr(
        metrics.MLP, "predict", staticmethod(lambda p, X: y.copy())
    )
    params = init_params((3, 4, 3), seed=0)
    report = per_class_metrics(params, X, y)

    for cls in [0, 1, 2]:
        assert report[cls]["precision"] == pytest.approx(1.0)
        assert report[cls]["recall"] == pytest.approx(1.0)
        assert report[cls]["f1"] == pytest.approx(1.0)
        assert report[cls]["support"] == 2


def test_per_class_metrics_all_wrong(monkeypatch):
    """All-wrong predictions should give precision=recall=F1=0 for every class."""
    from fedgraa import metrics

    y = np.array([0, 0, 1, 1])
    preds = np.array([1, 1, 0, 0])   # every sample misclassified
    X = np.zeros((len(y), 3))
    monkeypatch.setattr(
        metrics.MLP, "predict", staticmethod(lambda p, X: preds.copy())
    )
    params = init_params((3, 4, 2), seed=0)
    report = per_class_metrics(params, X, y)

    for cls in [0, 1]:
        assert report[cls]["precision"] == pytest.approx(0.0)
        assert report[cls]["recall"] == pytest.approx(0.0)
        assert report[cls]["f1"] == pytest.approx(0.0)


def test_per_class_metrics_empty_returns_empty():
    params = init_params((3, 4, 3), seed=0)
    report = per_class_metrics(params, np.zeros((0, 3)), np.zeros(0, dtype=int))
    assert report == {}


def test_classification_report_contains_macro_avg(monkeypatch):
    """classification_report must include a macro-average row."""
    from fedgraa import metrics

    y = np.array([0, 1, 2])
    monkeypatch.setattr(
        metrics.MLP, "predict", staticmethod(lambda p, X: y.copy())
    )
    params = init_params((3, 4, 3), seed=0)
    report_str = classification_report(
        params, np.zeros((3, 3)), y,
        class_names=["Benign", "Attack1", "Attack2"],
        rare_class_ids=[2],
    )
    assert "macro avg" in report_str
    assert "*" in report_str  # rare class marked with asterisk
