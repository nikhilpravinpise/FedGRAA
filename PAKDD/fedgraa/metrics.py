"""Evaluation metrics.

The paper reports two numbers per method: accuracy over **all classes** and
accuracy over the **last n** (protected rare) classes. ``last_n_class_accuracy``
computes the latter as the mean per-class recall over the rare classes, which is
what "accuracy on the rare classes" means under the paper's balanced setup.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .model import MLP, Params


def accuracy(params: Params, X: np.ndarray, y: np.ndarray) -> float:
    """Overall classification accuracy across all classes."""
    if len(y) == 0:
        return 0.0
    preds = MLP.predict(params, X)
    return float((preds == np.asarray(y)).mean())


def last_n_class_accuracy(
    params: Params,
    X: np.ndarray,
    y: np.ndarray,
    rare_class_ids: Sequence[int],
) -> float:
    """Mean per-class recall over the protected rare classes.

    For each rare class we measure the fraction of its samples classified
    correctly, then average over the rare classes that are present. This matches
    the paper's "(Last n classes)" accuracy column.
    """
    y = np.asarray(y)
    rare = list(rare_class_ids)
    if len(y) == 0 or not rare:
        return 0.0
    preds = MLP.predict(params, X)
    recalls = []
    for class_id in rare:
        mask = y == class_id
        if mask.any():
            recalls.append(float((preds[mask] == class_id).mean()))
    if not recalls:
        return 0.0
    return float(np.mean(recalls))
