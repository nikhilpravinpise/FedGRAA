"""Evaluation metrics.

The paper reports two numbers per method: accuracy over **all classes** and
accuracy over the **last n** (protected rare) classes. ``last_n_class_accuracy``
computes the latter as the mean per-class recall over the rare classes, which is
what "accuracy on the rare classes" means under the paper's balanced setup.

Gap 3.6 fix: ``per_class_metrics`` and ``classification_report`` are added so
per-class precision, recall, and F1 can be inspected independently — enabling
verification of the paper's class-level accuracy claims.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

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


# --------------------------------------------------------------------------- #
# Per-class precision / recall / F1  (gap 3.6 fix)
# --------------------------------------------------------------------------- #
def per_class_metrics(
    params: Params,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """Compute per-class precision, recall, F1 and support.

    Returns a dict ``{class_id: {"precision": p, "recall": r, "f1": f1,
    "support": n}}`` over the classes present in ``y``.
    """
    y = np.asarray(y)
    if len(y) == 0:
        return {}
    preds = MLP.predict(params, X)
    report: Dict[int, Dict[str, float]] = {}
    for cls in np.unique(y):
        tp = int(((preds == cls) & (y == cls)).sum())
        fp = int(((preds == cls) & (y != cls)).sum())
        fn = int(((preds != cls) & (y == cls)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        report[int(cls)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int((y == cls).sum()),
        }
    return report


def classification_report(
    params: Params,
    X: np.ndarray,
    y: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    rare_class_ids: Optional[Sequence[int]] = None,
) -> str:
    """scikit-learn-style per-class and macro-averaged classification report.

    Classes listed in ``rare_class_ids`` are marked with an asterisk so they
    stand out in the output.
    """
    report = per_class_metrics(params, X, y)
    if not report:
        return "No samples."
    rare_set = set(rare_class_ids) if rare_class_ids else set()
    header = (
        f"{'Class':>22s} | {'Precision':>9s} | {'Recall':>9s} | "
        f"{'F1':>9s} | {'Support':>7s}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for cls, m in sorted(report.items()):
        marker = "*" if cls in rare_set else " "
        name = (class_names[cls] if class_names and cls < len(class_names)
                else str(cls))
        lines.append(
            f"{marker}{name:>21s} | {m['precision']:>9.4f} | {m['recall']:>9.4f} "
            f"| {m['f1']:>9.4f} | {m['support']:>7d}"
        )
    n = len(report)
    macro_p = sum(m["precision"] for m in report.values()) / n
    macro_r = sum(m["recall"] for m in report.values()) / n
    macro_f1 = sum(m["f1"] for m in report.values()) / n
    total = sum(m["support"] for m in report.values())
    lines += [
        sep,
        f"{'macro avg':>22s} | {macro_p:>9.4f} | {macro_r:>9.4f} "
        f"| {macro_f1:>9.4f} | {total:>7d}",
    ]
    return "\n".join(lines)
