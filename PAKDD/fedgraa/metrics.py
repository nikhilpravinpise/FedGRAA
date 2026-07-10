"""Evaluation metrics.

The paper reports two numbers per method: accuracy over **all classes** and
accuracy over the **last n** (protected rare) classes. ``last_n_class_accuracy``
computes the latter as the mean per-class recall over the rare classes.

The paper never defines "accuracy on the last n classes" mathematically; we
commit to mean-per-class recall as the interpretation for strict reproduction.
This treats every protected rare class equally, matching the experiment's stated
purpose of helping minority clients regardless of their absolute sample count.

Gap 3.6 fix: ``per_class_metrics`` and ``classification_report`` are added so
per-class precision, recall, and F1 can be inspected independently — enabling
verification of the paper's class-level accuracy claims.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .model import MLP, Params


def _get_model(params: Params, X: np.ndarray) -> MLP:
    """Build an MLP from params shapes when no model instance is available."""
    from .model import _infer_layer_sizes

    return MLP(_infer_layer_sizes(params, X.shape[1]), seed=0)


def accuracy(
    model_or_params: MLP | Params,
    params_or_X: Params | np.ndarray,
    X_or_y: np.ndarray,
    y_or_none: Optional[np.ndarray] = None,
) -> float:
    """Overall classification accuracy across all classes.

    Supports two calling conventions:

    * ``accuracy(model, params, X, y)`` — efficient, used by the trainer.
    * ``accuracy(params, X, y)`` — convenience, builds a temporary MLP.
    """
    if y_or_none is None:
        # Backward-compatible form: accuracy(params, X, y)
        params = model_or_params  # type: ignore[assignment]
        X = params_or_X  # type: ignore[assignment]
        y = X_or_y
        model = _get_model(params, X)
        preds = model.predict(params, X)
    else:
        model = model_or_params  # type: ignore[assignment]
        params = params_or_X  # type: ignore[assignment]
        X = X_or_y
        y = y_or_none
        preds = model.predict(params, X)

    if len(y) == 0:
        return 0.0
    return float((preds == np.asarray(y)).mean())


def last_n_class_accuracy(
    model_or_params: MLP | Params,
    params_or_X: Params | np.ndarray,
    X_or_y: np.ndarray,
    y_or_rare: Optional[np.ndarray | Sequence[int]] = None,
    rare_class_ids: Optional[Sequence[int]] = None,
) -> float:
    """Accuracy over the protected rare classes.

    Computes mean per-class recall: for each rare class we measure the fraction
    of its samples classified correctly, then average over the rare classes that
    are present. This is the single interpretation used for strict reproduction.

    Supports two calling conventions:

    * ``last_n_class_accuracy(model, params, X, y, rare_class_ids)`` — efficient.
    * ``last_n_class_accuracy(params, X, y, rare_class_ids)`` — convenience.
    """
    # Distinguish the two calling conventions.  The 5-argument trainer form
    # passes y as the 4th positional argument and rare_class_ids either as the
    # 5th positional argument or as a keyword.  The 4-argument convenience form
    # passes rare_class_ids as the 4th positional argument.
    if isinstance(model_or_params, MLP):
        model = model_or_params
        params = params_or_X  # type: ignore[assignment]
        X = X_or_y
        if rare_class_ids is None:
            raise TypeError(
                "last_n_class_accuracy(model, params, X, y, rare_class_ids) "
                "missing rare_class_ids"
            )
        y = y_or_rare
    else:
        params = model_or_params  # type: ignore[assignment]
        X = params_or_X  # type: ignore[assignment]
        y = X_or_y
        if rare_class_ids is None:
            if y_or_rare is None:
                raise TypeError(
                    "last_n_class_accuracy(params, X, y, rare_class_ids) "
                    "missing rare_class_ids"
                )
            rare_class_ids = y_or_rare
        model = _get_model(params, X)

    y = np.asarray(y)
    rare = list(rare_class_ids)
    if len(y) == 0 or not rare:
        return 0.0
    preds = model.predict(params, X)

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
    model = _get_model(params, X)
    preds = model.predict(params, X)
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
