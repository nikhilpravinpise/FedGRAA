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

import warnings
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
    missing = []
    for class_id in rare:
        mask = y == class_id
        if mask.any():
            recalls.append(float((preds[mask] == class_id).mean()))
        else:
            missing.append(class_id)
    if missing:
        warnings.warn(
            f"last_n_class_accuracy: rare class id(s) {missing} have zero "
            f"samples in this evaluation split and are excluded from the "
            f"mean, which is now averaged over {len(recalls)}/{len(rare)} "
            "classes instead of the full protected set.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not recalls:
        return 0.0
    return float(np.mean(recalls))


# --------------------------------------------------------------------------- #
# Confusion matrix  (W1.4 -- FEDGRAA_V2_MASTER_PLAN.md T-1..T-3)
# --------------------------------------------------------------------------- #
def confusion_matrix(
    params: Params,
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Row-normalized-free integer confusion matrix, ``C[true, pred]``.

    A pooled confusion matrix is *additive* across clients: summing each
    client's own confusion matrix over the same ``(true, pred)`` grid gives
    exactly the pooled-population confusion matrix (T-1's additivity
    invariant) -- this is what lets per-class precision/recall/F1 (below) be
    derived identically whether computed once over pooled rows or as a sum of
    per-client matrices, which the v2 federated evaluation path relies on.
    """
    y = np.asarray(y)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    if len(y) == 0:
        return cm
    model = _get_model(params, X)
    preds = model.predict(params, X)
    np.add.at(cm, (y, preds), 1)
    return cm


def per_class_metrics_from_confusion(
    cm: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """Per-class precision/recall/F1/support derived from a pooled confusion
    matrix (T-2: macro-F1 from the pooled matrix must equal recomputing it
    directly). Only classes with nonzero support (row or column) are reported.
    """
    n = cm.shape[0]
    support = cm.sum(axis=1)
    pred_totals = cm.sum(axis=0)
    tp = np.diag(cm)
    report: Dict[int, Dict[str, float]] = {}
    for cls in range(n):
        if support[cls] == 0 and pred_totals[cls] == 0:
            continue
        precision = float(tp[cls] / pred_totals[cls]) if pred_totals[cls] > 0 else 0.0
        recall = float(tp[cls] / support[cls]) if support[cls] > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        report[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support[cls]),
        }
    return report


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

    Implemented on top of :func:`confusion_matrix` /
    :func:`per_class_metrics_from_confusion` (W1.4) so the pooled and the
    additive-over-clients computations are structurally the same code path.
    Behaviour is unchanged: classes present only as false positives (nonzero
    predicted count, zero true support) were never reported by the original
    per-``np.unique(y)`` loop either, since it iterated over classes present in
    ``y`` -- reproduced here by scoping ``n_classes`` to ``y``'s own range and
    then dropping any class absent from ``y``.
    """
    y = np.asarray(y)
    if len(y) == 0:
        return {}
    model = _get_model(params, X)
    preds = model.predict(params, X)
    # Size the matrix to cover every class id either truly present or ever
    # predicted, so a false-positive-only predicted class doesn't index out
    # of bounds; it is then dropped below to match the original semantics
    # (only classes present in y are reported).
    n_classes = int(max(y.max(), preds.max() if len(preds) else 0)) + 1
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y, preds), 1)
    full_report = per_class_metrics_from_confusion(cm)
    present = set(np.unique(y).tolist())
    return {cls: m for cls, m in full_report.items() if cls in present}


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
