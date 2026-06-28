"""Dataset loading and preprocessing.

Two paths are supported:

* :func:`load_dataset` reads the real CIC-IDS-2017 / UNSW-NB15 CSV exports from a
  directory, applies the standard IDS cleaning pipeline (drop inf/NaN,
  label-encode, standard-scale) and caps frequent classes at 10,000 exactly as
  the paper does.
* :func:`make_synthetic_dataset` fabricates a class-separable dataset whose
  per-class counts reproduce the "Used" column of Tables 1 and 2, so the full
  federated pipeline and every experiment run end-to-end with no downloads.

Both return ``(X, y, spec)`` with ``X`` standardized float64 features and ``y``
integer class labels aligned to ``spec.labels``.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DatasetSpec, get_dataset_spec


# --------------------------------------------------------------------------- #
# Synthetic data (default, no external files needed)
# --------------------------------------------------------------------------- #
def make_synthetic_dataset(
    spec: DatasetSpec,
    seed: int = 0,
    class_separation: float = 2.5,
) -> Tuple[np.ndarray, np.ndarray, DatasetSpec]:
    """Generate features that respect the exact per-class "Used" counts.

    Each class gets a random mean vector; samples are Gaussian around it. This
    yields linearly-separable-ish clusters so the network can genuinely learn,
    making the FedAvg-fails / FedGRAA-wins behaviour observable without real
    data. Counts equal ``spec.used_counts`` (already capped at 10,000).
    """
    rng = np.random.default_rng(seed)
    n_features = spec.n_features
    # A distinct, well-separated mean per class.
    class_means = rng.normal(0.0, class_separation, size=(spec.n_classes, n_features))

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    for class_id, count in enumerate(spec.used_counts):
        if count <= 0:
            continue
        samples = rng.normal(
            loc=class_means[class_id], scale=1.0, size=(count, n_features)
        )
        X_parts.append(samples)
        y_parts.append(np.full(count, class_id, dtype=int))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    # Shuffle then standard-scale (zero mean, unit variance per feature).
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    X = _standard_scale(X)
    return X, y, spec


# --------------------------------------------------------------------------- #
# Real CSV loader
# --------------------------------------------------------------------------- #
def load_dataset(
    name: str,
    data_dir: Optional[str] = None,
    label_column: Optional[str] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, DatasetSpec]:
    """Load a dataset by name.

    If ``data_dir`` is given and contains CSV files, the real data is loaded and
    preprocessed. Otherwise a synthetic dataset matching the paper's counts is
    returned (with a clear runtime notice handled by the caller).
    """
    spec = get_dataset_spec(name)
    if data_dir and os.path.isdir(data_dir):
        csvs = [
            os.path.join(data_dir, f)
            for f in sorted(os.listdir(data_dir))
            if f.lower().endswith(".csv")
        ]
        if csvs:
            return _load_real_csvs(csvs, spec, label_column=label_column, seed=seed)
    return make_synthetic_dataset(spec, seed=seed)


def _load_real_csvs(
    csv_paths: List[str],
    spec: DatasetSpec,
    label_column: Optional[str],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, DatasetSpec]:
    frames = [pd.read_csv(p, skipinitialspace=True) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]

    if label_column is None:
        # CIC files use "Label"; UNSW exports often use "attack_cat"/"label".
        for candidate in ("Label", "label", "attack_cat", "Attack", "class"):
            if candidate in df.columns:
                label_column = candidate
                break
    if label_column is None or label_column not in df.columns:
        raise ValueError(
            "Could not infer the label column; pass label_column explicitly."
        )

    y_raw = df[label_column].astype(str).str.strip()

    # Drop the label column plus other well-known non-feature columns so they
    # never leak into X (UNSW ships an 'id' and a binary 'label' alongside
    # 'attack_cat'; some CIC exports carry flow identifiers).
    drop_cols = {label_column}
    for extra in ("id", "label", "attack_cat", "Label", "Flow ID", "Timestamp"):
        if extra in df.columns:
            drop_cols.add(extra)
    X_df = df.drop(columns=list(drop_cols))

    # Keep numeric features only; clean inf / NaN (CIC-IDS contains both).
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(0.0)
    X = X_df.to_numpy(dtype=np.float64)

    # Map textual labels onto the spec's label ordering, tolerating the real
    # datasets' spelling variants (e.g. CIC "Web Attack - Brute Force").
    y = _match_labels(y_raw, spec)
    keep = y >= 0
    X, y = X[keep], y[keep]

    X, y = _cap_frequent_classes(X, y, spec, seed=seed)
    X = _standard_scale(X)
    return X, y, spec


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _normalize_label(text: str) -> Tuple[str, str]:
    """Return ``(normalized, singular)`` forms of a label string.

    Lower-cases, strips punctuation, collapses whitespace, drops a leading
    "web attack" prefix and a trailing plural 's' so real-world label spellings
    line up with the paper's class names.
    """
    import re

    t = text.strip().lower()
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    if t.startswith("web attack"):
        t = t[len("web attack"):].strip()
    singular = t[:-1] if (t.endswith("s") and len(t) > 3) else t
    return t, singular


# Explicit aliases (normalized form -> paper label) for cases normalization
# alone cannot resolve.
_LABEL_ALIASES = {
    "brute force": "Brute Force",
    "sql injection": "SQL Injection",
    "xss": "XSS",
    "backdoors": "Backdoor",
    "backdoor": "Backdoor",
    "dos hulk": "DoS Hulk",
    "benign": "BENIGN",
    "normal": "Normal",
}


def _match_labels(y_raw, spec: DatasetSpec) -> np.ndarray:
    """Map raw label strings onto ``spec`` class IDs, returning -1 for misses."""
    # Build a lookup keyed by normalized spec labels.
    norm_to_id = {}
    for i, lbl in enumerate(spec.labels):
        full, singular = _normalize_label(lbl)
        norm_to_id.setdefault(full, i)
        norm_to_id.setdefault(singular, i)
    # Fold in the explicit aliases that exist in this spec.
    label_index = {lbl: i for i, lbl in enumerate(spec.labels)}
    for alias_norm, target in _LABEL_ALIASES.items():
        if target in label_index:
            norm_to_id.setdefault(alias_norm, label_index[target])

    ids = np.empty(len(y_raw), dtype=int)
    for j, raw in enumerate(y_raw):
        full, singular = _normalize_label(str(raw))
        ids[j] = norm_to_id.get(full, norm_to_id.get(singular, -1))
    return ids


def _standard_scale(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std


def _cap_frequent_classes(
    X: np.ndarray, y: np.ndarray, spec: DatasetSpec, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Cap each class at ``spec.frequent_cap`` samples (Section 5)."""
    rng = np.random.default_rng(seed)
    keep_idx: List[np.ndarray] = []
    for class_id in range(spec.n_classes):
        idx = np.flatnonzero(y == class_id)
        if len(idx) > spec.frequent_cap:
            idx = rng.choice(idx, size=spec.frequent_cap, replace=False)
        keep_idx.append(idx)
    sel = np.concatenate(keep_idx)
    rng.shuffle(sel)
    return X[sel], y[sel]
