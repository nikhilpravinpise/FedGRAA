"""Non-IID data partitioning and validation-set splitting.

Implements the data setup of Section 5 exactly:

* **One attack type per client**, paired with an *equal number* of Benign /
  Normal samples (``partition_non_iid``).
* **90% train / 10% validation** on each client; the validation slices together
  form the server-side validation set ``D_V``.
* ``D_V`` is split into the ``n`` protected rare classes ``D_V^n`` and the
  remaining classes, which are divided by **median frequency** into the highly
  frequent ``D_V^h`` (above median) and moderately frequent ``D_V^m`` (below /
  at median) — see :func:`median_frequency_split`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .config import DatasetSpec


@dataclass
class ClientShard:
    """One client: a single attack type plus an equal number of benign rows."""

    client_id: int
    attack_id: int
    attack_label: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray

    @property
    def n_train(self) -> int:
        return len(self.y_train)


@dataclass
class ValidationSplit:
    """The median-frequency partition of ``D_V`` used by FedGRAA (eq. 8-9)."""

    X_h: np.ndarray  # highly frequent (above median)
    y_h: np.ndarray
    X_m: np.ndarray  # moderately frequent (below/at median)
    y_m: np.ndarray
    X_n: np.ndarray  # the n protected rare classes
    y_n: np.ndarray
    rare_class_ids: List[int]


@dataclass
class FederatedSplit:
    """Everything the trainer needs: clients, the full ``D_V``, and its split."""

    spec: DatasetSpec
    clients: List[ClientShard]
    X_val: np.ndarray
    y_val: np.ndarray
    val_split: ValidationSplit

    @property
    def n_clients(self) -> int:
        return len(self.clients)


def _train_val_split(
    X: np.ndarray, y: np.ndarray, val_fraction: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    n_val = max(1, int(round(n * val_fraction))) if n > 1 else 0
    perm = rng.permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    if len(train_idx) == 0:  # tiny class: keep at least one training sample
        train_idx, val_idx = perm[:1], perm[1:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def partition_non_iid(
    X: np.ndarray,
    y: np.ndarray,
    spec: DatasetSpec,
    n_rare: int = 3,
    val_fraction: float = 0.10,
    seed: int = 0,
) -> FederatedSplit:
    """Build the one-attack-per-client federated split.

    For every attack class the client receives all of that class's samples plus
    an equal number of benign samples drawn from the shared benign pool. Each
    client is then split 90/10 into train/validation.
    """
    rng = np.random.default_rng(seed)
    benign_idx_all = np.flatnonzero(y == spec.benign_id)

    clients: List[ClientShard] = []
    val_X_parts: List[np.ndarray] = []
    val_y_parts: List[np.ndarray] = []

    for client_id, attack_id in enumerate(spec.attack_ids):
        attack_idx = np.flatnonzero(y == attack_id)
        n_attack = len(attack_idx)
        if n_attack == 0:
            continue

        # Equal number of benign samples (Section 5, Data Settings).
        n_benign = min(n_attack, len(benign_idx_all))
        replace = n_benign > len(benign_idx_all)
        benign_pick = rng.choice(benign_idx_all, size=n_benign, replace=replace)

        idx = np.concatenate([attack_idx, benign_pick])
        rng.shuffle(idx)
        Xc, yc = X[idx], y[idx]

        Xtr, ytr, Xval, yval = _train_val_split(Xc, yc, val_fraction, rng)
        clients.append(
            ClientShard(
                client_id=client_id,
                attack_id=attack_id,
                attack_label=spec.labels[attack_id],
                X_train=Xtr,
                y_train=ytr,
                X_val=Xval,
                y_val=yval,
            )
        )
        val_X_parts.append(Xval)
        val_y_parts.append(yval)

    X_val = np.concatenate(val_X_parts, axis=0)
    y_val = np.concatenate(val_y_parts, axis=0)

    rare_class_ids = spec.rare_client_ids(n_rare)
    val_split = median_frequency_split(X_val, y_val, rare_class_ids)

    return FederatedSplit(
        spec=spec,
        clients=clients,
        X_val=X_val,
        y_val=y_val,
        val_split=val_split,
    )


def median_frequency_split(
    X_val: np.ndarray, y_val: np.ndarray, rare_class_ids: List[int]
) -> ValidationSplit:
    """Split ``D_V`` into ``D_V^h`` / ``D_V^m`` / ``D_V^n``.

    ``D_V^n`` holds the protected rare classes. Among the *remaining* classes,
    the per-class sample frequency is computed and the **median** is used as the
    threshold: classes with frequency strictly above the median go to ``D_V^h``
    (highly frequent), the rest go to ``D_V^m`` (moderately frequent).
    """
    rare_set = set(rare_class_ids)
    is_rare = np.isin(y_val, list(rare_set))

    X_n, y_n = X_val[is_rare], y_val[is_rare]
    X_rest, y_rest = X_val[~is_rare], y_val[~is_rare]

    if len(y_rest) == 0:
        empty_X = X_val[:0]
        empty_y = y_val[:0]
        return ValidationSplit(
            empty_X, empty_y, empty_X, empty_y, X_n, y_n, list(rare_class_ids)
        )

    classes, counts = np.unique(y_rest, return_counts=True)
    median_freq = float(np.median(counts))
    freq_by_class = dict(zip(classes.tolist(), counts.tolist()))

    high_classes = {c for c, f in freq_by_class.items() if f > median_freq}

    is_high = np.isin(y_rest, list(high_classes)) if high_classes else np.zeros(
        len(y_rest), dtype=bool
    )
    X_h, y_h = X_rest[is_high], y_rest[is_high]
    X_m, y_m = X_rest[~is_high], y_rest[~is_high]

    return ValidationSplit(
        X_h=X_h,
        y_h=y_h,
        X_m=X_m,
        y_m=y_m,
        X_n=X_n,
        y_n=y_n,
        rare_class_ids=list(rare_class_ids),
    )
