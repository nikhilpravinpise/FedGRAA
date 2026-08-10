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
from typing import List, Literal, Tuple

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
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random 90/10 train/validation split used by the paper."""
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

    For every attack class the client receives all of that class’s samples plus
    an equal number of benign samples drawn from the shared benign pool. Each
    client is then split 90/10 into train/validation.

    Gap 2.3 fix: benign samples are allocated sequentially from a single
    pre-shuffled pool (circular indexing) rather than by independent random
    draws. This minimises cross-client overlap and makes allocation deterministic.

    Gap M-3: if total benign demand exceeds the benign pool, a warning is issued
    because some benign samples will be shared across clients.
    """
    rng = np.random.default_rng(seed)

    # Pre-compute attack-class index arrays once.
    attack_idx_map: dict = {
        aid: np.flatnonzero(y == aid) for aid in spec.attack_ids
    }

    # --- Disjoint benign allocation (gap 2.3 fix) ------------------------ #
    benign_idx_all = np.flatnonzero(y == spec.benign_id)
    benign_pool = rng.permutation(benign_idx_all)
    pool_size = len(benign_pool)

    total_attack_demand = sum(len(attack_idx_map[aid]) for aid in spec.attack_ids)
    if total_attack_demand > pool_size:
        import warnings
        warnings.warn(
            f"Benign pool ({pool_size:,}) is smaller than total attack demand "
            f"({total_attack_demand:,}). {total_attack_demand - pool_size:,} "
            f"benign samples will be reused across clients (gap M-3).",
            RuntimeWarning,
            stacklevel=2,
        )

    benign_alloc: dict = {}
    cursor = 0
    for aid in spec.attack_ids:
        demand = len(attack_idx_map[aid])
        if demand == 0 or pool_size == 0:
            benign_alloc[aid] = np.empty(0, dtype=int)
        else:
            # Circular indexing: wrap around the pool rather than drawing
            # independently, so each additional sample is unique until the
            # pool is exhausted, then samples repeat in a deterministic order.
            indices = np.arange(cursor, cursor + demand) % pool_size
            benign_alloc[aid] = benign_pool[indices]
        cursor += demand
    # --------------------------------------------------------------------- #

    clients: List[ClientShard] = []
    val_X_parts: List[np.ndarray] = []
    val_y_parts: List[np.ndarray] = []

    for client_id, attack_id in enumerate(spec.attack_ids):
        attack_idx = attack_idx_map[attack_id]
        n_attack = len(attack_idx)
        if n_attack == 0:
            continue

        # Equal number of benign samples (Section 5, Data Settings).
        benign_pick = benign_alloc[attack_id]

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

    # Feature scaling: the paper does not specify it. The single committed
    # default (global standard scaling on the full pooled data) is applied
    # exactly once, upstream of this function, by the data loader
    # (``data.py::_load_real_csvs`` / ``make_synthetic_dataset``) before
    # partitioning ever runs. Re-scaling here a second time -- using only the
    # post-split validation pool's statistics -- would silently double-
    # normalize every feature and leak validation-only statistics into the
    # training data a second, narrower way. No further scaling happens here.
    rare_class_ids = spec.rare_client_ids(n_rare)
    val_split = median_frequency_split(X_val, y_val, rare_class_ids)

    return FederatedSplit(
        spec=spec,
        clients=clients,
        X_val=X_val,
        y_val=y_val,
        val_split=val_split,
    )


def _infer_n_classes(y: np.ndarray, rare_class_ids: List[int]) -> int:
    """Smallest class-count that covers every id appearing in ``y`` or
    ``rare_class_ids`` -- exactly enough to size a histogram/tag array so no
    id ever indexes out of bounds (F3's zero-held-out-rows corner: a rare
    class id can be present in ``rare_class_ids`` while absent from ``y``)."""
    hi = -1
    if len(y):
        hi = max(hi, int(np.max(y)))
    if len(rare_class_ids):
        hi = max(hi, int(max(rare_class_ids)))
    return hi + 1


def stratum_tags(counts: np.ndarray, rare_class_ids: List[int]) -> np.ndarray:
    """Server-side, histogram-only length-``C`` tag vector in ``{'h','m','n'}``.

    Carries **all** of :func:`median_frequency_split`'s arithmetic -- the
    median-frequency threshold and the gap-2.6 non-empty-``D_V^h`` fallback --
    but consumes a SecAgg-summed *count* vector instead of raw per-row labels
    (F2). A client can therefore tag its own validation rows from this
    broadcast vector alone, without ever seeing another client's rows: the
    global partition decision (which classes are "highly frequent") depends
    only on the pooled histogram, not on any individual row.

    ``counts[c]`` is the number of rows of class ``c`` in the population this
    split is computed over (e.g. the pooled held-out set, or the pooled full
    shard) -- callers needing the ``D_V`` split pass the held-out histogram;
    the rank histogram (F3's ``h_k^rank``) uses this function only indirectly,
    via :func:`fedgraa.config.DatasetSpec.rare_client_ids`'s ``BottomN``.
    """
    C = len(counts)
    tags = np.full(C, "m", dtype="<U1")
    rare = {int(r) for r in rare_class_ids}
    present = [c for c in range(C) if counts[c] > 0 and c not in rare]
    for r in rare:
        if 0 <= r < C:
            tags[r] = "n"
    if not present:
        return tags

    freqs = np.array([counts[c] for c in present], dtype=float)
    median_freq = float(np.median(freqs))
    high = {c for c, f in zip(present, freqs) if f > median_freq}

    # Gap 2.6 fix: if no class strictly exceeds the median, designate the
    # upper half (by count) as "high frequency" so D_V^h is never empty when
    # there are at least two non-rare classes present.
    if not high and len(present) > 1:
        order = [present[i] for i in np.argsort(freqs, kind="stable")]
        high = set(order[len(order) // 2:])

    for c in high:
        tags[c] = "h"
    return tags


def apply_tags(
    X: np.ndarray, y: np.ndarray, tags: np.ndarray, rare_class_ids: List[int]
) -> ValidationSplit:
    """Client-side: split *one client's own rows* using the broadcast tags.

    Pure and local (F2): needs only ``tags`` (broadcast by the server, derived
    from the pooled histogram via :func:`stratum_tags`) and this client's own
    ``(X, y)`` -- no other client's rows and no global label distribution.
    """
    lab = tags[y] if len(y) else np.empty(0, dtype="<U1")
    m_h, m_m, m_n = (lab == "h"), (lab == "m"), (lab == "n")
    return ValidationSplit(
        X[m_h], y[m_h], X[m_m], y[m_m], X[m_n], y[m_n], list(rare_class_ids)
    )


def median_frequency_split(
    X_val: np.ndarray, y_val: np.ndarray, rare_class_ids: List[int]
) -> ValidationSplit:
    """Split ``D_V`` into ``D_V^h`` / ``D_V^m`` / ``D_V^n``.

    ``D_V^n`` holds the protected rare classes. Among the *remaining* classes,
    the per-class sample frequency is computed and the **median** is used as
    the threshold: classes with frequency strictly above the median go to
    ``D_V^h`` (highly frequent), the rest go to ``D_V^m`` (moderately
    frequent).

    F2 / W1.1: re-expressed as the composition of :func:`stratum_tags`
    (server-side, histogram-only) and :func:`apply_tags` (client-side, given
    the broadcast tags) so the pooled (v1) computation and the per-client (v2)
    computation are structurally the same code path, not two implementations
    that must be kept in sync by hand. Every existing caller and test is
    unaffected -- this function's signature and behaviour are unchanged.
    """
    C = _infer_n_classes(y_val, rare_class_ids)
    counts = np.bincount(y_val, minlength=C).astype(np.int64) if C else np.zeros(0, dtype=np.int64)
    tags = stratum_tags(counts, rare_class_ids)
    return apply_tags(X_val, y_val, tags, rare_class_ids)


def local_label_histogram(
    shard: ClientShard,
    n_classes: int,
    scope: Literal["full", "heldout"] = "full",
) -> np.ndarray:
    """One client's label-count vector, the SecAgg-summed input to F3's two
    rarity histograms.

    ``scope="full"`` sums the client's **entire local shard** (train + held-out)
    -- feeds ``h_k^rank`` -> ``BottomN`` -> the rare-class ranking, chosen for
    its *determinism* (exactly ``used_counts``, zero seed variance -- F3).
    ``scope="heldout"`` sums only the client's held-out split -- feeds
    ``h_k^strat`` -> :func:`stratum_tags`, which must match
    ``median_frequency_split``'s pooled-``y_val`` counts exactly.
    """
    if scope == "full":
        y = np.concatenate([shard.y_train, shard.y_val])
    elif scope == "heldout":
        y = shard.y_val
    else:
        raise ValueError(f"scope must be 'full' or 'heldout', got {scope!r}")
    return np.bincount(y, minlength=n_classes).astype(np.int64)


def partition_by_silo(
    silos: List[Tuple[np.ndarray, np.ndarray, str]],
    spec: DatasetSpec,
    n_rare: int = 3,
    val_fraction: float = 0.10,
    seed: int = 0,
) -> FederatedSplit:
    """Build a federated split from **pre-existing, real** data silos (W6,
    FEDGRAA_V2_MASTER_PLAN.md §7.4, Option D) -- e.g. NF-UQ-NIDS-v2's four
    independently-collected NetFlow datasets, each treated as one client.

    This is the real-non-IID counterpart to :func:`partition_non_iid`'s
    synthetic one-attack-per-client construction. Under
    :func:`partition_non_iid`, "the server does not know client label
    distributions" is a claim about a simulation the experimenter designed;
    under a genuine multi-silo dataset, each client's label distribution is
    whatever that silo's original data collection produced, so the same
    privacy claim becomes a statement about a real deployment rather than an
    assumption about the partition.

    Each element of ``silos`` is ``(X_silo, y_silo, silo_name)`` -- already
    loaded, cleaned, and label-mapped onto ``spec``'s class ids (mirroring
    the numeric-feature/label-mapping pipeline :func:`fedgraa.data._load_real_csvs`
    already applies per-file). One :class:`ClientShard` is built per silo
    (``client_id`` = silo index, ``attack_label`` = ``silo_name``, since a
    silo is not restricted to a single attack class the way
    :func:`partition_non_iid`'s synthetic clients are); shards are then split
    90/10 exactly as :func:`partition_non_iid` does, and the pooled
    validation set is stratified by :func:`median_frequency_split` unchanged.

    Unlike :func:`partition_non_iid`, ``attack_id`` on the resulting
    :class:`ClientShard` is set to ``-1`` (a silo is not one attack class) --
    callers that key off ``attack_id`` (none currently do outside the
    synthetic-partition path) must not assume it is meaningful here.
    """
    if not silos:
        raise ValueError("partition_by_silo requires at least one silo")

    rng = np.random.default_rng(seed)
    clients: List[ClientShard] = []
    val_X_parts: List[np.ndarray] = []
    val_y_parts: List[np.ndarray] = []

    for client_id, (X_silo, y_silo, silo_name) in enumerate(silos):
        Xtr, ytr, Xval, yval = _train_val_split(X_silo, y_silo, val_fraction, rng)
        clients.append(
            ClientShard(
                client_id=client_id,
                attack_id=-1,
                attack_label=silo_name,
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
        spec=spec, clients=clients, X_val=X_val, y_val=y_val, val_split=val_split,
    )


def bottom_n_rare_classes(
    counts: np.ndarray, attack_ids: List[int], n: int
) -> List[int]:
    """The ``n`` rarest attack classes by count -- ``BottomN(p_hat, n)``
    (F3/F7): a SecAgg-summed, data-driven replacement for
    :meth:`fedgraa.config.DatasetSpec.rare_client_ids`'s hard-coded
    ``rare_label_order`` oracle. Restricted to ``attack_ids`` so the always-
    common Benign/Normal class can never be selected as "rare." Ties break by
    ascending class id (Python's stable ``sorted``), matching
    ``DatasetSpec.rare_client_ids``'s own non-oracle fallback path exactly
    (F7) -- this is the same computation, just fed a SecAgg-summed histogram
    instead of ``used_counts``.
    """
    return sorted(attack_ids, key=lambda i: counts[i])[:n]
