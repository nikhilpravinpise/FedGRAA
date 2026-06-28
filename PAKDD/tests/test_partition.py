"""Tests for the non-IID partition and median-frequency validation split."""

import numpy as np

from fedgraa.config import CIC_IDS_2017
from fedgraa.data import make_synthetic_dataset
from fedgraa.partition import median_frequency_split, partition_non_iid


def test_one_attack_per_client_with_equal_benign():
    X, y, spec = make_synthetic_dataset(CIC_IDS_2017, seed=0)
    split = partition_non_iid(X, y, spec, n_rare=3, val_fraction=0.10, seed=0)

    assert split.n_clients == spec.n_clients  # 14

    for client in split.clients:
        labels_present = set(np.unique(client.y_train).tolist()) | set(
            np.unique(client.y_val).tolist()
        )
        # exactly the attack class and benign
        assert labels_present <= {client.attack_id, spec.benign_id}
        # equal number of attack and benign across the whole client shard
        all_y = np.concatenate([client.y_train, client.y_val])
        n_attack = int((all_y == client.attack_id).sum())
        n_benign = int((all_y == spec.benign_id).sum())
        assert n_attack == n_benign


def test_train_val_ratio_roughly_90_10():
    X, y, spec = make_synthetic_dataset(CIC_IDS_2017, seed=0)
    split = partition_non_iid(X, y, spec, n_rare=3, val_fraction=0.10, seed=0)
    # pick a client with enough samples to measure the ratio reliably
    big = max(split.clients, key=lambda c: c.n_train)
    total = big.n_train + len(big.y_val)
    val_frac = len(big.y_val) / total
    assert 0.05 < val_frac < 0.15


def test_median_split_buckets_are_disjoint_and_complete():
    X, y, spec = make_synthetic_dataset(CIC_IDS_2017, seed=0)
    split = partition_non_iid(X, y, spec, n_rare=3, val_fraction=0.10, seed=0)
    vs = split.val_split

    n_total = len(split.y_val)
    n_buckets = len(vs.y_h) + len(vs.y_m) + len(vs.y_n)
    assert n_buckets == n_total

    # rare bucket only contains rare classes
    assert set(np.unique(vs.y_n).tolist()) <= set(vs.rare_class_ids)
    # h and m contain no rare classes
    assert not (set(np.unique(vs.y_h).tolist()) & set(vs.rare_class_ids))
    assert not (set(np.unique(vs.y_m).tolist()) & set(vs.rare_class_ids))


def test_median_split_threshold_logic():
    # 4 classes with counts 1, 2, 8, 10 (median = 5): classes >5 are "high".
    y = np.array([0] * 1 + [1] * 2 + [2] * 8 + [3] * 10)
    X = np.zeros((len(y), 3))
    vs = median_frequency_split(X, y, rare_class_ids=[])
    high_classes = set(np.unique(vs.y_h).tolist())
    mod_classes = set(np.unique(vs.y_m).tolist())
    assert high_classes == {2, 3}
    assert mod_classes == {0, 1}
