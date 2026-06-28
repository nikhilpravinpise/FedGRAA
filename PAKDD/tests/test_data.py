"""Tests for data generation and the dataset specifications."""

import numpy as np

from fedgraa.config import CIC_IDS_2017, UNSW_NB15
from fedgraa.data import make_synthetic_dataset


def test_cic_spec_matches_table1():
    assert CIC_IDS_2017.n_features == 78
    assert CIC_IDS_2017.n_classes == 15
    assert CIC_IDS_2017.n_clients == 14  # one per attack type
    assert CIC_IDS_2017.used_count("Heartbleed") == 11
    assert CIC_IDS_2017.used_count("SQL Injection") == 21
    assert CIC_IDS_2017.used_count("Infiltration") == 36
    # frequent classes capped at 10,000
    assert CIC_IDS_2017.used_count("DoS Hulk") == 10_000


def test_unsw_spec_matches_table2():
    assert UNSW_NB15.n_features == 43
    assert UNSW_NB15.n_classes == 10
    assert UNSW_NB15.n_clients == 9
    assert UNSW_NB15.used_count("Worms") == 174
    assert UNSW_NB15.used_count("Shellcode") == 1511
    assert UNSW_NB15.used_count("Backdoor") == 2329


def test_rare_client_ids_default_n3():
    cic = CIC_IDS_2017.rare_client_ids(3)
    assert cic == [
        CIC_IDS_2017.label_id("Heartbleed"),
        CIC_IDS_2017.label_id("SQL Injection"),
        CIC_IDS_2017.label_id("Infiltration"),
    ]
    unsw = UNSW_NB15.rare_client_ids(3)
    assert unsw == [
        UNSW_NB15.label_id("Worms"),
        UNSW_NB15.label_id("Shellcode"),
        UNSW_NB15.label_id("Backdoor"),
    ]


def test_synthetic_counts_match_used_column():
    X, y, spec = make_synthetic_dataset(CIC_IDS_2017, seed=0)
    assert X.shape[1] == 78
    for class_id, expected in enumerate(spec.used_counts):
        assert int((y == class_id).sum()) == expected


def test_synthetic_is_standardized():
    X, y, _ = make_synthetic_dataset(UNSW_NB15, seed=0)
    # per-feature mean ~0, std ~1 after standard scaling
    np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(X.std(axis=0), 1.0, atol=1e-6)
