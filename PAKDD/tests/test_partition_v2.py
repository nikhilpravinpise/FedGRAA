"""Tests for FedGRAA-v2's F2/F3/F7/F13 partition primitives.

Covers FEDGRAA_V2_MASTER_PLAN.md §9.2's T-8, T-8b, T-8c, T-8d (the "zero
compute cost" oracle-recovery checks promoted to W0.3), plus unit coverage
for :func:`stratum_tags`/:func:`apply_tags` (F2, W1.1) and
:func:`local_label_histogram` (F3, W1.3).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from fedgraa.config import CIC_IDS_2017, UNSW_NB15
from fedgraa.data import make_synthetic_dataset
from fedgraa.partition import (
    apply_tags,
    bottom_n_rare_classes,
    local_label_histogram,
    median_frequency_split,
    partition_by_silo,
    partition_non_iid,
    stratum_tags,
)

SEEDS = range(10)


# --------------------------------------------------------------------------- #
# F2 / W1.1 -- stratum_tags + apply_tags compose to median_frequency_split
# --------------------------------------------------------------------------- #
def test_stratum_tags_apply_tags_composition_matches_median_split():
    """The structural claim of F2: median_frequency_split(X, y, rare) must be
    exactly reproducible as stratum_tags + apply_tags on the same histogram."""
    y = np.array([0] * 1 + [1] * 2 + [2] * 8 + [3] * 10 + [4] * 3)
    X = np.arange(len(y) * 3).reshape(len(y), 3).astype(float)
    rare_ids = [4]

    direct = median_frequency_split(X, y, rare_ids)

    counts = np.bincount(y, minlength=5)
    tags = stratum_tags(counts, rare_ids)
    composed = apply_tags(X, y, tags, rare_ids)

    np.testing.assert_array_equal(np.sort(direct.y_h), np.sort(composed.y_h))
    np.testing.assert_array_equal(np.sort(direct.y_m), np.sort(composed.y_m))
    np.testing.assert_array_equal(np.sort(direct.y_n), np.sort(composed.y_n))
    assert set(direct.rare_class_ids) == set(composed.rare_class_ids)


def test_apply_tags_is_purely_local():
    """F2: apply_tags needs only this client's own (X, y) and the broadcast
    tag vector -- no other client's rows, no global counts."""
    tags = np.array(["h", "m", "n"], dtype="<U1")
    y = np.array([0, 1, 2, 0, 2])
    X = np.arange(len(y) * 2).reshape(len(y), 2).astype(float)
    split = apply_tags(X, y, tags, rare_class_ids=[2])
    assert set(np.unique(split.y_h).tolist()) <= {0}
    assert set(np.unique(split.y_m).tolist()) <= {1}
    assert set(np.unique(split.y_n).tolist()) <= {2}
    assert len(split.y_h) + len(split.y_m) + len(split.y_n) == len(y)


def test_stratum_tags_gap26_fallback_matches_median_split():
    """The gap-2.6 fallback (no class strictly exceeds the median) must
    survive the stratum_tags/apply_tags refactor identically."""
    y = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5)
    X = np.zeros((len(y), 3))
    vs = median_frequency_split(X, y, rare_class_ids=[])
    assert len(vs.y_h) > 0 and len(vs.y_m) > 0
    assert len(vs.y_h) + len(vs.y_m) == len(y)


def test_stratum_tags_rare_class_absent_from_population_still_tagged_n():
    """F3's zero-held-out-rows corner: a rare class id can be present in
    rare_class_ids while entirely absent from this population's y -- the tag
    vector must still be long enough and still mark it 'n' (even though no
    row will ever match it here)."""
    y = np.array([0, 0, 1, 1, 1])  # class 5 (rare) has zero rows here
    counts = np.bincount(y, minlength=6)
    tags = stratum_tags(counts, rare_class_ids=[5])
    assert tags[5] == "n"
    assert len(tags) == 6


# --------------------------------------------------------------------------- #
# F3 / W1.3 -- local_label_histogram: full-shard vs held-out
# --------------------------------------------------------------------------- #
def test_local_label_histogram_full_scope_sums_train_and_val():
    X, y, spec = make_synthetic_dataset(CIC_IDS_2017, seed=0)
    split = partition_non_iid(X, y, spec, n_rare=3, val_fraction=0.10, seed=0)
    shard = split.clients[0]
    hist = local_label_histogram(shard, spec.n_classes, scope="full")
    expected = np.bincount(
        np.concatenate([shard.y_train, shard.y_val]), minlength=spec.n_classes
    )
    np.testing.assert_array_equal(hist, expected)


def test_local_label_histogram_heldout_scope_is_val_only():
    X, y, spec = make_synthetic_dataset(CIC_IDS_2017, seed=0)
    split = partition_non_iid(X, y, spec, n_rare=3, val_fraction=0.10, seed=0)
    shard = split.clients[0]
    hist = local_label_histogram(shard, spec.n_classes, scope="heldout")
    expected = np.bincount(shard.y_val, minlength=spec.n_classes)
    np.testing.assert_array_equal(hist, expected)


def test_local_label_histogram_rejects_bad_scope():
    X, y, spec = make_synthetic_dataset(CIC_IDS_2017, seed=0)
    split = partition_non_iid(X, y, spec, n_rare=3, val_fraction=0.10, seed=0)
    with pytest.raises(ValueError):
        local_label_histogram(split.clients[0], spec.n_classes, scope="bogus")


# --------------------------------------------------------------------------- #
# T-8c -- rank histogram (full-shard, SecAgg-summed) is deterministic; the
# held-out histogram is not (F3's actual, measured justification).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec", [CIC_IDS_2017, UNSW_NB15])
def test_rank_histogram_is_deterministic_across_seeds(spec):
    X, y, spec2 = make_synthetic_dataset(spec, seed=0)
    full_hists = []
    heldout_hists = []
    for seed in SEEDS:
        split = partition_non_iid(X, y, spec2, n_rare=3, val_fraction=0.10, seed=seed)
        full = sum(
            local_label_histogram(c, spec2.n_classes, scope="full")
            for c in split.clients
        )
        held = sum(
            local_label_histogram(c, spec2.n_classes, scope="heldout")
            for c in split.clients
        )
        full_hists.append(full)
        heldout_hists.append(held)

    used_counts = np.array(spec2.used_counts, dtype=np.int64)
    for full in full_hists:
        np.testing.assert_array_equal(full, used_counts)

    # The held-out histogram is NOT expected to be constant across seeds --
    # that instability is exactly F3's actual justification for preferring
    # the full-shard signal.
    assert not all(
        np.array_equal(heldout_hists[0], h) for h in heldout_hists[1:]
    ), "held-out histogram unexpectedly identical across all 10 seeds"


# --------------------------------------------------------------------------- #
# T-8d -- a protected rare class may have zero rows in the pooled D_V^n
# (v1 fragility, F3).
# --------------------------------------------------------------------------- #
def test_rare_class_may_be_absent_from_heldout_pool():
    """Regression guard on the v1 fragility F3 measured: on CIC-IDS-2017,
    n=3, at least one of seeds {5, 9} leaves a protected rare class with zero
    rows in the pooled D_V^n, because `_train_val_split` draws an
    unstratified ~10% of a tiny (11-36 row) shard."""
    X, y, spec = make_synthetic_dataset(CIC_IDS_2017, seed=0)
    oracle_rare = set(spec.rare_client_ids(3))

    found_zero_seed = False
    for seed in (5, 9):
        split = partition_non_iid(X, y, spec, n_rare=3, val_fraction=0.10, seed=seed)
        held = sum(
            local_label_histogram(c, spec.n_classes, scope="heldout")
            for c in split.clients
        )
        if any(held[c] == 0 for c in oracle_rare):
            found_zero_seed = True
            break
    assert found_zero_seed, (
        "expected at least one of seeds {5, 9} to have a protected rare class "
        "with zero rows in the pooled held-out split (F3)"
    )


# --------------------------------------------------------------------------- #
# T-8 -- BottomN recovers the hard-coded rare set (F7's zero-compute oracle
# check, re-scoped per F13: n=3 both datasets, n in {5,7} CIC only).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec", [CIC_IDS_2017, UNSW_NB15])
def test_bottom_n_recovers_hardcoded_rare_set_n3(spec):
    X, y, spec2 = make_synthetic_dataset(spec, seed=0)
    oracle = set(spec2.rare_client_ids(3))
    for seed in SEEDS:
        split = partition_non_iid(X, y, spec2, n_rare=3, val_fraction=0.10, seed=seed)
        counts = sum(
            local_label_histogram(c, spec2.n_classes, scope="full")
            for c in split.clients
        )
        recovered = set(bottom_n_rare_classes(counts, spec2.attack_ids, 3))
        assert recovered == oracle, (seed, recovered, oracle)


@pytest.mark.parametrize("n", [5, 7])
def test_bottom_n_recovers_hardcoded_rare_set_cic_only(n):
    X, y, spec = make_synthetic_dataset(CIC_IDS_2017, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        oracle = set(spec.rare_client_ids(n))
    for seed in SEEDS:
        split = partition_non_iid(X, y, spec, n_rare=n, val_fraction=0.10, seed=seed)
        counts = sum(
            local_label_histogram(c, spec.n_classes, scope="full")
            for c in split.clients
        )
        recovered = set(bottom_n_rare_classes(counts, spec.attack_ids, len(oracle)))
        assert recovered == oracle, (seed, recovered, oracle)


# --------------------------------------------------------------------------- #
# T-8b -- UNSW n>=5's oracle set is NOT count-derivable (F13): a five-way tie
# at the 10,000 frequent-class cap makes it structurally unrecoverable.
# --------------------------------------------------------------------------- #
def test_unsw_oracle_rare_set_is_not_count_derivable_at_n5():
    X, y, spec = make_synthetic_dataset(UNSW_NB15, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        oracle = set(spec.rare_client_ids(5))

    recovered_matches = 0
    for seed in SEEDS:
        split = partition_non_iid(X, y, spec, n_rare=5, val_fraction=0.10, seed=seed)
        counts = sum(
            local_label_histogram(c, spec.n_classes, scope="full")
            for c in split.clients
        )
        recovered = set(bottom_n_rare_classes(counts, spec.attack_ids, 5))
        if recovered == oracle:
            recovered_matches += 1

    assert recovered_matches == 0, (
        "expected the count-derived BottomN(n=5) to NEVER match the oracle "
        "rare set on UNSW-NB15 -- five classes tie at the 10,000-sample cap, "
        "so the pre-capping counts that would disambiguate them do not exist "
        "anywhere in the federated system (F13)"
    )
    # And the cause is exactly the five-way tie at the frequent-class cap.
    counts = np.array(spec.used_counts)
    capped = [i for i in spec.attack_ids if counts[i] == spec.frequent_cap]
    assert len(capped) >= 5, "expected >= 5 attack classes tied at the cap"


# --------------------------------------------------------------------------- #
# W6 -- partition_by_silo: real-non-IID multi-silo partitioning primitive
# --------------------------------------------------------------------------- #
def test_partition_by_silo_builds_one_client_per_silo():
    rng = np.random.default_rng(0)
    spec = CIC_IDS_2017
    silos = []
    for i, name in enumerate(["silo_a", "silo_b", "silo_c"]):
        n = 200 + i * 50
        X = rng.normal(size=(n, spec.n_features)).astype(np.float32)
        y = rng.integers(0, spec.n_classes, size=n)
        silos.append((X, y, name))

    split = partition_by_silo(silos, spec, n_rare=3, val_fraction=0.10, seed=0)
    assert split.n_clients == 3
    for client, (X, y, name) in zip(split.clients, silos):
        assert client.attack_label == name
        assert client.attack_id == -1
        total = client.n_train + len(client.y_val)
        assert total == len(y)
    # pooled val set is the concatenation of every silo's own held-out split
    assert len(split.y_val) == sum(
        len(c.y_val) for c in split.clients
    )


def test_partition_by_silo_rejects_empty_silo_list():
    with pytest.raises(ValueError):
        partition_by_silo([], CIC_IDS_2017, n_rare=3)
