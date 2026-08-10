"""Tests for data generation and the dataset specifications."""

import os

import numpy as np
import pandas as pd
import pytest

from fedgraa.config import CIC_IDS_2017, UNSW_NB15, DatasetSpec
from fedgraa.data import load_nf_uq_nids_v2_silos, make_synthetic_dataset
from fedgraa.partition import partition_by_silo


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
    assert UNSW_NB15.n_features == 39  # 43 minus 3 categorical columns (proto/service/state) dropped by the numeric-only pipeline
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


# --------------------------------------------------------------------------- #
# W6 -- NF-UQ-NIDS-v2 multi-silo loader scaffold
# --------------------------------------------------------------------------- #
def _write_fake_silo_csv(dir_path: str, rng: np.random.Generator, labels, n: int) -> None:
    os.makedirs(dir_path, exist_ok=True)
    y = rng.choice(labels, size=n)
    df = pd.DataFrame({
        "feat1": rng.normal(size=n),
        "feat2": rng.normal(size=n),
        "feat3": rng.normal(size=n),
        "id": np.arange(n),
        "Attack": y,
    })
    df.to_csv(os.path.join(dir_path, "part.csv"), index=False)


def test_load_nf_uq_nids_v2_silos_missing_directory_raises_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="NF-UQ-NIDS-v2 silo directory not found"):
        load_nf_uq_nids_v2_silos(str(tmp_path), silo_names=["NF-BoT-IoT-v2"])


def test_load_nf_uq_nids_v2_silos_rejects_mismatched_column_schema(tmp_path):
    """Audit-found regression guard: two silos with the same NUMBER of
    numeric columns but different column names must raise, not silently
    concatenate unrelated features as if they were aligned by position."""
    rng = np.random.default_rng(0)
    silo_names = ["silo_a", "silo_b"]
    os.makedirs(str(tmp_path / "silo_a"))
    os.makedirs(str(tmp_path / "silo_b"))
    pd.DataFrame({
        "feat1": rng.normal(size=10), "feat2": rng.normal(size=10),
        "Attack": ["Benign"] * 10,
    }).to_csv(str(tmp_path / "silo_a" / "p.csv"), index=False)
    pd.DataFrame({
        "feat1": rng.normal(size=10), "feat3": rng.normal(size=10),
        "Attack": ["Benign"] * 10,
    }).to_csv(str(tmp_path / "silo_b" / "p.csv"), index=False)

    with pytest.raises(ValueError, match="schema mismatch"):
        load_nf_uq_nids_v2_silos(str(tmp_path), silo_names=silo_names)


def test_load_nf_uq_nids_v2_silos_derives_schema_from_data_not_a_table(tmp_path):
    """Nothing about the resulting DatasetSpec is hardcoded -- n_features,
    labels, and used_counts must all come from what was actually read."""
    rng = np.random.default_rng(0)
    silo_names = ["silo_a", "silo_b"]
    _write_fake_silo_csv(str(tmp_path / "silo_a"), rng, ["Benign", "DDoS"], n=40)
    _write_fake_silo_csv(str(tmp_path / "silo_b"), rng, ["Benign", "DDoS", "Bot"], n=25)

    silos, labels = load_nf_uq_nids_v2_silos(str(tmp_path), silo_names=silo_names)

    assert labels == sorted(["Benign", "DDoS", "Bot"])
    assert len(silos) == 2
    for X, y, name in silos:
        assert name in silo_names
        assert X.shape[1] == 3  # feat1, feat2, feat3 (id/Attack dropped)
        assert set(np.unique(y).tolist()) <= set(range(len(labels)))

    measured_used_counts = [0] * len(labels)
    for _X, y, _name in silos:
        for cid in y:
            measured_used_counts[cid] += 1
    assert sum(measured_used_counts) == 40 + 25

    spec = DatasetSpec(
        name="NF-UQ-NIDS-v2-TEST",
        n_features=silos[0][0].shape[1],
        labels=labels,
        used_counts=measured_used_counts,
        benign_id=labels.index("Benign"),
    )
    split = partition_by_silo(silos, spec, n_rare=1, val_fraction=0.2, seed=0)
    assert split.n_clients == 2


# --------------------------------------------------------------------------- #
# Loader/layout gap (docs/audit/nf_uq_nids_v2_provenance_audit.md §6): the
# official NF-UQ-NIDS-v2 release ships as one merged CSV with silo membership
# in a `Dataset` column, not four pre-split subdirectories.
# --------------------------------------------------------------------------- #
def _write_merged_csv(path: str, rng: np.random.Generator, silo_labels: dict, n_per_silo: int) -> None:
    frames = []
    for silo_name, labels in silo_labels.items():
        y = rng.choice(labels, size=n_per_silo)
        frames.append(pd.DataFrame({
            "feat1": rng.normal(size=n_per_silo),
            "feat2": rng.normal(size=n_per_silo),
            "feat3": rng.normal(size=n_per_silo),
            "id": np.arange(n_per_silo),
            "Attack": y,
            "Dataset": silo_name,
        }))
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


def test_load_nf_uq_nids_v2_silos_merged_csv_layout_matches_presplit(tmp_path):
    """The official release's actual layout: one CSV, silo tag in `Dataset`.
    Must produce the same silos/labels as the equivalent pre-split directories
    (small chunksize forces multiple chunks through the streaming path)."""
    rng = np.random.default_rng(0)
    silo_labels = {"silo_a": ["Benign", "DDoS"], "silo_b": ["Benign", "DDoS", "Bot"]}
    merged_path = str(tmp_path / "merged.csv")
    _write_merged_csv(merged_path, rng, silo_labels, n_per_silo=40)

    silos, labels = load_nf_uq_nids_v2_silos(
        merged_path, silo_names=list(silo_labels), chunksize=7,
    )

    assert labels == sorted({"Benign", "DDoS", "Bot"})
    assert len(silos) == 2
    for X, y, name in silos:
        assert name in silo_labels
        assert X.shape == (40, 3)  # feat1, feat2, feat3 (id/Attack/Dataset dropped)
        assert set(np.unique(y).tolist()) <= set(range(len(labels)))


def test_load_nf_uq_nids_v2_silos_merged_csv_missing_dataset_column_raises(tmp_path):
    merged_path = str(tmp_path / "merged.csv")
    pd.DataFrame({"feat1": [1.0, 2.0], "Attack": ["Benign", "DDoS"]}).to_csv(merged_path, index=False)

    with pytest.raises(ValueError, match="no 'Dataset' column"):
        load_nf_uq_nids_v2_silos(merged_path, silo_names=["silo_a"])


def test_load_nf_uq_nids_v2_silos_merged_csv_missing_silo_raises(tmp_path):
    rng = np.random.default_rng(0)
    merged_path = str(tmp_path / "merged.csv")
    _write_merged_csv(merged_path, rng, {"silo_a": ["Benign"]}, n_per_silo=10)

    with pytest.raises(ValueError, match="No rows found for silo"):
        load_nf_uq_nids_v2_silos(merged_path, silo_names=["silo_a", "silo_missing"])


# --------------------------------------------------------------------------- #
# B6 (docs/audit/PRE_COMPUTE_AUDIT_2026-07-28.md): an uncapped load of the
# real ~76M-row merged release was measured to need ~88 GB peak. These tests
# exercise the bounded-memory `max_per_class` path (two-pass tally + seeded
# per-(silo,class) selection) that fixes it, and the safety guard that keeps
# the old uncapped path from silently repeating the failure.
# --------------------------------------------------------------------------- #
def _write_merged_csv_variable(
    path: str, rng: np.random.Generator, silo_key_counts: dict,
) -> None:
    """Like ``_write_merged_csv`` but with an EXACT row count per
    ``(silo, label)`` key, so tests can put one key over a cap and another
    under it in the same file. ``silo_key_counts`` is
    ``{silo_name: {label: count}}``.
    """
    frames = []
    for silo_name, label_counts in silo_key_counts.items():
        for label, n in label_counts.items():
            frames.append(pd.DataFrame({
                "feat1": rng.normal(size=n),
                "feat2": rng.normal(size=n),
                "feat3": rng.normal(size=n),
                "id": np.arange(n),
                "Attack": label,
                "Dataset": silo_name,
            }))
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


def _key_counts(silos, labels) -> dict:
    """(silo, label_string) -> count, from a loader's ``(silos, labels)``
    return value."""
    out: dict = {}
    for X, y, name in silos:
        for cid in y:
            key = (name, labels[cid])
            out[key] = out.get(key, 0) + 1
    return out


def test_load_nf_uq_nids_v2_silos_max_per_class_caps_the_frequent_key_only(tmp_path):
    """A key over the cap is subsampled to exactly `max_per_class`; a key
    under the cap is kept in full -- capping must never drop rows it didn't
    need to."""
    rng = np.random.default_rng(0)
    silo_key_counts = {
        "silo_a": {"Benign": 200, "DDoS": 15},
        "silo_b": {"Benign": 30, "Bot": 5},
    }
    merged_path = str(tmp_path / "merged.csv")
    _write_merged_csv_variable(merged_path, rng, silo_key_counts)

    silos, labels = load_nf_uq_nids_v2_silos(
        merged_path, silo_names=list(silo_key_counts),
        chunksize=17, max_per_class=20, seed=0,
    )
    counts = _key_counts(silos, labels)

    assert counts[("silo_a", "Benign")] == 20   # 200 -> capped to 20
    assert counts[("silo_a", "DDoS")] == 15     # 15 -> kept in full (< cap)
    assert counts[("silo_b", "Benign")] == 20   # 30 -> capped to 20
    assert counts[("silo_b", "Bot")] == 5       # 5 -> kept in full (< cap)
    assert labels == sorted({"Benign", "DDoS", "Bot"})


def test_load_nf_uq_nids_v2_silos_capped_result_is_deterministic_given_seed(tmp_path):
    rng = np.random.default_rng(1)
    silo_key_counts = {"silo_a": {"Benign": 500, "DDoS": 8}}
    merged_path = str(tmp_path / "merged.csv")
    _write_merged_csv_variable(merged_path, rng, silo_key_counts)

    silos1, labels1 = load_nf_uq_nids_v2_silos(
        merged_path, silo_names=["silo_a"], chunksize=31,
        max_per_class=25, seed=7,
    )
    silos2, labels2 = load_nf_uq_nids_v2_silos(
        merged_path, silo_names=["silo_a"], chunksize=31,
        max_per_class=25, seed=7,
    )

    assert labels1 == labels2
    for (X1, y1, n1), (X2, y2, n2) in zip(silos1, silos2):
        assert n1 == n2
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_allclose(X1, X2)


def test_load_nf_uq_nids_v2_silos_capped_selection_is_chunksize_invariant(tmp_path):
    """Which rows get selected for a key depends only on (population size,
    max_per_class, seed) -- pass 1 counts exactly regardless of chunksize, so
    a different chunksize must not change which rows pass 2 keeps."""
    rng = np.random.default_rng(2)
    silo_key_counts = {"silo_a": {"Benign": 300, "DDoS": 40}}
    merged_path = str(tmp_path / "merged.csv")
    _write_merged_csv_variable(merged_path, rng, silo_key_counts)

    silos_small_chunk, _ = load_nf_uq_nids_v2_silos(
        merged_path, silo_names=["silo_a"], chunksize=9,
        max_per_class=30, seed=3,
    )
    silos_large_chunk, _ = load_nf_uq_nids_v2_silos(
        merged_path, silo_names=["silo_a"], chunksize=100_000,
        max_per_class=30, seed=3,
    )

    X1, y1, _ = silos_small_chunk[0]
    X2, y2, _ = silos_large_chunk[0]
    # Sort by first feature column so row order (which can differ across
    # chunk boundaries even for the same selected set) doesn't matter.
    order1 = np.argsort(X1[:, 0])
    order2 = np.argsort(X2[:, 0])
    np.testing.assert_allclose(X1[order1], X2[order2])
    np.testing.assert_array_equal(y1[order1], y2[order2])


def test_load_nf_uq_nids_v2_silos_max_per_class_larger_than_population_keeps_all(tmp_path):
    rng = np.random.default_rng(4)
    silo_key_counts = {"silo_a": {"Benign": 12, "DDoS": 6}}
    merged_path = str(tmp_path / "merged.csv")
    _write_merged_csv_variable(merged_path, rng, silo_key_counts)

    silos, labels = load_nf_uq_nids_v2_silos(
        merged_path, silo_names=["silo_a"], chunksize=5,
        max_per_class=1_000_000, seed=0,
    )
    counts = _key_counts(silos, labels)
    assert counts[("silo_a", "Benign")] == 12
    assert counts[("silo_a", "DDoS")] == 6


def test_load_nf_uq_nids_v2_silos_capped_missing_silo_still_raises(tmp_path):
    rng = np.random.default_rng(5)
    merged_path = str(tmp_path / "merged.csv")
    _write_merged_csv_variable(merged_path, rng, {"silo_a": {"Benign": 20}})

    with pytest.raises(ValueError, match="No rows found for silo"):
        load_nf_uq_nids_v2_silos(
            merged_path, silo_names=["silo_a", "silo_missing"],
            max_per_class=10,
        )


def test_load_nf_uq_nids_v2_silos_capped_feeds_partition_by_silo(tmp_path):
    """End-to-end sanity, mirroring the uncapped schema-derivation test: the
    capped path's output must still be a valid input to partition_by_silo."""
    rng = np.random.default_rng(6)
    silo_key_counts = {
        "silo_a": {"Benign": 100, "DDoS": 12},
        "silo_b": {"Benign": 80, "Bot": 9},
    }
    merged_path = str(tmp_path / "merged.csv")
    _write_merged_csv_variable(merged_path, rng, silo_key_counts)

    silos, labels = load_nf_uq_nids_v2_silos(
        merged_path, silo_names=list(silo_key_counts),
        chunksize=13, max_per_class=15, seed=0,
    )
    used_counts = [0] * len(labels)
    for _X, y, _name in silos:
        for cid in y:
            used_counts[cid] += 1

    spec = DatasetSpec(
        name="NF-UQ-NIDS-v2-CAPPED-TEST",
        n_features=silos[0][0].shape[1],
        labels=labels,
        used_counts=used_counts,
        benign_id=labels.index("Benign"),
    )
    split = partition_by_silo(silos, spec, n_rare=1, val_fraction=0.2, seed=0)
    assert split.n_clients == 2


def test_load_nf_uq_nids_v2_silos_uncapped_guard_raises_past_threshold(
    tmp_path, monkeypatch,
):
    """The uncapped path must refuse to proceed past the row-count safety
    guard rather than silently continuing toward the ~88 GB failure B6
    measured on the real file. The guard is monkeypatched down so the test
    can trigger it without generating millions of rows."""
    import fedgraa.data as data_mod

    monkeypatch.setattr(data_mod, "_UNCAPPED_ROW_GUARD", 50)
    rng = np.random.default_rng(8)
    merged_path = str(tmp_path / "merged.csv")
    _write_merged_csv(merged_path, rng, {"silo_a": ["Benign"]}, n_per_silo=100)

    with pytest.raises(ValueError, match="safety guard"):
        load_nf_uq_nids_v2_silos(merged_path, silo_names=["silo_a"])


def test_load_nf_uq_nids_v2_silos_uncapped_stays_under_guard_for_small_files(tmp_path):
    """The guard must not fire for the small files every existing test (and
    every legitimate small real-world silo) uses."""
    rng = np.random.default_rng(9)
    merged_path = str(tmp_path / "merged.csv")
    _write_merged_csv(merged_path, rng, {"silo_a": ["Benign", "DDoS"]}, n_per_silo=100)

    silos, labels = load_nf_uq_nids_v2_silos(merged_path, silo_names=["silo_a"])
    assert sum(len(y) for _X, y, _n in silos) == 100


def test_load_nf_uq_nids_v2_silos_max_per_class_rejected_for_directory_layout(tmp_path):
    """max_per_class is only implemented for the merged single-CSV layout
    (the real release's actual layout); passing it with a directory data_dir
    must raise loudly rather than silently ignore the memory-safety request."""
    rng = np.random.default_rng(0)
    _write_fake_silo_csv(str(tmp_path / "silo_a"), rng, ["Benign", "DDoS"], n=20)

    with pytest.raises(NotImplementedError, match="merged single-CSV layout"):
        load_nf_uq_nids_v2_silos(
            str(tmp_path), silo_names=["silo_a"], max_per_class=10,
        )


def test_load_nf_uq_nids_v2_silos_dtype_option_controls_output_dtype(tmp_path):
    rng = np.random.default_rng(10)
    merged_path = str(tmp_path / "merged.csv")
    _write_merged_csv(merged_path, rng, {"silo_a": ["Benign", "DDoS"]}, n_per_silo=30)

    silos_default, _ = load_nf_uq_nids_v2_silos(merged_path, silo_names=["silo_a"])
    silos_f32, _ = load_nf_uq_nids_v2_silos(
        merged_path, silo_names=["silo_a"], dtype=np.float32,
    )

    assert silos_default[0][0].dtype == np.float64
    assert silos_f32[0][0].dtype == np.float32
    # Same values (up to float32 precision), not a different computation.
    np.testing.assert_allclose(
        silos_default[0][0], silos_f32[0][0].astype(np.float64), atol=1e-5,
    )
