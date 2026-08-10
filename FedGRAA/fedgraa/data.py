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
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DatasetSpec, get_dataset_spec


# --------------------------------------------------------------------------- #
# UNSW-NB15 downloader (gap B-1)
# --------------------------------------------------------------------------- #
UNSW_URLS = {
    "UNSW_NB15_training-set.csv":
        "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECv4C4/download?path=%2F&files=UNSW_NB15_training-set.csv",
    "UNSW_NB15_testing-set.csv":
        "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECv4C4/download?path=%2F&files=UNSW_NB15_testing-set.csv",
    "UNSW-NB15_1.csv":
        "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECv4C4/download?path=%2F&files=UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv":
        "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECv4C4/download?path=%2F&files=UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv":
        "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECv4C4/download?path=%2F&files=UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv":
        "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECv4C4/download?path=%2F&files=UNSW-NB15_4.csv",
}


def download_unsw_nb15(
    target_dir: str,
    files: Optional[List[str]] = None,
    timeout: int = 120,
) -> List[str]:
    """Attempt to download the UNSW-NB15 CSV files.

    The official AARNet CloudStor URLs are tried first. If they fail (e.g.
    because the service requires authentication or has moved), the function
    prints manual-download instructions and returns an empty list.

    Returns the list of successfully downloaded file paths.
    """
    os.makedirs(target_dir, exist_ok=True)
    files = files or list(UNSW_URLS.keys())
    downloaded: List[str] = []
    for fname in files:
        url = UNSW_URLS.get(fname)
        if url is None:
            continue
        dest = os.path.join(target_dir, fname)
        if os.path.exists(dest):
            downloaded.append(dest)
            continue
        print(f"Downloading {fname} ...", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
            downloaded.append(dest)
            print(f"  saved {dest}", flush=True)
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"  failed to download {fname}: {exc}", flush=True)
            if os.path.exists(dest):
                os.remove(dest)
    if not downloaded:
        print(
            "\nCould not auto-download UNSW-NB15. Please obtain the CSV files "
            "manually from https://research.unsw.edu.au/projects/unsw-nb15-dataset "
            f"and place them in {target_dir}/",
            flush=True,
        )
    return downloaded


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
    cache_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, DatasetSpec]:
    """Load a dataset by name.

    If ``data_dir`` is given and contains CSV files, the real data is loaded and
    preprocessed. Otherwise a synthetic dataset matching the paper's counts is
    returned (with a clear runtime notice handled by the caller).

    Pass ``cache_dir`` to save / reload the preprocessed ``(X, y)`` arrays as
    ``.npy`` files. On a second run this skips the ~30 s CSV-loading phase
    entirely. The cache key encodes the dataset name and seed so different
    seeds don't collide.
    """
    spec = get_dataset_spec(name)

    if data_dir and os.path.isdir(data_dir):
        csvs = [
            os.path.join(data_dir, f)
            for f in sorted(os.listdir(data_dir))
            if f.lower().endswith(".csv")
        ]
        # Gap B-1: auto-download UNSW-NB15 if the directory is empty.
        if not csvs and name == "UNSW-NB15":
            print(f"[{name}] Data directory is empty; attempting auto-download ...")
            download_unsw_nb15(data_dir)
            csvs = [
                os.path.join(data_dir, f)
                for f in sorted(os.listdir(data_dir))
                if f.lower().endswith(".csv")
            ]
        if csvs:
            # --- NPY cache check ----------------------------------------- #
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                tag = f"{name.replace('-', '_').replace('/', '_')}_seed{seed}"
                cx = os.path.join(cache_dir, f"{tag}_X.npy")
                cy = os.path.join(cache_dir, f"{tag}_y.npy")
                if os.path.exists(cx) and os.path.exists(cy):
                    print(f"[{name}] Loading preprocessed cache from {cache_dir}",
                          flush=True)
                    X = np.load(cx)
                    y = np.load(cy)
                    return X, y, spec
            # --- real CSV path -------------------------------------------- #
            X, y, spec = _load_real_csvs(
                csvs, spec, label_column=label_column, seed=seed
            )
            if cache_dir:
                np.save(cx, X)
                np.save(cy, y)
                print(f"[{name}] Saved preprocessed cache to {cache_dir}",
                      flush=True)
            return X, y, spec

    return make_synthetic_dataset(spec, seed=seed)


def _load_real_csvs(
    csv_paths: List[str],
    spec: DatasetSpec,
    label_column: Optional[str],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, DatasetSpec]:
    print(f"  Loading {len(csv_paths)} CSV file(s)...", flush=True)

    frames = []
    for i, p in enumerate(csv_paths, 1):
        frames.append(pd.read_csv(p, skipinitialspace=True))
        print(f"    [{i}/{len(csv_paths)}] {os.path.basename(p)} "
              f"({len(frames[-1]):,} rows)", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    print(f"  Total rows: {len(df):,}", flush=True)
    del frames  # free memory before heavy processing

    if label_column is None:
        # CIC files use "Label"; UNSW exports use "attack_cat" (multiclass) alongside "label" (binary 0/1).
        # Prefer the multiclass column when available.
        for candidate in ("attack_cat", "Label", "label", "Attack", "class"):
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

    # Map labels first (fast: only normalises unique values) so we can drop
    # unknown rows before the expensive numeric conversion.
    print("  Mapping labels...", flush=True)
    y = _match_labels(y_raw, spec)
    keep_mask = y >= 0
    n_dropped = int((~keep_mask).sum())
    if n_dropped:
        print(f"  Dropped {n_dropped:,} rows with unrecognised labels.", flush=True)
    y = y[keep_mask]
    X_df = X_df[keep_mask]

    # Keep numeric features only; clean inf / NaN (CIC-IDS contains both).
    # Use select_dtypes to avoid the OOM-prone apply(to_numeric) path on large
    # DataFrames: apply triggers a np.vstack of all columns that can allocate
    # hundreds of MB of intermediates and crash on constrained machines.
    print("  Converting features to float64...", flush=True)
    X_df = X_df.select_dtypes(include=[np.number])
    X = X_df.to_numpy(dtype=np.float64)
    del X_df  # free the DataFrame before allocating the final capped array
    np.place(X, ~np.isfinite(X), 0.0)  # replace inf / NaN in one pass

    print("  Capping frequent classes...", flush=True)
    X, y = _cap_frequent_classes(X, y, spec, seed=seed)
    # The paper is silent on feature scaling. Global standard scaling on the
    # pooled data is the closest thing to an unmentioned default and is applied
    # before partitioning (partition_non_iid re-centers the same statistics).
    print("  Standard-scaling globally...", flush=True)
    X = _standard_scale(X)
    print(f"  Final dataset: {len(y):,} rows, {X.shape[1]} features, "
          f"{len(np.unique(y))} classes.", flush=True)
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


def _match_labels(y_raw: "pd.Series", spec: DatasetSpec) -> np.ndarray:
    """Map raw label strings onto ``spec`` class IDs, returning -1 for misses.

    Vectorised: normalises only the *unique* raw values (typically < 20) and
    then uses pandas.Series.map for the full-length assignment.  This avoids
    a Python-level loop over millions of rows that caused a ~14 s freeze when
    loading the full 2.8 M-row CIC-IDS-2017 dataset.
    """
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

    # Normalise only the unique values (typically < 20), then map the whole
    # Series in one pandas vectorised call.
    unique_raw = y_raw.unique()
    raw_to_id: dict = {}
    for raw in unique_raw:
        full, singular = _normalize_label(str(raw))
        raw_to_id[raw] = norm_to_id.get(full, norm_to_id.get(singular, -1))

    return y_raw.map(raw_to_id).fillna(-1).astype(int).to_numpy()


def _standard_scale(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std


# --------------------------------------------------------------------------- #
# NF-UQ-NIDS-v2 (W6, FEDGRAA_V2_MASTER_PLAN.md §7.4 Option D) -- real
# multi-silo non-IID loader scaffold.
# --------------------------------------------------------------------------- #
# ⚠️ Disclosure. NF-UQ-NIDS-v2 (Sarhan, Layeghy & Portmann's unified NetFlow
# feature set, merging four independently-collected NIDS datasets -- commonly
# referred to in the literature as NF-BoT-IoT-v2, NF-ToN-IoT-v2,
# NF-UNSW-NB15-v2 and NF-CSE-CIC-IDS2018-v2) is cited by
# ``docs/research_c/Final_Stage_C.md`` §5 ("Option D") and
# ``docs/research_c/FEDGRAA_V2_MASTER_PLAN.md`` §7.4 as the dataset that makes
# FedGRAA-v2's non-IID/privacy claim non-vacuous.
#
# ✅ Provenance verified, 2026-07-28 (``docs/audit/nf_uq_nids_v2_provenance_audit.md``).
# A local copy (``data/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.csv``, 75,987,976 rows) was
# independently measured against the dataset's official publication and
# project pages -- total/benign/attack split, feature count, constituent-silo
# names, and attack-category count all match to the exact record, with clean
# row-level integrity throughout. This is now audited to the same standard as
# CIC-IDS-2017 / UNSW-NB15 (``docs/audit/data_provenance_audit.md``). **No
# download URL is hardcoded in this module regardless** -- per this project's
# own standing rule ("never guess a URL"), the file was obtained by the
# project author using their own web access outside this codebase's tools,
# not fetched by any code here, so there is nothing to embed.
#
# What *is* implemented: a loader that, given either (a) the four per-silo
# CSVs placed by the caller in ``data_dir`` (one subfolder per silo) or (b)
# the official release's single merged CSV (silo membership carried in a
# ``Dataset`` tag column -- the layout the real file actually ships in, per
# the provenance audit's §6), derives everything (feature columns, label set,
# per-class counts, and the resulting :class:`~fedgraa.config.DatasetSpec`)
# **from the data itself** rather than from a hardcoded table -- so nothing
# here can silently drift from whatever the real files actually contain, and
# no unverified number ever enters ``fedgraa/config.py``.
NF_UQ_NIDS_V2_SILO_NAMES = (
    "NF-BoT-IoT-v2",
    "NF-ToN-IoT-v2",
    "NF-UNSW-NB15-v2",
    "NF-CSE-CIC-IDS2018-v2",
)


# ⚠️ B6 (docs/audit/PRE_COMPUTE_AUDIT_2026-07-28.md). An uncapped load of the
# real merged release (75,987,976 rows) was measured -- from a real 1M-row
# sample's deep pandas footprint, extrapolated -- to require ~88 GB peak
# across the grouped per-silo frames, the pd.concat copy, and the pooled
# scaling array. Above this many TOTAL rows accumulated across the tracked
# silos, an uncapped load refuses to proceed rather than silently repeat
# that failure: pass ``max_per_class`` instead (see :func:`load_nf_uq_nids_v2_silos`).
_UNCAPPED_ROW_GUARD = 5_000_000

# Kept local to the NF-UQ loader -- deliberately NOT unified with
# ``_load_real_csvs``'s own candidate tuple (line ~219), whose priority order
# differs (CIC/UNSW prefer "attack_cat" first; NF-UQ's real schema uses
# "Attack" first) -- merging them risks silently changing which column
# CIC/UNSW resolve to.
_NF_UQ_LABEL_COLUMN_CANDIDATES = ("Attack", "attack_cat", "Label", "label", "class")


def _resolve_nf_uq_label_column(
    columns: List[str], label_column: Optional[str]
) -> Optional[str]:
    if label_column is not None:
        return label_column
    for candidate in _NF_UQ_LABEL_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


def _tally_merged_csv(
    csv_path: str,
    dataset_column: str,
    label_column: Optional[str],
    chunksize: int,
) -> Tuple[Dict[Tuple[str, str], int], str, set]:
    """Pass 1 of the capped read (``max_per_class`` given): an *exact*
    per-``(silo, label)`` row count via one chunked scan.

    Holds nothing beyond the current chunk plus a small ``{key: count}``
    dict -- O(chunksize) + O(distinct keys) memory, independent of the
    file's total row count. This is what makes the subsequent selection
    (:func:`_select_positions_per_key`) exact rather than approximate: the
    population size for each key's sample is known precisely before any
    row is chosen.
    """
    # Resolve the two columns this pass actually needs from the header alone
    # (a single near-instant row-0 read) so the chunked scan below can pass
    # ``usecols=`` and skip parsing NF-UQ-NIDS-v2's other ~40 NetFlow feature
    # columns on every one of the real file's ~38 chunks -- this pass only
    # ever tallies (silo, label) counts, so those columns are pure overhead.
    header_cols = [
        c.strip()
        for c in pd.read_csv(csv_path, skipinitialspace=True, nrows=0).columns
    ]
    if dataset_column not in header_cols:
        raise ValueError(
            f"Merged NF-UQ-NIDS-v2 CSV {csv_path!r} has no {dataset_column!r} "
            f"column to group by (columns seen: {header_cols}). Pass "
            "dataset_column= if the real file uses a different silo-tag "
            "column name."
        )
    resolved_label_column = _resolve_nf_uq_label_column(header_cols, label_column)
    if resolved_label_column is None:
        raise ValueError(
            "Could not infer the label column across NF-UQ-NIDS-v2 "
            "silos; pass label_column explicitly once the real "
            "schema is confirmed."
        )

    keep_cols = {dataset_column, resolved_label_column}
    counts: Dict[Tuple[str, str], int] = {}
    seen_datasets: set = set()
    for chunk in pd.read_csv(
        csv_path, skipinitialspace=True, chunksize=chunksize,
        usecols=lambda c: c.strip() in keep_cols,
    ):
        chunk.columns = [c.strip() for c in chunk.columns]
        tags = chunk[dataset_column].astype(str).str.strip()
        labels = chunk[resolved_label_column].astype(str).str.strip()
        seen_datasets.update(tags.unique().tolist())
        key_counts = pd.DataFrame(
            {"_s": tags.to_numpy(), "_l": labels.to_numpy()}
        ).value_counts()
        for (silo, label), n in key_counts.items():
            key = (silo, label)
            counts[key] = counts.get(key, 0) + int(n)
    return counts, resolved_label_column, seen_datasets


def _select_positions_per_key(
    counts: Dict[Tuple[str, str], int],
    silo_names: List[str],
    max_per_class: int,
    seed: int,
) -> Dict[Tuple[str, str], np.ndarray]:
    """Deterministic (seeded), uniform-without-replacement selection of up
    to ``max_per_class`` row **positions** (0-indexed, in file-scan order)
    for each ``(silo, label)`` key.

    Computed from ``counts`` alone -- pass 2 (:func:`_read_merged_csv_selected_rows`)
    never needs to hold more than these small index arrays in memory, no
    matter how large the key's true population is.

    ``numpy.random.Generator.choice(n, size=k, replace=False)`` was measured
    (not assumed) to be O(k) time and memory in practice for large ``n`` and
    small ``k`` -- 10,000 draws from a 30,000,000 population completed in
    under 1ms using well under 1MB, confirmed directly before relying on it
    here, per this project's own "measure, don't assume" discipline
    (docs/audit/PRE_COMPUTE_AUDIT_2026-07-28.md).
    """
    rng = np.random.default_rng(seed)
    selected: Dict[Tuple[str, str], np.ndarray] = {}
    for (silo, label), n in counts.items():
        if silo not in silo_names:
            continue
        k = min(n, max_per_class)
        idx = rng.choice(n, size=k, replace=False)
        idx.sort()
        selected[(silo, label)] = idx
    return selected


def _read_merged_csv_selected_rows(
    csv_path: str,
    dataset_column: str,
    label_column: str,
    silo_names: List[str],
    chunksize: int,
    selected: Dict[Tuple[str, str], np.ndarray],
) -> Dict[str, "pd.DataFrame"]:
    """Pass 2 of the capped read: re-scan the file, keeping only the rows
    whose per-key position was chosen by :func:`_select_positions_per_key`.

    Peak memory is ``O(chunksize) + O(sum of selected sizes)`` --
    ``sum(selected sizes) <= n_keys * max_per_class`` -- never the full
    file, regardless of the real dataset's actual row count. Position
    counters are threaded across chunks so a key spanning multiple chunks
    (the normal case) is still selected correctly: pass 1 and pass 2 scan
    the file in the same order, so "row j of key (silo, label)" means the
    same physical row in both passes.
    """
    position_counters: Dict[Tuple[str, str], int] = {}
    per_silo_frames: Dict[str, List["pd.DataFrame"]] = {n: [] for n in silo_names}
    silo_set = set(silo_names)
    all_columns: Optional[List[str]] = None

    for chunk in pd.read_csv(csv_path, skipinitialspace=True, chunksize=chunksize):
        chunk.columns = [c.strip() for c in chunk.columns]
        if all_columns is None:
            all_columns = [c for c in chunk.columns if c != dataset_column]
        tags = chunk[dataset_column].astype(str).str.strip()
        in_scope = tags.isin(silo_set)
        if not in_scope.any():
            continue
        sub = chunk.loc[in_scope]
        sub_tags = tags.loc[in_scope]
        sub_labels = sub[label_column].astype(str).str.strip()
        key_df = pd.DataFrame(
            {"_s": sub_tags.to_numpy(), "_l": sub_labels.to_numpy()},
            index=sub.index,
        )
        for (silo, label), group_index in key_df.groupby(
            ["_s", "_l"], sort=False
        ).groups.items():
            key = (silo, label)
            n = len(group_index)
            start = position_counters.get(key, 0)
            position_counters[key] = start + n
            sel = selected.get(key)
            if sel is None or len(sel) == 0:
                continue
            positions = np.arange(start, start + n)
            mask = np.isin(positions, sel, assume_unique=True)
            if mask.any():
                rows = sub.loc[group_index].iloc[np.flatnonzero(mask)]
                per_silo_frames[silo].append(rows.drop(columns=[dataset_column]))

    return {
        name: (
            pd.concat(per_silo_frames[name], ignore_index=True)
            if per_silo_frames[name]
            else pd.DataFrame(columns=all_columns or [])
        )
        for name in silo_names
    }


def _read_merged_nf_uq_nids_v2_csv(
    csv_path: str,
    dataset_column: str,
    silo_names: List[str],
    chunksize: int,
    label_column: Optional[str] = None,
    max_per_class: Optional[int] = None,
    seed: int = 0,
) -> List[Tuple["pd.DataFrame", str]]:
    """Stream a single merged NF-UQ-NIDS-v2 CSV (silo membership carried in
    a ``dataset_column`` tag column, e.g. the official release's ``Dataset``
    column) in chunks, grouping rows by silo instead of requiring pre-split
    subdirectories.

    **Bounded-memory path (``max_per_class`` given, B6's fix).** Two chunked
    passes over the file -- :func:`_tally_merged_csv` counts exactly, then
    :func:`_select_positions_per_key` + :func:`_read_merged_csv_selected_rows`
    keep only a deterministically-sampled ``max_per_class`` rows per
    ``(silo, label)`` key. Peak memory is bounded by chunk size and the
    total selected rows -- **never** by the file's true row count. This is
    the path a real load of NF-UQ-NIDS-v2's ~76M rows must use.

    **Unbounded path (``max_per_class=None``, the historical behaviour).**
    Single chunked pass, still never materializing the whole *ungrouped*
    file at once -- but per-silo frames accumulate in full across chunks,
    which is exactly what made an uncapped load of the real file require
    ~88 GB peak (measured, B6). A running-total guard
    (:data:`_UNCAPPED_ROW_GUARD`) now refuses to proceed past 5,000,000
    accumulated rows with an actionable error instead of continuing toward
    an OOM with no diagnostic. Every existing caller of this function uses
    small (test) files far under that guard, so this path's behaviour and
    error messages are unchanged for them.
    """
    if max_per_class is not None:
        counts, resolved_label_column, seen_datasets = _tally_merged_csv(
            csv_path, dataset_column, label_column, chunksize,
        )
        selected = _select_positions_per_key(
            counts, silo_names, max_per_class, seed,
        )
        per_silo = _read_merged_csv_selected_rows(
            csv_path, dataset_column, resolved_label_column, silo_names,
            chunksize, selected,
        )
        missing = [name for name in silo_names if per_silo[name].empty]
        if missing:
            raise ValueError(
                f"No rows found for silo(s) {missing} in {dataset_column!r} "
                f"column of {csv_path!r}. Values actually present: "
                f"{sorted(seen_datasets)}. Pass silo_names= matching what "
                "the file actually contains."
            )
        return [(per_silo[name], name) for name in silo_names]

    # --- Unbounded path: unchanged behaviour, plus a safety guard -------- #
    per_silo_chunks: Dict[str, List["pd.DataFrame"]] = {name: [] for name in silo_names}
    seen_datasets = set()
    total_selected = 0
    for chunk in pd.read_csv(csv_path, skipinitialspace=True, chunksize=chunksize):
        chunk.columns = [c.strip() for c in chunk.columns]
        if dataset_column not in chunk.columns:
            raise ValueError(
                f"Merged NF-UQ-NIDS-v2 CSV {csv_path!r} has no {dataset_column!r} "
                f"column to group by (columns seen: {list(chunk.columns)}). Pass "
                "dataset_column= if the real file uses a different silo-tag "
                "column name."
            )
        tags = chunk[dataset_column].astype(str).str.strip()
        seen_datasets.update(tags.unique().tolist())
        for name in silo_names:
            sub = chunk.loc[tags == name].drop(columns=[dataset_column])
            if not sub.empty:
                per_silo_chunks[name].append(sub)
                total_selected += len(sub)
        if total_selected > _UNCAPPED_ROW_GUARD:
            raise ValueError(
                f"Uncapped NF-UQ-NIDS-v2 load has already accumulated "
                f"{total_selected:,} rows across the tracked silos, past the "
                f"{_UNCAPPED_ROW_GUARD:,}-row safety guard. An uncapped load "
                "of the real ~76M-row release measured ~88 GB peak memory "
                "(docs/audit/PRE_COMPUTE_AUDIT_2026-07-28.md, finding B6). "
                "Pass max_per_class= (e.g. 10_000, matching this project's "
                "existing DatasetSpec.frequent_cap convention) to bound "
                "memory via deterministic per-(silo,class) subsampling "
                "instead of an unbounded load."
            )

    missing = [name for name in silo_names if not per_silo_chunks[name]]
    if missing:
        raise ValueError(
            f"No rows found for silo(s) {missing} in {dataset_column!r} column of "
            f"{csv_path!r}. Values actually present: {sorted(seen_datasets)}. Pass "
            "silo_names= matching what the file actually contains."
        )

    return [
        (pd.concat(per_silo_chunks[name], ignore_index=True), name)
        for name in silo_names
    ]


def load_nf_uq_nids_v2_silos(
    data_dir: str,
    label_column: Optional[str] = None,
    silo_names: Optional[List[str]] = None,
    dataset_column: str = "Dataset",
    chunksize: int = 2_000_000,
    max_per_class: Optional[int] = None,
    seed: int = 0,
    dtype: type = np.float64,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, str]], List[str]]:
    """Load NF-UQ-NIDS-v2's four constituent silos from local CSVs.

    Two on-disk layouts are supported, auto-detected from ``data_dir``:

    * **Pre-split (default expectation):** ``data_dir/<silo_name>/*.csv`` for
      each of ``silo_names`` (default :data:`NF_UQ_NIDS_V2_SILO_NAMES``) --
      one subdirectory per silo.
    * **Merged (the official release's actual layout, confirmed by
      ``docs/audit/nf_uq_nids_v2_provenance_audit.md``):** ``data_dir`` is a
      path to a single CSV file with silo membership carried in a
      ``dataset_column`` tag column (default ``"Dataset"``). Rows are
      grouped by that column in place -- no re-split to disk -- per the
      audit's §6 preferred fix. Selected automatically whenever
      ``os.path.isfile(data_dir)``.

    Either way, the caller is responsible for having already obtained and
    placed the file(s) (see the module docstring above: this project does
    not embed a download URL for a dataset it has not independently
    verified).

    **``max_per_class`` (B6, docs/audit/PRE_COMPUTE_AUDIT_2026-07-28.md).**
    ``None`` (default) preserves every existing caller's behaviour exactly.
    For the real ~76M-row merged release, pass an explicit cap -- e.g.
    ``10_000``, matching :attr:`~fedgraa.config.DatasetSpec.frequent_cap`'s
    existing convention -- to bound memory via deterministic
    per-``(silo, class)`` subsampling (see :func:`_read_merged_nf_uq_nids_v2_csv`).
    Omitting it on a large file now raises rather than silently attempting
    an ~88 GB load. **Not implemented for the pre-split directory layout**
    (that path reads whole CSVs directly, and no real dataset today ships
    at a scale where it would matter) -- passing ``max_per_class`` with a
    directory ``data_dir`` raises ``NotImplementedError`` rather than
    silently ignoring the memory-safety request.

    Returns ``(silos, labels)`` where ``silos`` is the
    ``[(X_silo, y_silo, silo_name), ...]`` list :func:`~fedgraa.partition.partition_by_silo`
    consumes and ``labels`` is the **measured** sorted union of class strings
    seen across every silo (index in this list = the class id encoded into
    each ``y_silo``) -- feed straight to a
    :class:`~fedgraa.config.DatasetSpec` constructed by the caller from these
    measured values, never from a hardcoded count table.
    """
    silo_names = silo_names or list(NF_UQ_NIDS_V2_SILO_NAMES)

    if os.path.isfile(data_dir):
        per_silo_raw = _read_merged_nf_uq_nids_v2_csv(
            data_dir, dataset_column, silo_names, chunksize,
            label_column=label_column, max_per_class=max_per_class, seed=seed,
        )
    else:
        if max_per_class is not None:
            raise NotImplementedError(
                "max_per_class is only implemented for the merged single-CSV "
                "layout (data_dir pointing at a file), not the pre-split "
                "directory layout -- the real, provenance-verified "
                "NF-UQ-NIDS-v2 release ships as the former "
                "(docs/audit/nf_uq_nids_v2_provenance_audit.md §6), so this "
                "is the layout B6's memory fix targets. Point data_dir at "
                "the merged CSV, or omit max_per_class if the directory "
                "layout's total row count is known to be small."
            )
        per_silo_raw = []
        for name in silo_names:
            silo_dir = os.path.join(data_dir, name)
            if not os.path.isdir(silo_dir):
                raise FileNotFoundError(
                    f"NF-UQ-NIDS-v2 silo directory not found: {silo_dir!r}. "
                    "This loader does not download NF-UQ-NIDS-v2 automatically "
                    "(no URL is hardcoded in this repository -- see the module "
                    "docstring above and docs/audit/nf_uq_nids_v2_provenance_audit.md). "
                    f"Obtain the official per-silo CSV export(s) for {name!r} and place "
                    f"them under {silo_dir!r} (or point data_dir at the single "
                    "merged release CSV instead) before calling this function."
                )
            csvs = [
                os.path.join(silo_dir, f)
                for f in sorted(os.listdir(silo_dir))
                if f.lower().endswith(".csv")
            ]
            if not csvs:
                raise FileNotFoundError(f"No CSV files found in {silo_dir!r}")
            frames = [pd.read_csv(p, skipinitialspace=True) for p in csvs]
            df = pd.concat(frames, ignore_index=True)
            df.columns = [c.strip() for c in df.columns]
            per_silo_raw.append((df, name))

    if label_column is None:
        for candidate in ("Attack", "attack_cat", "Label", "label", "class"):
            if candidate in per_silo_raw[0][0].columns:
                label_column = candidate
                break
    if label_column is None:
        raise ValueError(
            "Could not infer the label column across NF-UQ-NIDS-v2 silos; "
            "pass label_column explicitly once the real schema is confirmed."
        )

    # Measure the label vocabulary across ALL silos before encoding, so every
    # silo's y is aligned to the same class ids (a class present in one silo
    # but not another must still get a consistent id everywhere).
    all_labels: set = set()
    for df, _name in per_silo_raw:
        all_labels.update(df[label_column].astype(str).str.strip().unique().tolist())
    labels = sorted(all_labels)
    label_to_id = {lbl: i for i, lbl in enumerate(labels)}

    silos: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for df, name in per_silo_raw:
        y_raw = df[label_column].astype(str).str.strip()
        y = y_raw.map(label_to_id).astype(int).to_numpy()

        drop_cols = {label_column}
        for extra in ("id", "Label", "label", "attack_cat", "Attack", "Flow ID", "Timestamp"):
            if extra in df.columns:
                drop_cols.add(extra)
        X_df = df.drop(columns=list(drop_cols)).select_dtypes(include=[np.number])
        X = X_df.to_numpy(dtype=dtype)
        np.place(X, ~np.isfinite(X), 0.0)
        silos.append((X, y, name, tuple(X_df.columns)))

    # ⚠️ Audit-found bug (2026-07-28): concatenating silos with the same
    # *number* of numeric columns but DIFFERENT column names would silently
    # align unrelated features by position -- e.g. one silo's 3rd numeric
    # column being "bytes_out" and another's being "duration" -- producing a
    # pooled matrix that looks fine (no shape error) but is semantically
    # corrupt. NF-UQ-NIDS-v2's whole premise is a schema *shared* across its
    # four constituent datasets, so this is exactly the kind of assumption
    # this project's own audit discipline requires checking, not trusting.
    ref_cols = silos[0][3]
    for _X, _y, name, cols in silos[1:]:
        if cols != ref_cols:
            raise ValueError(
                f"Numeric column schema mismatch between silo {silo_names[0]!r} "
                f"({len(ref_cols)} columns: {ref_cols}) and silo {name!r} "
                f"({len(cols)} columns: {cols}). NF-UQ-NIDS-v2's unified "
                "NetFlow feature set assumes every constituent silo shares "
                "the same numeric schema after dropping id/label columns; "
                "concatenating mismatched schemas would silently align "
                "unrelated features by column position. Confirm the real "
                "files share a schema, or pass a harmonized label_column / "
                "pre-align the columns before calling this loader."
            )
    silos = [(X, y, name) for X, y, name, _cols in silos]

    # Global standard scaling across every silo pooled together, matching
    # partition_non_iid's single-scaling-pass convention (data.py's own
    # docstring: scale once, upstream of partitioning, never re-scale after).
    # Mean/std are always accumulated in float64 regardless of `dtype` --
    # capping bounds the row count enough that this pooled array is small,
    # and computing the moments in the wider type avoids float32
    # accumulation error before the final cast back to `dtype`.
    X_all = np.concatenate([s[0] for s in silos], axis=0).astype(np.float64)
    mean = X_all.mean(axis=0, keepdims=True)
    std = X_all.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    silos = [
        (((X.astype(np.float64) - mean) / std).astype(dtype), y, name)
        for X, y, name in silos
    ]

    return silos, labels


def _cap_frequent_classes(
    X: np.ndarray, y: np.ndarray, spec: DatasetSpec, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Cap each class at ``spec.frequent_cap`` samples (Section 5).

    The paper says highly frequent *attacks* are capped at 10,000; the
    Benign/Normal class is not in that list and is therefore left untouched.
    """
    rng = np.random.default_rng(seed)
    keep_idx: List[np.ndarray] = []
    for class_id in range(spec.n_classes):
        if class_id == spec.benign_id:
            keep_idx.append(np.flatnonzero(y == class_id))
            continue
        idx = np.flatnonzero(y == class_id)
        if len(idx) > spec.frequent_cap:
            idx = rng.choice(idx, size=spec.frequent_cap, replace=False)
        keep_idx.append(idx)
    sel = np.concatenate(keep_idx)
    rng.shuffle(sel)
    return X[sel], y[sel]
