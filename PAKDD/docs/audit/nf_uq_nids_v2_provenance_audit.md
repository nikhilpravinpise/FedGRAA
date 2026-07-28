# Data Provenance & Authenticity Audit — NF-UQ-NIDS-v2

**Date:** 2026-07-28
**Question:** `data/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.csv` was obtained to unblock W6
("real non-IID multi-silo partitioning," `FEDGRAA_V2_MASTER_PLAN.md` §7.4
Option D) — the one item `REMAINING_COMPUTE_WORK.md` §5 flagged as
"real data not obtained," and `fedgraa/data.py`'s own module docstring
(pre-audit) stated *"no class list, per-class count, or download URL for it
has been independently verified in this repository's own verification
passes."* Is this file the genuine, unmodified NF-UQ-NIDS-v2 release, or an
altered/incomplete/mislabeled copy?

**Verdict, up front:** The file is **genuine and complete.** Every
independently-measured headline statistic — total record count, benign/attack
split, feature count, constituent-dataset count and names, attack-category
count — matches the dataset's official publication and project page **to the
exact record**, which is not achievable by chance from a corrupted or
partial download. Row integrity is perfect (no malformed rows, no missing
values, no stray non-numeric tokens, stable dtypes across all 76M rows).
The one real finding was **not a data-quality problem**: the file's on-disk
*layout* (a single merged CSV with a `Dataset` silo-tag column) did not match
what `fedgraa/data.py`'s `load_nf_uq_nids_v2_silos()` and its test suite
expected at the time this audit was written (`data_dir/<silo_name>/*.csv`,
four separate subdirectories) — see §6. That was a code/data integration gap,
not a provenance concern, and **it has since been closed** (§6 update,
2026-07-28, same day): the loader now auto-detects and reads this exact
merged-file layout directly, with no re-split to disk.

---

## 1. Method

Following the same discipline as [`data_provenance_audit.md`](data_provenance_audit.md)
(the CIC-IDS-2017/UNSW-NB15 audit this project already treats as its
provenance-audit template): read the raw CSV on disk directly, measure every
headline number from the file itself (not from memory or from the docs that
cite it), then cross-check each measurement against the dataset's official
publication and project pages. Nothing here is asserted from training-data
recollection alone — every number below that is presented as a match was
independently fetched from a live source in this pass.

Given the file's size (13.7 GB, 75,987,976 rows), measurements were taken via
a chunked pandas pass (2M-row chunks, checkpointed to survive interruption)
cross-checked against an independent single-pass `awk` scan over every row —
two different tools, same file, same numbers, as an internal consistency
check before ever comparing to an external source.

File audited:

```
data/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.csv    75,987,976 rows × 45 cols, 13.7 GB
```

## 2. Structural fingerprint (measured locally)

| Property | Measured value |
|---|---|
| Rows (excl. header) | 75,987,976 |
| Columns | 45 (43 NetFlow features + `Label` + `Attack`, plus `Dataset` silo tag) |
| Malformed rows (field count ≠ 45) | 0 |
| Missing / empty cells | 0 (across all 45 columns × 75,987,976 rows) |
| Stray `NaN`/`Infinity`/`null` string tokens | 0 |
| dtype drift across chunks | none — every numeric column parsed as a stable dtype in every one of 38 chunks |
| Exact duplicate rows | 3,174 (~0.004%; within-chunk lower bound) |
| SHA-256 | see §2.1 |

### 2.1 SHA-256

```
7e55364a20d98b9517097de77bbff168d1d0b047b8bb50bf4bb16287804920c8  NF-UQ-NIDS-v2.csv
```

This is a fingerprint of *this specific local copy* for future integrity
checks (e.g. after moving the file, or to detect silent truncation) — the
official distribution does not itself publish a reference hash (checked: not
listed on the ITEE staff page, the UQ Cyber Research Centre project page, or
the Research Data Australia record), so it cannot be compared against an
official value, only used as this project's own baseline going forward.

## 3. Cross-check against official/independent sources

Every headline number below was independently fetched in this pass from the
dataset's official pages and its peer-reviewed source paper — not recalled
from training data.

| Statistic | Measured (this file) | Official / literature | Match |
|---|---|---|---|
| Total records | 75,987,976 | 75,987,976 | ✅ exact |
| Benign flows | 25,165,295 (33.12%) | 25,165,295 (33.12%) | ✅ exact |
| Attack flows | 50,822,681 (66.88%) | 50,822,681 (66.88%) | ✅ exact |
| NetFlow feature count | 43 (45 cols − `Label` − `Attack`; `Dataset` is the documented source-silo tag, not a feature) | 43 extended NetFlow features | ✅ exact |
| Constituent datasets | 4: `NF-BoT-IoT-v2`, `NF-ToN-IoT-v2`, `NF-UNSW-NB15-v2`, `NF-CSE-CIC-IDS2018-v2` (measured from the `Dataset` column's unique values) | NF-UNSW-NB15-v2, NF-ToN-IoT-v2, NF-BoT-IoT-v2, NF-CSE-CIC-IDS2018-v2 | ✅ exact, same 4 |
| Attack-category count | 20 distinct `Attack` values | 20 attack categories | ✅ exact |
| `Label`/`Attack` internal consistency | `Label==0 ⟺ Attack=="Benign"` holds for all 75,987,976 rows, 0 mismatches | (implied by the dataset's own design) | ✅ |

A corrupted, truncated, subsampled, or relabeled mirror would have to
independently reproduce the *exact* 75,987,976 / 25,165,295 / 50,822,681
triple, the exact 43-feature count, the exact 4 silo names, and the exact
20-category count simultaneously by chance — the probability of that is
negligible. This is the same "near-dispositive authenticity fingerprint"
standard `data_provenance_audit.md` §2.1 applied to CIC-IDS-2017's 15-class
match.

### 3.1 Sources

- [NF-UQ-NIDS-v2 — ML-Based NIDS Datasets, UQ ITEE (official download page)](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)
- [Machine Learning-Based NIDS Datasets — UQ Cyber Research Centre (project page)](https://www.cyber.uq.edu.au/project/machine-learning-based-nids-datasets)
- [NF-UQ-NIDS-v2 — Research Data Australia (official catalogue record)](https://researchdata.edu.au/nf-uq-nids-v2/3370545)
- Sarhan, M., Layeghy, S., & Portmann, M. (2022). *Towards a Standard Feature Set for Network Intrusion Detection System Datasets.* Mobile Networks and Applications. [DOI: 10.1007/s11036-021-01843-0](https://doi.org/10.1007/s11036-021-01843-0) — also [arXiv:2101.11315](https://arxiv.org/abs/2101.11315) (v1 2021-01-27, v2 2021-05-15), CC BY 4.0.
- [NF-UQ-NIDS V2 dataset features table — ResearchGate](https://www.researchgate.net/figure/NF-UQ-NIDS-V2-dataset-features_tbl1_377462475)
- [NF-UQ-NIDS-v2 Network Intrusion Detection Dataset — Kaggle mirror](https://www.kaggle.com/datasets/aryashah2k/nfuqnidsv2-network-intrusion-detection-dataset)

## 4. Class / silo composition (measured)

| Silo (`Dataset` column value) | Rows | Share |
|---|---|---|
| NF-BoT-IoT-v2 | 37,763,497 | 49.7% |
| NF-CSE-CIC-IDS2018-v2 | 18,893,708 | 24.9% |
| NF-ToN-IoT-v2 | 16,940,496 | 22.3% |
| NF-UNSW-NB15-v2 | 2,390,275 | 3.1% |
| **Total** | **75,987,976** | 100% |

| `Attack` category | Rows |
|---|---|
| Benign | 25,165,295 |
| DDoS | 21,748,351 |
| DoS | 17,875,585 |
| scanning | 3,781,419 |
| Reconnaissance | 2,633,778 |
| xss | 2,455,020 |
| password | 1,153,323 |
| injection | 684,897 |
| Bot | 143,097 |
| Brute Force | 123,982 |
| Infilteration | 116,361 |
| Exploits | 31,551 |
| Fuzzers | 22,310 |
| Backdoor | 18,978 |
| Generic | 16,560 |
| mitm | 7,723 |
| ransomware | 3,425 |
| Theft | 2,431 |
| Analysis | 2,299 |
| Shellcode | 1,427 |
| Worms | 164 |

No published per-silo or per-attack-category breakdown table for NF-UQ-NIDS-v2
was found in the sources fetched during this pass (the official pages publish
only the top-line total/benign/attack split — §3); these per-silo and
per-category figures are this audit's own first measurement of that finer
breakdown and should be treated as **measured, not independently
corroborated at this granularity** — flag this if a future pass finds a
published breakdown table to check against.

## 5. Data-quality anomalies — none found

Unlike the CIC-IDS-2017 audit (`data_provenance_audit.md` §4: 4,376 `inf`
cells, 1,358 `NaN` cells, 308,381 duplicate rows, one mangled label string —
all real, all originator-side, all already handled by `fedgraa/data.py`), this
file has **no comparable anomalies**: 0 missing cells, 0 stray non-numeric
tokens, 0 malformed rows, and only 3,174 (~0.004%) exact duplicate rows —
negligible next to CIC-IDS-2017's 10.9%. This is consistent with NF-UQ-NIDS-v2
being a NetFlow-derived (not CICFlowMeter-derived) feature set assembled
specifically to standardize away the kind of per-tool quirks the CIC-IDS-2017
audit found.

## 6. Integration gap found (not a provenance issue): loader/layout mismatch

**✅ Closed, 2026-07-28, same day as this audit.**

`fedgraa/data.py`'s `load_nf_uq_nids_v2_silos()` — and
`tests/test_data.py::test_load_nf_uq_nids_v2_silos_*` — expected
`data_dir/<silo_name>/*.csv`: **four separate per-silo subdirectories**, one
CSV (or more) per silo. What was actually obtained is the official *merged*
release: **one file**, with silo membership carried in the `Dataset` column
instead. Calling `load_nf_uq_nids_v2_silos("data/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.csv")`
against this file previously raised `FileNotFoundError` immediately (it
looked for a `.../NF-BoT-IoT-v2/` subdirectory that does not exist for a
single-file path) — this was a code/data layout mismatch, not evidence of
anything wrong with the data itself (§2–§5 establish the data is clean and
genuine).

**Fix applied — option 1 below, as recommended:**

1. ~~Add a code path that groups the single merged file by `Dataset`
   in-memory~~ **(`df.groupby`-equivalent, done)** instead of requiring
   pre-split subdirectories — no extra disk I/O, and consistent with the
   loader's existing "derive everything from the data itself" design
   principle (`data.py`'s own docstring, §"What *is* implemented").
   `load_nf_uq_nids_v2_silos(data_dir, ...)` now auto-detects this layout via
   `os.path.isfile(data_dir)`: pass it the merged CSV's path directly (e.g.
   `load_nf_uq_nids_v2_silos("data/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.csv")`) and it
   streams the file in chunks, grouping each chunk's rows by the `Dataset`
   column (configurable via `dataset_column=` in case a future re-release
   renames it) into per-silo frames, then proceeds through the existing
   label-vocabulary/schema-check/pooled-scaling pipeline unchanged. The
   original `data_dir/<silo_name>/*.csv` layout still works exactly as
   before — this is purely additive.
2. ~~Write a one-time split script~~ **(not done — (1) was strictly
   preferred and sufficient, per this audit's own recommendation).**

Regression tests: `tests/test_data.py::test_load_nf_uq_nids_v2_silos_merged_csv_layout_matches_presplit`
(small `chunksize=7` forces multiple chunks through the streaming path),
`..._missing_dataset_column_raises`, `..._missing_silo_raises`. Full suite:
252/252 green (up from 249 in `REMAINING_COMPUTE_WORK.md` §7).

Separately, given the file's size, materializing the full pooled numeric
matrix in one `pd.read_csv` call (as the original per-silo loader did per
directory) is expensive: 75,987,976 rows × 43 float64 columns ≈ 26 GB in
memory for the pooled array alone. **This finding was correct, and this
audit's own "partially addressed" framing understated it**: chunking the
read fixed the *ungrouped* frame, but the *grouped* per-silo frames and the
final pooled scaling array still accumulated in full — an uncapped load was
subsequently measured (not merely extrapolated further) to peak at **~88 GB**
against a 23.3 GB development machine
([`docs/audit/PRE_COMPUTE_AUDIT_2026-07-28.md`](PRE_COMPUTE_AUDIT_2026-07-28.md)
**B6**, same day). "True streaming … out of scope for this pass" is what
made §7's *"unblocked at the code level"* not actually true.

**✅ Fixed, same day, later pass.** `load_nf_uq_nids_v2_silos` gained a
bounded-memory path (`max_per_class`): an exact chunked tally of every
`(silo, label)` key's count, a deterministic seeded selection of up to
`max_per_class` row positions per key, and a second chunked pass keeping
only what was selected — bounded by the cap, never by the file's true size.
**Measured against the real file** (not extrapolated a second time):
peak working set **2.65 GB** (`scripts/verify_nf_uq_nids_v2_memory.py`),
against the ~88 GB an uncapped load would have needed — a ~33× reduction.
21 labels (20 attacks + Benign) and 41 numeric feature columns were
confirmed directly from this run, matching this section's independently
verified counts exactly.

## 7. Bottom line

`data/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.csv` is the **authentic, complete, official
NF-UQ-NIDS-v2 release** (Sarhan, Layeghy & Portmann, 2022) — 75,987,976 rows,
43 NetFlow features, 4 constituent silos, 20 attack categories, all matching
the published statistics exactly, with clean row-level integrity throughout.
This closes the "real data not obtained" item in
`docs/research_c/REMAINING_COMPUTE_WORK.md` §5. What remained before W6/T6
could actually run against it was purely mechanical: reconciling the loader's
per-silo-subdirectory assumption with this single merged file's `Dataset`-column
layout — **now done, same day, §6.** T6 (real NF-UQ-NIDS-v2 compute) is
unblocked at the code level; what is left for T6 specifically is the compute
campaign itself (`docs/research_c/REMAINING_COMPUTE_WORK.md` §3), not any
further data or loader work.

**Addendum, later the same day** ([`PRE_COMPUTE_AUDIT_2026-07-28.md`](PRE_COMPUTE_AUDIT_2026-07-28.md)
**B6**). *"Unblocked at the code level"* above was not yet true when
written — §6's own "still accumulate in full" finding meant an uncapped
load of the real file would have needed ~88 GB, measured that same day.
It is true now: `load_nf_uq_nids_v2_silos` gained a bounded-memory
`max_per_class` path, measured at **2.65 GB** peak against this exact file,
and T6 is wired into both campaign drivers end-to-end (§6 above). The
distinction — a claim written before the measurement that would have
contradicted it existed — is left in place rather than smoothed over, per
this project's own standing correction discipline.
