# Stage C compute-session update — 2026-07-30

> **Purpose.** A dated, standalone record of the FedGRAA-v2 Stage C compute
> resumption, result verification, FedPGO result commit, and the primary-source
> correction discovered in this session. It does **not** supersede or silently
> rewrite [`Final_Stage_C.md`](Final_Stage_C.md) or
> [`REMAINING_COMPUTE_WORK.md`](REMAINING_COMPUTE_WORK.md). Those live documents
> remain pending an author decision on the FedPGO reimplementation below.

## 1. Starting point and recovery check

- Repository tip at resumption was `c5c8ac8` (`Implement FedPGO baseline
  (corrected earlier misattribution), build the frontier figure/table`), after
  `605d1f1` and `7f06718`.
- The detached Modal app `ap-7JAdAvySkrbHYteY1doi4c` was no longer listed by
  `modal app list`; nothing was killed or relaunched.
- `modal volume ls fedgraa-v2-results v2_frontier` returned 739 objects. The
  volume contained the 20 canonical FedPGO JSONs (two datasets × ten seeds),
  plus the pre-existing `_r2` smoke artefact and other harmless non-cell
  objects. Results were retrieved with `modal volume get --force`.

## 2. Campaign verification

The canonical checker completed successfully:

```text
python scripts/check_stage_c_complete.py --tier T1 T2 T3 T4 T5 T6 --t4-m 3

STAGE C (T1/T2/T3/T4/T5/T6) COMPLETE — all checks passed.
730/730 cells complete
```

The complete set is 19 arms per dataset: five FedGRAA variants
(`v1_projected`, `v2_m1`, `v2_m2`, `v2_m3`, `v2_m5`) and fourteen external
baselines, including `fedpgo`. The arithmetic is 19 arms, not 18 as an earlier
handoff sentence accidentally stated. The checker verified every cell's
internal consistency and all relative documentation links. See
[`scripts/check_stage_c_complete.py`](../../scripts/check_stage_c_complete.py)
and the persisted inputs in [`results/v2_frontier/`](../../results/v2_frontier/).

## 3. Frontier-artifact rebuild

- The system `python` lacked NumPy and Matplotlib, so its first invocation of
  the build/render scripts did not complete. The project environment was then
  used:

  ```text
  .venv313/Scripts/python.exe scripts/build_v2_frontier_table.py
  .venv313/Scripts/python.exe scripts/render_v2_frontier.py
  ```

- The builder reported `19/19 arms with data` for both CIC-IDS-2017 and
  UNSW-NB15.
- The rebuilt [`frontier_table.json`](../../results/v2_frontier/frontier_table.json)
  has 38 dataset–arm rows; every row contains exactly seeds 0–9. The corresponding
  [`frontier_table.md`](../../results/v2_frontier/frontier_table.md) adds the
  `fedpgo` row to each dataset table, each with `n seeds = 10`.
- `frontier_figure.png` was regenerated but was byte-identical to its tracked
  version, so it has no new Git diff. FedPGO is not a plotted frontier point.

## 4. Tests and committed result set

The complete suite was run in the project environment:

```text
.venv313/Scripts/python.exe -m pytest tests/ -q
441 passed, 554 warnings in 654.95s (0:10:54)
```

The 12 FedPGO-specific tests are included in that total, confirmed by
`pytest tests/test_baselines_t5_fedpgo.py --collect-only -q`, which collected
12 tests from
[`tests/test_baselines_t5_fedpgo.py`](../../tests/test_baselines_t5_fedpgo.py).
Thus the prior expectation of approximately 453 tests was a stale arithmetic
expectation, not a regression.

Commit `30fd064` (`Add completed FedPGO frontier results`) contains exactly:

- the 20 canonical files matching
  `results/v2_frontier/*_fedpgo_g1.0_seed[0-9].json`; and
- the regenerated `frontier_table.json` and `frontier_table.md`.

The smoke artefact
`results/v2_frontier/UNSW_NB15_fedpgo_g1.0_seed0_r2.json` is deliberately
untracked and was not committed.

## 5. Primary-source correction: the current FedPGO cells are a proxy, not a faithful reproduction

After the compute commit, the author added the full FedPGO paper locally as
[`FedPGO/FedPGO.md`](FedPGO/FedPGO.md). This resolves the former
full-text-access blocker, but it also invalidates the earlier
abstract-level reconstruction claim.

The primary text specifies:

1. AGO projects against the **previous round's global gradient**
   ([`FedPGO.md`, §3.5](FedPGO/FedPGO.md#35-gradient-decomposition-adaptive-gradient-orthogonalization),
   especially the text and Eq. 6 at source lines 415–429).
2. Its recomposition is
   `g'_i = α_i g_parallel + (1 - α_i) g_perp`, with a client-specific
   `α_i ∈ [0, 1]` (Eq. 8; source lines 425–443).
3. Its PGF score is
   `λ1 cos(g_i, g_G) + λ2 ||g_perp|| + λ3 ΔL_val`: alignment, orthogonal
   magnitude, and post-personalization validation-loss reduction (Eq. 9;
   source lines 447–455).

The committed implementation does not match those details:

- It derives `tau_g` as the **same-round FedAvg-weighted mean** of local
  deltas, not a stored prior-round global gradient
  ([`fedgraa/baselines_t5_fedpgo.py`](../../fedgraa/baselines_t5_fedpgo.py),
  lines 303–312).
- It recomposes as `parallel + orthogonal_weight * orthogonal`, not the
  paper's α/(1−α) mixture (lines 148–171).
- It scores ReLU-clipped cosine similarity plus an orthogonal *fraction*, and
  has no validation-loss-improvement term (lines 174–211).

Therefore, the 20 committed cells accurately record the behaviour of the
repository's disclosed **FedPGO-inspired proxy**, but must not be presented as
results of a faithful FedPGO reproduction. The table rows should be relabelled
or replaced only after an author decision.

## 6. Required decision before live documentation is updated

Do **not** make the planned “all 14 baselines complete” status wording in the
live documentation without preserving the correction above. Choose one path:

1. **Faithful-baseline path:** revise
   [`fedgraa/baselines_t5_fedpgo.py`](../../fedgraa/baselines_t5_fedpgo.py)
   to the primary-source mechanism, add/update tests, and rerun the 20 FedPGO
   cells before rebuilding the frontier artifacts. This path may change the
   730/730 result corpus and requires a new compute campaign.
2. **Proxy-disclosure path:** retain commit `30fd064` and the 730/730 campaign
   record, but explicitly label FedPGO as an inspired proxy in the table and
   both live status documents; list a faithful source-based implementation and
   rerun as pending work.

Until that decision, [`Final_Stage_C.md`](Final_Stage_C.md) and
[`REMAINING_COMPUTE_WORK.md`](REMAINING_COMPUTE_WORK.md) have intentionally
not been edited in this session. Their existing statements that FedPGO full
text is inaccessible are now historically true but operationally superseded by
the local primary source; any correction must be dated and retained in place,
not silently deleted.

## 8. Resolution (2026-07-31): faithful-baseline path taken, and a §4 factual correction

**Dated decision.** Path 1 of §6 (faithful-baseline path) was taken, as
planned in
[`STAGE_C_FEDPGO_FAITHFUL_PLAN_2026-07-31.md`](STAGE_C_FEDPGO_FAITHFUL_PLAN_2026-07-31.md).

**Outcome.**

- The proxy implementation was replaced by a faithful implementation of the
  primary text (Eqs. 6–10, Algorithms 1–3) in
  [`fedgraa/baselines_t5_fedpgo.py`](../../fedgraa/baselines_t5_fedpgo.py),
  with the eight judgment calls JC1–JC8 disclosed in its module docstring
  (commit `25611ce`).
- Its test file was rewritten (19 tests, including a gradient-memory
  regression test pinning the proxy's same-round-FedAvg bug); full suite:
  453 passed.
- The 20 FedPGO cells were rerun on Modal (all stamped `25611ce`, `git_dirty:
  true` — the tree carried the then-untracked plan document) and committed in
  commit `96adb05`. The rebuilt frontier table/figure include the faithful
  FedPGO point: CIC-IDS-2017 all=0.5145/rare=0.0676, UNSW-NB15
  all=0.5318/rare=0.1087 (means over seeds 0–9).
- The 20 proxy cells were archived, not deleted, at
  [`results/v2_frontier/_archive_fedpgo_proxy_2026-07-30/`](../../results/v2_frontier/_archive_fedpgo_proxy_2026-07-30/)
  with a dated README (plan decision D2; collision-free by construction —
  the table builder's glob is non-recursive and the checker verifies exact
  `cell.path`).
- `check_stage_c_complete.py --tier T1 T2 T3 T4 T5 T6 --t4-m 3` remains
  green at 730/730.

**Correction to §4 (dated 2026-07-31).** §4 above states that the smoke
artefact `results/v2_frontier/UNSW_NB15_fedpgo_g1.0_seed0_r2.json` is
"deliberately untracked and was not committed". That is wrong: the file was
committed in `9d20ac9` (57 lines, alongside `FedPGO/FedPGO.md`). The
CIC-IDS-2017 `_r2` smoke artefact was never tracked. Both proxy-era `_r2`
files now live in the archive directory; the 2026-07-31 faithful rerun's own
CIC-IDS-2017 `_r2` smoke artefact (commit `25611ce`) is committed at the
canonical location. The UNSW-NB15 `_r2` was not re-smoked.

## 7. Literature-search result carried forward

The existing duplication/scooping result remains unchanged: the FastTrack
pass was re-run in the prior Stage C pass and found no new competing paper;
`XAI-SOH-FL` was the closest hit and was already catalogued. The relevant
historical evidence remains in
[`Final_Stage_C.md`](Final_Stage_C.md), including §1.2 and the `[S9]`
FastTrack provenance entry. This session did not re-run that literature tool.
