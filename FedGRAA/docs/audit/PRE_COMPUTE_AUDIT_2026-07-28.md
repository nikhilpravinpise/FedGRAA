# Pre-Compute Audit — Full Repository Sweep Before the Modal Campaign

> **What this document is.** A whole-repository audit conducted immediately
> before launching the FedGRAA-v2 compute campaign (T1–T7), covering every
> source module, script, test, and document in `PAKDD/`. Its purpose is
> narrow and practical: **find everything that would make the campaign
> produce wrong, unusable, or unattributable numbers — before ~$27 and
> several hundred core-hours are spent producing them.**
>
> **Method.** Claims were checked by **execution**, not by reading. The full
> test suite was run; `verify_v2_exactness.py` and `verify_v2_plan_claims.py`
> were run; the completeness gate was run; the dataset loader's memory
> footprint was measured against the real 12.79 GB file; the SecAgg dropout
> failure probability was computed from the actual production threshold; and
> a purpose-built timing probe measured real per-round wall-clock and
> byte counts for the v1 and v2 arms. Where a document and the code disagree,
> **the code was executed to decide which is right.**
>
> **Headline.** The **science is sound** — F12's two-phase exact protocol is
> genuinely exact (re-measured this pass at `1.2×10⁻⁷`–`3.8×10⁻⁷`), the byte
> formula is genuinely `6×`/`2×` (re-measured this pass on a real run), and
> the suite is genuinely green at **252**. **The campaign harness is not
> ready.** Six blocking defects would each, on their own, invalidate part of
> the campaign; four of them are silent (they produce plausible-looking
> numbers rather than errors).
>
> **Compiled:** 2026-07-28. **Audits:** the working tree as of this date,
> including all uncommitted Stage C work.
>
> ---
>
> ## ✅ REMEDIATION COMPLETE (same day) — read this before §3
>
> **Five of the six blocking findings are fixed, with 40 new regression
> tests.** T1, T2 and T4 are launch-ready. **B6 (NF-UQ-NIDS-v2, ~88 GB peak)
> remains open and gates T6 only** — it is a genuine data-engineering task,
> scoped in §7 Phase 4.
>
> | # | Finding | Status | Where |
> |---|---|---|---|
> | **B1** | No Modal campaign driver | ✅ **Fixed** | [`scripts/modal_frontier_campaign.py`](../../scripts/modal_frontier_campaign.py) — pinned image, `verify_env` pre-flight, per-dataset memory split, Volume-persisted, resumable |
> | **B2** | Frontier anchor at `η′=0.1` | ✅ **Fixed** | `DEFAULT_ETA_PRIME = 0.01` pinned for **both** arms (author's decision); `--eta-prime` exposed |
> | **B3** | T4 paths collide with T1/T2 | ✅ **Fixed** | `CellSpec.config_tag`; T1 filenames unchanged; 400 tiers → 400 unique paths |
> | **B4** | `p_drop=0.3` crashes | ✅ **Fixed** | An unrecoverable round now **aborts that round** and is counted (`failed_secagg_rounds`), matching the real protocol's own semantics |
> | **B5** | DP server-side at `1/√K` | ✅ **Fixed** | Laplace moved into the per-client submit loop; aggregate noise is now `(Δ₁/ε)·√(2K)`, asserted against the **trainer**, not just the standalone function |
> | **B6** | NF-UQ-NIDS-v2 ~88 GB peak | ✅ **Fixed, measured on the real file** | Two-pass bounded-memory loader; **peak measured 2.65 GB** (down from an extrapolated ~88 GB) reading the actual 12.79 GB / 75,987,976-row release — §3's B6 entry below |
>
> **Also fixed:** H4 (v1 now bills bytes, so the frontier's `1×` anchor is
> measured), H3 (pinned versions confirmed exact on a real Modal build), M1
> (T-14/15/16/17 implemented, not withdrawn — see below), M2 (`secagg=False`
> now raises instead of being ignored), M3/M4 (stale counts and arithmetic),
> M6 (results carry full config + library versions + commit hash), M8, M9.
>
> **T6 is now fully wired**, not just the loader in isolation: `run_one_cell`
> has a complete NF-UQ-NIDS-v2 branch (`load_nf_uq_nids_v2_silos` +
> `partition_by_silo`, `DatasetSpec` derived from measured data), a `T6` tier
> exists in `run_v2_frontier_sweep.py`, and it is exercised end-to-end by
> `tests/test_run_one_cell_nf_uq.py` on synthetic data — the same code path,
> just not the 12.79 GB file, in the normal suite. **T6's Modal driver is a
> genuinely separate file and App**,
> [`scripts/modal_frontier_campaign_t6.py`](../../scripts/modal_frontier_campaign_t6.py)
> — the first version put it on the same App as T1/T2/T4 and that was a real
> bug, caught and fixed the same pass; see the box directly below.
>
> **Modal-vs-local equivalence, verified directly.** A fresh Modal
> `--smoke` run and an identical local `run_one_cell()` call, same seed,
> same commit, were diffed field by field rather than assumed: `bytes_sent`,
> `comm_rounds` and `final_all_acc` matched to full float64 precision
> between the two environments, on both arms. See §2.6.
>
> ### ⚠️ A second new defect, found while getting T6 launch-ready: T6's own Modal driver made T1 slow
>
> Wiring T6 into `modal_frontier_campaign.py` as a second `@app.function`
> image (`image_t6`, alongside the existing `image`) was itself wrong,
> caught by running it rather than by reading the code: a plain
> `--tier T1 --smoke` launch, which should finish in ~1 minute, instead sat
> for **15+ minutes** building an image containing the 12.79 GB NF-UQ-NIDS-v2
> directory that T1 never reads. Modal's `app.run()` (entered by `modal run`)
> syncs **every function registered on the App it activates**, not only the
> ones the invoked tier needs — so a second image on the *same* `modal.App`
> is built and uploaded regardless of which function is actually called.
>
> **Confirmed with a minimal, isolated experiment before committing to a
> fix**, not assumed: a throwaway file with two separate `modal.App` objects
> — App A with a trivial function and a `local_entrypoint`, App B (same
> file) with a function whose image does `add_local_dir` on the real
> 12.79 GB directory but *no* entrypoint of its own — ran App A's entrypoint
> and measured **12 seconds**, with App B's image never built.
>
> **Fix.** T6 now lives entirely in its own file and App,
> [`scripts/modal_frontier_campaign_t6.py`](../../scripts/modal_frontier_campaign_t6.py)
> (`fedgraa-v2-frontier-t6`), not a second image bolted onto the T1/T2/T4
> App. `modal_frontier_campaign.py`'s image once again excludes
> `data/NF-UQ-NIDS-v2` unconditionally, with no conditional path at all —
> there is no case in which that App should ever touch the file.
>
> **Verified, not just fixed:**
>
> | | Before the fix | After the fix |
> |---|---|---|
> | `modal run modal_frontier_campaign.py --tier T1 --verify-only` | 15+ min (killed) | **43 s** |
> | Functions built for a T1 launch | `run_cell_large`, `run_cell_small`, `verify_env`, **`run_cell_t6`, `verify_env_t6`** | `run_cell_large`, `run_cell_small`, `verify_env` only |
> | `datasets_present.NF-UQ-NIDS-v2` on the T1 image | — | `False` (correctly excluded) |
>
> This is the second real defect this pass found by running code rather than
> by reading it — matching the verification pass's own V1/V2/V3 pattern
> above, on a mechanism that pass did not touch. It is recorded here, not
> folded silently into a rewritten B6 section, for the same reason V1–V3
> were: **the failure mode worth naming is "claimed correct without being
> run," and the fix is not the interesting part — catching it before launch
> is.**
>
> ### One new defect, found by the tests written to close M1
>
> Implementing T-15 surfaced a live bug in [`fedgraa/stats.py`](../../fedgraa/stats.py)'s
> `paired_ttest`. Its degeneracy guard was an **exact** float comparison,
> `np.ptp(diffs) == 0.0`. The case that actually arises does not hit it
> exactly: `[0.7, 0.8] − [0.6, 0.7]` is a constant difference in exact
> arithmetic, but its two float64 entries differ by `1.1 × 10⁻¹⁶`, so the
> guard did not fire and `ttest_rel` divided by a `~10⁻¹⁷` standard error —
> returning **`t = 1.8 × 10¹⁵`, `p = 3.5 × 10⁻¹⁶`** from data that is
> identical. That is precisely the "spurious near-infinite significance" the
> function's own docstring claims to prevent.
>
> Now a **relative** test (`std(diffs) ≤ 10⁻¹² · scale`), plus guards for
> `n < 2` and non-finite `scipy` output. **No published number moved:**
> `audit_stage_a.py` and `audit_stage_b.py` — which recompute every
> transcribed value, including every Holm-corrected p-value, from the raw
> persisted JSON — both still exit 0. Stage B's real paired differences have
> `std ≈ 10⁻²`, twelve orders of magnitude above the new threshold.
>
> **This is the audit's own best argument for itself:** the four unimplemented
> test IDs were not bookkeeping. Writing one of them found a latent
> significance-reporting bug in code that the campaign was about to use.
>
> ---
>
> ## 🔍 VERIFICATION PASS (2026-07-28, after remediation)
>
> The remediation above was then **independently re-verified**, on the
> principle that a fix asserted is not a fix demonstrated. **Three real
> problems in the remediation itself were caught this way (V1, V2, V3) — none
> of them by a failing test.** V1 came from reading the actual output files,
> V3 from reading the launcher after the smoke test succeeded, and V2 from
> measuring a number the write-up had already asserted. All three are fixed
> and now carry regression tests.
>
> ### What was checked
>
> | Check | Result |
> |---|---|
> | 25 behavioural assertions, one per claimed fix, testing *behaviour* not source text | **25/25 pass** |
> | Full suite | **292 passed**, exit 0 |
> | `check_stage_a_complete`, `check_stage_b_complete`, `audit_stage_a`, `audit_stage_b`, `check_doc_links` | **all exit 0** |
> | `check_stage_c_complete --tier T1` | exit 1 — correct, campaign not run |
> | `compileall fedgraa scripts tests` | clean |
> | **Modal smoke test — 2 real cells, real CIC-IDS-2017 data, in containers** | **both completed** |
>
> The sharpest of the 25: B5's aggregate noise now measures
> `σ_empirical/σ_closed-form = **0.991**`. Had the server-side bug still been
> present, that ratio would read `1/√K = **0.500**`. And the `stats.py` fix is
> not over-aggressive — a genuine 4-seed difference still returns
> `p = 0.0275`.
>
> ### The smoke test is the first end-to-end evidence of the campaign path
>
> Two cells (`v1_projected` and `v2_m1`, 2 rounds, real cached CIC-IDS-2017)
> ran in Modal containers and wrote to the results Volume:
>
> | | `v1_projected` | `v2_m1` |
> |---|---|---|
> | `bytes_sent` | 9,344,160 | 56,071,800 |
> | `comm_rounds` | 2 | 22 = `2 × (1 + 5×2)` |
> | `final_all_acc` | 0.39803794272523985 | 0.39803794272523985 |
> | `ζ` at round 2 | — | `6.105 × 10⁻⁸` |
>
> **Measured byte ratio: `6.0007×`** — T-12's formula, now confirmed across
> *both arms* on *real data* on the *actual campaign platform*, which is
> only possible because H4 made the v1 denominator a measurement. The two
> arms agreeing to the last float digit at `m=1` with `ζ ≈ 6×10⁻⁸` is T-4's
> reduction property showing up end-to-end. (Accuracy is a discrete statistic,
> so bit-identity here is expected rather than remarkable — the gradient-level
> proof remains T-4's job, not this run's.)
>
> Both files landed as `..._seed0_r2.json`. The `_r2` tag is B3's fix working
> unprompted: **the smoke cells did not squat on the real T1 filenames.**
>
> ### ⚠️ Finding V1 — `git_commit` was `null` on every Modal cell
>
> Reading the actual result JSONs, not the code that writes them, showed both
> carrying `"git_commit": null, "git_dirty": null`. The Modal image ships a
> **copy** of the working tree, not a clone, and the container has no `git`
> binary — so `provenance()`'s subprocess call could never succeed there. Its
> best-effort design (correct, so a campaign never dies on provenance) turned
> that into a **silent null on exactly the platform the campaign runs on**,
> defeating the part of M6 that mattered most.
>
> Fixed: the launcher resolves commit and dirty-state locally, where git
> exists, and forwards them through the environment; `provenance()` reads the
> environment **before** attempting git. Three regression tests, including one
> that simulates `git` being absent entirely.
>
> **This one could only be found by inspecting output.** Every unit test for
> `provenance()` passed throughout — they ran on a machine with git.
>
> ### ⚠️ Finding V3 — the Modal path was not actually resumable
>
> The Modal launcher enumerated cells and dispatched **all** of them: it never
> called `missing_cells()`. Its own docstring said *"resumable"*, and the
> local CLI genuinely is — but `missing_cells()` checks **local** paths, and
> the Modal cells' paths live on the mounted Volume, so nothing was ever
> filtered. A relaunch after a partial failure would have re-run every
> completed cell. That is the exact property
> `COMPUTE_INFRASTRUCTURE_2026-07-27.md` §6.4 leans on to argue preemption
> costs one cell rather than the campaign — so the argument for **not**
> paying Modal's 3× non-preemptible surcharge rested on a property the code
> did not have.
>
> Fixed on both sides, sharing one definition so they cannot drift apart
> again:
>
> * `should_skip_cell(cell, force)` in `run_v2_frontier_sweep.py` — the single
>   source of truth, unit-tested (complete → skip; truncated JSON → re-run;
>   missing key → re-run; `--force` → override; per-cell, not per-tier).
> * The launcher filters against a Volume listing so a relaunch does not pay a
>   container start per finished cell.
> * Each container re-checks authoritatively after `results_volume.reload()` —
>   a mount is a snapshot taken at container start, so without the reload a
>   cell cannot see results other containers committed.
>
> **Verified on Modal, not asserted:** re-running the smoke test printed
> `resuming: 2 of 2 cells already complete on the volume, 0 to run` and
> dispatched nothing; `--force` re-ran both and produced identical byte counts.
>
> ### ⚠️ Finding V2 — this audit's own test counts were unverified
>
> The remediation write-up claimed **284** passing tests in four places and
> **294** in a fifth, and "32" vs "42" new tests. All were written before the
> suite finished, and the suite was interrupted. The measured figure is
> **292** (252 baseline + 31 + 9), now corrected everywhere. Note the
> correction itself had to be made *twice*: an intermediate value of 289 was
> written down, then invalidated by the three tests finding V3 added
> afterwards. Both passes are recorded rather than smoothed over — the
> failure mode is not a wrong number, it is writing any number before
> measuring the tree it describes.
>
> This is precisely §6.2's *"readiness claims outrunning the code"* — the
> pattern this audit was written to name — reproduced by the audit itself,
> within hours, on a smaller scale. It is recorded rather than quietly
> corrected because the lesson is the document's own thesis: **a number that
> was not measured is not a result, however confident the surrounding prose.**

---

## ✅ B6 CLOSURE PASS (2026-07-28, same day) — the one blocker the verification pass above did not touch

The verification pass re-checked B1–B5 and the campaign driver; B6
(NF-UQ-NIDS-v2's ~88 GB extrapolated peak) was still open at that point and
is closed here, in the same "measure, don't assume" spirit.

**Fix.** [`fedgraa/data.py`](../../fedgraa/data.py)'s
`load_nf_uq_nids_v2_silos` gained a bounded-memory path (used when a caller
passes `max_per_class`): an exact chunked tally of every `(silo, label)`
key's row count, a deterministic seeded selection of up to `max_per_class`
row positions per key computed from the counts alone (`numpy.random.Generator
.choice(n, size=k, replace=False)`, measured — not assumed — to be O(k), not
O(n), even at `n=30,000,000`), and a second chunked pass keeping only the
selected positions. The old unbounded path is byte-for-byte unchanged for
every existing small-file caller, now behind a safety guard that raises
rather than silently repeating the ~88 GB failure if a future caller omits
the cap.

**Measured against the real file, not extrapolated:**
[`scripts/verify_nf_uq_nids_v2_memory.py`](../../scripts/verify_nf_uq_nids_v2_memory.py)
(Windows peak working set via `GetProcessMemoryInfo`), `max_per_class=10_000`
(matching `DatasetSpec.frequent_cap`'s existing convention):

```
elapsed              :   1041.5 s   (~17.4 min, two full chunked passes)
PEAK WORKING SET     :   2714.5 MB  (2.65 GB)
labels (n=21)        :  20 attack categories + Benign -- matches the
                         provenance audit's independently-verified count
total selected rows  :  255,864 across the 4 real silos
```

**2.65 GB measured vs. ~88 GB extrapolated — a ~33× reduction**, comfortably
under every Modal container tier already sized for this campaign. A single
transient pandas `RuntimeWarning: overflow encountered in cast` was observed
during the run; reproduced separately on a smaller slice and confirmed
`np.isfinite` clean (no `inf`/`NaN` reached `X`) — noted for transparency,
not a defect.

**T6 is now wired end-to-end, not just the loader.** `run_v2_frontier_sweep.py`
gained a `T6` tier and a complete `run_one_cell` branch
(`load_nf_uq_nids_v2_silos` + `partition_by_silo`, `DatasetSpec` derived from
measured data, never hardcoded). T6's Modal driver is a genuinely separate
file/App, [`scripts/modal_frontier_campaign_t6.py`](../../scripts/modal_frontier_campaign_t6.py)
— **not** a second image on the T1/T2/T4 App, which is what the first
version did and which turned out to be a real bug (a plain `--tier T1`
launch built and uploaded the 12.79 GB file too; see the "second new defect"
box above for the measurement that caught it and the fix). This file has its
own `run_cell_t6` and `verify_env_t6` (the latter checking the CSV *file*
specifically, not just the directory).
[`tests/test_run_one_cell_nf_uq.py`](../../tests/test_run_one_cell_nf_uq.py)
exercises the exact same code path end-to-end on a small synthetic file, so
it runs in the normal suite in seconds.

**Remaining, honestly scoped, not hidden:** `partition_by_silo` creates one
client per silo — NF-UQ-NIDS-v2 has **4**, against T1's K=14/K=9. T6 is
documented as "T1's grid, on the one real-silo dataset" rather than "T1
repeated," since the client-count difference is structural, arguably the
*point* of a real (not synthetic) cross-silo benchmark, and should not read
as an oversight.

**Suite: 311 passed** (292 → 311; 19 new tests — 10 in `tests/test_data.py`
for the loader itself, 5 in `tests/test_run_v2_frontier_sweep.py` for T6
enumeration, 4 in the new `tests/test_run_one_cell_nf_uq.py`), `zero`
existing assertions weakened. `check_doc_links.py` and
`python -m compileall fedgraa scripts tests` both still clean.

**Modal-vs-local equivalence, checked directly rather than assumed** (the
user's own question: "is running this on Modal basically the same as running
it on my laptop?"). A fresh Modal `--smoke` run (`v1_projected` + `v2_m1`,
CIC-IDS-2017, seed=0) and an identical local `run_one_cell()` call, same
commit, were diffed field by field:

| | Local (Windows) | Modal (Linux) |
|---|---|---|
| `v1_projected` `final_all_acc` | `0.39803794272523985` | `0.39803794272523985` |
| `v1_projected` `bytes_sent` | `9,344,160` | `9,344,160` |
| `v2_m1` `final_all_acc` | `0.39803794272523985` | `0.39803794272523985` |
| `v2_m1` `bytes_sent` | `56,071,800` | `56,071,800` |
| `v2_m1` `comm_rounds` | `22` | `22` |

**Bit-identical on every field, on both arms.** local and Modal environments
resolve to identical pinned library versions (confirmed exactly, §2.7
above), so this is not a coincidence of two similar-looking floats — it is
the same computation. **Answer to the question as asked: yes — same code,
same pinned libraries, same seed, bit-identical results on every
deterministic field. The only difference is which physical machine executes
it**, plus the CPU-generation floating-point noise M7 already documents
(bounded, disclosed, and orders of magnitude below anything a published
number would report to).

---

## Table of contents

1. [Verdict and scoreboard](#1-verdict-and-scoreboard)
2. [What was verified green](#2-what-was-verified-green-by-execution)
3. [Blocking findings (B1–B6)](#3-blocking-findings)
4. [High-severity findings (H1–H6)](#4-high-severity-findings)
5. [Medium findings (M1–M9)](#5-medium-findings)
6. [Documentation audit](#6-documentation-audit)
7. [The ordered fix list](#7-the-ordered-fix-list)
8. [Method and reproducibility of this audit](#8-method-and-reproducibility-of-this-audit)

---

## 1. Verdict and scoreboard

**As found**, and after same-day remediation:

| Area | As found | After remediation | Detail |
|---|---|---|---|
| Core v2 algorithm (`aggregation.py`, `secagg.py`, `partition.py`) | ✅ **Sound** | ✅ unchanged | Exactness re-measured independently this pass; §2 |
| Test suite | ✅ **252 passed**, 444 s | ✅ **292 passed**, 516 s | +40 regression guards (31 in `test_pre_compute_audit_fixes.py`, 9 in `test_pipeline_invariants.py`); docs claiming 249 were stale (M3) |
| Communication instrumentation | ✅ **`6.0×` / `2.0×`** | ✅ + v1 `1×` now measured too | Measured on a real run, not from the formula; §2, H4 |
| Doc-link integrity | ✅ All links resolve | ✅ unchanged | `check_doc_links.py` exit 0 |
| Stage A/B correctness audits | ✅ exit 0 | ✅ exit 0 | Re-run after the `stats.py` fix — no published number moved |
| Literature/positioning (Stage C) | ✅ No defects surfaced | — | Out of this audit's scope |
| **Campaign driver** | ⛔ **Not launch-ready** | ✅ **Launch-ready** | B2, B3, M6, M8, M9 |
| **Modal execution layer** | ⛔ **Did not exist** | ✅ **`modal_frontier_campaign.py`** | B1 |
| **Dropout arm (T4)** | ⛔ **Crashes** | ✅ **Aborts the round, counts it** | B4 |
| **DP arm (T4)** | ⛔ **Wrong mechanism** | ✅ **Per-client, pre-SecAgg** | B5 |
| **NF-UQ-NIDS-v2 arm (T6)** | ⛔ **Infeasible** | ✅ **Fixed same day — B6 CLOSURE PASS below** | Real peak measured **2.65 GB** (was ~88 GB extrapolated); T6 wired end-to-end in both drivers |
| Stage C audit tooling | ⚠️ **Missing** | ⚠️ **Still missing** | H5 — `audit_stage_c.py` needs real T1 data to audit |
| Analysis pipeline (frontier figure) | ⚠️ **Missing** | ⚠️ **Still missing** | H6 — same dependency |

**Bottom line as found: T1 and T2 needed B1–B3; T4 needed B4 + B5 too; T6
needed B6, a genuine data-engineering task rather than a wiring fix.**
**Bottom line now (updated by the B6 CLOSURE PASS below, same day): T1, T2,
T4, and T6 are all launch-ready.** T6's loader fix and driver wiring are
complete and measured against the real file; only T3/T5 (external baseline
reproduction, out of this audit's scope) remain genuinely open work.

Four of the six blockers were **silent** — they did not raise, they produced
numbers. B2 produced a plausible frontier whose left-hand anchor was a
known-broken arm; B3 overwrote T1 results with T4 results and marked them
complete; B5 produced DP numbers that over-claim privacy by `√K`; B6 is the
only one that failed loudly (OOM). B4 failed loudly, but only after burning
compute.

**That silence is the finding behind the findings.** All six were present in a
tree whose suite was 252/252 green, whose exactness claim was genuinely
exact, and whose byte formula was genuinely correct. The mechanisms were
verified. The campaign path had never been run.

---

## 2. What was verified green, by execution

These are not inherited claims. Each was re-run this pass.

### 2.1 The exactness claim (F12) — holds

`scripts/verify_v2_exactness.py`, exit 0. The two-phase count-weighted
protocol reproduces the pooled `∇_P L'`:

| Cell | V1 naive sum | V3 corrected |
|---|---|---|
| UNSW-NB15 γ=1.0 | `1.599` | **`2.98×10⁻⁷`** |
| UNSW-NB15 γ=0.5 | `3.771` | **`3.02×10⁻⁷`** |
| Degenerate strata (6 cases, both datasets) | — | **`2.22×10⁻⁷` – `3.80×10⁻⁷`** |

The keystone contribution of FedGRAA-v2 is real and independently
re-derivable. **This is the part of the project that most deserves
confidence.**

### 2.2 The byte formula (T-12) — holds, measured on a real run

A purpose-built probe ran `FederatedTrainerV2` on real cached UNSW-NB15 data
for 10 rounds and read `bytes_sent` directly:

| Arm | Measured bytes | Baseline (`2·n_params·K·4·T`) | Ratio | Plan's §4.1 claim |
|---|---|---|---|---|
| `v2_m1` | 145,429,800 | 24,235,200 | **6.0006×** | 6× ✅ |
| `v2_m5` | 48,479,880 | 24,235,200 | **2.0004×** | 2× ✅ |

`comm_rounds` also matches exactly: `m=1` → 110 = `10 × (1 + 5×2)`; `m=5` →
30 = `10 × (1 + 1×2)`. The separate billing of bytes and *rounds* that F12
required is correctly implemented.

### 2.3 The compute budget survives contact with the v2 arms

**This was the audit's largest open risk, and it cleared.** Every Modal
calibration in `COMPUTE_INFRASTRUCTURE_2026-07-27.md` §5 measured the **v1**
`FederatedTrainer`. Eighty percent of T1/T2's cells are **v2** arms, which
add `⌈E'/m⌉` `two_phase_composite_grad` calls per round (per-client
forward+backward over each client's validation split) plus a `zeta_probe`
per eval round. If v2 were 2–3× more expensive, the campaign's cost and
wall-clock estimates would all be wrong.

Measured, same machine, same data, 10 rounds each:

| Arm | s/round | vs. v1 |
|---|---|---|
| `v1_projected` | 6.253 | 1.00× |
| `v2_m1` | 6.556 | **1.05×** |
| `v2_m5` | 6.044 | **0.97×** |

**v2 is not materially more expensive than v1.** The `~$26–29` / `~2.3–3 h`
figures in `COMPUTE_INFRASTRUCTURE_2026-07-27.md` §5.3/§6.3 stand for T1/T2.
(The per-client gradient work is small relative to the 200-round client
training loop, which dominates and is identical across arms.)

### 2.4 Plan claims re-derived

`scripts/verify_v2_plan_claims.py`, exit 0 — F3's DP-recovery table, F13's
UNSW `n≥5` unrecoverability, and the §7.2 `119 CPU-h` budget all reproduce.
`scripts/check_doc_links.py`, exit 0. `scripts/check_stage_c_complete.py`
correctly exits **1** on an empty results tree (the gate works; an earlier
`| tail` in this audit masked the true exit code — the gate itself is fine).

### 2.5 Test suite

```
252 passed, 366 warnings in 444.33s (0:07:24)
```

Matches `REMAINING_COMPUTE_WORK.md` §7 exactly. **Zero failures.**

---

## 3. Blocking findings

### B1 — ⛔ There is no Modal campaign driver. The docs say there is.

**Claimed.** `REMAINING_COMPUTE_WORK.md` §3 (line 75): *"T1, T2, and T4 can
run on Modal today, **unmodified**, using `scripts/run_v2_frontier_sweep.py`
and the validated pattern in `COMPUTE_INFRASTRUCTURE_2026-07-27.md`."*
`COMPUTE_INFRASTRUCTURE_2026-07-27.md` §7: *"nothing else is blocking it."*

**Measured.** Every file in the repository containing `modal.App` or
`@app.function`:

```
scripts/modal_calibration/modal_cic_memory_calibration.py
scripts/modal_calibration/modal_concurrency_probe.py
scripts/modal_calibration/modal_cpu_scaling_calibration.py
scripts/modal_calibration/modal_v2_gate_verification.py
```

All four are calibration/verification jobs. **None runs a campaign cell.**
`scripts/run_v2_frontier_sweep.py` mentions Modal only in comments; it has no
Modal import and no `@app.function`. The `.map()` wrapper both documents
describe as the next step **has not been written**.

**Two further gaps inside the same missing piece:**

1. **No result persistence.** `run_one_cell` writes to `cell.path`, a
   *relative* path inside the container filesystem, which is destroyed on
   exit. A Modal port must either mount a `Volume` or rely on `.map()`'s
   return values — the docs specify the Volume route (§3.1) but no Volume is
   declared anywhere.
2. **No data in the image.** `run_one_cell` calls `load_dataset(...,
   data_dir="data")`. The only *validated* image recipe
   (`modal_v2_gate_verification.py:28`) **excludes `data`**, because it only
   ran unit tests. See H2 for why the calibration scripts' recipe is now also
   wrong.

**Why it blocks.** This is the single item standing between the current tree
and a running campaign, and it is described in three documents as already
done.

---

### B2 — ⛔ The frontier's leftmost anchor is misconfigured: `η′=0.1`, not the specified `η′=0.01`

**The plan is explicit.** `FEDGRAA_V2_MASTER_PLAN.md` F1 (line 118):

> *"Everywhere this plan says 'v1' as v2's comparison reference, it means
> **v1-projected** (`projection + η′=0.01`, the Stage B **combined-fix**
> arm), not v1-literal."*

**The driver does not do this.** [`run_v2_frontier_sweep.py:184-188`](../../scripts/run_v2_frontier_sweep.py#L184-L188)
builds `FedConfig(...)` without `eta_prime`, so it takes the
[`config.py:49`](../../fedgraa/config.py#L49) default of **`0.1`**. There is
no `--eta-prime` CLI flag and no per-arm override anywhere in the driver.

**Why this is not a cosmetic difference.** `η′=0.1` **with** projection is
exactly the arm Stage B named **`projection-alone`**, and Stage B measured
what it does — [`fedgraa_convergence_defect.md:237`](../current/fedgraa_convergence_defect.md):

> *"on CIC-IDS-2017, `projection-alone` (η′ still at the paper's own 0.1)
> **stays at 0.0% rare-class accuracy at γ≤0.5** and only partially recovers
> at γ=1.0 (51.0%)"*

Against the combined fix's **69.7% / 81.1% / 89.3%** across γ.

**Consequence.** As configured, T1/T2 would produce a frontier whose
privacy-violating reference point scores **0.0% rare-class accuracy on
CIC-IDS-2017 at γ ∈ {0, 0.5}**. Every "v2 vs. v1" delta in the paper would be
measured against an arm the project's own Stage B campaign already
established as broken — and it would look like v2 winning enormously. This is
the most dangerous finding in this audit precisely because the resulting
numbers are *flattering*.

**Also unresolved:** the v2 arms inherit `η′=0.1` too. The plan says v2
subclasses `ProjectedFedGRAA` but never states v2's `η′`. This needs an
explicit decision, not a default.

---

### B3 — ⛔ T4 result files collide with, and silently overwrite, T1/T2 results

[`run_v2_frontier_sweep.py:93-99`](../../scripts/run_v2_frontier_sweep.py#L93-L99):

```python
@property
def path(self) -> str:
    ds_tag = self.dataset.replace("-", "_")
    return os.path.join(self.out_dir,
        f"{ds_tag}_{self.arm}_g{self.gamma}_seed{self.seed}.json")
```

`p_drop` and `dp_epsilon` are **not in the filename**. Verified by execution:

```
T1 path : results/v2_frontier\CIC_IDS_2017_v2_m1_g1.0_seed0.json
T4 pdrop: results/v2_frontier\CIC_IDS_2017_v2_m1_g1.0_seed0.json
T4 dp   : results/v2_frontier\CIC_IDS_2017_v2_m1_g1.0_seed0.json
COLLIDE : True
```

**Three compounding consequences:**

1. A T4 run **overwrites** the corresponding T1 result.
2. `missing_cells()` ([line 149](../../scripts/run_v2_frontier_sweep.py#L149))
   would then **skip** every T4 cell as "already complete," because a T1 file
   sits at its path — so a T4 campaign launched after T1 does nothing at all
   and reports success.
3. The result dict ([lines 209-223](../../scripts/run_v2_frontier_sweep.py#L209-L223))
   records `dataset, arm, gamma, seed, grad_refresh_every, rounds, all_acc,
   last_n_acc, zeta, final_all_acc, final_last_n_acc, bytes_sent,
   comm_rounds` — and **not** `p_drop`, `dp_epsilon`, `n_rare`, or
   `eta_prime`. So after the fact, a T4 result is **indistinguishable** from a
   T1 result by inspection. There is no way to recover which is which.

**Compounded by M9:** `--tier` only accepts `{T1, T2}`, so T4 must be
expressed as a T1 tier plus `--p-drop`/`--dp-epsilon` — precisely the
invocation that collides.

---

### B4 — ⛔ T4's dropout arm crashes deterministically under the production SecAgg threshold

[`trainer.py:549-552`](../../fedgraa/trainer.py#L549-L552) defaults the
threshold to a **majority**: `max(1, n_clients // 2 + 1)`. `SecAggChannel.aggregate()`
raises `SecAggDropoutError` whenever fewer than that many clients survive —
correct security behaviour (T-7), and fatal here because `run_one_cell`
([line 204](../../scripts/run_v2_frontier_sweep.py#L204)) constructs
`FederatedTrainerV2(split, cfg, measure_zeta=True)` with **no threshold
override**, and nothing catches the exception mid-run.

Computed from the binomial directly:

| Dataset | K | threshold | `p_drop` | P(a round fails) | **P(a 200-round cell completes)** |
|---|---|---|---|---|---|
| CIC-IDS-2017 | 14 | 8 | 0.1 | 0.0002 | 0.964 |
| CIC-IDS-2017 | 14 | 8 | **0.3** | **0.0933** | **3.1 × 10⁻⁹** |
| UNSW-NB15 | 9 | 5 | 0.1 | 0.0009 | 0.837 |
| UNSW-NB15 | 9 | 5 | **0.3** | **0.0988** | **9.2 × 10⁻¹⁰** |

**All 40 `p_drop=0.3` cells fail.** At `p_drop=0.1`, ~3.6% of CIC cells and
~16% of UNSW cells fail — a biased, non-obvious subset.

**The test suite does not catch this, and says so.**
[`tests/test_trainer_v2.py:141-146`](../../tests/test_trainer_v2.py#L141-L146)
carries the explanation in its own comment:

> *"the trainer's production-default threshold (majority, >n/2) is
> deliberately strict for security and **would legitimately raise
> SecAggDropoutError (T-7) at p_drop=0.3** on so few clients. T-13 is about
> robustness to dropout *given* a threshold chosen to tolerate it, so use a
> permissive threshold here explicitly."*

`test_dropout_robustness_end_to_end` passes only because it passes
`secagg_threshold=1`. **The campaign driver has no such parameter.** T-13
being green is therefore not evidence that T4 will run.

**This needs a design decision, not just a flag:** what threshold is
defensible for a dropout experiment? A permissive threshold weakens the very
security property F4 exists to enforce; a majority threshold makes
`p_drop=0.3` unrunnable at K=9–14. Either answer is publishable — silently
inheriting the majority default and watching 40 cells fail is not.

---

### B5 — ⛔ Differential privacy is applied in the wrong place, at `1/√K` of the modelled magnitude

**What the code does.** [`trainer.py:612-616`](../../fedgraa/trainer.py#L612-L616):

```python
counts_rank = ch_rank.aggregate()[0].astype(np.float64)   # exact SecAgg sum
if self.cfg.dp_epsilon is not None:
    counts_rank = np.clip(
        laplace_mechanism(counts_rank, self.cfg.dp_epsilon, rng=rng), 0.0, None)
```

One Laplace draw, applied **server-side, after the exact sum is already in
hand**. Delivered noise: `σ = (Δ₁/ε)·√2`.

**What every other artifact in the project models.** Per-client noise,
SecAgg-summed:

- [`privacy.py:69-82`](../../fedgraa/privacy.py#L69-L82) `aggregate_noise_std`:
  `σ = (Δ₁/ε)·√(2K)`
- `verify_v2_plan_claims.py:187` (the script F3's headline table comes from):
  `rng.laplace(0, b, (K, C)).sum(0)` — K independent per-client draws
- `tests/test_privacy.py:65-76` asserts `aggregate_noise_std` against the
  plan's own table (`10.58` at ε=1, K=14; `8.49` at ε=1, K=9)
- The plan's §9.2 **T-9** spec: *"Empirical σ within 10% of `(Δ₁/ε)·√(2K)`"*

**Two independent defects follow.**

**(a) Magnitude — the delivered noise is `√K` times too small.** At K=14
(CIC) that is **3.74×** less noise than the accountant assumes; at K=9
(UNSW), **3.00×**. Any ε reported from a T4 DP run would **over-claim
privacy** by that factor, because `composed_epsilon()` would be computed for
a mechanism the trainer does not implement.

**(b) Trust model — noising after aggregation protects nothing from the
server.** F4 is unambiguous that the **server** is the adversary
(`FEDGRAA_V2_MASTER_PLAN.md` F4: *"A server that sees any individual `g_k`
can therefore test, in one dot product, whether client `k` holds rare-class
data"*). In the implemented flow, the server computes the **exact**
histogram sum and *then* adds noise to its own copy. Against the stated
adversary this is a no-op. The correct placement is per-client noise added
**before** SecAgg submission (local/distributed DP), which is also exactly
what makes `σ = (Δ₁/ε)·√(2K)` the right formula.

**The tell.** `aggregate_noise_std` is exported, documented, and covered by
four tests — and is **never called by any production code path**
(`grep` confirms: only `privacy.py` and `test_privacy.py`). A function that
models the deployed mechanism but is never invoked by it is the signature of
exactly this class of drift.

**Consequence.** Every DP number in T4 would be wrong in a direction that
flatters the paper. Given that "the assumption is removable at an exactly
measured cost" is the contribution, a mis-stated privacy cost is not a
detail.

---

### B6 — ✅ **FIXED, and measured on the real file.** The NF-UQ-NIDS-v2 loader was going to need ~88 GB peak; a bounded-memory rewrite measures 2.65 GB

**Was.** `data/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.csv` is **12.79 GB**, 46 columns,
75,987,976 rows. Sampling 1,000,000 real rows gave a deep pandas footprint
of **578.2 bytes/row** and **41 numeric feature columns** after the loader's
drops, extrapolating to a **~88 GB peak** at the concat step against a
23.3 GB development machine — the original finding, and the reason
`REMAINING_COMPUTE_WORK.md`'s *"✅ obtained, verified, and **loadable**"* and
`nf_uq_nids_v2_provenance_audit.md` §7's *"unblocked at the code level"* did
not follow from their own §6 finding (*"the grouped per-silo frames … still
accumulate in full"*).

**Fix.** [`fedgraa/data.py`](../../fedgraa/data.py) gained a bounded-memory
path, used when a caller passes `max_per_class`: an exact chunked tally of
every `(silo, label)` key's row count
(`_tally_merged_csv`), a deterministic seeded selection of up to
`max_per_class` row *positions* per key computed from the counts alone
(`_select_positions_per_key` — `numpy.random.Generator.choice(n, size=k,
replace=False)` was measured, not assumed, to be O(k) time/memory even at
`n=30,000,000`), and a second chunked pass that keeps only the selected
positions (`_read_merged_csv_selected_rows`). Peak memory is now
`O(chunksize) + O(n_keys × max_per_class)` — bounded by the cap, never by
the file's true size. The old unbounded path is preserved byte-for-byte for
every existing (small-file) caller, now behind a `_UNCAPPED_ROW_GUARD`
safety check that raises rather than silently repeating the ~88 GB failure
if a future caller forgets to pass a cap.

**Measured, not estimated, against the actual file**
(`scripts/verify_nf_uq_nids_v2_memory.py`, Windows peak-working-set via
`GetProcessMemoryInfo`, `max_per_class=10_000`, matching
`DatasetSpec.frequent_cap`'s existing convention):

```
elapsed              :   1041.5 s  (~17.4 min, two full chunked passes)
PEAK WORKING SET     :   2714.5 MB  (2.65 GB)
labels (n=21)        : 20 attack categories + Benign -- matches the
                        provenance audit's independently-verified count
n_features            : 41
total selected rows   : 255,864  (NF-BoT-IoT-v2: 42,431 / NF-ToN-IoT-v2: 91,148
                         / NF-UNSW-NB15-v2: 61,853 / NF-CSE-CIC-IDS2018-v2: 60,432)
```

**2.65 GB measured vs. ~88 GB extrapolated for the uncapped path — a ~33×
reduction**, comfortably inside a normal development machine and well under
the smallest Modal container tier used elsewhere in this campaign. One
transient `RuntimeWarning: overflow encountered in cast` (pandas-internal,
during block consolidation of a mixed-dtype NetFlow byte-counter column)
was observed once during the run; investigated separately by reproducing
the read on a smaller slice and checking `np.isfinite` directly — no `inf`
or `NaN` reached `X`, and the loader's existing `np.place(X, ~np.isfinite(X),
0.0)` cleanup would neutralize it regardless. Noted for transparency, not
treated as a defect.

**T6 is now fully wired, not just the loader.** `run_v2_frontier_sweep.py`
gained a `T6` tier (`enumerate_cells` builds `NF_UQ_NIDS_V2_DATASET` cells
across all five arms at γ=1.0) and `run_one_cell` gained a complete
NF-UQ-NIDS-v2 branch: `load_nf_uq_nids_v2_silos` + `partition_by_silo`,
with `DatasetSpec.used_counts`/`benign_id` derived from what the loader
measured, never hardcoded (F3/F7's own discipline, applied here for the
first time to a real rather than synthetic dataset).
T6's Modal driver is
[`scripts/modal_frontier_campaign_t6.py`](../../scripts/modal_frontier_campaign_t6.py)
— a genuinely **separate file and App**, not a second image on the T1/T2/T4
App. (The first version *was* a second image on the same App, and that
turned out to be a real bug — a T1-only launch built and uploaded the
12.79 GB file too, caught by timing the launch rather than by reading the
code; see the "second new defect" box earlier in this document for the
measurement and the fix.) It has its own `run_cell_t6` function sized from
the measurement above and its own `verify_env_t6` pre-flight that checks the
CSV *file* is present, not just the directory.

**End-to-end, not just the loader in isolation.**
[`tests/test_run_one_cell_nf_uq.py`](../../tests/test_run_one_cell_nf_uq.py)
runs `run_one_cell` through both arms on a small synthetic class-separable
merged CSV — the exact code path Modal will execute for T6, minus the 76M
rows, so it runs in the normal suite in seconds rather than needing the real
file.

**Remaining, honestly scoped:** `partition_by_silo` creates **one client per
silo** — NF-UQ-NIDS-v2 has **4**, against T1's **K=14** (CIC) / **K=9**
(UNSW). T6 is documented as "T1's own grid, applied to the one real-silo
dataset" rather than "T1 repeated," since the client count is structurally
different and that is arguably the *point* of a real (not synthetic)
cross-silo benchmark, not an oversight.

---

## 4. High-severity findings

### H1 — ✅ Resolved by measurement: the v2 arms do not blow the compute budget

Recorded here because it was the audit's largest open risk and the answer is
load-bearing. **Every** Modal calibration measured the v1 trainer; 80% of
T1/T2 cells are v2. Measured ratio: **1.05× at m=1, 0.97× at m=5** (§2.3).
The cost and wall-clock estimates hold. **No action needed — but this should
be recorded in `COMPUTE_INFRASTRUCTURE_2026-07-27.md` §5, which currently
carries a v1-only measurement presented as the campaign's per-unit cost.**

### H2 — The validated Modal image recipe now uploads 13.9 GB

`COMPUTE_INFRASTRUCTURE_2026-07-27.md` §4.2's exclusion table lists `data` at
**889 MB** and, on that basis, the calibration scripts'
`_ignore` (`modal_cic_memory_calibration.py:40`) deliberately **includes**
`data/` in the image — the campaign needs it.

`data/` is now **13.9 GB**, because NF-UQ-NIDS-v2.csv (12.79 GB) landed in it:

| Directory | Size |
|---|---|
| `data/CIC-IDS-2017` | 844 MB |
| `data/UNSW-NB15` | 45 MB |
| `data/NF-UQ-NIDS-v2` | **13,093 MB** |

So **neither** existing recipe is correct for the campaign: the
gate-verification recipe excludes `data` entirely (no datasets), and the
calibration recipe would now push a 13.9 GB image for a campaign that needs
889 MB of it. The campaign image must exclude `data/NF-UQ-NIDS-v2`
specifically.

### H3 — ✅ **FIXED.** Dependencies were unpinned; the reproducibility baseline was not enforced

`requirements.txt` and both Modal image recipes use `>=` bounds only. The
recorded baseline (Python 3.13.14 / TF 2.21.0 / NumPy 2.5.1 / SciPy 1.18.0)
is what `pip` **happened** to resolve on the day the image was built —
`COMPUTE_INFRASTRUCTURE_2026-07-27.md` §3 confirms this by quoting the build
log, not a lockfile. A cache-busting rebuild mid-campaign would silently run
later cells against different library versions, with no error and no record.
Locally, `pandas` has already drifted to **3.0.3** against a `>=2.0` floor —
a major version.

For a project whose entire discipline is "measure, don't assume," running the
headline campaign on unpinned dependencies is the one remaining place where
an unrecorded variable can enter the results.

**✅ Fixed.** `scripts/modal_frontier_campaign.py` pins exact versions
(`PINNED`) and a `verify_env` pre-flight **asserts** the container resolved
them before any cell runs — a mismatch raises rather than silently producing
results from an unrecorded environment. Confirmed on a real Modal build:

```
Successfully installed … numpy-2.5.1 … pandas-3.0.3 … scikit-learn-1.9.0
                        scipy-1.18.0 … tensorflow-2.21.0
```

Every cell's result JSON also records its own resolved versions and commit
hash (M6), so the environment is recoverable per result even if the image
recipe later changes.

### H4 — The frontier's x-axis has no value for its leftmost point

`run_one_cell` sets `bytes_sent = None` and `comm_rounds = None` for
`v1_projected` ([lines 193-206](../../scripts/run_v2_frontier_sweep.py#L193-L206)),
because `FederatedTrainer` carries no instrumentation. But
`REMAINING_COMPUTE_WORK.md` §6.2 specifies the frontier figure as *"x = total
training traffic in GB from `bytes_sent`"* — so the `1×` anchor point has no
measured x-coordinate. Either add the same billing to `FederatedTrainer`
(preferred: it makes the `1×` claim a measurement, exactly as T-12 made the
`6×` claim one) or record the analytic derivation explicitly in the results.

### H5 — Stage C has no correctness audit, and no CI gate

Stage A and Stage B each have an `audit_stage_*.py` that **recomputes every
transcribed number from the raw persisted JSON**, and
`.github/workflows/docs-check.yml` runs both on every push. Stage C has:

- ✅ `check_stage_c_complete.py` — completeness only ("is every artifact
  present and shaped right?")
- ❌ no `audit_stage_c.py` — the "are the numbers actually true?" half
- ❌ no CI wiring for either

The README describes this two-tier discipline as the project's core quality
claim. The stage about to produce the paper's headline numbers is the one
stage it does not cover.

### H6 — No frontier-figure builder exists

Acknowledged honestly in `REMAINING_COMPUTE_WORK.md` §6.1 (*"genuinely cannot
be built or verified before real T1 data exists"*). Flagged here only because
it sits on the critical path between "campaign finishes" and "paper," and
because the v1 analogues (`build_ablation_table.py`, `render_ablation.py`)
show what it needs to be.

---

## 5. Medium findings

| # | Finding | Evidence |
|---|---|---|
| **M1** | ✅ **FIXED.** **Four specified tests did not exist.** `FEDGRAA_V2_MASTER_PLAN.md` §9.2 lists **T-14/15** (seed-count and significance-reporting invariants) and **T-16/17** (property-based: additivity over random partitions, `Δ₁=2` for neighbouring histograms) as *"Unchanged"*. No test referenced them, so `REMAINING_COMPUTE_WORK.md` §1's *"Every one of §9.2's T-4 through T-19 test IDs has a corresponding, passing test"* was **false**. Now implemented in [`tests/test_pipeline_invariants.py`](../../tests/test_pipeline_invariants.py) — **implemented rather than withdrawn, and writing T-15 immediately found a live `paired_ttest` bug** (see the remediation box above). `hypothesis` was deliberately **not** added: the same no-new-unvetted-dependency stance `privacy.py` took toward `dp-accounting`/`opacus`; both properties are covered by seeded randomized loops (200 cases each) that are reproducible without a shrinking engine. | Was: `grep -rhoE "T-[0-9]+" tests/` yielded T-4…T-13, T-18, T-19 only |
| **M2** | ✅ **FIXED.** **`FedConfig.secagg` was dead code.** [`config.py`](../../fedgraa/config.py) documents it as *"F4: SecAgg … is mandatory"*, but it was **never read** — `SecAggChannel` is constructed unconditionally, so `secagg=False` changed nothing while reading as though it disabled SecAgg. `__post_init__` now **rejects** `False` with an explanation, since F4 establishes there is no supported way to turn it off. | Was: `grep -rn "cfg.secagg"` → no hits |
| **M3** | ✅ **FIXED.** **Stale test counts.** README (×3) and `MENTOR_WALKTHROUGH.md` said **249**; measured **252**. `REMAINING_COMPUTE_WORK.md` §7 also said *"97 new tests"* and *"Two of the 94"* two paragraphs apart. | Suite run this pass |
| **M4** | ✅ **FIXED.** **Arithmetic/logic error.** `REMAINING_COMPUTE_WORK.md` §3: *"T1, T2, and T4 can run on Modal today … That is **130 CPU-hours** of the plan's **119-CPU-hour** 'minimum viable campaign' (**T1–T3**)."* T1+T2+T4 = 28+57+28 = **113**, not 130 — and **T4 is not a member of T1–T3**, so it cannot be a subset of that budget. Corrected in place, original struck rather than deleted. | §7.2 table |
| **M5** | **All Stage C work is uncommitted.** 41 paths dirty: 16 untracked `.py` files (including `fedgraa/privacy.py`, `fedgraa/secagg.py`, all six new test modules, and every campaign/Modal script) plus 9 untracked docs and 19 modified tracked files. Nothing is `.gitignore`d — it has simply never been committed. **A campaign launched from this tree has no commit hash to record in the reproducibility manifest**, and a machine failure loses the work. | `git status --short` |
| **M6** | ✅ **FIXED.** **Result JSONs were not self-describing.** No `eta_prime`, `n_rare`, `p_drop`, `dp_epsilon`, `secagg_threshold`, library versions, or commit hash — compounding B3 (a T4 result could not be told from a T1 one) and H3 (no record of what was installed). Every cell now carries a `config` block and a `provenance` block (commit + dirty flag + Python/TF/NumPy/SciPy/pandas/sklearn versions), captured best-effort so provenance can never fail a multi-hour run. | [`run_v2_frontier_sweep.py`](../../scripts/run_v2_frontier_sweep.py) |
| **M7** | ⚠️ **OPEN, and sharpened by a new measurement.** `COMPUTE_INFRASTRUCTURE_2026-07-27.md` §4.4 records `test_fedgraa_weight_gradient_matches_finite_difference` missing `atol=5e-3` by `0.00537` on a Modal container with *"AVX2, FMA only"*, and treats that as a property of Modal. **It is not — Modal's fleet is heterogeneous.** This audit's `verify_env` pre-flight, run on the real campaign image, reported **`avx512: True`**. So a 300-cell campaign will land on a *mix* of CPU generations, and §6.5's reasoning ("same-hardware, same-run internal comparisons, so the difference should cancel") holds **within** a cell but not **across** them. Consequence: (a) `docs-check.yml`'s `stage-gates` job runs the suite on `ubuntu-latest`, whose CPU generation is not pinned, so this can fail CI non-deterministically; (b) cross-cell numerical noise from CPU-generation differences is real and should be **disclosed in Limitations** alongside the `ζ` decomposition, not assumed away. The container's Python is also **3.13.3**, against the local baseline's **3.13.14** — a patch-level difference the `PINNED` check does not currently cover. | §4.4; this audit's `verify_env` output |
| **M8** | ✅ **FIXED.** **Dead assignment.** `spec = get_dataset_spec(cell.dataset)` was immediately overwritten by `load_dataset`'s third return value six lines later. | [`run_v2_frontier_sweep.py`](../../scripts/run_v2_frontier_sweep.py) |
| **M9** | ✅ **FIXED.** **`--tier` accepted only `{T1, T2}`** despite T4 being described as driver-supported, so T4 had to be expressed as a T1 tier plus `--p-drop`/`--dp-epsilon` — the exact invocation that triggered B3's collision. `T4` is now a first-class tier enumerating the plan's own 100 cells, with `--t4-m` selecting the arm. | [`run_v2_frontier_sweep.py`](../../scripts/run_v2_frontier_sweep.py) |

---

## 6. Documentation audit

### 6.1 What is in good shape

- **All relative links resolve** (`check_doc_links.py`, exit 0, enforced in CI).
- **The Option A/B/C/D scope confusion is fully resolved.** It was corrected
  three times in one day, and the corrections were propagated consistently to
  `Final_Stage_C.md` §10, `STAGE_DIRECTION_ASSESSMENT.md`, `MENTOR_WALKTHROUGH.md`,
  `README.md`, and `REMAINING_COMPUTE_WORK.md`. Dated snapshots were
  deliberately left alone. **No live contradiction remains** — this was
  checked directly and is a genuinely good outcome for a scope change of that
  size.
- **The correction-in-place discipline is real.** F3 records that its own
  first draft was refuted by measurement rather than deleting it; F12 records
  that the specification, not the test, was wrong. This is unusually honest
  and should be preserved.
- **Superseded documents are clearly marked** (`PAKDD_STATUS_REPORT.md`'s
  header, the `docs/archive/` tree).

### 6.2 The systematic problem: readiness claims outrunning the code

Every blocking finding in §3 has a corresponding document asserting the
opposite. This is not scattered staleness; it is one pattern:

| Document | Claim | Reality |
|---|---|---|
| `REMAINING_COMPUTE_WORK.md` §3 | *"T1, T2, and T4 can run on Modal today, unmodified"* | No Modal driver exists (B1); T4 collides (B3) and crashes (B4) |
| `COMPUTE_INFRASTRUCTURE_2026-07-27.md` §7 | *"nothing else is blocking it"* | B1, B2 |
| `REMAINING_COMPUTE_WORK.md` §3 (T4 row) | *"**None** — `p_drop`/`dp_epsilon` … already wired into the driver"* | Wired, but collides (B3), crashes (B4), and mis-implements DP (B5) |
| `REMAINING_COMPUTE_WORK.md` §5 | *"✅ obtained, verified, and **loadable**"* | ~88 GB peak; never run on the real file (B6) |
| `nf_uq_nids_v2_provenance_audit.md` §7 | *"T6 … unblocked at the code level"* | Contradicted by its own §6 |
| `REMAINING_COMPUTE_WORK.md` §1 | *"Every one of §9.2's T-4 through T-19 test IDs has a corresponding, passing test"* | T-14/15/16/17 have none (M1) |

**The common cause is a definition drift: "the mechanism is implemented and
unit-tested" was recorded as "the experiment can run."** Those are different
claims, and the gap between them is precisely §3. The unit tests are
excellent and they pass — but the *campaign path* (driver → config → Modal →
persisted result) has never been executed end-to-end for any arm, and the one
place the tests do touch it (T-13) had to override a production default to
get green (B4).

**Recommendation:** adopt a single explicit distinction in the docs —
**"mechanism verified"** vs. **"campaign path verified"** — and never let the
first be written as the second. A one-line "campaign path verified: no" in
`REMAINING_COMPUTE_WORK.md` §2's gate table would have surfaced all six
blockers weeks earlier.

### 6.3 Stale values to correct

| Location | Says | Should say |
|---|---|---|
| `README.md:32`, `:256`, `:368` | 249 tests | **252** |
| `MENTOR_WALKTHROUGH.md:48` | 249/249 | **252/252** |
| `REMAINING_COMPUTE_WORK.md` §7 | "Two of the 94" | 97 (or reconcile) |
| `REMAINING_COMPUTE_WORK.md` §3 | "130 CPU-hours … of the … 119-CPU-hour (T1–T3)" | 113 CPU-h, and T4 ∉ T1–T3 |
| `COMPUTE_INFRASTRUCTURE_2026-07-27.md` §4.2 | `data` = 889 MB | **13.9 GB** (H2) |
| `README.md` verification block | six commands | add `check_stage_c_complete.py` |

---

## 7. The ordered fix list

Ordered by what unblocks the most compute soonest. Each item is scoped so it
can be finished, not merely started.

> **Status, same day.** **Phase 1 items 1–4 and Phase 3 items 11–12 are
> done**, plus Phase 2 items 7, 8, 9 and 10. What remains: item 5 (the Modal
> smoke test and launch — an operational step, not a code one), item 6
> (`audit_stage_c.py`, which needs real T1 output to audit), and all of
> Phase 4 (**B6**). Each item below carries its own status.

### Phase 1 — launch T1/T2 (est. ~1 day) — ✅ **items 1–4 done**

1. ✅ **DONE — Decide and set `η′` explicitly (B2).** Add `--eta-prime` to the driver
   and set `eta_prime` on the `FedConfig` in `run_one_cell`. Recommended:
   `η′=0.01` for `v1_projected` to match the plan's stated combined-fix
   reference, and an explicit — not defaulted — choice for the v2 arms.
   Record the decision in the master plan; this is the one item that changes
   what the paper's headline comparison *means*.
2. ✅ **DONE — Put the full configuration in the filename and the result JSON (B3, M6).**
   Extend `CellSpec.path` with `p_drop`/`dp_epsilon` (and `eta_prime` if it
   varies), and add every config field, the library versions, and the commit
   hash to the result dict. Add a regression test asserting that two
   `CellSpec`s differing in any experimental field have different paths.
3. ✅ **DONE — Write the Modal campaign driver (B1).** `@app.function` around
   `run_one_cell`, `.map()` over `enumerate_cells(...)`, `cpu=1.0`,
   `memory=12288` (CIC) / `4096` (UNSW), forced `spawn`, results persisted to
   a `modal.Volume`, image excluding `data/NF-UQ-NIDS-v2` (H2) and pinning
   exact versions (H3).
4. ⏳ **Commit everything (M5)** so the campaign has a reproducible commit hash.
5. ⏳ **Smoke-test one real cell per arm on Modal** (`--rounds 2`), confirm the
   JSON lands in the Volume and `check_stage_c_complete.py` sees it, **then**
   launch T1.

### Phase 2 — make the results trustworthy (parallel with the campaign)

6. ⏳ **Write `audit_stage_c.py` (H5)** — genuinely blocked on real T1 output to audit. on the `audit_stage_b.py` pattern, and
   wire both it and `check_stage_c_complete.py` into `docs-check.yml`.
7. ✅ **DONE — Instrument `FederatedTrainer`'s bytes (H4)** so the frontier's `1×`
   anchor is measured rather than asserted.
8. ✅ **DONE — Correct the documentation (§6.3)** and add the "mechanism verified" vs.
   "campaign path verified" distinction (§6.2).
9. ✅ **DONE — Landed T-14/15/16/17 (M1)**, not withdrawn — and doing so found the `paired_ttest` bug; either is fine, the current
   state is not.
10. ✅ **DONE (M2, M8, M9)** · ⏳ **M7 open** — Fix M2, M7, M8, M9 — small, and M7 is a live CI flake.

### Phase 3 — T4 (needs decisions, not just code)

11. ✅ **DONE — Choose the dropout threshold policy (B4).** The majority threshold is kept (it is the security-correct one); an unrecoverable round now aborts and is counted rather than killing the job, which is the real protocol's own semantics and makes the abort *rate* a reportable measurement. `--secagg-threshold` exposed for explicit override. and expose it on the driver.
12. ✅ **DONE — Move DP noise to the client, pre-SecAgg (B5).** Each participant adds
    `Laplace(0, Δ₁/ε)` before `submit()`, making the aggregate noise
    `(Δ₁/ε)·√(2K)` — the quantity `aggregate_noise_std`, `test_privacy.py`,
    and F3's headline table already model. Then add the test that would have
    caught this: assert the *trainer's* end-to-end noise matches
    `aggregate_noise_std`, not just that the standalone function does.

### Phase 4 — T6 (a real data-engineering task)

13. **Streaming mean/std, float32, and a documented subsampling policy (B6);**
    then a real end-to-end run against the 12.79 GB file before any T6 cell
    is scheduled.
14. **Wire NF-UQ-NIDS-v2 into `enumerate_cells`/`run_one_cell`,** and
    **document the K=4 vs K=14/9 change** as a scope difference rather than
    "T1 repeated."

**T3/T5 (fourteen external baseline reproductions) remain the largest single
piece of remaining work and are unaffected by this audit** — they are
correctly scoped in `REMAINING_COMPUTE_WORK.md` §4 and were not re-examined
here.

---

## 8. Method and reproducibility of this audit

Every finding above is re-derivable. Commands, in order:

```powershell
# Suite state (252 passed, 444s)
.venv313\Scripts\python.exe -m pytest tests -q

# The verified-green claims (2.1, 2.4)
.venv313\Scripts\python.exe scripts\verify_v2_exactness.py       # exit 0
.venv313\Scripts\python.exe scripts\verify_v2_plan_claims.py     # exit 0
.venv313\Scripts\python.exe scripts\check_doc_links.py           # exit 0
.venv313\Scripts\python.exe scripts\check_stage_c_complete.py    # exit 1 (correct)

# B3 -- path collision
.venv313\Scripts\python.exe -c "from scripts.run_v2_frontier_sweep import CellSpec; a=CellSpec('CIC-IDS-2017','v2_m1',1.0,0); b=CellSpec('CIC-IDS-2017','v2_m1',1.0,0,p_drop=0.3); print(a.path==b.path)"

# B4 -- dropout failure probability
.venv313\Scripts\python.exe -c "from scipy.stats import binom; [print(K,p,binom.cdf(K//2,K,1-p)) for K,p in [(14,0.3),(9,0.3)]]"

# B5 -- the modelled function is never called
grep -rn "aggregate_noise_std" fedgraa/ scripts/

# B6 -- real per-row footprint
.venv313\Scripts\python.exe -c "import pandas as pd; c=next(pd.read_csv('data/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.csv',skipinitialspace=True,chunksize=1_000_000)); print(c.memory_usage(deep=True).sum()/len(c))"
```

The timing/byte probe used for §2.2 and §2.3 is reproduced in
`scripts/` as part of Phase 1 item 3's smoke test, or can be re-derived from
`FederatedTrainerV2.bytes_sent` / `.comm_rounds` on any short run.

**AI disclosure.** This audit was performed by an AI assistant (Claude Opus 5)
at the repository author's direction, in a single pass on 2026-07-28. Every
"blocking" claim is backed by an execution result or an arithmetic derivation
shown inline; no finding is asserted from reading alone. Where a document and
the code disagreed, the code was run to decide. No campaign cell was executed
and no Modal job was launched during this audit. The `~88 GB` figure in B6 is
an extrapolation from a measured 1,000,000-row sample, not a completed load —
stated as such because completing the load is precisely what the finding says
is infeasible on available hardware.
