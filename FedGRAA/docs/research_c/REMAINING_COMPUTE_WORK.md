# FedGRAA-v2 — What's Left, Now That the Code Is Done

> ## ⚠️ CORRECTED 2026-07-28 by [`../audit/PRE_COMPUTE_AUDIT_2026-07-28.md`](../audit/PRE_COMPUTE_AUDIT_2026-07-28.md) — read this box before §3
>
> A whole-repository audit run immediately before launching the campaign found
> that **this document conflated two different claims**: *"the mechanism is
> implemented and unit-tested"* and *"the experiment can run."* Everything
> below about the first is accurate and was re-verified. Several statements
> about the second were **not**, and are corrected here rather than quietly
> rewritten:
>
> | This document said | Actually | Now |
> |---|---|---|
> | §3: *"T1, T2, and T4 can run on Modal today, **unmodified**"* | **No Modal driver existed.** Only four calibration/gate scripts contained `@app.function`; none ran a campaign cell (audit **B1**) | ✅ Fixed — [`scripts/modal_frontier_campaign.py`](../../scripts/modal_frontier_campaign.py) |
> | §3 T4 row: *"**None** — `p_drop`/`dp_epsilon` … already wired into the driver"* | Wired, but **(a)** T4 cell paths **collided with and overwrote** T1's, and `missing_cells()` then skipped every T4 cell as complete (**B3**); **(b)** `p_drop=0.3` crashed on the majority SecAgg threshold with P(a 200-round cell finishing) ≈ 3×10⁻⁹ (**B4**); **(c)** DP was applied server-side *after* aggregation at 1/√K the modelled magnitude (**B5**) | ✅ All three fixed |
> | (unstated) The campaign's η′ | The driver inherited `FedConfig`'s **0.1**, not the **0.01** combined-fix value F1 requires — i.e. Stage B's `projection-alone` arm, measured at **0.0%** rare-class accuracy on CIC at γ≤0.5 (**B2**) | ✅ Pinned to 0.01 for both arms |
> | §1: *"Every one of §9.2's T-4 through T-19 test IDs has a corresponding, passing test"* | **T-14/15 and T-16/17 have none** (**M1**) | ⏳ Open — see the audit's fix list |
> | §5: NF-UQ-NIDS-v2 *"✅ obtained, verified, and **loadable**"* | Loader needs **~88 GB peak** on the real 12.79 GB file (measured); never executed against it (**B6**) | ✅ **Fixed and measured, same day** — bounded-memory two-pass loader, real peak **2.65 GB** on the actual file, T6 wired end-to-end. §5 below is corrected in place again |
>
> **What the audit re-verified as sound, by execution:** F12's exactness
> (`1.2×10⁻⁷`–`3.8×10⁻⁷`), the `6×`/`2×` byte formula (measured on a real
> run, not derived), and the suite (252 green, then 292 after the audit's
> first remediation pass, **now 311** after B6's fix and T6's wiring).
> **The science was never the problem — the harness was.**
>
> **Standing rule adopted as a result:** never write *"mechanism verified"* as
> *"campaign path verified."* §2's gate table now carries both.

> **✅ Scope correction, 2026-07-28 (later same day), itself corrected shortly after.** This document briefly recorded the decided Stage C scope as **"Option B+D"** ([`Final_Stage_C.md`](Final_Stage_C.md) §10's lettering: **B** = privacy-compliant redesign/FedGRAA-v2, **D** = NF-UQ-NIDS-v2 dataset migration) — the literature review's own recommendation, adopted as-is — then briefly over-corrected to "implement A, B, C, and D all in parallel." **Neither is right. The author has clarified the actual decision is `Final_Stage_C.md` §10's own Option C: "both, sequenced" — B+D now (already underway, unchanged), Option A (the few-shot/prototype-guided hybrid) later, as a follow-on, not started in parallel and not currently in scope for this pass.** Concretely: **A** = the few-shot/prototype hybrid, a second, independent novelty contribution, deliberately **not** being worked on right now; **C** = the sequencing decision itself (ship B+D, then layer A on afterward) — which is what's actually chosen, not a fourth simultaneous workstream. **B and D's status is unchanged throughout all of this** — everything §2–§7 below already documents for them stands, and remains the entirety of this pass's active scope. §3a below still exists, but its purpose is now to record Option A's scope **for the later follow-on phase**, not as work to pick up now.
>
> **What this document is.** The single, honest map of what remains after this pass implemented [`FEDGRAA_V2_MASTER_PLAN.md`](FEDGRAA_V2_MASTER_PLAN.md)'s W0–W6 workstreams end to end (§2 below) — that is, Option B in full. Everything left for **B** is compute, external-baseline reproduction, or real-data acquisition — no algorithm design, mechanism, or test-plan work remains undone *for B*. Option **A** (§3a) is a different, deliberately-deferred body of work, sequenced after B+D per Option C: it needs algorithm design before it needs compute, and that design work has not started and is not scheduled to start this pass. This document exists so the next session (or the mentor conversation, [`FEDGRAA_V2_MASTER_PLAN.md`](FEDGRAA_V2_MASTER_PLAN.md) §12.3) can pick up exactly where this pass left off, on Modal for B+D now, and on Option A's design once B+D's compute campaign is far enough along to free up that time, without re-deriving scope.
>
> **What this document is not.** It is not a new design document for B/D — every mechanism §2 references already exists, is tested, and is cited by file path. It supersedes nothing in [`FEDGRAA_V2_MASTER_PLAN.md`](FEDGRAA_V2_MASTER_PLAN.md); it records that plan's implementation status. §3a *is* new scope-tracking for Option A's eventual follow-on phase, since no other document owns that yet — it records what A will need, not work in progress.
>
> **Compiled:** 2026-07-28, against a working tree whose full test count is recorded in §7 below, from this pass's last full-suite run, across 6 wholly new test files (`test_secagg.py`, `test_privacy.py`, `test_aggregation_v2.py`, `test_partition_v2.py`, `test_trainer_v2.py`, `test_run_v2_frontier_sweep.py`) plus additions to `test_data.py` and `test_reproduction_gaps.py`, covering every W0–W3 mechanism this plan called for. **Updated later the same day**, twice: first to (incorrectly) broaden scope to "A+B+C+D in parallel," then corrected to Option C's actual meaning — B+D now, A sequenced later — above and in §3a.

---

## 1. Bottom line

**Every piece of FedGRAA-v2 that is *code* — as opposed to *compute* or *external-baseline reproduction* — is implemented, tested, and green.** The two-phase exact protocol (F12's fix) reproduces the pooled gradient to `~2×10⁻⁷` relative error on both datasets at three γ values, matching [`scripts/verify_v2_exactness.py`](../../scripts/verify_v2_exactness.py)'s own measurement exactly. The privacy-compliant rarity ranking (F3/F7) recovers the paper's hard-coded rare set exactly on every seed it can (and is proven, not just observed, to be unrecoverable exactly where F13 said it would be). SecAgg's individual-contribution-unreachability (F4) is enforced structurally and checked by introspection, not by convention. Every one of §9.2's T-4 through T-19 test IDs has a corresponding, passing test. **✅ Update, 2026-07-28, same day:** real NF-UQ-NIDS-v2 data was obtained and provenance-verified (`docs/audit/nf_uq_nids_v2_provenance_audit.md`), the loader/layout gap that audit found is now closed (§5 below), `scripts/check_stage_c_complete.py` (§6) now exists, and G3/G4's test files were re-run and confirmed green on the actual Modal image (§3.2). **What is left is: running the compute campaign (T1–T7) and reproducing ~14 other papers' methods as baselines.** Neither can be done by writing more FedGRAA-v2 code — the first requires machine-hours (deferred to Modal, per the author's own instruction and per [`COMPUTE_INFRASTRUCTURE_2026-07-27.md`](COMPUTE_INFRASTRUCTURE_2026-07-27.md)'s already-validated setup — the campaign itself, T1–T7, is *still* not launched as of this update; only its test-verification and tooling prerequisites were addressed), the second requires external literature reproduction this session's tools cannot shortcut (no live web access was used to build any of the code in this pass; see §4).

**Also decided, 2026-07-28 (same update, corrected twice later the same day):** the author has resolved the mentor-conversation decision `STAGE_DIRECTION_ASSESSMENT.md` flagged as the standing blocker on Stage C's scope — see that document's new resolution note and §4 below. The decision is `Final_Stage_C.md` §10's **Option C — "both, sequenced"**: implement B (privacy redesign) and D (dataset migration) now, which is already underway per §2–§6 and unaffected by this note, and pursue A (the few-shot/prototype hybrid) **afterward, as a follow-on**, not in parallel and not starting this pass — see §3a for what A will need when that phase begins. The specific T3/T5-vs-external-compute trade-off (§4) is deliberately deferred until real T1/T2 frontier numbers are in hand, not decided now; that deferral is unaffected by this correction.

> ✅ **Dated correction, 2026-08-02 (W-DOC1,** [`STAGE_C_REMAINING_WORK_MASTER_PLAN_2026-08-02.md`](STAGE_C_REMAINING_WORK_MASTER_PLAN_2026-08-02.md) **§0/§8).** This section's "reproducing ~14 other papers' methods as baselines" line is now stale: **all 14 T3/T5 external baselines are implemented and complete** — `fedgraa/baselines_t3_contrib.py` (CELM, SpectralFed, CoAst), `fedgraa/baselines_t3_trust.py` (FLTrust, RAGA, FedTruth), `fedgraa/baselines_t5_nids.py` (FedMADE, SBFedNIDS, FedAPA), `fedgraa/baselines_t5_adaptive.py` (FedAWA, DA-PFL, FedAWARE, AdaAggRL), `fedgraa/baselines_t5_fedpgo.py` (FedPGO, faithful since `25611ce`) — 730/730 cells, `check_stage_c_complete.py --tier T1 T2 T3 T4 T5 T6` green, re-verified live 2026-08-02. What remains of Stage C is **not** more baseline code: it is T7 (adversarial robustness, still zero code as of this correction), the classic-FL-optimizer coverage decision (§4 of the referenced plan — resolved 2026-08-02: build FedAvg/FedProx/FedOpt/SCAFFOLD/FedNova in full, `fedgraa/baselines_classic.py`), two analysis layers over already-completed data (significance testing, overhead reporting), and the manuscript itself, which has not been started. See the referenced plan's §0 table for the full, corrected ground truth.
>
> ✅ **Dated correction, 2026-08-03.** The W-DOC1 correction above stopped at
> "T7 ... still zero code as of this correction" — that was true on
> 2026-08-02 but this document's own §2 G5 row and §3 T7 row were never
> updated to match, and an independent audit later flagged both as stale.
> **Both are now resolved:** T7's attack harness (SecFedNIDS protocol +
> rare-class fabrication) was built and run — 90/90 cells complete,
> `check_stage_c_complete.py --tier T7 --seeds 0 1 2 3 4` green — and G5
> (T1–T3 complete, `check_stage_c_complete.py` passes) closed alongside the
> full 830/830-cell T1–T6/TC campaign. See
> [`../current/STAGE_C_MASTER_REPORT.md`](../current/STAGE_C_MASTER_REPORT.md)
> §10 for the full T7 account and
> [`STAGE_C_T7_JUDGMENT_CALLS_2026-08-02.md`](STAGE_C_T7_JUDGMENT_CALLS_2026-08-02.md)
> for the 18 disclosed judgment calls T7's implementation required. This
> document's §2/§3 tables below are left as the historical record of what
> was true on 2026-07-28; treat the two correction boxes above as the
> authoritative current status, not the tables under them.

---

## 2. What was implemented this pass, mapped to the master plan's own workstream IDs

| WS | Task | Where | Tests |
|---|---|---|---|
| W0.2 | `ProjectedFedGRAA` + `project_to_simplex` promoted into `fedgraa/aggregation.py` as a non-default class; `scripts/diagnose_fedgraa.py` now imports rather than redefines them | `fedgraa/aggregation.py` | `test_project_to_simplex_*`, `test_strict_reproduction_default_unchanged` (T-19) |
| W0.3 | Zero-compute oracle-recovery checks landed as permanent tests, at F13's corrected scope | `tests/test_partition_v2.py` | T-8, T-8b, T-8c, T-8d |
| W0.4 | `FedSNDTrainer` renamed `StaticDistributionAwareTrainer`; fabrication disclosed in its docstring; name purged from method keys, CLI help text, and every doc that scheduled it as a to-do (archival/audit-trail docs left untouched by design) | `fedgraa/baselines.py`, `fedgraa/experiment.py`, `run_experiments.py`, several `docs/current/*.md` | `tests/test_reproduction_gaps.py` (updated in place) |
| W1.1/F2 | `median_frequency_split` re-expressed as the composition of `stratum_tags` (server-side, histogram-only) + `apply_tags` (client-side, local-only) | `fedgraa/partition.py` | `test_stratum_tags_apply_tags_composition_matches_median_split`, all pre-existing `test_partition.py` tests unmodified |
| W1.2/W2.3a/F12 | Per-client composite-gradient contribution + the two-phase, count-weighted, byte-optimal SecAgg protocol | `fedgraa/aggregation.py`'s `two_phase_composite_grad` | T-4, T-4a, T-4b, degenerate-stratum corner (6/6) |
| W1.3/F3 | `local_label_histogram(shard, C, scope="full"\|"heldout")` | `fedgraa/partition.py` | T-8c, T-8d |
| W1.4 | `confusion_matrix` + `per_class_metrics_from_confusion`; `per_class_metrics` refactored on top, behaviour-preserving | `fedgraa/metrics.py` | pre-existing `test_metrics.py` unmodified |
| W1.5 | `FedConfig.grad_refresh_every` / `secagg` / `dp_epsilon` / `p_drop` | `fedgraa/config.py` | validated via every v2 test that constructs a `FedConfig` |
| W2.1/F4 | `SecAggChannel` — individual contributions unreachable *by construction* (never retained, not merely ungettable), Shamir threshold, dropout → `SecAggDropoutError` | `fedgraa/secagg.py` | T-6 (introspection-checked), T-7 |
| W2.2 | Laplace mechanism (`Δ₁=2`); advanced-composition **and** RDP accountants, both closed-form (no unvetted third-party DP package added) | `fedgraa/privacy.py` | T-9; advanced-composition matches the plan's own §5.3 worked example (`8.89` at ε=0.1×200 rounds) to 2 decimal places; RDP confirmed tighter |
| W2.3/F5 | `FedGRAAv2` — the `m`-refresh loop, protocol-agnostic via an injected `grad_provider` callback | `fedgraa/aggregation.py` | refresh-count-matches-`ceil(E'/m)` for `m∈{1,2,3,5}`; T-10 (simplex, every round) |
| W2.4 | `FederatedTrainerV2` — orchestrates F3's two histograms → SecAgg → F2's tags → the two-phase pass; bytes and communication *rounds* instrumented separately | `fedgraa/trainer.py` | T-5, T-11, **T-12 (bytes match `(1+⌈E'/m⌉)×baseline` to <1% for every `m∈{1,2,3,5}`)**, T-13 |
| W2.5 | `zeta_probe` — measurement-only, both grad_P computed fresh at the same point; scope-disclosed (does not capture `ζ_stale(m)` continuously — see the theory doc) | `fedgraa/trainer.py` | exercised inside T-5/T-11; isolated per-source in T-18 |
| W3 | Full new test suite: `test_secagg.py`, `test_privacy.py`, `test_aggregation_v2.py`, `test_partition_v2.py`, `test_trainer_v2.py` | `tests/` | all green, §7 |
| W5 | Theory workstream, §6 steps 1–5, mapped onto the actual implemented code | [`FEDGRAA_V2_THEORY.md`](FEDGRAA_V2_THEORY.md) | n/a (a document, self-checked against the test suite it cites) |
| W6 | Real-non-IID multi-silo partitioning primitive (`partition_by_silo`) + a schema-agnostic NF-UQ-NIDS-v2 loader that derives everything from the data it's given, never a hardcoded table. **Update 2026-07-28:** loader now also auto-detects and streams the official release's single-merged-CSV layout (`Dataset`-column silo tag), not just pre-split subdirectories — closes the gap `docs/audit/nf_uq_nids_v2_provenance_audit.md` §6 found. | `fedgraa/partition.py`, `fedgraa/data.py`'s `load_nf_uq_nids_v2_silos` | `test_partition_by_silo_*`, `test_load_nf_uq_nids_v2_silos_*` (incl. 3 new merged-layout tests) |
| — | `run_experiment(method="fedgraa_v2", ...)` wired into the existing experiment-orchestration layer | `fedgraa/experiment.py` | `test_run_experiment_dispatches_fedgraa_v2` |
| — | Modal-ready frontier-campaign driver (T1/T2), one function call per `(dataset, arm, gamma, seed)` cell — the exact shape a Modal `.map()` needs | `scripts/run_v2_frontier_sweep.py` | enumeration/resumability tests green; **one real cell manually run end-to-end this pass** (2 rounds, real cached CIC-IDS-2017 data) to confirm the wiring, not the full campaign |
| — | **Fixed a pre-existing Linux/Modal portability bug** found by the Modal smoke test (`COMPUTE_INFRASTRUCTURE_2026-07-27.md` §4.3): `fedgraa/experiment.py`'s two `ProcessPoolExecutor` sites now force `mp_context=multiprocessing.get_context("spawn")` instead of inheriting Linux's fork default, which deadlocks once TensorFlow is already imported in the parent | `fedgraa/experiment.py` | `tests/test_parallel_sweep.py` (5/5, confirms no Windows regression — Windows only ever used `spawn`) |

**Gate status (master plan §12.1), updated:**

| Gate | Condition | Status |
|---|---|---|
| G0 | `import fedgraa` succeeds; full suite green | ✅ still passing |
| G1 | W0.3 passes | ✅ landed as permanent tests |
| G2 | W1 complete, zero existing-test edits | ✅ confirmed (155/155 unmodified, then 234+/234+ with additions) |
| **G3** | T-4a, T-4b, then T-4 green, in that order | ✅ **all three green** |
| **G4** | T-12 green for all `m` | ✅ **green for `m∈{1,2,3,5}`, <1% error** |
| G5 | T1–T3 complete, `check_stage_c_complete.py` passes | ⏳ **not run — this is the compute campaign, §3** |
| G6 | Duplication test re-run at W4 exit | ⏳ blocked on G5 |

**Two gate columns, not one** — the distinction [`../audit/PRE_COMPUTE_AUDIT_2026-07-28.md`](../audit/PRE_COMPUTE_AUDIT_2026-07-28.md) §6.2 found this document had been collapsing:

| Tier | Mechanism verified (unit/integration tests green) | **Campaign path verified** (driver → config → Modal → persisted result) |
|---|---|---|
| T1 | ✅ since 2026-07-28 | ✅ 2026-07-28, after B1/B2/B3 |
| T2 | ✅ since 2026-07-28 | ✅ 2026-07-28, after B1/B2/B3 |
| T4 | ✅ since 2026-07-28 | ✅ 2026-07-28, after B3/B4/B5 |
| T6 | ✅ loader + `partition_by_silo` unit-tested | ✅ 2026-07-28 — B6 fixed, **real peak measured at 2.65 GB** (was ~88 GB), `run_one_cell`'s NF-UQ branch exercised end-to-end on synthetic data, wired into its own dedicated Modal driver `scripts/modal_frontier_campaign_t6.py` (§5) |
| T3/T5 | ❌ baseline code does not exist (§4) | ❌ |

A green unit test proves the mechanism. It does not prove the experiment can
run — four of the audit's six blocking findings were **silent**, producing
plausible numbers rather than errors, on a suite that was 252/252 green.

G3 and G4 are the two hard gates the master plan names explicitly ("T-4 gates every accuracy number," "T-12 gates every communication number") — **both are now closed.** Every v2 accuracy or byte-count number this plan's frontier figure will eventually report rests on a mechanism that is proven exact by unit test, not merely believed to be.

---

## 3. What remains: the compute campaign (T1–T7)

Nothing below requires new FedGRAA-v2 *design* work — every arm listed already has working code. What each needs is machine-hours.

| Tier | Contents | Runs | Est. CPU (§7.2) | Blocking dependency |
|---|---|---|---|---|
| **T1 — Frontier** | v1-projected + v2 at `m∈{1,2,3,5}` × 2 datasets × 10 seeds × γ=1.0 | 100 | ~28 h | **None** — `scripts/run_v2_frontier_sweep.py --tier T1` runs today |
| **T2 — γ interaction** | T1 arms at γ ∈ {0, 0.5} | 200 | ~57 h | **None** — `--tier T1 T2` |
| **T3 — Approximation baselines** | CELM, SpectralFed, CoAst, FedTruth, RAGA, FLTrust × 2 × 10 | 120 | ~34 h | **Baseline code does not exist** — §4 |
| **T4 — Robustness controls** | `p_drop∈{0.1,0.3}`, DP at `ε∈{1,4,10}` on the best `m` | 100 | ~28 h | **None** — `FedConfig.p_drop`/`dp_epsilon` + `--p-drop`/`--dp-epsilon` flags already wired into the driver |
| **T5 — Novelty-defence grid** | FedPGO ✅, FedMADE, SBFedNIDS, FedAPA, FedAWA, DA-PFL, FedAWARE, AdaAggRL × 2 × 10 | 140 remaining (FedPGO's 20 done) | ~39 h | **7 of 8 baselines lack code** — §4; FedPGO faithful since 2026-07-31 (`25611ce` impl, `96adb05` results) |
| **T6 — NF-UQ-NIDS-v2** | T1's grid, on the one real-silo dataset (**4** clients, not K=9/14 — a structural difference, not "T1 repeated") | 100 | ~28 h | **None** — the B6 closure pass ([`../audit/PRE_COMPUTE_AUDIT_2026-07-28.md`](../audit/PRE_COMPUTE_AUDIT_2026-07-28.md)) wired NF-UQ-NIDS-v2 into `enumerate_cells()`/`run_one_cell()` as a `T6` tier and gave it its own Modal driver, [`scripts/modal_frontier_campaign_t6.py`](../../scripts/modal_frontier_campaign_t6.py); the row this replaces ("not yet wired ... one remaining mechanical step") was accurate when written but went stale the same day once that pass landed |
| **T7 — Adversarial** | SecFedNIDS protocol + rare-class-fabrication attack | ~80 | ~23 h | **Attack harness does not exist** — not attempted this pass (master plan tags T7 "Nice — or a follow-on paper") |

~~**T1, T2, and T4 can run on Modal today, unmodified, using `scripts/run_v2_frontier_sweep.py` and the validated pattern in `COMPUTE_INFRASTRUCTURE_2026-07-27.md`.** That is 130 CPU-hours of the plan's 119-CPU-hour "minimum viable campaign" (T1–T3) already unblocked~~

> ⚠️ **Corrected 2026-07-28** ([`../audit/PRE_COMPUTE_AUDIT_2026-07-28.md`](../audit/PRE_COMPUTE_AUDIT_2026-07-28.md)). The struck sentence was wrong in three ways, kept here rather than deleted because the failure mode — writing "the mechanism works" as "the experiment can run" — is the one this project's verification discipline exists to catch:
> 1. **"unmodified" was false.** No Modal driver existed (**B1**); the `.map()` wrapper both this document and `COMPUTE_INFRASTRUCTURE_2026-07-27.md` §7 called "the next step" had not been written. It exists now: [`scripts/modal_frontier_campaign.py`](../../scripts/modal_frontier_campaign.py).
> 2. **T4 could not have produced usable output** — path collisions (**B3**), deterministic crashes (**B4**), and a mis-placed DP mechanism (**B5**). All fixed, each with a named regression test in [`tests/test_pre_compute_audit_fixes.py`](../../tests/test_pre_compute_audit_fixes.py).
> 3. **The arithmetic was wrong.** T1+T2+T4 = 28+57+28 = **113** CPU-hours, not 130 — and **T4 is not a member of T1–T3**, so it cannot be a subset of that tier's 119-hour budget (**M4**).

**Corrected statement.** T1 (28 h) and T2 (57 h) are launch-ready as of 2026-07-28 — **85 CPU-hours**, which is T1–T3's 119-hour minimum viable campaign minus only T3. T4 (28 h, "Should" not "Must") is launch-ready as of the same date, now that B3/B4/B5 are fixed. **The only piece of T1–T3 still gated is T3**, which needs baseline implementations (§4), not more FedGRAA-v2 code.

Launch:

```powershell
# Windows only, and required: Modal's CLI progress output contains a Unicode
# check mark that cp1252 cannot encode, which aborts the run before any
# container starts. Console-encoding fault, not a campaign fault.
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"

modal run scripts\modal_frontier_campaign.py --verify-only          # pinned env + data + volume pre-flight
modal run scripts\modal_frontier_campaign.py --tier T1 --smoke      # 2 real cells, 2 rounds
modal run --detach scripts\modal_frontier_campaign.py --tier T1     # the campaign

modal volume get fedgraa-v2-results v2_frontier results\            # pull results down
python scripts\check_stage_c_complete.py --tier T1                  # then verify
```

`--verify-only` is not optional politeness — it asserts the container
resolved the **pinned** library versions, that both datasets are actually
in the image, and that the results Volume is writable, before a single cell
runs. Every one of those is a failure that otherwise shows up 20 minutes
into a container.

### 3.1 Running it on Modal

Per [`COMPUTE_INFRASTRUCTURE_2026-07-27.md`](COMPUTE_INFRASTRUCTURE_2026-07-27.md) §4.5's validated script pattern: wrap [`run_one_cell`](../../scripts/run_v2_frontier_sweep.py) as a Modal `@app.function`, `.map()` it over `enumerate_cells(["T1","T2"])`'s output, force `multiprocessing.set_start_method("spawn", force=True)` if any nested pooling is ever added (none is, currently — each cell is single-process), and persist each cell's JSON to a Modal `Volume` (or have `run_one_cell` write straight to a mounted path) so preemption (§6.4 of that document) only costs the one preempted cell, not the whole campaign — `missing_cells()` already makes the driver resumable across a from-scratch restart.

**Cost and wall-clock are no longer estimates — [`COMPUTE_INFRASTRUCTURE_2026-07-27.md`](COMPUTE_INFRASTRUCTURE_2026-07-27.md) §5–6 measured them directly on Modal**, using the existing v1 `FederatedTrainer` code as a stand-in for one campaign unit (three purpose-built calibration scripts under `scripts/modal_calibration/`, real runs, dated 2026-07-28): `cpu=1.0` is confirmed strictly at least as fast as `cpu=2.0`/`cpu=4.0` for this workload (`fedgraa/trainer.py` pins `TF_NUM_INTRAOP_THREADS=1`, so extra cores don't help); the account's concurrency ceiling is empirically 99 containers, not merely read off the pricing page; and per-unit wall-clock/memory were measured directly (UNSW-NB15: 681.5 s, 2,086 MB peak; CIC-IDS-2017: 1,235.9 s, 8,630 MB peak, stable across all three γ). Built up from these measured per-unit numbers: **T1–T3 ≈ 112 core-hours, ≈ $13.47, ≈ 1.1–1.5 h wall-clock at the measured 99-container ceiling; the full T1–T7 stack ≈ 230 core-hours, ≈ $26–29, ≈ 2.3–3 h wall-clock** — both within 6% of the master plan's own independently-derived CPU-hour estimates, despite coming from a completely different measurement method. See that document's §5.3/§6.3 for the full derivation.

### 3.2 ✅ Done, 2026-07-28: G3/G4 confirmed on the actual Modal image

[`COMPUTE_INFRASTRUCTURE_2026-07-27.md`](COMPUTE_INFRASTRUCTURE_2026-07-27.md) §6.5 already flagged that the Modal container's CPU lacks AVX512 (vs. the local dev machine), which measurably shifted one finite-difference gradient test (`atol=5e-3`) by `0.00537` in the earlier v1 smoke test. T-4/T-4a/T-4b run at a much tighter internal tolerance (`<10⁻⁶`, same-container comparisons) so this was expected to mostly cancel, but per this project's own "measure, don't assume" discipline that was **confirmed, not assumed**: [`scripts/modal_calibration/modal_v2_gate_verification.py`](../../scripts/modal_calibration/modal_v2_gate_verification.py) re-ran `tests/test_aggregation_v2.py` + `tests/test_trainer_v2.py` (G3's and G4's own test files) on the real Modal image (same non-AVX512 container confirmed via `/proc/cpuinfo`, [`modal_v2_gate_result.json`](../../scripts/modal_calibration/modal_v2_gate_result.json)) — **34/34 passed, exit code 0.** G3/G4 are now verified on Modal specifically, not just inferred from the local Windows/AVX512 run. Only this one gate-verification run was executed on Modal — no compute-campaign cell beyond it.

---

## 3a. Deliberately deferred: Option A — the few-shot/prototype-guided hybrid (sequenced after B+D, not current-pass work)

**This section is scope-tracking for a later phase, not a to-do for right now.** The decision, per `Final_Stage_C.md` §10's **Option C ("both, sequenced")**, is to finish B+D first — which is what §2–§7 already describe and is this pass's entire active scope — and pick A up **afterward**, as a follow-on. Nothing below is implemented, and nothing below is scheduled to start until B+D's own remaining work (§3–§6) is far enough along. Recording it here now is so the follow-on phase doesn't have to rediscover what A needs from scratch, not a signal that A is in progress.

**What Option A is**, per [`Final_Stage_C.md`](Final_Stage_C.md) §10 and [`STAGE_DIRECTION_ASSESSMENT.md`](../current/STAGE_DIRECTION_ASSESSMENT.md) §5 (which calls the same thing "Option B" in its own, different lettering — see that document's 2026-07-28 note for the cross-reference): a new federated aggregation method combining FedGRAA's gradient-refinement idea with prototype exchange (PROTEAN, ESORICS 2025, is the closest same-domain precedent) or few-shot personalization. It is a second, independent novelty contribution — not a variant, extension, or refactor of the FedGRAA-v2 code in `fedgraa/aggregation.py`, `fedgraa/trainer.py`, etc. It would compete for the same paper's "what is the contribution" slot that B currently occupies, or form a second paper, depending on how the eventual manuscript is scoped — a question this document does not resolve.

**Why this project's own prior documents recommended building B first, and why the sequencing (Option C) still respects that reasoning:** `FEDGRAA_V2_MASTER_PLAN.md` §22.2 (line 755) noted Option A "has no design doc, no byte accounting, no test spec, no codebase mapping," and faces fresher 2025–2026 competition already published at AAAI/IJCAI/TIFS/ESORICS/NeurIPS — FeCoGraph (TIFS 2025), FedFSA (AAAI 2025), FedGMKD (NeurIPS 2024), PROTEAN (ESORICS 2025), DFML, FedP3E all landed since the literature review was compiled ([`Final_Stage_C.md`](Final_Stage_C.md) §4.5–§4.6). That asymmetry — B implementation-ready, A starting from zero — is exactly why Option C sequences B+D first: it is the lower-risk, already-substantially-done work, and A's larger, higher-risk, zero-progress build happens once that's landed, not competing with it for attention now.

**What it will need, in order, once the follow-on phase starts:**

1. **A design/theory document analogous to [`FEDGRAA_V2_MASTER_PLAN.md`](FEDGRAA_V2_MASTER_PLAN.md)** — the mechanism (how gradient-refinement composes with prototype exchange or few-shot personalization), the threat model if any, the communication/byte accounting, and a concrete test specification (the T-4/T-12-style hard gates B has). This does not exist anywhere in this repository today and is a genuine research-design task, not a mechanical one — it is not something this pass fabricates without that design work actually happening, consistent with this project's own standing rule against writing code the audit discipline (`audit_stage_a.py`/`audit_stage_b.py`) would have to later un-invent (cf. §2's W0.4 `FedSNDTrainer` disclosure, the one prior instance in this codebase of exactly that mistake).
2. **Implementation** in `fedgraa/` (a new module, most likely — the mechanism is not a mode of `FedGRAAv2`, it is a different aggregator) plus a full test suite, to the same bar §7 records for B (all-green, zero weakened assertions in the existing suite).
3. **Benchmarking against the baseline grid `STAGE_DIRECTION_ASSESSMENT.md` §5 and `Final_Stage_C.md` §10 both name for this angle**: FedAPA, FedAWA, FedMADE, SBFedNIDS, DA-PFL — five more external-paper reproductions, on top of (not instead of) T3/T5's existing fourteen (§4). None of that baseline code exists either.
4. **A frontier-campaign driver** analogous to [`scripts/run_v2_frontier_sweep.py`](../../scripts/run_v2_frontier_sweep.py), once 1–3 above exist, so Option A's own compute campaign can run on Modal the same way B's does.

**Status as of this update: step 0 of 4, and intentionally not started.** Per Option C's sequencing, this pass's active work is B+D (§2–§7) only. This section exists so the later follow-on phase does not have to rediscover Option A's scope — it isn't dropped, it's sequenced, and it starts from zero when its turn comes.

---

## 4. What remains: external-baseline reproduction (T3, T5 — B+D's baselines; A's own five are a later, sequenced addition)

**This is a genuinely separate, large piece of work, not a compute-only gap.** T3's six approximation baselines (CELM, SpectralFed, CoAst, FedTruth, RAGA, FLTrust) and T5's eight novelty-defence baselines (FedPGO, FedMADE, SBFedNIDS, FedAPA, FedAWA, DA-PFL, FedAWARE, AdaAggRL) are **other papers' methods** — none of their code exists anywhere in this repository, and faithfully reproducing even one of them is comparable in scope to what Stage A did for FedGRAA itself (read the paper closely enough to catch its own internal ambiguities, reimplement, verify against whatever the paper reports, document every judgment call). Implementing all fourteen would be a multi-week effort on its own and is **not** "more FedGRAA-v2 code" — it is fourteen more reproduction projects, and they are this pass's active scope (B+D). **§3a's Option A brings its own baseline grid** (FedAPA, FedAWA, FedMADE, SBFedNIDS, DA-PFL — mostly overlapping names already in T5's list, but Option A can't reuse T5's *runs*, since it needs its own head-to-head against its own new method, not against B) — but per the Option C sequencing decision (§1), that grid is a **later** addition, scheduled once A's design/implementation phase begins, not scheduled alongside T3/T5 now. *Corrected 2026-07-31: thirteen of the fourteen remain — FedPGO was reimplemented faithfully from its primary text (committed `25611ce`) and its 20 cells rerun (`96adb05`); the "none of their code exists anywhere" sentence predates that and is retained for its historical register.*

**Why this is bundled with "the compute part" rather than attempted in this pass:** a baseline reimplementation cannot be verified as correct without running it and checking its numbers against *something* (the source paper's reported figures, at minimum) — exactly the kind of claim this project's own audit discipline (`audit_stage_a.py`/`audit_stage_b.py`) refuses to accept un-derived. Writing baseline code without the compute to verify it produces a second, unverified codebase sitting next to the first — worse than not writing it, per this project's own standing rule of "measure, don't assume." **This is a deliberate scope boundary, not an oversight**, and it is the reason T3/T5 are listed as remaining work here rather than silently skipped.

**Recommended order, unchanged from the master plan's own framing (§10.2/§12.3):** T3 first (it is "Must" — the frontier's right-hand anchor) and only the **closest** baseline per family if time is short (CELM for T3's approximate-estimator comparison, FedPGO for T5's novelty defence) rather than all fourteen. *Corrected 2026-07-31: with FedPGO done, CELM is now the single closest unimplemented baseline; the "FedPGO for T5's novelty defence" clause is superseded by the committed faithful implementation and its rerun results.*

**Superseded, 2026-07-28 (corrected twice later the same day — see §1's top note for the full history).** The master plan's §12.3 framing — trade T5 in full for external lab/cluster compute time — was written for a mentor conversation that, per `STAGE_DIRECTION_ASSESSMENT.md`, had not yet happened. **That conversation has now been resolved by the author directly: the decision is `Final_Stage_C.md` §10's Option C — implement T3 and T5 in full now (B+D's scope), and pursue Option A plus its own baseline grid afterward, as a sequenced follow-on, not in parallel.** This is a scope decision, not a resourcing one yet for B+D — the author's own framing is to implement B+D fully now and weigh the actual cost/benefit trade-off once real T1/T2 frontier numbers are in hand, rather than deciding the trade-off blind now. Concretely: T3/T5 baseline implementation remains open work for this pass (unaffected — see the "not implemented" disclosure above); Option A's baseline grid is open work for the **later** phase, not this one. Whether every baseline across T3/T5 is worth running to completion, or only the closest-per-family subset, is now explicitly a decision to revisit after T1/T2 land, not a decision already made.

---

## 5. NF-UQ-NIDS-v2 real data — ✅ obtained, verified, loadable, and wired (2026-07-28)

> ✅ **Closed 2026-07-28, same day as B6 was found** ([`../audit/PRE_COMPUTE_AUDIT_2026-07-28.md`](../audit/PRE_COMPUTE_AUDIT_2026-07-28.md) **B6**). This section previously read *"obtained, verified, and **loadable**"* with the loadability half **false** — an uncapped load of the real file was measured to extrapolate to ~88 GB peak. It is now genuinely true, **measured against the real file, not extrapolated**.
>
> [`fedgraa/data.py`](../../fedgraa/data.py) gained a bounded-memory path: an exact chunked tally of every `(silo, label)` key's count, a deterministic seeded selection of up to `max_per_class` row positions per key, and a second chunked pass keeping only what was selected. Peak memory is now `O(chunksize) + O(n_keys × max_per_class)` — bounded by the cap, never by the file's true size.
>
> **Measured** ([`scripts/verify_nf_uq_nids_v2_memory.py`](../../scripts/verify_nf_uq_nids_v2_memory.py), Windows peak working set, `max_per_class=10_000`):
>
> ```
> elapsed              :   1041.5 s   (~17.4 min, two full chunked passes)
> PEAK WORKING SET     :   2714.5 MB  (2.65 GB)
> total selected rows  :  255,864 across the 4 real silos
> ```
>
> **2.65 GB measured vs. ~88 GB extrapolated for the uncapped path — a ~33× reduction.** `nf_uq_nids_v2_provenance_audit.md` §6 had found the right underlying number but its §7 conclusion ("unblocked at the code level") did not follow from it; both are now actually true.
>
> **T1/T2/T4 remain unaffected** — no cell in those tiers touches this dataset, and `modal_frontier_campaign.py`'s image excludes it unconditionally. T6 runs from a wholly separate file/App, `scripts/modal_frontier_campaign_t6.py` — not a second image on the same App, which the first version tried and which turned out to be a real bug (a T1-only launch built and uploaded the 12.79 GB file too; see `../audit/PRE_COMPUTE_AUDIT_2026-07-28.md`'s "second new defect" box for the measurement and fix).

### 5.1 The data itself — ✅ obtained and verified (2026-07-28)

[`fedgraa/data.py`](../../fedgraa/data.py)'s `load_nf_uq_nids_v2_silos` (W6) is complete, tested code, and as of this update it has real, verified data to run against too:

1. **✅ Done — the file, obtained and provenance-verified.** `data/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.csv` (75,987,976 rows, the official merged release) was obtained by the project author using their own web access (not by this session's tools — consistent with this project's own "never guess a URL" discipline) and independently audited: every headline statistic matches the dataset's official publication and project pages to the exact record, with clean row-level integrity throughout. Full detail: [`docs/audit/nf_uq_nids_v2_provenance_audit.md`](../audit/nf_uq_nids_v2_provenance_audit.md).
2. **✅ Done — the layout gap the provenance audit found.** The official release ships as this single merged CSV (silo membership in a `Dataset` column), not the four pre-split subdirectories the loader originally expected. `load_nf_uq_nids_v2_silos` now auto-detects and streams this layout directly (`os.path.isfile(data_dir)` dispatch, chunked `Dataset`-column grouping) — no re-split to disk, no loader-side data engineering left. See the provenance audit's §6 for the full before/after.
3. **✅ Done — schema confirmation.** The real file's label column is `"Attack"`, already first in the loader's auto-detect candidate list, so no `label_column=` override is needed in practice. Measured: **21 labels** (20 attack categories + Benign), **41** numeric feature columns.
4. **✅ Done — bounded-memory loading (B6).** §5 above; `max_per_class` caps memory via deterministic per-`(silo,class)` subsampling, measured at 2.65 GB peak against the real file.
5. **✅ Done — wired into the campaign driver.** `scripts/run_v2_frontier_sweep.py`'s `enumerate_cells()` now has a `T6` tier building `NF_UQ_NIDS_V2_DATASET` cells across all five arms; `run_one_cell` has a complete NF-UQ-NIDS-v2 branch (`load_nf_uq_nids_v2_silos` + `partition_by_silo`, with `DatasetSpec.used_counts`/`benign_id` built from the *measured* labels/counts, never a hand-typed table, per `tests/test_data.py::test_load_nf_uq_nids_v2_silos_derives_schema_from_data_not_a_table`'s existing pattern). [`scripts/modal_frontier_campaign_t6.py`](../../scripts/modal_frontier_campaign_t6.py) — its own file and `modal.App`, not a second image bolted onto `modal_frontier_campaign.py` — carries `run_cell_t6` and `verify_env_t6`. [`tests/test_run_one_cell_nf_uq.py`](../../tests/test_run_one_cell_nf_uq.py) exercises the whole path end-to-end on synthetic data.

**T6 is now genuinely launch-ready** — `modal run scripts/modal_frontier_campaign_t6.py --smoke` runs the real code path; a full (undetached) launch is the only step left, and it is a compute-campaign launch, not more engineering. `partition_by_silo` yields **4 clients** against T1's K=14/K=9, so T6 is documented as "T1's own grid, on the one real-silo dataset" rather than a literal repeat — a structural, intentional difference, not an oversight.

---

## 6. What to do when the compute campaign completes

Once T1–T3 (minimum) or T1–T7 (full) finish on Modal:

1. Run [`scripts/check_stage_c_complete.py`](../../scripts/check_stage_c_complete.py) (clone of `check_stage_b_complete.py`'s completeness-gate pattern, per §7.1 of the master plan). **✅ Written, 2026-07-28** — directly against `run_one_cell`'s own result-dict schema (`scripts/run_v2_frontier_sweep.py`), which was already fully known from the implemented, tested driver code without needing a real campaign's output first. Validated end-to-end against one real manually-run cell (`--tier T1 --datasets CIC-IDS-2017 --seeds 0 --rounds 2`, using real cached CIC-IDS-2017 data): correctly reports `0/N` complete against an empty results directory, and `N/N` complete + exit 0 once that one real cell's JSONs exist. Scoped to the per-cell JSON layer only — a consolidated frontier-figure builder (the v2 analogue of `build_ablation_table.py`/`render_ablation.py`) still does not exist and genuinely cannot be built or verified before real T1 data exists to build it from; that piece remains open, listed here rather than silently deferred.
2. Build the frontier figure (master plan §4.2: x = total training traffic in GB from `bytes_sent`, already logged per cell by `run_one_cell`; y = per-minority-class recall with bootstrap 95% CI, using the existing `fedgraa/stats.py` machinery unchanged).
3. Re-run the duplication/scooping search (G6) before drafting.
4. Only then write §10.2's contribution sentence into a manuscript draft — it is explicitly marked **not yet writable** in the master plan (§10.2) because its second half asserts what the frontier's accuracy axis shows, which T1–T3 alone can answer.

---

## 7. Final test count for this pass

```
.venv313\Scripts\python.exe -m pytest tests/ -q
252 passed, 366 warnings in 459.28s (0:07:39)
```

**252, up from the 155-test regression floor this plan's own §13.3 recorded** — 97 new tests, all green, zero failures, zero modifications required to weaken any of the original 155 assertions (`test_reproduction_gaps.py` and `tests/test_data.py` gained additions for the rename/W6 work respectively, but no existing assertion in either file was weakened or removed). This is the number [`FEDGRAA_V2_MASTER_PLAN.md`](FEDGRAA_V2_MASTER_PLAN.md)'s own W1-exit-gate ("the full existing suite passes with zero modifications to any existing test") and G2 ask for, confirmed directly rather than assumed.

**249 → 252 reflects this same-day 2026-07-28 update pass** (loader/layout gap closure, §5/§6 above): 3 new regression tests for the merged-CSV loading path (`test_load_nf_uq_nids_v2_silos_merged_csv_layout_matches_presplit`, `..._missing_dataset_column_raises`, `..._missing_silo_raises`), all in `tests/test_data.py`, none weakening any prior assertion.

**Two of the 94 are audit-found regression guards, not Pass-5 originals.** A same-day independent audit pass (`FEDGRAA_V2_MASTER_PLAN.md` §14.1 Pass 6) adversarially probed the two newest, least-exercised modules and found: (1) `SecAggChannel.aggregate()` could be called twice silently instead of raising, inconsistent with its documented single-use contract (`test_calling_aggregate_twice_raises`); (2) `load_nf_uq_nids_v2_silos` would silently concatenate two silos with same-count-but-different-name numeric columns, corrupting the pooled matrix by aligning unrelated features by position rather than raising (`test_load_nf_uq_nids_v2_silos_rejects_mismatched_column_schema`). Both are fixed; 247 → 249 reflects both fixes' regression tests. Neither was reachable by any current production code path (each SecAgg channel is used once per round in practice; no real multi-silo data has been loaded yet), but both are exactly the class of defect this project's own audit discipline exists to catch before they become reachable.

---

## 8. AI disclosure

This document was compiled by an AI assistant (Claude Sonnet 5) at the repository author's direction, in the same pass that implemented [`FEDGRAA_V2_MASTER_PLAN.md`](FEDGRAA_V2_MASTER_PLAN.md)'s W0–W6 workstreams. Every "done" claim in §2 is backed by a named, currently-passing test; every "remaining" claim in §3–§5 is a genuine dependency on machine-hours, external-paper reproduction, or data this session had no way to fabricate responsibly (no URL for NF-UQ-NIDS-v2 is invented anywhere here, consistent with the standing instruction never to guess a URL). No compute campaign was run and no baseline paper was reimplemented in this pass; both are stated as open, not silently assumed complete or silently skipped.

**Update, 2026-07-28, same day.** At the author's explicit instruction ("do everything in the list except the compute on modal part... you can run the tests on modal but not the actual campaign"), a follow-up pass closed every item from that instruction that did not require the T1–T7 compute campaign itself: the NF-UQ-NIDS-v2 loader/layout gap (§5/§6, `docs/audit/nf_uq_nids_v2_provenance_audit.md` §6), `scripts/check_stage_c_complete.py` (§6), and re-running G3/G4's test files on the real Modal image (§3.2, `scripts/modal_calibration/modal_v2_gate_verification.py`, 34/34 passed). **No compute-campaign cell was run on Modal in this update beyond that one test-verification job** — T1/T2/T4's actual cells (§3) remain explicitly not launched, per the author's instruction to defer that specifically. The author also separately resolved the mentor-conversation decision `STAGE_DIRECTION_ASSESSMENT.md` flagged as pending (see that document's resolution note and §4 above) — recorded here, not decided by this assistant.

**Update, 2026-07-28, later the same day.** The author clarified that the previously-recorded "Option B+D" scope decision was a misreading of what was wanted: the actual instruction was read (at that point) as implementing all four of `Final_Stage_C.md` §10's options (A, B, C, D) in full, in parallel, not the literature review's recommended B+D subset. This pass corrected the top-of-document scope note, the §1 "Also decided" bullet, §4's framing, and added §3a documenting Option A (few-shot/prototype-guided hybrid) as newly-committed work — no design doc, code, or tests for it exist anywhere in this repository as of this update, and none were fabricated to fill that gap, consistent with this document's standing disclosure discipline. The same correction was propagated to every other document that repeated the "Option B+D" framing (`STAGE_DIRECTION_ASSESSMENT.md`, `Final_Stage_C.md` §10, `FEDGRAA_V2_MASTER_PLAN.md`, `docs/audit/STAGE_C_IMPACT_ON_STAGE_AB_2026-07-27.md`, `MENTOR_WALKTHROUGH.md`, `README.md`); dated literature-review and audit snapshots (e.g. `STAGE_C_LITERATURE_MASTER.md`, `STAGE_C_DEEP_RESEARCH_2026-07-25.md`, `RESEARCH_DIRECTION_AUDIT_2026-07-25.md`, `nf_uq_nids_v2_provenance_audit.md`) were deliberately left as-is, since they record what was true/recommended at the time they were written, not the live scope decision.

**Update, 2026-07-28, a third time the same day — correcting the correction above.** The author clarified further: "I meant option C, doing both sequentially, doing B and D for now." The "A, B, C, D in parallel" reading above was itself wrong. The actual decision is `Final_Stage_C.md` §10's **Option C — "both, sequenced"**: finish B+D now (§2–§7, unaffected throughout all of this), and take up Option A **afterward**, as a follow-on phase, not simultaneously and not starting this pass. §1's top note, the §1 "Also decided" bullet, §3a's heading/framing, and §4's framing were all corrected again to reflect this. This is now the third and current statement of scope in this document; earlier "B+D only" and "A+B+C+D in parallel" framings were both transient misreadings, corrected in sequence rather than left standing, per this project's own standing discipline of stating what changed and why rather than quietly overwriting prior claims.
