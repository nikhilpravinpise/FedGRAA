# FedGRAA-v2 — Master Plan

> **What this document is.** The single planning entry point for **FedGRAA-v2**: the privacy-compliant redesign selected as Stage C's angle (Option **B + D**) by [`Final_Stage_C.md`](Final_Stage_C.md) §10. It converts the literature verdict into an ordered, testable, costed build plan — design, theory, experiments, code, tests, positioning, risks, and gates.
>
> **What this document is not.** It is **not** a re-derivation of work already done. Three documents remain authoritative and are cited by pointer, never restated:
>
> | Source | Remains authoritative for |
> |---|---|
> | [`Final_Stage_C.md`](Final_Stage_C.md) | The literature corpus, the novelty claim, the citation ledger, the do-not-write list |
> | [`privacy_compliant_fedgraa_redesign.md`](privacy_compliant_fedgraa_redesign.md) | The 7-family / 6-approach taxonomies, DP arithmetic, threat-model actors, byte counts |
> | [`../current/fedgraa_convergence_defect.md`](../current/fedgraa_convergence_defect.md) | The simplex defect, the corrected proposition, the Stage B factorial evidence |
>
> **What is new here.** §2 lists **thirteen findings** that materially change the plan — including **a measured fault in the exactness claim on which the whole contribution rests** (F12), five specification bugs in the redesign document that would have caused silent failures, and one endemic-flaw finding that upgrades the paper's contribution class.
>
> **This document has been through three passes** (§14.1): a planning pass, a verification pass that executed the codebase rather than reasoning about it, and a post-environment-fix audit. Pass 2 added F12 and F13 and **refuted both stated grounds of F3** — recorded in place rather than quietly rewritten. Pass 3 **measured F12**: the protocol both source documents specify is **160–557% wrong**, and the corrected one is exact to 2 × 10⁻⁷.
>
> **Status: unblocked.** The environment fault that stopped all execution is resolved; the suite is green at **155 tests** (§2 F11, §13). Every numeric claim here is re-derivable by running [`verify_v2_plan_claims.py`](../../scripts/verify_v2_plan_claims.py) and [`verify_v2_exactness.py`](../../scripts/verify_v2_exactness.py), both of which exit non-zero if any claim stops holding.
>
> **Compiled:** 2026-07-27. **Supersedes:** `privacy_compliant_fedgraa_redesign.md` §13 (action checklist) and §15 (code mapping) only. All other sections of that document stand.
>
> ---
>
> ### ⚠️ Read [`../audit/PRE_COMPUTE_AUDIT_2026-07-28.md`](../audit/PRE_COMPUTE_AUDIT_2026-07-28.md) before running the campaign
>
> A whole-repository audit run immediately before launch **re-verified this plan's two load-bearing measurements independently** — F12's exactness (`1.2×10⁻⁷`–`3.8×10⁻⁷`) and §4.1's byte multipliers (measured `6.0006×` at `m=1`, `2.0004×` at `m=5`, on a real run rather than from the formula). **The findings in §2 stand.**
>
> It also found **six blocking defects in the campaign *harness*** — none in the algorithm — of which five are now fixed. Three touch this document's own statements:
>
> | This plan | Audit finding |
> |---|---|
> | **F1** (line 118): v2's reference is *"v1-projected (`projection + η′=0.01`)"* | The campaign driver inherited `FedConfig`'s `η′=0.1`, i.e. Stage B's **`projection-alone`** arm — **0.0%** rare-class accuracy on CIC at γ≤0.5. ✅ Now pinned to `0.01` for **both** arms, so the frontier varies only `m` and the protocol |
> | **§9.2** lists **T-14/15** and **T-16/17** as *"Unchanged"* | Neither had any implementation. ✅ Both now in [`../../tests/test_pipeline_invariants.py`](../../tests/test_pipeline_invariants.py) — and writing T-15 found a live `paired_ttest` bug that returned `p = 3.5×10⁻¹⁶` from identical data |
> | **§7.2 T4** (`p_drop ∈ {0.1, 0.3}`, DP at `ε ∈ {1,4,10}`) | At the production majority SecAgg threshold, `p_drop=0.3` gave a 200-round cell a **~3×10⁻⁹** chance of completing; and DP was applied server-side *after* aggregation at `1/√K` the modelled noise. ✅ Both fixed — an unrecoverable round now aborts and is counted, and DP moved per-client pre-SecAgg |
>
> **Still open: B6.** T6's NF-UQ-NIDS-v2 loader needs **~88 GB peak** on the real 12.79 GB file (measured from a 1M-row sample) and has never been executed against it. T6 also runs at **K=4** silos against T1's K=14/K=9 — a different experiment, not "T1 repeated." T1/T2/T4 are unaffected.
>
> **Standing rule this produced:** *"mechanism verified"* is not *"campaign path verified."* All six defects were present in a tree that was 252/252 green.
>
> ---
>
> **✅ IMPLEMENTATION UPDATE (2026-07-28).** Everything in §8's W0–W3 workstreams, plus W5 (theory) and W6 (dataset-engineering scaffold), is now **implemented, tested, and green** — `ProjectedFedGRAA`/`FedGRAAv2`/`two_phase_composite_grad` in `fedgraa/aggregation.py`; `SecAggChannel` in `fedgraa/secagg.py`; the Laplace/advanced-composition/RDP accountants in `fedgraa/privacy.py`; `stratum_tags`/`apply_tags`/`local_label_histogram`/`bottom_n_rare_classes`/`partition_by_silo` in `fedgraa/partition.py`; `FederatedTrainerV2`/`zeta_probe` in `fedgraa/trainer.py`. **Both hard gates (§12.1 G3, G4) are closed**: T-4/T-4a/T-4b pass at `<10⁻⁶` relative error, and T-12's byte formula matches measured bytes to `<1%` for every `m ∈ {1,2,3,5}`. The theory workstream (§6 steps 1–5) is written up in [`FEDGRAA_V2_THEORY.md`](FEDGRAA_V2_THEORY.md); a Modal-ready campaign driver (`scripts/run_v2_frontier_sweep.py`) covers T1/T2/T4 and has been manually verified against one real cell (not the full campaign). **What remains is exactly the compute campaign (T1–T7), external-baseline reproduction (T3/T5, ~14 other papers' methods), and real NF-UQ-NIDS-v2 data acquisition** — none of it is algorithm or test-design work. The full, itemized remaining-work map, gate-by-gate, is [`REMAINING_COMPUTE_WORK.md`](REMAINING_COMPUTE_WORK.md). §8's tables below are annotated in place with `✅ IMPLEMENTED` rather than rewritten, so the original plan's own language stays intact as the historical record it always was.

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Thirteen findings that change the plan](#2-thirteen-findings-that-change-the-plan)
3. [The corrected FedGRAA-v2 design](#3-the-corrected-fedgraa-v2-design)
4. [The exactness–communication frontier — the actual contribution](#4-the-exactnesscommunication-frontier--the-actual-contribution)
5. [Threat model and privacy accounting, corrected](#5-threat-model-and-privacy-accounting-corrected)
6. [Theory workstream](#6-theory-workstream)
7. [Experimental design — two grids, tiered](#7-experimental-design--two-grids-tiered)
8. [Implementation plan — seven workstreams](#8-implementation-plan--seven-workstreams)
9. [Test plan](#9-test-plan)
10. [Positioning, claims, and the do-not-write list](#10-positioning-claims-and-the-do-not-write-list)
11. [Risk register](#11-risk-register)
12. [Sequencing, gates, and the mentor conversation](#12-sequencing-gates-and-the-mentor-conversation)
13. [Environment — resolved, recorded for the reproducibility manifest](#13-environment--resolved-recorded-for-the-reproducibility-manifest)
14. [Method, verification, and AI disclosure](#14-method-verification-and-ai-disclosure)

---

## 1. Executive summary

### 1.1 The thesis, in one paragraph

FedGRAA-v1 computes its aggregation weights from a **server-held labelled validation set** and a **hard-coded list of rare classes** — two assumptions no real federated deployment can satisfy. FedGRAA-v2 removes both without touching Eq. 7–10, by replacing the central validation loss with a **federated backward pass** whose secure-aggregated result reproduces `∇_P L'` *exactly*, and the hard-coded rare list with a **SecAgg-summed class histogram**. Exactness is not free: it costs a **6× multiplier on total training traffic**. The plan's central move is to stop treating that as a caveat and start treating it as **the result** — by building a *tunable* refresh interval that traces a **communication–exactness frontier from 6× down to 2×**, and benchmarking the whole frontier against the near-zero-cost approximate estimators (CELM, SpectralFed). The paper's question becomes *"how much exactness does rare-attack recall actually need, and what does each increment cost?"* — a question with a publishable answer at every point on the curve.

> ⚠️ **Exactness requires a protocol neither source document specifies.** A plain SecAgg-**sum** of per-client gradients — what `privacy_compliant_fedgraa_redesign.md` §4.1 calls *"mathematically identical"* — is **not** equal to `∇_P L'`, for two independent reasons: the loss is a `reduce_mean`, so contributions must be **count-weighted**, not summed; and `min(L_h, L_m)` is a **global** arm selection no client can make locally. Both are fixable at **< 0.02% extra bytes** — so the frontier above stands — but at the cost of **two communication rounds per refresh instead of one**. See **F12**, the most consequential finding in this document.

### 1.2 The finding that upgrades the contribution class

**FedMADE holds a labelled server-side auxiliary dataset too — and so does the rest of this niche.** Verified verbatim from the FedMADE full text this pass ([arXiv:2408.07152](https://arxiv.org/html/2408.07152v1)):

> *"In each training round, when the server collects local models from the local client gateways, it utilizes an auxiliary dataset to calculate the CPM for each local model."* … *"For our experiments, we construct a small auxiliary dataset of ten samples from each class."*

FedMADE is Stage C §4.4's **must-beat** baseline — the 71.07%-minority-improvement paper. Its aggregation weights are derived from a **labelled, per-class-stratified server dataset**, which is a *stronger* violation than FedGRAA-v1's. Combined with the precedents Stage C §7.1 already catalogues — FLTrust's root dataset, FedGreed still assuming it in 2025, FedTruth and CoAst both naming it impractical, CELM's abstract restating it independently — this changes what kind of paper v2 is:

| Framing | Reads as | Reviewer response |
|---|---|---|
| ❌ *"We fixed two flaws in our own method"* | Errata | "So the v1 results were invalid?" |
| ✅ *"Validation-driven adaptive aggregation for FL-IDS rests on a server-side labelled set that the deployment model forbids — including in the strongest published minority-class baseline. We show the assumption is removable at an exactly measured cost, and chart the cost curve."* | A result about the subfield | "What is the cheapest point that still works?" |

**Take the second framing.** It costs nothing extra to write and it converts a defensive section into the paper's contribution.

### 1.3 Recommendation

| # | Decision | Recommendation | Confidence |
|---|---|---|---|
| 1 | Stage C angle | **Option B + D**, as [`Final_Stage_C.md`](Final_Stage_C.md) §10 recommends — unchanged and re-affirmed. **✅ Update, 2026-07-28:** the author has since chosen Option C — "both, sequenced": build B + D now (this plan, unchanged), pursue Option A afterward as a follow-on — see `../current/STAGE_DIRECTION_ASSESSMENT.md`'s resolution note and `REMAINING_COMPUTE_WORK.md` §3a. This plan's own scope (§§2–9 below) is unaffected — it remains the B-only build, which is this pass's active work; A is separate, deliberately-deferred work with no plan of its own yet | High |
| 2 | What v2 is built on | **The projected (fixed) algorithm, not the literal one.** Non-negotiable — see §2 F1 | Very high |
| 3 | The headline contribution | **The exactness–communication frontier** (§4), not "we removed two flaws" | High |
| 3b | How exactness is achieved | **The two-phase, count-weighted protocol of §3** — *not* the plain SecAgg-sum both source documents specify, which is provably not exact (§2 **F12**). Write **T-4a/T-4b before** the v2 trainer | Very high |
| 4 | Rarity-signal source | **Full local shard histogram**, not the held-out split — for **determinism** (exactly `used_counts`, zero seed variance vs. a 0–6 range that hits zero on 2/10 seeds), not for the correctness or DP reasons this plan's first draft gave, both of which measurement refuted (§2 F3) | High |
| 5 | SecAgg on the gradient channel | **Mandatory, not optional** — the redesign doc's "optionally" is a security hole (§2 F4) | Very high |
| 6 | Dataset move | **NF-UQ-NIDS-v2 migration** in parallel from day 1 — it is what makes the privacy claim non-vacuous (§7.4) | High |
| 7 | Baseline strategy | **Two separate grids** (novelty-defence vs. contribution-evidence), tiered by compute (§7.2) | High |
| 8 | First action | ~~W0.2 — promote `ProjectedFedGRAA` into `fedgraa/`.~~ ✅ **W0–W3, W5, W6 all implemented and tested (2026-07-28)** — see `REMAINING_COMPUTE_WORK.md`. **The next action is the compute campaign (T1–T2 on Modal)**, not more code | Certain |

### 1.4 What gates the work

Not literature — Stage C closed that. Not the environment either — ✅ **resolved 2026-07-27, suite green at 155 tests** (§2 F11). Two things gate FedGRAA-v2:

1. **The mentor conversation** — four questions, unchanged from [`../current/STAGE_B_MASTER_REPORT.md`](../current/STAGE_B_MASTER_REPORT.md) §22.2, plus one new input (§12.3).
2. **Compute reality** — §7.2 shows the §9.1 literature checklist's 24-baseline grid is roughly **6–9 CPU-weeks**. It must be tiered, not attempted whole; T1–T3 is **119 CPU-hours** and is the paper.

**Neither blocks starting.** W0.2 → W1 → W2 is **6–9 working days** (§8) that need no decision and no new compute, and W5 (theory) and W6 (dataset) are parallel from day one.

---

## 2. Thirteen findings that change the plan

Findings are tagged **F1–F13**. Each states what was believed, what is true, how it was checked, and what changes.

> **Read F12 and F13 first.** Both came from the passes that ran **after** this document's first draft (§14.1) — pass 2 executed the partition code and re-derived every number from the live config; pass 3, once the environment was fixed, measured F12 against the real model. They are the two most consequential findings here: **F12 shows the exactness claim, which is the paper's entire contribution, is false as specified in both this plan's first draft and the source redesign document — 160–557% wrong, measured.** F3 was substantially **wrong** in its first form and is rewritten in place rather than deleted.
>
> Numbering is chronological, not by importance, so that the `[Fn]` tags used throughout §§3–13 stay stable. **By consequence the order is F12 → F1 → F13 → F5 → the rest.**

---

### F1 — v2 cannot be built on the literal algorithm, and no existing document says so

**Believed.** [`privacy_compliant_fedgraa_redesign.md`](privacy_compliant_fedgraa_redesign.md) §15.1 states that `_project_simplex(w)` is *"already used inside `FedGRAA.update()`"* in `aggregation.py`, and §11.2's equivalence test asserts agreement on *"`w^t` after the Adam step and simplex projection."*

**True.** There is **no `_project_simplex` in `fedgraa/aggregation.py`.** [aggregation.py:206-230](../../fedgraa/aggregation.py#L206-L230) says the opposite, explicitly:

> *"No simplex projection is applied afterwards, because Algorithm 2 does not contain such a step."*

The projection exists **only** as a diagnostic-only function, [`scripts/diagnose_fedgraa.py:61`](../../scripts/diagnose_fedgraa.py#L61) `project_to_simplex`, deliberately never imported by the reproduction package — a standing "no silent patching" decision recorded in [`../current/fedgraa_convergence_defect.md`](../current/fedgraa_convergence_defect.md) §7.

**Why this gates everything downstream** *(F12 is the most consequential finding; this is the one that must be fixed **first**)*. Stage B established that the unprojected algorithm **violates feasibility at round 1 on 100% of cells** — all 60 literal-arm cells, 10 seeds × 3 γ × 2 datasets, zero exceptions — with composite-loss blowup up to **1.86 × 10⁶×** on CIC-IDS-2017. FedGRAA-v2's entire empirical programme is *measuring the estimation error `ζ` between the true and federated composite gradients*. **Measuring a small perturbation against a reference that diverges by six orders of magnitude is meaningless.** Every v2 number — the ζ distribution, the v1-vs-v2 accuracy delta, the frontier curve — would be noise on top of a divergence.

**Changes.**
- **Workstream 0 is now a prerequisite, not a nice-to-have** (§8.1): promote `ProjectedFedGRAA` from `scripts/` into `fedgraa/aggregation.py` as a first-class, **non-default** class. The strict-reproduction default stays literal and untouched; v2 subclasses the projected variant.
- Everywhere this plan says "v1" as v2's comparison reference, it means **v1-projected** (`projection + η′=0.01`, the Stage B *combined-fix* arm), not v1-literal. The paper must report both and say why.
- `privacy_compliant_fedgraa_redesign.md` §15.1's projection row and §11.3 test 10 are **withdrawn** and replaced by §9.2 T-10 below.

---

### F2 — The equivalence test cannot pass as specified: the pseudocode broadcasts the wrong object

**Believed.** §4.2's pseudocode broadcasts `rare_classes_t ← BottomN(p̂, n)` to clients, which then compute their own `∇_P L_k'`.

**True.** A client cannot stratify its own rows from the rare set alone. [partition.py:203-244](../../fedgraa/partition.py#L203-L244) `median_frequency_split` assigns a row to `D_V^h` vs `D_V^m` by whether **its class count exceeds the median of the counts of all non-rare classes present in the pooled validation set** — a *global* quantity. A client that knows only `rare_classes_t` and its own rows cannot reproduce that partition. Test #4 (`test_fedgraa_v2_matches_v1_at_zeta_zero`) would fail, and would fail for a *specification* reason that looks like an implementation bug.

**Changes.** The server must broadcast a **length-`C` stratum-tag vector** `τ ∈ {h, m, n}^C`, derived server-side from `p̂` by applying the *unmodified* `median_frequency_split` arithmetic to the histogram instead of to raw labels. Clients tag their own rows by lookup. Cost: `C` bytes/round (15 or 10) — negligible against the 163 KiB / 131 KiB model messages of §4.1. This also *preserves* the gap-2.6 fallback in [partition.py:232-239](../../fedgraa/partition.py#L232-L239) that keeps `D_V^h` non-empty, which a naive client-side reimplementation would have silently dropped.

**Refactor implied.** `median_frequency_split(X_val, y_val, rare_ids)` splits into:
- `stratum_tags(counts: np.ndarray, rare_ids) -> np.ndarray` — pure, server-side, histogram-only, carries all the existing median/fallback logic;
- `apply_tags(X, y, τ) -> ValidationSplit` — pure, client-side, no global knowledge.

`median_frequency_split` is then re-expressed as the composition of the two, so **all existing tests keep passing unchanged** and the equivalence is structural rather than asserted.

---

### F3 — ⚠️ **REWRITTEN AFTER MEASUREMENT.** The full-shard histogram is still right, but both of the first draft's reasons were wrong

> **This finding's first draft claimed (a) the held-out histogram breaks ζ=0 equivalence "by construction" and (b) DP noise makes it unusable. The verification pass executed the partition code across 10 seeds × 2 datasets and 2,000 DP trials per (dataset, ε). Both claims are false.** They are recorded here rather than deleted, because the failure mode — arguing from a plausible SNR calculation instead of running the partition — is exactly the one this project's verification discipline exists to catch.

**Believed (redesign doc §4.2).** `h_k ← client k's local label-count vector for its held-out split`.

**Measured** ([`scripts/verify_v2_plan_claims.py`](../../scripts/verify_v2_plan_claims.py), on the live `config.py` + `partition.py`, strict-reproduction defaults `val_fraction = 0.10`, `n_rare = 3`, seeds 0–9):

| Claim in first draft | Measured result | Verdict |
|---|---|---|
| Held-out histogram breaks `BottomN` → T-4 fails by construction | **10/10 seeds recover the oracle rare set on both datasets** | ❌ **False** |
| DP noise makes the held-out histogram unusable at ε=10 | **100.00%** recovery at ε=4 and ε=10; **99.95% / 99.85%** at ε=1 (2,000 trials each) | ❌ **False** |
| Full-shard histogram equals `used_counts` | **Exact on every class, every seed, both datasets** — including BENIGN (ratio 1.000, so the circular-indexing benign pool does *not* double-count at these configs) | ✅ True |

**Why the SNR argument failed.** It computed *relative error on a count* (`σ/count`), when the quantity that governs `BottomN` is the **gap to the next-rarest class, relative to σ**. On CIC-IDS-2017 that gap is Infiltration (36) → XSS (652), i.e. ≈62 held-out counts against `σ = 10.58` at ε=1 — a ~6σ separation. The ranking is robust *because the rare classes are separated by orders of magnitude*, which is the same property that motivates FedGRAA in the first place. **The redesign doc's §7.5 relative-error table makes the same conflation** — and half-notices it (*"BottomN … only needs relative ordering, not exact counts"*) before presenting the relative-error framing as the finding anyway.

**What is still true, and is the actual reason to prefer the full shard.** The held-out histogram is **unstable**, and the instability is not benign:

| Oracle-rare class | Full-shard count (all 10 seeds) | Held-out count across seeds 0–9 |
|---|---|---|
| Heartbleed | **11** — zero variance | 1, 2, 2, 2, 1, 1, 2, 1, 1, **0** |
| SQL Injection | **21** — zero variance | 1, 1, 3, 3, 3, 3, 3, 2, 3, **0** |
| Infiltration | **36** — zero variance | 3, 3, 2, 3, 4, **0**, 6, 3, 5, 3 |

**On 2 of 10 seeds at least one protected rare class has *zero* held-out rows** (seed 5: Infiltration; seed 9: Heartbleed **and** SQL Injection). `BottomN` survives it, but the pooled `D_V^n` does not: on those seeds the γ-term **cannot see that class at all**, because `_train_val_split` draws an unstratified ~10% of a 22-row shard. That is a previously-undocumented fragility **in FedGRAA-v1 itself**, independent of v2, and it belongs in Limitations.

**Changes (unchanged in substance, corrected in justification).** Two histograms, two roles:

| Histogram | Source | Feeds | Why |
|---|---|---|---|
| `h_k^rank` | Client `k`'s **full local shard** (train + val) | `BottomN(p̂, n)` → `rare_classes_t` | **Deterministic** — exactly `used_counts`, zero seed variance; never zero for a present class |
| `h_k^strat` | Client `k`'s **held-out split only** | `stratum_tags(·)` → `τ` (F2) | Must match `median_frequency_split`'s pooled-`y_val` counts exactly |

**And one piece of good news the redesign doc understates.** DP on the rarity histogram is **empirically viable at ε = 1** on both datasets and both sources (≥99.85% recovery, 2,000 trials). The redesign doc's pragmatic "use SecAgg-only, no DP, for the ranking signal" is more conservative than the measurement requires. Report the measured recovery rates; they are a stronger privacy-utility result than the doc's analytical caveat.

---

### F4 — SecAgg on the gradient channel is mandatory; "optionally" reopens the flaw the redesign closes

**Believed.** §4.1/§4.2 describe the gradient sum as *"optionally through SecAgg"* and *"optionally SecAgg-wrapped."*

**True.** The composite loss is `L_k' = (1−γ)·L_{a,k} + γ·L_{u,k}`, where `L_{u,k}` is evaluated on client `k`'s **rare-class rows**. A client holding no rare-class rows contributes a **structurally zero** `γ`-component. A server that sees any individual `g_k` can therefore test, in one dot product, whether client `k` holds rare-class data — **recovering exactly the oracle knowledge that flaw #1 exists to remove.** Under the one-attack-per-client partition this identifies the protected clients outright. The `(1−γ)` component leaks the client's stratum membership by the same argument.

**And SecAgg costs nothing here — a structural gift worth stating in the paper.** [aggregation.py:110-120](../../fedgraa/aggregation.py#L110-L120) `_grad_wrt_weights` computes `dL'/dw_k = ⟨∇_P L', P_k⟩` from the **total** gradient and the client parameters the server already holds. The algorithm **never needs an individual `g_k`** — only the sum. SecAgg is therefore a **zero-information-loss drop-in**, not a lossy privacy tax. Most SecAgg deployments cannot say that.

**Changes.** Delete "optionally" everywhere. State the leak, state that the algorithm's sum-only structure makes closing it free, and make it a **testable invariant**: the SecAgg mock's interface must make an individual `g_k` *unreachable by construction*, not merely unused (§9.2 T-6).

---

### F5 — The 6× cost is a knob, not a constant — and the knob is the paper

**Believed.** ≈6× is *the* cost of privacy compliance, reported as a fixed caveat.

**True.** The 6× arises because [aggregation.py:224-228](../../fedgraa/aggregation.py#L224-L228) recomputes `aggregate(...)` and `_composite_grad_P(...)` **inside** the `for _ in range(self.weight_epochs)` loop — so each of `E'=5` inner epochs needs a fresh `θ'_g` broadcast and a fresh gradient return. Refreshing the federated gradient every `m` inner epochs instead of every epoch gives:

`total traffic = (1 + ⌈E'/m⌉) × baseline`

| Variant | `m` | Multiplier | CIC-IDS-2017 | UNSW-NB15 ⚠️ | Comm. rounds / refresh | Gradient exactness |
|---|---|---|---|---|---|---|
| v2-exact | 1 | **6×** | 5.606 GB | **2.908 GB** | 2 `[F12]` | Exact (SecAgg quantization only) |
| v2-lazy-2 | 2 | **4×** | 3.738 GB | **1.939 GB** | 2 | Stale ≤ 1 inner epoch |
| v2-lazy-3 | 3 | **3×** | 2.803 GB | **1.454 GB** | 2 | Stale ≤ 2 inner epochs |
| v2-lazy-5 | 5 | **2×** | **1.869 GB** | **0.969 GB** | 2 | Stale ≤ 4 inner epochs (one refresh/round) |
| v1 (privacy-violating) | — | 1× | 0.934 GB | **0.485 GB** | 1 | Exact, via a server-held `D_V` |
| CELM / SpectralFed | — | ≈1× | ≈0.93 GB | ≈0.49 GB | 1 | Approximate, frozen after warm-up |

> ⚠️ **The UNSW-NB15 column corrects an error inherited from `privacy_compliant_fedgraa_redesign.md` §7.1.** That document computes UNSW's message size from **`n_features = 43`**; the live [config.py:224](../../fedgraa/config.py#L224) uses **`n_features = 39`** (42 raw columns minus the 3 categorical fields the numeric-only pipeline drops). Correct parameter count is **33,660**, not 34,460; message **134,640 B**, not 137,840 B; baseline **0.485 GB**, not 0.496 GB — a **2.4% overstatement** throughout that document's UNSW row. Re-derived here directly from `spec.model_config().layer_sizes` ([`scripts/verify_v2_plan_claims.py`](../../scripts/verify_v2_plan_claims.py)); the CIC-IDS-2017 figures were checked the same way and **match the redesign doc exactly** (41,715 params, 166,860 B, 0.934 GB).

**This is well-precedented, not a hack.** Staleness folds cleanly into Assumption 5's `ζ` — a lazy/inexact-gradient method, whose canonical convergence treatment is **Schmidt, Le Roux & Bach, *"Convergence Rates of Inexact Proximal-Gradient Methods for Convex Optimization,"* NeurIPS 2011, arXiv:1109.2415** ✅ *(verified this pass: title, authors, 12 Sep 2011 submission, DBLP `conf/nips/SchmidtRB11`; the paper's own result is that inexact methods retain the error-free rate provided errors decay appropriately)*. **This reference is absent from all ten Stage C passes** and is the single most load-bearing citation the theory workstream was missing.

**Changes.** `m` becomes a first-class config parameter (`FedConfig.grad_refresh_every`), the frontier becomes the headline experiment (§4), and "is exact worth 6×?" becomes a curve with five points instead of a rhetorical question with none.

---

### F6 — The codebase ships a baseline named after a paper that does not exist

**True.** [`fedgraa/baselines.py:278`](../../fedgraa/baselines.py#L278) defines `FedSNDTrainer`, docstringed *"FedSND baseline: simple distribution-aware static aggregation weights."* [`Final_Stage_C.md`](Final_Stage_C.md) §3.2 records FedSND as a **confirmed hallucination-equivalent** — zero arXiv, web and DBLP hits — and it is *FedGRAA's own reference [12]*, flagged for removal in checklist item §9.12.

Nothing connects these two facts. A comparison table containing a "FedSND" column is a table comparing against a **nonexistent method**, and a reviewer who checks reference [12] finds it immediately.

**Changes.** Before any v2 result is reported: rename to what the code actually implements (`StaticDistributionAwareTrainer`), add a docstring recording that the FedSND citation is unverifiable and the class is retained only as a generic distribution-aware static-weight control, and remove the name from every artifact key, plot legend and results JSON. Tracked as **W0.4** (§8.1). Cheap, and a real credibility hazard if missed.

---

### F7 — `rare_client_ids`'s non-oracle path already exists but has never executed

**True, and worth more than the redesign doc claims.** [config.py:139-155](../../fedgraa/config.py#L139-L155): when `rare_label_order` is empty, `rare_client_ids` falls back to size-ranking `used_counts`. Both `CIC_IDS_2017` and `UNSW_NB15` are constructed **with** a non-empty `rare_label_order` (the paper's hard-coded oracle), so the fallback has never run.

**What the redesign doc missed.** This is not merely "a smaller change." It is a **free correctness oracle**: the fallback path computes exactly what `BottomN(p̂, n)` must compute when `p̂` is exact. So test T-8 (§9.2) — *does online inference recover the hard-coded set?* — can be run **today, at zero compute cost, before any SecAgg or federated-gradient code exists**, by simply constructing the two `DatasetSpec`s with `rare_label_order=[]` and comparing. That converts the plan's riskiest correctness assumption into a **day-one, sub-second check**.

**Changes.** T-8 is promoted to **W0.3**. ✅ **Already executed** (F3, F13 tables) — what remains is landing it as a permanent test at the corrected scope (`n=3` both datasets; `n ∈ {5,7}` CIC only, per F13).

---

### F8 — The right baselines for the *contribution* are not the right baselines for the *novelty claim*

**True.** [`Final_Stage_C.md`](Final_Stage_C.md) §9.1 item 2 lists ~24 baselines led by FedPGO. That priority ordering is correct for defending **novelty** — FedPGO is the closest paper, and omitting it is the likeliest fatal miss. It is the **wrong** ordering for evidencing **v2's contribution**, which is about *removing the server-side validation set*. FedPGO is a personalization paper; beating or losing to it says nothing about whether exact federated validation beats approximate data-free estimation.

**Changes.** Two grids, run and reported separately (§7.2):

| Grid | Question it answers | Leads with |
|---|---|---|
| **Contribution-evidence** | Is exact federated validation worth its cost vs. approximating or avoiding it? | v1-projected, v2 frontier (5 points), **CELM**, **SpectralFed**, CoAst, FedTruth, RAGA, FLTrust |
| **Novelty-defence** | Is FedGRAA a distinct point in an occupied space? | **FedPGO**, FedMADE, SBFedNIDS, FedAPA, FedAWA, DA-PFL, FedAWARE, AdaAggRL |

The contribution grid is the paper's Results section. The novelty grid is its Related Work plus one comparison table. Conflating them is what makes the §9.1 list read as 24 unreachable baselines instead of two tractable ones.

---

### F9 — The theory workstream's target theorem is now fully citable, with no gaps

**True.** Assembling what four documents hold separately, plus F5's new reference, the target theorem has **every ingredient sourced**:

| Ingredient | Source | Status |
|---|---|---|
| Constraint set + feasibility | Duchi et al. 2008 simplex projection; Stage B §4 corrected proposition | Already written |
| Non-convex stationarity measure | Projected-gradient-mapping, `min_{t≤T}‖G_η(w^t)‖² = O(1/T)` | Already written (Stage B §4) |
| Inexact gradient tolerance | **Schmidt, Le Roux & Bach, NeurIPS 2011, arXiv:1109.2415** ✅ | **New this pass (F5)** |
| Bounded estimation error assumption | Assumption 5, `E‖∇̂ − ∇‖² ≤ ζ²` | Already written (redesign §5) |
| Weights *derived from* the bound | FedAAW, KBS 2024, DOI 10.1016/j.knosys.2024.112484 | Catalogued, unread |
| Alternating-minimization proof architecture | Parsa et al., arXiv:2511.03529 | Catalogued |
| Non-vacuous heterogeneity measure | FedAWARE consensus dynamics, arXiv:2310.02702 | Catalogued |
| PL condition (replaces convexity) | Karimi, Nutini & Schmidt, ECML-PKDD 2016 | Catalogued |

**Target statement.** *Projected inexact (sub)gradient descent on the probability simplex, under PL, with bounded gradient error `ζ`:* `min_{t≤T} ‖G_{η'}(w^t)‖² = O(1/T) + O(ζ²)`, with **feasibility structural at every `t`**.

Three properties make this the right theorem: it reduces to Stage B's already-proved proposition at `ζ = 0`; the `O(ζ²)` term is exactly what the **frontier experiment measures**, so theory and experiment test the same quantity; and it covers every point on the frontier — v2-exact and every lazy variant — with one statement rather than one per variant.

---

### F10 — Two smaller items, recorded so they are not re-derived

**F10a — `ClientShard` already carries per-client `X_val`/`y_val`.** [partition.py:25-39](../../fedgraa/partition.py#L25-L39). The pooled `FederatedSplit.X_val` at [partition.py:180-181](../../fedgraa/partition.py#L180-L181) is *literally the concatenation* of the per-client splits — verified **bit-identical** by execution. The federated backward pass therefore needs **no new data plumbing**: it iterates `split.clients` instead of reading `split.val_split`.

⚠️ **Corrected by F12 — do not read this as "the equivalence is a summation-order identity."** This plan's first draft did, and it is wrong: the *data* is identical, but the *reduction* is not, because `reduce_mean` and the global `min` arm both act on the pooled set. What F10a establishes is narrower and still useful — **the risk lives entirely in the reduction, not in the plumbing**, which is precisely where T-4a/T-4b point. It remains good evidence for "methodology patch, not rewrite," but not for exactness.

**F10b — the client-Adam deviation compounds with `ζ`.** [`../current/fedgraa_convergence_defect.md`](../current/fedgraa_convergence_defect.md) §7 records that `client_train_round` resets client *model* parameters every round but never rebuilds the client Adam optimizer, so momentum persists across 200 rounds of shifting start points. Orthogonal to the simplex fix — but **not** orthogonal to `ζ` measurement: it is a second source of round-to-round variance that will show up in the ζ distribution's tail and must be **held constant across all frontier arms** so the curve measures refresh interval and nothing else. Disclose it in Limitations; do not fix it mid-campaign.

---

### F11 — ✅ **RESOLVED.** The environment blocker is fixed; the suite is green at 155 tests

**Was.** A Windows Application Control policy blocked `_multiprocessing.pyd` on the `.venv313` interpreter. TensorFlow imports `multiprocessing.pool` transitively via `tensorflow.python.distribute.combinations`, so `import fedgraa` failed, so **16 of 20 test modules could not collect** (14 tests collected, 16 collection errors). Verified twice, sandboxed and unsandboxed.

**Now.** Fixed by the author, 2026-07-27. Re-verified:

```
$ .venv313/Scripts/python.exe -c "import multiprocessing.pool; import fedgraa"
mp OK / fedgraa OK

$ .venv313/Scripts/python.exe -m pytest tests/ -q
155 passed, 366 warnings in 320.21s (0:05:20)
```

**Consequences — all positive, and one of them is large.**

1. **G0 passes; W0.0 and W0.1 are complete.** The critical path starts at W0.2.
2. **The baseline is 155 tests, not the 142 that `STAGE_B_MASTER_REPORT.md` §22.1 records** — the 2026-07-26 audit commit added 13. Use 155 as the regression floor.
3. **F1 is now confirmed by the project's own suite, not only by inheritance.** `tests/test_aggregation.py::test_fedgraa_default_hyperparameters_drift_off_simplex` and `…_reduced_eta_prime_stays_near_simplex` both pass — the unprojected weight learner *is* measured drifting off the simplex at the paper's own `η'=0.1`. W0.2 rests on a live, passing test.
4. **F12 could be measured rather than argued** — see F12, which is the finding this unblocked.

**Changes.** §13 is retained as a diagnosis record, not a task. Risk R1 is closed.

---

### F12 — ⛔ **The exactness claim is false as specified — now confirmed by measurement.** A plain SecAgg-sum of per-client gradients does *not* reproduce `∇_P L'`

**This is the most consequential finding in this document.** Exactness *is* the contribution (§4, §10.2). Both the redesign doc and this plan's first draft specify a protocol that does not deliver it.

> ✅ **Measured, once the environment was fixed** ([`scripts/verify_v2_exactness.py`](../../scripts/verify_v2_exactness.py)). Four variants of `∇_P L'` computed from identical inputs, compared against the v1 pooled reference, on both datasets at γ ∈ {0.5, 1.0}:
>
> | Variant | CIC γ=1.0 | CIC γ=0.5 | UNSW γ=1.0 | UNSW γ=0.5 |
> |---|---|---|---|---|
> | **V1 — naive sum** *(what both design docs specify)* | **1.86** | **5.57** | **1.60** | **3.77** |
> | **V2 — global tags, still mean-normalized** *(isolates obstruction A)* | 1.86 | 6.15 | 1.60 | 3.75 |
> | **V3 — the §3 two-phase count-weighted protocol** | **1.2 × 10⁻⁷** | **2.2 × 10⁻⁷** | **3.0 × 10⁻⁷** | **3.0 × 10⁻⁷** |
>
> *(relative L2 error vs. the pooled `∇_P L'`; V3's residual is float32 summation-order noise over ~4 × 10⁴ parameters)*
>
> **The naive protocol is not slightly off — it is 160% to 557% wrong.** It returns a substantially different vector, not a perturbed one. **V2 is no better than V1** (and at CIC γ=0.5 slightly worse, 6.15 vs 5.57), so **obstruction (A) alone is fatal** and fixing only the tag broadcast (F2) would not have rescued it. **V3 reproduces the pooled gradient exactly**, confirming the corrected design.
>
> **Two details in this table are themselves findings.** *(i)* V1 and V2 are **identical at γ=1.0** on both datasets, because `(1−γ) = 0` deletes the `L_a` term entirely — the stratum tags cannot matter when the arm they select is multiplied by zero. *(ii)* They **diverge at γ=0.5** (5.57 vs 6.15; 3.77 vs 3.75). Together these **confirm F2 empirically and bound it**: the tag-broadcast defect is real, and it is active **only for γ < 1**. Since the frontier campaign runs γ ∈ {0, 0.5, 1.0} (§7.2 T1–T2), it is in scope and must be fixed — but a γ=1.0-only test would not have detected it.

**Believed.** [`privacy_compliant_fedgraa_redesign.md`](privacy_compliant_fedgraa_redesign.md) §4.1: *"`∇_P L' = Σ_k ∇_P L_k'(θ'_g)` computed via … summation of client-returned gradients — **mathematically identical value**, just computed as a distributed sum."* This plan's first-draft §3 pseudocode inherited it verbatim.

**True — two independent obstructions, both verified in the live source.**

**(A) The loss is a mean, so the sum of per-client gradients is the wrong quantity.** [model.py:166-184](../../fedgraa/model.py#L166-L184) uses `tf.reduce_mean`. The pooled gradient is `∇L_h = (1/n_h)·Σ_i ∇ℓ_i`. A plain sum of per-client *mean* gradients gives `Σ_k (1/n_{h,k})·Σ_{i∈k} ∇ℓ_i` — every client weighted by `1/n_{h,k}` instead of by `n_{h,k}/n_h`. **Under the one-attack-per-client partition this is not a rounding error:** client shard sizes span three orders of magnitude (Heartbleed's client holds 22 rows; a capped-majority client holds 20,000), so the naive sum inverts the intended weighting almost exactly.

**(B) `min(L_h, L_m)` is a *global* decision that no client can make locally.** [aggregation.py:184](../../fedgraa/aggregation.py#L184) selects the arm from the pooled scalars via `_composite_min_arm(L_h, L_m)`. A client evaluating `min(L_{h,k}, L_{m,k})` on its own rows can select a *different* arm than the global one; summing per-client arm-selected gradients then mixes `g_h` and `g_m` contributions that the pooled computation would never mix. Under the one-attack-per-client partition, where each client holds one attack class plus benign, **local and global arm selection routinely disagree.**

**The corrected protocol.** Both obstructions are fixable at **negligible byte cost**, which is why the §4 frontier survives — but neither is fixable by summation alone:

| Obstruction | Fix | Extra bytes | Extra leakage |
|---|---|---|---|
| **(A)** mean vs. sum | Clients return **unnormalized (sum-form)** gradients `n_{s,k}·∇L_{s,k}` and their counts `n_{s,k}`; server divides by `n_s = Σ_k n_{s,k}`, which SecAgg already supplies from the histogram | **Zero** — same array shape | **None** — the counts are already SecAgg-summed for `p̂` |
| **(B)** global min-arm | **Two-phase refresh.** Phase 1: clients return three unnormalized scalars (`Σℓ` over their h/m/n rows) plus three counts → SecAgg-sum → server computes global `L_h`, `L_m`, applies the **unmodified** `_composite_min_arm`, broadcasts the 1-byte arm choice. Phase 2: clients return the selected-arm gradient and the `L_u` gradient → SecAgg-sum | ≈24 B/client/refresh against a 134,640 B model message — **< 0.02%** | **None** — scalars are SecAgg-summed; the server learns only the pooled `L_h`, `L_m` it already learns in v1 |

**What this does and does not cost.**
- **Bandwidth: unchanged.** The `(1 + ⌈E'/m⌉)×` multipliers in §4.1 **stand** — the added scalars are four orders of magnitude below the model messages.
- **Latency: doubled per refresh.** Each refresh becomes two sequential communication rounds instead of one. That is a real deployment cost and must be reported separately from bytes; conflating the two is how the redesign doc's "6×" came to feel like the whole story.
- **Rejected alternative:** have clients return *both* arm gradients so the server can select post hoc. One round instead of two, but **3× the return traffic** — strictly worse on the frontier, which is the paper's axis. Mention it as the latency-vs-bandwidth trade-off, choose the scalar route.

**⚠️ A measured nuance that makes this more dangerous, not less: Adam hides it.** Tracking the error one step downstream — through `_grad_wrt_weights` and a single Adam step on `w` — the naive protocol's 186%-wrong gradient moves `w` by only `4.3 × 10⁻⁷` from the reference on CIC γ=1.0, because Adam's per-coordinate normalization divides out gradient magnitude. On UNSW γ=0.5, where the *sign structure* differs rather than just the scale, the same step moves `w` by **0.40** — six orders of magnitude more. **So the defect is invisible in the first round on three of the four cells tested and catastrophic on the fourth, then compounds over `T=200` rounds × `E'=5` inner epochs.** A smoke test that trained for a few rounds and compared accuracies would very likely have passed. This is precisely why T-4a/T-4b assert on the **gradient**, not on downstream accuracy.

**Why this was caught, and what it validates.** T-4 (`v2 ≡ v1-projected at ζ=0`) is precisely the test that fails under the naive protocol — for **both** obstructions, independently. **The test design was right; the specification was wrong.** That is the strongest available argument for §9.3's rule that T-4 gates every reported number: had the campaign been run on the naive protocol, every v2 result would have been wrong in a way that looks like a *method* failing rather than a *spec* being incorrect.

**A reference implementation now exists.** `verify_v2_exactness.py` prototypes `stratum_tags`, `apply_tags` and the two-phase `v3_corrected` protocol, and asserts V3 ≡ pooled / V1 ≢ pooled on both datasets at two γ. **W1.1 and W2.3a are ports of working code, not fresh designs**, and T-4a/T-4b are ports of assertions that already pass.

**Changes.** §3's pseudocode is corrected below. Add sub-tests **T-4a** (unnormalized-sum equivalence) and **T-4b** (global-arm equivalence) so the two obstructions are tested in isolation before the combined T-4 (§9.2). Update `privacy_compliant_fedgraa_redesign.md` §4.1's "mathematically identical" row when that document is next revised.

---

### F13 — On UNSW-NB15 at `n ≥ 5`, the oracle rare set is **unrecoverable by any count-based inference** — and that is evidence *for* v2

**Measured** ([`scripts/verify_v2_plan_claims.py`](../../scripts/verify_v2_plan_claims.py), 10 seeds, exact arithmetic, zero noise):

| Dataset | `n=3` | `n=5` | `n=7` |
|---|---|---|---|
| CIC-IDS-2017 | full 10/10 · held-out 10/10 | 10/10 · 10/10 | 10/10 · 10/10 |
| **UNSW-NB15** | 10/10 · 10/10 | **0/10 · 1/10** | **0/10 · 1/10** |

**Cause — and it is structural, not numerical.** [config.py:253-259](../../fedgraa/config.py#L253-L259) sets UNSW's `rare_label_order[4] = "Exploits"`, annotated in the config's own comment as *"44,525 original, capped at 10,000."* Four other UNSW classes (Generic, Fuzzers, Reconnaissance, DoS) are **also capped at exactly 10,000**. Size-ranking the as-used counts therefore hits a five-way tie and returns the lowest class id (Generic, id 1); the oracle returns Exploits (id 7). **The information that distinguishes them — the pre-capping counts — does not exist anywhere in the federated system.** No histogram over the data the clients actually hold can recover it, at any `ε`, with any amount of communication.

**Consequence 1 — a test in the source spec is unsatisfiable.** `privacy_compliant_fedgraa_redesign.md` §11.3 test 8 requires `BottomN` to recover the hard-coded set *"for at least `n ∈ {3,5}` on **both** datasets."* On UNSW at `n=5` that is **impossible by construction**. §9.2's T-8 is re-scoped accordingly. *(Separately: UNSW `n=7` truncates to 5 classes with a `RuntimeWarning` — the documented ambiguity #10 path, confirmed firing.)*

**Consequence 2 — this is the cleanest demonstration available that `rare_label_order` is oracle knowledge.** The premise of the entire v2 redesign is that the hard-coded rare list encodes information a real server cannot have. At `n=3` that is an argument. **At `n≥5` on UNSW it is a demonstrated fact**: the list provably cannot be derived from the federated system's observable state. Put this in the paper. It converts [`Final_Stage_C.md`](Final_Stage_C.md) §7.1's flaw #1 from an assertion into a measurement, and it costs one table.

**Consequence 3 — the `n`-sweep must be scoped.** v1-vs-v2 comparisons on UNSW at `n ≥ 5` do not hold the rare set fixed, so they measure two changes at once. Either restrict head-to-head comparison to `n=3` (recommended), or run both and **report the set difference explicitly** as part of the result rather than letting it contaminate the accuracy delta. CIC-IDS-2017 is unaffected at every `n` and can carry the full sweep.

---

## 3. The corrected FedGRAA-v2 design

Superseding [`privacy_compliant_fedgraa_redesign.md`](privacy_compliant_fedgraa_redesign.md) §4.2's pseudocode, with F2/F3/F4/F5 folded in. **Changes from the original are marked `[Fn]`.**

```
Server executes:
  initialize θ_g^0, w^0 (static reparam, Eq. 7)
  for round t = 1 .. T:
    S_t ← sample of K clients
    for k in S_t (parallel):
      θ_k^t     ← ClientUpdate(k, θ_g^{t-1})              # unchanged (Alg. 1)
      h_k^rank  ← label counts over k's FULL local shard   # [F3] not held-out
      h_k^strat ← label counts over k's HELD-OUT split     # [F3] new, distinct role

    p̂_rank  ← SecAgg-sum({h_k^rank})                       # server sees only the sum
    p̂_strat ← SecAgg-sum({h_k^strat})
    rare_t  ← BottomN(p̂_rank, n)                           # [F3] replaces oracle list
    τ       ← stratum_tags(p̂_strat, rare_t)                # [F2] length-C tag vector

    for e' = 1 .. E':
      if (e' − 1) mod m == 0:                              # [F5] m = grad_refresh_every
        θ'_g ← Σ_k w_k θ_k^t
        broadcast (θ'_g, τ, γ) to S_t                      # [F2] τ, not just rare_t

        # ---- PHASE 1: global arm selection [F12-B] ------------------------
        for k in S_t (parallel):
          (Sℓ_h,k, Sℓ_m,k, Sℓ_u,k) ← UNNORMALIZED loss sums over k's h/m/n rows
          (n_h,k,  n_m,k,  n_u,k)  ← the corresponding row counts
        (Sℓ_s, n_s) ← SecAgg-sum over k, for s ∈ {h, m, u}   # ~24 B/client
        L_h ← Sℓ_h/n_h ;  L_m ← Sℓ_m/n_m ;  L_u ← Sℓ_u/n_u
        arm ← _composite_min_arm(L_h, L_m)                  # UNCHANGED from v1
        broadcast arm to S_t                                # 1 byte

        # ---- PHASE 2: gradients, sum-form [F12-A] -------------------------
        for k in S_t (parallel):
          ĝ_a,k ← n_arm,k · ∇_P L_arm,k(θ'_g)               # UNNORMALIZED
          ĝ_u,k ← n_u,k   · ∇_P L_u,k  (θ'_g)               # UNNORMALIZED
        (ĝ_a, ĝ_u) ← SecAgg-sum over k                      # [F4] MANDATORY
        ∇_P L' ← (1−γ)·(ĝ_a / n_arm) + γ·(ĝ_u / n_u)        # server-side normalize

      w ← Π_Δ( w − η' · ⟨∇_P L', θ_·^t⟩ )                  # [F1] projection — Eq. 10 + Π_Δ

    w^t ← w ; θ_g^t ← Σ_k w_k^t θ_k^t                      # unchanged
  return θ_g^T
```

**Why the normalization is exact.** `ĝ_a = Σ_k n_{a,k}·∇L_{a,k} = Σ_k Σ_{i∈k} ∇ℓ_i` — the per-client `1/n_{a,k}` from `reduce_mean` cancels against the count weight, leaving a plain sum over *rows*. Dividing by `n_a = Σ_k n_{a,k}` then reproduces the pooled mean gradient **exactly**, not approximately. The same cancellation gives the phase-1 scalars: `Sℓ_h/n_h` is the pooled `L_h` that [aggregation.py:175-177](../../fedgraa/aggregation.py#L175-L177) computes in one call.

✅ **One edge case the server must replicate — specified, then verified.** v1's `_loss_and_grad_P` returns `(inf, zeros)` for an **empty** split ([aggregation.py:54-56](../../fedgraa/aggregation.py#L54-L56)), which is what fires `_composite_min_arm`'s NaN-symmetric fallback. In v2 the equivalent condition is `n_s = 0`, so the server must map `n_s = 0 → L_s = inf` **before** calling `_composite_min_arm`, rather than evaluating `0/0`. This is not hypothetical: F3 measured **2 of 10 CIC seeds where a protected rare class has zero rows in `D_V^n`**.

> **Measured** ([`verify_v2_exactness.py`](../../scripts/verify_v2_exactness.py), degenerate-strata block): forcing each of the `h`, `m`, `n` strata empty in turn, on both datasets, the corrected protocol still reproduces the pooled gradient — **6/6 cases, rel. err 2.2 × 10⁻⁷ to 3.8 × 10⁻⁷.** This corner was specified in an earlier draft of this plan and *not* tested; it is tested now, because "specified but untested" is precisely the category F12 came out of.

**Seven invariants this design must preserve, each with a test in §9:**

| # | Invariant | Test |
|---|---|---|
| I1 | Eq. 7–10 are **unchanged**; only the *source* of `∇_P L'` differs | T-4, T-5 |
| I2 | At `m = 1`, no DP, no dropout: v2 ≡ v1-projected to float tolerance | **T-4** (the plan's keystone test) |
| **I2a** | **Count-weighted normalization: `Σ_k n_{s,k}·∇L_{s,k} / n_s` = pooled `∇L_s`** `[F12-A]` | **T-4a** |
| **I2b** | **The arm is selected from the *pooled* `L_h`, `L_m`, never per-client** `[F12-B]` | **T-4b** |
| I3 | The server never observes any individual `h_k`, scalar, or `g_k` | **T-6** (interface-level, not usage-level) |
| I4 | `w^t ∈ Δ` for every `t`, structurally | T-10 |
| I5 | Measured bytes match `(1 + ⌈E'/m⌉) × baseline` to within 1%; **communication *rounds* are reported separately from bytes** `[F12]` | **T-12** |

---

## 4. The exactness–communication frontier — the actual contribution

### 4.1 The reframing

[`Final_Stage_C.md`](Final_Stage_C.md) §10 already made the key move: the interesting question is not *"does the privacy fix work?"* but ***"is exact worth 6×?"*** F5 completes it — with `m` as a knob, that question has a **curve**, and every point on the curve is a publishable answer:

| Outcome | Reading | Publishable? |
|---|---|---|
| Rare-class recall is flat across `m ∈ {1,2,3,5}` | Exactness is **not** what matters; 2× buys everything 6× does | ✅ Strongly — a negative result that saves the field 4× traffic |
| Recall degrades monotonically with `m` | Exactness **is** load-bearing; here is the price of each increment | ✅ Strongly — the trade-off nobody has quantified |
| CELM/SpectralFed at ≈1× match v2-lazy-5 at 2× | Approximation dominates; **say so** and cite them as the better choice | ✅ Honestly — and it is the finding a reviewer will trust most |
| v2 at any `m` beats approximation on rare-class recall specifically | Exactness matters **for rare classes** even where it doesn't for accuracy | ✅ Best case — precisely FedGRAA's evaluation axis |

**There is no arm of this experiment that produces an unpublishable result.** That property is rare and is the strongest argument for Option B.

### 4.2 The headline figure

One figure, two panels (CIC-IDS-2017 | UNSW-NB15). **x** = total training traffic (GB, log scale); **y** = per-minority-class recall, mean over ≥10 seeds with bootstrap 95% CI. Points: v1-projected (0.93/0.50 GB, *privacy-violating* — mark it, don't hide it), v2 at `m ∈ {5,3,2,1}`, CELM, SpectralFed. Annotate each v2 point with its **measured** median `ζ_t`.

That figure is the paper. Everything else supports it.

### 4.3 Why the two datasets must both appear

⚠️ *This subsection's first draft rested on the redesign doc's §7.5 DP finding. §5.2 refuted that finding, so the prediction is re-grounded on evidence that survived.*

**Expect the frontier to have a different shape on each dataset**, for two reasons that are measured rather than assumed:

1. **Stage B found exactly this dataset-dependence, for a different lever.** In the projection-vs-η′ factorial the two datasets were **near mirror images**: on CIC the η′ cut did the work and projection-alone sat at 0% rare-class accuracy at γ≤0.5; on UNSW projection-alone tied the combined fix at every γ while the η′ cut contributed nothing ([`../current/fedgraa_convergence_defect.md`](../current/fedgraa_convergence_defect.md) §5.3). A lever that reverses between these two datasets is the norm here, not the exception.
2. **The client-skew severity differs by orders of magnitude.** CIC's three rarest classes hold 11 / 21 / 36 samples against UNSW's 174 / 1,511 / 2,329, and the same feasibility defect costs CIC up to **1.86 × 10⁶×** composite-loss blowup against UNSW's **46.85×**. Gradient staleness is a perturbation of the same weight-space extrapolation, so it should bite harder where the client models are more disparate.

**Predict divergence, pre-register the prediction, and report both panels.** A single-dataset frontier would be the paper's weakest claim; two divergent frontiers with a mechanistic explanation is one of its strongest. **Do not offer the DP-noise story as that explanation** — §5.2 measured both datasets as DP-viable down to ε=1, so it no longer supports the prediction it was originally introduced for.

---

## 5. Threat model and privacy accounting, corrected

[`privacy_compliant_fedgraa_redesign.md`](privacy_compliant_fedgraa_redesign.md) §10 remains authoritative for actors and composition arithmetic. Four corrections:

**5.1 `[F4]` The gradient channel is a named leak, closed by a mandatory mechanism.** State the attack explicitly (a per-client `g_k` reveals rare-class holding via its `γ`-component), state that SecAgg closes it, and state that the closure is **free** because `_grad_wrt_weights` needs only the sum. Reviewers reward a paper that names its own attack and shows the defence is structural.

**5.2 `[F3]` The DP story is better than the redesign doc concludes — measured, not argued.** `BottomN` recovery rate over 2,000 Laplace trials per cell, `σ = (Δ₁/ε)·√(2K)`, `n=3`:

| Dataset | `σ` @ ε=1 / 4 / 10 | `p̂_rank` (full shard) | `p̂_strat` (held-out) |
|---|---|---|---|
| CIC-IDS-2017 (K=14) | 10.58 / 2.65 / 1.06 | **100.00% / 100.00% / 100.00%** | 99.95% / 100.00% / 100.00% |
| UNSW-NB15 (K=9) | 8.49 / 2.12 / 0.85 | **100.00% / 100.00% / 100.00%** | 99.85% / 100.00% / 100.00% |

**Both signals are DP-viable down to ε = 1**, because what governs `BottomN` is the *gap to the next-rarest class* relative to `σ` — on CIC that gap is Infiltration (36) → XSS (652), a ~6σ separation at ε=1 — not the per-count relative error the redesign doc's §7.5 table reports. The **full-shard signal is still preferred**, but on determinism grounds (F3), not on DP grounds.

**Report the measured recovery rates, not the analytical caveat.** The redesign doc's pragmatic "use SecAgg-only for the ranking signal" is more conservative than the evidence requires, and a formal `(ε,δ)` guarantee on the ranking signal is available essentially for free. Still report the §7.5 relative-error tables for both datasets — they remain the correct statement about *count* accuracy, which matters for anything that consumes counts rather than ranks (the stratification threshold, and any histogram value logged outside the training loop).

> ⚠️ **Scope caveat added 2026-07-28** (independent review alongside
> [`../audit/PRE_COMPUTE_AUDIT_2026-07-28.md`](../audit/PRE_COMPUTE_AUDIT_2026-07-28.md)).
> The `p̂_strat` column above is computed by
> [`scripts/verify_v2_plan_claims.py`](../../scripts/verify_v2_plan_claims.py),
> which adds independent Laplace noise to **both** the rank and stratification
> histograms to produce this table. The shipped `FederatedTrainerV2`
> (`fedgraa/trainer.py`) does **not** do this: `cfg.dp_epsilon` noises only the
> full-shard/`h_rank` histogram that feeds `BottomN` rare-class selection
> (`fedgraa/config.py`'s own comment: *"Laplace-DP budget on **the** rarity
> histogram"*, singular). The held-out/stratification histogram that feeds
> `stratum_tags` — and therefore the exact two-phase gradient split — is
> SecAgg-summed but released to the server **without DP noise**, always,
> regardless of `cfg.dp_epsilon`. This is intentional (see the comment at
> `trainer.py`'s `_histograms_and_tags`) and is covered by a regression test
> asserting exactly `K` Laplace draws per round (one per client, rank channel
> only). **Consequence for T4 and the paper:** the `p̂_strat` numbers above
> describe a stronger DP guarantee (both channels protected) than what T4's
> `ε∈{1,4,10}` cells will actually run (rank channel only). If the paper
> states the protocol satisfies `(ε,δ)`-DP without qualifying which channel,
> that overclaims by this table's own `p̂_strat` half. Either (a) state the DP
> guarantee as covering the rank/rarity-selection channel specifically, not
> "the histograms" generally, or (b) extend the trainer to noise both
> channels before T4 runs, matching this table exactly. (a) requires no code
> change and is the smaller claim already true today.

**5.3 Composition, unchanged.** Report the **composed** `(ε,δ)` over all `T=200` rounds, never a per-round `ε`. Advanced composition gives `ε_total ≈ 8.89` at `ε_round=0.1, δ=10⁻⁵`; use RDP (Mironov, CSF 2017) for the tighter number and report both.

**5.4 Out of scope, stated not assumed.** Honest-but-curious server; SecAgg's standard Shamir collusion threshold; **malicious server and Byzantine clients explicitly excluded**; the `θ_k^t` channel remains DLG-exposed exactly as in v1. Add one sentence noting that [`Final_Stage_C.md`](Final_Stage_C.md) §4.11b's premise conflict — FedGRAA up-weights the statistically anomalous, poisoning defences suppress it — means the Byzantine exclusion is **load-bearing, not incidental**, and points at the natural follow-on paper.

---

## 6. Theory workstream

### 6.1 Target theorem

> **Theorem (informal).** Let `L'` have bounded subgradients on `Δ` and satisfy the PL condition on `Δ`. Let the server's composite-loss gradient estimate satisfy `E‖∇̂_P L' − ∇_P L'‖² ≤ ζ²` (Assumption 5). Then projected inexact subgradient descent `w^{t} = Π_Δ(w^{t-1} − η'∇̂)` with `η' ≤ 1/β_eff` satisfies
> `min_{t≤T} ‖G_{η'}(w^t)‖² = O(1/T) + O(ζ²)`,
> and `w^t ∈ Δ` for every `t ≥ 0` **by construction**.

### 6.2 Build order

| Step | Task | Source | Effort |
|---|---|---|---|
| 1 | Restate Stage B's proposition as the `ζ = 0` special case | [`../current/fedgraa_convergence_defect.md`](../current/fedgraa_convergence_defect.md) §4 — already written | Editing |
| 2 | Add Assumption 5 and the `O(ζ²)` term | Redesign §5 + **Schmidt/Le Roux/Bach 2011** ✅ | ~1 day |
| 3 | Decompose `ζ` into named sources | New: `ζ² ≲ ζ²_secagg + ζ²_dp + ζ²_stale(m) + ζ²_dropout` | ~1 day |
| 4 | Swap convexity for PL | Karimi/Nutini/Schmidt 2016 | ~1 day |
| 5 | Partial participation | Li et al., ICLR 2020 | ~1 day |
| 6 | **Optional, high leverage:** derive `w` *from* the bound | FedAAW (KBS 2024) — requires obtaining the paper | ~1 week |
| 7 | Harden `β_eff`, the `min()` sub-differential, the rate constant | Stage B §4's own three flagged items — **mentor work** | Mentor-gated |

**Step 3 is the one that ties theory to experiment**, and it is the step no existing document has. Decomposing `ζ` by source means the frontier experiment (§4) is a **direct empirical measurement of `ζ_stale(m)`**, with `ζ_secagg` and `ζ_dp` measured independently as controls. Theory and experiment then test the same object — which is what makes a theory section load-bearing rather than decorative.

⚠️ **`[F12]` The decomposition has no `ζ_protocol` term — and that is a claim, not an omission.** The two-phase count-weighted protocol (§3) is designed so that the federated computation is *algebraically identical* to the pooled one, leaving `ζ_protocol = 0` exactly. **The naive summation protocol would contribute a large, `m`-independent, client-size-dependent `ζ_protocol` that no amount of refreshing removes** — it would sit as a constant floor under the entire frontier and would be indistinguishable, in the results, from "exactness does not help." T-4a/T-4b are what license writing `ζ_protocol = 0` in the theorem; until they are green, that term must be carried explicitly.

### 6.3 Standing discipline

Stage B's `min(L_h, L_m)` sub-differential treatment is a **standard-practice sketch, not a formal derivation**, and says so. Keep that honesty. Do not upgrade any hedge in Stage B's rigor flag without doing the work — the project's credibility is built on exactly that habit.

---

## 7. Experimental design — two grids, tiered

### 7.1 What is already built and must not be rebuilt

Per [`../current/STAGE_B_MASTER_REPORT.md`](../current/STAGE_B_MASTER_REPORT.md) §22.1, Stage C inherits, green and tested:

- [`fedgraa/stats.py`](../../fedgraa/stats.py) — `bootstrap_ci`, `paired_ttest`, `wilcoxon_signed_rank`, `cohens_d_paired`, `holm_bonferroni`, with tested degenerate-cell contracts.
- [`scripts/run_ablation_seed_sweep.py`](../../scripts/run_ablation_seed_sweep.py) — resumable, worker-capped, `(arm, seed)`-enumerated. **v2 arms plug into the existing enumeration; do not write a new driver.**
- [`scripts/build_ablation_table.py`](../../scripts/build_ablation_table.py) / [`render_ablation.py`](../../scripts/render_ablation.py) — consolidation and rendering.
- [`scripts/check_stage_b_complete.py`](../../scripts/check_stage_b_complete.py) — the completeness-gate pattern; clone as `check_stage_c_complete.py`.
- Two forensically-audited canonical datasets, and **155 passing tests** — ⚠️ not the 142 that report records; the 2026-07-26 audit commit added 13 (§13.3).

### 7.2 Compute reality, and the tiering it forces

At Stage B's measured **~15–20 min per (arm, seed)** on CPU, the [`Final_Stage_C.md`](Final_Stage_C.md) §9.1 grid — ~24 baselines × 2 datasets × 10 seeds, plus 3 γ for γ-bearing methods — is **≈6–9 CPU-weeks**. It cannot be run as listed. Tiering:

| Tier | Contents | Runs | Est. CPU | Gate |
|---|---|---|---|---|
| **T1 — Frontier** *(contribution)* | v1-projected + v2 at `m∈{1,2,3,5}` × 2 datasets × 10 seeds × γ=1.0 | 100 | **28 h** | **Must** — this is the paper |
| **T2 — γ interaction** | T1 arms at γ ∈ {0, 0.5} | 200 | 57 h | **Must** — γ is the contribution FedGRAA has |
| **T3 — Approximation baselines** | CELM, SpectralFed, CoAst, FedTruth, RAGA, FLTrust × 2 × 10 | 120 | 34 h | **Must** — the frontier needs its right-hand anchor |
| **T4 — Robustness controls** | `p_drop ∈ {0.1, 0.3}`, DP at `ε ∈ {1,4,10}` on best `m` | 100 | 28 h | **Should** |
| **T5 — Novelty-defence grid** | FedPGO, FedMADE, SBFedNIDS, FedAPA, FedAWA, DA-PFL, FedAWARE, AdaAggRL × 2 × 10 | 160 | 45 h | **Should** — one comparison table, not the Results section |
| **T6 — NF-UQ-NIDS-v2** | T1 repeated on the cross-silo benchmark | 100 | 28 h + data eng. | **Should** — see §7.4 |
| **T7 — Adversarial** | SecFedNIDS protocol + the rare-class-fabrication attack | ~80 | 23 h | **Nice** — or a follow-on paper |

*(Derived in [`scripts/verify_v2_plan_claims.py`](../../scripts/verify_v2_plan_claims.py) at Stage B's measured 15–20 min/run, midpoint 17 min. Treat as ±20%.)*

**T1–T3 = 119 CPU-hours ≈ 5.0 days wall-clock at 1 worker, 1.7 days at 3 workers.** That is the honest minimum viable campaign, and it is entirely reachable on the current machine once §13 is fixed. **T5 is the item to trade for lab GPU/cluster access in the mentor conversation** — it is the largest block and the least essential to the contribution.

⚠️ **`n_rare` scoping, per F13.** Every grid above fixes **`n = 3`**. Any `n`-sweep extension must **exclude UNSW-NB15 at `n ≥ 5`** from v1-vs-v2 head-to-head comparison, because the oracle and inferred rare sets provably differ there and the comparison would confound two changes. CIC-IDS-2017 carries the full `n ∈ {3,5,7}` sweep unaffected.

### 7.3 Statistical protocol — inherited, unchanged

≥10 seeds; mean + bootstrap 95% CI (B=10,000, fixed RNG); paired *t* **and** Wilcoxon; Cohen's *d*; **Holm–Bonferroni across the full comparison set within each dataset × γ cell**; fixed disclosed seed list `[0..9]`. Two Stage B lessons that must carry over:

- **The Wilcoxon floor.** At n=10 the smallest attainable two-sided p is `2/2¹⁰ = 0.001953`. Report it as *"as significant as 10 seeds can demonstrate,"* never as "p = 0.002."
- **Always test the second metric.** Stage B §5.3 found that two headline "ties" on rare-class accuracy **dissolved** when all-class accuracy was tested — one arm had traded 29pp of majority-class accuracy for rare-class parity. **Every frontier point must report per-minority-class recall *and* all-class accuracy.** A frontier that only reports the headline metric will find a hollow tie and publish it.

### 7.4 Why NF-UQ-NIDS-v2 (Option D) is not optional for *this* angle

Under the one-attack-per-client synthetic partition, "the server does not know client label distributions" is a claim about a simulation the experimenter designed. Under **NF-UQ-NIDS-v2** — four independently-collected NIDS datasets in one 43-feature schema, one per silo — non-IID structure is *real*, and the privacy claim becomes a statement about a deployment. It closes [`Final_Stage_C.md`](Final_Stage_C.md) §9 gaps #4 and #5 **and** the §7.1 oracle-partition objection simultaneously, with **data engineering only and no algorithmic risk**. Start the feature-extraction pass in parallel with W1 so T6 is not serialized behind the frontier campaign.

---

## 8. Implementation plan — seven workstreams

### 8.1 W0 — Prerequisites *(no new algorithm; ~1–2 days)*

| ID | Task | Files | Gate |
|---|---|---|---|
| ~~W0.0~~ | ~~Fix the interpreter (§13)~~ | environment | ✅ **DONE** 2026-07-27 |
| ~~W0.1~~ | ~~Re-establish the green baseline~~ | `tests/` | ✅ **DONE** — **155 passed** in 320 s. This is the regression floor (not the 142 in `STAGE_B_MASTER_REPORT.md` §22.1) |
| **W0.2** | Promote `ProjectedFedGRAA` from `scripts/` into `fedgraa/aggregation.py` as a **non-default** class `[F1]` | `aggregation.py`, `trainer.py` | Strict-reproduction default **unchanged** — assert it in a test. ✅ **IMPLEMENTED** — `test_strict_reproduction_default_unchanged` (T-19) |
| **W0.3** | **Zero-compute oracle check** `[F7]`,`[F13]` — ✅ **already executed in the verification pass** (§14.1). Land it as tests T-8/T-8b/T-8c/T-8d at the **corrected scope**: `n=3` both datasets, `n ∈ {5,7}` **CIC only** | `config.py`, `partition.py`, new test | Already de-risked; landing it makes the result permanent. ✅ **IMPLEMENTED** — `tests/test_partition_v2.py` |
| W0.4 | Rename `FedSNDTrainer` → `StaticDistributionAwareTrainer`; purge the name from artifacts, legends, JSON keys `[F6]` | `baselines.py`, `plot.py`, `results/` | Before any published table. ✅ **IMPLEMENTED** |

### 8.2 W1 — Refactor for federation *(behaviour-preserving; ~2–3 days)*

Every W1 task is a **pure refactor with the existing test suite as its oracle**. No new behaviour, no new numbers.

| ID | Task | Files |
|---|---|---|
| W1.1 | Split `median_frequency_split` → `stratum_tags(counts, rare_ids)` + `apply_tags(X, y, τ)`; re-express the original as their composition `[F2]` | `partition.py` — ✅ **IMPLEMENTED** |
| W1.2 | Split `_composite_grad_P(params, val)` → per-client `composite_grad_contribution(params, client_val, τ, γ)`; server path becomes a sum over clients | `aggregation.py` — ✅ **IMPLEMENTED** as `two_phase_composite_grad` (W2.3a, byte-optimal single-array variant, see below) |
| W1.3 | Add `local_label_histogram(shard, C, scope="full"\|"heldout")` `[F3]` | `partition.py` — ✅ **IMPLEMENTED** |
| W1.4 | Add `confusion_matrix(params, X, y, n_classes) -> np.ndarray`; derive `per_class_metrics` from a *pooled* matrix | `metrics.py` — ✅ **IMPLEMENTED** |
| W1.5 | Add `FedConfig` fields: `grad_refresh_every: int = 1`, `secagg: bool = False`, `dp_epsilon: Optional[float] = None`, `p_drop: float = 0.0` | `config.py` — ✅ **IMPLEMENTED** (`secagg` defaults `True`, per F4 "mandatory not optional") |

**W1 exit gate: the full existing suite passes with zero modifications to any existing test.** If an existing test needs editing, the refactor changed behaviour and must be redone.

### 8.3 W2 — New mechanisms *(~3–4 days)*

| ID | Task | New file |
|---|---|---|
| W2.1 | `SecAggChannel` — interface contract makes individual contributions **unreachable**, not merely unused `[F4]`; Shamir threshold `t`; dropout handling | `fedgraa/secagg.py` — ✅ **IMPLEMENTED**; T-6 checks unreachability by introspection, T-7 checks the dropout threshold boundary |
| W2.2 | Laplace mechanism, `b = Δ₁/ε`, `Δ₁ = 2`; advanced-composition **and** RDP accountants (wrap a standard one, do not hand-roll) | `fedgraa/privacy.py` — ✅ **IMPLEMENTED** as closed-form accountants (no third-party DP package was pinned in `requirements.txt`; adding one unvetted mid-project was judged worse than a cited closed-form derivation — see `FEDGRAA_V2_THEORY.md`); matches this doc's own §5.3 worked example (8.89) to 2 d.p. |
| W2.3 | `FedGRAAv2(ProjectedFedGRAA)` — accepts an externally-supplied `∇_P L'`; implements the `m`-refresh loop `[F5]` | `fedgraa/aggregation.py` — ✅ **IMPLEMENTED**; refresh count verified `== ceil(E'/m)` for every `m ∈ {1,2,3,5}` |
| **W2.3a** | ⚠️ **`[F12]` The two-phase exact protocol** — phase-1 scalar exchange + server-side `_composite_min_arm`; phase-2 **count-weighted, sum-form** gradient exchange. **This is the task the contribution depends on; T-4a/T-4b must be written before it, not after** | `fedgraa/aggregation.py` — ✅ **IMPLEMENTED** as `two_phase_composite_grad`, refined beyond the §3 pseudocode: phase 2 broadcasts `n_arm`/`n_n` (safe, pooled aggregates) so each client pre-combines into **one** model-sized array instead of two, which is what makes T-12's measured bytes match this doc's own `(1+⌈E'/m⌉)×` formula exactly rather than ~1.5× over it |
| W2.4 | `FederatedTrainerV2` — orchestrates histogram → SecAgg → tags → two-phase backward pass; **instruments every message for T-12, counting bytes *and* communication rounds separately** `[F12]` | `fedgraa/trainer.py` — ✅ **IMPLEMENTED**; T-12 green at `<1%` error for `m ∈ {1,2,3,5}` |
| W2.5 | `zeta_probe` — computes `∇_P L'` twice per eval round (oracle pooled + federated) and logs `ζ_t` | `fedgraa/trainer.py` — ✅ **IMPLEMENTED**, with a disclosed scope limit: the per-round log captures `ζ_secagg`/`ζ_dp`'s downstream effect but not `ζ_stale(m)` continuously (see `FEDGRAA_V2_THEORY.md` §3) |

### 8.4 W3–W6

| WS | Content | Depends on | Effort |
|---|---|---|---|
| **W3** | Test suite (§9) — **T-4 and T-12 are hard gates** on every number after them | W2 | ~3 days |
| **W4** | T1–T3 campaign; `check_stage_c_complete.py`; consolidation + frontier figure | W3 green | ~1 week (2 days compute) |
| **W5** | Theory writeup (§6 steps 1–5) | Independent — **start in parallel with W1** | ~1 week |
| **W6** | NF-UQ-NIDS-v2 ingest + T6 | Independent — **start in parallel with W1** | ~1 week |

**Critical path: W0 → W1 → W2 → W3 → W4 ≈ 3 weeks of focused work.** W5 and W6 are genuinely parallel and should be started immediately, not queued.

---

## 9. Test plan

Superseding `privacy_compliant_fedgraa_redesign.md` §11. Of its 17 tests, **6 stand unchanged and 11 are corrected, re-scoped, replaced or added** — the churn is concentrated on the tests that F12, F13, F3 and F1 touched, which is the expected shape given that those four findings hit the specification rather than the implementation.

### 9.1 Conventions

One file per module, pytest, fixtures from [`tests/conftest.py`](../../tests/conftest.py). New files: `test_aggregation_v2.py`, `test_secagg.py`, `test_privacy.py`, `test_trainer_v2.py`, `test_partition_v2.py`.

### 9.2 The tests

| ID | Test | Status | Assertion |
|---|---|---|---|
| T-1..3 | Confusion-matrix additivity, macro-F1 from pooled matrix, missing-class handling | Unchanged | Exact integer equality; `1e-12` on derived ratios |
| **T-4** | **`test_v2_matches_v1_projected_at_zeta_zero`** | ⚠️ **Corrected `[F1]`** — reference is **v1-projected**, not v1-literal | `∇_P L'`, `grad_w`, `w^t` all match within `1e-9`. **The keystone test. No v2 number is trustworthy before it is green.** Decomposes into T-4a/T-4b — write those first, because each isolates one of F12's two obstructions and a combined failure is otherwise undiagnosable |
| **T-4a** | **`test_count_weighted_sum_equals_pooled_gradient`** | 🆕 **New `[F12-A]`** | For randomly-sized shards, `Σ_k n_k·∇L_k / Σ_k n_k` equals the pooled `∇L` within `1e-9`, **and** the naive `Σ_k ∇L_k` does **not** — the second assertion is what stops the bug from being reintroduced |
| **T-4b** | **`test_arm_selected_globally_not_per_client`** | 🆕 **New `[F12-B]`** | Construct a split where ≥1 client's local `argmin(L_h, L_m)` differs from the pooled one (trivial under one-attack-per-client); assert v2 selects the **pooled** arm and matches v1, and that a per-client-arm variant measurably diverges |
| T-5 | `test_v2_multi_round_trajectory_matches_v1_projected` | ⚠️ Corrected `[F1]` | `T=10`; `all_acc`, `last_n_acc`, `composite_loss` within `1e-6` at every eval round |
| **T-6** | **`test_secagg_channel_hides_individual_contributions`** | ⚠️ **Strengthened `[F4]`** | Not "the server doesn't use `g_k`" but **"the server-side API has no method returning an individual contribution."** Structural, checked by introspection |
| T-7 | `test_secagg_dropout_tolerance` | Unchanged | Valid renormalized vector when survivors ≥ `t`; **explicit error**, never a silent wrong answer, when survivors < `t` |
| **T-8** | **`test_bottom_n_recovers_hardcoded_rare_set`** | ⚠️ **Re-scoped `[F7]`,`[F13]`** — the redesign doc's *"`n ∈ {3,5}`, both datasets"* is **unsatisfiable** | **`n = 3` on both datasets** (measured 10/10) **plus CIC-only at `n ∈ {5,7}`** (measured 10/10). Runs at **W0.3**, zero compute |
| **T-8b** | **`test_unsw_oracle_rare_set_is_not_count_derivable_at_n5`** | ⚠️ **Replaced `[F13]`** — the first draft's version asserted a difference that does not exist | **Documents the known divergence as a test:** at `n=5` on UNSW, `BottomN(used_counts, 5)` returns Generic while `rare_label_order[:5]` returns Exploits. Asserts the disagreement and its cause (five classes tied at the 10,000 cap), so a future config change cannot silently alter it |
| **T-8c** | **`test_rank_histogram_is_deterministic_across_seeds`** | 🆕 **New `[F3]`** | Full-shard summed histogram equals `used_counts` **exactly for every class on every seed**; held-out histogram does not. This is the assertion that actually encodes F3's real justification |
| **T-8d** | **`test_rare_class_may_be_absent_from_heldout_pool`** | 🆕 **New `[F3]`** | Regression guard on a **v1** fragility: on seeds 5 and 9 (CIC, `n=3`) a protected rare class has **zero** rows in the pooled `D_V^n`. Pin the behaviour so the Limitations claim stays true, and so any future stratified-split change is detected |
| T-9 | `test_dp_noise_matches_closed_form_std` | Unchanged | Empirical σ within 10% of `(Δ₁/ε)·√(2K)`, N≥1000, 3+ seeds |
| **T-10** | **`test_v2_weights_stay_on_simplex`** | ⚠️ **Rewritten `[F1]`** | Original referenced nonexistent v1 projection tests. New: for every v2 arm and every `m`, `|Σw − 1| < 1e-6` and `min(w) ≥ 0` at **every** round — mirroring Stage B's 60-cell zero-exception result |
| T-11 | `test_v2_full_run_comparable_accuracy` | Unchanged | Held-out seed; `\|Δ all_acc\| < 0.02`; `last_n_acc(v2) > 0.5 × last_n_acc(v1p)` |
| **T-12** | **`test_measured_bytes_match_frontier_formula`** | ⚠️ **Generalized `[F5]`** | For **every** `m ∈ {1,2,3,5}`, measured bytes within 1% of `(1 + ⌈E'/m⌉) × baseline`. **Hard gate on any published communication number** |
| T-13 | `test_dropout_robustness_end_to_end` | Unchanged | Completes at `p_drop ∈ {0,0.1,0.3}`; accuracy non-increasing in `p_drop` |
| T-14/15 | ≥10 seeds present; no significance claim without p-value + effect size | Unchanged | Pipeline invariants, fail loudly |
| T-16/17 | Property-based: additivity over random partitions; `Δ₁ = 2` for neighbouring histograms | Unchanged | `hypothesis` |
| **T-18** | **`test_zeta_decomposition_sources_are_separable`** | 🆕 **New `[F5]`, §6.2 step 3** | With exactly one noise source active at a time, measured `ζ` attributes to that source and is ≈0 for the others — the empirical check that the theory's `ζ²` decomposition is well-posed |
| **T-19** | **`test_strict_reproduction_default_unchanged`** | 🆕 **New `[F1]`** | `FederatedTrainer(method="fedgraa")` still constructs the **unprojected** `FedGRAA`. Guards the no-silent-patching mandate against the W0.2 promotion |

### 9.3 Two hard gates

- **T-4 (with T-4a and T-4b) gates every accuracy number.** Until v2 provably reduces to v1-projected at `ζ=0`, any v2 result could be a bug rather than a method. **F12 is the proof that this gate earns its cost**: the naive protocol both source documents specify fails T-4a *and* T-4b, and would have produced a full campaign of confidently wrong numbers. Write T-4a and T-4b **before** the v2 trainer, not after.
- **T-12 gates every communication number.** The 6×/4×/3×/2× multipliers are the paper's quantitative spine; they must be *measured facts about the code*, not hand-derived estimates. **The redesign doc's 2.4% UNSW overstatement (§4.1) is what hand-derived byte counts look like when they go wrong.** T-12 must assert against `spec.model_config().layer_sizes`, never against a literal.

---

## 10. Positioning, claims, and the do-not-write list

### 10.1 Inherit Final_Stage_C §8.1 verbatim

[`Final_Stage_C.md`](Final_Stage_C.md) §8.1's Related Work paragraph is **final wording**. Use it. It is narrow, concedes overlaps pre-emptively, and does not lean on the venue qualifier that *that document's* §6.1 shows is quantitatively weak.

### 10.2 The v2 contribution sentence

Building on [`Final_Stage_C.md`](Final_Stage_C.md) §7.4's narrowed statement, with `[F5]` and the FedMADE finding folded in:

> "Validation-driven adaptive aggregation for federated intrusion detection rests on a server-held labelled set — in FedGRAA, in FLTrust's root dataset, in FedGreed (2025), and in FedMADE, whose server constructs an auxiliary dataset of ten samples per class to compute the aggregation weights. We remove the assumption via secure-aggregated class-frequency estimation and a federated backward pass that reproduces the composite-loss gradient **exactly**, and we show that exactness is a **tunable** property: a refresh-interval parameter traces a frontier from 6× to 2× total training traffic. Benchmarked against contemporary data-free contribution estimators that approximate the same signal at near-zero cost (CELM, SpectralFed, 2026), we report what each increment of exactness buys in per-minority-class recall and what it costs in bytes — making FedGRAA the first fully FL-compliant gradient-refined adaptive-aggregation method **for intrusion detection**, with the compliance cost measured rather than asserted."

⚠️ **This sentence is not yet writable.** Its second half asserts *what exactness buys in per-minority-class recall* — the frontier's accuracy axis, which **T1–T3 have not run** (§14.3). Everything before "Benchmarked against" is already evidenced; everything after it is a promise. Do not paste this into a draft before G5.

### 10.3 The do-not-write list — inherited, plus four

[`Final_Stage_C.md`](Final_Stage_C.md) §8.3's three stand:
1. ❌ "The intersection is unoccupied" — FedPGO occupies it on CIC-IDS2017.
2. ❌ "Existing adaptive aggregators all suppress minorities" — the counter-current is five papers deep.
3. ❌ "We are the first to add a simplex projection" — CELM treats it as a basic design requirement.

Four more from this pass:

4. ❌ **Any sentence asserting what FedPGO *fails* to do.** Its full text is unobtainable across six channels ([`Final_Stage_C.md`](Final_Stage_C.md) §9.0a). Cite at abstract-level confidence only. *Dated correction, 2026-07-31: the constraint on **reading** FedPGO is lifted — the full text became local on 2026-07-30 (committed `9d20ac9`, [`Final_Stage_C.md`](Final_Stage_C.md) §14) and the mechanism was reimplemented faithfully (`25611ce`) with the rerun results committed (`96adb05`). What survives of item 4 is its original shape: any sentence about what FedPGO fails to do must now be grounded in the primary text and the faithful implementation's disclosed judgment calls (JC1–JC8), not in the abstract.*
5. ❌ **"Negligible overhead."** It is 2–6×. Say the number. `[F5]`
6. ❌ **Any comparison table containing "FedSND."** `[F6]`
7. ❌ **"FedGRAA-v2 converges"** without the `O(ζ²)` term and the stationarity-not-global-optimality qualifier. Stage B's honesty on this is the project's strongest habit; do not spend it.

### 10.4 Venue

PAKDD is **CORE B** (ICORE2026, re-confirmed). Computer Networks — FedPGO's venue — has **3.1×** LNCS's 2-year mean citedness. **Do not argue novelty from venue rank.** Position PAKDD honestly as a near-term target and the frontier result as the AAAI/IJCAI-tier extension.

---

## 11. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| ~~R1~~ | ~~**Environment stays blocked** `[F11]`~~ | — | — | ✅ **CLOSED** 2026-07-27. Suite green at 155 tests |
| R2 | **T-4 never goes green** — v2 has a real arithmetic discrepancy with v1 | ⚠️ **Low** *(was Medium)* `[F12]` | High | ✅ Largely retired by measurement: the naive protocol was measured *failing* (160–557% error) **and** the §3 protocol measured *passing* (≈2 × 10⁻⁷), so [`verify_v2_exactness.py`](../../scripts/verify_v2_exactness.py) is a working reference implementation. Residual risk is only that the port into `fedgraa/` regresses it — which T-4a/T-4b catch. **Write those two tests first regardless** |
| R3 | **Frontier is flat** — no measurable recall difference across `m` | Medium | **Low** — that is a publishable negative result (§4.1) | Pre-register the prediction; report either way |
| R4 | **CELM/SpectralFed dominate at ≈1×** | Medium | Medium | Report it honestly; the rare-class axis is where exactness should matter and is FedGRAA's actual claim |
| R5 | **Compute overrun** on T5 | High | Medium | T5 is explicitly Should-tier; trade it for lab compute in the mentor conversation (§12) |
| R6 | **Someone publishes the same fix** | Low-medium | High | Saturation is 2× annualized ([`Final_Stage_C.md`](Final_Stage_C.md) §1.3 N1). Re-run the duplication test at W4 exit; ECGR (arXiv:2601.03584) is the named term-adjacent watch item |
| R7 | **Provenance question unresolved** — who owns FedGRAA | Open | **Branches everything** | Mentor conversation, question 1. Do not draft a resubmission before it is answered |
| R8 | **NF-UQ-NIDS-v2 feature extraction is harder than scoped** | Medium | Low | It is Should-tier and parallel; T1–T3 stand alone on the existing two datasets |
| R9 | **Reviewer asks why γ is still manual** | **High** | Low | One sentence in Limitations citing **XAI-SOH-FL** as the same-domain precedent for automating it ([`Final_Stage_C.md`](Final_Stage_C.md) §4.12). Cheap; do not skip |
| R10 | **Dataset labelling errors undermine rare-class numbers** | Medium | Medium | Cite **Flood et al., EuroS&P 2024** (not El Mahdaouy et al.); **lead with** the ≥10-seed/Wilcoxon/Holm apparatus for rare classes rather than appending it |

---

## 12. Sequencing, gates, and the mentor conversation

### 12.1 Gate table

| Gate | Condition | Blocks |
|---|---|---|
| ~~**G0**~~ | ~~`import fedgraa` succeeds; full suite green~~ | ✅ **PASSED** 2026-07-27 — 155 tests |
| ~~**G1**~~ | ~~W0.3 passes — online rarity inference recovers the oracle set~~ | ✅ **PASSED** 2026-07-28 — `tests/test_partition_v2.py` |
| ~~**G2**~~ | ~~W1 complete with **zero** existing-test edits~~ | ✅ **PASSED** 2026-07-28 — 155/155 unmodified, confirmed before W2 began |
| ~~**G3**~~ | ~~**T-4a, T-4b, then T-4 green**~~ | ✅ **PASSED** 2026-07-28 — `tests/test_aggregation_v2.py`, `<10⁻⁶` rel. error |
| ~~**G4**~~ | ~~**T-12 green** for all `m`~~ | ✅ **PASSED** 2026-07-28 — `tests/test_trainer_v2.py`, `<1%` error for `m ∈ {1,2,3,5}` |
| **G5** | T1–T3 complete, `check_stage_c_complete.py` passes | ⏳ **Not run — the compute campaign.** `check_stage_c_complete.py` also not yet written (no campaign output to check the shape of yet); see `REMAINING_COMPUTE_WORK.md` §6 |
| **G6** | Duplication test re-run at W4 exit | ⏳ Blocked on G5 |

### 12.2 Order of operations

```
[DONE] W0.0 env ──► [DONE] W0.1 155 green ──► [DONE] W0.2 ──► [DONE] W0.3 [G1 ✅]
                                                              │
                        ┌─────────────────────────────────────┤
                        ▼                    ▼                ▼
              [DONE] W1 refactor [G2 ✅]  [DONE] W5 theory  [DONE] W6 NF-UQ-NIDS-v2 scaffold
                                                                (real data acquisition still open)
                        │
                        ▼
              [DONE] W2 mechanisms  (W2.3a shipped as a byte-optimal refinement
                                     of the measured v3_corrected, see §8.3)
                        │
                        ▼
              [DONE] W3 tests [G3 ✅][G4 ✅]  -- 2026-07-28, this pass
                        │
                        ▼
                  W4 campaign T1-T3 [G5] ──► frontier figure ──► draft ──► [G6]

           ◄──────────── you are here (2026-07-28) ────────────►
           All code/test workstreams (W0-W3, W5, W6) complete.
           What remains is compute (T1-T7 on Modal), external-baseline
           reproduction (T3/T5), and real NF-UQ-NIDS-v2 data acquisition --
           see REMAINING_COMPUTE_WORK.md for the itemized map.
```

### 12.3 The mentor conversation — four questions plus one new input

The four questions from [`../current/STAGE_B_MASTER_REPORT.md`](../current/STAGE_B_MASTER_REPORT.md) §22.2 are unchanged: **provenance**, **Stage C angle**, **compute**, **authorship**. Two things this plan adds:

**New input — the angle question is now cheaper to answer than it looks.** The trade-off has shifted since §22.2 was written. Option B's cost is no longer "implement a redesign and hope it is interesting"; it is **~3 weeks on the critical path to a frontier experiment with no unpublishable outcome** (§4.1), on top of infrastructure that already exists (§7.1). Option A (few-shot/prototype hybrid) still has **no design doc, no byte accounting, no test spec, no codebase mapping**, and faces fresher 2025 competition at AAAI/IJCAI/TIFS. That asymmetry is now much larger than the literature review could see.

**Concrete ask — trade T5 for compute.** The novelty-defence grid (**45 CPU-hours, 160 runs**) is the single largest block and the least essential to the contribution. It is the natural thing to ask lab GPU/cluster time for, and framing it that way makes the compute request specific rather than open-ended.

---

## 13. Environment — ✅ resolved, recorded for the reproducibility manifest

### 13.1 What the failure was

```
$ .venv313/Scripts/python.exe -m pytest tests/ -q --collect-only
14 tests collected, 16 errors in 78.16s

ImportError: DLL load failed while importing _multiprocessing:
An Application Control policy has blocked this file.
```

**Chain:** `import fedgraa` → `fedgraa/model.py:22 import tensorflow` → `tensorflow.python.distribute.combinations` → `multiprocessing.pool` → `_multiprocessing.pyd` → blocked by Windows Application Control. Not a missing dependency — the `.pyd` was present; the policy blocked that interpreter. Stage B ran clean on 2026-07-26, so it was newly enforced.

### 13.2 How it was resolved, and why the resolution matters

**Fixed by the author, 2026-07-27, on the *original* `.venv313` interpreter** — the Application-Control route, not the interpreter-swap fallback this plan's first draft recommended. That is the better of the two outcomes and worth recording explicitly:

> **There is no version drift.** The suite now runs on the *same* interpreter and the *same* resolved package set that produced Stage A's and Stage B's numbers, so every cross-stage comparison — the v1-vs-v2 deltas, the frontier against Stage B's factorial arms — remains valid without re-baselining. Had the venv been rebuilt on a different interpreter, that comparability would have needed re-establishing first, which the first draft flagged as the worse failure mode.

### 13.3 Resolved environment — for the reproducibility manifest

| Component | Version | Satisfies `requirements.txt` |
|---|---|---|
| Interpreter | `C:\Dev\Research\PAKDD\.venv313` — **unchanged from Stage A/B** | — |
| Python | 3.13.14 | ✅ (3.9–3.13 per the TF constraint) |
| TensorFlow | 2.21.0 | ✅ `>=2.21.0` |
| NumPy | 2.5.1 | ✅ `>=1.24` |
| SciPy | 1.18.0 | ✅ `>=1.10` |

```
$ .venv313/Scripts/python.exe -m pytest tests/ -q
155 passed, 366 warnings in 320.21s (0:05:20)
```

**155 is the regression floor** — 13 higher than the 142 recorded in [`../current/STAGE_B_MASTER_REPORT.md`](../current/STAGE_B_MASTER_REPORT.md) §22.1, from the 2026-07-26 audit commit. Any workstream in §8 that lands with fewer than 155 passing has regressed something.

### 13.4 One durable note

[`scripts/verify_v2_plan_claims.py`](../../scripts/verify_v2_plan_claims.py) deliberately loads `config.py`/`partition.py` **without** importing the `fedgraa` package, so it keeps working if the TF import path ever breaks again. Keep that property when editing it — it is what let this plan's numbers be verified while the blocker was live.

---

## 14. Method, verification, and AI disclosure

### 14.1 What this pass was — two passes, and the second corrected the first

**Pass 1 — planning, 2026-07-27.** Read [`Final_Stage_C.md`](Final_Stage_C.md) in full; read `privacy_compliant_fedgraa_redesign.md` §4, §5, §7, §11, §13, §15; read `fedgraa_convergence_defect.md` §4–§7 and `STAGE_B_MASTER_REPORT.md` §22; read the live source of `aggregation.py`, `partition.py`, `trainer.py`, `config.py`, `baselines.py`, `metrics.py`, `stats.py`; execute the test suite; cross-check every load-bearing design claim against the code. Produced F1–F11.

**Pass 2 — verification, same day, at the author's instruction to "verify and review it completely."** Re-read every cited line range; **executed** `partition_non_iid` across 10 seeds × 2 datasets × `n ∈ {3,5,7}`; ran 2,000 Laplace-DP trials per (dataset, ε); and re-derived every number in §4 and §7.2 from `spec.model_config().layer_sizes` rather than from the source document's arithmetic. Produced **F12** and **F13**, and **refuted both stated grounds of F3**.

**What pass 2 changed:**

| Item | Pass-1 status | Pass-2 outcome |
|---|---|---|
| **F12** — exactness protocol | Not seen. Pass 1 reproduced the redesign doc's "mathematically identical" summation verbatim | 🆕 **Refuted.** Two independent obstructions found in the live source; §3's pseudocode rewritten |
| **F13** — UNSW `n≥5` oracle | Not seen | 🆕 **New.** A test in the source spec is unsatisfiable; the finding is *evidence for* v2's premise |
| **F3** — "fails by construction", "DP unusable" | Asserted from an SNR calculation | ❌ **Both refuted by measurement.** Recommendation survives; justification replaced |
| §4 UNSW byte column | Inherited from the redesign doc | ⚠️ **Corrected.** `n_features=43` in the doc vs. **39** live → 2.4% overstatement |
| §7.2 compute totals | Rounded up per tier, then summed | ⚠️ **Corrected.** 119 CPU-h, not 130; T5 45 h, not 55 h |
| F10a — pooled `D_V` | Inferred from reading | ✅ **Upgraded to bit-identical, verified by execution** |
| Benign double-counting | Flagged as a possible hazard | ✅ **Not an issue** at current configs (ratio 1.000); the circular-indexing wrap path still exists |

**Pass 3 — post-environment-fix audit, 2026-07-27.** The author resolved the Application Control blocker (§13), which lifted the one constraint pass 2 could not work around. Re-ran the full test suite (**155 passed**), re-ran pass 2's claim script (all PASS), and — the substantive item — **built a measurement for F12**, which §14.3 had explicitly flagged as the plan's last "argument, not measurement."

**What pass 3 changed:**

| Item | Before pass 3 | After |
|---|---|---|
| **F12** — the exactness fault | Argued from reading `model.py` and `aggregation.py` | ✅ **Measured.** Naive protocol **160–557%** relative error; corrected protocol **≈2 × 10⁻⁷**. Obstruction (A) shown fatal on its own |
| **F12** — Adam masking | Not seen | 🆕 **New.** Adam's normalization hides the defect in the first step on 3 of 4 cells (`‖Δw‖ ≈ 4 × 10⁻⁷`) and exposes it on the 4th (`0.40`). A short smoke test would likely have passed |
| **F11 / §13** | Blocker; §13 was a task | ✅ **Resolved on the original interpreter** — no version drift, cross-stage comparability intact |
| **F1** | Inherited from Stage B's 60-cell campaign | ✅ **Confirmed by the live suite** — both `test_..._drift_off_simplex` tests pass |
| Regression floor | 142 (per `STAGE_B_MASTER_REPORT.md` §22.1) | ⚠️ **155** |
| W1.1 / W2.3a / T-4a / T-4b | Designs to be written | ✅ **Working reference implementation exists** |

**The one lesson worth carrying.** F3's first form was a confident, mechanically detailed argument from a correct formula applied to the wrong quantity — relative error on a count, where the mechanism depends on the *gap between counts*. It read as rigorous and was wrong. F12 was the mirror image: an argument that turned out **right, and understated** — the real error is 160–557%, not a subtle bias. Both point the same way, and it is the same signature `Final_Stage_C.md` §3.2 names for hallucinated citations, in a numeric register: **plausibility is not evidence, in either direction; running it is.**

**Reproducibility.** Two scripts, both re-runnable, both exit non-zero on regression:

| Script | Covers | Checks | Last run |
|---|---|---|---|
| [`scripts/verify_v2_plan_claims.py`](../../scripts/verify_v2_plan_claims.py) | §4 byte counts, §5.2 DP rates, §7.2 compute, F3/F10a/F13 | 29 | ✅ all PASS |
| [`scripts/verify_v2_exactness.py`](../../scripts/verify_v2_exactness.py) | **F12** — V0/V1/V2/V3 gradient equivalence (2 datasets × 2 γ) + the §3 empty-stratum corner | 18 | ✅ all PASS |

The first deliberately avoids importing the `fedgraa` package, so it survives a future TF-import break (§13.4); the second necessarily needs the real model and does not.

**Pass 4 — final audit, 2026-07-27.** Read the document end to end after ~25 edits and repaired **nine defects introduced by the editing itself**: three references to scratchpad scripts that never existed in the repo (F3, F13, §7.2); a contradiction where F10a still called the ζ=0 equivalence "a summation-order identity" that F12 had disproved; two competing "most consequential finding" claims (F1 vs F12); §4.3 still resting its dataset-divergence prediction on the DP finding §5.2 had refuted; a stale "~55 CPU-hours" for T5 against §7.2's corrected 45; a test-churn count of "five" against an actual 11; and four cross-document `§`-references ambiguous between this plan and its sources. Then closed the one substantive gap the audit found: the §3 empty-stratum edge case had been **specified but never tested**, and is now verified 6/6.

**Pass 5 — implementation, 2026-07-28.** At the author's instruction to implement the plan completely except the compute campaign (deferred to Modal per `COMPUTE_INFRASTRUCTURE_2026-07-27.md`), built every W0–W3 mechanism this document specifies, plus W5 (theory) and W6 (dataset-engineering scaffold): `ProjectedFedGRAA`/`FedGRAAv2`/`two_phase_composite_grad` in `fedgraa/aggregation.py`; `SecAggChannel` in `fedgraa/secagg.py`; the Laplace/advanced-composition/RDP accountants in `fedgraa/privacy.py`; `stratum_tags`/`apply_tags`/`local_label_histogram`/`bottom_n_rare_classes`/`partition_by_silo` in `fedgraa/partition.py`; `FederatedTrainerV2`/`zeta_probe` in `fedgraa/trainer.py`; a Modal-ready campaign driver (`scripts/run_v2_frontier_sweep.py`); and six new test files plus additions to two existing ones. **What this pass changed relative to the plan as written:**

| Item | Plan said | What was built |
|---|---|---|
| W2.3a's phase-2 message shape | Two arrays per client (`n_{arm,k}·∇L_{arm,k}`, `n_{n,k}·∇L_{n,k}`), combined server-side | **One** array per client, client-pre-combined using the phase-1-broadcast `n_arm`/`n_n` — equally exact (re-derived and verified: `sum_k [(1-γ)/n_arm · n_{arm,k}∇L_{arm,k} + γ/n_n · n_{n,k}∇L_{n,k}] = grad_P L'` exactly), strictly fewer bytes, and the reason T-12's measured bytes match this doc's `(1+⌈E'/m⌉)×` formula to `<1%` rather than landing ~1.5× over it. A genuine, checked refinement — not a deviation that changes any claim in §4/§10, since the accuracy/exactness properties are identical, only the byte-optimality is sharper. |
| W2.2's "wrap a standard [DP] one" | Implied a third-party accountant package | No DP-accounting package is pinned in `requirements.txt`; rather than add one unvetted mid-project, both accountants (advanced composition, RDP) are implemented directly from their published closed forms (Dwork & Roth 2014 Thm 3.20; Mironov 2017's Laplace-mechanism RDP curve + standard RDP→DP conversion) and cross-checked against this plan's own §5.3 worked example. Disclosed as a deliberate substitution, not silently done. |
| `zeta_probe`'s scope | "computes ∇_P L' twice per eval round ... and logs ζ_t" | Implemented exactly as specified, **plus an explicit scope disclosure** (`FEDGRAA_V2_THEORY.md` §3): comparing both sides fresh at the same point captures `ζ_secagg`/`ζ_dp`'s downstream effect but not `ζ_stale(m)` continuously — an initial version conflated "last-refreshed gradient" against "current oracle," producing a misleadingly large ζ that a comparison against T-4's known-exact baseline caught and corrected, in the same spirit as this document's own F3/F12 findings. |
| §6.2's theory steps 1–5 | To be written | Written up in full in `FEDGRAA_V2_THEORY.md`, each step mapped onto the actual implemented code rather than left abstract; steps 6–7 (deriving `w` from the bound; hardening `β_eff`/the rate constant) remain explicitly out of scope, as the plan itself scopes them. |
| T3/T5 baselines | ~14 other papers' methods, listed by name | **Not implemented.** Faithfully reproducing 14 other papers' methods is its own multi-week reproduction effort (comparable in scope to what Stage A did for FedGRAA itself), cannot be verified as correct without the compute to check it against, and is therefore bundled with "the compute part" in `REMAINING_COMPUTE_WORK.md` rather than attempted un-verified. |
| NF-UQ-NIDS-v2 | Cited as the Option-D dataset | **No download URL, schema, or per-class counts are asserted anywhere in this codebase** for it — this project's own "never guess a URL" / "measure, don't assume" discipline applies here too. What is built: a loader (`load_nf_uq_nids_v2_silos`) that derives everything from whatever real files it is given, and a real-non-IID multi-silo partitioning primitive (`partition_by_silo`) it feeds into — both tested against synthetic fixtures standing in for the real files. |

Every claim in the table above is backed by a currently-passing test named in `REMAINING_COMPUTE_WORK.md` §2. **G3 and G4 — the plan's own two named hard gates — are closed as of this pass.**

**Pass 6 — independent audit, 2026-07-28, same day.** At the author's instruction to "audit and verify what is done," re-ran every independent verification script (`verify_v2_plan_claims.py`, `verify_v2_exactness.py`, the full test suite, both Stage A/B completeness gates and correctness audits, the doc-link checker) fresh rather than trusting Pass 5's own summary, then adversarially probed the two newest, least-exercised modules (`SecAggChannel`, the NF-UQ-NIDS-v2 loader) for edge cases the existing test suite didn't cover. **Found and fixed two real defects, in the same spirit as F3/F12's own discipline of measuring rather than assuming:**

| # | Defect | Found by | Fix |
|---|---|---|---|
| 1 | `SecAggChannel.aggregate()` could be called twice on the same channel and would silently re-return the same total the second time, instead of raising -- `submit()` checked `self._closed`, `aggregate()` set it but never checked it. Not exploitable by any current caller (every production code path constructs one channel per round and calls `aggregate()` exactly once), but inconsistent with the class's own documented "single-use" contract. | Direct adversarial probing of `SecAggChannel` (calling `aggregate()` twice) | `aggregate()` now raises `RuntimeError` if `self._closed`, mirroring `submit()`'s own check. Regression test: `test_calling_aggregate_twice_raises`. |
| 2 | `load_nf_uq_nids_v2_silos` (W6) would silently concatenate two silos whose numeric feature columns had the **same count but different names** (e.g. one silo's 3rd column being `"bytes_out"`, another's being `"duration"`) -- no shape error, but a semantically corrupt pooled matrix that silently aligns unrelated features by position. Worse than the cryptic-error failure mode this module's own docstring worried about. | Constructing two synthetic silos with mismatched-but-same-count schemas and confirming no exception was raised | Added an explicit column-name-equality check across silos before concatenating, raising a `ValueError` naming both mismatched silos and their column sets. Regression test: `test_load_nf_uq_nids_v2_silos_rejects_mismatched_column_schema`. |

Both fixes are additive (new checks that reject previously-silently-accepted input) and neither changes any previously-passing test's expected behavior. Full suite after both fixes: see `REMAINING_COMPUTE_WORK.md` §7 for the exact count. **Nothing else audited (T-4/T-4a/T-4b/T-12/T-18's own numeric claims, the DP accountants' match to the §5.3 worked example, F13's UNSW unrecoverability, the byte-optimal protocol's exactness) turned up a discrepancy** -- all re-verified to the same values Pass 5 reported.

**Pass 7 — loader/layout gap closure + Modal gate verification + mentor-decision resolution, 2026-07-28, same day.** At the author's instruction to "do everything [from a prioritized next-steps list] except the compute on Modal part" (tests on Modal explicitly permitted, the T1/T2/T4 campaign itself explicitly not), closed every remaining item that did not require the compute campaign:

| Item | What was done | Evidence |
|---|---|---|
| NF-UQ-NIDS-v2 loader/layout gap (`docs/audit/nf_uq_nids_v2_provenance_audit.md` §6, found the same day as this pass) | `fedgraa/data.py`'s `load_nf_uq_nids_v2_silos` now auto-detects a single-merged-CSV path (`os.path.isfile(data_dir)`) and streams it in chunks, grouping by the `Dataset` column, instead of only accepting four pre-split subdirectories -- the official release's actual layout. Additive: the original directory layout still works unchanged. | 3 new tests in `tests/test_data.py`; full suite 249 -> 252, all green |
| `scripts/check_stage_c_complete.py` (`REMAINING_COMPUTE_WORK.md` §6, previously the one piece of W4 tooling not yet built) | Written directly against `run_one_cell`'s already-known result-dict schema (no real campaign output needed to know the shape) | Validated end-to-end against one real, manually-run cell: `0/N` on an empty results dir, `N/N` + exit 0 once that cell's JSONs exist |
| G3/G4 re-verification on the actual Modal image (`REMAINING_COMPUTE_WORK.md` §3.2, previously flagged as "cheap insurance" not yet taken) | `scripts/modal_calibration/modal_v2_gate_verification.py` re-ran `tests/test_aggregation_v2.py` + `tests/test_trainer_v2.py` on Modal (confirmed non-AVX512, same container profile as the earlier v1 smoke test) | **34/34 passed, exit code 0** -- `scripts/modal_calibration/modal_v2_gate_result.json` |
| Mentor-conversation decision (`STAGE_DIRECTION_ASSESSMENT.md` §5/§7, the standing blocker this document's own §12.3 pointed at) | Author resolved it directly rather than waiting on the external meeting: implement the full B+D scope (privacy redesign + dataset migration) now, deferring the T3/T5-vs-external-compute trade-off until real T1/T2 numbers are in hand, and **pursuing Option A (few-shot/prototype hybrid) afterward as a sequenced follow-on (`Final_Stage_C.md` §10's Option C)** — not in parallel, not traded away | `STAGE_DIRECTION_ASSESSMENT.md`'s new resolution note (and its own 2026-07-28 scope-correction addenda); `REMAINING_COMPUTE_WORK.md` §4's updated framing and new §3a; `docs/audit/STAGE_C_IMPACT_ON_STAGE_AB_2026-07-27.md` action item 4 marked resolved (and corrected) |

**What this pass explicitly did not do, by the author's own instruction:** launch the T1/T2/T4 compute campaign itself. `scripts/run_v2_frontier_sweep.py` remains exercised only against hand-run single cells (both in Pass 5 and again in this pass, to validate `check_stage_c_complete.py`), never a real tier. G5/G6 remain exactly as `REMAINING_COMPUTE_WORK.md` §2's gate table already stated: blocked on that campaign.

### 14.2 Tooling — disclosed in full, matching `Final_Stage_C.md` §13.2's discipline

**The `FastTrack_Literature_Open`, `alphaXiv` and `Consensus` MCP servers were again not connected in this session.** The deferred-tool registry was searched for them and returned only `WebSearch`, `WebFetch`, `TodoWrite` and `ExitPlanMode`. This pass therefore made **no** attempt to close `Final_Stage_C.md` §13.4's items 1–4; they remain exactly as that document leaves them. Two `WebSearch` calls also returned `unavailable` mid-pass and were not retried, since neither was load-bearing.

**What was fetched, and what it established:**

| Target | Tool | Result |
|---|---|---|
| arXiv:1109.2415 | WebFetch | ✅ **Schmidt, Le Roux & Bach**, *"Convergence Rates of Inexact Proximal-Gradient Methods for Convex Optimization,"* submitted 12 Sep 2011, NeurIPS 2011, DBLP `conf/nips/SchmidtRB11`. Abstract confirmed verbatim. **New to the corpus** (F5, §6.2) |
| arXiv:2408.07152 (FedMADE, abstract) | WebFetch | Title/authors/venue confirmed; **abstract does not mention the auxiliary dataset** — insufficient alone |
| arXiv:2408.07152v1 (FedMADE, full text) | WebFetch | ✅ Three verbatim sentences confirming a **labelled server-side auxiliary dataset of ten samples per class** used to compute the aggregation weights (§1.2) |
| Scooping search: server-validation-free FL-IDS aggregation, 2026 | WebSearch | **Negative result.** Returned only FedMADE, NIDS-FGPA, PerFedHypID — all already catalogued. No new occupant of the v2 angle |

⚠️ **One near-miss worth recording, because it is the corpus's own §3.2 failure mode.** A WebSearch summary asserted the FedMADE auxiliary-dataset fact confidently *before* it was verified. The abstract fetch **did not support it**. Only the full-text fetch did. Had the search summary been taken at face value, §1.2's load-bearing claim would have rested on a tool's paraphrase. **The "no link, no citation" test applied symmetrically, and it is the reason that claim is quotable.**

### 14.3 What is asserted vs. what is verified

- **Verified by execution of the live codebase** (`verify_v2_plan_claims.py`, all PASS): every byte count and frontier multiplier in §4.1; every DP recovery rate in §5.2; F3's determinism and zero-count results; F10a's bit-identical pooled `D_V`; F13's recoverability table; §7.2's compute totals.
- **Verified by execution against the real model** (`verify_v2_exactness.py`, all PASS): **F12 in both directions** — that the naive protocol is wrong (160–557% relative L2) *and* that the §3 two-phase count-weighted protocol is exact (≈2 × 10⁻⁷), on both datasets at two γ, plus the Adam-masking observation.
- **Verified by the project's own test suite:** F1's simplex drift (`tests/test_aggregation.py`, 2 tests); the 155-test regression floor; §13's resolved environment.
- **Verified by reading live source, with file and line range cited inline:** F2, F4, F6, F7, and both of F12's mechanisms (`model.py:166-184`'s `reduce_mean`; `aggregation.py:184`'s global arm selection).
- **Verified by external fetch:** F5's Schmidt et al. citation (arXiv:1109.2415); §1.2's FedMADE quotes (arXiv HTML full text).
- **Still derived, not executed:** the `√(2K)` DP-aggregate-noise formula and the composed `(ε,δ)` figures are analytical and carry the redesign doc's derivation, not a fresh one. The **§8/§12 effort estimates** are planning judgements. The **frontier's *accuracy* axis is entirely unmeasured** — §4 establishes what each point costs in bytes, not what it buys in recall; that is what T1–T3 exist to find out, and nothing in this plan predicts the answer.
- **Judgement, offered as recommendation not decision:** §1.3's eight recommendations, §4's reframing, §7.2's tiering, §10.2's contribution sentence, §12.3's mentor framing. Per the same boundary [`../current/STAGE_DIRECTION_ASSESSMENT.md`](../current/STAGE_DIRECTION_ASSESSMENT.md) §5 observes: **the Stage C angle is the user's and the mentor's decision, not this document's.** This plan assumes Option B+D because `Final_Stage_C.md` §10 recommends it. **✅ Update, 2026-07-28: the author has since selected Option C — "both, sequenced"** (B+D now, A afterward as a follow-on — see `STAGE_DIRECTION_ASSESSMENT.md`'s resolution note), confirming the caveat this line originally flagged: §§3–9 here are B-only and do not transfer to A; Option A needs its own plan, not yet written and not scheduled to be written until B+D's own remaining work is further along (`REMAINING_COMPUTE_WORK.md` §3a tracks that gap).
- **Not verified, carried forward:** everything inherited from the ten prior passes, at those passes' verification level. ~140 entries in `Final_Stage_C.md` §11 carry a prior pass's confidence, not this one's.

### 14.4 Effort estimates

The day/week figures in §8 and §12 are **planning estimates, not measurements** — the only measured rate in this project is Stage B's ~15–20 min per (arm, seed) on CPU, which is what §7.2's compute column is built from. Treat implementation effort as order-of-magnitude and the compute column as reliable.

### 14.5 AI-usage disclosure

This document was compiled by an AI assistant (Claude Opus 5) at the repository author's direction, over **three passes** (§14.1): reading the Stage C literature corpus and three design/theory documents; reading the live `fedgraa/` source; executing the test suite and diagnosing its failure; performing two external verification fetches and one scooping search; drafting this plan; then, on the author's instruction to verify it completely, **executing the partition and configuration code to test the plan's own claims** — which refuted part of it; then, after the author resolved the environment fault, **building a measurement for the plan's central claim** and landing both verifications as re-runnable scripts.

**Every finding in §2 names the file, line range, command output, or fetched source it rests on.** No verdict is asserted from model knowledge. Where a later pass contradicted an earlier one, the contradiction is stated in place rather than silently corrected (F3, the §4.1 byte column) — matching the register `Final_Stage_C.md` §13.5 established for this series. Interpretive readings — the contribution reframing (§1.2, §4), the tiering (§7.2), the risk weightings (§11) — are inferences from the cited evidence, offered for the author and the mentor conversation, not decisions taken.

**Two standing caveats.** No text here should enter a manuscript without the author's own read of the underlying sources. And the frontier's **accuracy axis is unmeasured** — this plan establishes what each point on it costs, never what it buys; any sentence asserting that exactness helps rare-class recall is unsupported until T1–T3 run.

---

*Compiled 2026-07-27. Plans, does not supersede, the Stage C literature work in [`Final_Stage_C.md`](Final_Stage_C.md). Supersedes `privacy_compliant_fedgraa_redesign.md` §13 and §15 only; all other sections of that document stand and remain the authority for the taxonomies, DP arithmetic, and threat-model actors this plan builds on.*
