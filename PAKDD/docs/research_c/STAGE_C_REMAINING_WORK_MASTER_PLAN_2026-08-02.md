# FedGRAA-v2 — everything left before this is a submittable PAKDD paper (2026-08-02)

> **Purpose.** A single, complete, execution-ready inventory of every remaining
> piece of work on FedGRAA-v2's Stage C, written before any of it is executed
> — the same discipline
> [`STAGE_C_FEDPGO_FAITHFUL_PLAN_2026-07-31.md`](STAGE_C_FEDPGO_FAITHFUL_PLAN_2026-07-31.md)
> used. It supersedes nothing; it corrects
> [`REMAINING_COMPUTE_WORK.md`](REMAINING_COMPUTE_WORK.md)'s now-stale
> compute-status table in place (§0) and extends its scope to the one tier
> that document only ever gestured at — **T7 (adversarial robustness)** —
> plus everything else that has to happen before Stage C is a finished paper,
> not just a finished compute campaign. Nothing below is implemented yet
> unless explicitly marked done in §0. This document is the map; execution is
> tracked turn-by-turn against it, the same way the FedPGO plan's own §11
> became its execution record.

**Trigger.** The author asked directly: *"Make a plan .md for all the
remaining work of fedgraa v2, i.e T7, etc"* — with an explicit standing
instruction to do the complete thing, not a partial one, and to search before
building rather than fabricate. This document is that plan. Three things in
it were verified today, live, rather than assumed from memory (§7.2) —
consistent with that instruction and with this project's own "measure, don't
assume" discipline running through every prior Stage C document.

---

## 0. Ground truth, corrected (§ REMAINING_COMPUTE_WORK.md is stale)

[`REMAINING_COMPUTE_WORK.md`](REMAINING_COMPUTE_WORK.md) was compiled
2026-07-28 and says, in its §3/§4 tables, that **T3/T5 baseline code does not
exist**. That was true on 2026-07-28. It has been false since 2026-07-29.
Ground truth as of this document (verified by `git log`, `ls fedgraa/`, and a
live `check_stage_c_complete.py` run today):

| Claim | `REMAINING_COMPUTE_WORK.md` (2026-07-28) said | Actually true (2026-08-02) |
|---|---|---|
| T3 baselines (CELM, SpectralFed, CoAst, FedTruth, RAGA, FLTrust) | "Baseline code does not exist" | ✅ **Implemented** — `fedgraa/baselines_t3_contrib.py` (CELM, SpectralFed, CoAst), `fedgraa/baselines_t3_trust.py` (FLTrust, RAGA, FedTruth). All 6 arms complete, 10 seeds × 2 datasets |
| T5 baselines (FedMADE, SBFedNIDS, FedAPA, FedAWA, DA-PFL, FedAWARE, AdaAggRL, FedPGO) | "7 of 8 lack code" | ✅ **Implemented** — `fedgraa/baselines_t5_nids.py` (FedMADE, SBFedNIDS, FedAPA), `fedgraa/baselines_t5_adaptive.py` (FedAWA, DA-PFL, FedAWARE, AdaAggRL), `fedgraa/baselines_t5_fedpgo.py` (FedPGO, faithful since `25611ce`). All 8 arms complete |
| Overall gate | "G5 ⏳ not run" | ✅ **`check_stage_c_complete.py --tier T1 T2 T3 T4 T5 T6 --t4-m 3` → 730/730, doc-links green** (re-run today, 2026-08-02) |
| Full suite | 252 passed (2026-07-28 count) | ✅ **453 passed** (re-run today) |
| T7 | "Attack harness does not exist — not attempted, tagged 'Nice'" | ❌ **Still true, unchanged.** Zero adversarial/poisoning code anywhere in `fedgraa/` (verified by grep today — every "attack" hit in the codebase means *network-intrusion attack class*, the IDS domain noun, not an adversarial-ML attack on training) |
| Manuscript | "No manuscript exists anywhere" (`STAGE_DIRECTION_ASSESSMENT.md`, 2026-07-25) | ⚠️ **Corrected 2026-08-02 — this document's own claim was wrong.** `docs/paper/PAKDD_Final_Version_Submit_v3.md` (+ its rendered `.pdf`) and `docs/paper/draft1.md` are a full v1-only submission draft that already existed; `docs/paper/FedGRAA_Simulated_Peer_Review.md` is a full 5-agent simulated review of it (2026-07-25, same day as the "no manuscript exists" claim above), recommending **Reject — Resubmit Encouraged** on exactly three structural grounds (missing simplex projection, a convexity assumption contradicting the paper's own non-convex architecture, and single-run/unseeded/missing-baseline results). The filesystem search this document's §0 originally cited apparently missed `docs/paper/`. **All seven required revisions (R1-R7) the review names are now independently resolved by this project's own subsequent Stage B/C work** — simplex projection (`ProjectedFedGRAA`), the corrected non-convex stationarity theorem (`FEDGRAA_V2_THEORY.md`), 10-seed statistical significance testing (§3 of this document), and the 19-baseline comparison (§4) among them. The corrected resubmission draft is `docs/paper/FedGRAA_v2_PAKDD_Submission.md`. |

**Action taken on this correction:** recorded here as a dated note, per this
project's own §13.5-style convention (dated corrections retained in place,
not silently rewritten). A one-line dated correction should also be appended
to `REMAINING_COMPUTE_WORK.md` §1 and §4 pointing here — tracked as work
item **W-DOC1** in §8 below, not done silently as part of writing this plan.

**Bottom line entering this document:** the *compute campaign* for the
privacy-compliant redesign (B) and the dataset migration (D) — everything
`Final_Stage_C.md` §10's Option C called "B+D now" — is **finished**: 730/730
cells, 14/14 external baselines faithfully reproduced (FedPGO to
primary-source fidelity; the other 13 to the standard this project's own
disclosure convention holds every baseline to), full suite green. What is
left is **not** more FedGRAA-v2 mechanism code. It is: one more compute tier
(T7) that is a **genuinely new sub-project** (an attack/defense harness, not
a variant of the existing aggregator), two analysis layers the existing data
hasn't been run through yet (§2–§3), and the **entire manuscript** (§6) —
Stage 2 (WRITE) of the ARS pipeline has not started.

---

## 1. Complete inventory, prioritized

| # | Item | Status | Blocks drafting? | Est. effort |
|---|---|---|---|---|
| 1 | **T7 — adversarial robustness** (attack suite + defense baselines + campaign) | Not started, 0% | **Yes** — `Final_Stage_C.md` §9.1 item 3 names this a *premise conflict that must be argued, not merely disclosed*; omitting it is not "incomplete," it's citably dishonest given the paper's own literature review already flagged the conflict | Largest single item — comparable to implementing 2–3 more T3/T5-sized baselines plus novel attack design |
| 2 | **Statistical significance layer for the v2 frontier** (paired Wilcoxon + Holm–Bonferroni vs. all 14 baselines) | Infrastructure exists (`fedgraa/stats.py`) but wired only into Stage B's *v1* ablation, not the v2 frontier | Yes — item 8 of the A\*/A checklist names this explicitly, and "FedGRAA-v2 wins" is not a defensible sentence without it | Small–medium (extend `build_v2_frontier_table.py`, no new mechanism) |
| 3 | **Classic-FL-baseline coverage decision** (FedAvg/FedProx/FedOpt exist in code but aren't v2-frontier arms; SCAFFOLD/FedNova don't exist at all) | Open decision, not yet made | Advisory — see §4 for why this may not gate drafting | Small if skipped-with-rationale; medium if wired in; large if SCAFFOLD/FedNova built from scratch |
| 4 | **Overhead-reporting write-up** (the ≈6× exact-vs-approximate narrative against CELM/SpectralFed) | Data exists (`bytes_sent` logged per cell); no analysis script or written narrative | Yes for the manuscript's own overhead paragraph (item 7) | Small (one script + one paragraph) |
| 5 | **Duplication/scooping re-check (G6)** | Last run inside `Final_Stage_C.md`'s own pass (§1.2); `REMAINING_COMPUTE_WORK.md` §6 step 3 requires one more, immediately pre-draft | Yes — explicit pipeline gate before writing the contribution sentence | Trivial (one tool run + a paragraph recording the result) |
| 6 | **The manuscript itself** (ARS Stage 2 WRITE, all sections) | 0% — no file exists | N/A, this *is* drafting | Largest recurring item; the ARS pipeline (Stage 2 → 2.5 → 3 → 4 → 3′ → 4′ → 4.5 → 5 → 6) that was interrupted last turn |
| 7 | **Documentation debt** (dated corrections to stale docs; §0 above is the template) | Partially done (this document) | No | Small |
| 8 | **Camera-ready / submission packaging** (reproducibility statement, code-release cleanup, PAKDD formatting/page limits, artifact appendix) | 0% | No (post-acceptance-shaped, but PAKDD's own submission format requirements gate the *first* submission too) | Medium |
| 9 | Option A (few-shot/prototype hybrid) | Deliberately deferred | **Not FedGRAA-v2's remaining work** — see §9 | Explicitly out of scope here |

Sequencing (§10) puts items 2–5 before 6, and 1 in parallel with 2–5 since it
has no dependency on them; 7 happens continuously; 8 happens once 6 exists.

---

## 2. T7 — Adversarial robustness evaluation

### 2.1 Why this is not optional

This project's own literature review — not an external reviewer, its own
prior work — already wrote the sentence that makes skipping T7 indefensible:

> "FedGRAA is therefore not merely *undefended* against poisoning — it is a
> mechanism that **structurally amplifies exactly the signal a poisoning
> defense suppresses**, and an adversary who makes an update look like a
> rare-class contribution gets up-weighted by construction."
> — `STAGE_C_LITERATURE_MASTER.md` line 398 (verbatim); `Final_Stage_C.md`
> line 415 makes the identical finding in slightly tighter wording ("not
> merely *undefended* — it structurally amplifies...")

A reviewer who has read AdaAggRL, MESAS, or CrowdGuard (all already in this
project's own bibliography) will ask the up-weighting-as-attack-surface
question in the first review round, unprompted. Not having an answer is a
**structural gap in the contribution**, not a missing nice-to-have — this is
why `Final_Stage_C.md` §9.1 places it at position 3 of a 12-item A\*/A
checklist, ahead of dataset scale (item 5) and tied in urgency with the
convergence theorem (item 1, already substantially addressed by
`FEDGRAA_V2_THEORY.md`'s W5 workstream).

### 2.2 Decisions (author-approved scope; mirrors the FedPGO plan's own §1)

| # | Decision | Choice | Rationale |
|---|---|---|---|
| D1 | Which attacks | **Four**, in ascending order of novelty: (a) label-flipping — the field-standard baseline attack, trivial to implement, needed as a sanity floor; (b) model-poisoning ("pill"-style parameter-space perturbation, Guo et al. AAAI 2026) — a stronger generic attack; (c) **the mechanism-specific attack**: an adversary that fabricates a gradient update whose AGO-style signature (high orthogonal-component norm, low alignment with the global direction) mimics a genuine rare-class contribution to farm PGF-style up-weighting — this is FedGRAA-v2's own attack surface, not reproduced from any paper, and is the one no baseline defense in the literature was designed against; (d) AutoAdapt-style adaptive attacker (Krauß et al., NDSS 2024) as the ceiling — an attacker that optimizes against whatever defense is active, used only against the strongest defense arm, not the full grid |
| D2 | Which defenses to compare against | **Two external reproductions + one internal reuse**: SecFedNIDS (Zhao et al., FGCS 2022 — public PyTorch code, same datasets as this project) and WeiDetect (Sameera et al., *JISA* 2025 — full text openly available, arXiv:2504.04367) as external baselines; **AdaAggRL is reused, not reimplemented** — it already exists as a T5 novelty-defence arm (`fedgraa/baselines_t5_adaptive.py`) and Final_Stage_C.md itself frames it as "the adversarial mirror-image of FedGRAA," so it doubles as a T7 defense arm for free once the attack-injection harness exists |
| D3 | FedGRAA-v2's own posture | **No new defense is added to FedGRAA-v2 itself.** T7 evaluates the *existing* mechanism (γ-tunable up-weighting) under attack, honestly, including the negative result if the attack succeeds. Per `Final_Stage_C.md` item 11, the manuscript's Limitations section states whether γ trades detection performance against poisoning exposure — that sentence can only be written after this campaign runs. A defended variant is explicitly **not** in scope; that would be Option A/B-shaped new-mechanism work this document does not authorize (see §9) |
| D4 | Evaluation harness | **SoK/FLPoison-aligned** (arXiv:2502.03801 — full text open, confirmed today, §2.3): attacker-fraction and attack-strength as the two primary knobs, task accuracy + rare-class recall under attack (this project's own existing metric) as the effectiveness axis, and attack-success-rate as the attack-side axis, rather than inventing a bespoke protocol |
| D5 | Datasets | Reuse the existing two (CIC-IDS-2017, UNSW-NB15) — **not** SecFedNIDS's or WeiDetect's own datasets (CIC-Darknet2020, CSE-CIC-IDS2018). Consistent with every other T3/T5 baseline in this project: reproduce the *mechanism* faithfully, evaluate on *this project's own* harness, disclose the resulting number mismatch rather than chase paper-reported figures (the FedPGO plan's own §10 "honest verification bar" pattern) |

### 2.3 Primary-source status (verified today, 2026-08-02 — not from memory)

Per this project's own hard-won rule (the FedPGO proxy had to be entirely
rebuilt once the primary text arrived, `STAGE_C_FEDPGO_FAITHFUL_PLAN_2026-07-31.md`
§2) — **no attack or defense below gets implemented from an abstract-level
summary.** Full-text acquisition is execution step 1, not step 3. Status,
checked live via WebFetch today rather than assumed:

| Source | Status | Detail |
|---|---|---|
| **WeiDetect** (arXiv:2504.04367) | ✅ **Full text openly obtainable** — PDF and HTML both live at arxiv.org, confirmed today. Abstract-level mechanism (confirmed): server-side validation-score generation per local model, then a **Weibull distribution fit over those scores** used to identify and remove malicious models — a two-phase server-side filter, not a client-side defense. Exact fitting/thresholding procedure requires the full-text read (not yet done — this is execution work, §2.6 step 1) |
| **SoK: Benchmarking Poisoning Attacks and Defenses in FL / FLPoison** (arXiv:2502.03801) | ✅ **Full text openly obtainable**, CC-BY 4.0, TeX source available, confirmed today. Benchmarks 15 attacks × 17 defenses; code on GitHub (exact URL needs the full-text pull, not blocked) |
| **SecFedNIDS** (Zhao et al., *FGCS* 2022) | ✅ **Public reference code confirmed live** at `github.com/zhangzhao156/SecFedNIDS` today — `main_poisoned_data_det.py` (data-level filtering) and `main_poisoned_model_det.py` (model-level detection), PyTorch, evaluated on UNSW-NB15 (one of this project's own two datasets already) — checked today, not assumed from the literature review's secondary description. This is a **stronger** starting position than FedPGO ever had: a public reference implementation to cross-check against, not just a primary text |
| **ARES 2023** (attack parameterization, DOI 10.1145/3600160.3605090) | ⚠️ **Paywalled — confirmed today.** `dl.acm.org` returns HTTP 403, the same ACM DL access pattern this project has hit repeatedly for other papers (`Final_Stage_C.md` §9.0 item (i)). No preprint mirror checked yet (not attempted this pass — the search-effort budget for this document was spent confirming the three sources above, not exhausting ARES's access channels). **Consequence:** the mechanism-specific attack (D1c) and the label-flipping/model-poisoning attacks (D1a/b) do not strictly need ARES's own parameterization scheme to be implemented — SoK's own harness (D4) already supplies attacker-fraction/attack-strength knobs — so ARES's inaccessibility does **not** block T7. If ARES becomes available later, its specific parameterization can be folded in as a documented judgment-call refinement, the same way this project treats every other partially-accessible source |
| **AdaAggRL** | ✅ **Already implemented**, `fedgraa/baselines_t5_adaptive.py` — no new acquisition needed, reused per D2 |
| **Poisoning with a Pill** (Guo et al., AAAI 2026, arXiv:2407.15389) | Not checked this pass — arXiv, so presumptively obtainable; checking it is execution step 1 alongside WeiDetect/SoK, not deferred further |
| **AutoAdapt** (Krauß et al., NDSS 2024) | Not checked this pass; needed only for D1d (the adaptive-ceiling attack), which is the lowest-priority of the four attacks — check when D1a–c are implemented and green |

**What this means for sequencing:** unlike FedPGO (one paywalled paper, one
implementation), T7 has **three obtainable primary sources plus one public
reference codebase** and only **one** paywalled source whose absence doesn't
block the work. There is no reason to defer T7 pending access-logistics the
way the FedPGO proxy had to wait — the honest blocker for T7 is
**implementation effort**, not **access**, and that distinction should be
stated plainly rather than let the two blur the way `STAGE_C_COMPUTE_SESSION_UPDATE_2026-07-30.md`
§5–§6 records happening once already for FedPGO.

### 2.4 Module architecture (mirrors the T3/T5/T6 pattern already established)

New files, following the existing `fedgraa/baselines_t{3,5}_*.py` +
`scripts/modal_frontier_campaign_t6.py` naming convention exactly:

1. **`fedgraa/attacks.py`** — the attack layer, orthogonal to the trainer/
   aggregator split. Exposes attacker client wrappers that intercept a
   client's trained update before it reaches the aggregator:
   - `label_flip_attack(X, y, flip_map, fraction)` — D1a
   - `model_poison_attack(params, scale, target_layers)` — D1b, "pill"-style
     once Guo et al. is read; a documented judgment call if not
   - `rare_class_signature_attack(tau_k, g_G, alpha, eps)` — D1c, **the
     mechanism-specific attack**: constructs a delta whose AGO decomposition
     (reusing `fedgraa.baselines_t5_fedpgo.fedpgo_decompose`'s own math,
     since FedGRAA-v2's up-weighting signal is not identical to PGF's but
     the orthogonality concept is the shared attack surface — needs a
     precise mapping to FedGRAA-v2's own `two_phase_composite_grad`
     weighting function in `fedgraa/aggregation.py`, which is genuine design
     work, not a reproduction, and must be written up with the same
     judgment-call disclosure discipline as every JC1–JC8 table in this
     project) — high orthogonal norm, engineered to score as rare-class-like
     under FedGRAA-v2's own weighting rule
   - `adaptive_attack(defense_fn, ...)` — D1d, wraps any defense and
     optimizes against it (AutoAdapt-style), applied only to the strongest
     defense arm
2. **`fedgraa/baselines_t7_adversarial.py`** — the two external defense
   reproductions (`SecFedNIDSDetector`, `WeiDetectFilter`), each with the
   same module-docstring judgment-call-disclosure convention every T3/T5
   baseline in this codebase already uses, and each citing the exact
   primary-source line numbers once §2.3's full-text reads happen
3. **`scripts/run_v2_frontier_sweep.py`** — extend `enumerate_cells()` with a
   `T7` tier: `(dataset, attack, defense, attacker_fraction, seed)` cells,
   reusing the existing `arms:` filter parameter (already built for FedPGO's
   rerun, `25611ce`) so a `--tier T7 --arm secfednids` smoke run is possible
   without touching T1–T6's enumeration paths
4. **`scripts/modal_frontier_campaign_t7.py`** — its own file/`modal.App`,
   following T6's own precedent of never bolting a second dataset/workload
   onto an existing App (`PRE_COMPUTE_AUDIT_2026-07-28.md`'s "second new
   defect" box is the named reason this pattern exists)

### 2.5 Grid size and cost estimate

Per `REMAINING_COMPUTE_WORK.md` §3's own T7 row (80 runs, ~23 core-hours) —
re-derived here rather than only cited, since that document's baseline-code
estimates just proved stale (§0):

- 2 datasets × 4 attacks × 3 defense-arms (SecFedNIDS, WeiDetect, AdaAggRL)
  × {2 attacker-fractions} × 5 seeds ≈ **240** cells if run at full
  cross-product — larger than the 80-run estimate in the older document,
  because that estimate predates D1's four-attack decision.
- **Recommended reduced grid** (author decision needed, flagged here rather
  than decided unilaterally): full cross-product for the two datasets ×
  4 attacks × 3 defenses at **one** attacker-fraction (the field-standard
  30%) × 5 seeds = 2×4×3×1×5 = **120** cells, plus a **trend-line addendum
  for D1c only**: the mechanism-specific attack run against FedGRAA-v2
  directly (undefended, i.e. no defense cross-product — the point of this
  addendum is measuring the vulnerability's own shape, not re-testing the
  three defenses) at a second attacker-fraction, for the 2 datasets, with 5
  *additional* seeds per (dataset, fraction) to bring D1c's own seed count
  from the base grid's 5 up to 10 — 2 datasets × 1 fraction (the new one) ×
  5 additional seeds = **20** cells. Total 120 + 20 = **140** cells, closer
  to the original 80-run estimate's order of magnitude while still covering
  every decision in §2.2. (Arithmetic double-checked here rather than left
  to round to a stated total, per this document's own §7.2 discipline.)
- At the measured per-cell cost from `COMPUTE_INFRASTRUCTURE_2026-07-27.md`
  §5–6 (UNSW-NB15: 681.5 s/cell; CIC-IDS-2017: 1,235.9 s/cell; ≈$0.12/core-hour
  blended), 140 cells ≈ **37 core-hours ≈ $4.5**, well inside the "afternoon,
  not a multi-day commitment" finding that document's headline result
  already established for the full T1–T7 stack.

### 2.6 Execution order (this is the part that must happen next, in this order)

1. **Full-text reads**: WeiDetect (arXiv:2504.04367), SoK/FLPoison
   (arXiv:2502.03801), Guo et al. "Poisoning with a Pill" (arXiv:2407.15389),
   SecFedNIDS's actual source (`main_poisoned_data_det.py` /
   `main_poisoned_model_det.py`) — before writing a line of `fedgraa/attacks.py`
   or `baselines_t7_adversarial.py`. Attempt ARES 2023 via one non-ACM
   channel (Sci-Hub is out of scope for this project on ethical/legal
   grounds — try the authors' own institutional page or a conference
   proceedings mirror instead); proceed without it if inaccessible, per §2.3.
2. **Judgment-call table**, written *before* implementation, in the same
   format as the FedPGO plan's §4 — every place the primary sources are
   ambiguous or where FedGRAA-v2's own architecture doesn't map 1:1 onto
   SecFedNIDS's or WeiDetect's original setting gets a disclosed JC entry.
3. **`fedgraa/attacks.py`** + tests (attack functions are pure, testable
   without touching the trainer — unit-test each in isolation: label-flip
   changes only the targeted class fraction, model-poison scales only the
   targeted layers, the mechanism-specific attack's constructed delta scores
   above a threshold on FedGRAA-v2's own weighting function by construction).
4. **`fedgraa/baselines_t7_adversarial.py`** + tests, faithful to the
   full-text reads from step 1, same JC-disclosure docstring convention.
5. **Wire attacker injection into `run_one_cell`** — an attacker client's
   update is computed normally then passed through the active
   `attacks.py` function before aggregation; defense arms wrap the
   aggregation step with the detector/filter.
6. **`enumerate_cells()` T7 tier + `modal_frontier_campaign_t7.py`**, smoke
   test (2 cells, 2 rounds) before a full `--force` run, exactly the T5/
   FedPGO rerun pattern (`STAGE_C_FEDPGO_FAITHFUL_PLAN_2026-07-31.md` §7).
7. **Full suite green**, **`check_stage_c_complete.py` extended to accept
   `--tier T7`** (or a parallel `check_stage_c_t7_complete.py` if the schema
   genuinely differs enough — decide during implementation, not here).
8. **Results table + one paragraph** answering `Final_Stage_C.md` item 11's
   question directly: does γ trade detection performance against poisoning
   exposure, backed by the actual T7 numbers, not asserted in the abstract.

### 2.7 Test spec (mirrors the FedPGO plan's §5, ~15 tests)

1. Label-flip attack changes exactly the targeted fraction of labels;
   untargeted clients unaffected.
2. Model-poison attack scales only the targeted parameter arrays; shared vs.
   personalized split (where applicable) respected.
3. Mechanism-specific attack: constructed delta's orthogonal-component norm
   exceeds a benign client's by construction, on a synthetic fixture.
4. Mechanism-specific attack, applied through the real aggregation pipeline,
   demonstrably increases the attacker's aggregation weight vs. an
   unmodified delta of the same underlying gradient (the actual claim the
   manuscript will make — must be pinned by a regression test, not asserted).
5. SecFedNIDS detector: on a synthetic fixture with a known-poisoned model
   injected, flags it; zero false positives on an all-benign fixture.
6. WeiDetect filter: Weibull fit converges on a benign-only fixture; correctly
   down-weights/removes an injected outlier score.
7. AdaAggRL (reused from T5) correctly receives attacker-fraction metadata
   without requiring changes to its existing T5 implementation (an
   integration test, not a new unit test of AdaAggRL itself).
8. Adaptive attacker (D1d): optimizes its perturbation against a mocked
   defense score function; converges to a lower detection score than a
   non-adaptive attack of the same base type, on a synthetic fixture.
9. End-to-end: one full T7 cell (2 rounds, real cached data) runs without
   NaN collapse, produces valid accuracy/attack-success-rate outputs.
10. Determinism: same seed → same attacker selection, same attack outcome.
11–15. Validation/error-path tests for each new public function's parameter
   bounds (attacker_fraction ∈ [0,1], defense threshold parameters,
   etc.), mirroring every other baseline module's validation-test pattern
   in this codebase.

### 2.8 Gate

`check_stage_c_complete.py --tier T1 T2 T3 T4 T5 T6 T7` → expect
**730 + (grid size from §2.5) / same total**, doc-links green, full suite
green with the new T7 test file included. Only after this gate passes does
§6's manuscript drafting get to write the adversarial-robustness paragraph
with real numbers instead of a promissory "future work" placeholder.

---

## 3. Statistical significance layer for the v2 frontier

**Gap, precisely stated.** `fedgraa/stats.py` has the machinery (used by
Stage B's `build_ablation_table.py`/`render_ablation.py` for the *v1*
ablation). `scripts/build_v2_frontier_table.py` computes **bootstrap 95% CI**
per arm (§ "verified today," line `ci_95 = list(bootstrap_ci(...))`) but
never runs a **paired hypothesis test** of FedGRAA-v2 against any of the 14
external baselines, and never applies a **multiple-comparison correction**
across the 14-baseline family. A frontier plot with error bars is not the
same claim as "FedGRAA-v2 significantly outperforms baseline X" — the
manuscript cannot respectably make the second claim without this layer,
and `Final_Stage_C.md` item 8 says so explicitly ("≥10 seeds, paired
Wilcoxon, Holm–Bonferroni across the baseline grid").

**Plan.**
1. Extend `scripts/build_v2_frontier_table.py` (or a new
   `scripts/build_v2_significance_table.py`, decide during implementation
   based on how invasive the change is to the existing script's output
   schema) to compute, per dataset and per FedGRAA-v2 arm (`v2_m1..v2_m5`)
   against each of the 14 external baselines: a paired Wilcoxon signed-rank
   test over the 10 matched seeds' rare-class-recall values, then a
   Holm–Bonferroni correction across the resulting 14×2×5 = 140 comparisons
   (or scoped per-dataset if that's the more defensible unit — a judgment
   call to disclose, not silently choose).
2. Tests: the existing `fedgraa/stats.py` functions are presumably already
   unit-tested (Stage B); confirm, and add a regression test that the new
   script's output matches a hand-computed Wilcoxon statistic on a fixed
   synthetic pair.
3. Output: one more table (`significance_table.md`), consumed directly by
   the manuscript's Results section.

**Not blocked on anything** — the underlying seed data (10 seeds × 19 arms
× 2 datasets) already exists from the completed T1–T6 campaign. This is the
single cheapest, highest-value item in this entire document: no new compute,
no new primary-source reading, pure analysis code over data already on disk.

---

## 4. Classic-FL-baseline coverage — a decision, not a default

`Final_Stage_C.md` item 2's priority list is: *"FedAvg, FedProx, SCAFFOLD,
FedNova, FedOpt + FedPGO (top priority), ..."* — the classic optimizers are
listed **first**, ahead of the 14 external NIDS-specific baselines that are
now fully implemented. As things stand:

- `FedAvg` **was actually run and reported** against v1 FedGRAA in Stage B's
  own ablation (`build_ablation_table.py`'s `COMPARISON_ARMS = ["literal",
  "projection_alone", "eta_alone", "fedavg", "static"]` — confirmed by
  reading the script directly).
- `FedProx` and `FedOpt` are a **weaker claim than "exist and were compared"
  — corrected here after checking, not left at the more flattering framing
  a memory-only pass would have used.** `FedProxTrainer`/`FedOptServer`
  classes exist in `fedgraa/baselines.py` and `fedgraa/experiment.py`'s
  `run_experiment()` can dispatch to them (`method == "fedprox"`/`"fedopt"`),
  but a repo-wide search today (`grep` across every `scripts/*.py`, plus a
  filename search for any `fedprox`/`fedopt` result JSON) found **zero**
  evidence either was ever actually executed or reported — `COMPARISON_ARMS`
  above does not include them, and no result artifact of either exists
  anywhere in the repository. They are **dispatchable, unexecuted code**,
  not a completed comparison. Wiring them into the v2 frontier is therefore
  not "zero new mechanism code, just more cells" (as this document
  originally, incorrectly, implied before this audit pass) — it also
  requires **first verifying these v1-era classes are even compatible with
  the v2 privacy-compliant harness** (`FederatedTrainerV2`,
  `two_phase_composite_grad`, SecAgg) at all, since neither has ever been
  run through it, and adapting them if not.
- `SCAFFOLD` and `FedNova` **do not exist anywhere** in this repository
  (confirmed by a case-insensitive repo-wide grep today; the only matches
  were the unrelated English word "scaffolding" in two docstrings/comments).

**This document does not decide the resolution — it names the decision
explicitly so it isn't silently defaulted either way**, consistent with this
project's own standing practice (e.g. `STAGE_C_FEDPGO_FAITHFUL_PLAN_2026-07-31.md`
§1's decision table, author-approved before execution):

| Option | What it costs | What it buys |
|---|---|---|
| **Skip, with a stated rationale** | Zero — write one sentence in Limitations: "classic non-personalized optimizers (FedAvg/FedProx/SCAFFOLD/FedNova/FedOpt) were evaluated against FedGRAA v1 in Stage B (cite); the v2 frontier compares against the *personalization-aware and NIDS-specific* competitor set (T3/T5's 14 baselines) precisely because those are FedGRAA-v2's actual positioning, per `Final_Stage_C.md`'s own §8.2 differentiation argument" | Nothing extra; relies on the Stage B numbers still being defensible context |
| **Wire in FedAvg** (already run once, v1) **+ verify/adapt FedProx/FedOpt** (code exists, never executed — §4's correction above) into the v2 frontier grid as 3 more arms | Medium, not small — FedAvg is genuinely cheap (proven-compatible code, just needs the v2 harness's SecAgg/two-phase path wired in); FedProx/FedOpt carry real unknown risk since neither has ever run end-to-end and may need adaptation before their 3 × 2 × 10 = 60 cells (~few core-hours) can even execute | A reviewer-proof "yes, we compared against the field-standard optimizers too, on the same privacy-compliant harness" row |
| **Build SCAFFOLD + FedNova from scratch too** | Large — two more from-scratch reproductions, each comparable in scope to one T3 baseline | Completeness against the literature review's own priority-ordered list in full |

**Recommendation for the author to confirm, not a decision made here:** the
middle option, with its cost stated honestly per the correction above —
FedAvg is cheap to wire in, FedProx/FedOpt need a compatibility check first
and may need adaptation, so "wire in the 3 that exist" is medium effort, not
small. SCAFFOLD/FedNova remain diminishing-return work given the paper's
positioning is explicitly against personalization-aware/NIDS-specific
methods, not generic FL optimizers, and item 2's own list treats them as one
undifferentiated bullet rather than a hard requirement the way item 3 (T7)
is a hard requirement.

---

## 5. Overhead-reporting write-up

**Gap.** `run_one_cell` already logs `bytes_sent` per cell (used by T-12's
byte-exactness gate and referenced throughout `REMAINING_COMPUTE_WORK.md`).
`Final_Stage_C.md` item 7 wants this turned into the specific comparison:
*"report the ≈6× [communication overhead] against CELM/SpectralFed's
near-zero-cost approximation, making 'is exact worth 6×?' an experiment
rather than a caveat."* No script currently produces this comparison table.

**Plan.** One small script, `scripts/build_overhead_table.py`: per dataset,
`bytes_sent` for FedGRAA-v2 (`v2_m1..v2_m5`) vs. CELM and SpectralFed (the
two T3 "near-zero-cost approximation" baselines), expressed as a ratio, with
the accompanying rare-class-recall delta from the already-built frontier
table so the "is it worth it" question has both axes on one row. No new
compute, no new mechanism — pure derived-metric reporting from data already
on disk, same category as §3.

---

## 6. Duplication/scooping re-check (G6)

`REMAINING_COMPUTE_WORK.md` §6 step 3 requires this **immediately before**
drafting the contribution sentence, not once, historically, in the
literature-review pass. `Final_Stage_C.md` §1.2 already ran this multiple
times through Stage C's literature phase and found nothing closer than
FedPGO — but that was as of 2026-07-27/31, before this pass's own T7 design
work; a same-day, one-more-time re-check costs one FastTrack/Consensus query
and one paragraph recording the (expected-null) result, and is the cheapest
possible insurance against drafting a contribution sentence that a paper
published in the interim quietly invalidates. **Do this the day drafting
actually starts** (§6 below), not today — a check run now would just have to
be re-run again once T7's numbers land and the actual draft begins, and
would be stale by then anyway.

---

## 7. The manuscript (ARS Stage 2 WRITE and onward)

> ⚠️ **Dated correction, 2026-08-02 (see §0's manuscript row).** This
> section was written assuming a from-zero draft. That premise was wrong: a
> full v1-only submission draft (`docs/paper/PAKDD_Final_Version_Submit_v3.md`)
> already existed, already went through a full simulated peer review
> (`docs/paper/FedGRAA_Simulated_Peer_Review.md`, 2026-07-25), and was
> rejected on exactly the three structural grounds this project's own
> subsequent Stage B/C work happens to have independently fixed (simplex
> projection, the non-convex convergence theorem, and multi-seed
> statistically-tested baselines). What follows is executed as a
> **corrected resubmission** of that draft
> (`docs/paper/FedGRAA_v2_PAKDD_Submission.md`), not a new paper — the
> section dependencies below are otherwise unchanged and still govern
> drafting order.

This is the item the previous turn's `academic-pipeline` dispatch was
already mid-launch on when this planning request interrupted it. It resumes
here, now correctly scoped to depend on §2–§5's outputs rather than jumping
straight to Results with an incomplete evidence base:

**Sections and their dependencies:**

| Section | Depends on | Status |
|---|---|---|
| Abstract, Introduction | Nothing new — synthesizable now from `Final_Stage_C.md`/`FEDGRAA_V2_MASTER_PLAN.md` | Ready to draft |
| Related Work | `Final_Stage_C.md` §8.2's citation list, §14's FedPGO resolution | **Ready to draft now** — this was last turn's scoped task and nothing in this document changes it |
| Methods (v1 critique + v2 redesign + theory) | `FEDGRAA_V2_MASTER_PLAN.md`, `FEDGRAA_V2_THEORY.md` (W5, the convergence theorem — item 1 of the A\*/A checklist, already substantially written) | Ready to draft now |
| Experimental Setup | T1–T6 campaign (done), T7 campaign (§2, not done) | Partially ready — the T1–T6 half can draft now, T7's protocol section needs §2 to land first |
| Results — main frontier | `frontier_table.md`/`.json` (done), §3's significance layer (not done), §5's overhead table (not done) | **Blocked on §3 and §5** — reporting frontier numbers without the significance layer is exactly the kind of unsupported-claim gap this project's own integrity discipline (ARS Stage 2.5) would catch anyway, so doing §3/§5 first avoids a guaranteed Stage 2.5 bounce |
| Results — adversarial robustness | §2 (T7), entirely | **Blocked on §2** |
| Discussion / Limitations | Items 11 (γ/poisoning trade-off — needs §2's numbers) and 12 (bibliography hygiene — verify no FedSND/CIC-IDS2023 residue survives into the reference list) | Blocked on §2 for item 11; item 12 is a drafting-time checklist, not a blocker |
| Conclusion, References | Everything else | Last |

**Recommended draft order therefore:** Related Work + Methods first (fully
unblocked, and exactly last turn's original scope) → §3/§5 (cheap, unblocks
the main Results section) → main Results → §2/T7 in parallel with main
Results drafting → adversarial Results + Limitations once T7 lands →
Abstract/Intro/Conclusion last (they're easiest to write accurately once the
rest exists) → G6 re-check (§6) the day final drafting of the contribution
sentence happens → ARS Stage 2.5 integrity gate → Stage 3 review → revision
loop → finalize.

This is a resumption of the `academic-pipeline` dispatch from the previous
turn, not a new decision — the same skill invocation applies once §3/§5 are
done, with its scope widened to the full paper rather than Related
Work/Results alone, since those two sections turned out to depend on
analysis layers (§3, §5) and a campaign (§2) that didn't exist yet.

---

## 8. Documentation debt

| ID | Item | Where |
|---|---|---|
| W-DOC1 | Dated correction appended to `REMAINING_COMPUTE_WORK.md` §1/§4 recording that T3/T5 are done (this document's §0 is the source of truth to summarize from) | `REMAINING_COMPUTE_WORK.md` |
| W-DOC2 | Same correction propagated to `FEDGRAA_V2_MASTER_PLAN.md`'s own "what remains" framing (§41's implementation-update box) if it still says T3/T5 baselines are open | `FEDGRAA_V2_MASTER_PLAN.md` |
| W-DOC3 | `STAGE_DIRECTION_ASSESSMENT.md`'s "no manuscript exists anywhere" line — still true today, so no correction needed yet, but flag it as the thing that becomes false once §6 lands | `docs/current/STAGE_DIRECTION_ASSESSMENT.md` |
| W-DOC4 | This document itself gets a "Final status — execution record" section appended (mirroring the FedPGO plan's §11) once §2–§7 actually execute, not written speculatively now | This document |

Low priority relative to §1–§7; do opportunistically, not as a blocking
gate on any of them.

---

## 9. Explicitly out of scope: Option A

**Not included above, on purpose.** The few-shot/prototype-guided hybrid
(`Final_Stage_C.md` §10's Option A) is, by this project's own prior and
repeated author decision (`STAGE_DIRECTION_ASSESSMENT.md`'s resolution note,
`REMAINING_COMPUTE_WORK.md` §3a), a **second, independent novelty
contribution** — not a variant, extension, or remaining task of FedGRAA-v2
itself — sequenced **after** B+D as a deliberate follow-on, with its own
from-zero design/implementation/baseline/campaign arc that
`REMAINING_COMPUTE_WORK.md` §3a already scopes in detail. Re-including it
here would contradict a decision the author already made explicitly three
times (`REMAINING_COMPUTE_WORK.md`'s own §8 disclosure records the two false
starts and the final correction). If the author wants Option A folded into
this plan's active scope, that is a new scope decision to make explicitly,
not an inference this document draws from "do the whole thing."

---

## 10. Sequencing / dependency graph

```
§3 (significance layer) ──┐
§5 (overhead table)    ───┼──► §6 Results (main frontier) ──┐
§4 (baseline decision) ───┘                                  │
                                                               ├──► §6 Discussion/Limitations ──► §6 Abstract/Intro/Conclusion ──► §6 G6 re-check ──► ARS Stage 2.5 ──► Stage 3 review ──► ... ──► Stage 6
§2 (T7: sources → attacks → defenses → campaign → gate) ─────┘
     (fully parallel with §3/§4/§5 — no shared dependency)

§6 Related Work + Methods ──► (already unblocked, draft first, feeds ARS Stage 2.5 same as everything else)

§7 (doc debt) ── continuous, no gate
§8 (Option A) ── not scheduled
```

§2 (T7) is the long pole — it is the only item with a multi-step
primary-source-then-implement-then-campaign arc. §3/§4/§5 can all complete
in the time it takes to do §2.6 steps 1–2 (the literature reads).

---

## 11. Effort & cost summary

| Item | Compute cost | Engineering effort (relative) |
|---|---|---|
| §2 T7 | ≈37 core-hours ≈ $4.5 (§2.5) | **Largest** — new module, 4 attacks, 2 external defense reproductions, 1 reused, ~15 new tests, new Modal driver |
| §3 significance layer | $0 (reads existing results) | Small |
| §4 classic-baseline wiring (if chosen) | ~few core-hours, ~$1 | Small for FedAvg; medium for FedProx/FedOpt (compatibility with the v2 harness unverified — §4); large if SCAFFOLD/FedNova built from scratch |
| §5 overhead table | $0 | Small |
| §6 manuscript | $0 (ARS pipeline token cost, per that skill's own budget-transparency step) | Large, multi-session, the actual deliverable |
| §8 doc debt | $0 | Trivial |

Total incremental compute for everything code-shaped in this document:
**under $10 and well under an hour of Modal wall-clock**, once §2.6's
literature-read step is done. The bottleneck for the whole remaining-work
set is engineering/writing time, not compute budget or access-logistics —
worth stating plainly since every other Stage C planning document up to now
has had compute or access as its named bottleneck, and this one doesn't.

---

## 12. Definition of done

Stage C (all of it, T1–T7 plus the manuscript) is complete when, and only
when, every one of the following is independently checkable, not asserted:

1. `check_stage_c_complete.py --tier T1 T2 T3 T4 T5 T6 T7` passes, doc-links
   green, at whatever the final T7 grid size turns out to be (§2.5).
2. Full test suite green, including every new T7 test (§2.7) and the
   significance-layer regression test (§3).
3. `significance_table.md` and `overhead_table.md` exist and are consumed by
   the manuscript's Results section, not left as unused artifacts.
4. A manuscript file exists (`.md` and/or `.tex`, per §6's ARS Stage 5
   finalize step), has passed ARS Stage 2.5 (integrity) and Stage 4.5
   (final integrity) with zero outstanding issues, and has been through at
   least one full Stage 3 review cycle.
5. Every item in `Final_Stage_C.md` §9.1's 12-item A\*/A checklist is either
   ✅ closed with a citable artifact, or explicitly and honestly stated as an
   Acknowledged Limitation in the manuscript itself — never silently dropped.
6. This document's §8 doc-debt items are closed.

---

## 13. AI disclosure

This document was compiled by an AI assistant (Claude Sonnet 5) at the
repository author's direct request ("make a plan .md for all the remaining
work... T7, etc"), continuing the same session that verified the FedPGO
faithful-reimplementation commits and began (then paused, on the author's
own follow-up question) the ARS `academic-pipeline` dispatch for Related
Work/Results drafting. Three claims in this document were verified live
today via WebFetch/GitHub rather than recalled from the existing literature
review: WeiDetect's and the SoK benchmark's full-text accessibility
(arXiv, confirmed open), SecFedNIDS's public reference code (confirmed real
and non-empty), and ARES 2023's ACM DL paywall (confirmed 403, consistent
with this project's own prior pattern for ACM DL access). Every cost and
core-hour figure in §2.5/§11 is derived from `COMPUTE_INFRASTRUCTURE_2026-07-27.md`'s
own measured per-cell numbers, not a fresh estimate. No code was written and
no compute was run in producing this document — it is a plan, per the
author's explicit request, not an execution record; §2.6/§10 define the
execution this document authorizes but does not itself perform.

---

## 14. Final status — execution record (2026-08-02, same day, second session)

Per this document's own §8 (W-DOC4) and the FedPGO plan's §11 precedent, this
section is appended once §2–§7 actually execute, not written speculatively.
Author instruction for this pass: implement the complete remaining-work
inventory, not a partial slice — recorded here as the standing rationale for
why every item below was executed the same session the plan was read, rather
than staged across multiple turns.

**§3 (significance layer) — done.** `scripts/build_v2_significance_table.py`
+ `tests/test_build_v2_significance_table.py` (9 tests). Corrected the
plan's own 140-comparison estimate to 112 (v2 has 4 frontier arms, not 5 —
`v2_m4` was never part of the T1/T2 campaign design). Real result:
**111/112 comparisons significant** at Holm-corrected α=0.05 (sole exception:
FedAWARE vs. `v2_m1` on CIC-IDS-2017).

**§5 (overhead table) — done.** `scripts/build_overhead_table.py` +
`tests/test_build_overhead_table.py` (7 tests). Confirms the frontier's own
byte formula exactly: **6.00×** at `m=1`, dropping to **2.00×** at `m=5`,
against CELM/SpectralFed.

**§4 (classic-FL baselines) — done, author chose the largest-scope option**
("build all five from scratch, including SCAFFOLD/FedNova") over an
`AskUserQuestion` prompt, not defaulted. FedAvg wired via a thin adapter;
FedProx/FedOpt wired via existing v1-era classes (no incompatibility found);
SCAFFOLD and FedNova built from scratch, faithful to their primary sources
(Karimireddy et al. ICML 2020; Wang et al. NeurIPS 2020), each with a
disclosed judgment-call table (JC-TC-1..7) in
`fedgraa/baselines_classic.py`'s module docstring. 24 new tests, all passing,
including a regression test confirming FedNova provably reduces to FedAvg
when local-step counts are uniform (the sanity check that the normalization
math is implemented correctly, not merely running). **Real campaign result
(100 cells, 10 seeds × 2 datasets × 5 arms, Modal):** every classic optimizer
scores ≈0% rare-class recall on both datasets except SCAFFOLD (≈30%) — all
of them well below every FedGRAA-v2 refresh interval (65–93%).

**§2 (T7, adversarial robustness) — done, in full**, including the one item
the original plan flagged as "genuinely new design work": the
mechanism-specific attack (D1c), derived directly from FedGRAA-v2's own
printed weight-update gradient (`_grad_wrt_weights`) rather than the
literature review's informal "orthogonal-component norm" language — a more
precise and more faithful design than the plan's own first pass proposed,
recorded as a judgment-call correction (JC-T7-13) rather than silently
substituted. Primary sources for WeiDetect, SoK/FLPoison, and "Poisoning
with a Pill" were read in full via a background research agent before any
code was written, per this project's own standing rule; SecFedNIDS's public
reference code was read directly. 19 judgment calls disclosed in
`docs/research_c/STAGE_C_T7_JUDGMENT_CALLS_2026-08-02.md`, written before
implementation, including three scope-narrowing corrections to the plan's
own §2.5 grid arithmetic (JC-T7-16, JC-T7-18): the final T7 grid is **90
cells**, not the plan's original 140, because (a) AdaAggRL's own
unmodified T5 implementation cannot be hooked for parameter-space attacks
without editing code the plan says must stay unchanged, so it is evaluated
under label-flipping only, and (b) the mechanism-specific attack (D1c) is
calibrated specifically to FedGRAA-v2's own weighting formula and is
therefore evaluated only against undefended FedGRAA-v2, not the SecFedNIDS/
WeiDetect defense cross-product, where it would exercise nothing
method-specific. New modules: `fedgraa/attacks.py` (4 attacks, 19 tests),
`fedgraa/baselines_t7_adversarial.py` (SecFedNIDS + WeiDetect, including a
from-scratch Stochastic Outlier Selection implementation rather than a new
`pyod` dependency, 19 tests), a new `_fused_importance_step` in
`fedgraa/model.py`, D1c wired directly into `FederatedTrainerV2` (6 tests),
full `enumerate_cells`/`run_one_cell` T7 dispatch in
`scripts/run_v2_frontier_sweep.py`, `scripts/modal_frontier_campaign_t7.py`
(its own file/App, per T6's established precedent), and `--tier T7` support
in `check_stage_c_complete.py`. **Real campaign result (90 cells, 5 seeds ×
2 datasets, Modal):** the mechanism-specific attack up-weights its
attacker's aggregation share to 2.0–3.1× uniform, confirming the literature
review's prediction directly rather than only in theory; SecFedNIDS's
reproduction is a weak defense on these datasets (53–75% miss rate);
WeiDetect is stronger but specifically degraded by the adaptive attacker;
and — the most consequential finding — **every defended cell's rare-class
recall collapses to ≈0%**, meaning the two external defenses stop the
poisoning attack only by discarding FedGRAA-v2's own rare-class-sensitive
signal, not by preserving it.

**§6 (manuscript) — done**, as a **corrected resubmission**, not a new
paper (§0/§7's dated correction: a full v1-only submission draft and its
own simulated peer-review rejection already existed in `docs/paper/`,
missed by this document's own original filesystem search).
`docs/paper/FedGRAA_v2_PAKDD_Submission.md` resolves every one of the
review's required revisions (R1–R7) using the real work above: the missing
simplex projection, the corrected non-convex/ζ-inexact stationarity
theorem, the 19-baseline statistically-tested comparison, the resolved
FedAvg-formula ambiguity, and — beyond the review's own asks — the full
privacy-compliant redesign (§5) and adversarial-robustness evaluation (§7)
neither the original draft nor its review addressed at all. G6 (duplication
re-check) run the same day drafting started, per this document's own §6:
expected-null result, no scooping match found.

**§7 (documentation debt) — done.** W-DOC1/W-DOC2 (dated corrections to
`REMAINING_COMPUTE_WORK.md` and `FEDGRAA_V2_MASTER_PLAN.md`), W-DOC4 (this
section). W-DOC3 remains correctly not-yet-applicable (superseded by this
section's own manuscript-exists correction instead).

**Full suite:** 551 tests passing (up from 453 at this document's own
starting point), zero regressions, checked three times across the session
(after TC, after T7 wiring, and after the final campaign pull). Stage C
completeness gate: **830/830 cells** across every tier (T1–T7, TC),
doc-links green.

**What remains, honestly.** Option A (the few-shot/prototype hybrid,
explicitly out of scope per §9) is unchanged and still deferred. Camera-
ready packaging (§1 item 8) — reproducibility statement, PAKDD-specific
formatting/page-limit compliance, artifact-release cleanup — has not been
started; it is post-acceptance-shaped and was not part of this pass's
scope. The manuscript's reference list (§ References) carries the
highest-priority citations in full and abbreviates the remainder with a
pointer to the verified full catalog (`Final_Stage_C.md` §11); expanding
every citation inline is mechanical remaining work, not a content gap.

---

## 15. Documentation sweep + reference-list completion (2026-08-02, same day, third pass)

Per the author's follow-up instruction ("do everything except option A,
make thorough changes to all documentation everywhere accordingly"), this
pass closed the two items §14 flagged as not-yet-done and swept every other
document this session's own work made stale:

- **Manuscript reference list completed in full** — all 34 inline citation
  keys used in `docs/paper/FedGRAA_v2_PAKDD_Submission.md` now resolve to a
  complete, verified entry (previously 24 of 34, with the remainder pointed
  at the external catalog rather than inlined).
- **Reproducibility statement + submission-compliance section added** to the
  manuscript: exact seed/config provenance, the completeness/correctness
  commands that independently re-verify every reported number, code/
  artifact availability, and an honest (not asserted) note on the remaining
  LaTeX-conversion/page-limit-verification step before camera-ready.
- **Every document this session's own findings made stale, corrected in
  place, dated, not silently rewritten**: `README.md` (status table, test
  count, Key Files, the Stage C gate commands, the "one open question"
  pointer), `MENTOR_WALKTHROUGH.md`, `docs/current/publication_roadmap.md`,
  `docs/current/STAGE_DIRECTION_ASSESSMENT.md` (all three still carried
  "no manuscript exists" language from before this session's discovery of
  `docs/paper/`'s existing draft), and the three `docs/paper/` files
  themselves (`PAKDD_Final_Version_Submit_v3.md`, `draft1.md`,
  `FedGRAA_Simulated_Peer_Review.md`) each now carry a dated pointer to the
  corrected resubmission at the top, unedited otherwise, per this project's
  own "retained as historical record" convention.
- **Re-verified before commit:** `check_doc_links.py` (green),
  `python -m compileall -q fedgraa scripts tests` (clean — this is what
  the `doc-links-and-syntax` CI job runs), and the full 551-test suite,
  matching exactly what `.github/workflows/docs-check.yml`'s `stage-gates`
  job runs on every push to `main`.

**What remains, unchanged from §14:** Option A (explicitly, repeatedly out
of scope per the author's own instruction this pass and every prior one),
and the mechanical LaTeX/page-limit camera-ready conversion, now explicitly
disclosed in the manuscript's own Submission Compliance section rather than
left implicit.
