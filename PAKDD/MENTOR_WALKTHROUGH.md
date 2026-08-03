# FedGRAA / PAKDD — Mentor Walkthrough

**Purpose of this document.** A single, stand-alone briefing you can read from top
to bottom before (or during) a meeting with your mentor/professor, or hand to
them directly. It explains what the project is, what has actually been done,
what the evidence is, where the work is honestly weak, and — most
importantly — the specific decisions that need a mentor conversation before
the next phase can start. Every number below is traceable to a file in this
repo; the file map is in the appendix.

**Compiled:** 2026-07-26, against `HEAD` (`9ee076a`), main branch, working
tree clean.

> **✅ UPDATE (2026-07-28).** Since this document was written, [`docs/research_c/FEDGRAA_V2_MASTER_PLAN.md`](docs/research_c/FEDGRAA_V2_MASTER_PLAN.md) was compiled (2026-07-27) and its entire non-compute implementation (W0–W3, W5 theory, W6 dataset-engineering scaffold) has now been built, tested, and landed (2026-07-28) — every mechanism the plan specifies for FedGRAA-v2 exists in `fedgraa/` and is green, including both of the plan's own named hard gates (T-4/T-4a/T-4b exactness, T-12 byte accounting). §8's "Stage C is at 0%" below is therefore **no longer accurate for the engineering** — see [`docs/research_c/REMAINING_COMPUTE_WORK.md`](docs/research_c/REMAINING_COMPUTE_WORK.md) for the exact status. **What this update does not do:** it does not mean the mentor conversation below has happened, or that "Option B+D" is confirmed as the chosen angle by anyone other than the literature review's own recommendation ([`Final_Stage_C.md`](docs/research_c/Final_Stage_C.md) §10, "high confidence"). Building the Option-B+D implementation ahead of that conversation was a deliberate bet on that recommendation, consistent with the master plan's own framing that the implementation work "needs no decision" to start — but the four questions in §8 (provenance, angle **confirmation**, compute, authorship) are unchanged and still need an actual conversation before a manuscript is drafted. If the mentor selects a different angle, this implementation does not transfer.
>
> **✅ UPDATE (2026-07-28, later the same day, corrected again the same day) — the angle question above is now settled as follows.** The author decided directly, without the mentor meeting: `Final_Stage_C.md` §10's **Option C — "both, sequenced."** Build B (privacy-compliant redesign, described throughout this document) and D (NF-UQ-NIDS-v2 migration) now — this is exactly the "Option B+D" scope this document already describes, unaffected by this note — and pursue A (the few-shot/prototype hybrid) **afterward, as a follow-on**, not simultaneously and not starting this pass. A is committed for that later phase but has no design, code, or tests yet — see [`docs/research_c/REMAINING_COMPUTE_WORK.md`](docs/research_c/REMAINING_COMPUTE_WORK.md) §3a. Every "Option B+D" reference below (§8, the status table) is therefore accurate as this document's **current, active** scope; A is tracked separately in §3a as deferred, not concurrent, work.
>
> **✅ UPDATE (2026-08-02) — Stage C is now complete, B+D+T7, and a manuscript exists.** Everything the two updates above still called open — the compute campaign, baseline reproduction, the T7 adversarial-robustness tier this document doesn't otherwise mention, and the manuscript itself — is now done: 830/830 compute cells (T1–T7, TC) across both datasets, 10-seed statistical significance testing (111/112 comparisons significant), and a corrected, resubmission-ready manuscript, [`docs/paper/FedGRAA_v2_PAKDD_Submission.md`](docs/paper/FedGRAA_v2_PAKDD_Submission.md), that resolves every finding of the simulated peer-review rejection of the original draft this document's §1/§8 material predates. Full account: [`docs/research_c/STAGE_C_REMAINING_WORK_MASTER_PLAN_2026-08-02.md`](docs/research_c/STAGE_C_REMAINING_WORK_MASTER_PLAN_2026-08-02.md) §0/§7/§14. **This does not resolve the mentor-conversation questions below on its own** — provenance, authorship, and Option A's later scheduling are still real questions for that conversation, per the author's own standing instruction that building ahead of the conversation was "a deliberate bet," not a substitute for it. What it does mean: the "compute" and "baseline reproduction" items in §8's four open questions no longer describe unstarted work — they describe work to review, not work to schedule.

---

## 0. The 60-second version

You are reproducing a published-looking federated-learning paper —
*"Federated Gradient-Refined Adaptive Averaging for Intrusion Detection
Systems"* (FedGRAA) — from scratch, on real intrusion-detection data. The
paper claims its aggregation scheme protects rare attack classes (e.g. 89%
accuracy on minority classes). **Your faithful, line-for-line reimplementation
shows the algorithm as printed does not converge**: a missing step (projecting
the aggregation weights back onto the probability simplex) causes the weights
to become invalid in the very first training round, on both datasets, every
time — collapsing rare-class accuracy to ~0% instead of the claimed ~89%. You
found the exact mechanism, proved (with a corrected, weaker theorem) why the
paper's own convergence proof cannot cover this, and validated a minimal fix
across a statistically rigorous 10-seed experiment.

That's Stage A (reproduction) and Stage B (formal critique + ablation), and
both are **complete, tested, and independently audited** — including an
audit you ran on your own work yesterday that found real bugs and got them
fixed the same day. Stage C (the novel contribution needed for an A/A*-tier
submission) is **at 0%**, not because of missing compute or missing ideas,
but because it's genuinely blocked on decisions only your mentor can make —
laid out in §8.

| | Status |
|---|---|
| Stage A — faithful reproduction | ✅ Complete |
| Stage B — formal critique + statistical ablation | ✅ Complete |
| Stage C — novel A/A\*-tier extension (Option B+D now, FedGRAA-v2; Option A, few-shot/prototype hybrid, sequenced as a follow-on per Option C) | B+D: ✅ **Code/tests complete** (2026-07-28) · ⏳ compute campaign, baseline reproduction, and the mentor conversation itself still open. A: ⏳ **deliberately deferred** — committed as a later, sequenced phase, no design/code/tests yet — see `docs/research_c/REMAINING_COMPUTE_WORK.md` §2–§7 (B/D, active) and §3a (A, deferred) |
| Tests | **311/311 passing** as of 2026-07-28 (252 after the Stage C implementation pass, +40 regression guards from `docs/audit/PRE_COMPUTE_AUDIT_2026-07-28.md`'s remediation, +19 from that same audit's B6 fix and T6 wiring; this doc's own 155 above was the pre-Stage-C-implementation baseline); **559/559 current** now that Stage C's compute campaign, T7, and classic-FL baselines have landed — see `docs/current/STAGE_C_MASTER_REPORT.md` §12 |
| Completeness gates (structure) | 2/2 green |
| Correctness audits (values re-derived from raw data) | 2/2 green |
| Manuscript | Does not exist yet — everything below is analysis docs, not a paper draft |

---

## 1. How this project is organized, and why

This lives in a personal research repo (`Research/`) with one standing rule
for every project in it: **reimplement the target paper faithfully first,
measure honestly, and if something doesn't reproduce, diagnose *why* instead
of quietly patching it until it works.** FedGRAA (`PAKDD/`) is the flagship
and, so far, only project.

Two kinds of automated checks exist for every claim of "done," and this
matters for how much you should trust anything in this document:

- **Completeness gates** (`check_stage_a_complete.py`, `check_stage_b_complete.py`)
  — is every artifact the docs claim actually present and correctly shaped?
- **Correctness audits** (`audit_stage_a.py`, `audit_stage_b.py`) — is every
  reported number *independently re-derived from the raw persisted JSON*,
  rather than just read back from the document that asserts it?

This distinction exists because of two real incidents: a documentation
reorganization silently broke 34 relative links, and an earlier report
claimed a "byte-identical" reproduction that turned out to be an overclaim.
Both are now impossible to miss silently — they're gated. This habit of
"trust nothing, re-derive it" is the single most defensible thing about the
project, and §6 shows it catching real problems in the project's own work as
recently as yesterday.

---

## 2. The paper being reproduced: FedGRAA

**Problem it addresses.** In federated intrusion detection, each client
typically holds one attack type (plus benign traffic). Standard FedAvg-style
aggregation — weighting each client's contribution by how much data it has —
systematically drowns out clients with few samples, which are exactly the
rare attack types (SQL Injection, Heartbleed, Infiltration, ...) a real IDS
most needs to catch. FedGRAA proposes to *learn* the aggregation weights `w`
each round by gradient descent on a composite objective that explicitly
trades off common-class performance against rare-class performance, via a
personalization coefficient `γ ∈ [0,1]`.

**Architecture:** `Dense(200, tanh) → Dense(100, tanh) → Dense(50, tanh) →
Dense(n_classes, softmax)`. 78 input features on CIC-IDS-2017 (15 classes),
39 on UNSW-NB15 (10 classes).

**Algorithm 1 (federated training loop)** — standard: each of `K` clients
does local SGD/Adam updates for `E` epochs, then the server aggregates.

**Algorithm 2 (the FedGRAA weight learner)** — the actual contribution. Each
round, before aggregating: run `E′` epochs of gradient descent on the
aggregation-weight vector `w` itself, against a composite validation loss
`L' = (1−γ)·L_a + γ·L_u` where `L_a` is common-class loss and `L_u` is
rare-class loss. As printed in the paper, Algorithm 2 has **exactly 8
lines**. There is no line 9. **No projection, clipping, or renormalization
step of any kind exists anywhere in the algorithm as published** — verified
by direct grep of the reimplementation and by reading the paper's own
algorithm box character-by-character. This one missing line turns out to be
the whole story — see §3.

**Datasets:** CIC-IDS-2017 and UNSW-NB15, both standard, public,
intrusion-detection benchmarks. Both were forensically audited against their
raw CSVs (byte-level column/label checks, class-count cross-checks against
canonical published distributions) specifically to rule out "the data is
corrupted or synthetic" as an explanation for anything found later. **Verdict:
both are genuine, complete, correctly labeled.** This matters because it
means the central finding below is an algorithmic property, not a data
artifact.

---

## 3. Stage A — the faithful reproduction (✅ complete)

**Goal** (self-imposed, gated by `check_stage_a_complete.py`): every
table/figure in the original paper gets a corresponding persisted,
reproduced-or-honestly-diverging counterpart. Every place the reimplementation
had to make a judgment call because the paper was ambiguous is logged
explicitly (9 such "strict reproduction" decisions — e.g., forcing round-0
accuracy to display as 0.00% "to match the paper's table" even though a
randomly-initialized network doesn't actually score 0%, a choice that's
disclosed in code but easy to miss if you only read the output tables).

### 3.1 The central finding, walked through step by step

This is the headline result of the entire project so far, so it's worth
spelling out mechanically rather than just quoting the conclusion.

1. **The gap in the algorithm.** Algorithm 2 updates `w` with gradient
   descent and stops. `w` is supposed to represent a probability distribution
   over clients (each `w_k ≥ 0`, `Σw_k = 1` — the "probability simplex").
   Nothing in the 8-line algorithm keeps it there. A gradient step can — and,
   empirically, always does — push some component of `w` negative or push the
   sum away from 1.

2. **Measured, not inferred.** Instrumenting every round of training and
   printing `w` directly: **on both datasets, at every tested `γ ∈ {0, 0.5,
   1.0}`, `w` leaves the simplex in round 1.** In the later, larger Stage B
   campaign (10 seeds × 3 γ × 2 datasets = 60 unprojected runs, 120 counting
   both affected arms), this held **120 times out of 120, zero exceptions** —
   `Σw` ranged as far as `[−2.62, +2.77]` on CIC-IDS-2017, with individual
   weight components as negative as `−1.157`. A negative "weight" on a client
   means the aggregation is *subtracting* that client's model from the
   global one — not a subtle numerical wobble, a structurally broken update.

3. **The consequence.** The paper's own Lemma 1 asserts the composite loss
   `L'` is non-increasing round over round — the entire basis of its claimed
   convergence guarantee. With `w` off the simplex, that's violated
   immediately. Measured composite-loss blowup: **2.7×–193× within the
   3-seed Stage-A regime**, and as extreme as **1.86 million times** the
   initial loss in the fuller 10-seed Stage B campaign (heavy-tailed — see §4).

4. **The downstream effect — what you'd actually see if you just ran the
   paper's own code.** Rare-class ("last-n") accuracy, the paper's headline
   metric, collapses to **~0%** at the paper's own 200-round horizon, on both
   datasets, under the algorithm exactly as printed. Meanwhile the naive
   "Static Reparameterization" baseline (fixed inverse-size weighting, no
   learning at all) gets **84–92% rare-class accuracy for free** — the
   learned mechanism FedGRAA is proposing actively underperforms doing
   nothing clever.

5. **Confirming the mechanism, not just the symptom.** A diagnostic-only
   ablation adds a Euclidean projection of `w` back onto the simplex after
   each gradient step, plus lowering the weight-learner's own learning rate
   `η′` from the paper's 0.1 to 0.01 (`E′` unchanged). This **is not a tweak
   to make numbers look better** — it directly targets the mechanism
   identified in step 1. Result: the loss blowup collapses to **exactly
   1.0×** on CIC-IDS-2017 (i.e., genuinely non-increasing, as Lemma 1
   promised) and **≤1.5×** on UNSW-NB15, and rare-class accuracy recovers to
   **72–94%** (mean across seeds) on CIC and **0–75%** (γ-dependent) on
   UNSW. This fix is deliberately kept diagnostic-only and never silently
   merged into the strict-reproduction default — the project's standing
   "no silent patching" rule, so the paper's algorithm-as-published and the
   corrected version are never conflated.

### 3.2 Headline numbers — paper claim vs. what actually happens

γ=1, n=3 (3 protected rare clients), 200-round literal run vs. 60-round
diagnostic fix (mean ± std across seeds):

| | Paper's claim | Literal reproduction (algorithm exactly as printed) | Diagnostic fix (projection + η′=0.01) |
|---|---|---|---|
| CIC-IDS-2017, rare-class acc. | **88.88%** | **0.00%** | **94.44% ± 3.93%** |
| UNSW-NB15, rare-class acc. | **21.09%** | **33.33%** | **74.79% ± 1.99%** |

**Scale of the reproduction itself:** 155 automated tests (unit, gradient,
end-to-end), a full Keras/TensorFlow reimplementation (not the earlier,
retired hand-written NumPy version), both datasets' γ-sweeps and n-sweeps
run and persisted as raw JSON (not just plotted and discarded), and every one
of 17 places where the paper's own text is internally ambiguous or
contradictory logged explicitly (§5).

---

## 4. Stage B — turning the diagnosis into a citable result (✅ complete)

Stage A found and confirmed the mechanism. Stage B's job was to turn "we
diagnosed a bug" into something that reads as a rigorous paper section on its
own: formal theory plus statistics that would survive peer review, not just
a 3-seed spot check.

**What was built:**
- A **corrected convergence proposition** (`fedgraa_convergence_defect.md`
  §4): the paper's original proof requires the composite loss to be convex
  and β-smooth — an assumption the paper's own tanh-MLP violates on its face
  (§5 below has more on this). The corrected version proves feasibility (the
  projected update stays on the simplex) and stationarity, which is honestly
  *weaker* than the paper's original unconditional convergence claim — the
  project does not overclaim its own fix either.
- A **statistics module** (`fedgraa/stats.py`) implementing bootstrap
  confidence intervals, paired t-test, Wilcoxon signed-rank, Cohen's *d*, and
  Holm–Bonferroni multiple-comparison correction, with disclosed and tested
  behavior for degenerate edge cases (e.g. zero-variance data).
- A **10-seed, 2×2 factorial ablation** (projection on/off × η′ paper-value/
  reduced-value), crossed with 3 γ-values and 2 datasets = **240 measured
  cells**, run and persisted (not summarized from memory).

### 4.1 The three findings that survived statistical testing

**Finding 1 — the feasibility result is unconditional, exactly as the
corrected theory predicts.** Across all 120 unprojected cells:
`first_simplex_violation_round = 1`, zero exceptions. Across all 120
projected cells: never violated, weight-sum error within float32 round-off
(`≤1.2×10⁻⁷`). A theory that predicts a structural guarantee and then
matches it with zero exceptions across 240 runs is a genuinely strong result.

**Finding 2 — which lever fixes the accuracy is dataset-dependent, and this
overturned the project's own pre-registered hypothesis.** Before running the
full campaign, the working hypothesis (from a smaller 3-seed preview) was
"both the projection and the smaller η′ are always independently necessary."
The full 10-seed run **overturned that**: on CIC-IDS-2017, the η′ reduction
alone carries almost all of the rare-class recovery; on UNSW-NB15, the
pattern inverts and projection alone carries it. Critically — and this is
the kind of check that separates careful work from a paper hunting for a
clean story — reporting *only* rare-class accuracy would have made the
"η′-alone" arm on CIC look statistically equivalent to the full fix. Checking
**all-class accuracy too** reveals it isn't: that arm buys rare-class parity
by starving every common class down to 15.2% all-class accuracy (vs. 44.3%
for the full fix). **No single-lever arm is safe on both metrics at every
γ; only the combined fix never fails catastrophically on either metric, on
either dataset.** That's the actual, defensible claim — not "our fix is
best," but "our fix is the only one that isn't sometimes disastrous."

**Finding 3 — on UNSW, the literal algorithm's failure mode is
unpredictability, not uniform collapse.** On CIC-IDS-2017 the literal
algorithm's rare-class accuracy is an exact, seed-independent 0.0% at two of
three γ-values (10/10 seeds). On UNSW-NB15 at γ=0.5, the *same* unmodified
algorithm produces rare-class accuracy anywhere from 0.4% to 44.0% depending
purely on the random seed. For an IDS you'd actually deploy, an algorithm
whose behavior swings wildly with the RNG seed is arguably a worse failure
mode than one that reliably fails — and this pattern was only visible once
the seed count went from 3 to 10.

### 4.2 Headline numbers (mean rare-class accuracy, 95% bootstrap CI, N=10 seeds, γ=1)

| Method | CIC-IDS-2017 | UNSW-NB15 |
|---|---|---|
| FedAvg (baseline) | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| Static Reparam. (baseline, no learning) | 84.4% [77.2, 91.4] | 48.3% [40.8, 56.0] |
| **Literal FedGRAA (paper exactly as published)** | **10.0% [0.0, 30.0]** | **27.6% [15.0, 39.9]** |
| Projection-only fix | 51.0% [24.2, 77.1] | 67.2% [59.6, 73.6] |
| η′-reduction-only fix | 88.9% [68.9, 100.0] | 36.4% [33.7, 39.8] |
| **Both fixes together (recommended)** | **89.3% [83.6, 94.6]** | **69.2% [63.4, 73.8]** |

---

## 5. The paper itself was independently critiqued — and it doesn't hold up well

Separately from "does the code reproduce," the paper's own text was read
line-by-line, purely as a document (no reference to the reimplementation),
looking for internal contradictions. **17 distinct issues** were catalogued,
tagged by type:

- **Direct contradictions** — the algorithm boxes/equations describe plain
  gradient descent (`θ ← θ − η∇L`), while the prose says the optimizer is
  Adam. Adam's update rule is categorically different (it maintains
  first/second moment estimates); these cannot both be literally describing
  the same computation, and the paper never resolves the tension. This
  happens **twice** — once for the client updates, once for the FedGRAA
  weight-learner update itself. Separately, two different equations are both
  labeled "FedAvg" in the same section despite computing different things
  whenever client dataset sizes differ (they always do).
- **Undefined terms** — the paper's own headline metric, "last-n accuracy,"
  is used throughout §5 and never mathematically defined anywhere in the
  paper.
- **A convexity assumption the paper's own model violates** — the
  convergence proof (Appendix A.1) requires the composite loss to be convex
  and β-smooth. §5 of the same paper describes the actual network as a
  multi-layer tanh-MLP with softmax output — not convex. The proof's premise
  contradicts the paper's own architecture section.
- **Statistically suspicious reported numbers** — several headline accuracy
  values in the paper's own summary table are simple repeating-decimal
  fractions (88.88%, 66.66%, 77.77%, 33.33% — i.e. 8/9, 2/3, 7/9, 1/3), and
  several figures show implausibly flat accuracy curves across long stretches
  of training rounds. This is the kind of pattern much more consistent with
  hand-picked placeholder values than genuine measured neural-network
  results.
- **Underspecified procedures** — no disclosed random seed, no repetition
  count, no variance/error bars anywhere in the paper; the "automated method"
  for splitting classes into frequency tiers is asserted but never specified
  as an actual algorithm; UNSW-NB15 doesn't have enough genuinely rare
  classes to support the paper's own claimed `n=7` experiment configuration.

**A second, independent audit pass (yesterday, 2026-07-26) went further and
checked the paper's abstract against this project's own measured data,
plus its bibliography against external sources.** Findings worth flagging
directly to your mentor:

- **The abstract's headline "88% accuracy on minority classes" claim is not
  obtainable from the algorithm as the paper's own methodology section
  describes it.** Running the algorithm exactly as printed (η′=0.1, no
  projection — literally what the paper specifies) gives ~10% rare-class
  accuracy at γ=1 on CIC-IDS-2017, not 88%. The 88.9% number only appears
  under the corrected/fixed arm (η′ lowered to 0.01 and/or projection
  added) — changes this project made, not anything the submitted algorithm
  contains. If a paper goes out with this abstract unmodified, it would be
  making exactly the claim this project's own evidence refutes.
- **Four bibliography references have real problems**, independently
  web-verified: one ("FedSND," ref [12]) appears to be a **fabricated
  citation** — zero trace on arXiv, Semantic Scholar, or DBLP, and its own
  cited arXiv ID postdates the venue/year it's supposedly published at by
  three years. Two others cite the wrong authors or wrong venue for a real
  paper. One has an internally self-contradictory year (cited as "NeurIPS
  2021" while its own arXiv ID is from November 2023).
- **For balance:** the directional shape of the paper's results (FedGRAA
  favoring minority classes more as γ increases, at some cost to overall
  accuracy) is consistent with what this project's faithful reproduction
  actually shows, and the paper's conclusion section is more hedged than its
  abstract. The problems are concentrated in specific numbers and
  unconditional claims, not the overall idea.

---

## 6. Why you can trust these numbers — the audit discipline

This section exists because "we found bugs in a paper" is a claim that
invites the question "how do I know your numbers are right?" The honest
answer is a project habit worth describing to your mentor directly, because
it's arguably as much a research contribution as the finding itself: **the
project audits its own audits, and does not treat "we wrote a report saying
it's done" as equivalent to "it's verified."**

**What an independent, from-scratch re-derivation found (2026-07-26):** five
parallel review passes were run against primary sources only — raw JSON,
`git log`, the actual test files, external web search for citations — not
against the project's own prior documents. On direct re-computation from raw
data (not re-reading prose):

- The "120 of 120 cells leave the simplex in round 1" claim is **exactly
  correct** — every relevant file, every seed, every γ, re-checked.
- **Every** p-value, Cohen's *d*, and Holm-corrected p-value in both ablation
  tables **matches the underlying JSON exactly**, across all 120
  comparisons per metric.
- The Holm–Bonferroni correction's own mathematical guarantee (corrected
  p-value ≥ raw p-value) holds with **zero violations** across 120 checks —
  no bug in the statistics code.
- Every headline number quoted in the top-level README traces back exactly
  to the raw ablation JSON.

**What the same audit found wrong, and what happened next:** real, if mostly
minor, code bugs — e.g., two "parallel" implementations of the composite-loss
calculation disagreed on which of two possible `NaN` cases they silently
dropped (one path fed the numbers actually used to plot results, the other
fed the numbers actually used to drive the weight update — a real,
reachable inconsistency, though no evidence it corrupted any already-published
number); a metric that could silently shrink its own denominator when a rare
class had zero samples in a given round; feature-scaling statistics computed
before the train/validation split, meaning a small amount of validation data
leaked into training normalization. It also found process gaps: the two
"is this stage complete" scripts check *structure* (do the files exist, are
they shaped right) but never recompute the actual statistics, so a
hand-edited or stale number would pass silently; and a status-tracking
document that itself listed 12 items as "still open" that had, per `git log`,
actually all been closed — the exact failure mode the audit trail exists to
catch, occurring inside the artifact built to catch it.

**All of the above code and process bugs were fixed the same day**, in the
very next commit: the NaN-handling asymmetry, the silent-denominator warning,
the stale status document, and — notably — the audit gates (`audit_stage_a.py`
/ `audit_stage_b.py`) were wired into CI (`.github/workflows/docs-check.yml`)
so they run on every push instead of relying on someone remembering to run
them by hand.

The net effect worth telling your mentor plainly: **wherever this audit
checked a specific scientific number against its primary source, the number
was correct.** The defects that mattered were concentrated in the paper
draft's own claims (§5), a handful of real-but-currently-minor code bugs
(now fixed), and a documentation layer that occasionally described a more
finished state than was true (now also fixed). None of the core experimental
conclusions in §3–4 moved as a result of this audit.

---

## 7. Status snapshot as of 2026-07-28 (pre-Stage-C; see the 2026-08-02
   update banner at the top of this document for what changed since)

| Item | Value |
|---|---|
| Tests | 155/155 passing as of this snapshot (2026-07-28); **559/559 current** — see `docs/current/STAGE_C_MASTER_REPORT.md` §12 |
| Completeness gates | `check_stage_a_complete.py` exit 0, `check_stage_b_complete.py` exit 0 |
| Correctness audits | `audit_stage_a.py`, `audit_stage_b.py` — all headline numbers re-derived and matched |
| CI | GitHub Actions runs all four gates + doc-link check on every push (added yesterday) |
| Datasets | CIC-IDS-2017, UNSW-NB15 — both independently verified genuine |
| Diagnostic cells run (Stage B) | 240 (10 seeds × 4 arms × 3 γ × 2 datasets) |
| Paper ambiguities catalogued | 17, independently, paper-text-only |
| Bibliography issues found | 4 references with real, externally-verified problems, incl. 1 likely fabricated |
| Manuscript status | **Does not exist** — a placeholder draft (`docs/paper/draft1.md`) was started yesterday; not reviewed or vetted against the abstract-accuracy problem in §5 yet |
| Extension baselines implemented but never run against FedGRAA | FedProx, FedOpt, FedSND (`fedgraa/baselines.py`) |
| Extension baselines not yet implemented | AFL, q-FedAvg, FedAdp (flagged as "must-have" comparisons by the literature review) |
| Stage C literature review | Substantially complete — ~170 papers surveyed, hallucinated citations caught and removed from the *project's own* corpus, competitor set identified (FedAPA, FedAWA, FedMADE, SBFedNIDS, DA-PFL); ~140 bibliography entries still need independent re-verification before Related Work is written on top of them |
| Venue note | Literature review corrected its own earlier claim — **PAKDD is CORE B, not A** — cuts both ways (novelty fit vs. reviewer unfamiliarity with this specific niche) |

---

## 8. What's actually blocking progress — decisions for the mentor

> **Update, 2026-07-28:** items 2 and 3 below have moved since this section was written. Item 2 (the angle) is narrowed to "Option B+D" by [`Final_Stage_C.md`](docs/research_c/Final_Stage_C.md) §10 with high confidence, and that option's entire implementation now exists and is tested (top of this document) — the decision is now "confirm this angle," not "choose among two unscoped options." Item 3 (compute) is resolved as infrastructure — Modal is validated and ready (`docs/research_c/COMPUTE_INFRASTRUCTURE_2026-07-27.md`) — though **T3/T5's baseline reproduction (~14 other papers' methods) is a real, separately-scoped effort that still needs deciding whether to fully build or partially trade for lab compute** (`docs/research_c/REMAINING_COMPUTE_WORK.md` §4). Items 1 (provenance) and 4 (authorship) are unchanged and still need the actual conversation.
>
> **Update, 2026-07-28, later the same day, corrected again the same day.** Item 2 above needs one refinement, not a reversal: the author confirmed "Option B+D" as the **current** angle and active scope, exactly as this section already describes — and *additionally* chose Option C ("both, sequenced"), meaning Option A (few-shot/prototype hybrid) is committed as a **later follow-on**, not concurrent work and not this pass's scope. Option A has none of B+D's implementation progress and isn't expected to yet — no design doc, code, or tests exist for it (`docs/research_c/REMAINING_COMPUTE_WORK.md` §3a). Item 3's T3/T5 baseline-reproduction scope is unchanged for now; Option A brings its own baseline grid (FedAPA, FedAWA, FedMADE, SBFedNIDS, DA-PFL) later, in its own follow-on phase, mostly overlapping T5's existing list but requiring its own head-to-head runs (`REMAINING_COMPUTE_WORK.md` §4).

Everything technical that can be resolved by writing more code or running
more compute has been resolved. ~~Stage C is at 0%, and it is blocked on one
conversation, not on missing ideas or missing machines.~~ **Stage C's code is
now built; the conversation below is still the blocker for everything that
depends on judgment rather than engineering** — Four concrete
questions, ideally one sitting:

1. **Provenance — the single highest-leverage question.** Is FedGRAA the
   lab's own (as-yet-unpublished) work, or a third party's manuscript being
   independently reproduced and critiqued? Two web searches plus a DBLP and
   Semantic Scholar sweep found **no published trace** of this exact paper
   anywhere. This branches everything downstream:
   - *If it's the lab's own work* — Stage B was never really a "critique of
     someone else's paper"; it's internal QA, and the defect + fix + corrected
     theory become sections of a corrected resubmission, likely co-authored.
   - *If it's a third party's paper* — the "critique as a standalone,
     citable paper section" plan needs rethinking; Stage B becomes internal
     validation and raw material for Stage C's Method section instead.
2. **The Stage C angle.** Two scoped-but-unbuilt options: a **few-shot /
   prototype-guided hybrid aggregation** method (the literature review's
   recommended angle — competitive positioning against 2025 papers like
   FedAPA/FedAWA), or the **privacy-compliant redesign** (a ~730-line design
   doc already exists: the original paper assumes the server can see which
   classes are rare and holds a labelled validation set, both of which
   arguably violate federated learning's core privacy premise — a real,
   citable flaw independent of the convergence defect). Or scope both if time
   allows. This has been "the next decision" for three consecutive project
   passes without being decided.
3. **Compute.** Everything so far is deliberately CPU-only — a GPU attempt
   was tried and measured **25% slower** for a model this small, so it was
   rejected. Stage C's planned baseline comparison (6 extra methods × 2
   datasets) is materially larger than anything run to date, and a
   scalability experiment (50–200 clients, vs. the current 9–14) may
   change the GPU calculus. What's actually available in the lab?
4. **Authorship** on whatever Stage C becomes — worth raising explicitly and
   early rather than leaving implicit.

---

## 9. A workable agenda for the meeting

1. Walk through §3–4 (the core finding + the ablation) — this is the
   strongest, most defensible material and the actual basis for asking the
   provenance question.
2. Ask the provenance question directly (§8.1) — it determines the shape of
   everything else.
3. Show §5's abstract-vs-data gap and the citation issues — these need
   fixing regardless of the provenance answer, before any draft goes further.
4. Decide the Stage C angle (§8.2) together — don't decide alone per the
   project's own working notes.
5. Ask about compute (§8.3) and raise authorship (§8.4).
6. If useful, hand over `docs/current/STAGE_A_MASTER_REPORT.md` and
   `docs/current/STAGE_B_MASTER_REPORT.md` directly — they're written to be
   read stand-alone and are the most detailed evidence available.

---

## Appendix A — glossary (for anyone unfamiliar with FL terms)

| Term | Meaning |
|---|---|
| **Federated learning (FL)** | Training one shared model across multiple clients' data without centralizing the data itself — only model updates are exchanged. |
| **Aggregation** | Combining clients' locally-updated models into one global model each round, typically a weighted average. |
| **Probability simplex** | The set of valid weight vectors: every component ≥ 0, all components sum to 1 — required for a weighted average to make sense as an average. |
| **γ (gamma)** | FedGRAA's personalization coefficient: how much the aggregation objective favors rare-class performance (γ=1) vs. common-class performance (γ=0). |
| **η′ (eta-prime)** | The learning rate used specifically for FedGRAA's weight-learning step (distinct from the model's own learning rate η). |
| **Last-n / rare-class accuracy** | Mean accuracy on the n rarest/protected client classes — the paper's central claimed advantage, though never mathematically defined in the paper itself. |
| **Simplex projection** | The missing operation: after a gradient step potentially pushes weights off the simplex, project them back onto it (the nearest valid point). |
| **Holm–Bonferroni correction** | A statistical adjustment applied when running many hypothesis tests at once, to control the chance of a false positive from sheer multiple comparisons. |
| **Completeness gate vs. correctness audit** | Structural check ("does the artifact exist and is it shaped right") vs. substantive check ("is the reported number actually derivable from the raw data"). |

## Appendix B — where things live

| Ask for this | Find it here |
|---|---|
| The single entry point for "what's left to do" | `PAKDD/docs/current/publication_roadmap.md` |
| Full reproduction record (Stage A) | `PAKDD/docs/current/STAGE_A_MASTER_REPORT.md` |
| Full theory + ablation record (Stage B) | `PAKDD/docs/current/STAGE_B_MASTER_REPORT.md` |
| The mechanism, in full detail | `PAKDD/docs/current/fedgraa_divergence_diagnosis.md`, `fedgraa_convergence_defect.md` |
| The 17-issue paper critique | `PAKDD/docs/current/PAPER_AMBIGUITY_REPORT.md` |
| Yesterday's independent full-repo audit | `PAKDD/docs/audit/INDEPENDENT_FULL_AUDIT_2026-07-26.md` |
| Data provenance verification | `PAKDD/docs/audit/data_provenance_audit.md` |
| Stage C literature review + venue analysis | `PAKDD/docs/research_c/literature_review_and_publication_strategy.md`, `STAGE_C_LITERATURE_MASTER.md` |
| Privacy-redesign extension proposal | `PAKDD/docs/research_c/privacy_compliant_fedgraa_redesign.md` |
| **FedGRAA-v2 master build plan (the current Stage C angle)** | `PAKDD/docs/research_c/FEDGRAA_V2_MASTER_PLAN.md` |
| **FedGRAA-v2 theory workstream (W5)** | `PAKDD/docs/research_c/FEDGRAA_V2_THEORY.md` |
| **What's left for FedGRAA-v2 — the itemized remaining-work map** | `PAKDD/docs/research_c/REMAINING_COMPUTE_WORK.md` |
| **Compute platform selection + validated Modal setup** | `PAKDD/docs/research_c/COMPUTE_INFRASTRUCTURE_2026-07-27.md` |
| Beginner-friendly walkthrough of FL + this paper | `PAKDD/understanding/00_START_HERE.md` onward |
| The reimplementation itself | `PAKDD/fedgraa/` (`aggregation.py`, `trainer.py`, `model.py`, `stats.py`) |
| **FedGRAA-v2's own code** | `PAKDD/fedgraa/secagg.py`, `privacy.py`; `aggregation.py`'s `FedGRAAv2`/`two_phase_composite_grad`; `trainer.py`'s `FederatedTrainerV2` |
| Raw persisted results (not just plots) | `PAKDD/results/` |

## Appendix C — reproduce it yourself

```powershell
git clone https://github.com/N1KH1LT0X1N/Research.git
cd Research/PAKDD
python -m venv .venv313
.venv313\Scripts\Activate.ps1
pip install -r requirements.txt

pytest tests --tb=short                                              # 559 passed
python scripts\check_stage_a_complete.py                             # exit 0
python scripts\check_stage_b_complete.py                             # exit 0
python scripts\check_stage_c_complete.py --tier T1 T2 T3 T4 T5 T6 TC --t4-m 3  # exit 0
python scripts\check_stage_c_complete.py --tier T7 --seeds 0 1 2 3 4 # exit 0
python scripts\audit_stage_a.py             # re-derives every Stage A number from raw JSON
python scripts\audit_stage_b.py             # re-derives every Stage B number from raw JSON
python scripts\audit_stage_c.py             # re-derives every Stage C number from raw JSON
```

Datasets are not tracked in git — see `PAKDD/README.md § Datasets` for where
to get CIC-IDS-2017 and UNSW-NB15.
