# Stage B Correctness Audit — Are the Numbers Actually True?

**Compiled:** 2026-07-25 · **Audited tree:** `HEAD = 2feb18c` (Stage B as committed) · **Instrument:** [`../../scripts/audit_stage_b.py`](../../scripts/audit_stage_b.py), new, committed, negative-control tested · **Stance:** adversarial — the goal was to *break* the committed package, not to confirm it.

**Relationship to the other audits.** [`STAGE_AB_VERIFICATION_AUDIT_2026-07-25.md`](STAGE_AB_VERIFICATION_AUDIT_2026-07-25.md) is a broad, human-written verification sweep across Stage A **and** Stage B, covering documentation hygiene, version control, and strategy alongside the numbers. This document is narrower and deeper: it covers Stage B's *numbers only*, and its findings are backed by an **automated, re-runnable gate** rather than a one-time reading. The two overlap on one finding (metric-divergence, F2 there / **A4** here) and agree; where this document goes further is that the check is now executable and will fail a future regression. Read that one for "is the project in good shape?"; read this one for "can I trust each specific number, and what stops it from drifting?"

---

## 0. Verdict

**The science held; the reporting did not, in four places.** Every scientific claim survived — the feasibility theorem's empirical prediction, the significance testing, the factorial decomposition, and the raw runs themselves were all confirmed correct, to the digit, by independent recomputation. **Zero** numerical errors were found in the computed results.

All four defects were in **how results were summarized and described** — the layer between correct data and a reader's understanding. That is not a lesser category of error: a correct number reported in a misleading way misinforms exactly as effectively as a wrong one, and three of these four would have been caught by a competent reviewer.

| # | Defect | Severity | Class |
|---|---|---|---|
| **A1** | Composite-loss blowup reported as the **mean of a heavy-tailed ratio** — `185764×`, a figure describing no run that occurred | **High** | Misleading summary statistic |
| **A2** | **Seed-0 worst-case figures presented as the 10-seed campaign's** — understated the CIC/UNSW severity gap by three orders of magnitude | Medium | Stale figure |
| **A3** | **Run-count overstatement** — "80 fresh diagnostic runs"; 71 were fresh, 9 reused | Medium | Reproducibility claim |
| **A4** | **"Statistically indistinguishable" was rare-accuracy-only** — an arm at 15.2% all-class accuracy read as equivalent to one at 44.3% | **High** | Incomplete significance reporting |
| **A5** | *(found by negative control)* The audit script **crashed while reporting a failure** (`UnicodeEncodeError` on `η′`) | Medium | Tooling |
| **A6** | *(found by the audit's own strictness)* A **malformed significance marker** in the theory doc — `\***` where p=0.042 warrants one star | Low | Transcription |

Every one is fixed, and each fix is now **guarded by an automated check** that fails if it recurs (§4).

---

## 1. Why a second gate was needed

Stage B already had [`check_stage_b_complete.py`](../../scripts/check_stage_b_complete.py), and it passed. It was not enough, and the reason is worth stating precisely because it generalizes.

That script answers **"is every artifact present and shaped right?"** — do all 80 diagnostic JSONs exist, does each carry a `diagnosis` block, does the table contain every arm×γ cell and all five comparisons, do the figures render non-empty, is every manifest path git-tracked. Those are real checks and they caught real gaps during Stage B.

But **every single defect in §0 passes that script.** A blowup mean of 185,764× is a well-formed float in a correctly-shaped cell. A stale 7.6× is a valid number in a valid table. "80 fresh runs" is prose. A rare-accuracy-only significance claim uses correctly-computed p-values in the right JSON block. Shape validation is structurally blind to all of it.

The gap is between **completeness** and **correctness** — and correctness has to be checked against something *other than the artifact asserting it*. That is the design principle of the new gate: every number is re-derived from the most primitive source available (the raw per-seed training output), and every statistic is recomputed with an independent implementation (SciPy directly, never the repo's own `fedgraa/stats.py`).

---

## 2. Method

[`scripts/audit_stage_b.py`](../../scripts/audit_stage_b.py) runs six check families, ordered by what they can catch:

| Check | What it re-derives | What it would catch that shape-checking cannot |
|---|---|---|
| **A. Raw-run integrity** | SHA-256 of all 80 diagnostic JSONs; every cell's persisted `config` vs. the arm it is filed under (projection flag, η′, E′, seed, rounds, γ) | A run duplicated and filed as two seeds; an arm whose data was generated with the wrong flags — i.e. an ablation that isn't actually ablating |
| **B. Table-vs-raw fidelity** | Every `per_seed` array re-read from the raw JSONs and compared exactly; mean, median, CI bracketing and data-range containment | A globbing or seed-mapping bug in the builder silently pulling the wrong files |
| **C. Independent statistics** | Every paired-*t* p, Wilcoxon p, Cohen's *d*, Holm adjustment and factorial effect, recomputed via SciPy **without** `fedgraa/stats.py` | A bug in the project's own statistics module — which testing the module against itself cannot find |
| **D. Doc transcription** | All **262** numbers hand-typed into the theory doc's tables, parsed back out and re-derived; every significance star checked against its own p-value | Copy-paste drift between the data and the prose that cites it — the most likely failure mode for hand-written tables |
| **E. Presentation soundness** | Detects cells where mean/median diverge >3× and requires the rendered tables not to show the mean | **A1** — a correct number that is a misleading summary |
| **F. Metric divergence** | Re-derives the all-class-accuracy comparison family; flags any comparison that ties on the metric of record but differs materially on all-class accuracy | **A4** — a *true* "not significant" that is nonetheless misleading |

**The independence principle matters most in C.** `fedgraa/stats.py` has 19 unit tests, and they pass — but unit tests written alongside a module share its author's assumptions. Recomputing the same 90 comparisons through SciPy is a genuinely independent check. It confirmed the module: bootstrap CIs match an independent percentile-bootstrap implementation to **0.000%**, and all p-values, effect sizes, and Holm adjustments match to machine precision.

### 2.1 Negative control — an audit that cannot fail is worthless

Before trusting a green result, three defects were deliberately injected into a scratch copy of the tree:

| Injected defect | Detected? | Message produced |
|---|---|---|
| A transcribed number changed in the doc (`69.7%` → `71.2%`) | ✅ | `D: rare-acc mean ... both ... g0.0: doc 71.2 vs json 69.7` |
| A p-value corrupted in the table JSON (`1.59e-07` → `0.0424`) | ✅ | `C: ... p_ttest 0.0424242 vs scipy 1.59e-07` **plus 3 cascading Holm failures** — the correction is recomputed jointly, so one corrupted p is caught five times |
| A run file overwritten with a copy of another seed | ✅ | `A: ... seed7 ... is byte-identical to ... seed5` **plus** `config seed: got 5 want 7` at all three γ |

**This exercise found defect A5**: the script detected the first injected error correctly and then died with `UnicodeEncodeError` while *printing* the failure message, because the row label contains `η′`, which the Windows console's cp1252 codec cannot encode. An auditor that exits non-zero with no readable reason is barely better than none. Fixed by reconfiguring `stdout` to UTF-8 with `errors="replace"`.

That is the clearest argument for negative-controlling a verification tool: the bug was **in the failure path**, which by construction never executes while everything is passing.

---

## 3. The findings in detail

### A1 — Blowup reported as the mean of a heavy-tailed ratio *(High)*

The CIC-IDS-2017 literal arm at γ=0 was reported as **`blowup 185764.05×`**. The per-seed values are:

```
[3.78, 7.59, 29.76, 2.58, 3.62, 3.59, 2.47, 3.14, 1857580.09, 3.87]
```

Nine seeds sit near 3×. One diverged to **1.86 million×**. The mean describes none of them: it is 50,000× the median, and no run landed within two orders of magnitude of it.

**Why it mattered.** Composite-loss blowup is a *ratio*, not a bounded score like accuracy — it is heavy-tailed by construction, and a mean is the wrong summary. The reported figure implied a typical catastrophic run when the truth is "usually ~4×, occasionally astronomically worse." A reviewer inspecting the per-seed array — which was, to be fair, persisted in the JSON all along — would have flagged it immediately.

**Fix.** `_cell()` now carries `median`/`min`/`max` alongside the mean; the renderer reports **median (and max)** for blowup; the choice is documented in each rendered table and in the theory doc. Two regression tests lock it in, one of them using this exact per-seed array as its fixture.

**The corrected form is strictly more informative** — and rhetorically stronger, not weaker:

> `blowup 3.70× med, 1.86×10⁶× max`

It also surfaced a finding the mean had flattened: **only the `both` arm has median *and* max of 1.00× at every γ.** `projection-alone` still reaches 5.60× and `η′-alone` still reaches 30.02× somewhere in the γ range. Feasibility makes the *unbounded* excursion structurally impossible; the η′ cut flattens the bounded residual. That is a two-lever story the single mean had hidden.

### A2 — Seed-0 worst cases presented as the campaign's *(Medium)*

§6 read: *"CIC-IDS-2017's worst-case blowup is **193×**; UNSW-NB15's worst-case, also at γ=1, is **7.6×**"* — describing them as differing "by an order of magnitude." Both are **seed-0** values, written before the 10-seed campaign existed and never revisited after it landed.

Over the full seed set the maxima are **1.86×10⁶×** (CIC) and **46.85×** (UNSW) — roughly **four** orders of magnitude apart, not one.

**Why it mattered.** The qualitative claim ("CIC is worse") survived, but the paper's own severity argument was understated by three orders of magnitude — an error in the *conservative* direction, which is how it evaded notice. Stale-figure defects are insidious precisely when the new data strengthens the claim.

**Fix.** §2, §4 and §6 now separate *typical run* (median: CIC 2.71–14.19×, UNSW 1.50–8.26×) from *tail severity* (max), and state the distinguishing feature explicitly: the medians are close, so what separates the datasets is **how catastrophic CIC's tail is**, not the typical run. §2 additionally now records the far stronger 10-seed result that `first_simplex_violation_round = 1` holds on **all 60 literal-arm cells**, not just seed 0.

### A3 — Run-count overstatement *(Medium)*

The master report claimed **"80 fresh diagnostic runs."** In fact 71 were newly executed; 9 were Stage-A-era files whose arm, config, and seed already matched the factorial and which the driver's resumability logic correctly reused.

**Why it mattered.** This is a reproducibility claim, not a rounding error. It overstates the compute performed and understates the (entirely legitimate, and separately verified) reuse. Check **A** independently confirms all 9 reused files carry the correct config, so the reuse was sound — but the description was not.

**Fix.** Corrected to "80 (arm, seed) runs — 71 newly executed, 9 reused," with the reuse mechanism named.

### A4 — "Statistically indistinguishable" was rare-accuracy-only *(High)*

This is the finding with real scientific consequence, and the one this audit is most glad to have caught.

Stage B pre-registered rare-class accuracy as the metric of record and ran all five pairwise comparisons on it — correctly. §5.1 then reported that on CIC-IDS-2017, `η′-alone` is *"statistically indistinguishable from `both`"* at γ=1.0 (Holm p = 0.97), and concluded that **"the η′ cut does the great majority of the work."**

Every number in that sentence is correct. The conclusion is wrong.

Running the same five comparisons on **all-class accuracy**, on the same seeds:

| CIC-IDS-2017, γ=1.0 | rare-acc | all-class acc |
|---|---|---|
| `both` (the fix) | 89.3% | **44.3%** |
| `η′-alone` | 88.9% | **15.2%** |
| Holm-corrected p | 0.97 *(ties)* | **5.6×10⁻⁴** *(d = +1.99)* |

`η′-alone` achieves rare-class parity **by starving every frequent class** — all-class accuracy collapses to 15.2%. It is the same over-correction pathology the doc already documented for η′=0.001, reappearing at η′=0.01 once γ=1 removes `L_a`'s counterweight. An ablation reporting only its metric of record scores that arm as *equivalent to the fix*.

**Why it mattered.** This is subtler than a wrong number — it is a *true statement that misleads*. It cannot be caught by recomputation, because there is nothing to recompute: the p-value is right. It is caught only by asking whether the reported metric is sufficient to support the conclusion drawn from it.

**Fix.** Four parts:
1. The builder now computes the **full five-comparison family on all-class accuracy** as well (`comparisons_all_acc`), Holm-corrected within its own family.
2. The renderer emits both families side by side and **auto-flags rows where they disagree** — `**(diverges)** ties on rare, differs on all-acc`. The flag is machine-generated, so it cannot be forgotten.
3. §5.1, §5.2 and §5.3 of the theory doc were rewritten around the dual-metric evidence.
4. Check **F** enforces disclosure: any comparison that ties on rare-acc but differs significantly on all-acc with a ≥5pp gap **fails the audit** unless the doc discloses a metric divergence.

**What the corrected analysis concludes** is more nuanced, and more defensible, than either the original preview hypothesis or the first corrected version:

- At CIC γ=0.5 the `η′-alone` tie is **genuine on both metrics** — there the η′ cut really does carry the arm.
- At CIC γ=1.0 it is **hollow** — 29pp of all-class accuracy traded away.
- The trade-off runs both ways: at CIC γ=1.0 `projection-alone` is *significantly better than `both`* on all-class accuracy (48.4% vs 44.3%, p=0.042).
- The mirror check on UNSW γ=1.0 flags the same divergence in `both`'s favour but at **0.4pp** — statistically clean, materially trivial. That the check is not simply flagging everything is itself evidence it is measuring something real.

So: **no arm dominates on both metrics at every γ.** The defensible claim is narrower and more useful — `both` is the only factorial arm **never catastrophically wrong on either metric anywhere**. Deficits against it are single-digit percentage points; deficits against the single-lever arms reach 29pp and 38pp. For anyone applying the correction without per-dataset, per-γ retuning, that worst-case profile is the argument.

### A5 — The audit crashed while reporting a failure *(Medium, tooling)*

Covered in §2.1. Found only because the tool was negative-controlled.

### A6 — Malformed significance marker *(Low)*

While validating the *new* dual-metric tables, check **D** rejected a cell reading `**0.0424 \*** (d=−0.88)**` — three stars where p = 0.042 warrants one. A mistyped bold delimiter, introduced while writing the A4 fix, caught within minutes by the guard built for a different purpose.

Worth recording for two reasons. First, it is evidence the transcription guard earns its keep on live editing, not just historical data. Second, the same run exposed a bug in the guard's own regex (an alternation matching `\*` before `\*\*\*`, reading three stars as one) — so a single failing run surfaced both a doc error and a checker error. Both fixed.

---

## 4. What is now guarded

Every defect above has a corresponding automated check, so none can silently recur:

| Defect | Guard | Fails if… |
|---|---|---|
| A1 heavy-tailed mean | Check **E** + `test_median_is_robust_to_a_single_diverging_seed` + `test_render_reports_blowup_as_median_not_mean` | any rendered blowup shows a mean exceeding 3× its median |
| A2 stale figures | Check **D** (262 numbers re-derived) | any doc number stops matching the JSON |
| A3 run-count claim | Check **A** (config-vs-arm on all 80 runs) | a run's persisted config contradicts its filed arm |
| A4 metric-only significance | Check **F** | a rare-acc tie hides a ≥5pp all-acc gap that the doc doesn't disclose |
| A5 crash-on-report | UTF-8 `stdout` reconfiguration | — (negative-control the tool after changes) |
| A6 star markup | Check **D** star validation, per metric family | a significance run disagrees with its own p-value |

**Gate status on `HEAD`:**

```
python -m pytest tests                    → 142 passed
python scripts/check_stage_b_complete.py  → exit 0   (completeness: presence + shape)
python scripts/audit_stage_b.py           → exit 0   (correctness: 262 doc numbers,
                                                      90 comparisons, 240 configs)
```

---

## 5. What this audit does *not* cover

Stated plainly, so the green result is not over-read:

- **Stage A is out of scope.** Its numbers were verified by [`full_audit_2026-07-23.md`](full_audit_2026-07-23.md) and again by [`STAGE_AB_VERIFICATION_AUDIT_2026-07-25.md`](STAGE_AB_VERIFICATION_AUDIT_2026-07-25.md), but **not** by this kind of automated, re-runnable transcription gate. Extending `audit_stage_b.py`'s approach to Stage A's tables is the obvious next hardening step.
- **It cannot validate the training itself.** It verifies that persisted outputs are faithfully summarized and described. If `fedgraa/` computed a metric incorrectly, every check here would still pass — that is what the 142-test suite and the Stage A reproduction audit are for.
- **It does not check the theory.** §1–4 of the theory doc are mathematical claims; the audit only verifies that the *empirical* figures cited in support of them match the data.
- **Statistical power is unchanged.** The audit confirms the tests were applied correctly; it cannot make n=10 into a larger sample. The Wilcoxon floor (many p-values sit at exactly `0.001953` = 2/2¹⁰, the smallest two-sided p attainable at n=10) is a genuine resolution limit, now disclosed in the theory doc's limitations.
- **Doc-link integrity, cross-stage consistency, and version-control hygiene** are covered by the companion audit's F1/F6/F9, not here.

---

## 6. The generalizable lesson

Three of the six defects share a structure: **a correct number, reported in a way that supports a wrong conclusion.** A heavy-tailed mean, a stale-but-valid figure, and a true-but-insufficient significance test. None is a computational error; all three would mislead a reader; none was catchable by any completeness or shape check, and two were not catchable by recomputation either.

The check that caught the most important one (A4) is not a numerical check at all. It asks: *does the evidence reported actually support the sentence written next to it?* Mechanizing that fully is not possible, but the useful approximation turned out to be cheap — compute the obvious alternative metric, and flag disagreement. That single idea, in about forty lines, caught the audit's highest-value finding.

The corollary for future stages: **when a result is summarized, the summary is a claim and needs its own verification.** Persisting the per-seed array is necessary but not sufficient — the data was right in the JSON the whole time, and the reported figure was still wrong by five orders of magnitude.

---

*Instrument: [`../../scripts/audit_stage_b.py`](../../scripts/audit_stage_b.py) · Fixes committed alongside · Companion: [`STAGE_AB_VERIFICATION_AUDIT_2026-07-25.md`](STAGE_AB_VERIFICATION_AUDIT_2026-07-25.md) · Capstone: [`../current/STAGE_B_MASTER_REPORT.md`](../current/STAGE_B_MASTER_REPORT.md) §7.1*
