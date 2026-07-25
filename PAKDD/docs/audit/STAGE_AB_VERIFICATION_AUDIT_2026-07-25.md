# Stage A + Stage B Verification Audit — Independent Re-Derivation of Every Claim

**Compiled:** 2026-07-25 · **Scope:** everything Stage A and Stage B claim, verified against raw artifacts rather than against the documents that assert them · **Tree audited:** working tree at `HEAD = 4459f53` (plus uncommitted changes, see F1) · **Method:** re-ran both completeness gates and the full test suite; independently recomputed every ablation cell, confidence interval, *p*-value, effect size, and multiplicity correction from raw per-seed JSON using SciPy directly rather than the repo's own `fedgraa/stats.py`; re-derived every number in `STAGE_A_MASTER_REPORT.md` §5–§8 from the persisted sweeps; verified the projection operator against a numerical optimizer.

---

## 0. Verdict

**Stage A and Stage B are genuinely complete, and the numbers are trustworthy.** Every quantitative claim I could check reconciled — not approximately, but to the last digit reported, across roughly 400 independent checks. I found **zero** numerical errors, zero fabricated or drifted values, and zero statistical misapplications in the reported results.

Every defect I did find lives in the **documentation and version-control layer**, not in the science:

| # | Finding | Severity | Domain |
|---|---|---|---|
| **F1** | The corrected ablation tables and Stage A's 976-line master report are **not committed** — a clean checkout of `HEAD` reproduces the superseded numbers | **High** | Version control |
| **F2** | "Statistically indistinguishable" claims are **rare-accuracy-only**; on all-class accuracy the same comparisons reverse and become highly significant | **High** | Reporting completeness |
| **F3** | The prior audit's **CRITICAL** provenance finding (D1 — "published" framing) remains entirely unactioned | **High** | Strategy |
| **F4** | `publication_roadmap.md` — the designated entry point — is ~2 stages behind and still instructs "Execute Stage A" | Medium | Staleness |
| **F5** | The two newest strategy docs describe the Stage B campaign as unrun and give instructions that are now actively wrong | Medium | Staleness |
| **F6** | 37 broken relative links across 63 markdown files; neither completeness gate checks doc links | Medium | Hygiene |
| **F7** | `PAKDD_STATUS_REPORT.md` stale (79/79 tests, wrong headline fix, 2 broken links) | Low | Staleness |
| **F8** | "byte-identical" (UNSW n=5 vs n=7) is literally false — metrically identical, one metadata field differs | Low | Precision |
| **F9** | Stage A's capstone still frames blowup severity on seed-0 statistics that Stage B has since corrected | Low | Cross-doc consistency |

**Answer to "are we going good?" — yes, on the evidence.** The reproduction and the critique are in materially better shape than the documentation around them suggests. The work is ahead of its own bookkeeping, which is a much better problem than the reverse.

---

## 1. What was verified, and how

### 1.1 The gates and the suite

| Check | Result |
|---|---|
| `python -m pytest tests` | **139 passed**, 134.6 s — matches the claimed count exactly |
| `python scripts/check_stage_a_complete.py` | **exit 0** — 18 checks, all pass |
| `python scripts/check_stage_b_complete.py` | **exit 0** — 8 arm×dataset seed-coverage checks, 2 table loads, 4 render checks, theory-doc grep, 27-path manifest check |

Both gates are honest instruments, not rubber stamps. Stage A's checker actually re-renders the n-sweep figures in memory and counts `Line2D` objects per panel (≥5) — a *rendered* guard against the historical `plot_n_sweep` defect, not merely a file-existence check. Stage B's checker independently walks the full 4-arm × 10-seed × 2-dataset grid and validates every `diagnosis` block field. Both resolve manifest paths against `git ls-files`, so a listed-but-untracked artifact fails.

### 1.2 Structural integrity of the campaign data

Loaded all **240** `(dataset, arm, γ, seed)` diagnosis blocks — 4 arms × 2 datasets × 10 seeds × 3 γ — and checked each against what its filename asserts:

- `project` flag matches the arm (literal/η′-alone `False`; projection-alone/both `True`) — **240/240**
- `eta_prime` matches the arm (0.1 / 0.01) — **240/240**
- `weight_epochs = 5` — **240/240**
- `config.seed` equals the seed encoded in the filename — **240/240**
- `rounds = 60`, `eval_every = 1`, `n_rare = 3` — **240/240**
- `history.gamma` matches the γ key it is filed under — **240/240**
- all five history arrays exactly 61 points long — **240/240**

No arm is silently mislabelled, no seed is duplicated under two filenames, and no run drifted off the fixed diagnostic horizon. This matters more than it sounds: the entire factorial's validity rests on exactly one lever differing between adjacent cells, and that is now verified at the config level rather than assumed from the file-naming convention.

### 1.3 Independent statistical re-derivation

I rebuilt the consolidated tables from raw per-seed JSON without importing `build_ablation_table.py` or `fedgraa/stats.py`, using SciPy's `ttest_rel` / `wilcoxon` and my own bootstrap and Holm implementations, then diffed against the persisted tables.

| Quantity | Cells compared | Mismatches |
|---|---|---|
| Per-seed arrays (rare-acc, all-acc, blowup) | 24 arms×γ × 3 metrics, both datasets | **0** |
| Cell means | 72 | **0** (exact to 1e-12) |
| Bootstrap 95% CIs | 72 | **0** (within 2e-3, B=10,000) |
| Baseline per-seed arrays vs `seed_sweep_*_r60.json` | 8 | **0** |
| Paired *t p*-values | 30 | **0** (exact to 1e-9) |
| Wilcoxon *p*-values | 30 | **0** |
| Cohen's *d* (paired) | 30 | **0** |
| Holm-adjusted *p* | 30 | **0** |
| Factorial main effects + interaction (mean, CI, *p*) | 18 | **0** |

Every headline number in `STAGE_B_MASTER_REPORT.md` §4–5 and `fedgraa_convergence_defect.md` §5.1–5.2 traces back to raw data. The `git_sha` recorded inside both table JSONs (`a17a443…`) is real and consistent with the manifest.

### 1.4 The two load-bearing scientific claims

**Feasibility (the corrected proposition's part 1) — holds with zero exceptions.** Across all **60** `both`-arm cells (10 seeds × 3 γ × 2 datasets): `final_weight_sum = 1.0` within float32 round-off, `final_weight_min ≥ 0`, and `first_simplex_violation_round is None`. **0 violations.** This is the pre-registered falsifiable prediction and it survived at full scale.

**The defect (§2's claim) — holds with zero exceptions.** Across all **60** `literal`-arm cells: `first_simplex_violation_round = 1`. **60/60.** The round-1 feasibility failure is not a seed artifact.

**Blowup, checked against the documents' exact wording:**

| Claim in docs | Verified |
|---|---|
| CIC `both`: exactly 1.0× on 29/30 cells, one at 1.016× | ✅ 29/30, max 1.0161 |
| UNSW `both`: reduced but not flattened, never exceeding 1.87× | ✅ 14/30 exactly 1.0, max 1.8720 |
| CIC literal γ=0 per-seed `[3.78, 7.59, 29.76, 2.58, 3.62, 3.59, 2.47, 3.14, 1857580.09, 3.87]` | ✅ exact |
| UNSW literal reaches 46.85× at γ=1 over 10 seeds | ✅ exact |

### 1.5 Stage A's numbers

Re-derived **~90 values** in `STAGE_A_MASTER_REPORT.md` §5–§8 directly from `results/*.json`: both γ-sweeps' final-round all/rare accuracy for all 5 methods; both n-sweeps' final-round rare accuracy for all 5 methods × 3 n; both literal diagnostics' blowup, `Σw`, `min(w)`, and accuracies at all 3 γ; both combined-fix seed bands' means. **Zero mismatches.**

Including the γ=1 CIC composite-loss trace quoted in §5.2, digit-for-digit:

```
2.98  2.77  4.53  6.06  40.52  75.55  261.48  290.24  319.11  205.30  149.79
```

**Round-0 convention.** The manifest claims the strict/ditto convention on all four base sweeps. Verified: **30/30 curves** across all four artifacts have `all_acc[0] = last_n_acc[0] = 0.0`. The config-vs-artifact self-contradiction that `stage_a_plan.md` §3 flagged is genuinely resolved, not merely declared resolved.

### 1.6 The projection operator

`project_to_simplex` (Duchi et al. 2008) is the algorithmic foundation of the corrected proposition, so I tested it as an optimization problem rather than reading it. Over **300 random vectors** (K ∈ [2,15], N(0,3) entries), compared against SLSQP solving `min ‖x − w‖² s.t. x ≥ 0, Σx = 1` directly:

- max deviation **9.34×10⁻⁷** — the float32 representation limit, not an algorithmic error
- `Σ = 1` and `min ≥ 0` on all 300
- edge cases correct: singleton → `[1.0]`; all-zeros → uniform; all-negative → uniform; extreme `[1e6, −1e6]` → `[1, 0]`

The `if not np.any(cond)` guard returning uniform is a defensive branch that the correct Duchi condition should make unreachable; it is harmless and never fired in testing.

### 1.7 `fedgraa/stats.py` degenerate-cell handling

The module documents conservative contracts for zero-variance data. I checked whether those paths actually fire on the real campaign:

- **Bootstrap point-CI path:** fires on **6 of 28** cells (CIC literal γ=0/0.5, CIC projection-alone γ=0/0.5, and FedAvg on both datasets — all exactly 0.0% rare-acc on every seed). Correctly returns a degenerate point CI, not `NaN`.
- **Degenerate paired-*t* path** (constant paired differences → returns `(0.0, 1.0)`): fires on **0 of 30** comparisons. The conservative contract therefore affects **no reported number** — worth recording, because that contract would be anti-conservative if inverted, and it is currently untested by live data rather than merely untriggered.

### 1.8 What the campaign got right that a weaker pass would have missed

Two things deserve explicit credit, because both are the kind of error that surfaces in review rather than in self-audit:

1. **The heavy-tailed blowup.** CIC literal γ=0 has one seed at 1.86×10⁶× against nine near 3×. The mean (185,764×) describes no run that occurred. This was caught, and `build_ablation_table.py` / `render_ablation.py` now carry median/min/max with an explicit rationale in the code docstring, the rendered tables, and the theory doc. The `§6` severity narrative was correspondingly rewritten from the seed-0 "one order of magnitude" framing to the correct "four orders on the tail, medians far closer." That is exactly the right correction.
2. **The n=3 hypothesis was overturned and reported as such.** The preview data supported "both levers are always independently necessary." The n=10 factorial showed the load-bearing lever is dataset-dependent (η′ on CIC, projection on UNSW), and the docs lead with that rather than burying it. Reporting a pre-registered hypothesis's refutation is the single most reliable signal that a result set was not tuned.

---

## 2. Findings

### F1 — High — The corrected work is not committed; `HEAD` reproduces superseded numbers

`HEAD` is `4459f53 "Stage B: convergence-defect theory + consolidated 10-seed ablation with significance testing"` (2026-07-25 03:01). The working tree has diverged from it, and the divergence contains **the scientific correction described in §1.8**.

**Modified but uncommitted (tracked files):**

```
M PAKDD/results/ablation/CIC_IDS_2017_ablation_table.json    (+122 lines: median/min/max)
M PAKDD/results/ablation/CIC_IDS_2017_ablation_table.md      (mean -> median/max reporting)
M PAKDD/results/ablation/UNSW_NB15_ablation_table.json
M PAKDD/results/ablation/UNSW_NB15_ablation_table.md
M PAKDD/scripts/build_ablation_table.py                       (+22 lines)
M PAKDD/scripts/render_ablation.py                            (+34 lines)
M PAKDD/tests/test_build_ablation_table.py                    (+30 lines)
M PAKDD/tests/test_render_ablation.py                         (+20 lines)
```

**Untracked:**

```
?? PAKDD/docs/current/STAGE_A_MASTER_REPORT.md          (976 lines — Stage A's capstone)
?? PAKDD/docs/current/STAGE_DIRECTION_ASSESSMENT.md
?? PAKDD/docs/audit/RESEARCH_DIRECTION_AUDIT_2026-07-25.md
?? PAKDD/docs/paper/FedGRAA_Simulated_Peer_Review.md
?? PAKDD/docs/research_c/                                (3,818 lines across 8 docs)
?? PAKDD/results/ablation/_logs/
```

**Staged deletions** (5 docs moved `docs/current/` → `docs/research_c/`, move not committed).

**Why this matters more than ordinary uncommitted work:**

- A clean checkout of `HEAD` yields ablation tables reporting **blowup 185764.05×** — the statistic the project has since correctly determined "describes no run that actually happened."
- `check_stage_b_complete.py` passes on the working tree and its manifest check passes, because it verifies paths are *git-tracked*, not that their *content* is committed. The gate cannot distinguish "committed and correct" from "modified in the working tree." That is a reasonable design, but it means the green signal does not currently mean what a reader would assume.
- `STAGE_A_MASTER_REPORT.md` is cited as authoritative by `STAGE_DIRECTION_ASSESSMENT.md`, `RESEARCH_DIRECTION_AUDIT_2026-07-25.md`, and `fedgraa_convergence_defect.md`, and has never been committed. Stage A's own plan declares it complete "against commit `be4ef03`"; the document making that case does not exist in any commit.

**Fix:** commit the working tree in two logical commits — (a) the blowup median/max correction with its tests and regenerated tables, (b) the doc reorganization plus the four untracked capstone/strategy docs. Decide whether `results/ablation/_logs/` belongs in git or `.gitignore`.

**Optional hardening:** have `check_stage_*_complete.py` additionally warn when any manifest-referenced path is dirty relative to `HEAD` (`git diff --quiet -- <path>`). Three lines; closes the gap between "the gate is green" and "the gate is green on what is committed."

---

### F2 — High — The "statistical tie" conclusions are rare-accuracy-only, and reverse on all-class accuracy

This is the one finding that touches the science rather than the bookkeeping.

Both capstones draw their central cross-dataset conclusion from comparisons run **only on rare-class accuracy**:

> "at γ=0.5–1.0 it still beats literal/FedAvg significantly, but is statistically indistinguishable from `η′-alone` (Holm p = 1.0 and 0.97) — **the η′ cut is doing most of the work on this dataset**." — `STAGE_B_MASTER_REPORT.md` §4

> "`projection-alone` is statistically indistinguishable from `both` at **every** γ (Holm p = 0.606, 0.418, 0.544)" — `fedgraa_convergence_defect.md` §5.2

All-class accuracy is carried in every cell and in the table JSON, but **is never significance-tested**. I ran the same paired *t* + Holm protocol on it. The conclusions do not survive:

**CIC-IDS-2017 — all-class accuracy, `both` vs. reference arm**

| γ | vs. arm | arm all-acc | both all-acc | Δ (pp) | *p* | Cohen's *d* | Holm *p* |
|---|---|---|---|---|---|---|---|
| 0.0 | η′-alone | 71.67% | 62.34% | **−9.34** | 0.0174 | −0.92 | 0.052 (n.s.) |
| 0.5 | literal | 26.67% | 54.24% | +27.56 | 7.7e-04 | 1.57 | **0.0023 \*\*** |
| 1.0 | projection-alone | 48.36% | 44.32% | **−4.03** | 0.0212 | −0.88 | **0.042 \*** |
| 1.0 | **η′-alone** | **15.17%** | **44.32%** | **+29.16** | 1.4e-04 | **1.99** | **4.2e-04 \*\*\*** |

**UNSW-NB15 — all-class accuracy**

| γ | vs. arm | arm all-acc | both all-acc | Δ (pp) | *p* | Cohen's *d* | Holm *p* |
|---|---|---|---|---|---|---|---|
| 0.5 | η′-alone | 49.25% | 53.20% | +3.95 | 0.0147 | 0.95 | **0.044 \*** |
| 1.0 | literal | 29.15% | 52.19% | +23.04 | 0.0046 | 1.18 | **0.0092 \*\*** |
| 1.0 | **projection-alone** | 51.81% | 52.19% | +0.38 | 0.0063 | 1.12 | **0.0092 \*\*** |
| 1.0 | **η′-alone** | **13.33%** | **52.19%** | **+38.86** | 3.2e-06 | **3.20** | **9.7e-06 \*\*\*** |

**Three consequences:**

1. **The CIC "η′ does most of the work" conclusion is metric-specific and, stated unqualified, misleading.** At γ=1.0, `η′-alone` ties `both` on rare accuracy (89% vs 89%) while collapsing all-class accuracy to **15.2%** against `both`'s 44.3% — a 29pp gap at *d* = 1.99, Holm *p* = 4.2e-04. An arm that recovers the rare classes by destroying the majority classes has not "done the work"; it has relocated the failure.

2. **The canonical η′-alone arm exhibits the same starvation pathology §6 attributes only to the η′=0.001 over-correction row.** §6 currently draws a clean line: the 0.001/E′=1 run "starves every non-rare client… all-class accuracy collapses to 0.8%," and is "*not* the same intervention as the factorial's canonical η′-alone cell (η′=0.01, E′=5 — a 10× reduction, not 100×)." The data says the canonical cell shows a milder version of exactly that pathology — 15.2% (CIC) and 13.3% (UNSW) all-class accuracy at γ=1. That is a difference of degree, not of kind, and the current framing implies otherwise.

3. **The UNSW "projection-alone ties both at every γ" conclusion also weakens.** On all-class accuracy at γ=1.0, `both` beats `projection-alone` at Holm *p* = 0.0092, *d* = 1.12 — significant, even though the point-estimate gap is only +0.38pp (the paired differences are tightly consistent across seeds, which is what a paired test is designed to detect).

**This finding strengthens the paper's actual recommendation.** `both` is already the recommended arm on "performs well across both datasets" grounds; adding the all-class dimension gives it a *statistically significant* basis on both datasets rather than a "ties on the primary metric, wins on robustness grounds" one. The current under-reporting is costing the project its strongest available argument.

**Fix:**
- Extend `build_ablation_table.py` to run the 5 pre-registered comparisons on `all_acc` as well as `rare_acc`. The per-seed arrays are already stored; this is a loop over a second metric plus a second `comparisons_all_acc` block, ~30 lines, no re-training. Extend `check_stage_b_complete.py` and the renderer to match.
- Pre-register the secondary-metric comparisons explicitly so this reads as planned analysis rather than post-hoc fishing — the honest framing is "rare-class accuracy is the metric of record; all-class accuracy is a pre-specified secondary endpoint, corrected within its own family."
- Qualify §5.1's and §5.2's narrative paragraphs and `STAGE_B_MASTER_REPORT.md` §4/§5/§6 accordingly.
- Revise §6's η′=0.001 paragraph to place the canonical η′-alone cell on the same continuum rather than opposite it.

---

### F3 — High — The prior audit's CRITICAL finding is entirely unactioned

`RESEARCH_DIRECTION_AUDIT_2026-07-25.md` (compiled 01:04 today) raised finding **D1 — CRITICAL: Stage B's anchor contribution has no venue, because the target paper is not published**, evidenced by two zero-result web searches for the FedGRAA title and acronym, and by the manuscript itself being an anonymized submission draft ("No Author Given").

As of this audit, **nothing downstream of D1 has changed**:

- `fedgraa_convergence_defect.md` is still titled "Why a **Published** FL-IDS Aggregation Algorithm Doesn't Converge, and How to Fix It."
- Its §7 limitations list does not mention the provenance question.
- No document in the tree records the manuscript's origin, authorship, or submission outcome.
- The audit's §5 action list ("settle provenance," "force the Stage C angle decision," "ask the compute question," "raise authorship") shows no evidence of having been started.

That audit was written *before* the Stage B campaign ran and explicitly advised against running it at full size pending the provenance answer (see F5). The campaign ran anyway and produced good results — that is not a bad outcome, and the sunk cost is modest. But **D1 remains the single highest-leverage open question in the project, and it is answerable in one conversation.** Everything about how Stage B is framed, titled, and eventually submitted branches on it.

I did not re-run the provenance search this pass and take that audit's negative result as given; it is a strong negative, not a confirmed one, and its own §6 says so.

**Fix:** ask the mentor. Until answered, at minimum soften the theory doc's title and add a provenance line to §7 — "published" is currently an unverified factual assertion in the title of the project's flagship artifact.

---

### F4 — Medium — `publication_roadmap.md` is ~2 stages stale and still says "Execute Stage A"

The roadmap is explicitly designed as "a single entry point for 'what's left to do to genuinely publish this'." Its mtime is 2026-07-23 23:38 — it predates all of Stage B.

**§1 "What's missing for a submission"** still lists as outstanding, all of which are now done:

| Roadmap says missing | Actual state |
|---|---|
| "only the CIC-IDS-2017 γ-sweep (n=3) is persisted" | All four base sweeps persisted, Stage A gate green |
| "The n=5/7 sweeps and the UNSW-NB15 γ-sweep are runnable but not yet run" | Both run and committed |
| "The combined fix has only been seed-swept on CIC, not UNSW" | UNSW band exists |
| "No formal statistical testing (only 3-seed variance bands)" | 10-seed bootstrap CIs + paired *t* + Wilcoxon + Holm |
| "No ablation table packaging the projection-vs-η′ isolation tests" | Full 2×2 factorial, both datasets |
| "No corrected convergence theory for the fixed algorithm" | `fedgraa_convergence_defect.md` §4 |

**§6 "Immediate next actions"** still opens *"Stage A is now fully specified in `stage_a_plan.md`… Execute Stage A in this order"* and enumerates six Stage A steps.

This was already flagged by `STAGE_DIRECTION_ASSESSMENT.md` §3 (01:00 today) with a "~20–30 min mechanical" fix estimate. It was not done. It is the document a returning session or the mentor reads first, and it currently understates the project by two complete stages.

---

### F5 — Medium — The two newest strategy docs are stale in their central premise and give now-wrong instructions

Established chronology (git log + file mtimes):

| Time | Event |
|---|---|
| 01:00 | `STAGE_DIRECTION_ASSESSMENT.md` written — "Stage B: one compute task remaining" |
| 01:04 | `RESEARCH_DIRECTION_AUDIT_2026-07-25.md` written — "Stage B campaign unrun" |
| 01:37–02:49 | The four baseline seed-sweeps run |
| 02:55–02:59 | Ablation manifest, `stage_b_plan.md`, `STAGE_B_REMAINING_TASKS.md` updated to "complete" |
| 03:01 | Commit `4459f53` |
| 04:58–05:03 | Blowup median/max correction; tables rebuilt; theory doc and Stage B capstone finalized |

Both strategy docs were overtaken within ninety minutes of being written. Neither carries a superseded banner — unlike `STAGE_B_REMAINING_TASKS.md`, which correctly got one.

The consequence is not cosmetic. `RESEARCH_DIRECTION_AUDIT_2026-07-25.md` §5 action 5 reads:

> "**Cut the Stage B campaign to what a paper section actually needs.** Do not run 71 fresh (arm, seed) cells… most likely 4 arms × 2 datasets × 3 γ at **5 seeds**, not 10 — then run exactly that."

Acting on that today would mean discarding a completed 10-seed campaign to rebuild it at 5 seeds. Anyone reading these two docs cold — including the mentor — would form a materially wrong picture of the project's state.

**Fix:** add a dated superseded/partially-superseded banner to both, in the style `STAGE_B_REMAINING_TASKS.md` already uses. `RESEARCH_DIRECTION_AUDIT`'s D1/D2/D3/D4/D6 findings remain live and valuable; only its Stage-B-status premise and action 5 are overtaken. Say exactly that, so the still-valid findings are not discarded along with the stale ones.

---

### F6 — Medium — 37 broken relative links across 63 markdown files

Scanned all 347 relative markdown links and image references in every project `.md`. **37 do not resolve.** Neither completeness gate checks documentation links — both only validate `results/…` paths.

| Root cause | Count | Detail |
|---|---|---|
| `docs/current/` → `docs/research_c/` move (uncommitted) | 11 | Moved docs' own outbound links to `fedgraa_convergence_defect.md`, `publication_roadmap.md`, `stage_b_plan.md` now resolve inside `research_c/`. Affects `STAGE_C_LITERATURE_MASTER.md` (6), `STAGE_C_DEEP_RESEARCH_2026-07-25.md` (3) |
| `FedGRAA_Simulated_Peer_Review.md` placed in `docs/paper/` | 11 | Links to `PAPER_AMBIGUITY_REPORT.md`, `fedgraa_convergence_defect.md`, `publication_roadmap.md` as siblings; they live in `docs/current/` |
| Top-level docs pointing at pre-reorganization paths | 5 | `README.md` → `docs/strict_reproduction_report.md` (now `docs/archive/`); `PAKDD_STATUS_REPORT.md` → `docs/fedgraa_divergence_diagnosis.md` (now `docs/current/`) |
| `PAKDD_Final_Version_Submit%20(3).pdf` | 5 | Unescaped `)` in the filename truncates the link target at the first paren. Affects `REPRODUCTION_AUDIT.md`, `PAPER_AMBIGUITY_REPORT.md`, and 3 others. Renders broken in any standard markdown viewer |
| `scripts/archive/gpu/README.md` → pre-reorganization `docs/` paths | 2 | Archived; low priority |

Most of these are direct fallout from the in-flight doc reorganization (F1) and will need fixing as part of committing it. The `%20(3` cases are independent and predate it — escaping to `PAKDD_Final_Version_Submit%20%283%29.pdf`, or simply renaming the file to drop the parentheses, fixes all five.

**Optional hardening:** the link checker used for this audit is ~30 lines; adding it to the completeness gates (or a pre-commit hook) would make this class of breakage impossible to reintroduce.

---

### F7 — Low — `PAKDD_STATUS_REPORT.md` is stale

Dated 2026-07-23. Describes the divergence diagnosis as the project's current frontier; cites **79/79 tests** (now 139); presents `η′=0.001` as the headline intervention with its "collapsing overall accuracy instead" caveat, rather than the validated combined fix; and carries 2 of F6's broken links. `README.md` links to it as the current-state pointer, and `latest_baseline.md`'s (correct, well-written) superseded banner points readers to it as the authoritative current source.

By contrast, `latest_baseline.md` is a model of how to handle this — an explicit dated superseded banner naming the fixing commit. `PAKDD_STATUS_REPORT.md` needs either the same treatment or a refresh.

---

### F8 — Low — "byte-identical" is literally false for UNSW n=5 vs n=7

`results/README.md` and `STAGE_A_MASTER_REPORT.md` §7.2 both state the UNSW n=5 and n=7 rows are "byte-identical — confirmed directly against the JSON."

Verified precisely: **every metric curve is exactly identical** — `all_acc` and `last_n_acc`, all 5 methods, max elementwise difference exactly `0`. The JSON objects are *not* byte-identical: `history.n_rare` records `5` in one and `7` in the other, which is correct metadata behaviour.

The scientific claim is sound and important; only the word is over-strong. "Metrically identical (every curve exact to the bit; only the recorded `n_rare` label differs)" is both true and more informative — it shows the truncation happened at the right layer.

---

### F9 — Low — Stage A's capstone frames severity on seed-0 statistics that Stage B has since corrected

`STAGE_A_MASTER_REPORT.md` §5.8/§5.9 and its executive summary frame the cross-dataset severity story as *"CIC's worst was 193×; UNSW's worst, also γ=1, is 7.6×"* and *"blowup magnitude is an order of magnitude smaller."*

Those are correct **seed-0** figures and Stage A is honestly scoped to a 3-seed regime, so this is not an error in context. But `fedgraa_convergence_defect.md` §6 has since replaced that framing with the 10-seed reading — worst case 1.86×10⁶× (CIC γ=0) vs 46.85× (UNSW γ=1), roughly four orders apart on the tail, while the medians are much closer (CIC 2.71–14.19× vs UNSW 1.50–8.26×), and at γ=0 the CIC median (3.70×) is actually **lower** than UNSW's (5.10×).

A reader going A → B hits two different headline framings of the same phenomenon. One forward-reference line in Stage A §5.8 — "superseded at N=10 by `fedgraa_convergence_defect.md` §6" — closes it. Worth doing before the package goes to the mentor, since both capstones are intended to be read together.

---

## 3. What this audit did not cover

- **The provenance question (F3)** — not re-investigated; the prior audit's negative result is taken as given, with its own stated caveat that a strong negative is not a confirmed one.
- **The correctness of the corrected convergence proposition as mathematics.** I verified its empirical predictions (feasibility 60/60, blowup behaviour) and the correctness of the projection operator it relies on, and read the proof sketch for internal consistency. I did not independently re-derive the `O(1/T)` gradient-mapping rate or evaluate whether the Clarke-subdifferential treatment of `min(L_h, L_m)` would satisfy a referee. §4's own "rigor flag" already scopes those three items for the mentor, correctly.
- **The Stage C literature corpus** (3,818 lines across 8 docs) — its citation verification was audited separately on 2026-07-24/25 and is out of Stage A/B scope. That audit's own disclosure that ~140 carried-over bibliography entries were never independently re-verified remains an open exposure.
- **Re-running any training.** All experimental values were verified against persisted artifacts, not regenerated. Determinism across a re-run is asserted by pinned seeds, not tested here.
- **The 200-round base sweeps' internal trajectories** beyond final-round values, round-0 values, and the specific composite-loss figures quoted in the capstones.

---

## 4. Recommended order

1. **Commit the working tree** (F1) — two commits: the blowup correction with its tests and regenerated tables, then the doc reorganization plus the four untracked capstones. Until this is done, `HEAD` misrepresents the project's own best numbers.
2. **Settle provenance with the mentor** (F3) — one conversation; it gates how Stage B is framed and titled. Fold in the Stage C angle decision, the compute question, and authorship, per the prior audit's §5.
3. **Add the all-class-accuracy comparisons** (F2) — ~30 lines of builder code, no re-training, and it converts the project's central recommendation from a robustness argument into a statistically significant one on both datasets.
4. **Refresh `publication_roadmap.md`** (F4) and **banner the two stale strategy docs** (F5) — mechanical, ~45 min total, and they are what a mentor reads first.
5. **Fix the 37 broken links** (F6) as part of committing the reorganization; escape or rename the manuscript PDF; consider adding the link check to the completeness gates.
6. **F7–F9** — refresh or banner `PAKDD_STATUS_REPORT.md`; soften "byte-identical"; add the Stage A → Stage B forward-reference.

---

## 5. Closing assessment

The instinct behind this request — "are we actually good, or does it just look good?" — is the right one to have, and in this case the answer is that it is good. I went looking for drifted numbers, unsupported claims, statistical shortcuts, and gates that pass without checking anything, and found none. The ablation reproduces exactly from raw data under an independent implementation. The two structural claims hold at 60/60 and 60/60. The projection operator is correct to float32. Stage A's ~90 reported values reconcile to the digit, including a loss trace quoted to two decimals across eleven rounds.

More telling than any individual check: the project caught its own heavy-tailed-blowup problem and corrected the reporting rather than shipping a mean that flattered the narrative, and it published the refutation of its own n=3 hypothesis as a headline finding. Those are the habits that make the rest of the numbers believable.

The gap is that **the work has outrun its bookkeeping**. The best version of this project currently exists only in an uncommitted working tree, described by an entry-point document that thinks Stage A hasn't started, alongside two strategy documents that think Stage B hasn't run. That is a genuinely fixable, half-day problem — and it is worth fixing before the package goes to the mentor, because right now the repository undersells the work in it.

The one substantive scientific item is F2, and it is an opportunity rather than a defect: the missing all-class analysis is the strongest argument the project has for its own recommended fix, and it costs no compute to add.

---

*Verification performed this pass: `pytest tests` (139 passed); `check_stage_a_complete.py` and `check_stage_b_complete.py` (both exit 0); independent recomputation of 240 diagnosis blocks, 72 cell means, 72 bootstrap CIs, 30 paired-t, 30 Wilcoxon, 30 Cohen's d, 30 Holm adjustments, and 18 factorial effects via SciPy; re-derivation of ~90 Stage A values from `results/*.json`; round-0 convention check across 30 curves; UNSW n=5/n=7 elementwise curve comparison; `project_to_simplex` validated against SLSQP over 300 random vectors plus 4 edge cases; degenerate-path firing analysis on `fedgraa/stats.py`; relative-link resolution across 347 links in 63 markdown files; `git status`/`git diff`/`git log` chronology reconstruction.*
