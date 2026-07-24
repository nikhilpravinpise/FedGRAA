# FedGRAA / PAKDD — Stage B Master Report

### Packaging the divergence diagnosis as a rigorous, citable component: the corrected convergence theory, the statistically-tested ablation, and the anchor contribution ready to hand to the PhD mentor.

**Status:** Stage B ✅ **complete.** **Baseline commit:** `a17a443` (Stage B plan and the completed data campaign were both written/run against this tree). **This document compiled:** 2026-07-25. `python scripts/check_stage_b_complete.py` exits `0`; `python -m pytest tests` reports **139/139 passed**.

**What this document is.** The capstone record of Stage B, mirroring [`STAGE_A_MASTER_REPORT.md`](STAGE_A_MASTER_REPORT.md)'s structure: an executive summary, the theory recap (defect → why the paper's proof fails → corrected proposition), the finalized 10-seed ablation table's headline numbers, test-suite/completeness status, and the Stage C handoff. It supersedes no other document — [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) remains the authoritative, full-detail theory-and-data section — but a reader who opens only this file gets the complete Stage B story.

**How to navigate this document:**

1. [Executive summary](#1-executive-summary)
2. [Recap — the defect, why the proof fails, the corrected proposition](#2-recap--the-defect-why-the-proof-fails-the-corrected-proposition)
3. [The completed ablation campaign — what was run](#3-the-completed-ablation-campaign--what-was-run)
4. [Headline results — CIC-IDS-2017](#4-headline-results--cic-ids-2017)
5. [Headline results — UNSW-NB15](#5-headline-results--unsw-nb15)
6. [Cross-dataset synthesis — the honest, dataset-dependent finding](#6-cross-dataset-synthesis--the-honest-dataset-dependent-finding)
7. [Statistics module & verification](#7-statistics-module--verification)
8. [Test suite & completeness verification](#8-test-suite--completeness-verification)
9. [Artifact index](#9-artifact-index)
10. [Honest limitations (recap)](#10-honest-limitations-recap)
11. [Stage C handoff](#11-stage-c-handoff)

---

## 1. Executive summary

Stage A ([`STAGE_A_MASTER_REPORT.md`](STAGE_A_MASTER_REPORT.md)) established, empirically, that the FedGRAA paper's Algorithm 2 lets its aggregation-weight vector `w` leave the probability simplex in round 1 of training, at every tested `γ`, on both datasets — a direct, measured violation of the paper's own Lemma 1 convergence guarantee. Stage B's job was to turn that empirical diagnosis into a **citable, self-contained paper section**: formal theory (why the defect happens, why the paper's own proof doesn't catch it, and a corrected guarantee for a fixed algorithm) plus a statistically rigorous ablation (bootstrap CIs, paired significance tests, a 2×2 factorial) quantifying the fix — the "fallback/anchor contribution, real even if Stage C stalls or runs long" per [`publication_roadmap.md`](publication_roadmap.md) §2.

That goal is now fully met. [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) states the feasibility-set defect (§2), a three-part proof sketch of why Appendix A.1 doesn't cover it (§3), and a corrected convergence proposition — projected (sub)gradient descent on the simplex, converging to a stationary/KKT point rather than the paper's claimed global minimum (§4). The empirical half is a **complete 10-seed × 4-arm × 2-dataset factorial campaign**: 80 (arm, seed) diagnostic runs — 71 newly executed for Stage B, 9 reused from Stage-A-era files whose arm/config/seed already matched — plus 4 baseline seed-sweeps, consolidated through a new statistics module ([`../../fedgraa/stats.py`](../../fedgraa/stats.py)) into bootstrap-CI'd tables with 5 pre-registered pairwise significance tests and a factorial main-effect/interaction read, per dataset per γ (§5 of the theory doc; §3–6 below).

**The campaign's central structural finding held with zero exceptions at full scale:** across all 60 `both`-arm cells (10 seeds × 3 γ × 2 datasets), the projected fix's weight vector satisfies `Σw = 1.0000` (to float32 precision) and `min(w) ≥ 0` on **every single cell** — the round-1 blowup is not merely rare under the fix, it is structurally impossible, exactly as §4's proposition predicts.

**The campaign's central empirical surprise — and the reason a real 10-seed factorial was worth running rather than trusting the n=3 preview** — is that the two levers of the fix (simplex projection, and the smaller η′) are **not equally load-bearing on both datasets**. On CIC-IDS-2017, the η′ cut alone already recovers most of the rare-class-accuracy gain (statistically tying the full combined fix at γ=0.5–1.0). On UNSW-NB15, the pattern inverts: projection alone statistically ties the combined fix at every γ, while the η′ cut contributes nothing significant. Only the **combined fix** performs well on both datasets at every γ — which is why it remains the recommendation — but the factorial decomposition is a more honest and more interesting finding than the n=3 preview's "both levers are always necessary" hypothesis, and Stage B reports it as such (§6).

**Headline numbers** (rare-class accuracy, mean [95% bootstrap CI], N=10 seeds, 60-round horizon):

| | CIC-IDS-2017 | UNSW-NB15 |
|---|---|---|
| FedAvg (baseline) | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| Static Reparam. (baseline) | 84.4% [77.2, 91.4] | 48.3% [40.8, 56.0] |
| Literal (paper default, η′=0.1, no proj.), γ=1 | 10.0% [0.0, 30.0] | 27.6% [15.0, 39.9] |
| **Both** (the fix), γ=1 | **89.3% [83.6, 94.6]** | **69.2% [63.4, 73.8]** |

Stage B does not include a new method, baselines beyond FedAvg/Static, scalability experiments, or a fully machine-checked proof — those remain Stage C (§11).

---

## 2. Recap — the defect, why the proof fails, the corrected proposition

*(Full derivations: [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) §1–4. Summarized here for a self-contained capstone.)*

**The defect (§2 of the theory doc).** Algorithm 2 updates `w ← w − η'∇_w L'(w)` with no projection, clipping, or renormalization step — "Algorithm 2 has exactly 8 lines. There is no line 9." On real data, `w` leaves the feasible simplex `Δ = {w : w_k ≥ 0, Σw_k = 1}` at `first_simplex_violation_round = 1`, at every tested γ ∈ {0, 0.5, 1.0}, on both datasets, with zero exceptions. Off `Δ`, the aggregated model `θ_g = Σ_k w_k θ_k` becomes an unbounded affine extrapolation rather than a convex combination, and the paper's own Lemma 1 descent guarantee (`L'(w^t) ≤ L'(w^{t-1}) − (η'−β(η')²/2)‖∇L'‖²`) is violated immediately — at γ=1 on CIC-IDS-2017, composite loss goes `2.98 → 319.11` within 8 rounds.

**Why Appendix A.1's proof doesn't cover this (§3 of the theory doc) — three compounding gaps:**
1. **Assumption 1 (`L'` convex and β-smooth) is false** for the paper's own tanh-MLP (200→100→50) — a standard non-convex network (paper ambiguity report item #13).
2. **`L_a = min(L_h, L_m)` is non-smooth exactly where training pushes it** — the optimization is actively driven toward the crossing boundary `{L_h=L_m}` by Eq. 8's own design, so no finite β-smoothness constant exists there, and Assumption 3's step-size condition `η' ≤ 1/β` is either vacuous or arbitrarily restrictive (item #6).
3. **Even granting smoothness by fiat, Lemma 2 tacitly assumes `w ∈ Δ`** — an assumption Assumption 2 (bounded gradients) never supplies and Algorithm 2 never enforces. This is the gap most directly responsible for the specific §2 failure mode.

**The corrected proposition (§4 of the theory doc).** Replacing line 6 with `w^t = Π_Δ(w^{t-1} − η'∂L'(w^{t-1}))` (Euclidean projection onto the simplex) turns the weight-learner into projected (sub)gradient descent on a compact convex set. This yields two honest guarantees, stated explicitly weaker than the paper's Theorem 1:
1. **Feasibility is structural, not empirical** — `w^t ∈ Δ` for every `t` by construction, so the round-1 blowup is *provably impossible*, not merely unobserved in a particular run.
2. **Stationarity, not global optimality** — because `L'` is non-convex, the method is shown (via the standard projected-gradient-mapping argument, non-expansive projection + bounded subgradients on a compact set) to reach a stationary/KKT point at rate `O(1/T)`, not the paper's claimed global minimum.

The proposition explicitly flags three items left for hardening with the PhD mentor (an exact `β_eff` estimate, a fully formal `min(L_h,L_m)` sub-differential treatment, and a worked-out rate constant) — consistent with this project's standing no-silent-patching mandate. §3–6 below report the completed empirical test of both guarantees at full statistical power.

---

## 3. The completed ablation campaign — what was run

Per [`stage_b_plan.md`](stage_b_plan.md) §3 and the runbook in (formerly) [`STAGE_B_REMAINING_TASKS.md`](STAGE_B_REMAINING_TASKS.md), the campaign covers the 2×2 factorial (projection on/off × η′=0.1/0.01) plus FedAvg/Static references, at N=10 seeds, on both datasets, all at the fixed 60-round/eval-every-1 diagnostic horizon:

| Arm | What it isolates | CIC-IDS-2017 | UNSW-NB15 |
|---|---|---|---|
| **Literal** | paper default (η′=0.1, no projection) | 10 seeds | 10 seeds |
| **Projection-alone** | projection on, η′ still 0.1 | 10 seeds | 10 seeds |
| **η′-alone (canonical)** | η′=0.01/E′=5, no projection | 10 seeds | 10 seeds |
| **Both (the fix)** | projection + η′=0.01/E′=5 | 10 seeds | 10 seeds |
| FedAvg (reference) | data-size-weighted average, γ-independent | 10 seeds | 10 seeds |
| Static Reparam. (reference) | inverse-size weighting, γ-independent | 10 seeds | 10 seeds |

**Totals:** 80 (arm, seed) diagnostic runs (4 arms × 10 seeds × 2 datasets, each covering all 3 γ in one run), of which **71 were newly executed** for Stage B and **9 were reused** from Stage-A-era files whose arm, config, and seed already matched the factorial (the driver's resumability check recognizes them by filename convention and verifies the persisted config); plus 40 baseline seed-runs (2 methods × 2 datasets × 10 seeds). Every diagnostic JSON carries the `diagnosis` block (`final_weight_sum`, `final_weight_min`, `composite_loss_blowup_ratio`, `first_simplex_violation_round`, `monotone_lemma1`, `final_all_acc`, `final_last_n_acc`) that the table builder and completeness checker both consume directly.

**Consolidation.** [`../../scripts/build_ablation_table.py`](../../scripts/build_ablation_table.py) reads every per-seed JSON and the 4 baseline files, and — via [`../../fedgraa/stats.py`](../../fedgraa/stats.py) — computes, per (dataset, arm, γ): the mean and percentile-bootstrap 95% CI (B=10,000, fixed RNG seed) of rare-class accuracy, all-class accuracy, and composite-loss blowup; and, per (dataset, γ), the 5 pre-registered pairwise comparisons (**both** vs. each of literal / projection-alone / η′-alone / FedAvg / Static — paired *t*-test, Wilcoxon signed-rank, Cohen's *d*, all paired by seed, Holm–Bonferroni-corrected across the 5) plus the 2×2 factorial main-effect/interaction read. [`../../scripts/render_ablation.py`](../../scripts/render_ablation.py) renders the markdown table and a grouped-bar-chart figure per dataset. Full outputs are embedded in [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) §5.1–5.2 and reproduced in §4–5 below.

---

## 4. Headline results — CIC-IDS-2017

*(Full table with all pairwise-comparison p-values and the factorial read: [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) §5.1; raw data: [`../../results/ablation/CIC_IDS_2017_ablation_table.json`](../../results/ablation/CIC_IDS_2017_ablation_table.json).)*

| Arm | γ=0.0 rare-acc | γ=0.5 rare-acc | γ=1.0 rare-acc |
|---|---|---|---|
| FedAvg | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| Static Reparam. | 84.4% [77.2, 91.4] | 84.4% [77.2, 91.4] | 84.4% [77.2, 91.4] |
| Literal (η′=0.1) | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 10.0% [0.0, 30.0] |
| Projection-alone | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 51.0% [24.2, 77.1] |
| η′-alone (canonical) | 27.5% [12.5, 43.3] | 78.5% [62.6, 92.2] | 88.9% [68.9, 100.0] |
| **Both (the fix)** | **69.7% [61.1, 78.9]** | **81.1% [66.4, 92.2]** | **89.3% [83.6, 94.6]** |

`Both` beats every reference arm significantly (Holm p < 0.008) at γ=0.0; at γ=0.5–1.0 it still beats literal/FedAvg significantly, but is statistically indistinguishable from `η′-alone` (Holm p = 1.0 and 0.97) — the η′ cut is doing most of the accuracy work on this dataset. Feasibility (`Σw=1.0000`, `min(w)≥0`) holds on all 30 cells, and blowup is exactly 1.0× on 29/30 (the exception at 1.016×).

**The blowup column carries a finding the accuracy columns cannot.** The literal arm's blowup is heavy-tailed: at γ=0 its per-seed values are `[3.78, 7.59, 29.76, 2.58, 3.62, 3.59, 2.47, 3.14, 1857580.09, 3.87]` — median 3.70×, but **1.86×10⁶× on seed 8**. Stage B therefore reports blowup as median (and max) rather than mean throughout, since the mean (185,764×) describes no run that actually occurred. Only the `both` arm is flat at *every* seed and γ (median 1.00×, max 1.00×); `projection-alone` still reaches 5.60× and `η′-alone` still reaches 30.02× somewhere in the γ range. Feasibility is what makes the *unbounded* 1.86×10⁶ excursion structurally impossible; the η′ cut is what flattens the bounded residual.

---

## 5. Headline results — UNSW-NB15

*(Full table: [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) §5.2; raw data: [`../../results/ablation/UNSW_NB15_ablation_table.json`](../../results/ablation/UNSW_NB15_ablation_table.json).)*

| Arm | γ=0.0 rare-acc | γ=0.5 rare-acc | γ=1.0 rare-acc |
|---|---|---|---|
| FedAvg | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| Static Reparam. | 48.3% [40.8, 56.0] | 48.3% [40.8, 56.0] | 48.3% [40.8, 56.0] |
| Literal (η′=0.1) | 0.1% [0.0, 0.3] | 26.9% [16.8, 36.2] | 27.6% [15.0, 39.9] |
| Projection-alone | 6.4% [0.0, 16.0] | 47.7% [35.0, 57.6] | 67.2% [59.6, 73.6] |
| η′-alone (canonical) | 3.4% [0.4, 7.8] | 12.1% [5.2, 19.6] | 36.4% [33.7, 39.8] |
| **Both (the fix)** | 1.7% [0.0, 4.6] | **37.1% [28.4, 45.7]** | **69.2% [63.4, 73.8]** |

`Both` beats literal, FedAvg, and Static significantly at γ=1.0, and beats η′-alone significantly at γ=0.5 and γ=1.0 — but is statistically indistinguishable from `projection-alone` at **every** γ (Holm p = 0.61, 0.42, 0.54). On this dataset, projection is doing the great majority of the accuracy-recovery work; the η′ cut's main effect is negative or negligible throughout. Feasibility again holds with zero exceptions across all 30 `both`-arm cells; composite-loss blowup is reduced (median 1.00–1.32×, never exceeding 1.87× on any seed) but not flattened to exactly 1.0× the way CIC's is — consistent with the proposition's boundedness-not-flatness prediction (§4 of the theory doc). Checked per-seed rather than only on means, the fix's blowup is strictly below the literal algorithm's on the same seed at the same γ in **all 30 of 30** UNSW cells (literal reaches 46.85× at γ=1).

---

## 6. Cross-dataset synthesis — the honest, dataset-dependent finding

The n=3 preview data available before this campaign supported a simple hypothesis: "neither lever alone recovers rare-class accuracy; only the combination does." The full n=10 factorial **replaces that hypothesis with a sharper, more honest one**:

- **The projection fix is structurally necessary and dataset-independent** — it is what makes `Σw=1.0000`, `min(w)≥0` hold with zero exceptions on all 60 `both`-arm cells regardless of dataset (§4 of the theory doc's feasibility claim).
- **Which lever drives the *rare-class-accuracy recovery* is dataset-dependent.** On CIC-IDS-2017, `η′-alone` statistically ties `both` at γ≥0.5 — the η′ cut does most of the work. On UNSW-NB15, `projection-alone` statistically ties `both` at every γ — projection does most of the work. Neither single-lever arm dominates on the *other* dataset: `η′-alone` is markedly worse than `both`/`projection-alone` on UNSW (3.4–36.4% vs. 67–69%), and `projection-alone` is markedly worse than `both`/`η′-alone` on CIC at γ≤0.5 (0.0% vs. 70–89%).
- **`Both` is the only arm that performs well across both datasets and all three γ.** This is the practically relevant conclusion for anyone applying the fix without per-dataset retuning, even though the factorial decomposition shows the two levers are not equally load-bearing everywhere — a genuinely more nuanced and more defensible claim than the preview data could support, and exactly the kind of result a statistically rigorous n=10 factorial is for.

This finding, its full pairwise-comparison and factorial-effect evidence, and its implications are written up in [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) §5.3 and folded into §6's discussion and §7's honest limitations (the two-dataset caveat).

---

## 7. Statistics module & verification

[`../../fedgraa/stats.py`](../../fedgraa/stats.py) (new this stage) implements, pure NumPy/SciPy:

- `bootstrap_ci` — percentile bootstrap, B=10,000, fixed RNG seed; returns a degenerate point CI (not `NaN`) on a constant vector.
- `paired_ttest`, `wilcoxon_signed_rank` — both zero-diff-safe; a paired *t*-test on identical paired differences returns `(0.0, 1.0)` rather than a `0/0` `NaN`.
- `cohens_d_paired` — 0.0 on constant-zero diffs.
- `holm_bonferroni` — order-preserving adjusted p-values.

This module exists specifically because the ablation's real data has cells with **exactly zero variance across seeds** (e.g. FedAvg's rare-acc is 0.0% on every single seed, every dataset, every γ — a genuine, correctly-handled degenerate case visible directly in §4–5's tables above, not a bug). `tests/test_stats.py` (19 cases) checks every degenerate-cell edge against hand-computed references and SciPy where applicable; `tests/test_build_ablation_table.py` (13 cases) checks the wiring from raw per-seed dicts through to table cells and comparison rows; `tests/test_render_ablation.py` (5 cases) smoke-tests the markdown/figure renderer.

---

## 8. Test suite & completeness verification

```
.venv313/Scripts/python.exe -m pytest tests --tb=short        → 142 passed
.venv313/Scripts/python.exe scripts/check_stage_b_complete.py → exit 0   (completeness)
.venv313/Scripts/python.exe scripts/audit_stage_b.py          → exit 0   (correctness)
```

`check_stage_b_complete.py` verifies **presence and shape**: all 4 arms × 10 seeds × 2 datasets of per-seed diagnostic JSONs carry complete `diagnosis` blocks; both consolidated `*_ablation_table.json` files load with every arm×γ cell, all 5 comparisons, and the factorial read; both rendered `.md`/`.png` pairs exist and are non-empty; the theory doc contains every required section; and every `results/...` path referenced in [`../../results/ablation/README.md`](../../results/ablation/README.md) exists on disk and is git-tracked.

[`../../scripts/audit_stage_b.py`](../../scripts/audit_stage_b.py) verifies **correctness** — the harder question, and a check a passing completeness script cannot answer:

- **Raw-run integrity.** All 80 diagnostic JSONs are byte-distinct (no run duplicated and filed as two seeds) and each one's *persisted config* matches the arm it is filed under — projection flag, η′, E′, seed, rounds, γ, across all 240 cells.
- **Table-vs-raw fidelity.** The consolidated table's `per_seed` arrays are re-read from the raw JSONs and compared exactly, catching any globbing or seed-mapping error in the builder.
- **Independent statistics.** Every p-value, Cohen's *d*, Holm adjustment, and factorial effect is recomputed from scratch against SciPy, deliberately *not* through `fedgraa/stats.py`, so the module is checked against an independent implementation rather than itself.
- **Doc transcription.** All **232** numbers hand-typed into the theory doc's tables are parsed back out and re-derived from the JSON, including that every significance star agrees with its own numeric p-value.
- **Presentation soundness.** Flags any cell where a mean is a misleading summary of a heavy-tailed ratio.

**This audit was run adversarially against the first committed version of Stage B and found three real defects, all since fixed** (§7.1). It is negative-control tested: deliberately injected errors — a wrong transcribed number, a corrupted p-value, a duplicated run file — are each caught and named precisely.

### 7.1 What the audit caught, and what changed

| Defect found | Why it mattered | Fix |
|---|---|---|
| **Blowup reported as a heavy-tailed mean.** CIC γ=0 literal per-seed blowups are `[3.78, 7.59, 29.76, 2.58, 3.62, 3.59, 2.47, 3.14, 1857580.09, 3.87]`; the table showed the mean, **185,764×** | The mean described no run that actually happened — one diverging seed set it. A reviewer checking the per-seed array would have caught it | Blowup is now reported as **median (and max)** everywhere; `_cell()` carries `median`/`min`/`max`; two regression tests lock it in. The corrected presentation is also *more* striking: "3.70× med, 1.86×10⁶× max" |
| **Stale seed-0 worst-case figures presented as the campaign's.** §6 cited "CIC 193× vs UNSW 7.6×, an order of magnitude apart" — both seed-0 values | With 10 seeds the true maxima are 1.86×10⁶× and 46.85× — about *four* orders apart. The qualitative claim held but the numbers understated it | §6 now reports the 10-seed maxima and medians separately, and distinguishes "typical run" from "tail severity" |
| **Run-count overstatement.** The report said "80 fresh diagnostic runs" | 71 were newly executed; 9 were reused Stage-A-era files. Overstating fresh compute is a reproducibility claim, not a rounding error | Corrected to "80 (arm, seed) runs — 71 newly executed, 9 reused" with the reuse mechanism explained |
| *(found by negative control)* **The audit script itself crashed while reporting a failure** — `UnicodeEncodeError` on `η′` under a cp1252 console | An auditor that dies while naming a defect is worse than none: it exits non-zero with no readable reason | `sys.stdout` reconfigured to UTF-8 with `errors="replace"` |

Two things the audit **confirmed rather than corrected**, worth recording because they were the claims most at risk: the feasibility result (`Σw=1`, `min(w)≥0` on all 60 `both`-arm cells) is exactly as stated, and `first_simplex_violation_round = 1` holds on **all 60 literal-arm cells** — the round-1 defect is unconditional at n=10, not a seed-0 artifact.

---

## 9. Artifact index

| Artifact | Role |
|---|---|
| [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) | The standalone section — theory (§1–4), ablation (§5), discussion/limitations (§6–7) |
| [`../../fedgraa/stats.py`](../../fedgraa/stats.py) | Bootstrap CI, paired tests, Cohen's d, Holm-Bonferroni (T4) |
| [`../../scripts/run_ablation_seed_sweep.py`](../../scripts/run_ablation_seed_sweep.py) | Resumable, worker-capped campaign driver (T3 infra) |
| [`../../scripts/build_ablation_table.py`](../../scripts/build_ablation_table.py) | Consolidates per-seed JSONs → table JSON (T5) |
| [`../../scripts/render_ablation.py`](../../scripts/render_ablation.py) | Table JSON → markdown + figure (T6) |
| [`../../scripts/check_stage_b_complete.py`](../../scripts/check_stage_b_complete.py) | Single green/red completeness signal (§7) |
| [`../../results/ablation/README.md`](../../results/ablation/README.md) | Manifest: every artifact → generating command, seeds, claim supported |
| `results/ablation/{CIC_IDS_2017,UNSW_NB15}_ablation_table.{json,md}`, `_ablation.png` | The consolidated, rendered ablation (6 files) |
| `results/ablation/seed_sweep_{CIC_IDS_2017,UNSW_NB15}_{fedavg,static}_r60.json` | 60-round baseline reference bands (4 files) |
| `results/diagnostics/{CIC_IDS_2017,UNSW_NB15}_{literal,projected,literal_etaonly_eta0.01_E5,projected_combined_eta0.01_E5}[_seed*]_r60.json` | 80 per-seed diagnostic runs (4 arms × 10 seeds × 2 datasets) |
| `tests/test_stats.py`, `test_run_ablation_seed_sweep.py`, `test_build_ablation_table.py`, `test_render_ablation.py` | 54 new tests (139 total in the full suite) |

---

## 10. Honest limitations (recap)

*(Full text: [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) §7.)*

- **Stationarity, not global optimality** — the corrected proposition claims strictly less than the paper's Theorem 1, by design, because the loss is non-convex.
- **The fix is diagnostic-only** — `ProjectedFedGRAA` lives solely in `scripts/diagnose_fedgraa.py`; the strict-reproduction default in `fedgraa/` is unchanged.
- **N=10 seeds is still a small sample** — several comparisons are not significant after Holm correction despite visually large point-estimate gaps (e.g. CIC γ=1.0 `both` vs. `projection-alone`, Holm p=0.063, d=0.88); both parametric and non-parametric tests are reported for exactly this reason.
- **CPU-only training** throughout, Stage A and B alike.
- **The `min(L_h,L_m)` sub-gradient treatment is a standard-practice sketch**, not a from-scratch formal derivation — flagged for the PhD mentor.
- **Which lever drives accuracy recovery is dataset-dependent, and only two datasets were tested** (§6 above) — with only two data points, it is not possible to say which pattern is more typical of real IDS data in general.
- **This is a critique of one paper's one algorithm**, not a general claim about learned-aggregation-weight FL schemes.

---

## 11. Stage C handoff

Per [`publication_roadmap.md`](publication_roadmap.md) §6.6, the next action is to bring the combined Stage A + Stage B package to the PhD mentor to jointly scope Stage C's novelty angle and target venue. Stage B stands on its own as the anchor contribution in the meantime — a complete, statistically rigorous, citable "why this doesn't converge, and how to fix it" section, independent of whether Stage C's new-method work proceeds.

**Explicitly out of Stage B's scope, deferred to Stage C** (unchanged from [`stage_b_plan.md`](stage_b_plan.md) §9):
- A new method (few-shot/prototype-guided hybrid, or the privacy-compliant redesign).
- Baselines beyond FedAvg/Static (FedProx/FedOpt/FedSND implemented-but-unrun; AFL/q-FedAvg/FedAdp unimplemented).
- Scalability (50–200 clients) and extreme non-IID experiments.
- A fully machine-checked proof with exact β constants and a formal `min()` sub-differential treatment.
- Any change to the strict-reproduction default in `fedgraa/`.

**What Stage C inherits, ready-made:** a formally stated defect and corrected convergence guarantee that needs only constant-hardening, not re-derivation; a statistics module reusable for any future ablation (Stage C's new method will need the same bootstrap/paired-test/factorial machinery); and a fully verified data pipeline and completeness-check pattern (`check_stage_*_complete.py`) that a Stage C plan can extend rather than reinvent.
