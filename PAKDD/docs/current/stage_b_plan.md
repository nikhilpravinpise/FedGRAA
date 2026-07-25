# Stage B Execution Plan — Package the Critique as a Rigorous, Citable Component

**Parent:** [`publication_roadmap.md`](publication_roadmap.md) §2 (Stage B).
**Status:** ✅ **complete.** Builds directly on the completed Stage A artifacts (commit `be4ef03`): the persisted diagnostic JSONs, the combined-fix seed bands, and the `results/` manifest. This plan was written against commit `a17a443`. See [`STAGE_B_MASTER_REPORT.md`](STAGE_B_MASTER_REPORT.md) for the capstone summary of the completed campaign and [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) for the full theory + populated ablation section this plan specifies. *(This plan's target framing below quotes the manuscript as "published" — that was the working assumption at the time this was written. It is now an open question; see [`STAGE_B_MASTER_REPORT.md`](STAGE_B_MASTER_REPORT.md) §21. Retained verbatim as the historical record of what was planned and executed.)*
**Scope decisions (locked):** **full theoretical closure** (defect theorem + corrected proposition, both with proof sketches) and **N = 10 seeds** for significance testing.
**Baseline commit:** `a17a443`.

---

## 0. What "done" means (definition of done)

> **Stage-B goal (verbatim from the roadmap):** *"turn the diagnosis from 'internal audit trail' into something that reads as a paper section on its own — this is the fallback/anchor contribution, real even if Stage C stalls or runs long."*

Concretely, Stage B is **done** when all of the following are true and committed:

1. A standalone formal doc — [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) — states, in paper-section prose: **(a)** the feasibility-set defect (no simplex projection → weight vector `w` leaves the simplex `Δ` → the aggregated model `θ_g` becomes unbounded extrapolation → Lemma 1's descent guarantee fails at round 1), **(b)** a proof sketch of *why* the paper's Appendix-A.1 proof doesn't cover this (Assumption 1 convexity false for the tanh-MLP; β-smoothness false for `min(L_h, L_m)`), and **(c)** a **corrected convergence proposition** for the projected algorithm (projected subgradient descent on `Δ` → stationary/KKT point; `L′` non-increasing).
2. A single **consolidated ablation table** (data JSON + rendered markdown + figure) covers the **2×2 factorial** — literal / projection-alone / η′-alone / both — plus the **FedAvg** and **Static-Reparam** reference rows, per γ ∈ {0, 0.5, 1}, on **both** datasets, each cell reporting **mean + bootstrap 95 % CI over N = 10 seeds**.
3. **Significance testing is real:** bootstrap CIs on every cell, plus the pre-registered pairwise comparisons (paired *t* + Wilcoxon signed-rank, paired by seed, with Cohen's *d* and Holm–Bonferroni correction), plus the factorial main-effects/interaction read. Exact *p*-values persisted, not just stars.
4. A new **statistics module** [`../../fedgraa/stats.py`](../../fedgraa/stats.py) implements all of the above with unit tests against SciPy/NumPy references; the full `pytest tests` suite stays green.
5. The **projection-alone** and **canonical η′-alone** arms — currently single-seed / not-yet-run — are generated across the full 10-seed set on both datasets and persisted, so **every** ablation cell is significance-testable.
6. A single command — `python scripts/check_stage_b_complete.py` (new, §7) — exits `0`, proving the theory-doc + ablation + stats artifact set is complete on any clean checkout.
7. A manifest — [`../../results/ablation/README.md`](../../results/ablation/README.md) — maps every new artifact → exact command, git SHA, seed set, and the claim it supports (same backticked-path convention Stage A's manifest uses).
8. The standalone section reads end-to-end as **"Why a published FL-IDS aggregation algorithm doesn't converge, and how to fix it,"** ready to hand to the PhD mentor alongside the Stage A package (roadmap §6.6). A short capstone `STAGE_B_MASTER_REPORT.md` records completion (mirroring `STAGE_A_MASTER_REPORT.md`).

Stage B does **not** include: a new method (few-shot/prototype hybrid or the privacy redesign), baselines beyond FedAvg/Static (FedProx/FedOpt/FedSND are implemented but unrun; AFL/q-FedAvg/FedAdp are not implemented), scalability/non-IID experiments, or a fully machine-checked proof with exact constants. Those are Stage C and are explicitly out of scope (§9).

---

## 1. Grounding — what this plan verified in the code

This plan is written against the tree, not the roadmap prose. Key facts established by reading the code and the persisted results:

| Question the roadmap left open | Answer (verified) | Evidence |
|---|---|---|
| Does any significance-testing / bootstrap / *t*-test code already exist? | **No — build from scratch.** A repo-wide search for `scipy.stats` / `bootstrap` / `ttest` / `wilcoxon` / `percentile` matched only doc prose; both seed-sweep tools compute `np.mean`/`np.std` only. | grep across `*.py`; [`../../scripts/seed_sweep_combined_fix.py`](../../scripts/seed_sweep_combined_fix.py) L47–69 |
| Is SciPy available? | **Yes, transitively** (via `scikit-learn≥1.3`), installed in `.venv313`, but **not a direct dependency** — `requirements.txt` doesn't list it. Add `scipy` explicitly (§6.1). | `.venv313\Lib\site-packages\scipy\`; [`../../requirements.txt`](../../requirements.txt) |
| Do the four ablation arms already have a runner? | **Yes.** [`../../scripts/diagnose_fedgraa.py`](../../scripts/diagnose_fedgraa.py) `run_fine_grained(...)` covers every arm via flags — no new trainer needed. `ProjectedFedGRAA` + `project_to_simplex` (Duchi 2008) live there. | `diagnose_fedgraa.py` `run_fine_grained`, `analyze_history`, `ProjectedFedGRAA` |
| Which arms lack multi-seed data? | **projection-alone** (CIC seed-0 only; UNSW none) and **η′-alone** (only the off-spec η′=0.001/E′=1 file). The **both** arm and FedAvg/Static have 3 seeds. So Stage B re-runs proj-alone + a *new canonical* η′-alone arm and extends all arms to 10. | `results/diagnostics/` listing |
| What exact fields does the table read per run? | `diagnosis.final_all_acc`, `diagnosis.final_last_n_acc`, `diagnosis.composite_loss_blowup_ratio`, plus `first_simplex_violation_round`, `first_monotonicity_violation_round`, `final_weight_sum`, `final_weight_min`. | `diagnose_fedgraa.py` `analyze_history`; `seed_sweep_combined_fix.py` `aggregate` |
| Is the existing aggregator reusable? | **Yes.** `seed_sweep_combined_fix.py::aggregate` is dataset-general mean/std. The Stage B builder reuses its math and adds CI + *p*-value columns for **all four** arms (not just "both"). | `seed_sweep_combined_fix.py` L47–69 |
| What is the rare-class metric, exactly? | `last_n_class_accuracy` = **mean per-class recall** over the *n* rare classes ([`../../fedgraa/metrics.py`](../../fedgraa/metrics.py)). This is the table's metric of record. | `metrics.py` `last_n_class_accuracy` |
| What does the proof scaffolding provide? | **Theorem 1** (`L′→` global min), **Assumptions 1–4** (1 = convex + β-smooth; 3 = `η′ ≤ 1/β`), **Lemma 1** descent inequality `L′(wᵗ) ≤ L′(wᵗ⁻¹) − (η′ − β(η′)²/2)‖∇L′‖²`, **Lemma 2** (`θ_g` = converging convex combo). | [`../../understanding/05_convergence_proof.md`](../../understanding/05_convergence_proof.md); paper Appendix A.1 in [`../paper/`](../paper/) |
| Which paper items ground the defect? | **#13** (Assumption 1 "L′ convex and β-smooth" vs. the §5 tanh-MLP 200→100→50), **#6** (`L_a = min(L_h, L_m)` non-differentiable at the branch the optimizer approaches → breaks β-smoothness), **#1/#2** (Adam-vs-plain-GD notation — the descent bound assumes plain GD). | [`PAPER_AMBIGUITY_REPORT.md`](PAPER_AMBIGUITY_REPORT.md) #6, #13, #1/#2 |

---

## 2. The critique-packaging matrix — what exists, what's missing

Legend: ✅ done & committed · ❌ missing · ⚠️ exists but single-seed / scattered / off-spec.

| Deliverable | Scope | Persisted counterpart (expected path) | Status | Task |
|---|---|---|---|---|
| Formal defect statement (feasibility argument) | theory | `docs/current/fedgraa_convergence_defect.md` §1–2 | ❌ (scattered in diagnosis §2) | **T1** |
| Proof sketch — *why the paper's proof fails* | theory | same doc §3 | ❌ | **T1** |
| Corrected convergence proposition | theory | same doc §4 | ❌ | **T2** |
| Arm: **literal** (proj off, η′=0.1, E′=5) | CIC | `results/diagnostics/CIC_IDS_2017_literal_r60.json` | ⚠️ 1 seed → 10 | **T3** |
| Arm: **literal** | UNSW | `results/diagnostics/UNSW_NB15_literal_r60.json` | ⚠️ 1 seed → 10 | **T3** |
| Arm: **projection-alone** (proj on, η′=0.1) | CIC | `results/diagnostics/CIC_IDS_2017_projected_r60.json` | ⚠️ 1 seed → 10 | **T3** |
| Arm: **projection-alone** | UNSW | *(none)* | ❌ | **T3** |
| Arm: **η′-alone, canonical** (proj off, η′=0.01, E′=5) | both | *(new)* `{DS}_literal_etaonly_eta0.01_E5_r60.json` — caveat (a) | ❌ | **T3** |
| Arm: **both** (proj on, η′=0.01, E′=5) | CIC | `..._projected_combined_eta0.01_E5[_seed{1,2}]_r60.json` | ⚠️ 3 seeds → 10 | **T3** |
| Arm: **both** | UNSW | `UNSW_NB15_projected_combined_eta0.01_E5[_seed{1,2}]_r60.json` | ⚠️ 3 seeds → 10 | **T3** |
| Reference: **FedAvg / Static** | CIC | existing `seed_sweep_CIC_IDS_2017_{fedavg,static}.json` (3 seeds) → regen 10 @ 60r in `results/ablation/` | ⚠️ | **T3** |
| Reference: **FedAvg / Static** | UNSW | `results/ablation/seed_sweep_UNSW_NB15_{fedavg,static}_r60.json` | ❌ (absent) | **T3** |
| **Statistics module** | code | `fedgraa/stats.py` + `tests/test_stats.py` | ❌ | **T4** |
| Consolidated ablation table (data) | both | `results/ablation/{DS}_ablation_table.json` | ❌ | **T5** |
| Ablation table (rendered) | both | `results/ablation/{DS}_ablation_table.md` + `{DS}_ablation.png` | ❌ | **T6** |
| Completeness check | code | `scripts/check_stage_b_complete.py` | ❌ | **T7** |
| Manifest | doc | `results/ablation/README.md` | ❌ | **T5/T7** |
| Standalone Stage B section | doc | `docs/current/fedgraa_convergence_defect.md` (full) | ❌ | **T8** |

**Net gap:** two theory writeups (T1 defect + why-it-fails, T2 corrected proposition), one multi-seed data campaign (T3: 4 arms × 2 datasets × 10 seeds + baseline extension), one new stats module (T4), the table builder + renderer (T5/T6), a check script + manifest (T7), and the standalone section assembling it all (T8).

> **Two matrix caveats.**
> **(a) η′-alone must be *canonicalized*.** The existing "η′-alone" file (`CIC_IDS_2017_literal_real_smalleta_r60.json`) uses **η′=0.001, E′=1** — the *single-lever over-correction* that collapsed all-class accuracy to 0.8 % (diagnosis §6.1). That configuration does **not** complete the 2×2 factorial. The primary ablation instead runs η′-alone at **η′=0.01, E′=5, no projection**, so the *only* difference from **both** is the projection lever (§3). The η′=0.001/E′=1 result is retained as a documented **secondary "over-correction" row**, not a factorial cell.
> **(b)** The **both** arm (3 seeds) and the FedAvg/Static references (3 seeds) are **extended** to 10 (reuse seeds 0–2; add 3–9), not re-run from scratch.

### The design: a 2×2 factorial (this is what makes it a *table*, not a list)

At fixed E′ = 5, the four arms are the corners of a factorial over two levers:

| | **η′ = 0.1** (paper default) | **η′ = 0.01** (fix) |
|---|---|---|
| **projection off** | literal | η′-alone *(canonical, new)* |
| **projection on** | projection-alone | **both** |

This yields, per (dataset, γ): the **main effect of projection**, the **main effect of the η′ cut**, and their **interaction** — the clean answer to "which lever does the work, and do they need each other?" The diagnosis already hints the answer is *both are necessary* (projection alone leaves rare-acc ≈ 0 %; η′ alone at 0.001 collapses all-acc), which the factorial + significance testing will establish rigorously rather than anecdotally.

---

## 3. Global run conventions (apply to every task)

- **Working dir:** `c:\Dev\Research\PAKDD`. **Interpreter:** `.venv313\Scripts\python.exe` (shown as `python` below).
- **Data/cache:** always `--data-dir data --cache-dir .cache`. First UNSW seed-1…9 runs pay a one-time ~30 s preprocessing cost each, then cache.
- **Diagnostic horizon:** `--rounds 60 --eval-every 1` for **every** arm — identical to the existing arms so all cells are comparable. Do **not** switch to the 200-round base-sweep horizon here; comparability with the committed seed-0 arms depends on holding the horizon fixed.
- **The four factorial arms (one lever at a time, all at E′ = 5).** `diagnose_fedgraa.py` names files `{DS}_{variant}{_tag}_r{rounds}.json`, where `variant` = `projected` if `--project` else `literal`, and **the seed is folded into `--tag`** — there is *no* `--seed` component in the filename, so seed 0 omits it and seed *s* > 0 appends `_seed{s}` (this is why the existing multi-seed files read `..._E5_seed1_r60.json`). At seed 0:
  - **literal** — no flags · variant `literal`, **no `--tag`** → `{DS}_literal_r60.json`
  - **projection-alone** — `--project` · variant `projected`, **no `--tag`** → `{DS}_projected_r60.json`
  - **η′-alone (canonical)** — `--eta-prime 0.01 --weight-epochs 5` (no `--project`) · `--tag etaonly_eta0.01_E5` → `{DS}_literal_etaonly_eta0.01_E5_r60.json`
  - **both** — `--project --eta-prime 0.01 --weight-epochs 5` · `--tag combined_eta0.01_E5` → `{DS}_projected_combined_eta0.01_E5_r60.json`
  - *(secondary, non-factorial)* **η′=0.001 over-correction** — `--eta-prime 0.001 --weight-epochs 1` — documented row only, not required at 10 seeds.
  The new `run_ablation_seed_sweep.py` (§6.2) encapsulates this arm→(flags, tag) mapping so filenames stay consistent and the existing seed-0 files (`{DS}_literal_r60.json`, `CIC_IDS_2017_projected_r60.json`, the `combined` seeds) are recognized for reuse.
- **Seeds:** N = 10 (seeds 0–9). Reuse any existing per-seed diagnostic JSON whose arm+config+dataset already matches (both: seeds 0–2 both datasets; literal: seed 0 both; projection-alone: CIC seed 0); generate the remainder. The canonical η′-alone arm is new at every seed.
- **Baseline references (FedAvg/Static):** run via the *existing* [`../../scripts/run_seed_sweep.py`](../../scripts/run_seed_sweep.py) — **one `--method` per call**, a **space-separated** `--seeds 0 1 … 9` list (its CLI takes a list, **not** a `0-9` range), at the **60-round ablation horizon** (`--rounds 60 --eval-every 1`) with `--out results/ablation/…` so Stage A's committed bands stay untouched. FedAvg/Static are **γ-independent** (γ is a FedGRAA-only knob), so each is a *single* reference value shown across all three γ columns, not a factorial cell. Only the CIC bands exist today (at a different horizon/location); UNSW bands are generated fresh.
- **Metric of record:** rare-class accuracy = `last_n_class_accuracy` (mean per-class recall). Secondary = all-class accuracy. Diagnostic = composite-loss blowup ratio, first simplex-violation round, Lemma-1 monotonicity. These are the table's columns.
- **Significance protocol (pre-registered — decide before looking at 10-seed results):**
  - **Per cell:** percentile bootstrap 95 % CI, B = 10 000 resamples, fixed RNG seed.
  - **Pre-registered pairwise comparisons** (per dataset, per γ): **both** vs {literal, projection-alone, η′-alone, FedAvg, Static}. Each: paired *t*-test **and** Wilcoxon signed-rank (paired by seed) + Cohen's *d* (paired). **Holm–Bonferroni** correction across the 5 comparisons within each (dataset, γ).
  - **Factorial read (complementary):** projection main effect, η′ main effect, interaction — reported as effect sizes with CIs; a two-way repeated-measures ANOVA where cell variances permit (see the zero-variance caveat in §4).
  - Report **exact** *p*, effect size, and CI — never a bare star.
- **Parallelism & RAM:** CIC diagnostic runs each load the 1.46 GB array; the campaign is many *independent* (arm, seed) runs, so parallelize **across runs** with a worker cap (§6.2) to stay under ~11 GB. UNSW arrays are ~47 MB — no memory risk.
- **Determinism:** every training run pins its seed; the bootstrap uses a fixed RNG seed so CIs reproduce exactly; the manifest records SHA + the full seed set.

---

## 4. The gap that must be closed first: single-seed arms + zero statistics infrastructure

Two things block a rigorous ablation table and must be generated/built **before** the table exists — the Stage B analogue of Stage A's "fix the plot defect before rendering figures."

**(1) Two arms cannot be significance-tested at all yet.** Projection-alone is CIC-seed-0-only (UNSW: none), and a *canonical* η′-alone arm doesn't exist (only the off-spec η′=0.001 file). A CI or paired test needs the full 10-seed set. → **T3** generates them.

**(2) There is no statistics code anywhere in the repo.** Both seed-sweep tools compute only `np.mean`/`np.std`; there is no bootstrap, paired test, effect size, or multiplicity control. Shipping "significance testing" therefore means building [`../../fedgraa/stats.py`](../../fedgraa/stats.py) from scratch (§6.1) **with tests**, landed green *before* the table is generated so its numbers are trustworthy on the first pass.

> **Numerical subtlety the stats module must handle (real, not hypothetical):** the rare-acc metric is a bounded mean of per-class recalls, and several cells have **exactly zero variance across seeds** — e.g. the combined fix at γ=0.5 is 91.7 % for all three current seeds. A percentile bootstrap on a constant vector returns a degenerate point CI (correct), but a **paired *t*-test on identical paired differences is undefined** (0/0). `stats.py` must detect degenerate cells and fall back gracefully (report an exact/point result, defer to the sign of the difference, and lean on Wilcoxon), rather than emitting `NaN`. This is exactly the kind of edge a reviewer probes; handling it explicitly is part of the rigor Stage B is selling.

---

## 5. Task-by-task plan

Each task lists what it produces, effort, dependencies, and acceptance criteria. Theory tasks (T1, T2, T8) need no compute; the data campaign (T3) is the only long-running phase.

### Task T1 — Formal defect statement + "why the paper's proof fails" (theory; no compute)

Write [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) §1–3:
- **§1 Setup.** Restate Algorithm 2's `w`-update, the composite loss `L′` (Eq. 9 γ-blend), the feasible simplex `Δ = {w : wₖ ≥ 0, Σwₖ = 1}`, and Lemma 1's descent guarantee (from `understanding/05`).
- **§2 The defect (feasibility).** Algorithm 2 applies no projection; empirically `w` exits `Δ` in **round 1 at every γ on both datasets** (`first_simplex_violation_round = 1`). Off `Δ`, `θ_g = Σwₖθₖ` is no longer a convex combination inside `conv{θₖ}` but **unbounded linear extrapolation outside it** → `L′` blows up → Lemma 1's `L′(wᵗ) ≤ L′(wᵗ⁻¹)` is violated at round 1. Cite the γ=1 CIC loss trace `2.98 → … → 319.11` over rounds 0–10.
- **§3 Why Appendix A.1 doesn't cover it** (three compounding gaps, all already documented — *assembling, not discovering*): **(i)** Assumption 1 (`L′` convex + β-smooth) is false for the tanh-MLP 200→100→50 (#13); **(ii)** `L_a = min(L_h, L_m)` is non-smooth at the branch boundary the optimizer actively approaches (#6), so no finite β exists and thus **no η′ satisfies Assumption 3 (η′ ≤ 1/β)** — the descent bracket `(η′ − β(η′)²/2)` can go negative; **(iii)** even granting smoothness, Lemma 2 tacitly assumes `w ∈ Δ` so `θ_g` stays bounded — Assumption 2 bounds *gradients*, never the *iterate* `w`, and Algorithm 2 never enforces `w ∈ Δ`.
- **Produces:** `fedgraa_convergence_defect.md` §1–3.
- **Effort:** ~2–3 h writing.
- **Accept when:** every claim cites either a persisted metric (violation round, loss trace) or a paper equation/assumption; a reader who never saw the audit trail can follow defect → cause.

### Task T2 — Corrected convergence proposition (theory; no compute)

Write [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) §4:
- **Fixed algorithm.** `wᵗ = Π_Δ(wᵗ⁻¹ − η′ ∂L′(wᵗ⁻¹))`, `Π_Δ` = Euclidean projection onto the simplex (Duchi 2008; already implemented as `project_to_simplex` in `diagnose_fedgraa.py`) ⇒ the method becomes **projected (sub)gradient descent** on the convex set `Δ`.
- **Proposition (stated honestly for a non-convex objective).** With `w ∈ Δ`, `θ_g ∈ conv{θₖ}` is bounded, so `L′(θ_g)` is bounded on `Δ` → the round-1 blowup is **structurally impossible** (matches measured `Σw = 1.0000`, blowup `1.0×` on all runs). Under Assumption 2 (bounded subgradients) and small enough step (empirically η′ ≤ 0.01), the **projected-gradient-mapping norm → 0**, i.e. iterates converge to a **stationary/KKT point** of `min_{w∈Δ} L′`, and `L′` is non-increasing along the trajectory.
- **Proof sketch.** Projection is non-expansive (`‖Π_Δx − Π_Δy‖ ≤ ‖x − y‖`); standard projected-subgradient descent on a compact convex set with bounded subgradients gives `min_t ‖G_η(wᵗ)‖² = O(1/T)` (gradient-mapping convergence) ⇒ stationarity. Treat the `min(L_h, L_m)` non-smoothness via subgradient selection / a Moreau envelope, noting non-smoothness is confined to the measure-zero set `{L_h = L_m}`. **Contrast explicitly with the paper's Theorem 1:** we claim only *stationarity*, not the global minimizer, because the objective is non-convex — this is the honest, correct guarantee.
- **Empirical corroboration.** All 9 current seed×γ combined-fix runs show blowup exactly `1.0×`, Lemma-1 monotone, rare-acc recovered — the proposition's predictions hold; the 10-seed campaign (T3) confirms at scale.
- **Rigor flag.** Exact constants, the formal `min()` subgradient treatment, and an estimated β vs. the η′ threshold are to be **hardened with the PhD mentor/prof**; §4 states the guarantee and proof sketch at a level sufficient for the standalone section.
- **Produces:** `fedgraa_convergence_defect.md` §4.
- **Effort:** ~3–4 h.
- **Accept when:** the proposition states its assumptions explicitly, distinguishes stationarity from the paper's global-min claim, and is cross-checked against the persisted blowup=1.0× / monotone data.

### Task T3 — Multi-seed ablation campaign (4 arms × 2 datasets × 10 seeds + baselines)

Generate the missing (arm, dataset, seed) diagnostic runs; reuse existing JSONs. Driven by the new resumable, worker-capped `scripts/run_ablation_seed_sweep.py` (§6.2) so it's one command per dataset. Per-run command pattern (each covers 3 γ):

```bash
python scripts/diagnose_fedgraa.py --dataset {DS} --gammas 0 0.5 1.0 \
    --rounds 60 --eval-every 1 [ARM FLAGS] --seed {s} --tag {ARM_TAG}[_seed{s} if s>0]
```

ARM → flags (+ base `--tag`), per §3: **literal** → *(no flags, no tag)* · **projection-alone** → `--project` *(no tag)* · **η′-alone** → `--eta-prime 0.01 --weight-epochs 5 --tag etaonly_eta0.01_E5` · **both** → `--project --eta-prime 0.01 --weight-epochs 5 --tag combined_eta0.01_E5`. For seed *s* > 0 append `_seed{s}` to the tag (so literal/projection-alone use `--tag seed{s}` at *s* > 0). Then generate the FedAvg/Static references at 10 seeds via **four** `run_seed_sweep.py` calls (fedavg/static × both datasets), each with space-separated `--seeds 0 1 … 9 --rounds 60 --eval-every 1 --out results/ablation/…` (§3). The existing CIC bands are at a different horizon/location, so the ablation regenerates them here; **UNSW baseline bands don't exist at all yet**.

- **Produces:** the full grid of per-seed diagnostic JSONs under `../../results/diagnostics/` (naming per the generalized `_seed_filename` convention, per arm), for both datasets; extended baseline seed-sweep JSONs.
- **Configs:** 4 arms × 2 datasets × 10 seeds = 80 (arm, seed) runs (3 γ each), minus ~10 reusable ≈ **70 fresh diagnostic runs**; plus **4 baseline calls** (fedavg/static × 2 datasets), 10 seeds each.
- **Effort:** ~15–20 min per (arm, seed) [3 γ × 60 rounds, eval_every=1]. Parallel across runs (worker-capped) ≈ **6–10 h unattended**; sequential ~18–23 h. Baselines add ~1–2 h. The driver is resumable, so an interruption costs ≤ 1 run.
- **Depends on:** §6.2 driver (nice-to-have; the raw `diagnose_fedgraa.py` loop works without it).
- **Accept when:** for every (arm, dataset, seed) a diagnostic JSON exists carrying the `diagnosis` block; `check_stage_b_complete.py` counts 10 seeds per arm per dataset.

### Task T4 — Statistics module `fedgraa/stats.py` (code; tested)

Build the module (§6.1): `bootstrap_ci`, `paired_ttest`, `wilcoxon_signed_rank`, `cohens_d_paired`, `holm_bonferroni`, all degenerate-cell-safe. Unit tests against SciPy references + hand-computed cases + the zero-variance edge.
- **Effort:** ~3–4 h incl. tests.
- **Accept when:** `pytest tests` green; bootstrap CI matches a seeded reference; paired tests match `scipy.stats` within tolerance; Holm matches a worked example; a zero-variance cell returns a point CI without `NaN`.

### Task T5 — Consolidated ablation-table builder `scripts/build_ablation_table.py` (code)

Reads all per-seed diagnostic JSONs (4 arms) + the baseline seed-sweep JSONs for a dataset and emits `../../results/ablation/{DS}_ablation_table.json`: per (arm, γ) the per-seed arrays + mean + bootstrap CI for rare-acc / all-acc / blowup; per (comparison, γ) the paired-test *p*, Cohen's *d*, and Holm-adjusted *p*; and the factorial main-effect/interaction summary. Reuses `seed_sweep_combined_fix.py::aggregate`'s math + `fedgraa/stats.py`. + unit test (feed tiny dicts, assert cell + *p* wiring). Also writes/updates `../../results/ablation/README.md`.
- **Effort:** ~3–4 h incl. test + manifest.
- **Accept when:** JSON emitted for both datasets with all arm×γ cells + the 5 comparisons + the factorial read; unit test green.

### Task T6 — Render the table + figure `scripts/render_ablation.py` (code)

Emits `../../results/ablation/{DS}_ablation_table.md` (publication-ready: rows = arms, cols = γ, cells = "rare-acc mean [95 % CI] · all-acc · blowup", significance markers keyed to the persisted exact *p*) and `../../results/ablation/{DS}_ablation.png` (grouped bar chart of rare-acc by arm × γ with CI error bars). Reuses `fedgraa/plot.py` idioms + `matplotlib.use("Agg")`. + a smoke test.
- **Effort:** ~2–3 h.
- **Accept when:** md + png render for both datasets; where the data supports it, **both** visibly dominates the single-lever arms with non-overlapping CIs.

### Task T7 — Completeness check + manifest `scripts/check_stage_b_complete.py` (code/doc)

Mirror `check_stage_a_complete.py`'s style (§7): standalone script, aggregate-all-failures, exit 0/1. + finalize `results/ablation/README.md`.
- **Effort:** ~2 h.
- **Accept when:** exits `0` on the completed tree; exits `1` with a specific message when any artifact is missing.

### Task T8 — Assemble the standalone Stage B section (doc)

Finalize [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) as an end-to-end paper section: framing ("Why a published FL-IDS aggregation algorithm doesn't converge, and how to fix it") → §1–4 theory (T1/T2) → §5 the consolidated ablation table (embed the rendered md + figure) → §6 discussion (severity is dataset-dependent — CIC's 193× blowup vs UNSW's 7.6×; the η′=0.001 over-correction row) → §7 honest limitations (stationarity ≠ global min; the fix is diagnostic-only; N = 10 seeds; CPU-only). This is the **anchor contribution**, real even if Stage C stalls. Write the `STAGE_B_MASTER_REPORT.md` capstone alongside (mirrors Stage A).
- **Effort:** ~3–4 h.
- **Accept when:** reads as a self-contained section; every number traces to a persisted artifact via the manifest; ready to hand to the mentor with the Stage A package.

---

## 6. Code changes required (small, deterministic, testable — no big compute)

Do these **first** so T3's runs and the table are correct on the first pass.

### 6.1 New `fedgraa/stats.py` (REQUIRED — blocks T5/T6)

Pure-Python + NumPy/SciPy, no training deps. Signatures:

```python
def bootstrap_ci(values, statistic=np.mean, n_boot=10_000, alpha=0.05, seed=0): ...   # -> (lo, hi); point CI if constant
def paired_ttest(a, b): ...            # -> (t, p); (0.0, 1.0) when all paired diffs identical
def wilcoxon_signed_rank(a, b): ...    # -> (W, p); scipy.stats.wilcoxon, zero-diff-safe
def cohens_d_paired(a, b): ...         # -> d (paired); 0.0 when diffs constant at 0
def holm_bonferroni(pvals): ...        # -> adjusted p-values (list), order-preserving
```

Add `scipy>=1.10` to [`../../requirements.txt`](../../requirements.txt) (explicit; currently only transitive via scikit-learn). Bootstrap uses `np.random.default_rng(seed)` for reproducibility.

### 6.2 New `scripts/run_ablation_seed_sweep.py` (REQUIRED — drives T3)

Enumerate `(arm, seed)` for a dataset; **skip** any diagnostic JSON already on disk (resumable); call `diagnose_fedgraa.run_fine_grained` per `(arm, γ, seed)`; `--workers N` cap. + a unit test of the enumeration/skip logic (mock the runner) so a re-run provably re-does nothing already present.

### 6.3 New `scripts/build_ablation_table.py` (REQUIRED — T5)

Consolidates all four arms + baselines into the table JSON via `fedgraa/stats.py`. Leave `seed_sweep_combined_fix.py` **as-is** (it stays the "both"-only band that Stage A's check depends on) and reuse its `aggregate()` math rather than mutating it. + unit test (`test_build_ablation_table.py`, importlib-load per `test_seed_sweep_combined_fix.py`'s pattern).

### 6.4 New `scripts/render_ablation.py` (REQUIRED — T6) + smoke test (`test_render_ablation.py`, `pytest.importorskip("matplotlib")` + Agg, per `test_plot.py`).

### 6.5 `results/ablation/README.md` manifest (REQUIRED — reproducibility). Every results path in backticks (`` `results/ablation/...` ``) so a `check_manifest_paths`-style on-disk + git-tracked check works.

### 6.6 New `scripts/check_stage_b_complete.py` (REQUIRED — §7).

### 6.7 *(Optional, recommended)* a `--seeds 0-9` range-shorthand parser. The **new** `run_ablation_seed_sweep.py` accepts range syntax natively; the **existing** `run_seed_sweep.py` currently takes a space-separated list (`--seeds 0 1 … 9`). Optionally factor a shared parser so both express the seed set the same way — otherwise use the space-separated form for `run_seed_sweep.py` (as the §8 runbook does).

### 6.8 Tests, in the existing style — keep the full suite green

- `tests/test_stats.py` — bootstrap CI, paired *t*, Wilcoxon, Cohen's *d*, Holm, **zero-variance edge** (pure-Python, fast).
- `tests/test_build_ablation_table.py` — importlib-load the script; feed tiny hand-written per-seed dicts; assert cell values + CI/*p* wiring (mirrors `test_seed_sweep_combined_fix.py`).
- `tests/test_render_ablation.py` — `importorskip` matplotlib, Agg, assert md + png produced (mirrors `test_plot.py`).

Run `.venv313\Scripts\python.exe -m pytest tests --tb=short` — the suite (Stage A left it green; confirm the exact count with a run) plus the new tests must all pass.

---

## 7. Verification — make "is Stage B done?" a single green/red signal

`scripts/check_stage_b_complete.py` (same idioms as `check_stage_a_complete.py`: standalone, `fail()`/`ok()`, aggregate all failures, `sys.exit(main())`) asserts:

- Each of the **4 arms** has **10** per-seed diagnostic JSONs per dataset (glob + count), each carrying the `diagnosis` block fields.
- `results/ablation/{CIC-IDS-2017,UNSW-NB15}_ablation_table.json` exist, load, and contain **every** arm×γ cell with `{rare_acc_mean, rare_acc_ci, all_acc_mean, all_acc_ci, blowup_mean, …}` and the **5 pre-registered comparisons** with `{p_value, p_holm, cohens_d}`.
- The rendered `{DS}_ablation_table.md` and `{DS}_ablation.png` exist and are non-empty.
- `docs/current/fedgraa_convergence_defect.md` exists and contains the required section headings (defect, why-the-proof-fails, corrected proposition, ablation, limitations) — a grep-level guard that the theory sections are present.
- Every `` `results/…` `` path in `results/ablation/README.md` exists on disk **and** is git-tracked (reuse Stage A's `git ls-files` regex pattern).

Keep it out of `pytest tests` (decoupled from large committed binaries). Run it last; cite its `0` exit in the capstone.

**Also run the full suite** after the §6 changes: `.venv313\Scripts\python.exe -m pytest tests --tb=short` must stay green (existing tests + the new `test_stats` / `test_build_ablation_table` / `test_render_ablation`).

---

## 8. One-shot runbook (ordered; copy-paste)

```bash
cd /c/Dev/Research/PAKDD

# --- Phase 1: code first (no big compute) ---
#  6.1 stats.py (+ scipy in requirements) · 6.2 ablation driver · 6.3 table builder
#  6.4 renderer · 6.6 check script · 6.8 tests
.venv313/Scripts/python.exe -m pytest tests --tb=short          # must stay green (+ new tests)

# --- Phase 2: multi-seed ablation campaign (unattended; resumable; worker-capped) ---
python scripts/run_ablation_seed_sweep.py --dataset CIC-IDS-2017 --seeds 0-9 --workers 3
python scripts/run_ablation_seed_sweep.py --dataset UNSW-NB15   --seeds 0-9
# Baselines: one --method per call, space-separated seeds (existing CLI), 60r horizon, --out keeps Stage A bands untouched
python scripts/run_seed_sweep.py --dataset CIC-IDS-2017 --method fedavg --seeds 0 1 2 3 4 5 6 7 8 9 --rounds 60 --eval-every 1 --out results/ablation/seed_sweep_CIC_IDS_2017_fedavg_r60.json
python scripts/run_seed_sweep.py --dataset CIC-IDS-2017 --method static --seeds 0 1 2 3 4 5 6 7 8 9 --rounds 60 --eval-every 1 --out results/ablation/seed_sweep_CIC_IDS_2017_static_r60.json
python scripts/run_seed_sweep.py --dataset UNSW-NB15   --method fedavg --seeds 0 1 2 3 4 5 6 7 8 9 --rounds 60 --eval-every 1 --out results/ablation/seed_sweep_UNSW_NB15_fedavg_r60.json
python scripts/run_seed_sweep.py --dataset UNSW-NB15   --method static --seeds 0 1 2 3 4 5 6 7 8 9 --rounds 60 --eval-every 1 --out results/ablation/seed_sweep_UNSW_NB15_static_r60.json

# --- Phase 3: consolidate + render ---
python scripts/build_ablation_table.py --dataset CIC-IDS-2017
python scripts/build_ablation_table.py --dataset UNSW-NB15
python scripts/render_ablation.py      --dataset CIC-IDS-2017
python scripts/render_ablation.py      --dataset UNSW-NB15

# --- Phase 4: theory doc + manifest, then verify + commit ---
#  write docs/current/fedgraa_convergence_defect.md (T1/T2/T8) and results/ablation/README.md (6.5)
python scripts/check_stage_b_complete.py           # must exit 0
git add fedgraa/stats.py scripts/ results/ablation/ tests/ requirements.txt \
        docs/current/fedgraa_convergence_defect.md docs/current/stage_b_plan.md
git commit -m "Stage B: convergence-defect theory + consolidated ablation with significance testing"
```

*(The arm loop and `--seeds 0-9` expansion live inside the driver, so PowerShell users run the same commands — no Bash loops needed.)*

---

## 9. Explicitly out of scope (deferred to Stage C)

- **A new method** (few-shot/prototype-guided hybrid, or the privacy-compliant redesign) → Stage C ([`publication_roadmap.md`](publication_roadmap.md) §2 Stage C).
- **Baselines beyond FedAvg/Static** (FedProx/FedOpt/FedSND implemented but unrun; AFL/q-FedAvg/FedAdp not implemented) → Stage C.
- **Scalability (50–200 clients) and extreme non-IID** experiments → Stage C.
- **Fully rigorous / machine-checked proof** with exact β constants and the formal `min()` subgradient treatment → hardened with the mentor in Stage C; Stage B ships the proposition + proof sketch.
- **Changing the strict-reproduction default** in `fedgraa/` — `ProjectedFedGRAA` stays diagnostic-only in `scripts/`, per the standing no-silent-patching mandate ([`../archive/strict_reproduction_report.md`](../archive/strict_reproduction_report.md)).

---

## 10. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Bootstrap / paired test misapplied on a zero-variance or bounded cell → `NaN`/garbage *p* | **Med** | Wrong significance claim | `stats.py` detects degenerate cells (§4); tests cover them; report exact + Wilcoxon backup |
| η′-alone run at off-spec η′=0.001 instead of canonical 0.01 → non-isolated ablation | Med | Factorial breaks | §2 caveat (a) + §3 pin η′=0.01/E′=5 no-projection; driver tags arms distinctly; 0.001 kept as a labeled secondary row |
| 10-seed CIC campaign OOM under parallel (`8 × 1.46 GB`) | Med (<16 GB) | Task stalls | `--workers 3` cap (§6.2); driver resumable so a crash loses ≤ 1 run |
| Paired tests over-claimed on n=10 (small sample) | Med | Reviewer skepticism | Report effect sizes + CIs + non-parametric Wilcoxon, not bare *p*<0.05; Holm correction; comparisons pre-registered (§3) |
| Corrected theory over-reaches (claims global min) | Low | Wrong theoretical claim | Proposition claims **stationarity only**, explicitly contrasted with the paper's Theorem 1; flagged for mentor hardening |
| Scope creep into Stage C (adding a method/baselines mid-flight) | Med | Stage B never ships | §9 hard boundary; Stage B is the shippable anchor contribution |
| Long campaign interrupted (laptop sleep/crash) | Med | Lost compute | Driver skips existing JSONs (resumable); or run overnight on AC power |
| Non-determinism across a re-run | Low | Numbers drift | Seeds + bootstrap RNG pinned; manifest records SHA + seed set |

---

## 11. Effort summary

| Phase | Work | Wall time |
|---|---|---|
| 1 | `stats.py` + driver + builder + renderer + check + tests (§6) | ~12–16 h hands-on (code) |
| 2 | 10-seed ablation campaign + baseline extension (T3) | ~6–10 h compute (parallel), unattended |
| 3 | Consolidate + render (T5/T6) | ~15 min compute |
| 4 | Theory doc T1/T2/T8 + manifest + verify + commit | ~8–11 h hands-on (writing) |
| **Total** | | **~20–27 h hands-on + ~6–10 h unattended compute** |

Sequential-only (no parallel) roughly triples the campaign to ~18–23 h compute but removes the RAM risk — a reasonable overnight option. The theory (T1/T2/T8) and code (§6) phases have no compute dependency and can proceed while the campaign runs.

---

*When Phases 1–4 are complete and `check_stage_b_complete.py` exits 0, Stage B is closed. Per [`publication_roadmap.md`](publication_roadmap.md) §6.6, the next action is to bring the combined Stage A + B package to the PhD mentor to jointly scope Stage C's novelty angle and target venue — with Stage B standing on its own as the anchor contribution in the meantime.*

---

## Appendix — grounding (facts verified against the tree, commit `a17a443`)

This plan was written against the live code and the persisted Stage A results. Verified, not assumed:

- **No statistics code exists:** repo-wide grep for `scipy.stats`/`bootstrap`/`ttest`/`wilcoxon`/`percentile` matched only documentation prose. `stats.py` is genuinely new.
- **SciPy is importable but not a direct dependency** (transitive via `scikit-learn`); `requirements.txt` must add it.
- **All four arms already run via `diagnose_fedgraa.py --project / --eta-prime / --weight-epochs`;** no new trainer is needed — the campaign is a seed sweep over an existing runner.
- **Seed coverage is asymmetric:** the `both` arm has 3 seeds (both datasets); projection-alone is CIC-seed-0-only (UNSW absent); a *canonical* η′-alone (0.01/E′=5) arm does not exist — the off-spec 0.001/E′=1 file is the only η′-alone data.
- **Baseline seed-bands exist for CIC only** (`seed_sweep_CIC_IDS_2017_{fedavg,static}.json`); **UNSW baseline bands are absent.** The ablation regenerates both at the 60-round horizon into `results/ablation/`. Confirmed against the `results/diagnostics/` listing.
- **`run_seed_sweep.py` runs one `--method` per call** (default `static`) and takes a **space-separated `--seeds` list** (`nargs="+"`), not a `0-9` range, with an `--out` override — verified by reading the script. The `0-9` shorthand is native only to the new `run_ablation_seed_sweep.py` (§6.2/6.7).
- **The theory gaps are already catalogued** (`PAPER_AMBIGUITY_REPORT.md` #6/#13; `understanding/05_convergence_proof.md` Lemma 1 / Assumptions 1–4) — T1/T2 assemble them into a formal section, they don't rediscover them.
- **Filename convention (verified in `diagnose_fedgraa.py` `main`):** output is `{DS}_{variant}{_tag}_r{rounds}.json`, `variant` ∈ {`literal`, `projected`} from `--project`; there is **no `--seed` component** — the seed is encoded in `--tag` (seed 0 omits it), which is why the existing multi-seed files read `..._combined_eta0.01_E5_seed1_r60.json`. `run_fine_grained` is at L98, `analyze_history` (the `diagnosis` block) at L137, `ProjectedFedGRAA`/`project_to_simplex` at L80/L61. The new driver honors this to reuse seed-0 files.
- **`run_seed_sweep.py` supports `--out`** (custom output path), so baseline ablation runs write to `results/ablation/` without touching Stage A's committed `results/diagnostics/` bands.
- **Section skeleton mirrors `stage_a_plan.md`** (§0 done-definition → §1 grounding table → §2 matrix → §3 conventions → §4 gap-to-close-first → §5 tasks → §6 code → §7 verification → §8 runbook → §9 out-of-scope → §10 risks → §11 effort → Appendix), so the two plans read as siblings.
