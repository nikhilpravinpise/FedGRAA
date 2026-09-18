# Internship Report Megaplan - FedGRAA (20-25 pages, LaTeX -> PDF)

**One-sentence summary:** Produce a complete, compile-ready, 20-25 page research-internship report in LaTeX that follows IIT conventions (title page, certificate, acknowledgement, abstract, TOC, lists of figures/tables, nomenclature, numbered chapters, bibliography) and tells the full FedGRAA internship arc - faithful reproduction of a published FL-IDS paper, diagnosis of its missing-simplex-projection defect, corrected convergence theory, and the novel FedGRAA-v2 privacy-compliant extension with a 960-cell statistically-tested evaluation - using only numbers and figures that already exist and are verified in this repository.

---

## 0. Decisions already locked (from user)

| Decision | Choice |
|---|---|
| Output format | LaTeX -> PDF (`.tex` master, compiled PDF deliverable) |
| Framing | Full arc; FedGRAA-v2 is the "proposed methodology" payload |
| Location | New `internship_report/` folder at repo root (`C:\Dev\FedGRAA\internship_report\`) |
| Front matter | Clearly-marked placeholders (`<Roll No.>`, `<Professor Name>`, etc.) - user fills in later |
| Page target | 20-25 pages of compiled PDF counted on the arabic-folio pages (Ch 1 through Bibliography). Roman-folio front matter (~6-7 pages) is additional, so the full document runs ~28-31 pages total. If the user's institute counts total document pages instead, apply the compression path in §9 option B |

**Writing-style constraints (binding):**
- No em-dash "—" anywhere in the report source (global user rule). Use "-" or restructure the sentence.
- Formal technical English, complete sentences, IIT report tone.
- Honest-negative-results framing matching the repo's own style: report the one non-significant comparison, the weak SecFedNIDS reproduction, and the defense-collapses-rare-recall finding plainly.
- Every quantitative claim must trace to a file in this repo (raw JSON, generated table, or master report). No invented numbers. The number-source table in §6 makes this auditable.
- Internship period attribution: the work was performed under an IIT Guwahati professor with day-to-day PhD-student mentorship (per `docs/current/publication_roadmap.md`); all personal/institutional specifics are placeholders.

---

## 1. What "IIT internship report" means here (research summary)

Two authoritative sources were used: IIT Guwahati's own ME-401 Summer Training Report template (`iitg.ac.in/mech/documents/570/Internship_Report_Template.pdf`) and IIT Bombay's report-writing guidelines (`ese.iitb.ac.in` + `che.iitb.ac.in`). Convergent structure across both:

**Preliminary pages (small-roman folios i, ii, iii..., each starting on a fresh page):**
1. Title page - report title, student name + roll no, "submitted in partial fulfilment of...", guide/supervisor name, department, institute, place, month-year
2. Certificate page - signed-by-guide boilerplate
3. Acknowledgement - brief, non-flowery
4. Abstract - single paragraph, ~150-200 words, covering purpose/method/findings/conclusion in that order
5. Table of Contents
6. List of Figures
7. List of Tables
8. Nomenclature - Roman capitals, then lowercase, then Greek letters, then acronyms; every symbol used in the report, alphabetical

**Main pages (arabic folios):** numbered chapters, max three levels (chapter > section x.y > subsection x.y.z), each chapter's sections carrying titles.

**IIT Guwahati chapter skeleton** (maps 1:1 onto the user's requested structure):
- Ch 1 Introduction (Motivation, Objectives, Outline of the report)
- Ch 2 Literature Review
- Ch 3 Methodology
- Ch 4 Results and Discussions
- Ch 5 Conclusions and Future Scope
- Bibliography, Appendix

The user's requested structure - Introduction, Related Work, Proposed Methodology, Experimental Results, Discussion, Conclusion - is adopted verbatim as chapter titles, with Discussion split out of Results (a common and accepted variant of the IITG skeleton).

---

## 2. Deliverables (exact file list)

```
internship_report/
  plan.md                       <- this file (the megaplan)
  report.tex                    <- master LaTeX document (single file; no \input splitting - easier to compile and diff)
  references.bib                <- BibTeX bibliography (~40 entries, all from the repo's verified catalog)
  figures/                      <- PNGs copied verbatim from results/ (names below); nothing regenerated
  build/                        <- tectonic/pdflatex output dir (gitignored)
  report.pdf                    <- compiled deliverable (committed, per repo convention of committing artifacts)
  REPORTING_NOTES.md            <- 1-page "how to fill placeholders + how to recompile" note for the user
```

`.gitignore` addition (append to root `.gitignore`): `internship_report/build/`.

---

## 3. LaTeX toolchain (no TeX currently installed)

`where pdflatex xelatex tectonic latexmk` finds nothing on this machine. Plan, in order of preference:

1. **Tectonic** (recommended): single self-contained binary that auto-downloads packages; `winget install Tectonic.Tectonic` (fall back to name-search `winget install Tectonic` if the ID is not found) or unzip the GitHub release. Compile: `tectonic --outdir build report.tex` then copy `build/report.pdf` up. Two passes are automatic.
2. Fallback: MiKTeX (`winget install MiKTeX.MiKTeX`), then `pdflatex -output-directory build report.tex` x2.
3. Document class: `report` (built-in, gives `\chapter`) at 11pt, A4, `geometry` margins 1in. No `llncs` dependency (that class is for the PAKDD paper, not an internship report).
4. Packages: `geometry, graphicx, booktabs, amsmath, amssymb, algorithm, algpseudocode, hyperref, xurl, caption, subcaption, cite, enumitem, fancyhdr, titlesec, setspace, microtype`. All auto-fetched by tectonic; no manual `.sty` files.
5. Bibliography: BibTeX via `references.bib` + `\bibliographystyle{plain}` (numbered `[n]` style, matching engineering-report convention). Tectonic runs bibtex automatically.

**If tectonic install is blocked:** the plan still produces the complete `.tex` + figures + `.bib`; compilation instructions go in `REPORTING_NOTES.md` for Overleaf (upload zip) as the zero-install path. This is a fallback, not the plan - every effort goes to compiling locally first.

---

## 4. Report structure, section by section (the full blueprint)

Page budget targets 11pt/A4/1in-margins/~1.15 spacing. F = figure, T = table. The chapter budgets below sum to ~21-23 arabic pages + ~1.5-2 for the bibliography, landing 22-25 inside the 20-25 target; front matter (~6-7 roman pages) is on top of that. Knobs to steer inside the range are in §9.

### Front matter (roman folios, ~6-7 pages; each item on a fresh page per IIT convention)

| Item | Content |
|---|---|
| Title page | "Internship Report: [title]" - proposed title: **"Federated Gradient-Refined Adaptive Averaging for Intrusion Detection: Faithful Reproduction, Convergence Defect Diagnosis, and a Privacy-Compliant Extension (FedGRAA-v2)"** (note: the original algorithm/paper expand FedGRAA with "Averaging" - the v2 manuscript alone uses "Aggregation" in its own title, so the report title must use "Averaging" to stay consistent with the acronym it defines). Fields: `<Student Name>` (Nikhil Pravin Pise pre-filled from `FedGRAA/CITATION.cff`), `<Roll No.>`, `<Home Institution>`, `<Department>`, "submitted in partial fulfilment of the requirements for the research internship", `<Professor Name, IIT Guwahati>` (guide - named only as "an IIT Guwahati professor" in `docs/current/publication_roadmap.md`), `<Mentor Name>` (day-to-day PhD-student mentor; a candidate name appears in `docs/audit/MENTOR_*` provenance docs - pre-fill only after user confirmation, else leave as placeholder), `<Place>`, `<Month Year>` |
| Certificate | Standard IITG boilerplate with signature lines for mentor + professor |
| Acknowledgement | 1 short paragraph, placeholders |
| Abstract | 150-200 words, purpose/method/findings/conclusion order; headline numbers: 0.00% reproduced rare-acc on CIC vs the paper's claimed 88.88% (UNSW: 33.33% reproduced vs 21.09% claimed - the gap is CIC-stark, and the paper's own "Acc." metric is ambiguity-flagged/undefined), missing simplex projection, 120/120 round-1 violations in unprojected cells, 89.3%/69.2% fixed rare-acc, 93.47%/69.66% v2 best, 111/112 significant, 6.00x->3.00x overhead, attacker up-weighted to 2.0-3.1x uniform share, defended cells collapse rare recall to ~0% (30/30 CIC, 19/30 UNSW) |
| TOC + LoF + LoT | auto-generated |
| Nomenclature | all symbols used: theta_g, theta_k, w, w_k, eta, eta', E, E', B, T, K, n, gamma, Delta, Pi_Delta, L', L_a, L_h, L_m, L_u, D_V, D_V^h/m/n, G_eta'(w), zeta, m, P, S_t, ASR, CI, FL, IDS, NIDS, SecAgg, DP, MLP, IID/non-IID |

### Chapter 1 - Introduction (~3 pages)

- 1.1 Motivation: centralised IDS privacy problem -> FL; FL's own size-proportional aggregation dilutes rare-attack clients (infiltration, SQL injection, Heartbleed) - the operational classes that matter most. Source: `docs/paper/FedGRAA_v2_PAKDD_Submission.md` S1 + `understanding/02_the_problem.md`.
- 1.2 The internship project: reproduce *"Federated Gradient-Refined Adaptive Averaging for Intrusion Detection Systems"* (PAKDD submission draft, anonymized) from scratch in TF/Keras; honestly measure; diagnose divergences; correct the theory; build a novel extension. Three stages (A reproduction, B critique+theory+ablation, C novel extension).
- 1.3 Objectives (numbered list, each maps to a results section): O1 faithful reproduction with persisted artifacts; O2 root-cause any divergence mechanistically; O3 correct the convergence theory and validate a minimal fix with statistical rigor; O4 remove the design's privacy violations; O5 evaluate against the full baseline frontier incl. adversarial robustness.
- 1.4 Summary of contributions (5 bullets: the defect, the fix, the corrected theorem, FedGRAA-v2, the evaluation + honest negative results).
- 1.5 Outline of the report (one line per chapter).

### Chapter 2 - Related Work (~3.5 pages)

Drawn from `FedGRAA_v2_PAKDD_Submission.md` S2 (the venue-vetted version) + `docs/research_c/Final_Stage_C.md` + `understanding/10_SOTA_adaptive_aggregation.md`. Subsections:

- 2.1 Federated learning and FedAvg (McMahan 2017; eq. for size-weighted aggregation; the non-IID problem).
- 2.2 Adaptive aggregation - three signal families: update-direction alignment (FedAWA), loss/accuracy signals (FedNolowe, PACT), outlier filtering (MESAS, CrowdGuard, FreqFed) - and why all three suppress exactly the rare-class clients FedGRAA protects.
- 2.3 The counter-current: FedAWARE (gradient diversity), Jiang2024 + AdaAggRL (RL-learned), CELM/SpectralFed/CoAst (data-free contribution estimation), precision-weighted (Reyes2021), FedPGO + NIDS-FGPA (same NIDS domain, gradient orthogonality/similarity).
- 2.4 Trust-based robust aggregation: FLTrust, RAGA, FedTruth - all sharing the server-held-reference assumption FedGRAA-v1 shares; quote FedTruth/CoAst/CELM's own admissions it is impractical.
- 2.5 Classic FL optimizers: FedProx, FedOpt, SCAFFOLD, FedNova.
- 2.6 Poisoning attacks and defenses in FL-NIDS: label-flip, model-poison ("pill", Guo et al. AAAI 2026), SecFedNIDS, WeiDetect.
- 2.7 Positioning paragraph: the specific unoccupied point (gamma-tunable, exact-gradient, single-global-model, evaluated on rare-class recall) - not claiming the region is empty (FedPGO is adjacent on the same dataset), claiming this point is unreported and evaluating it against 19 methods directly.

### Chapter 3 - Proposed Methodology (~5.5-6 pages)

The chapter is the report's core; it carries the reproduction work as "what we found in the printed algorithm" plus the full corrected design.

- 3.1 Problem formulation: FL round loop (Alg 1), client update, aggregation theta_g = sum w_k theta_k; the simplex Delta = {w_k>=0, sum=1}; composite objective L' = (1-gamma)L_a + gamma L_u, L_a = min(L_h, L_m); D_V splits; gamma semantics. Source: `STAGE_A_MASTER_REPORT.md` S2 + manuscript S3/S4.1.
- 3.2 FedGRAA as published (Algorithm box): the printed Algorithm 2 with Adam weight-update on w (confirmed in both the manuscript's own correction note and `aggregation.py:166,251`, which label it "Adam step on w (eq. 10)"); the gradient math dL'/dw_k = <grad_P L', P_k>. Include the algorithm float.
- 3.3 The diagnosed defect (headline internship finding): no projection step; w leaves Delta in round 1, 120/120 cells; off-simplex aggregation = unbounded affine extrapolation; Lemma 1 violated immediately (loss trace 2.98 -> 149.79 over 10 rounds; blowup up to 1.86e6x). Why Appendix A.1's proof cannot cover it: (i) convexity assumption false for tanh-MLP; (ii) min(L_h,L_m) non-smooth exactly where training drives it; (iii) Lemma 2 tacitly assumes w in Delta. Source: `fedgraa_convergence_defect.md` S2-S3, `fedgraa_divergence_diagnosis.md`.
- 3.4 The minimal correction: w^t = Pi_Delta(w^{t-1} - eta' dL') with Duchi et al. 2008 projection + eta'=0.01 (10x smaller), E'=5 unchanged; diagnostic-only discipline (never merged into the reproduction baseline).
- 3.5 Corrected convergence theory: Stage B proposition (feasibility structural + gradient-mapping stationarity O(1/T), explicitly weaker than the paper's global-optimality claim); Stage C extension to zeta-inexact oracle: min ||G_eta'(w^t)||^2 = O(1/T) + O(zeta^2); zeta decomposition (zeta_secagg, zeta_dp, zeta_stale(m), zeta_dropout). Statement-level only + proof sketch pointer; PL-condition caveat stated honestly.
- 3.6 FedGRAA-v2: the privacy-compliant redesign: (a) oracle rare-class list -> SecAgg-summed class-frequency histogram (+ optional client-side Laplace noise, advanced-composition epsilon_total ~8.89); (b) server-held D_V -> federated backward pass two_phase_composite_grad reconstructing the exact pooled gradient (regression-test bound <1e-6 relative error; measured ~1-4e-7 depending on gamma and dataset); honest-but-curious threat model stated; communication cost formula (1+ceil(E'/m))x.
- 3.7 Attack-surface analysis: mechanism-specific attack derived from the weight-update gradient (P_fake = P_global - eta_fake * grad_P L'); why up-weighting unusual updates is exploitable by construction.
- 3.8 Implementation/verification discipline (one subsection, what makes this internship's methodology unusual): completeness gates + correctness audits + 565-test suite + CI; "never read a number back from the document asserting it"; provenance-embedded JSON artifacts.

### Chapter 4 - Experimental Results (~6-7 pages)

- 4.1 Setup: datasets (CIC-IDS-2017 2,830,743 rows/78 features/15 classes, UNSW-NB15 39 features/10 classes; frequent classes capped at 10k; one attack class per client K=14/9; 10% per-client validation split; SHA-256-verified provenance), architecture (Dense 200->100->50 tanh -> softmax), training (Adam eta=0.01 for BOTH client updates and the w-update per the manuscript's own correction note, E=2, B=64, T=200, eta'=0.1 literal / 0.01 corrected, E'=5; seed regimes: 3 seeds Stage A, 10 seeds Stage B ablation + Stage C mains, 5 seeds T7/T7G), compute (CPU-only by measurement - GPU attempt measured 25% slower at this model size; Modal cloud for the 960-cell campaign, pinned env), statistical protocol (bootstrap 95% CI, paired Wilcoxon, Holm-Bonferroni). 19 baselines enumerated by family.
- 4.2 Stage A - reproduction vs claims: the headline paper-vs-reproduced table (CIC 88.88% claimed -> 0.00% reproduced; UNSW 21.09% claimed -> 33.33% reproduced - UNSW literal nominally exceeds the claim but is seed-unstable, see 4.3; the paper's "Acc." is itself one of the 17 flagged ambiguities) + gamma-sweep table covering BOTH datasets (CIC: FedAvg 61.0/0.0, Static 52.3/91.7, FedGRAA g0 62.5/0.0, g0.5 18.6/0.0, g1 17.4/0.0; UNSW: FedAvg 50.0/0.0, Static 54.9/31.5, FedGRAA g0 49.6/0.0, g0.5 43.3/25.1, g1 38.0/33.3) + the two Figure-1 reproduction PNGs; n-sweep Figure 2-3 PNGs; the honest-divergence statement.
- 4.3 Diagnosis: T simplex-violation table (first_violation_round = 1 in 120/120 unprojected cells; 0/120 in projected cells, sum-w within ~1.2e-7 of 1); composite-loss blowup trace figure (literal composite loss PNG); weight-trajectory PNG; UNSW milder-but-same-mechanism note; seed-dependence finding (UNSW g0.5 per-seed rare-acc swings 0.4%-44.0%).
- 4.4 Stage B - the fix, statistically: the 2x2 factorial ablation tables (both datasets, the §5.1/5.2 master tables condensed), ablation PNGs; the dataset-dependent-lever finding (eta' dominates CIC, projection dominates UNSW, only both safe everywhere); dual-metric check (eta'-alone buys rare-acc parity at 15.2% all-class); blowup medians/maxes.
- 4.5 Stage C - FedGRAA-v2 frontier: frontier_figure.png; condensed per-arm table (top rows + best baseline); 93.47%/69.66% best-m; v1_projected parity (privacy fix costs no accuracy).
- 4.6 Significance: 111/112 Wilcoxon-Holm; the one exception reported (FedAWARE vs v2_m1 CIC, p_holm=0.0625); per-arm significance table (13/14, 14/14 x3, 14/14 x4 UNSW).
- 4.7 Communication overhead: overhead_ratio.png + table (6.00x m=1, 4x, 3x, 2x; formula (1+ceil(E'/m))x exact); classic_fl_comparison.png (SCAFFOLD 31.67%/30.34% rare-acc, the other four classics at or near 0% - FedAvg 0/0, FedProx 0/1.33, FedOpt 0/0.44, FedNova 0/4.62).
- 4.8 Adversarial robustness (T7+T7G): t7_attacker_weight_trajectory.png (attacker up-weighted to ~3.1x/~2.7x uniform share at 30% attacker fraction); defense ASR table (SecFedNIDS misses 53-75%, WeiDetect 0-20% static / 60% adaptive on CIC; full 6-row grid from STAGE_C S10.4); the consequential finding (successfully-defended cells collapse rare-recall to ~0%: exactly 30/30 CIC, 19/30 UNSW SecFedNIDS+WeiDetect cells - recompute per §6, do NOT claim all UNSW cells); AdaAggRL rare-recall collapses to 0.0% CIC / ~2% UNSW under label-flip; T7G gamma table (non-monotonic in gamma, peaks ~0.5 in 3/4 cells, g0 seed-to-seed std-dev 10-20x larger than g>=0.5).

### Chapter 5 - Discussion (~2-2.5 pages)

- 5.1 What the reproduction gap means for FL-IDS evaluation practice (single-run unseeded claims vs 10-seed reality; UNSW's literal-arm variance is the worse failure mode).
- 5.2 The gamma trade-off and attack surface are the same mechanism (rare-class sensitivity IS the poisoning exposure; non-monotonic in gamma; threshold hypothesis at ~0.5, 3-point grid caveat).
- 5.3 No defense is free: external filters discard the up-weighting signal; a defended FedGRAA-v2 preserving rare-class sensitivity is open.
- 5.4 Exactness vs communication cost: m=3 non-obvious optimum; when 6x traffic is/isn't worth it vs near-zero-cost approximations (CELM/SpectralFed ~0% rare-recall).
- 5.5 Limitations (honest list): honest-but-curious server only; plaintext parameter channel still inversion-exposed; manual gamma; n=3 protected classes; K=9-14 clients (not 50-200 silo scale); Adam-as-local-solver deviation for SCAFFOLD/FedNova theory; a second documented deviation - per-client Adam optimizer state is built once and never reset across rounds (`INDEPENDENT_FULL_AUDIT_2026-07-26.md`, labeled internally as "a known deviation, not a bug"); a reachable-but-untriggered NaN path plus a mislabelled warning string in `aggregation.py` (same audit; no NaN found in any persisted trace, so no reported number is corrupted); NF-UQ-NIDS-v2 real-silo migration measured only as T6 benchmark, not the headline grid; UNSW n=7 infeasibility of the original paper's own sweep; provenance of the reproduced manuscript unresolved (stated neutrally).
- 5.6 Engineering/research-methodology takeaways (what the internship demonstrated): reproduce-then-audit discipline, completeness gates vs correctness audits, AI-assisted drafting with disclosure, mentor-decision vs engineering-decision separation. This subsection is what distinguishes an *internship* report from a paper - include it.

### Chapter 6 - Conclusion (~1-1.25 pages)

- Recap of contributions mapped to objectives O1-O5 with the headline numbers.
- Future work: Option A few-shot/prototype hybrid (already scoped in `REMAINING_COMPUTE_WORK.md` S3a); automated gamma (XAI-SOH-FL precedent); rare-class-preserving defense; extended threat model; real multi-silo evaluation at scale; PAKDD resubmission status (`docs/paper/FedGRAA_v2_PAKDD_Submission.{md,tex}`).

### Bibliography (~1.5-2 pages, ~40 entries)

All entries exist in the repo's multi-pass-verified catalog (`FedGRAA_v2_PAKDD_Submission.md` References + `Final_Stage_C.md` S11). Full BibTeX entries for: FedAvg, FedProx, FedOpt/Reddi, SCAFFOLD, FedNova, FedGRAA-target-paper placeholder (cited as "anonymized manuscript under review" or the mentor-provenance handling - see Open Question), FedAWA, FedAWARE, FedNolowe, PACT, MESAS, CrowdGuard, FreqFed, Jiang2024, AdaAggRL, CELM, SpectralFed, CoAst, FLTrust, RAGA, FedTruth, FedMADE, SBFedNIDS, FedAPA, DA-PFL, FedPGO, NIDS-FGPA, SecFedNIDS, WeiDetect, Bonawitz2016 SecAgg, Duchi2008 projection, Karimi2016 PL condition, CIC-IDS-2017 (Sharafaldin), UNSW-NB15 (Moustafa), NF-UQ-NIDS-v2, XAI-SOH-FL, MaYang2026 survey, pill-attack Guo2026, McMahan/Kairouz FL surveys if cited.

### Appendix A - Reproducibility manifest (~1 page, optional within budget)

Environment pin, seeds, the nine CI verification commands, artifact map (JSON tree), git-commit provenance note. Included only if page budget permits; otherwise a paragraph inside 5.6/4.1.

---

## 5. Figure inventory (all exist; copy into `internship_report/figures/`)

| Report fig | Source file (copy verbatim) | Used in |
|---|---|---|
| F4.1a/b | `FedGRAA/results/figures/CIC-IDS-2017_figure1.png`, `UNSW-NB15_figure1.png` | 4.2 (reproduction gap) |
| F4.2a/b | `FedGRAA/results/figures/CIC-IDS-2017_figure2_3.png`, `UNSW-NB15_figure2_3.png` | 4.2 (n-sweep) |
| F4.3 | `FedGRAA/results/figures/diagnosis_literal_composite_loss.png` | 4.3 (Lemma-1 violation trace) |
| F4.4 | `FedGRAA/results/figures/diagnosis_literal_weights_g1p0.png` | 4.3 (weight leaves simplex) |
| F4.5a/b | `FedGRAA/results/ablation/CIC_IDS_2017_ablation.png`, `UNSW_NB15_ablation.png` | 4.4 (2x2 ablation) |
| F4.6 | `FedGRAA/results/v2_frontier/frontier_figure.png` | 4.5 (v2 frontier) |
| F4.7a/b | `FedGRAA/results/v2_frontier/overhead_ratio.png`, `classic_fl_comparison.png` | 4.7 |
| F4.8 | `FedGRAA/results/v2_frontier/t7_attacker_weight_trajectory.png` | 4.8 (attack up-weighting) |
| F4.9 | `FedGRAA/results/v2_frontier/t7_defense_asr.png` | 4.8 (defense ASR) |
| F3.1 (optional) | `FedGRAA/docs/paper-files/figures/Figure1.png` (paper's own fig, only if a side-by-side "claimed vs reproduced" panel is wanted - verify licensing/fidelity note first; default: skip) | 3.3 |

All confirmed present and renderable (visually inspected: frontier, ablation, figure-1, diagnosis-loss PNGs are real plots with legends/CI).

## 6. Number-source map (audit trail: every reported number -> its file)

| Report claim | Source of truth |
|---|---|
| 0% rare-acc literal vs 88.88% claim; gamma-sweep table | `results/{CIC-IDS-2017,UNSW-NB15}_gamma_sweep.json` + `STAGE_A_MASTER_REPORT.md` S6 |
| 120/120 round-1 simplex violations (unprojected arms; projected 0/120); 1.86e6x / 46.85x max blowups | `results/diagnostics/*.json` + `results/ablation/*_ablation_table.json` via `audit_stage_b.py` + `STAGE_B_MASTER_REPORT.md` |
| Ablation arm table (89.3% [83.6,94.6] CIC, 69.2% [63.4,73.8] UNSW at gamma=1) | `results/ablation/*_ablation_table.json` + `fedgraa_convergence_defect.md` S5 |
| v2 frontier 93.47%/69.66% m=3, per-arm recalls, traffic GB | `results/v2_frontier/frontier_table.{json,md}` + `STAGE_C_MASTER_REPORT.md` S6 |
| 111/112 Holm-significant; FedAWARE p=0.0625 exception | `results/v2_frontier/significance_table.{json,md}` |
| 6.00x/3.00x overhead, (1+ceil(E'/m))x formula | `results/v2_frontier/overhead_table.{json,md}` |
| Attack up-weighting 3.1x/2.7x; T7G gamma table | `results/v2_frontier/*v2_raresig*.json` + `STAGE_C_MASTER_REPORT.md` S10 |
| Defense ASR grid (75/65/55, 15/20/60, 60/60/53.3, 0/6.7/33.3) | `STAGE_C_MASTER_REPORT.md` S10.4 + T7 JSONs |
| Defended-cell rare-recall collapse (30/30 CIC, 19/30 UNSW) | recomputed per-cell `final_last_n_acc == 0` over `*secfednids*`*`*weidetect*` T7 JSONs - NOT a printed table; the 19/30 UNSW bound is load-bearing, do not round up |
| AdaAggRL rare-recall collapse (0.0% CIC / ~2% UNSW under label-flip) | `results/v2_frontier/*adaaggrl_labelflip*.json` `final_last_n_acc` + `STAGE_C_MASTER_REPORT.md` S10.4 finding text + Appendix B (§17) per-arm table |
| T7G attacker up-weighting (2.05/3.34/3.07, 2.33/2.91/2.71 at 30%; g0 std-dev 10-20x) | `results/v2_frontier/*v2_raresig*_seed*_af*.json` `attacker_weight_history[-1]` x K clients |
| Classic-FL comparison (SCAFFOLD 31.67/30.34; others at/near 0, max 4.62 FedNova UNSW) | `results/v2_frontier/frontier_table.json` classic arms + `STAGE_C_MASTER_REPORT.md` S9 |
| ~1-4e-7 exactness (<1e-6 test bound) | `scripts/verify_v2_exactness.py` + `test_aggregation_v2.py` + manuscript S5 |
| 960 cells, 565 tests, 19 baselines | `STAGE_C_MASTER_REPORT.md` S1/S12 + `pytest --collect-only` |
| 17 paper ambiguities (incl. retracted #9 handling) | `docs/current/PAPER_AMBIGUITY_REPORT.md` |
| 4 bibliography defects incl. 1 fabricated ref | `docs/audit/INDEPENDENT_FULL_AUDIT_2026-07-26.md` |

Rule: report text cites numbers rounded exactly as in the generated `*_table.md` files; where the manuscript and master report differ in phrasing, `STAGE_C_MASTER_REPORT.md` is authoritative (its own §0 declares this).

## 7. Implementation steps (execution order)

1. `mkdir internship_report/{figures,build}`; copy the 13 PNGs from §5.
2. Install tectonic (winget or portable zip); verify `tectonic --version`.
3. Write `references.bib` (all ~40 entries; keys = short names used in the manuscript, e.g. `fedavg`, `fedpgo`, `secfedsnids`, `cicids2017`, `unsw nb15` -> normalized keys).
4. Write `report.tex` front matter block (custom title/certificate/acknowledgement/abstract pages; `\pagenumbering{roman}`; TOC/LoF/LoT; nomenclature table; `\pagenumbering{arabic}` at Ch 1).
5. Draft Ch 1-6 per §4, pulling prose from the repo's own documents where possible (the manuscript's Related Work and Methodology sections are already venue-vetted and compress well; stage reports supply results text). Keep paragraphs tight; every figure referenced in text; every table `booktabs`.
6. Compile: `tectonic --outdir build report.tex`; iterate until 0 errors; inspect warnings (overfull hboxes, undefined refs all resolved).
7. Page-count pass: adjust to land 20-25 arabic-folio pages via the knobs in §9 (Option A default).
8. Self-audit pass: cross-check every number in the PDF against the §6 source files (spot-check all headline figures + every table cell); grep the .tex for "—" (must be zero); verify all placeholders are `<...>`-wrapped and listed in REPORTING_NOTES.md.
9. Write `REPORTING_NOTES.md` (placeholder fill list + recompile commands + Overleaf fallback).
10. Update root README/docs map? NO - repo convention is docs live under FedGRAA/docs; the internship_report/ folder is self-contained with its own README pointer inside REPORTING_NOTES.md. Leave the repo's navigation untouched (out of scope).
11. `git add internship_report/` (figures are binaries, fine to commit per repo's own artifact-committing convention; build/ gitignored).

## 8. Equations and algorithms to typeset (LaTeX-ready)

- FedAvg: `θ_g^{t+1} = Σ_k (|D_k|/|D|) θ_k^{t+1}`; Static: `θ_g^{t+1} = Σ_k [(1/|D_k|)/Σ_j(1/|D_j|)] θ_k^{t+1}`
- Composite: `L_a = min(L_h, L_m)`; `L'(w) = (1-γ)L_a + γL_u`; `∂L'/∂w_k = ⟨∇_P L'(P), P_k⟩`
- Simplex: `Δ = {w ∈ R^K : w_k ≥ 0, Σw_k = 1}`; corrected update `w^t = Π_Δ(w^{t-1} - η'∂L'(w^{t-1}))`
- Gradient mapping + rate: `G_η'(w) = (w - Π_Δ(w - η'∂L'(w)))/η'`; `min_{t≤T} ‖G_η'(w^t)‖² = O(1/T) + O(ζ²)`
- Two-phase exactness: `∇_P L' = Σ_k ∇_P L_k'(θ'_g)`; overhead `(1+⌈E'/m⌉)×`
- Algorithm floats: Algorithm 1 (federated loop), Algorithm 2 literal (as published, marked "as printed - defective"), Algorithm 2' corrected (projected, marked "proposed"), FedGRAA-v2 two-phase protocol box.
- Greek letters in nomenclature sorted per IITB convention.

## 9. Page-budget knobs (to land 20-25 arabic-folio pages)

**Interpretation (locked):** the 20-25 target counts arabic-folio pages = Ch 1 Introduction through Bibliography. Roman-folio front matter (~6-7 pages: title, certificate, acknowledgement, abstract, TOC, LoF+LoT, nomenclature) is additional. Estimated total document: ~28-31 pages.

- **Option A (default): 20-25 arabic pages.** Base estimate ~22-24 at the §4 budgets. If over 25: shrink figure widths (full-width -> 0.8), merge the F4.2 n-sweeps into one line-reference, tighten Related Work by folding 2.5/2.6, move Appendix A out. If under 20: restore full ablation tables (the condensed versions), add the Stage-A n-sweep full per-cell table, expand Appendix A with the artifact manifest, add "Chapter summary" paragraphs IITG-style at each chapter end.
- **Option B (contingency, only if the user's institute counts TOTAL document pages):** compress front matter to ~4-5 pages (LoF+LoT share one page; nomenclature folded to a compact table), then re-budget chapters to ~13-16 arabic pages: Ch1 2, Ch2 2.5, Ch3 4, Ch4 5 (9 figure floats -> 6 via tighter subfigure grouping; ablation tables merged to one), Ch5 1.5, Ch6 1, Refs 1.5-2. Total lands ~20-24. This trades Related Work and Methodology depth for the stricter count - do not switch to B without user confirmation, since A is the standard IIT reading.

## 10. Acceptance criteria (definition of done)

- [ ] `tectonic --outdir build report.tex` exits 0; zero undefined references/citations; no missing figures.
- [ ] `report.pdf` has 20-25 arabic-folio pages (Ch 1 through Bibliography); ~6-7 roman-folio front-matter pages on top.
- [ ] Structure = front matter (title/certificate/acknowledgement/abstract/TOC/LoF/LoT/nomenclature) + chapters Introduction, Related Work, Proposed Methodology, Experimental Results, Discussion, Conclusion + Bibliography (+ optional Appendix).
- [ ] All 13 figures from §5 render; all tables use booktabs; every figure/table referenced in text and numbered.
- [ ] Every quantitative claim matches its §6 source (spot-check of every headline number + every table cell).
- [ ] Zero "—" characters in report.tex (user rule); zero fabricated references; placeholders clearly marked.
- [ ] Honest-negative-results preserved (111/112 not 112/112; weak SecFedNIDS reproduction; defended-cell collapse stated as exactly 30/30 CIC and 19/30 UNSW - never rounded up to "all"; AdaAggRL collapse; UNSW literal-arm seed variance).
- [ ] `REPORTING_NOTES.md` lets the user fill placeholders + recompile unaided.
- [ ] No changes outside `internship_report/` except the `.gitignore` build-dir line.

## 11. Open questions for the user (non-blocking; defaults chosen)

- Provenance framing: the reproduced manuscript is anonymized with unresolved publication status and circumstantial evidence linking it to the mentor's own earlier work (`docs/audit/MENTOR_*` docs). Default plan: refer to it neutrally as "the FedGRAA manuscript under reproduction (PAKDD submission draft)" and keep provenance discussion OUT of the report body; mention the unresolved status once, in Discussion S5.5, exactly as the repo's own public README does. The user can decide later whether to name it differently.
- Internship dates/duration: left as placeholders (`<Start Date>` - `<End Date>`); git history spans 2026-06-28 to 2026-08-13 (~7 weeks, Jun-Aug 2026) but is not asserted in the report.
- Whether the report should mention AI-assisted work at all: default = one sentence in Acknowledgement that tooling assistance was used under the intern's direction, matching the manuscript's own AI disclosure ethos; flagged for user confirmation since it is a judgment call, not a default the repo dictates.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| No LaTeX toolchain; winget blocked | Portable tectonic zip; Overleaf zip-upload path documented in REPORTING_NOTES.md as fallback |
| 20-25 page target missed | Budget table + knob list (§9) applied in a dedicated sizing pass before delivery |
| A number drifts from source | §6 source map + the self-audit step; the repo's own `audit_stage_*.py` already guarantee the underlying numbers |
| Report reads like the paper, not an internship | Ch 1 objectives, Ch 4 staged narrative (A->B->C), and Discussion 5.6 are internship-framing by design; the report's voice is first-person-singular research-intern, not anonymized-submission |
| Em-dash creep from copied prose | `grep -c "—"` gate in the self-audit (must be 0); repo prose itself uses "—" heavily so pasted text must be scrubbed |
