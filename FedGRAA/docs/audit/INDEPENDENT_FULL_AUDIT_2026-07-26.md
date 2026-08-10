# Independent Full-Repository Audit — FedGRAA / PAKDD

**Date:** 2026-07-26
**Auditor:** Claude (Sonnet 5), acting as an independent senior-researcher reviewer, on explicit instruction to "grill and scrutinize" the entire repository line by line and hold nothing back.
**Status of this document:** static snapshot as of the commit below. Not wired into any gate. Update the status column or supersede this file if items close — see §7 for why that specific failure mode is the one thing this repo keeps getting wrong.
**Reviewed at:** `HEAD = 0720010` ("Merge pull request #1 ... Audit remediation 2026-07-25"), working tree clean.

## 0. Method

This audit did not take any prior document's word for anything. Five independent review passes were run in parallel, each with its own scope and instructed to verify claims against primary sources (raw JSON, git history, the actual test files, external web search for citations) rather than against other documents in the repo:

1. **Core algorithm & statistics** — every line of `fedgraa/*.py`, cross-checked against the four current theory/reproduction docs.
2. **Documentation-trail consistency** — all 8 `docs/audit/*.md` files, all 11 `docs/current/*.md` + status files, cross-checked against `git log` and by actually re-running the gate scripts.
3. **Tests & completeness gates** — all ~20 files in `tests/`, plus every script in `scripts/`, checked for whether they verify *values* or only *structure*.
4. **Numeric claims vs. raw data** — every specific number quoted in the top-level docs, re-derived independently from the underlying `results/**/*.json` files with a fresh Python session, not by re-reading prose.
5. **Paper draft & research corpus** — the actual submission draft, the AI-simulated peer review, the `understanding/` series, and the `research_c/` literature corpus, including external verification of citations via web search.

I then independently re-verified the single highest-stakes claim myself (paper abstract vs. Stage B measured numbers) by reading the primary files directly rather than trusting any agent's report. Findings below are ranked by actual consequence, not by where they were found. Every item cites a file and, where applicable, a line number.

**Top-line verdict, stated plainly:** the *science* in this repository is unusually well-verified — on independent re-derivation, essentially every headline statistic (the 120/120 simplex-violation count, every p-value/Cohen's d/Holm-corrected value in both ablation tables, every factorial effect, all internal mean/median arithmetic) reconciles exactly against raw data. That is a genuinely rare property for a repo this size and worth stating up front so it isn't buried under the rest of this report. The problems are concentrated in three other places: **(a)** the actual paper draft makes claims the project's own data contradicts, **(b)** a small number of real, non-cosmetic code bugs exist in the aggregation/metrics layer, and **(c)** the verification *infrastructure* — the gates and the audit-tracking documents themselves — has a self-referential blind spot: nothing audits the auditors, and the one gate humans actually rely on (`check_stage_*_complete.py`) checks shape, not substance.

---

## 1. SEVERITY 1 — The paper draft itself makes claims the project's own evidence contradicts

This is the most consequential tier because it's the one artifact meant to leave the repository. Everything else here is infrastructure; this is the output.

### 1.1 The abstract's headline numbers are not reproducible from the algorithm as published

`PAKDD/docs/paper/PAKDD_Final_Version_Submit_v3.md:7` (Abstract), verified verbatim:

> "Extensive experiments on CICIDS 2017 and UNSW-NB15 demonstrate that FedGRAA achieves 88% accuracy on minority classes, significantly outperforming *FedAvg* while preserving client privacy. Additionally, as the number of clients with minority samples increases, FedGRAA maintains its advantage, outperforming *FedAvg* by up to 55%."

This project's own Stage B ablation — independently re-verified from raw JSON in this audit, not just quoted from the doc — measured the algorithm **exactly as printed** (Algorithm 2, Eq. 10, literal gradient descent, no simplex projection, η′=0.1 as the paper states), 10 seeds, and found (`docs/current/fedgraa_convergence_defect.md:151`):

| Arm | γ=0.0 | γ=0.5 | γ=1.0 |
|---|---|---|---|
| Literal (η′=0.1, no proj. — **the paper as published**) | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | **10.0% [0.0, 30.0]** |

The 88.9% figure the abstract's "88%" apparently traces to only appears at line 153–154 of the same table, under an arm the paper never describes: **η′ silently lowered from the paper's own stated 0.1 to 0.01**, and/or an **undisclosed simplex projection added** that Algorithm 2 as printed does not contain. Both changes are this project's own corrective invention (Stage B's contribution), not anything in the submitted algorithm box. I confirmed this discrepancy myself by reading both files directly — it is not merely an audit-agent's inference.

**Why this matters:** the abstract is presenting a number this project has already, rigorously, shown is not obtainable from the algorithm as specified, under any of the 10 tested seeds. If this draft goes out as-is, it makes exactly the claim Stage B exists to refute.

The "outperforming FedAvg by up to 55%" clause has no traceable source anywhere in the draft's own figures or tables — no caption or in-text computation produces "55%." (`understanding/04_experiments_and_results.md` §4.6 guesses it derives from the n-sweep panels, but that is this project's own inference after the fact, not something the paper computes.)

### 1.2 The convergence guarantee is stated unconditionally, against the paper's own stated architecture

`PAKDD_Final_Version_Submit_v3.md:147`: "ensuring better generalization and faster convergence (see Section A.1 for convergence proof)." Appendix A.1's Assumption 1 requires the composite loss to be **convex and β-smooth**; §5 of the same paper describes the model as a multi-layer tanh-MLP with softmax output — not convex. `docs/current/fedgraa_convergence_defect.md` §3–4 already contains this project's own detailed account of exactly why this assumption fails and what the honestly-weaker replacement guarantee is (stationarity on the simplex, not global optimality). The paper's methodology section states the strong claim with no qualification; nothing in the draft signals to a reader that this project has already derived, in the same repository, why that specific claim doesn't hold for its own network.

### 1.3 The summary-table numbers are statistically implausible as genuine measurements

`PAKDD_Final_Version_Submit_v3.md:893` and surrounding cells: 88.88 / 66.66 / 77.77 / 33.33 — simple repeating-decimal fractions (8/9, 2/3, 7/9, 1/3). `docs/current/PAPER_AMBIGUITY_REPORT.md` items #8–#9 (this project's own prior audit) already flags this and additionally notes long flat stretches across dozens of evaluation rounds in Figures 2–3, "far more consistent with hand-picked placeholder values... than with genuine measured neural-network accuracies." This is independently corroborated by the Devil's-Advocate persona in `FedGRAA_Simulated_Peer_Review.md:339`, arriving at the same conclusion via a different route. Three independent readings (this project's own ambiguity report, the simulated review, and this audit) converge on the same statistical-implausibility finding — worth treating as more than a hunch.

### 1.4 Privacy is claimed unconditionally against a method that requires a server-side labelled validation set

Abstract: "preserving client privacy." Algorithm 1/2 require the server to hold a labelled `D_V` and to know in advance which classes are rare — named explicitly in §5 (Heartbleed, SQL Injection, Infiltration). `docs/research_c/privacy_compliant_fedgraa_redesign.md` §1 (this project's own document) states this "already contradicts federated learning's own founding privacy premise." The abstract's claim is unqualified against a gap this project has already written 700+ lines about internally.

### 1.5 Citation problems in the paper's own reference list

- **Reference [12] ("FedSND," Zhang et al.) is a fabricated citation.** `docs/research_c/STAGE_C_LITERATURE_MASTER.md` §3.2 confirms zero trace across arXiv, web search, and DBLP, across two independent passes. Independently and more simply: the reference gives arXiv ID `2306.14443` (June 2023) while claiming publication at "the 38th ICML... (2020)" — three years *before* the cited preprint was uploaded, and ICML's 2020 edition was in fact the 37th, not the 38th. Cited in the Introduction as motivating prior work.
- **Reference [8] (Reisizadeh, Hassani, Mokhtari — "The Non-IID Data Quagmire...")** is, per external verification, actually authored by Hsieh, Phanishayee, Mutlu & Gibbons (ICML 2020/PMLR), not the cited authors, and was not published in IEEE Trans. Signal Processing as stated.
- **Reference [4] (Kim et al., "FedFN")** — arXiv ID `2311.13267` (November 2023) is cited as "NeurIPS (2021)" in the same reference entry — a self-contained internal contradiction, no external check required. The paper is actually a NeurIPS 2023 workshop paper.
- **Reference [11] (Wang, Zhang, Zhang)** — cited as ICML 2021/PMLR, but its own URL points to an IEEE Xplore *journal* article (`computer.org/csdl/.../2024/`), not conference proceedings — venue and year both disagree with the citation's own link.
- Two in-text citation mismatches against the bibliography's own author order ("Lee & Kim [4]" for a paper whose first author is Kim; "Pranab et al. [9]" citing by first name rather than surname).

None of this appears to have been cross-checked against `STAGE_C_LITERATURE_MASTER.md`, which already flags [12] as fabricated, before the draft was assembled.

### 1.6 Where the paper draft is on solid ground

Worth stating for balance: the underlying comparative shape of the results (FedGRAA favoring minority classes as γ increases, at some cost to global accuracy) is directionally consistent with what Stage A's faithful reproduction actually measured, and the paper's conclusion section is comparatively hedged relative to its abstract. The problem is concentrated in the abstract/intro's specific numbers and the unconditional convergence/privacy claims, not the paper's overall shape.

**Scope caveat, stated in the project's own `publication_roadmap.md` §6 and worth repeating here:** the provenance of this draft (own work vs. third-party manuscript being reproduced-and-critiqued) is explicitly unresolved in this repository — DBLP/Semantic Scholar searches for the title return zero hits. Every finding in this section is about the draft as it currently sits in the repo; whoever owns this project needs to resolve whose claims these are before deciding what "fixing the abstract" even means.

---

## 2. SEVERITY 2 — Real code-correctness bugs

### 2.1 `composite_loss()` and `_composite_grad_P()` handle NaN asymmetrically — confirmed by direct interpreter test

`fedgraa/aggregation.py:111` (`composite_loss`) vs. `:137` (`_composite_grad_P`):

```python
L_a = min(L_h, L_m)                    # line 111 — Python min(), not NaN-symmetric
...
if L_h <= L_m:                         # line 137 — <=, not NaN-symmetric
    L_a, g_a = L_h, g_h
else:
    L_a, g_a = L_m, g_m
```

Verified directly: `min(5.0, nan) = 5.0` (NaN silently vanishes; `isfinite()` reports `True`), while `min(nan, 5.0) = nan` (correctly caught). The two "parallel" computations of the same quantity therefore **disagree on which of the two NaN cases they catch**:

- If `L_m` is NaN and `L_h` finite: `composite_loss()` silently drops it (`min(L_h, nan) = L_h`), no warning — and this is the value that gets **persisted to `TrainHistory.composite_loss` and plotted**. `_composite_grad_P` correctly catches this same case.
- If `L_h` is NaN and `L_m` finite: it's the reverse — `composite_loss()` correctly flags it, but `_composite_grad_P` silently falls through with no warning, even though this is the value that actually drives the weight update.

Given the documented blowups reach 1.86×10⁶× the initial loss (`fedgraa_convergence_defect.md:151`), a NaN is not a far-fetched event in this code path. No NaN values were found in the currently-persisted per-round traces (all reported blowups are large-but-finite), so this bug likely has **not** corrupted any number already published — but it is a live, reachable path, not a hypothetical, and the secondary warning text at `aggregation.py:115-120` ("D_V^h and D_V^m are both empty") is factually wrong when the actual trigger is a numerical blow-up rather than an empty split — it would mislabel a real instability as a data-partitioning edge case.

### 2.2 `last_n_class_accuracy` silently shrinks its own denominator

`fedgraa/metrics.py:116-123` drops a rare class from the mean entirely if it has zero samples in that evaluation round's split (plausible for classes as small as 11 total samples), with no warning. The paper's headline metric can silently become an average over fewer classes than claimed, for a subset of rounds/seeds, with no visible signal anywhere in the output.

### 2.3 Validation-set statistics leak into feature scaling before the train/val split

`fedgraa/data.py:264-266` computes per-feature mean/std over the **entire** pooled dataset before `partition_non_iid` splits off each client's validation slice — so the scaling statistics used for training already incorporate the rows that later become `D_V^h/D_V^m/D_V^n`. Likely small in absolute effect given large N, but it is real, undisclosed leakage.

### 2.4 Round-0 accuracy is hardcoded to 0.0, not measured

`fedgraa/config.py:62-67` (`round_zero_zero: bool = True`, default on) forces round-0 accuracy to display as exactly 0.00% "to match the paper's table," per its own docstring, even though a randomly-initialized classifier does not actually score 0%. This is transparently documented in code but not flagged anywhere a reader of the reproduced tables would see it — someone skimming the output would reasonably (and incorrectly) conclude 0% was measured.

### 2.5 Client-side Adam momentum persists across all 200 rounds while model weights reset every round ("Gap 2.2")

`fedgraa/trainer.py:280-288`: each client's optimizer is built once and never reset, even though `client_train_round` resets the *model* parameters to the aggregated global every round. Already known and labeled internally (`docs/archive/v1_regression_root_cause_analysis.md:245-247`, called "a known deviation, not a bug") — but it is **not mentioned in any of the four current theory/reproduction docs**, meaning a reader of the current doc set has no way to know this second, independent deviation from Algorithm 1 exists alongside the headline simplex-projection defect.

### 2.6 Lower-severity/latent items (not observed to have affected any published number)

- `metrics.MLP.predict()` and `aggregation.FedGRAA`'s internal helper `MLP` both hardcode `seed=0`, decoupled from the experiment's actual seed — currently inert because every call site overwrites weights via `set_params()` first, except the unused `forward()` method (dead code — no call site anywhere in the package).
- `stats.wilcoxon_signed_rank` (`stats.py:115-118`) catches bare `except ValueError` around `scipy.stats.wilcoxon`, which would also swallow a genuine caller bug (e.g., mismatched array lengths), not just the intended degenerate case.
- `experiment._run_single_config` (`experiment.py:132-143`) catches worker exceptions in the parallel sweep and **drops the config from the results dict** rather than raising or recording the failure visibly — a partial results dict from a crashed worker is externally indistinguishable from one that was never requested. `run_n_sweep_parallel` has zero test coverage (see §3).
- `median_frequency_split` (`partition.py:230-243`) has an uncovered edge case (exactly one non-rare class remaining) that would silently leave `D_V^h` empty with no warning — not triggered by either shipped dataset at tested `n`, but undocumented.
- `paired_ttest` and `cohens_d_paired` (`stats.py`) return contradictory-looking fallbacks for the same degenerate input (`(0.0, 1.0)` "no effect" vs. `±inf` "maximal effect") — each individually well-reasoned and disclosed in its own docstring, and both are reported side-by-side in the actual tables, but a reader checking only one column could see an apparently contradictory pair of numbers.
- Blanket `np.place(X, ~np.isfinite(X), 0.0)` (`data.py:258`) zero-fills CIC-IDS-2017's known `Infinity` artifacts (zero-duration flows) with no count or per-class breakdown printed, unlike the label-drop path a few lines earlier which does print a count. Worth a spot check given how central rare-class recall is to the paper's claim.
- Two near-duplicate fused training-step implementations (`model.py:252-284` and `:287-314`) that could silently diverge under a future edit to one but not the other (verified currently consistent).

---

## 3. SEVERITY 3 — The verification infrastructure has a structure-vs-substance gap

### 3.1 The two completeness gates check shape, not values

`scripts/check_stage_a_complete.py` and `scripts/check_stage_b_complete.py` — the two scripts every top-level doc points to as the authoritative "is this stage done" signal — verify that JSON keys exist, that files are non-empty, that a markdown doc contains certain substrings, and that `per_seed` lists have the right length. **They never recompute a mean, CI, or p-value from the persisted per-seed data and compare it to what's stored in the summary.** A hand-edited or stale `mean`/`p_holm`/`cohens_d` sitting next to `per_seed` data that implies something completely different would pass both gates cleanly.

This is *not* uncaught in the abstract — separate scripts, `scripts/audit_stage_a.py` and `scripts/audit_stage_b.py`, do exactly this recomputation correctly and rigorously (independently re-derived via SciPy, with documented negative controls). But **nothing forces anyone to run them** — there is no CI, no Makefile, no pre-commit hook anywhere in the repository (confirmed by search). They are two more scripts a human has to remember to run, and the numeric-verification pass in this audit (§4 below) is, in effect, the first time in this audit cycle that the *values* — not just the presence — of the headline statistics were independently checked against raw data by an entity other than the project's own audit script.

### 3.2 The model's actual prediction function is mocked out in every test that touches it

`MLP.predict()` (`model.py:146-151`) is never exercised as a real forward pass anywhere in the test suite — every test in `test_metrics.py` monkeypatches `metrics.MLP.predict` to a fixed lambda. A real bug in `predict()` (wrong axis, stale weights, off-by-one) would be caught by none of these tests.

### 3.3 The project's own "headline result" test would pass on a tie

`tests/test_trainer.py::test_fedgraa_beats_fedavg_on_rare_class` (lines 46-58), commented as testing "the whole point of the paper," asserts `graa.final_last_n_acc >= avg.final_last_n_acc`. `>=` rather than `>` means this test provides no protection against a regression to exact parity between FedGRAA and FedAvg.

### 3.4 Other tests that verify callability/shape, not behavior, for code paths with real logic

- `tests/test_parallel_sweep.py` (entire 27-line file) only asserts the parallel-sweep function is callable and has the right parameter names — never runs it. The function it "tests" contains real orchestration logic including the silent-exception-swallow noted in §2.6.
- `tests/test_checkpoint.py::test_optimizer_state_extraction` — docstring admits "we just verify the signature" — the function determining whether checkpoint/resume actually restores optimizer momentum is never called.
- `tests/test_run_experiments_cli.py` tests a hand-duplicated copy of the real CLI's argparse definition, not the real parser; its one drift-guard is a raw substring search that would pass even if a flag's default value changed.
- Several `test_plot.py`/`test_render_ablation.py` cases check only `fig is not None` / `len(md) > 0`, though this is partially offset by other tests in the same files that do check exact line counts / formatted values.

### 3.5 Coverage gaps (zero test references, confirmed by grep)

`run_n_sweep_parallel`, `run_seed_sweep`, `summarize_gamma_sweep`, `tabulate_history` (all in `experiment.py`), the FedProx/FedOpt gradient math in `baselines.py` (only NaN-smoke-tested), and all `FedConfig`/`DatasetSpec` `__post_init__` validation paths (`config.py`) have no test exercising them at all.

### 3.6 Where the infrastructure is genuinely strong

The pytest suite (151 tests) is fully synthetic/self-contained — every test that needs data builds a small `DatasetSpec` fixture or writes to `tmp_path`; no test depends on the real committed CSVs or result files, which is a real reproducibility strength. `fedgraa/stats.py`'s degenerate-case handling (zero-variance CI, zero-variance t-test/Wilcoxon, signed-infinity Cohen's d) is deliberately designed, documented, and independently re-verified by `audit_stage_b.py`. `build_ablation_table.py`/`render_ablation.py` correctly report medians (not means) for blow-up ratios specifically to avoid a heavy-tailed mean misleading a reader, and this specific defense is itself tested. No hardcoded/fabricated numbers were found anywhere in the statistical pipeline scripts.

---

## 4. SEVERITY 4 — Documentation-trail integrity: the audit-of-audits problem

This repository has an unusual amount of prior self-auditing (8 documents in `docs/audit/`, several dated the day before this one). The pattern that emerges from checking them against each other and against `git log` is consistent: **whenever someone actually re-runs the numbers, they check out; the recurring failure is in the tracking layer that describes whether that happened.**

### 4.1 The remediation plan's own status column was never updated after its own items closed

`docs/audit/AUDIT_REMEDIATION_PLAN_2026-07-25.md` describes itself as "live — update the status column as items close" and lists twelve items, R1–R12, **every one still marked `Open` (R2 `Partly open`) in the file as it sits in the repo right now.** Cross-checked against `git log`: all twelve were in fact closed by dedicated, self-labeled commits (e.g. `eb23950` "...Addresses R3", `1ab0fbc` "...Addresses R12"), merged to `main` at `0720010`. The document whose entire purpose is "don't let this project's docs describe a state that no longer exists" is itself the most out-of-date document in the repository on this exact axis. This is worth fixing not because the underlying work is incomplete — it verifiably is complete — but because the failure mode is the one this whole audit trail exists to catch, occurring inside the artifact built to catch it.

### 4.2 A mojibake regression is silently present in both completeness gates' own success banners

Running `check_stage_a_complete.py` and `check_stage_b_complete.py` directly (not just reading about them) prints a replacement character (`�`) in place of an em-dash in the final "STAGE A/B COMPLETE" banner — a Windows-console-codepage encoding bug. This is the identical defect class `STAGE_B_CORRECTNESS_AUDIT_2026-07-25.md` (finding A5) already found and fixed in `audit_stage_b.py` specifically (reconfiguring `stdout` to UTF-8) — the fix was not applied to the two *completeness* gates every current document points to as the authoritative green signal. Non-fatal (exit code still 0), but no prior audit caught it because none of them re-ran the gates and read their literal console output.

### 4.3 Numeric drift between documents describing the same measurement

- **Test count:** root `README.md` says "142 passing tests" (stale); `PAKDD/README.md` correctly says 151 in prose but its own project-layout ASCII tree still says "142" three lines later, in the same file. Direct count (`def test_` across all `tests/*.py`) confirms **151 is current**; 142 was accurate as of an earlier commit, before `test_check_doc_links.py` (9 tests) was added.
- **"232" vs. "262" doc-transcribed numbers re-derived** — `results/ablation/README.md:31` and `STAGE_B_MASTER_REPORT.md:795` describe the identical audit check (`audit_stage_b.py` Check D) and disagree, apparently a digit transposition. Unresolved by this audit — no script-computed value for either number could be located to arbitrate.
- **"90 comparisons recomputed" vs. an actual count of 78** — `STAGE_B_MASTER_REPORT.md` §17.3 claims 90; direct enumeration of `comparisons` (30) + `comparisons_all_acc` (30) + `factorial` (18) in both ablation table JSONs totals 78. Unreconciled.
- **Two manifest "test case" counts off by one or two** against actual grep counts (`test_build_ablation_table.py`: claimed 13, actual 15; `test_render_ablation.py`: claimed 5, actual 6) — same drift pattern as the test-count issue, not independent.
- **A specific numeric correction (η′=0.001 vs. 0.01) that Stage A's own master report named by section number as needing a one-line fix** in `PAKDD_STATUS_REPORT.md` was never actually applied — the file was given a "superseded" banner instead, three audit passes after the error was identified, and the wrong sentence is still there underneath the banner.

None of these numeric-drift items touch a scientific conclusion; every one is stale bookkeeping the gates don't check because the gates check structure, not prose-number consistency (see §3.1).

### 4.4 The audits are honest about being AI-auditing-AI, and the one item that requires a human hasn't happened

Several audit docs explicitly disclose this ("This audit is itself AI-generated, and is subject to the same disclosure obligation," `RESEARCH_DIRECTION_AUDIT_2026-07-25.md` §6). That same document, and several after it, name **the manuscript-provenance question** (own work vs. third-party paper) as "the single highest-leverage open question... answerable in one conversation" — and, per every document's own disclosure, it still has not happened as of the latest commit. The project has closed every documentation-layer finding it could close by re-running scripts and editing prose, and left untouched the one finding that requires a human conversation outside the audit loop. `RESEARCH_DIRECTION_AUDIT_2026-07-25.md` finding D2 independently names this exact pattern for a different document (`STAGE_C_LITERATURE_MASTER.md`, audited four times in two days for "one citation upgrade... none of which changes a single decision") and correctly diagnoses it: *"auditing is legible progress and drafting is not."* That diagnosis applies to this document too, and to the fact that a sixth audit pass (this one) was run before the one blocking human conversation happened.

### 4.5 Understanding-series didactic docs have gone stale relative to the project's own later findings

`understanding/05_convergence_proof.md` walks through the paper's original convergence proof and hedges only with the generic "real neural nets aren't convex" caveat every FL paper's proof shares — it never mentions the simplex-feasibility failure that `fedgraa_convergence_defect.md` (which explicitly supersedes this file's theoretical content) found: the weight vector leaves the feasible set at round 1, on 120/120 tested configurations, causing the loss to blow up rather than merely deviate from an idealized bound. `understanding/08_answers_researched.md` Q10 has the same gap — a genuinely well-sourced answer to a *generic* convergence-assumption question, written before or independent of the project's own more specific, more important later finding. Neither file is factually wrong; both are answering a narrower question than the one this project ultimately needed answered, and nothing signals that to a reader treating the series as current.

### 4.6 Where the documentation layer is genuinely well-run

Worth stating plainly: staleness is mostly *well-handled* in this repo. `latest_baseline.md`, `PAKDD_STATUS_REPORT.md`, `STAGE_B_REMAINING_TASKS.md`, `STAGE_DIRECTION_ASSESSMENT.md`, and `RESEARCH_DIRECTION_AUDIT_2026-07-25.md` all carry dated, scoped, accurate "superseded" banners that correctly preserve what's still live. The relationship between `STAGE_C_LITERATURE_MASTER.md` and its reaudit addendum is a model of disclosed, tracked revision — the reaudit explicitly lists every edit it applied to the master doc, rather than silently diverging. The archived `sagitra_report.md` carries an unambiguous banner pointing to its replacement (though see the caveat below). And the Simulated Peer Review (`FedGRAA_Simulated_Peer_Review.md`) is a genuinely adversarial review, not a soft pass — it reaches a Reject verdict, explicitly overrides two of its own softer-leaning reviewer personas, independently rediscovers the core algorithmic defect from the algorithm box alone, and tags exactly which findings are code-corroborated versus visible on the page. Its one disclosed limitation is architectural (five personas, one underlying model) rather than leniency.

### 4.7 One remaining live risk in the archived research corpus

`docs/research_c/sagitra_report.md` is correctly banner-marked as archived and superseded, but its body still confidently presents, with direct quotes, several "papers" (CT-BRFL, FedHDE, FedCCW) that `STAGE_C_LITERATURE_MASTER.md` later determined are hallucinations via a "fluent description, zero matching link" signature. A reader who opens the archived file directly rather than starting from the current master doc could copy a fabricated, quoted citation before ever reaching the correction.

---

## 5. What checks out — stated explicitly, not buried

This section exists because a report that only lists problems would misrepresent what was actually found. On direct, independent re-derivation from raw JSON (not from re-reading the docs' prose):

- **The "120 of 120 unprojected cells leave the simplex in round 1" claim is exactly correct** — all 40 relevant per-seed files × 3 γ-keys = 120 cells, all round 1, zero exceptions; the 120 projected-arm cells all show `first_simplex_violation_round: None` with final weight-sum deviation ≤1.19×10⁻⁷ (float32 epsilon).
- **Every pairwise p-value, Cohen's d, and Holm-corrected p-value quoted in both master tables matches the underlying JSON exactly**, to the precision shown, across all 120 rare-acc + all-acc comparisons in both datasets.
- **The Holm–Bonferroni monotonicity property (`p_holm ≥ p_ttest`) holds with zero violations** across all 120 comparisons — no bug in the correction.
- **All mean/median/min/max fields in both ablation table JSONs reconcile exactly with their own `per_seed` arrays** (80 cells checked, zero discrepancies, tolerance <1e-8).
- **Every root-README headline number** (the "minimal fix" table, blow-up ratios up to 1.86×10⁶×, all six arms × both datasets) **matches the ablation JSON exactly.**
- **Round-by-round composite-loss traces quoted in the master reports match the raw persisted arrays exactly**, spot-checked across four separate traces including the most extreme blow-up.
- The 10-seed × 3-γ × 2-dataset × 4-arm ablation structure (80 diagnostic files) is exactly as claimed; the campaign logs' "71 fresh + 9 reused" run count is exactly consistent with the manifest.
- Client class-imbalance numbers quoted in prose (Heartbleed=11, SQL Injection=21, etc.) match `fedgraa/config.py` verbatim.

**Conclusion of this section:** wherever this audit checked a specific scientific number against its primary source, it was correct. The defects that matter are concentrated in the paper draft's own framing (§1), a handful of real but currently-inert-or-minor code bugs (§2), and a verification/documentation layer that checks structure rather than substance and doesn't audit its own currency (§3–4) — not in the underlying experimental results.

---

## 6. Ranked action list

1. **Do not submit the paper draft with its current abstract.** Either recompute the headline numbers from the algorithm exactly as the paper's own methodology section describes it (which, per this project's own data, yields ~10% not 88%), or explicitly present the corrected/projected variant as a correction with its own hyperparameters disclosed, not as the base method's result.
2. **Fix or remove references [4], [8], [11], [12]** in the paper draft — cross-check the entire bibliography against `STAGE_C_LITERATURE_MASTER.md`'s existing hallucination findings before any resubmission.
3. **Fix the NaN asymmetry in `aggregation.py`** (§2.1) — make `composite_loss()` and `_composite_grad_P()` agree on which NaN case each catches; fix the misleading "both empty" warning text to distinguish emptiness from numerical blow-up.
4. **Wire `audit_stage_a.py`/`audit_stage_b.py` into something that actually runs** — a pre-merge check, at minimum a documented required step before any "complete" claim, since they are the only layer that verifies values rather than shape.
5. **Update `AUDIT_REMEDIATION_PLAN_2026-07-25.md`'s status column** (or mark the whole document superseded/closed) — five-minute fix, highest symbolic value given §4.1.
6. **Reconcile the 142/151, 232/262, and 90/78 numeric-drift items** (§4.3) — mechanical, low-effort, but each currently makes a "current" document self-contradictory.
7. **Un-mock `MLP.predict()` in at least one test**, and add real assertions (not signature checks) to `test_parallel_sweep.py` and `test_optimizer_state_extraction`.
8. **Have the one open human conversation** (`docs/audit/RESEARCH_DIRECTION_AUDIT_2026-07-25.md`'s own top recommendation) about this manuscript's provenance before investing further in Stage C — every audit pass that has looked at this, including this one, converges on the same conclusion that this is the actual bottleneck, not another audit.

---

*This document was produced by five independent parallel review passes plus direct firsthand verification of the highest-stakes claim, per the user's request for a full, unsparing, line-by-line grill of the repository. It is, itself, another AI-generated audit — see §4.4 for why that fact should not be allowed to make this the seventh audit-of-audits instead of the trigger for item 8.*
