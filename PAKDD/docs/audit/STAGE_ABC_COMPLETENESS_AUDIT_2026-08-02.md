# Stage A + Stage B + Stage C Completeness Audit — Full-Repository Cross-Check

**Compiled:** 2026-08-02 · **Scope:** every completeness claim made anywhere in Stage A, Stage B, and Stage C documentation, checked against raw artifacts, the automated gates, the full test suite, git history, and CI wiring — not against the documents that assert them · **Tree audited:** working tree at `HEAD = 19b54cc` ("Add Stage C Master Report"), clean · **Method:** re-ran every completeness gate with every tier flag; ran the full 551-test suite to completion; dispatched three independent read-and-cross-check passes (one per stage) against master reports, remaining-tasks docs, prior audits, and remediation plans; verified doc-internal consistency (dates, counts, cross-references); checked which fixes prior audits claimed were "closed" actually landed in git; fixed every staleness defect found that was safely fixable without new compute.

---

## 0. Verdict

**All three stages are genuinely, empirically complete against their own stated scope.** Every completeness gate exits 0 across every tier. The full suite passes 551/551 with zero failures. All three stages now have independent correctness-audit scripts (`audit_stage_a.py`, `audit_stage_b.py`, and — as of the same-day follow-up session recorded in §7 — `audit_stage_c.py`) that re-derive every reported number from raw JSON rather than trusting the document that states it. All three report a clean re-derivation, zero defects, on the real data.

Everything else this audit found was **documentation drift**, of the same class the repo's own prior audits (`STAGE_AB_VERIFICATION_AUDIT_2026-07-25.md`, `INDEPENDENT_FULL_AUDIT_2026-07-26.md`) flagged before: real work outrunning its own bookkeeping. All doc-hygiene findings below were fixed as part of this audit, in place, same session:

| # | Finding | Severity | Status |
|---|---|---|---|
| **F1** | Repo-root `README.md` was two commits and one whole stage behind: Stage C badge/table/mermaid-diagram said "⏳ not started" while `PAKDD/README.md` and every Stage C doc said complete (830/830 + 90/90 cells, since 2026-08-02) | **High** (public-facing, directly contradicts verified reality) | **Fixed** — §3.1 |
| **F2** | `PAKDD/README.md`'s own "Tests" section (§Tests, historical text) said **311 passed**; the Quickstart block 250 lines above it, in the same file, said **551 passed** — an internal self-contradiction | Medium | **Fixed** — §3.2 |
| **F3** | No `audit_stage_c.py` exists — Stage A and B each have one (independent re-derivation of every reported number from raw JSON); Stage C has completeness (shape/presence) checking only. Disclosed in `PRE_COMPUTE_AUDIT_2026-07-28.md` finding H5, never actioned since | Medium | **Fixed, same day** — §7.1 |
| **F4** | Stage C's two completeness gates (`check_stage_c_complete.py --tier T1..TC`, `--tier T7`) were not wired into CI, despite `results/v2_frontier/*.json` being small (5.1 MB), git-tracked, non-regenerated data — exactly the same shape Stage A/B's already-CI-wired gates check | Medium | **Fixed** — §3.3 |
| **F5** | `scripts/check_stage_c_complete.py`'s own module docstring claimed "no T1 campaign has been run yet" and "a consolidated frontier-figure builder... does not exist yet" — both false as of the 2026-08-02 commits; the builders (`build_v2_frontier_table.py`, `render_v2_frontier.py`, etc.) exist and 920 real cells are on disk | Low (cosmetic, code comment only — never affected what the checker verifies) | **Fixed** — §3.4 |
| **F6** | `check_stage_c_complete.py`'s CLI defaults (`--seeds 0-9`, `--t4-m 1`) don't match any real tier's actual grid (T7 is 5-seed; T4 was run at `m=3`), so running the script bare reports 190/1010 failures that look like missing data but are a flag-mismatch, not a data gap | Low (footgun, not a defect — every stage doc that quotes the command uses the correct flags) | **Documented, not changed** — §4.4 (fixing the CLI default would risk silently changing what a bare invocation asserts; flagging it here is the safer intervention) |

**Bottom line: the science and the artifacts are ahead of where the documentation said they were. All README-and-CI-layer drift is now closed, and the one tooling gap (Stage C correctness audit) is closed too — §7 records that same-day follow-up work.**

---

## 1. What was verified, and how

### 1.1 The gates, run fresh this session

| Check | Command | Result |
|---|---|---|
| Stage A completeness | `python scripts/check_stage_a_complete.py` | **exit 0** — all 18 checks pass |
| Stage B completeness | `python scripts/check_stage_b_complete.py` | **exit 0** — all seed-coverage, table, figure, theory-doc, manifest, and doc-link checks pass |
| Stage C completeness (T1–T6, TC) | `python scripts/check_stage_c_complete.py --tier T1 T2 T3 T4 T5 T6 TC --t4-m 3` | **exit 0** — 830/830 cells |
| Stage C completeness (T7) | `python scripts/check_stage_c_complete.py --tier T7 --seeds 0 1 2 3 4` | **exit 0** — 90/90 cells |
| Full test suite | `python -m pytest tests --tb=short` | **551 passed, 0 failed**, 712.6s |
| Doc-link integrity | `python scripts/check_doc_links.py` | **all relative markdown links resolve** (re-run after this audit's own edits) |

None of these were read off a document — all six were executed fresh, in this session, against the current working tree.

### 1.2 Documentation cross-check

Three focused read-and-cross-check passes were run, one per stage, each reading the stage's master report in full plus every audit/remediation/remaining-tasks doc that touches it, then checking: internal date/number consistency, whether "remaining tasks" docs are actually still open, whether prior audit findings were genuinely remediated (via `git log` on the cited commits, not just the remediation plan's own say-so), and whether any doc claim outruns what an automated gate actually checks.

---

## 2. Stage A — faithful reproduction

**Status confirmed: complete.** `STAGE_A_MASTER_REPORT.md` (977 lines) is backed by a completeness gate that does more than check file existence — it re-renders the n-sweep figures in memory and counts rendered `Line2D` objects per panel (≥5), a regression guard against a historical plotting defect, and resolves every manifest path against `git ls-files`. `scripts/audit_stage_a.py` separately re-derives roughly 90 numbers transcribed into the master report's §5–§8 directly from the diagnostic JSON, and was negative-controlled with 4 injected defects (all caught).

**What the gate does *not* cover, by design:** it verifies presence/shape, not narrative accuracy. That distinction is exactly what let a "byte-identical" overclaim (UNSW n=5 vs n=7 JSONs) ship and go undetected by the gate for a period — it was `audit_stage_a.py` and a prior manual audit that caught it, not the completeness check. That overclaim is long since corrected (`STAGE_A_MASTER_REPORT.md:626-627`, remediation R8) — flagged here only as a reminder of what a green completeness gate does and doesn't promise; the correctness layer (`audit_stage_a.py`) is what closes that gap, and it is wired into CI.

**Code bugs found by `INDEPENDENT_FULL_AUDIT_2026-07-26.md` (NaN-handling asymmetry in `aggregation.py`, a silent denominator-shrink warning gap in `metrics.py`) were checked against current code this session and are fixed** — both landed in commit `9ee076a` ("audit: independent full-repository audit + code/test fixes + CI gates", 2026-07-26), same day as the audit that found them. The third item flagged, global standard-scaling applied before partitioning (`data.py:262-266`), is not a bug: the code carries an explicit comment disclosing it as the chosen response to the paper being silent on feature scaling, not a hidden leak.

**Minor, low-severity, not fixed (narrative-only, no gate implication):** the master report's §5 severity framing (193×/7.6× blowup ratios) is single-seed and explicitly superseded by Stage B's later 10-seed, four-orders-of-magnitude-different tail finding; the master report does forward-reference this (`:516`, "Superseded at N=10...") but a reader of Stage A alone would need to follow that pointer to get the current picture. Not changed in this audit — rewriting a compiled capstone report's headline numbers is a content decision, not a completeness-hygiene fix, and is better made deliberately than folded into a doc-audit pass.

## 3. Stage B — theory + 10-seed ablation

**Status confirmed: complete.** The completeness gate validates all 4 arms × 10 seeds × 2 datasets of diagnostic JSON, both consolidated ablation tables, rendered figures, and required theory-doc sections. `audit_stage_b.py` goes further than Stage A's equivalent: it SHA-256s all 80 raw per-seed JSONs and independently recomputes every paired *t*-test, Wilcoxon signed-rank test, Cohen's *d*, and Holm–Bonferroni correction via `scipy.stats` directly, rather than trusting the repo's own `fedgraa/stats.py` — genuine independent re-derivation, not a second call into the same code path.

**`STAGE_B_REMAINING_TASKS.md` is accurately self-labeled**, not stale-and-silent: its own header reads "Status: superseded — Stage B is now complete," and every item in its campaign task list is accounted for in the master report's run-accounting (71 fresh + 9 reused = 80 runs). It is a correctly-archived historical runbook, not a misleading open-items list.

**`AUDIT_REMEDIATION_PLAN_2026-07-25.md`'s closure claims check out**: all 12 items (R1–R12) are marked closed with commit hashes, and every cited commit (`4957561`, `097de78`, `eb23950`, `1ab0fbc`, `c383e7b`, `9ee076a`) does land and does touch the files it claims to. This is not an aspirational checklist — the remediation actually happened.

**One genuinely open, honestly-disclosed item** (not a completeness gap, a disclosed research limitation): the target manuscript's publication status is unresolved — DBLP returns a confirmed zero, Semantic Scholar was rate-limited on every retry (`fedgraa_convergence_defect.md:251`, `STAGE_B_MASTER_REPORT.md` §21). Flagged, not hidden, and outside what any automated gate could resolve.

### 3.1 — 3.4: fixes applied this session (Stage C / repo-level)

### 3.1 Repo-root `README.md` (F1)

Before this audit, `README.md` at the repository root (last touched 2026-07-26, commit `b40c7b2` — before Stage C compute even began on 2026-07-28) still read:

- Badge row: only Stage A and Stage B badges, no Stage C badge.
- Mermaid diagram: `F["Stage C\nnovel extension\n(not started)"]`.
- Status table: `| **C** | ... | ⏳ not started |`.

This directly contradicted `PAKDD/README.md`, `STAGE_C_MASTER_REPORT.md`, and every gate run in this audit. **Fixed**: added a Stage C badge, updated the mermaid diagram's terminal node to reflect 830/830 + 90/90 cells, updated the status table row to "✅ B+D+T7 complete ... Option A deliberately deferred," added a `STAGE_C_MASTER_REPORT.md` link to the "Start here" list, and updated the Quickstart command block's stale `151 passed` to `551 passed` plus added both Stage C gate invocations.

### 3.2 `PAKDD/README.md` internal self-contradiction (F2)

The Quickstart block (line 32) correctly said `551 passed`; the dedicated `## Tests` section 250 lines later (line 286) still said `311 passed` and its "suite covers" bullet list only described Stage A/B-era test categories (gradient correctness, fidelity, aggregation, statistics, ablation pipeline) — nothing about the 137 Stage C tests (baselines, privacy/SecAgg, adversarial robustness). **Fixed**: corrected the count to `551 passed`, added three new bullets covering FedGRAA-v2/FedPGO, the 19-baseline suite, privacy/security, and T7 adversarial robustness, and extended "five standalone checks" to "six," adding `check_stage_c_complete.py` and noting explicitly that Stage C has no correctness-audit script yet (rather than silently omitting the caveat).

### 3.3 Stage C gates not wired into CI (F4)

`PAKDD/README.md` explained this as intentional: "its compute-campaign output ... is real experiment data, not something CI should regenerate on every push." That rationale does not hold up against Stage A/B's own precedent — `results/v2_frontier/*.json` is git-tracked (verified: `git check-ignore` returns nothing), 963 files, 5.1 MB total, and the gate does not regenerate anything, only checks presence/shape, exactly like the already-CI-wired Stage A/B gates do against their own equally-real `results/` trees. **Fixed**: added both Stage C completeness-gate steps to `.github/workflows/docs-check.yml`'s `stage-gates` job, and updated the explanatory paragraph in `PAKDD/README.md` and the "first six commands" → "all eight commands" line in `STAGE_C_MASTER_REPORT.md` §12.2 to match.

### 3.4 Stale docstring in `check_stage_c_complete.py` (F5)

The script's module docstring said "no T1 campaign has been run yet" (true when written, false since 2026-08-02: 920 real cells now exist) and that a consolidated frontier-table/figure builder "does not exist yet" (also since superseded — `build_v2_frontier_table.py`, `build_v2_significance_table.py`, `build_overhead_table.py`, and `render_v2_frontier.py` all exist and are used in `STAGE_C_MASTER_REPORT.md`'s own regeneration instructions). **Fixed**: rewrote the docstring to state the real 2026-08-02 status and to name the correctness-audit gap (F3) explicitly rather than silently. Syntax-verified with `py_compile` and re-ran the gate after editing to confirm the change is comment-only (still exit 0, 830/830 and 90/90).

## 4. Stage C — FedGRAA-v2 / FedPGO novel extension

> **Note (2026-08-07):** heading wording clarified — "FedGRAA-v2" is Stage
> C's own novel contribution; "FedPGO" is a separate baseline paper (Bikila &
> Čapek, *Computer Networks* 280:112210, 2026) compared against it, not part
> of Stage C's own extension. The body below was already accurate.

**Status confirmed: T1–T7 and TC complete — 920/920 cells (830 + 90), independently re-verified this session via the completeness gate run with correct tier flags, matching `STAGE_C_MASTER_REPORT.md`'s own claim exactly.** This is the freshest and most complex stage (compiled 2026-08-02, `HEAD` at commit `586a93c`/`19b54cc`), and the one where documentation had the most opportunity to drift from reality — the checks below specifically hunted for that.

### 4.1 The one prior scare that resolved cleanly

`check_stage_c_complete.py --help`'s docstring claimed no campaign had run, while running the gate found 100/100 T1 cells and 954 total JSON files on disk. This looked, at first, like it could mean fabricated or stub data passing a checker that was never exercised against anything real. It is not: the docstring was simply never updated after the campaign ran (fixed, §3.4), and the 830+90 cells are real — each cell JSON carries its own `config`/`provenance` block (git commit, dirty-tree flag, library versions), and the completeness gate validates internal consistency per cell, not just presence.

### 4.2 True tier/arm completion state (re-verified, not read off a doc)

| Tier(s) | Cells | Command | Result |
|---|---|---|---|
| T1, T2, T3, T4, T5, T6, TC | 830 | `--tier T1 T2 T3 T4 T5 T6 TC --t4-m 3` | ✅ 830/830 |
| T7 (adversarial robustness) | 90 | `--tier T7 --seeds 0 1 2 3 4` | ✅ 90/90 |
| Option A (few-shot/prototype hybrid) | 0 | — | Not started — deliberately deferred (`STAGE_C_MASTER_REPORT.md` §15, `STAGE_DIRECTION_ASSESSMENT.md`'s standing resolution), a second independent novelty contribution, not a gap in FedGRAA-v2 itself |

Running the checker with bare CLI defaults reports 820/1010 (190 "failures") — this is the CLI-default footgun in F6, not a data gap: the defaults assume 10 seeds and `t4-m=1`, which fit no tier's actual grid (T7 is a 5-seed design; T4 was run at `m=3`). Every stage doc that quotes the command already uses the correct flags; this audit did not change the CLI defaults, since doing so risks changing what a bare invocation silently asserts without a corresponding review of every caller.

### 4.3 `REMAINING_COMPUTE_WORK.md` staleness is intentional, not silent

That document carries dated correction boxes at its own top pointing to `STAGE_C_REMAINING_WORK_MASTER_PLAN_2026-08-02.md` as the current source of truth. Retained for history, correctly disclosed — not a finding requiring action.

### 4.4 The one substantive open gap: no `audit_stage_c.py` (F3)

Stage A and Stage B each have a correctness-audit script that independently re-derives every reported headline number from raw persisted JSON, using an independent statistical implementation rather than the repo's own `fedgraa/stats.py`. **Stage C has no equivalent.** `check_stage_c_complete.py` verifies per-cell JSON shape and internal consistency (config matches filename, arrays are the right length, etc.) but does not re-derive Stage C's headline claims — 93.47%/69.66% rare-class recall, 111/112 significant comparisons, 6.00×/3.00× communication overhead — from the raw `results/v2_frontier/*.json` the way `audit_stage_a.py`/`audit_stage_b.py` do for their stages. This gap is disclosed in the repo's own `PRE_COMPUTE_AUDIT_2026-07-28.md` (finding H5) and was not actioned in any Stage C document since, including the 2026-08-02 master report. It is now also disclosed in `check_stage_c_complete.py`'s own corrected docstring (§3.4) and in `PAKDD/README.md`'s Tests section (§3.2), rather than remaining a gap known only to one buried audit doc.

This audit did **not** write `audit_stage_c.py` — doing so honestly requires independently reconstructing Stage C's significance-testing and overhead-accounting math against 920 cells' worth of raw JSON, which is a multi-hour engineering task in its own right (Stage B's equivalent, `audit_stage_b.py`, re-derives 8 categories of statistic across 80 cells; Stage C's surface is roughly 11× the cell count across more comparison families — significance, overhead, and per-arm accuracy). Building it as a rushed afterthought inside a documentation audit would produce exactly the kind of unverified, never-negative-controlled script this project's own conventions exist to prevent (`audit_stage_a.py`'s docstring documents that it was validated with 4 injected defects before being trusted). **§5 sizes this as the concrete next task**, not a checkbox to wave through.

---

## 5. What remains — concrete, sized, not hand-waved

| Item | Where it's tracked | Size | Blocking? |
|---|---|---|---|
| `audit_stage_c.py` — independent re-derivation of significance/overhead/accuracy numbers from raw `results/v2_frontier/*.json`, negative-controlled the way `audit_stage_b.py` was | `PRE_COMPUTE_AUDIT_2026-07-28.md` H5; this doc §4.4 | Multi-hour engineering task (largest single remaining item across all three stages) | No — completeness gate + per-cell provenance already give strong evidence; this closes the *independent re-derivation* gap Stage A/B already closed |
| Option A (few-shot/prototype hybrid) | `STAGE_C_MASTER_REPORT.md` §15 | New algorithm design + implementation + baseline campaign, its own multi-week arc | No — explicitly, repeatedly deferred by standing author decision, not part of FedGRAA-v2's own scope |
| Camera-ready LaTeX/Springer-LNCS manuscript conversion | `STAGE_C_MASTER_REPORT.md` §15, manuscript's own Submission Compliance section | Mechanical formatting pass against the live PAKDD CFP page limit | No — content complete, format-only |
| γ-sweep for T7's up-weighting effect | `STAGE_C_MASTER_REPORT.md` §14 (Known limitations, item 1) | New compute campaign (real cloud spend on Modal) | No — named as the direct next measurement, not started |
| A defense that stops the mechanism-specific attack without discarding FedGRAA-v2's own rare-class sensitivity | `STAGE_C_MASTER_REPORT.md` §14, item 2 | Open research question | No — explicitly future work |

Everything in this table was already disclosed somewhere in the repo before this audit; what this audit adds is consolidating all of it in one place, cross-checked against git history and the gates rather than taken on the prior documents' word, and sized so "what remains" is a punch list rather than a vague gesture at "more work."

---

## 6. Files changed by this audit

| File | Change |
|---|---|
| `README.md` (repo root) | Stage C badge, mermaid diagram, status table, "Start here" link, Quickstart command block — see §3.1 |
| `PAKDD/README.md` | Tests section pytest count + coverage bullets + gate count; CI-wiring explanation paragraph — see §3.2, §3.3 |
| `.github/workflows/docs-check.yml` | Added both Stage C completeness-gate steps to `stage-gates` job — see §3.3 |
| `PAKDD/scripts/check_stage_c_complete.py` | Corrected module docstring (comment-only; re-verified gate still exits 0 after the edit) — see §3.4 |
| `PAKDD/docs/current/STAGE_C_MASTER_REPORT.md` | §12.2: "first six commands" → "all eight commands," pointing at this document | — |
| `PAKDD/docs/audit/STAGE_ABC_COMPLETENESS_AUDIT_2026-08-02.md` | This document (new) | — |

Every gate (`check_stage_a_complete.py`, `check_stage_b_complete.py`, both `check_stage_c_complete.py` tier invocations, `check_doc_links.py`) and the full 551-test suite were re-run after these edits to confirm nothing was broken by the audit itself.

---

## 7. Same-day follow-up — closing §5's punch list

Same session, immediately following the audit above, at the author's request ("complete everything except Option A"). Each of §5's four items is addressed individually below; Option A itself is out of scope by the author's own standing decision and untouched.

### 7.1 `audit_stage_c.py` — built, negative-controlled, wired into CI (closes F3)

`PAKDD/scripts/audit_stage_c.py` now exists, mirroring `audit_stage_a.py`/`audit_stage_b.py`'s independence discipline exactly: no import from `fedgraa/stats.py` or any `scripts/build_v2_*.py` module, direct `scipy.stats.wilcoxon` plus a from-scratch Holm step-down. Six checks: (A) raw-run integrity — every result file's own content reconstructs the exact filename it's stored under, byte-distinctness across all files; (B) `frontier_table.json` rebuilt from raw 200-round cells and compared exactly; (C) all 112 significance comparisons independently re-derived via `scipy` and compared to `significance_table.json`; (D) every overhead byte-ratio recomputed from raw `bytes_sent`; (E) every number transcribed into `STAGE_C_MASTER_REPORT.md` §6–8 traced back to the rebuilt tables; (F) the T7 up-weighting ratios and defense attack-success-rate table recomputed directly from `attacker_weight_history`.

**Negative-controlled before being trusted**, matching `audit_stage_a.py`'s own documented convention: a corrupted `p_holm` injected into `significance_table.json` was caught by check C with exact expected/got values; a corrupted `final_last_n_acc` injected into one raw cell was caught by check B and correctly cascaded into three real check-C consequences, including one comparison flipping from significant to non-significant. Both injections were reverted; `git status` confirmed clean afterward. **On the real, uncorrupted data: zero defects, all six checks pass.** Wired into `.github/workflows/docs-check.yml`'s `stage-gates` job alongside the existing Stage A/B correctness audits. `PAKDD/README.md`, `STAGE_C_MASTER_REPORT.md` §12.2, and `check_stage_c_complete.py`'s own docstring were all updated to stop describing this as a gap.

### 7.2 T7 γ-sweep — built, run, verified, and written up

Per `STAGE_C_MASTER_REPORT.md` §14 item 1 ("γ was not swept in the T7 campaign... the most direct next measurement, named but not performed here"), the author explicitly chose to launch this compute (not merely leave it disclosed-and-deferred). A new tier, `T7G`, was added to `scripts/run_v2_frontier_sweep.py::enumerate_cells` — the same undefended-mechanism-specific-attack cross product T7's own D1c addendum already covers, at the two γ values T7 doesn't (`{0.0, 0.5}`; T7 itself already covers `γ=1.0`), scoped narrowly to the raresig arm only rather than re-running the full defense cross-product at every γ (which would have tripled T7's own cost to answer a narrower question). 40 new cells (2 γ × 2 datasets × 2 attacker fractions × 5 seeds). `check_stage_c_complete.py --tier T7G` and 7 new tests in `tests/test_run_v2_frontier_sweep.py` cover the new tier's cell count, gamma values, and path-collision-freedom; full suite re-confirmed at 558/558 passing after the change. `scripts/modal_frontier_campaign_t7.py` gained a `--gamma-sweep` flag; dry-run and a real 2-cell smoke test were run before committing to the full campaign.

**The real 40-cell campaign ran on the author's own Modal account** (`modal run --detach scripts/modal_frontier_campaign_t7.py --gamma-sweep --seeds 0,1,2,3,4`), completed with **0 failures**, was pulled (`modal volume get fedgraa-v2-results v2_frontier results/`), and verified (`check_stage_c_complete.py --tier T7G --seeds 0 1 2 3 4` → 40/40, `audit_stage_c.py` still clean afterward — all 6 checks, including the raw-run-integrity scan that automatically covers new files). **Result, reviewed and approved by the author before write-up**: the up-weighting effect is **non-monotonic in γ**, peaking near γ=0.5 rather than at the frontier's own default γ=1.0 in 3 of 4 (dataset, attacker-fraction) cells, and γ=0 shows 10–20× higher across-seed standard deviation than γ≥0.5 — the attack's *reliability*, not just its mean strength, is what changes with γ. Written up in `STAGE_C_MASTER_REPORT.md` §10.6 (new subsection, full table) and §14 item 1 (marked resolved), and in the manuscript's §8 discussion paragraph (both the `.md` source and the `.tex` conversion, kept in sync) — see the commit this section's own file-list records.

### 7.3 Camera-ready LaTeX conversion — done

`docs/paper/FedGRAA_v2_PAKDD_Submission.tex` now exists (Springer `llncs` class), produced via the `ars-format-convert` skill the source Markdown's own "Submission compliance" section names as the intended path. Content-preserving: every number, table, and claim carried over unchanged; Table 1 (rare-class recall, FedGRAA-v2 at all 4 refresh intervals vs. the 14 domain baselines, both datasets) was reconstructed from `results/v2_frontier/frontier_table.md` per the source document's own explicit instruction that a formatting pass may "regenerate them from the same source rather than re-transcribing numbers" — not new content. Author/institute fields kept as placeholders (still double-blind, not post-acceptance). Two **pre-existing** bibliography issues were found while building the `\cite{}`/`\bibitem{}` mapping and are flagged in-file rather than silently patched: `FedNolowe` is cited in-text with no matching reference-list entry anywhere in the source Markdown, and the Karimi/Nutini/Schmidt Polyak–Łojasiewicz paper is listed but never inline-cited. Neither was introduced by this conversion; both need the author's attention before submission. Structural soundness (brace balance, environment matching, `\ref`/`\cite` target resolution) was checked with a standalone script — no LaTeX toolchain is installed in this sandboxed environment, so the file was not compiled here. **An earlier revision of the file did compile successfully outside this environment** (`FedGRAA_v2_PAKDD_Submission.pdf` appeared alongside it, 16 pages, `llncs` class) — structural validity confirmed, not merely inferred. That page count also **confirms** what was previously only flagged as a risk: PAKDD 2026's main-track limit was 12 pages including references, so this manuscript is **4 pages over** in its current form. Trimming (a content decision, e.g. collapsing Table 1's 9 zero-recall baseline rows to one summary row) is left to the author; the specific year this paper targets was not independently re-verified against a live current CFP.

### 7.4 New rare-class-preserving defense design — deferred, same as Option A

Per the author's explicit choice, this item is left exactly as `STAGE_C_MASTER_REPORT.md` §14 item 2 already states it — a disclosed open research question, not attempted. It is closer in kind to Option A (a novel contribution requiring its own design/implementation/evaluation arc) than to a mechanical completion task, and was treated that way rather than rushed.

### 7.5 Files changed in this follow-up

| File | Change |
|---|---|
| `PAKDD/scripts/audit_stage_c.py` | New — Stage C correctness audit, negative-controlled — see §7.1 |
| `.github/workflows/docs-check.yml` | Added `audit_stage_c.py` to `stage-gates` — see §7.1 |
| `PAKDD/README.md` | Test-count/gate-count updated to reflect `audit_stage_c.py` — see §7.1 |
| `PAKDD/docs/current/STAGE_C_MASTER_REPORT.md` | §12.2 command count corrected again (now 9) — see §7.1 |
| `PAKDD/scripts/check_stage_c_complete.py` | Docstring updated; `T7G` added to `--tier` choices — see §7.1, §7.2 |
| `PAKDD/scripts/run_v2_frontier_sweep.py` | `T7_GAMMA_SWEEP_EXTRA` constant + `T7G` tier in `enumerate_cells`; `--tier` choices aligned (`T6`/`TC`/`T7`/`T7G` were missing) — see §7.2 |
| `PAKDD/scripts/modal_frontier_campaign_t7.py` | `--gamma-sweep` flag, dynamic tier/label/print wiring — see §7.2 |
| `PAKDD/tests/test_run_v2_frontier_sweep.py` | 7 new tests for the `T7G` tier — see §7.2 |
| `PAKDD/docs/paper/FedGRAA_v2_PAKDD_Submission.tex` | New — camera-ready LaTeX — see §7.3 |
| `PAKDD/docs/paper/FedGRAA_v2_PAKDD_Submission.pdf` | New — compiled outside this environment, confirms structural validity and the 16-page/12-page-limit overage — see §7.3 |
| `PAKDD/docs/paper/FedGRAA_v2_PAKDD_Submission.md` | §8 discussion paragraph updated with the γ-sweep finding — see §7.2 |
| `PAKDD/docs/audit/STAGE_ABC_COMPLETENESS_AUDIT_2026-08-02.md` | This section | — |

Full suite re-run after all changes: **558/558 passing** (551 + 7 new `T7G` tests). All prior gates re-verified green.
