# Independent Repository Audit — Complete Pass (2026-08-02)

**Requested by:** repo owner ("audit and verify the whole repo and docs...
for correctness and completeness... go through each and every line of code
and doc files"). Performed by Claude (Sonnet 5), continuing directly from
`INDEPENDENT_REPO_AUDIT_2026-08-02_EXTERNAL.md` (an earlier pass in this same
session lineage that was interrupted mid-task by a platform usage limit and
explicitly disclosed exactly what it hadn't covered). This pass's entire
job was to close every gap that document listed in its own §1.3, re-verify
its findings still hold, and add anything new. It does.

**Tree audited:** working tree at `HEAD = d7d72c6`, clean except for the two
audit documents themselves (both untracked), `main` branch — unchanged
since the prior pass.

**Status: complete.** Every file the prior pass flagged as unread has now
been read in full, line by line, by ten parallel deep-read passes, each
independently instructed to be skeptical and to actively hunt for the same
bug classes the prior pass had already found (stale numbers, broken links,
circular verification, citation fabrication, silent data loss). In addition,
every one of the 9 automated gates was re-run fresh in this session
(independent of the prior pass's run), and one specific numeric discrepancy
a sub-agent couldn't resolve without a live Python environment was closed
out directly against `pytest --collect-only`.

**Coverage now:** 24/24 `fedgraa/` files, 28/28 `scripts/` files, 42/42 test
files (inherited from the prior pass, itself independently corroborated
below), and effectively 100% of the ~31,700 lines of tracked Markdown/TeX
(the only material not read line-by-line is the two compiled PDF exports in
`docs/paper/`, which are build artifacts of `.md`/`.tex` sources that *were*
read, and the ~14 GB of raw CSV data under `data/`, which is validated
instead by the audit scripts' independent statistical re-derivation).

---

## 0. Bottom line

**The prior pass's headline verdict holds and is now on firmer ground:**
the code is exceptionally well-engineered, and the automated correctness
machinery genuinely works. Closing the coverage gap added exactly **one**
new code-level bug (a dormant, undisclosed parameter override in an
adversarial-defense baseline) out of 24,000+ additional lines read, plus a
handful of MEDIUM-severity "disclosure fragility" patterns in the
table/figure-rendering scripts. Nothing resembling a correctness bug was
found in the aggregation math, privacy accounting, secure aggregation, or
any of the 16 external baseline reproductions.

**The documentation layer is, again, where the real defects are — and this
pass found substantially more of them, at every layer of the doc tree, not
just the top-level README the prior pass sampled.** The most important new
findings are:

1. **A real, currently-live functional bug in `run_experiments.py`**, not
   just a doc typo: the `--round-zero-zero` CLI flag's actual default
   (`False`, via bare `store_true`) silently overrides `FedConfig`'s
   intended default (`True`) whenever the flag isn't passed — meaning
   `python run_experiments.py` with no flags today runs with different
   convergence behavior than both the README and the library itself claim
   is the default. This was invisible to the test suite because the CLI
   test file re-implements a shadow parser with the identical bug instead
   of importing the real one (§2.2).
2. **A specific, previously-"fixed" doc-drift bug has recurred a second
   time on the same document**: `docs/current/publication_roadmap.md`
   still says "Last refreshed 2026-07-25" and "142/142 tests passing" after
   two more stages and 400+ more tests landed — the exact defect class
   `AUDIT_REMEDIATION_PLAN_2026-07-25.md`'s R4 already closed once on this
   same file (§3.4).
3. **`STAGE_C_MASTER_REPORT.md`'s own "new tests this stage" arithmetic
   doesn't reconcile with itself** — a table summing to 105 (112 with the
   T7G addendum) sits next to prose claiming 137 (144 with T7G), a
   consistent, unexplained gap of 32 in both cases. The document's
   *headline* total (558) is independently correct — verified fresh via
   `pytest --collect-only` in this pass — so this is a breakdown-level
   inconsistency, not a wrong final number, but it is the same "three
   different counts for the same thing in one document" bug class the
   prior pass found in `README.md`, now confirmed to also live in the
   project's own newest capstone document (§3.2).
4. **A planning document (`REMAINING_COMPUTE_WORK.md`) still describes T7
   as "not attempted" and a G5 gate as "not run"**, when the current master
   report shows T7 fully executed (90/90 cells) — a stale-status bug of
   exactly the kind the audit brief asked us to hunt for (§3.6).
5. **A genuinely funny, genuinely real, self-inflicted finding**: the prior
   pass's own report — `INDEPENDENT_REPO_AUDIT_2026-08-02_EXTERNAL.md` —
   broke 5 of the repo's 9 CI gates the moment it sat in the tree, because a
   sentence in its own §3.3 used literal markdown-link syntax (square
   brackets immediately followed by parentheses) as a prose example to
   illustrate a blind spot in the link checker, and the checker's regex
   parsed that illustrative example as a real, broken link (§1). Fixed as
   part of this pass, in both documents.

None of these are hidden or subtle — they are exactly the class of "doc
bookkeeping didn't keep up with a fast-moving repo" issue the prior pass's
meta-finding (§3.5 there) predicted would keep recurring. This pass confirms
that prediction was correct, at a larger scale than either pass alone would
have shown.

---

## 1. Fresh ground-truth verification (re-run independently, this session)

| Check | Command | Result |
|---|---|---|
| Full test suite | `pytest tests --tb=short -q` | **558 passed, 0 failed**, 404.2s |
| Test count cross-check | `pytest tests --collect-only -q` | **558 tests collected** — independently confirms the 558 figure by a second method |
| Stage A completeness | `check_stage_a_complete.py` | **9/10 sections pass**; fails only on the doc-link check below |
| Stage B completeness | `check_stage_b_complete.py` | **9/10 sections pass**; fails only on the doc-link check below |
| Stage C completeness (T1–T6, TC) | `check_stage_c_complete.py --tier T1 T2 T3 T4 T5 T6 TC --t4-m 3` | **830/830 cells pass**; fails only on the doc-link check below |
| Stage C completeness (T7) | `check_stage_c_complete.py --tier T7 --seeds 0 1 2 3 4` | **90/90 cells pass**; fails only on the doc-link check below |
| Stage A correctness audit | `audit_stage_a.py` | **exit 0**, all 4 sub-checks pass |
| Stage B correctness audit | `audit_stage_b.py` | **exit 0**, all 5 sub-checks pass |
| Stage C correctness audit | `audit_stage_c.py` | **exit 0**, all 6 sub-checks (A–F) pass, including 972-file byte-integrity, 112-comparison independent scipy re-derivation, and 94 master-report numbers traced to raw data |
| Doc-link integrity | `check_doc_links.py` | **found and fixed 6 broken links mid-pass, 0 remaining** — see below |

### 1.1 The one failure encountered mid-pass, and how it was fixed

Partway through this pass, `check_doc_links.py` (and, because they call the
same library function, the four completeness gates too) started reporting a
broken link inside `INDEPENDENT_REPO_AUDIT_2026-08-02_EXTERNAL.md`:

```
[FAIL] docs/audit/INDEPENDENT_REPO_AUDIT_2026-08-02_EXTERNAL.md
       -> path
```

This was **not a defect in the repository's committed content.** It was
caused entirely by the untracked prior-pass audit document sitting in the
working tree. That document's own §3.3 (correctly) explains that
`check_doc_links.py`'s link-detection regex only matches literal
markdown-link syntax — square brackets immediately followed by
parentheses — and, to illustrate the point, used exactly that syntax as a
prose example inside a sentence about a different file's citation. That
illustrative example was itself syntactically a markdown link whose target
resolved to nothing. The checker has no way to distinguish "a real link"
from "a link used as a typographic example inside a sentence about links,"
so it correctly (by its own stated, honestly-disclosed logic) flagged it.

While compiling *this* report, the same mistake was independently repeated
here — this section originally quoted that exact syntax verbatim to explain
it, which the checker flagged too (5 more broken-link hits, all inside this
document). **Both instances are now fixed**: every place either document
needed to refer to markdown-link syntax now describes it in prose (as
"square brackets immediately followed by parentheses") instead of
reproducing the literal, checker-matchable characters. Re-running
`check_doc_links.py` after the fix (§1, table above) confirms zero broken
links with both audit documents present in the tree — the previously
"self-inflicted" CI break no longer exists as of this commit.

---

## 2. Code findings

### 2.1 `fedgraa/` core library — now 24/24 files read in full

The prior pass read 18/24 files and found zero correctness bugs. This pass
closed the remaining 6 — the Stage C external-baseline reproductions
(`baselines_t3_contrib.py`, `baselines_t3_trust.py`,
`baselines_t5_adaptive.py`, `baselines_t5_fedpgo.py`,
`baselines_t5_nids.py`, `baselines_t7_adversarial.py`; CELM, SpectralFed,
CoAst, FLTrust, RAGA, FedTruth, FedMADE, SBFedNIDS, FedAPA, FedAWA, DA-PFL,
FedAWARE, AdaAggRL, FedPGO, SecFedNIDS, WeiDetect — 4,953 lines).

**One real finding:**

- **[MEDIUM] `fedgraa/baselines_t7_adversarial.py:186–188`** —
  `secfednids_consensus_filter`'s per-array top-K selection silently
  ignores the caller-supplied `top_k_frac` for any parameter array with
  ≤1000 elements (most bias vectors and several smaller weight matrices),
  hardcoding `0.1` instead:
  ```python
  k = max(1, int(round(top_k_frac * flat.size))) if flat.size > 1000 else \
      max(1, int(round(0.1 * flat.size)))
  ```
  `max(1, ...)` already prevents `k=0`, so nothing forces this branch to
  exist — it reads as a copy/paste artifact. It is currently **dormant**
  only because `top_k_frac`'s default (0.1) happens to equal the hardcoded
  fallback; confirmed via grep that no current caller ever overrides
  `top_k_frac`. The moment someone does (a realistic scenario — this file
  explicitly rescales other SecFedNIDS hyperparameters, JC-T7-9, for this
  project's smaller client counts), the override silently vanishes for
  roughly half the model's parameter arrays, with no warning, no docstring
  mention, and no `JC-T7-*` disclosure ID — breaking the file's own
  numbered-disclosure convention used everywhere else. Not exercised by
  `tests/test_baselines_t7_adversarial.py` (no test passes a non-default
  value), so it would not be caught by the suite as it stands.

**Everything else checked out clean**, at the same standard as the 18
previously-read files: every RNG path is seeded (`np.random.default_rng`
or a disjoint offset), every denominator traced is epsilon-guarded with a
documented uniform-fallback convention, hand-derived gradients (FedAWA's
`_fedawa_grad_lambda`, the Frank-Wolfe closed form) check out by hand, and
every deviation from a cited paper is numbered and disclosed — including
FedPGO's JC1–JC8 table (resolving a real internal inconsistency in
Algorithm 3 of the FedPGO paper itself, where it trains before computing
"the gradient" with an optimizer that makes a literal gradient
unrecoverable) and SBFedNIDS's explicit disclosure that it's a
paywalled-abstract *reconstruction*, not a verified reproduction. All 6
files have dedicated, substantial test files (2,431 lines total) that
exercise real trainers end-to-end — zero use of `mock`/`monkeypatch`/`patch`
found, the opposite of the "test the copy, not the code" anti-pattern
already found once elsewhere in this repo's test suite.

### 2.2 `run_experiments.py` — real code, real bug

Read in full as part of the top-level-misc pass. This is the project's CLI
entry point and was **not** covered by the prior pass at all.

- **[HIGH] `run_experiments.py:81-84,118` vs. `fedgraa/config.py:67` —
  silent default mismatch.** `--round-zero-zero` is declared
  `action="store_true"` with no explicit default, so argparse's default is
  `False`. Line 118 unconditionally does
  `round_zero_zero=args.round_zero_zero`, which **always** overrides
  `FedConfig`'s own dataclass default of `True` — even when the user passes
  no flag at all. So `python run_experiments.py` today silently runs with
  `round_zero_zero=False`, contradicting both `PAKDD/README.md:180`
  ("default True") and the library's own intended default. This is not a
  documentation typo — it changes actual experiment output. It has gone
  undetected because `tests/test_run_experiments_cli.py`'s CLI tests use a
  hand-duplicated shadow parser that reproduces the identical bug rather
  than importing the real `argparse` setup (a gap the prior pass's
  independent `tests/` sub-pass had already flagged in general terms;
  this pass shows it has a concrete, live consequence).

### 2.3 `scripts/` — now 28/28 files read in full

The prior pass read 4/28 (`audit_stage_a.py`, `audit_stage_b.py`,
`check_stage_a_complete.py`, `check_doc_links.py`) and rated them
"genuinely rigorous, no holes found." This pass closed the remaining 24
across two independent passes.

**Verification/audit tooling (13 files: `check_stage_b_complete.py`,
`check_stage_c_complete.py`, `audit_stage_c.py`, `verify_v2_plan_claims.py`,
`verify_v2_exactness.py`, `verify_nf_uq_nids_v2_memory.py`,
`diagnose_fedgraa.py`, `real_reproduction.py`, `render_diagnostics.py`,
`run_seed_sweep.py`, `sensitivity_sweep.py`, `seed_sweep_combined_fix.py`,
`run_ablation_seed_sweep.py`) — same high bar as the previously-read four.
`audit_stage_c.py` in particular independently recomputes Wilcoxon and
Holm statistics via raw `scipy`, byte-hashes every raw result file, and
reconstructs expected filenames from each cell's own content — genuinely
non-circular. Two minor findings, neither exploitable today:

- **[MEDIUM] `audit_stage_c.py:533`** — the Sec-6 doc-transcription block
  boundary is found via `doc.index("###", start+10)`, which matches *any*
  three consecutive `#` characters — a `####` sub-heading inside that
  section would truncate the checked range early. No such heading
  currently exists there, so this is latent, not live.
- **[LOW] `check_stage_c_complete.py`'s `check_cell_json`** only asserts
  `bytes_sent`/`comm_rounds` are non-`None`, not numeric-typed (unlike the
  adjacent accuracy fields, which do get an `isinstance` check).

**Build/render/campaign tooling (11 files: `build_overhead_table.py`,
`build_v2_frontier_table.py`, `build_v2_significance_table.py`,
`render_v2_frontier.py`, `render_stage_c_figures.py`, `render_ablation.py`,
`build_ablation_table.py`, `modal_frontier_campaign.py`/`_t6.py`/`_t7.py`,
`run_v2_frontier_sweep.py`) — no data-fabrication or transcription bug
found, spot-checked against the live `frontier_table.json` and
`significance_table.json`. Two real disclosure-fragility findings:

- **[MEDIUM] `build_v2_significance_table.py:138-139`** — the rendered
  markdown header hardcodes the literal string `"10 matched seeds per
  comparison"` instead of deriving it from the correctly-computed
  per-comparison `n_paired_seeds`. Currently accurate (verified: all 112
  live comparisons do have exactly 10 paired seeds), but nothing ties the
  header text to the real value — a future partial Modal failure that
  dropped one seed for one comparison would still compute the Wilcoxon
  test correctly over the seeds that remain, while the header kept
  claiming "10" for every row, silently overclaiming statistical power.
- **[MEDIUM] `render_v2_frontier.py:169-170`** — same pattern: the figure
  suptitle hardcodes `"mean over >=10 seeds"` without checking per-cell
  `n_seeds`. The companion markdown table is honest (prints the real
  `n_seeds` per row), so only the PNG caption carries this risk.
- **[MEDIUM, cross-file] Modal campaign scripts have no persisted failure
  manifest.** All three campaign drivers handle partial failure well —
  per-cell exceptions are caught individually, logged with full traceback,
  and loudly surfaced in a final "N/M cells FAILED" summary, never silently
  dropped or padded — but that signal only lives in console output, not on
  the results Volume. A later table-build pass sees "fewer files than
  expected" with no on-disk record of *why*.
- Two LOW findings: `build_v2_frontier_table.py`'s `_is_t1_cell` doesn't
  gate on `attacker_fraction==0` (currently unreachable given today's arm
  naming); `render_stage_c_figures.py` deliberately (and disclosedly)
  excludes `adaaggrl_labelflip` from one T7 defense-ASR figure.

**Overall §2 verdict:** 24/24 `fedgraa/` files and 28/28 `scripts/` files
now read in full. One real MEDIUM code bug (§2.1) and one real HIGH
functional bug (§2.2) found across ~13,600 newly-read lines; everything
else is either clean or a disclosure-fragility pattern (hardcoded claims
about data shape that are true today but not structurally guaranteed to
stay true). This is consistent with, not a reversal of, the prior pass's
verdict that the code is unusually well-engineered — the defect *rate*
stayed low even once every remaining file was actually read.

---

## 3. Documentation findings

The prior pass read 4 top-level docs in full and found 4 findings, all in
`PAKDD/README.md` and `docs/archive/README.md`. This pass read the
remaining ~24,000 lines of Markdown — `docs/current/`, `docs/paper/`,
`understanding/`, `docs/audit/` (12 of 13 remaining files), `docs/archive/`
(12 of 12 remaining files), `docs/research_c/` (18 files), and top-level
misc (`MENTOR_WALKTHROUGH.md`, `PAKDD_STATUS_REPORT.md`, `latest_baseline.md`,
root `README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `LICENSE`,
`requirements.txt`) — and found substantially more. Organized by area
below; severity in brackets.

### 3.1 Top-level docs (`PAKDD/README.md`, root `README.md`, `CONTRIBUTING.md`, `MENTOR_WALKTHROUGH.md`, etc.)

**Prior pass's two `PAKDD/README.md` findings, re-confirmed with fresh line
numbers:**
- **[HIGH]** Three stale test counts: lines 32, 296 ("551 passed"; should be
  558), and line 438 ("311 unit + ... tests" in the project-layout tree —
  even further out of date than the other two).
- **[MEDIUM]** "All nine commands above" (line 46) refers to a code block
  (lines 31–37) that shows only six commands; the other three
  (`check_stage_c_complete.py` ×2, `audit_stage_c.py`) are never shown as
  runnable lines in that block.

**New findings in the same file:**
- **[HIGH]** `--round-zero-zero` is documented as "(default True)"
  (line 180) — this is what the library *intends*, but per §2.2 above the
  actual CLI default is `False`, so the README and the shipped behavior
  disagree.
- **[MEDIUM]** The n-sweep run-count table (lines 175–176) claims
  `--sweep n` = 9 runs and `--sweep both` = 14 runs. The current code
  (`fedgraa/experiment.py:352-392`, with default `--ns 3 5 7`) actually
  produces 15 and 20 respectively — the code's own comment confirms "9"
  was the *pre-fix* behavior from a since-resolved bug (Gap C-3), and the
  README table was never updated when that fix landed.
- **[MEDIUM]** The CLI flag-reference table (lines 171–186) omits 5 real
  flags that exist in `run_experiments.py`: `--workers`, `--ns`, `--seed`,
  `--data-dir`, `--verbose`.
- **[LOW]** The project-layout tree never mentions the tracked, real
  `archive/` top-level folder (`archive/FedGRAA_reproduction.ipynb`).
- Root-level `./README.md:101` carries the same stale "551 passed" figure
  as its own instance of the same root-cause bug.

**`CONTRIBUTING.md`:**
- **[MEDIUM]** The PR-gate checklist (lines 12–19) lists only 6 of the 9
  commands `docs-check.yml` actually runs — it omits both
  `check_stage_c_complete.py` invocations and `audit_stage_c.py`. A
  contributor following this file verbatim would miss a third of the real
  CI surface. (Root `README.md` does list all of Stage C's commands, so
  this is specific to `CONTRIBUTING.md`.)

**`MENTOR_WALKTHROUGH.md`:**
- **[MEDIUM]** Three different, all-stale test counts coexist inside one
  document that itself carries a 2026-08-02 "Stage C complete" banner:
  line 50 says "311/311 passing," §7 (line 410) says "155/155 passing,"
  and Appendix C (line 534) says "# 155 passed." None were touched when the
  document was updated for Stage C. File/command references spot-checked
  elsewhere in the document are accurate.

**Files confirmed working as intended:** `PAKDD_STATUS_REPORT.md` and
`latest_baseline.md` are both explicitly self-labeled "SUPERSEDED —
historical snapshot" with correct forward-pointers and honestly-framed
historical numbers — no new issues. `requirements.txt` cross-checked
against every third-party import in `fedgraa/`, `scripts/`, `tests/`: no
missing or unused entries. Root `CODE_OF_CONDUCT.md` and `LICENSE` are
coherent, standard, non-placeholder text.

### 3.2 `docs/current/` — master reports and plans (11 files, 4,944 lines)

This is the repo's "living, currently-accurate" tier — and the one place
this pass most expected to find more instances of the README's own bug
class. It did.

- **[HIGH] `STAGE_C_MASTER_REPORT.md` — internal test-count arithmetic
  doesn't reconcile.** The document states "558 tests passing" three times
  (correct — independently re-verified this session via
  `pytest --collect-only`). But its own §12.1 table of "new tests this
  stage" sums to **105** (9+7+24+19+19+6+8+6+7 across 9 rows), while the
  same section's prose (line 369) says **"137 new tests this stage (144
  including the same-day T7G follow-up)."** 105 vs. 137, and 112 (105+7)
  vs. 144 — a consistent, unexplained gap of exactly 32 in both framings.
  This is the same "N different counts for the same fact in one document"
  bug class already found in `README.md`, now confirmed in the project's
  newest capstone report too. The grand total (558) is right; the
  breakdown that's supposed to justify it is not self-consistent.
- **[MEDIUM]** `STAGE_C_MASTER_REPORT.md`'s header ("Compiled: 2026-08-02,
  against HEAD after commit `586a93c`") is stale relative to its own
  content: §10.6 (T7G addendum) and §15 (camera-ready LaTeX) describe work
  that only landed in the *next* commit, `d7d72c6` — the actual current
  HEAD — whose message matches those sections verbatim. The header wasn't
  bumped when the document was extended.
- **[MEDIUM]** `fedgraa_convergence_defect.md` §7 discloses a **second**
  algorithm deviation (client-side Adam momentum persisting across rounds
  while weights reset every round) and explicitly states it was previously
  unmentioned anywhere. Confirmed: neither `STAGE_A_MASTER_REPORT.md` nor
  `STAGE_B_MASTER_REPORT.md` — both of which present themselves as
  sufficient standalone capstones — mentions it.
- **[MEDIUM]** `STAGE_A_MASTER_REPORT.md`'s Appendix B source-document
  index cites `docs/current/literature_review_and_publication_strategy.md`
  and `docs/current/privacy_compliant_fedgraa_redesign.md` — both files
  actually live under `docs/research_c/`. (`STAGE_B_MASTER_REPORT.md` and
  `publication_roadmap.md` cite the correct directory for the same two
  files — this is specific to Stage A's appendix.)
- **[LOW]** `fedgraa_divergence_diagnosis.md`'s sections skip from §6.2
  straight to "§8" — there is no §7 anywhere in the file.
- **[LOW]** `publication_roadmap.md` cites the Stage A capstone as "976
  lines"; it is actually 977.

**What held up well:** `stage_a_plan.md` and `stage_b_plan.md` both carry
explicit "✅ complete, see master report" banners — no stale-plan content
found. `STAGE_B_REMAINING_TASKS.md` opens with an explicit
"superseded — Stage B is now complete" banner, avoiding the classic
stale-task-list bug entirely. The oft-repeated CIC γ=1 loss trace central
to the Stage A/B narrative was spot-verified byte-for-byte against raw
`results/diagnostics/CIC_IDS_2017_literal_r60.json` — exact match. Every
"fixed" claim checked was corroborated, not contradicted, by the master
reports' own tables. Zero broken links across all 11 files (~60 targets
checked).

### 3.3 `docs/paper/` and `understanding/` (16 files, ~5,963 lines)

This pass's most consequential finding lives here: **the plain-language
`understanding/` explainer docs are frozen at the paper's original,
since-retracted state**, with no connection to the project's own later
corrections.

- **[HIGH]** `draft1.md`'s own banner claims it was "kept unedited for the
  historical record," but the body contains dated post-review edits
  (an appendix section explicitly "added 2026-07-26," and multiple
  `<!-- corrected 2026-07-26 -->` reference annotations) — it is not
  unedited, and the claimed chronology ("earlier iteration... kept
  unedited") is internally contradictory.
- **[HIGH]** `draft1.md`'s rewritten abstract disclaims the original
  88%/55% headline numbers as untrustworthy and reports corrected figures
  of "70-89%/37-69%" — but §5's result tables in that same file are
  byte-identical to the old numbers (88.88%, 66.66%, etc.). The corrected
  figures the abstract promises appear nowhere in the file's own body.
- **[MEDIUM]** A citation the project's own audit trail confirmed
  fabricated ("Zhang et al. [12]... FedSND," confirmed by three
  bibliographic tools to have "zero trace") was correctly removed from the
  paper's bibliography, but `understanding/10_SOTA_adaptive_aggregation.md:343`
  still cites it as legitimate ("FedSND... already covered"), with no
  cross-reference to the correction.
- **[MEDIUM]** `v3.md` reference [12] cites an arXiv ID from June 2023
  attributed to a paper described as ICML 2020 — a second citation-
  integrity defect in the same slot as the confirmed-fabricated FedSND
  entry, suggesting the whole reference was never real (superseded, not
  separately diagnosed, in `draft1.md`).
- **[MEDIUM]** None of `understanding/00`–`10` mention the peer review, the
  simplex-projection defect, FedGRAA-v2, or the withdrawn 55% claim.
  `05_convergence_proof.md` presents Theorem 1 with only a soft, generic
  caveat about non-convex losses — materially softer than the project's
  own finding that this specific gap is CRITICAL and invalidates the
  theorem's guarantee for the model actually used.
  `06_glossary_and_cheatsheet.md:56` still states the ~88% headline number
  as plain fact.
- **[LOW-MEDIUM]** Confirmed directly against code: `fedgraa/aggregation.py`
  documents that the base `FedGRAA` class has no simplex projection "on
  purpose," with `ProjectedFedGRAA`/`FedGRAAv2` as the fixed variants.
  `understanding/03_methodology_fedgraa.md` describes only the unprojected,
  literally-broken original algorithm and never mentions the corrected
  variants the codebase actually implements and tests.

**What's genuinely solid:** the paper-version lineage itself is unusually
well self-audited — the project found its own algorithmic defect and its
own fabricated citation, and produced a corrected v2 with real compute.
Headline numbers in the *current* paper (`FedGRAA_v2_PAKDD_Submission.md`/
`.tex`) were cross-checked against `STAGE_C_MASTER_REPORT.md` and
`README.md` (93.5%/69.7%, 111/112 significance, 6.00×/3.00× overhead,
baseline comparators) — all match exactly, no drift found. The problem is
that this correction discipline stopped at the paper layer and never
propagated down into the docs meant to explain the paper to a newcomer.

### 3.4 `docs/audit/` — the audit trail auditing itself (12 files, 3,884 lines)

Sampling strategy: every HIGH/CRITICAL finding across all 12 documents was
checked against the current codebase/docs; lower-severity findings were
spot-checked. This is the section most directly relevant to whether the
repo's self-verification claims can be trusted.

- **[HIGH] `publication_roadmap.md` has gone stale on the exact axis it was
  already fixed for once.** Last touched by commit `586a93c`
  (2026-08-02), it still reads "Last refreshed: 2026-07-25" and "142/142
  tests passing" — while the real count is 558. This is precisely the
  defect `AUDIT_REMEDIATION_PLAN_2026-07-25.md`'s finding R4 already closed
  ("Fixed — `6ed6fc1`") on 2026-07-25. It regressed a second time and no
  subsequent audit — including both 2026-08-02 audits — caught it before
  this pass.
- **[HIGH] D1 ("is FedGRAA published?") — the single most-repeated
  commitment in this entire corpus — is still open.** Named as the
  "highest-leverage... answerable in one conversation" open question in
  `RESEARCH_DIRECTION_AUDIT_2026-07-25.md`, echoed in
  `STAGE_AB_VERIFICATION_AUDIT_2026-07-25.md`, deferred-not-blocking in
  `AUDIT_REMEDIATION_PLAN_2026-07-25.md`, and reported as "still has not
  happened" in `INDEPENDENT_FULL_AUDIT_2026-07-26.md`. Current
  `publication_roadmap.md:54`: **"The provenance question is unresolved."**
  Still true. Meanwhile `STAGE_C_IMPACT_ON_STAGE_AB_2026-07-27.md` records
  that Stage C's direction was decided unilaterally, without the external
  mentor conversation this exact finding said should gate it. Spans 4+
  documents and 8+ days without closing.
- **[MEDIUM]** `PRE_COMPUTE_AUDIT_2026-07-28.md`'s M7 (CI hardware/AVX512
  non-determinism) was closed by *documenting* the risk
  (`COMPUTE_INFRASTRUCTURE_2026-07-27.md`), not by the structural fix it
  actually recommended — `docs-check.yml` still runs on unpinned
  `ubuntu-latest` with no patch-version pin.
- **[LOW]** `INDEPENDENT_FULL_AUDIT_2026-07-26.md`'s own "232 vs 262" doc-
  drift finding appears to be a false positive — the two cited numbers
  describe genuinely different quantities in the current tree, not a
  contradiction.

**No hard contradictions found** among the 12 documents' scientific claims;
three spot-checked "✅ Fixed" claims were verified genuinely fixed in
current code (the FedSND citation annotation, the removed "Published"
framing, the NaN-asymmetry fix in `aggregation.py`). Zero broken links
found across all 12 files (programmatically checked).

**Overall verdict on the audit trail:** genuinely rigorous on the
*scientific/numeric* axis (every recomputation checked across ~1,000+
independent spot-checks in this corpus alone came back correct, and
multiple documents run real negative controls to prove their own checkers
can fail) but shows a real, repeated **whack-a-mole pattern specifically in
the documentation/bookkeeping layer** — the same defect class recurs on the
same documents after being marked fixed, because the gates check artifact
*structure* (does the file exist, is the JSON shaped right), and nothing
in this repo checks whether an entry-point document's prose-numbers are
still current. Several of the audit documents name this dynamic explicitly
themselves.

### 3.5 `docs/archive/` (12 files, 4,547 lines)

The prior pass found 3 inaccuracies in this folder's own `README.md`
supersession table (wrong path, one file miscatalogued as archived when
it lives elsewhere, two files omitted). This pass read the 12 archived
documents' actual content.

- **[HIGH] 3 of 12 files are missing the "⚠️ ARCHIVED" banner** the
  folder's own README promises every file carries: `GPU_DECISION_AND_
  NEXT_STEPS.md`, `speedup_plan_v2.md`, `strict_reproduction_report.md`.
  The latter two are also the two files the prior pass found omitted from
  the supersession table — so they're both untabled *and* unbannered.
  `speedup_plan_v2.md` is actively cited elsewhere as ground truth, so it
  arguably shouldn't be "archived" at all.
- **[MEDIUM]** `reproduction_gaps.md` duplicates its own entire content:
  lines 1–315 mark every gap "✅ Fixed"; lines 318–694 repeat the identical
  gap descriptions with no status markers, framed as "Fix required" —
  three layered, undated claims (not-fixed → fixed → reverted, per the
  file's own banner) about the same code paths, no way to tell from the
  file alone which is current.
- **[Important negative result, checked directly]** The single most
  important question for this folder — *are any of these "archived" gap
  analyses actually still-live bugs?* — was checked by grepping current
  `fedgraa/` code for the most concrete claims in
  `paper_vs_code_line_by_line_gap_analysis.md` and `reproduction_gaps.md`
  (removed config toggles, the NumPy-vs-Keras backend gap, UNSW-NB15 data
  presence, the `round_zero_zero` default). **All were confirmed genuinely
  resolved** — none is a live bug hiding under an "archived" label.

### 3.6 `docs/research_c/` (19 files, 10,329 lines) — literature/citations and planning/compute

**Citation integrity — assessed LOW risk**, the opposite of what the prior
pass's "highest-risk unread area" framing anticipated. The corpus actively
hunts and excludes suspected-fabricated citations across 6 dated passes
(FedSND, FeCWGAN-GP, CT-BRFL, pFedCross, and others are explicitly tagged
"do not cite"), and a live web search run during this audit reproduced the
corpus's own hallucination-detection signature exactly (a fluent, detailed
description with zero real matching links) — independently corroborating
both that the phenomenon is real and that the corpus's own detection method
works. `FedPGO.md`'s algorithm (Eqs. 6–10, Algorithms 1–3) was checked
line-by-line against `fedgraa/baselines_t5_fedpgo.py` and matches exactly,
including 8 explicitly disclosed judgment calls for the paper's own
internal ambiguities. One cosmetic finding: `sagitra_report.md` and
`docs/archive/README.md` both point to the same wrong path
(`../current/privacy_compliant_fedgraa_redesign.md` instead of
`../research_c/...`) — copy-pasted from each other, low practical risk
since `sagitra_report.md` is itself explicitly disclaimed as non-citable.

**Planning/compute docs — one real stale-status bug found, as the audit
brief specifically asked us to check for:**

- **[HIGH] `REMAINING_COMPUTE_WORK.md`** still describes T7 as "Attack
  harness does not exist — not attempted this pass" and its G5 gate as
  "⏳ not run," and its top-line "what is left" summary still lists the
  full compute campaign as outstanding. Per `STAGE_C_MASTER_REPORT.md`,
  T7 is fully implemented and run (90/90 cells) and a manuscript exists.
  The file has one dated correction (2026-08-02) for a *different* section
  (T3/T5 framing) that never touches the T7 row or the "what is left"
  conclusion — a reader trusting the file's own body over its one pointer
  gets a materially wrong picture.
- **[HIGH]** `docs/archive/README.md`'s framing of
  `privacy_compliant_fedgraa_redesign.md` as something the archive
  supersedes is substantively wrong, not just mis-pathed: two other current
  documents (`FEDGRAA_V2_MASTER_PLAN.md`, `Final_Stage_C.md`) call it
  "still authoritative" for specific sections and cite it as a live design
  reference. It is not an archived artifact at all.
- **[MEDIUM]** `Final_Stage_C.md` reports "≈2.98 GB" UNSW-NB15 traffic;
  `FEDGRAA_V2_MASTER_PLAN.md`'s own same-day F5 finding corrects this exact
  figure to 2.908 GB (a stale `n_features` assumption), but
  `Final_Stage_C.md` still carries the uncorrected number.
- **[LOW-MEDIUM]** `STAGE_C_T7_JUDGMENT_CALLS_2026-08-02.md` contains
  exactly 18 numbered judgment-call entries (JC-T7-1 through JC-T7-18);
  two other documents (the remaining-work plan, the master report) both
  cite "19" — an apparent miscount propagated across two downstream docs.
- **[LOW]** `STAGE_C_COMPUTE_SESSION_UPDATE_2026-07-30.md`'s sections run
  `5 → 6 → 8 → 7` — out of order.
- **Confirmed consistent, no bug:** `FEDGRAA_V2_THEORY.md`'s description of
  `ProjectedFedGRAA`/`FedGRAAv2`/`two_phase_composite_grad` matches
  `fedgraa/aggregation.py` exactly. T7's 90-cell grid and FedPGO's
  faithful-rerun numbers both cross-check to the decimal against
  `STAGE_C_MASTER_REPORT.md`. Zero broken links across all 19 files.

---

## 4. Cross-cutting patterns

Two patterns recur strongly enough, across independently-assigned agents
that couldn't see each other's work, to be worth naming explicitly:

1. **Doc-drift is concentrated in "entry-point" and "status/plan" documents,
   never in the numerically-derived tables themselves.** Every table this
   pass or the prior one traced back to raw JSON via the audit scripts
   checked out exactly. Every place a number is *hand-typed into prose* —
   a test count, a "last refreshed" date, a "what's remaining" list, a
   README quickstart table — is where drift accumulates. The repo's gates
   are excellent at "does this artifact exist and is it shaped right" and
   have no mechanism at all for "does this document's own prose agree with
   itself, or with the newer document three commits later that superseded
   its claim." This is the same meta-finding the prior pass's §3.5
   identified from a much smaller sample; this pass shows it generalizes
   across the whole doc tree, not just `README.md`.
2. **Corrections applied at one layer don't reliably propagate to
   downstream/explainer layers.** The paper was corrected; `understanding/`
   wasn't told. A byte-count was corrected in one plan doc; a sibling
   summary doc kept the old number. A judgment-call count was fixed in one
   place; two other documents still cite the pre-fix count. No single
   instance is severe, but the pattern is structural: this project has
   real, working correction discipline at the point where a defect is
   *found*, and no working mechanism for propagating that correction to
   every place the corrected fact is repeated.

Neither pattern implicates the underlying science. Every audit script
re-derivation (`audit_stage_{a,b,c}.py`, re-run fresh in §1) passed, and no
raw-data or statistical-methodology defect was found anywhere in this pass.
The risk surface is entirely in the prose layer describing that science —
which is exactly what "correctness and completeness" auditing is supposed
to catch, and exactly why this pass was worth completing.

---

## 5. Consolidated findings, all severities, this pass only

*(Excludes the 4 findings already reported in the prior pass's §3, which
still stand and are re-summarized in §3.1 above.)*

| # | Area | File | Severity | One-line summary |
|---|---|---|---|---|
| 1 | Code | `run_experiments.py:81-118` | HIGH | `--round-zero-zero` CLI default silently contradicts library/README default |
| 2 | Docs | `docs/current/STAGE_C_MASTER_REPORT.md:355-369` | HIGH | New-test-count table (105) doesn't reconcile with prose (137/144) |
| 3 | Docs | `docs/current/publication_roadmap.md` | HIGH | Recurrence of an already-"fixed" stale-status bug (test count, refresh date) |
| 4 | Docs | `docs/audit/*` (4+ files) | HIGH | D1 publication-provenance question open 8+ days, work proceeded anyway |
| 5 | Docs | `docs/research_c/REMAINING_COMPUTE_WORK.md` | HIGH | Claims T7/G5 not attempted; actually 90/90 cells complete |
| 6 | Docs | `docs/archive/README.md` | HIGH | `privacy_compliant_fedgraa_redesign.md` wrongly framed as archived; it's live/authoritative elsewhere |
| 7 | Docs | `docs/archive/{GPU_DECISION...,speedup_plan_v2,strict_reproduction_report}.md` | HIGH | 3 files missing the promised "⚠️ ARCHIVED" banner |
| 8 | Docs | `docs/paper/draft1.md` | HIGH | Self-contradictory "unedited" banner; body has dated post-review edits |
| 9 | Docs | `docs/paper/draft1.md` §Abstract vs §5 | HIGH | Rewritten abstract's corrected numbers absent from the file's own body tables |
| 10 | Meta | `docs/audit/INDEPENDENT_REPO_AUDIT_2026-08-02_EXTERNAL.md:327` | HIGH (self-inflicted, easy fix) | Prose example parses as a broken link, breaks 5/9 CI gates if committed as-is |
| 11 | Code | `fedgraa/baselines_t7_adversarial.py:186-188` | MEDIUM | Dormant undisclosed `top_k_frac` override in SecFedNIDS filter |
| 12 | Code | `scripts/audit_stage_c.py:533` | MEDIUM | `"###"` substring match could truncate a doc-transcription check early |
| 13 | Code | `scripts/build_v2_significance_table.py:138-139` | MEDIUM | Hardcoded "10 matched seeds" text not tied to real per-comparison count |
| 14 | Code | `scripts/render_v2_frontier.py:169-170` | MEDIUM | Hardcoded ">=10 seeds" caption not tied to real per-cell count |
| 15 | Code | `scripts/modal_frontier_campaign*.py` | MEDIUM | Cell-failure signal not persisted, only in console output |
| 16 | Docs | `understanding/10_SOTA_adaptive_aggregation.md:343` | MEDIUM | Endorses a citation the project's own audit confirmed fabricated |
| 17 | Docs | `docs/paper/FedGRAA_v2_PAKDD_Submission.tex` ref [12] | MEDIUM | arXiv-ID/venue mismatch in a since-superseded reference slot |
| 18 | Docs | `understanding/03,05,06_*.md` | MEDIUM | Frozen at pre-review state; omits the project's own critical convergence-proof finding |
| 19 | Docs | `docs/audit/PRE_COMPUTE_AUDIT_2026-07-28.md` (M7) | MEDIUM | CI hardware-pinning risk closed by disclosure, not by the recommended fix |
| 20 | Docs | `docs/archive/reproduction_gaps.md` | MEDIUM | Entire content duplicated with contradictory undated fixed/not-fixed framing |
| 21 | Docs | `PAKDD/README.md:175-176` | MEDIUM | n-sweep run-count table stale (9/14 claimed vs. actual 15/20) |
| 22 | Docs | `PAKDD/README.md:171-186` | MEDIUM | CLI flag table omits 5 real flags |
| 23 | Docs | `CONTRIBUTING.md:12-19` | MEDIUM | PR checklist omits Stage C's 3 CI gates |
| 24 | Docs | `MENTOR_WALKTHROUGH.md` (3 locations) | MEDIUM | 3 different stale test counts in one document |
| 25 | Docs | `docs/research_c/Final_Stage_C.md` | MEDIUM | Stale UNSW byte figure (2.98GB, corrected elsewhere to 2.908GB) |
| 26 | Docs | `docs/research_c/STAGE_C_T7_JUDGMENT_CALLS_2026-08-02.md` | LOW-MEDIUM | 18 actual entries vs. "19" cited in two other docs |
| 27 | Docs | `docs/current/STAGE_A_MASTER_REPORT.md` Appendix B | MEDIUM | Wrong directory (`docs/current/` vs `docs/research_c/`) for 2 files |
| 28 | Docs | `docs/current/fedgraa_convergence_defect.md` §7 | MEDIUM | Second disclosed algorithm deviation not propagated to Stage A/B master reports |
| 29 | Code | `scripts/build_v2_frontier_table.py:65-86` | LOW | `_is_t1_cell` doesn't gate on `attacker_fraction==0` (latent, unreachable today) |
| 30 | Docs | `docs/current/fedgraa_divergence_diagnosis.md` | LOW | Section numbering skips §7 |
| 31 | Docs | `docs/current/publication_roadmap.md` | LOW | Off-by-one line-count citation (976 vs 977) |
| 32 | Docs | `docs/research_c/STAGE_C_COMPUTE_SESSION_UPDATE_2026-07-30.md` | LOW | Section numbers out of order (5,6,8,7) |
| 33 | Docs | `PAKDD/README.md` project-layout tree | LOW | Omits the real `archive/` folder |
| 34 | Docs | `docs/current/STAGE_C_MASTER_REPORT.md` header | MEDIUM | "Compiled against 586a93c" stale relative to content describing the next commit |

---

## 6. Recommendations, in priority order

1. **Fix the self-inflicted CI break before committing this audit or the
   prior one** (§1.1) — trivial text change, otherwise the next push
   breaks 5 gates over a documentation artifact about documentation
   hygiene.
2. **Fix `run_experiments.py`'s `--round-zero-zero` default** (§2.2) — this
   is the one finding in this whole pass that changes actual program
   behavior, not just prose. Either give it an explicit `default=True` or
   stop unconditionally overriding `FedConfig`'s default when the flag is
   absent. Then fix `tests/test_run_experiments_cli.py` to import the real
   parser instead of a hand-duplicated shadow one, so this class of bug is
   caught by the suite going forward.
3. **Reconcile `STAGE_C_MASTER_REPORT.md`'s new-test-count table against
   its own prose** (§3.2) — find the missing 32, most likely an omitted
   row or two in the §12.1 table.
4. **Re-refresh `publication_roadmap.md`** (§3.4) a second time, and this
   time consider adding the "does every test-count/date mention in this
   file agree with every other one" self-check the prior pass's §3.5
   already sketched — this is the second time this specific document has
   gone stale on the identical axis.
5. **Close or explicitly re-scope D1** (§3.4) — 8+ days and 4+ documents of
   "this is the single highest-leverage next step" with no resolution, while
   work proceeded past the gate this finding said should block it, is worth
   a direct decision either way rather than continuing to carry it forward.
6. **Batch-fix the `docs/archive/` bannering and cataloguing issues**
   (§3.5, §3.6): add "⚠️ ARCHIVED" banners to the 3 missing files (or
   un-archive `speedup_plan_v2.md`/`GPU_DECISION_AND_NEXT_STEPS.md` if
   they're actually still-current ground truth, which the evidence
   suggests), fix the `privacy_compliant_fedgraa_redesign.md` path and
   "archived" framing in `docs/archive/README.md`, and de-duplicate
   `reproduction_gaps.md`.
7. **Propagate the paper's own corrections into `understanding/`** (§3.3) —
   this is the highest-value single documentation fix available: a
   newcomer following the intended onboarding path currently learns a
   materially wrong account of both the algorithm's correctness and its
   headline numbers, from files whose entire purpose is to be that
   onboarding path.
8. **Update `REMAINING_COMPUTE_WORK.md`'s T7/G5 rows and bottom-line
   summary** (§3.6) to match `STAGE_C_MASTER_REPORT.md`'s actual completion
   status.
9. **Make the significance/frontier table & figure scripts derive their
   seed-count claims from real data instead of hardcoding them** (§2.3,
   findings 13–14) — cheap, and removes the one place a future partial
   compute failure could silently overclaim statistical power in a
   human-facing artifact.
10. Smaller, contained fixes with no dependencies on the above, safe to
    batch: `PAKDD/README.md`'s n-sweep table and flag-reference gaps (§3.1),
    `CONTRIBUTING.md`'s PR checklist (§3.1), `MENTOR_WALKTHROUGH.md`'s 3
    stale counts (§3.1), the citation cross-references in `understanding/10`
    and the paper's ref [12] (§3.3), the JC-T7 count and UNSW byte-count
    mismatches in `docs/research_c/` (§3.6), and the two missing/
    out-of-order section numbers (§3.2, §3.6).

---

## 7. What this pass did not find

Consistent with the prior pass: no security-relevant logic error in
`privacy.py` or `secagg.py` (both fully read across the two passes). No
test found to be actively lying about what it verifies. No numeric claim
spot-checked against raw JSON, in either pass, was found wrong — every
single re-derivation via the audit scripts, in both passes, reproduced
exactly. No evidence of fabricated citations having actually been
*incorporated* anywhere in the current paper or literature docs (the
opposite — active, working self-policing was independently corroborated
via a live web search in §3.6). No silent data-fabrication or padded
sample counts in any generated table or figure, spot-checked against live
JSON.

## 8. Coverage statement

Combined across both passes: **100% of `fedgraa/` (24/24 files), 100% of
`scripts/` (28/28 files), 100% of `tests/` (42/42 files, 9,444 lines),
and effectively 100% of tracked Markdown/TeX** (~31,700 lines across
`README.md`, `docs/current/`, `docs/paper/`, `docs/audit/`, `docs/archive/`,
`docs/research_c/`, `understanding/`, and top-level misc files) **have now
been read in full, line by line**, across the two passes. The only
material not read directly is: the two compiled PDF exports in
`docs/paper/` (build artifacts of sources that were read), and the raw
CSV/data files under `data/` (~14 GB — validated instead via the audit
scripts' independent statistical re-derivation from that same raw data,
which is the stronger check for that material in any case). This closes
every gap the prior pass's §1.3 disclosed as unread.

---

## 9. Remediation record (applied same session, after this report was written)

The repo owner asked for every finding above to be fixed. All 34 items in
§5's table, plus the 4 findings inherited from the prior pass's §3, were
addressed directly in the working tree immediately after this report was
compiled, including all 10 items in §6's recommendation list. That includes
every code fix (§2: the `run_experiments.py` default bug, the
`baselines_t7_adversarial.py` dead branch, the `audit_stage_c.py` and
`check_stage_c_complete.py` robustness gaps, the hardcoded seed-count text
in the significance/frontier renderers, the `build_v2_frontier_table.py`
latent gap, and a new persisted failure manifest for the Modal campaign
scripts) and every documentation fix (§3: stale test counts and dates
across `README.md`, root `README.md`, `CONTRIBUTING.md`,
`MENTOR_WALKTHROUGH.md`, `publication_roadmap.md`, and
`STAGE_C_MASTER_REPORT.md`; the `docs/archive/` banners and supersession
table; the `understanding/` correction pointers; the `draft1.md` banner and
citation fixes; and the `docs/research_c/` status/count corrections). The
test suite and all 9 gates were re-run fresh after every fix; results are
in the commit this document ships with, not restated here to avoid this
report itself becoming another stale snapshot the moment a future commit
moves the numbers again — check the gates directly rather than trusting
this paragraph's numbers after the fact.

The one exception, by design: this section and the two `docs/audit/`
report files themselves are left as an honest historical record of what
was found and when, per §1.1's fix (which corrected the self-referential
link bug but did not rewrite the surrounding narrative to pretend the
findings were never true). §5's table and §6's recommendations describe
the state *as discovered*, not the state after remediation.

---

*This document is an independent, ad hoc audit output, written to a file
in this session's working tree per the repo owner's request, and later
committed at the owner's direction alongside the fixes it recommended (§9).
It is not part of the repo's own numbered `docs/audit/*` sequence in the
sense of being produced by that sequence's own tooling, but it now ships
in the same commit as the fixes it describes.*
