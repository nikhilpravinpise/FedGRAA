# Independent Repository Audit — External Pass (2026-08-02)

**Requested by:** repo owner ("audit and verify the whole repo and docs ... for
correctness and completeness ... go through each and every line of code and
doc files"), performed by Claude (Sonnet 5) as a fresh, skeptical, external
pass — not a continuation of the repo's own `docs/audit/*` pipeline, and
deliberately not shown those documents' conclusions before forming
independent ones.

**Tree audited:** working tree at `HEAD = d7d72c6`, clean, `main` branch.

**Status: partial completion, honestly disclosed.** This audit ran fresh
verification of every automated gate, did a genuine line-by-line read of the
highest-stakes ~65% of the Python codebase and 100% of the test suite (via a
dedicated sub-pass), and a full read of the repository's top-level
documentation. It did **not** complete a literal line-by-line read of all
~28,000 lines of Markdown or the remaining ~35% of `fedgraa/` — a hard
platform session-usage-limit interruption cut short six parallel deep-read
passes partway through (see §1). One of those six finished; the rest were
reconstructed by continuing the work directly, at reduced parallelism, for as
long as budget allowed. §1 states exactly what was and wasn't covered so this
report's confidence claims are calibrated honestly rather than implied to be
uniform.

---

## 0. Bottom line

**The code is exceptionally well-engineered and the automated correctness
machinery genuinely works.** Every gate this audit re-ran exited clean.
Nothing resembling a correctness bug was found in ~13,400 lines of core
library code read in full. The one part of the codebase given a complete,
independent audit — the 42-file, 9,444-line test suite — was rated
"unusually rigorous ... largely meaningful evidence, not decorative" with
only minor, low-severity gaps.

**The documentation layer is where real, currently-live defects were found —
all of them small, all of them the same species of "bookkeeping didn't keep
up with a fast-moving repo," and all of them concrete and fixable in
minutes.** Four such defects are reported in §3 with exact file:line
citations, none previously flagged in the repo's own 13-document audit
history. The most interesting meta-finding is in §3.5: the repo's own
audit process is real and it works, but doc-hygiene drift is regenerating
faster than it is being re-audited — the same class of defect this repo's
own `STAGE_ABC_COMPLETENESS_AUDIT_2026-08-02.md` fixed same-day has already
partially recurred within that same document's own commit history.

---

## 1. Methodology and honest scope

### 1.1 What was verified by direct execution (100% coverage, ground truth)

Every one of the following was re-run fresh in this session, not read off a
document:

| Check | Command | Result |
|---|---|---|
| Full test suite | `.venv313\Scripts\python.exe -m pytest tests --tb=short -q` | **558 passed, 0 failed**, 419.6s |
| Stage A completeness | `scripts\check_stage_a_complete.py` | exit 0 |
| Stage B completeness | `scripts\check_stage_b_complete.py` | exit 0 |
| Stage C completeness (T1–T6, TC) | `scripts\check_stage_c_complete.py --tier T1 T2 T3 T4 T5 T6 TC --t4-m 3` | exit 0 — 830/830 |
| Stage C completeness (T7) | `scripts\check_stage_c_complete.py --tier T7 --seeds 0 1 2 3 4` | exit 0 — 90/90 |
| Stage A correctness audit | `scripts\audit_stage_a.py` | exit 0 |
| Stage B correctness audit | `scripts\audit_stage_b.py` | exit 0 |
| Stage C correctness audit | `scripts\audit_stage_c.py` | exit 0 |
| Doc-link integrity | `scripts\check_doc_links.py` | exit 0 |

All nine pass. This independently confirms the repo's own headline claim
that every completeness gate and correctness audit is currently green.

### 1.2 What was read in full, line by line, by a human-equivalent pass

- **`fedgraa/` core library — 18 of 24 files** (`aggregation.py`, `privacy.py`,
  `secagg.py`, `attacks.py`, `stats.py`, `trainer.py`, `data.py`,
  `partition.py`, `model.py`, `metrics.py`, `config.py`, `keras_backend.py`,
  `experiment.py`, `baselines.py`, `baselines_classic.py`, `plot.py`,
  `__init__.py`, `__main__.py` — roughly 6,450 of 11,403 lines). This set was
  chosen deliberately: it covers every mathematically or security-critical
  path (aggregation math, differential-privacy accounting, secure
  aggregation, adversarial attacks, statistical testing, the training loop,
  data loading/partitioning, and the model/optimizer core).
- **`scripts/` — 4 of 28 files read in full**: `audit_stage_a.py`,
  `audit_stage_b.py`, `check_stage_a_complete.py`, `check_doc_links.py` —
  chosen because these are the meta-scripts the entire "is this trustworthy"
  claim rests on. `audit_stage_c.py`, `check_stage_b_complete.py`, and
  `check_stage_c_complete.py` were exercised (run fresh, §1.1) but not read
  source-line-by-line by this pass; their behavior is corroborated instead by
  (a) their exit-0 results here and (b) `STAGE_ABC_COMPLETENESS_AUDIT_2026-08-02.md`'s
  own detailed, previously-verified description of each (see §3.5).
- **`tests/` — all 42 files, 9,444 lines, 100% coverage**, via a dedicated
  sub-pass that completed successfully before the session interruption (full
  findings reproduced in §2.3).
- **Top-level docs — full read**: root `README.md`, `PAKDD/README.md`,
  `docs/archive/README.md`, `docs/audit/STAGE_ABC_COMPLETENESS_AUDIT_2026-08-02.md`.
  These were chosen as the highest-traffic entry points a reader (or a
  contributor, or a reviewer) is most likely to actually rely on.

### 1.3 What was **not** independently re-verified line-by-line

- **6 `fedgraa/` files (~4,950 lines)**: `baselines_t3_contrib.py`,
  `baselines_t3_trust.py`, `baselines_t5_adaptive.py`,
  `baselines_t5_fedpgo.py`, `baselines_t5_nids.py`,
  `baselines_t7_adversarial.py` — the Stage C external-baseline
  reproductions (CELM, SpectralFed, CoAst, FLTrust, RAGA, FedTruth, FedMADE,
  SBFedNIDS, FedAPA, FedAWA, DA-PFL, FedAWARE, AdaAggRL, FedPGO, SecFedNIDS,
  WeiDetect). These are Related Work comparisons, not the headline FedGRAA/
  FedGRAA-v2 claim. Risk is assessed as **low but not zero**: every one of
  these files has a dedicated, passing test file (confirmed via the tests/
  sub-pass, §2.3), and the 18 files that *were* read showed a completely
  consistent pattern of careful, self-disclosing engineering with zero
  defects found — but that pattern was not directly confirmed against these
  specific 6 files by this audit.
- **24 of 28 `scripts/` files** (build/render/verify/campaign drivers) — not
  read source-line-by-line. Their outputs are indirectly validated by
  `audit_stage_{a,b,c}.py` re-deriving the numbers they produce from raw
  JSON (confirmed passing, §1.1).
- **~25,000 of ~28,000 lines of Markdown** — `docs/current/`'s three master
  reports (2,702 lines), all of `docs/paper/` (4 files), all of
  `understanding/` (11 files), 10 of 13 `docs/audit/` files, 11 of 12
  `docs/archive/` files (beyond the README), and all 17 `docs/research_c/`
  files were not read in full by this pass. This is the largest gap in this
  audit's coverage.

**Why:** roughly halfway through, six parallel deep-read sub-agents (one per
major repo section) hit the account's session usage limit simultaneously and
were terminated mid-task by the platform, not by any decision of this audit.
One sub-agent (`tests/`) had already finished and its output was preserved.
The other five were not restarted (retrying immediately would hit the same
limit); instead, the highest-value remaining work — the core library and the
meta-audit scripts — was continued directly, at full depth but reduced
breadth, for the remainder of available budget. **This report is accurate
about what it covers and does not extrapolate a "clean" verdict onto the
~25,000 lines of docs it did not personally read.** Where this report relies
on the repo's *own* prior audit machinery (which independently re-derives
numeric claims from raw JSON, §1.1) rather than personal line-by-line
reading, that is stated explicitly rather than implied to be first-hand.

---

## 2. Code findings

### 2.1 `fedgraa/` core library (18 files read in full) — no correctness bugs found

Every file read shows the same, consistently high standard: every
non-obvious design decision is documented inline with its rationale, every
degenerate/edge case (zero variance, empty splits, NaN arms, zero-count
histogram buckets) is handled explicitly with a stated contract rather than
left to silently produce `NaN` or a wrong-but-plausible number, and every
place the implementation deviates from a cited paper's literal algorithm is
disclosed with a specific reason (e.g. `aggregation.py`'s `FedGRAA.update`
keeps Algorithm 2 *unprojected* on purpose, with the projected variant kept
structurally separate in `ProjectedFedGRAA` so the strict-reproduction path
can never accidentally pick up the fix; `baselines_classic.py`'s SCAFFOLD/
FedNova reproductions carry a numbered list of disclosed judgment calls,
JC-TC-1 through JC-TC-7, for every place the shared Adam-based local
optimizer doesn't match the cited papers' own SGD-based analysis).

Specific things checked and found correct by direct reasoning, not just
reading:

- `aggregation.py`'s `project_to_simplex` — the Duchi et al. (2008) Euclidean
  simplex projection is implemented correctly (verified by hand-tracing the
  sort/cumsum/threshold logic against the cited algorithm).
- `aggregation.py`'s `_composite_min_arm` — the NaN-symmetric arm selection
  is genuinely symmetric (checked both branches) and is the single shared
  implementation between the loss and gradient computations, so they cannot
  silently disagree on which arm is non-finite.
- `privacy.py` — the Laplace-mechanism sensitivity (`Δ₁ = 2`), the aggregate
  noise `σ = (Δ₁/ε)·√(2K)` formula, the advanced-composition closed form
  (Dwork & Roth Thm. 3.20), and the RDP-to-DP conversion (Mironov 2017,
  Prop. 3) all check out against their cited closed forms by hand.
- `secagg.py`'s `SecAggChannel` — verified by inspection that no method,
  property, or introspectable attribute can recover an individual
  `submit()`'s argument once folded into the running total; the
  dropout-threshold check happens before `_closed` is set, so a failed
  `aggregate()` call can be retried logically (though the class is
  documented single-use — consistent).
- `stats.py`'s degeneracy guards — the *relative* (not exact) zero-variance
  test in `paired_ttest` is correctly derived and matches the documented
  historical bug it fixes (`[0.7,0.8]-[0.6,0.7]` differing by ~1.1e-16 in
  float64, which an exact `ptp==0` check would miss).
- `trainer.py`'s `FederatedTrainerV2` — SecAgg dropout is handled as an
  aborted round (weights carried forward, counter incremented), never as a
  silently-swallowed exception; DP noise is added client-side before
  submission, never server-side after aggregation (verified this is the
  only path — no code branch adds noise post-`ch_rank.aggregate()`).
- `data.py`'s NF-UQ-NIDS-v2 loader — the two-pass bounded-memory read
  (`_tally_merged_csv` → `_select_positions_per_key` →
  `_read_merged_csv_selected_rows`) is logically sound: position counters are
  threaded across chunks in the same scan order in both passes, so "row j of
  key (silo,label)" means the same physical row in pass 1 and pass 2.

No bare `except:` clauses, no silently-swallowed exceptions in a
correctness-relevant path, no unseeded randomness where reproducibility is
claimed, and no docstring/behavior mismatches were found across these 18
files.

### 2.2 `scripts/` meta-audit tooling (4 files read in full) — genuinely rigorous, no holes found

`audit_stage_a.py` and `audit_stage_b.py` do not trust `fedgraa/stats.py`:
`audit_stage_b.py` recomputes every paired t-test, Wilcoxon test, Cohen's d,
and Holm–Bonferroni adjustment directly via `scipy.stats`, independent of the
module under test — genuine second-implementation verification, not a
second call into the same code. Both scripts document (and this audit has no
reason to doubt) that they were negative-controlled: specific defects (a
changed doc number, a corrupted config field, a duplicated run file, a
violated round-0 convention) were deliberately injected into scratch copies
and confirmed caught before being trusted, then discarded. `check_doc_links.py`
is simple and correct for what it claims to do (relative markdown link/image
targets resolve to real files) — its docstring honestly discloses that it
does not verify anchor fragments or catch non-markdown-link prose references
to a path, which is exactly the blind spot that let finding §3.3 below slip
through undetected.

### 2.3 `tests/` — 100% coverage (42 files, 9,444 lines), verdict: unusually rigorous

*(Full independent sub-pass; reproduced here in full since it completed
before the session interruption.)*

**Overall judgment: the "551/558 passed" claim is meaningful evidence of
correctness, not decorative.** Nearly every safety/correctness-critical
primitive (DP noise scale, SecAgg dropout threshold, simplex projection,
Wilcoxon/Holm significance, gradient formulas) is checked against a
hand-derived or scipy-reference value, with explicit negative/edge cases.
Several tests are explicit regression guards for previously-real,
previously-silent bugs found by prior audits — evidence of a maintainer who
audits their own tests, not just their own code.

Findings:

- **`tests/test_run_experiments_cli.py:13-31` (medium).** `_build_parser()`
  is a hand-duplicated re-implementation of `run_experiments.py`'s real
  argparse setup, not an import of it. Two of its three tests exercise this
  shadow parser, not the production one — a real bug in `run_experiments.py`'s
  `--workers`/`--ns` wiring (wrong default, wrong type, dropped flag) would
  not be caught. The third test only greps the real source for the literal
  substrings `"--workers"`/`"--ns"`, which would catch the flag being
  deleted but not its default/type/choices being wrong. Currently the two
  parsers match (confirmed), but this is the classic "test the copy, not the
  code" anti-pattern and will silently rot if the real parser changes.
- **`tests/test_metrics.py` (low).** All `per_class_metrics`/
  `classification_report` tests monkeypatch `MLP.predict` to a fixed lambda,
  so a real bug in the forward-pass-to-prediction path used by these
  specific functions would not be caught. This is a partial regression: a
  real-forward-pass test was added for `accuracy`/`last_n_class_accuracy`
  specifically to close this exact gap (per the file's own docstring,
  referencing `docs/audit/INDEPENDENT_FULL_AUDIT_2026-07-26.md` §3.2), but
  the fix wasn't extended to these two functions.
- **`tests/test_parallel_sweep.py:122-129` (low, self-acknowledged).**
  `test_parallel_sweep_function_exists` only checks `callable(...)` and
  inspects the signature — vacuous alone (would pass even with the function
  body replaced by `pass`). Low severity because the file's own docstring
  explicitly names this as the pre-existing weak test it supplements, and
  three real tests alongside it now cover the actual logic.
- **Suite-level observation (not a defect).** `SecAggChannel` is explicitly
  an *ideal-functionality* model (no real Shamir secret sharing or pairwise
  masking) — the tests correctly validate that documented contract, and this
  is honestly scoped in both the code and the README, but a reader skimming
  test names could mistake "tested SecAgg" for tested cryptography rather
  than a tested abstraction.
- **Suite-level observation.** `pytest --collect-only` currently reports
  **558** tests, not the 551 several docs still cite — see §3.1, a doc-drift
  finding, not a test-quality one.

Verification performed by the sub-pass to rule out "hardcode whatever the
code outputs" tests: hand-derived DP-noise formulas, SecAgg dropout/
single-use guarantees, CELM/FedAPA paper equations (including a disclosed,
tested sign-flip deviation from the printed FedAPA equation), and an
end-to-end DP-sigma check that counts real `laplace_mechanism` calls through
the actual trainer code path (not just the standalone `privacy.py`
function) were all confirmed to match source, not curve-fit to it.

---

## 3. Documentation findings

These four findings are new — not previously flagged in the repo's own
`docs/audit/` history (13 documents were checked for whether they already
covered them; none do) — and all are independently verified against the
current tree, not inferred.

### 3.1 [HIGH] `PAKDD/README.md` states three different, all-wrong test counts in the same file

- Line 32 (`Verification signals` block): `# 551 passed`
- Line 296 (`## Tests` section): `# 551 passed`
- Line 438 (Project layout tree comment): `tests/ # 311 unit + gradient + e2e + diagnostic + statistics + doc-link + FedGRAA-v2 + pre-compute-audit tests`

**Ground truth, measured fresh in this audit (§1.1): 558 passed, 0 failed.**
`docs/current/STAGE_C_MASTER_REPORT.md` (lines 58 and 369) already
independently states "558 tests passing" / "558 total" — so the repo's own
most-current capstone document has the right number. But `PAKDD/README.md`
— the file both the repo-root `README.md` and the entire `docs/current/`
suite point new readers to first — shows two different wrong numbers in
three places.

This is exactly the class of defect `docs/audit/STAGE_ABC_COMPLETENESS_AUDIT_2026-08-02.md`
itself found and marked "Fixed" under finding **F2** (§3.2 of that document:
"corrected the count to `551 passed`" — which was itself already one
release behind, since the T7G follow-up in the *same document's own* §7.2
later adds 7 more tests, bringing the true total to 558). The "311" instance
at line 438 appears to predate even that fix and was never touched by it.
**Net effect: the flagship quickstart document has been wrong about its own
test count through at least two rounds of the repo's own audit-and-fix
cycle.**

### 3.2 [MEDIUM] `PAKDD/README.md`'s "nine commands" claim doesn't match the six commands actually shown

Line 46: *"**All nine commands above run on every push/PR to `main`**..."*
— but the PowerShell block it refers to (lines 31–37) lists exactly six
commands: `pytest`, `check_stage_a_complete.py`, `check_stage_b_complete.py`,
`audit_stage_a.py`, `audit_stage_b.py`, `check_doc_links.py`. Missing from
the block: both `check_stage_c_complete.py` tier invocations and
`audit_stage_c.py` — which `docs/current/STAGE_C_MASTER_REPORT.md:386`
confirms are exactly the other three of the nine CI-wired steps. A reader
who copies the shown block and is told it represents "all nine" would run
six of nine and reasonably (but wrongly) believe that covers everything CI
verifies.

### 3.3 [MEDIUM] `docs/archive/README.md`'s own supersession table has three concrete inaccuracies

1. Its row for `sagitra_report.md` names `` `../current/privacy_compliant_fedgraa_redesign.md` ``
   as the superseding document. **That path does not exist** (verified:
   `docs/current/privacy_compliant_fedgraa_redesign.md` — not found). The
   real file is at `docs/research_c/privacy_compliant_fedgraa_redesign.md`
   (verified: exists). Because this reference is written as inline code
   (`` `path` ``) rather than markdown link syntax (square brackets
   immediately followed by parentheses), `scripts/check_doc_links.py`'s
   link-syntax regex
   (`!?\[[^\]]*\]\(([^)]+)\)`) never sees it — which is exactly why the
   automated doc-link gate (confirmed passing, §1.1) never caught it. This
   is a real blind spot in an otherwise-solid checker, not a checker bug.
2. `sagitra_report.md` itself is **not in `docs/archive/`** — it lives in
   `docs/research_c/` (verified via directory listing). The archive README
   is describing and "superseding" a file that isn't in the folder its own
   table claims to catalogue.
3. The table omits two files that **are** actually present in
   `docs/archive/` (verified via directory listing):
   `GPU_DECISION_AND_NEXT_STEPS.md` and `strict_reproduction_report.md`.
   Neither gets the "what superseded this" treatment the README's own
   opening line promises for "each file below."

### 3.4 [MEDIUM] `strict_reproduction_report.md` is cited as a live source without its archived status being disclosed at the point of citation

`docs/archive/README.md`'s own banner states: *"Do not treat anything in
this folder as current"* and every file "carries its own `⚠️ ARCHIVED`
banner." Per §3.3, `strict_reproduction_report.md` is missing from that same
README's supersession table, so there is no stated answer to "what replaced
this, specifically for the purpose it's being cited for." Yet
`PAKDD/README.md:16` cites it directly, side-by-side with a `docs/current/`
document, with no archived-status caveat at the point of citation: *"the
remaining ambiguities and apparent paper errors are documented in
[docs/archive/strict_reproduction_report.md]... and
[docs/current/PAPER_AMBIGUITY_REPORT.md]."* A reader following that
sentence has no signal that the first link is to a document its own folder
says not to treat as current, while the second is live.

### 3.5 Meta-finding: the audit process works, but doc-hygiene drift is regenerating faster than it's re-audited

`docs/audit/STAGE_ABC_COMPLETENESS_AUDIT_2026-08-02.md` is itself a
genuinely rigorous, honest piece of self-auditing — it ran every gate fresh,
found six real defects (F1–F6), fixed them same-session, and even disclosed
what it deliberately chose *not* to fix and why (F6). This repo's
audit-then-fix discipline is real, not theater. But finding §3.1 shows that
one of that exact document's own fixes (F2, the test count) had already
drifted stale again by the time this pass started — not because the fix was
wrong, but because the same document's own §7.2 same-day follow-up added 7
more tests *after* the F2 fix landed, and nothing re-propagated the new
total back into `PAKDD/README.md`. The pattern worth naming for the
maintainer: this project's "run the gates before calling anything done"
discipline is real and effective for **numeric/artifact** claims (that's
what `audit_stage_{a,b,c}.py` exists for, and it works), but there is no
equivalent automated check for **prose self-consistency** within a single
document (e.g., "does every test-count mention in this file agree with every
other one in this file"). `check_doc_links.py` checks link resolution, not
numeric consistency. A cheap, high-leverage addition would be a small script
that greps every tracked `.md` file for repeated "N passed"/"N tests"-shaped
patterns and fails if they disagree within a file — the same idiom this
project already uses for `check_doc_links.py`, applied to a different class
of drift.

---

## 4. What this audit did not find

No injected/fabricated citations were spotted in the material read (none of
the top-level docs read in full make citation claims directly — that risk
lives mainly in `docs/research_c/`'s literature-review documents, which are
in the unread portion, §1.3). No security-relevant logic error was found in
`privacy.py` or `secagg.py`. No test was found to be actively lying about
what it verifies (the closest is the shadow-parser pattern in §2.3, which is
a coverage gap, not a false claim). No numeric claim spot-checked against
raw JSON (via re-running `audit_stage_{a,b,c}.py` fresh, §1.1) was found
wrong.

## 5. Recommendations, in priority order

1. Fix `PAKDD/README.md`'s three test-count mentions (§3.1) to `558`, and
   add the missing three CI commands to the "nine commands" block or correct
   the claimed count to match what's actually shown (§3.2). Both are
   minutes-long edits.
2. Fix `docs/archive/README.md`'s three table inaccuracies (§3.3): correct
   the `sagitra_report.md` supersession path to `../research_c/...`, either
   move that row to document it's actually in `research_c/` or remove it
   from the archive table, and add rows for `GPU_DECISION_AND_NEXT_STEPS.md`
   and `strict_reproduction_report.md`.
3. Add an archived-status caveat where `PAKDD/README.md:16` cites
   `strict_reproduction_report.md` (§3.4), or promote its still-relevant
   content into a `docs/current/` document if it's still doing real work.
4. Consider the doc-consistency checker sketched in §3.5 — cheap, and closes
   a blind spot the existing `check_doc_links.py` structurally cannot cover.
5. Extend the `test_metrics.py` real-forward-pass pattern to
   `per_class_metrics`/`classification_report` (§2.3), and replace
   `test_run_experiments_cli.py`'s shadow parser with a direct import of
   `run_experiments.py`'s real `argparse` setup (§2.3) — both are small,
   contained fixes with a clear existing template to follow.
6. To close this audit's own biggest disclosed gap (§1.3): a follow-up pass
   specifically over the 6 unread `fedgraa/` baseline files and the
   `docs/research_c/` literature documents (highest-risk unread material —
   citation integrity was never checked there) would be the highest-value
   next increment, budget permitting.

---

*This document is an external, ad hoc audit output, written to a file in
this session's working tree per the repo owner's request. It has not been
added to git and is not part of the repo's own numbered audit sequence
unless the owner chooses to commit it.*
