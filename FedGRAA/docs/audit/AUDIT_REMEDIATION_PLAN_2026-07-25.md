# Audit Remediation Plan — Closing Every Open Finding from the Stage A/B Audits

**Compiled:** 2026-07-25 · **Author:** remediation pass over [`../audit/STAGE_AB_VERIFICATION_AUDIT_2026-07-25.md`](../audit/STAGE_AB_VERIFICATION_AUDIT_2026-07-25.md) (F1–F9) and [`../audit/STAGE_B_CORRECTNESS_AUDIT_2026-07-25.md`](../audit/STAGE_B_CORRECTNESS_AUDIT_2026-07-25.md) (A1–A6)
**Tree this plan was verified against:** `HEAD = a6c4f16` + working tree, 2026-07-25
**Status of this document: CLOSED, 2026-07-26.** All twelve items (R1–R12) below were closed by dedicated, self-labeled commits merged to `main` by `0720010` — see the `Status` column for the exact commit per item. This update was itself prompted by [`../audit/INDEPENDENT_FULL_AUDIT_2026-07-26.md`](INDEPENDENT_FULL_AUDIT_2026-07-26.md) §4.1, which found that this document's status column had never been updated even though the underlying work was verifiably complete — the exact failure mode this document's own header warns against, occurring inside the artifact built to catch it. Retained (not deleted) as the historical record of the remediation; do not add new open items here — file new findings in a new dated audit doc instead.

---

## 0. Read this first: the audits are partly out of date, and that matters

The broad verification audit (F1–F9) was compiled against **`HEAD = 4459f53`**. Two commits have landed since:

| Commit | What it did |
|---|---|
| `2feb18c` | Fixed the heavy-tailed blowup reporting (A1), stale seed-0 figures (A2), run-count overstatement (A3); added [`scripts/audit_stage_b.py`](../../scripts/audit_stage_b.py); gitignored `results/ablation/_logs/` |
| `a6c4f16` | Added the full dual-metric significance family (A4 / F2); added the Stage B correctness audit doc; rewrote theory §5.1–5.3 and `STAGE_B_MASTER_REPORT` §21 |

**Consequence: reading F1–F9 literally would cause rework on items already fixed.** Before planning, every finding was re-verified against the current tree. The result:

- **A1–A6: all 6 closed and guarded.** No action.
- **F2 (all-class accuracy): substantively closed**, with one residual gate gap.
- **F1 (uncommitted work): roughly half closed** — the code/table/theory corrections are committed; the *documents* are not.
- **F3 (provenance): partly closed** — fully disclosed in `STAGE_B_MASTER_REPORT` §21, still unaddressed in the flagship theory doc.
- **F4–F9: all fully open.** Every one re-confirmed by direct inspection this pass.

Independent re-verification run for this plan, all green:

```
.venv313/Scripts/python -m pytest tests            -> 142 passed (120.4 s)
.venv313/Scripts/python scripts/check_stage_a_complete.py -> exit 0
.venv313/Scripts/python scripts/check_stage_b_complete.py -> exit 0
.venv313/Scripts/python scripts/audit_stage_b.py          -> exit 0
```

> **Environment note, worth recording because it cost time:** the project's dependencies live in **`PAKDD/.venv313/`**. Neither `python` on `PATH` (`~/miniconda3`) nor the WindowsApps shim has `numpy`, so every gate fails with `ModuleNotFoundError` under the default interpreter. This is not documented in `README.md` and should be — see **R11**.

**The one-line summary of what is actually wrong:** the science is verified correct and the code corrections are committed; **the documentation layer is what is broken** — 6 doc paths (~2,900 lines, including Stage A's 976-line capstone) exist in no commit, the designated entry-point doc is two stages stale, and 34 links do not resolve.

---

## 1. Live defect register

Re-verified against the working tree on 2026-07-25. `Source` maps back to the originating audit finding.

| ID | Source | Defect (as it exists *now*) | Sev | Status |
|---|---|---|---|---|
| **R1** | F1 | 6 doc paths untracked (~2,900 lines) incl. Stage A's capstone; doc-move deletions unstaged-for-commit | **High** | **Closed** — `1cddd2c`, `155828c`, `a190f93` |
| **R2** | F3 | Theory doc title asserts "**Published**"; §7 has no provenance bullet. `STAGE_B_MASTER_REPORT` §21 discloses it fully — the flagship doc does not | **High** | **Closed** — `4957561`, `097de78` |
| **R3** | F2 | `check_stage_b_complete.py` never validates the `comparisons_all_acc` family it now depends on | Medium | **Closed** — `eb23950` |
| **R4** | F4 | `publication_roadmap.md` — the designated entry point — lists 6 completed items as "missing" and says "Execute Stage A" | Medium | **Closed** — `6ed6fc1` |
| **R5** | F5 | `STAGE_DIRECTION_ASSESSMENT.md` and `RESEARCH_DIRECTION_AUDIT_2026-07-25.md` carry no superseded banner; the latter's action #5 instructs discarding a finished campaign | Medium | **Closed** — `6ed6fc1` |
| **R6** | F6 | **34 broken relative links** / 398 checked, across 9 files | Medium | **Closed** — `ed8ad07` |
| **R7** | F7 | `PAKDD_STATUS_REPORT.md` stale (79/79 tests, wrong headline fix, 2 broken links, no banner) | Low | **Closed** — `d504b07` |
| **R8** | F8 | "byte-identical" false at `results/README.md:53` and `STAGE_A_MASTER_REPORT.md:624` | Low | **Closed** — `5c65435` |
| **R9** | F9 | Stage A capstone frames severity on seed-0 (`193×`/`7.6×`) with no forward-reference to the 10-seed correction | Low | **Closed** — `5c65435` |
| **R10** | new | No doc-link gate exists; R6 is reintroducible on the next doc move | Medium | **Closed** — `c383e7b` |
| **R11** | new | `.venv313` interpreter requirement undocumented; every gate fails under default `python` | Low | **Closed** — `0790726` |
| **R12** | B-audit §5 | Stage A has **no** automated transcription/correctness gate — `audit_stage_b.py`'s method is unextended to Stage A | Medium | **Closed** — `1ab0fbc` |

All twelve re-verified independently on 2026-07-26 by reading the diffs at the cited commits and re-running the definition-of-done commands in §4 on `HEAD = 0720010` — `pytest` (151 passed, count updated post-`c383e7b`'s `test_check_doc_links.py`), all six gates (`pytest`, both completeness gates, both correctness audits, the doc-link checker) exit 0. See [`INDEPENDENT_FULL_AUDIT_2026-07-26.md`](INDEPENDENT_FULL_AUDIT_2026-07-26.md) for the fresh findings raised independently of this plan.

**Closed, no action** (recorded so a future reader does not re-open them): A1 heavy-tailed mean · A2 stale seed-0 figures · A3 run-count · A4/F2 dual-metric significance · A5 audit crash-on-report · A6 star markup · F1's `_logs/` gitignore decision · F1's code/table/theory commits · F6's `README.md` entries.

---

## 2. Remediation plan

Seven workstreams, ordered by dependency. **W1 must go first** — every doc fix below it edits files that are currently untracked, so doing them in the other order produces a diff no reviewer can read.

---

### W1 — Commit the documentation layer (R1)

**Why first.** `README.md:23` cites `docs/current/STAGE_A_MASTER_REPORT.md` as the authoritative Stage A deliverable. That file — 976 lines, Stage A's entire capstone — exists in **no commit**. `stage_a_plan.md` declares Stage A complete "against commit `be4ef03`"; the document making that case is untracked. Until this lands, a clean checkout cannot substantiate the project's own headline claim.

Untracked paths and their weights:

| Path | Lines | Note |
|---|---|---|
| `docs/current/STAGE_A_MASTER_REPORT.md` | 976 | Stage A capstone — cited by README, 3 audits, the theory doc |
| `docs/current/STAGE_DIRECTION_ASSESSMENT.md` | — | needs W3's banner; commit first, banner second |
| `docs/audit/RESEARCH_DIRECTION_AUDIT_2026-07-25.md` | 178 | needs W3's banner |
| `docs/audit/STAGE_AB_VERIFICATION_AUDIT_2026-07-25.md` | 362 | one of the two source audits |
| `docs/paper/FedGRAA_Simulated_Peer_Review.md` | 520 | 15 of the 34 broken links live here |
| `docs/research_c/` (8 docs) | 3,818 | destination of the `docs/current/` move |
| `archive/FedGRAA_reproduction.ipynb` | — | notebook move; confirm intent before committing |

Plus staged deletions for 5 docs moved `docs/current/` → `docs/research_c/`, and top-level `docs/` deletions.

**Tasks**

1. Confirm the `docs/current/` → `docs/research_c/` move is intentional and final. Stage it as a **rename** (`git add -A` detects it) so history follows the content.
2. Confirm `archive/FedGRAA_reproduction.ipynb` and `PAKDD/archive/` are an intended archival move, not stray files.
3. Commit in **three** logical commits, not one:
   - **(a) doc reorganization** — the `docs/current/` → `docs/research_c/` renames + top-level `docs/` deletions + `README.md` update. Pure moves, reviewable at a glance.
   - **(b) Stage A capstone + strategy docs** — `STAGE_A_MASTER_REPORT.md`, `STAGE_DIRECTION_ASSESSMENT.md`, `FedGRAA_Simulated_Peer_Review.md`.
   - **(c) audit docs** — `RESEARCH_DIRECTION_AUDIT_2026-07-25.md`, `STAGE_AB_VERIFICATION_AUDIT_2026-07-25.md`, and this plan.
4. Do **not** fold the W2–W6 content fixes into these commits. Move first, then edit — otherwise the rename detection breaks and the diff becomes unreviewable.

**Acceptance**
- `git status --porcelain` reports clean (or only deliberately-ignored paths).
- `git ls-files docs/current/STAGE_A_MASTER_REPORT.md` returns the path.
- `git stash -u && <all four gates> && git stash pop` — all four still green on a clean tree.

---

### W2 — Close the provenance claim in the flagship doc (R2)

**State.** `STAGE_B_MASTER_REPORT.md` §21 is a genuinely excellent treatment — it lays out both branches, refuses to assert non-publication on two zero-result searches, and names the word "published" as the one unbacked claim in the package. **That honesty has not propagated to the document that actually makes the claim.**

- [`fedgraa_convergence_defect.md:1`](../current/fedgraa_convergence_defect.md) — *"Why a **Published** FL-IDS Aggregation Algorithm Doesn't Converge, and How to Fix It"*
- Its §7 "Honest limitations" — 8 bullets, **none** on provenance.

A title is the most-quoted line in any document. An unverified factual assertion should not live there.

**Tasks**

1. **Retitle** to drop the unbacked word while keeping the claim's force. Recommended: *"Why an FL-IDS Aggregation Algorithm Doesn't Converge, and How to Fix It"* — the critique does not depend on publication status.
2. **Add a §7 bullet** cross-referencing `STAGE_B_MASTER_REPORT` §21 rather than restating it, e.g.:
   > **The target manuscript's publication status is unresolved.** Two zero-result searches are a strong negative, not a confirmed one; the escalation path this project's own literature docs mandate (DBLP, Semantic Scholar) has not been run. Nothing in §1–§6 depends on the answer — the algorithm's behaviour on real data is what it is. What does depend on it is whether this section can stand as a standalone citable critique. See [`STAGE_B_MASTER_REPORT.md` §21](../current/STAGE_B_MASTER_REPORT.md#21-the-open-framing-risk--provenance-of-the-target-manuscript).
3. **Run the mandated escalation** — DBLP and Semantic Scholar title/acronym searches. Cheap, and it converts a strong negative into a confirmed one either way. Record the result (including a null result, with date and query strings) in §21.
4. **Raise it with the mentor.** This is the single highest-leverage open question in the project and it resolves in one conversation. Fold in the three companion questions from `RESEARCH_DIRECTION_AUDIT` §5: the Stage C angle decision, the compute question, and authorship.

**Acceptance**
- No unqualified "published" in the theory doc's title or abstract.
- §7 carries the provenance bullet, linked to §21.
- §21 records the DBLP/Semantic Scholar queries and their date.

> **Sequencing note.** W2.1–W2.3 are unblocked and should be done now. W2.4 is a human conversation and gates only *framing*, not content — do not hold W3–W7 behind it.

---

### W3 — Refresh the stale strategy layer (R4, R5, R7)

These three are one workstream because they share a failure mode: **documents that describe a project state that no longer exists, with no marking to say so.** This layer is what a mentor reads first, and it currently undersells the work by two complete stages.

#### W3a — `publication_roadmap.md` (R4)

Confirmed stale at these lines:

| Line | Says | Actual |
|---|---|---|
| 11 | "79/79 tests passing" | **142** |
| 25 | "only the CIC-IDS-2017 γ-sweep (n=3) is persisted"; n=5/7 and UNSW "not yet run" | all four base sweeps persisted; Stage A gate green |
| 25 | "combined fix has only been seed-swept on CIC" | UNSW band exists |
| 27 | "No formal statistical testing (only 3-seed variance bands)" | 10-seed bootstrap CIs + paired *t* + Wilcoxon + Holm, **dual-metric** |
| 28 | "No ablation table" | full 2×2 factorial, both datasets, both metrics |
| 29 | "No corrected convergence theory" | `fedgraa_convergence_defect.md` §4 |
| 120 | *"Execute Stage A in this order"* + six Stage A steps | Stage A complete |

**Task.** Rewrite §1 and §6 against reality. §6 becomes "Stage C is the frontier; it is gated on the provenance/angle conversation (W2.4)." Keep the router structure — the document's value is that it is one entry point, and that is worth preserving.

> This exact fix was already flagged by `STAGE_DIRECTION_ASSESSMENT.md` §3 with a "~20–30 min mechanical" estimate, and was not done. It is the highest ratio of reader-impact to effort in this entire plan.

#### W3b — Banner the two overtaken strategy docs (R5)

Verified: **neither** `STAGE_DIRECTION_ASSESSMENT.md` nor `RESEARCH_DIRECTION_AUDIT_2026-07-25.md` carries a superseded banner. (An earlier grep suggested otherwise; it was matching the word "superseded" in body prose.)

Both were overtaken **within ninety minutes of being written** — they were compiled at 01:00 and 01:04, and the Stage B campaign completed at 03:01. The consequence is concrete: `RESEARCH_DIRECTION_AUDIT` §5 action 5 says

> *"Do not run 71 fresh (arm, seed) cells… most likely 4 arms × 2 datasets × 3 γ at **5 seeds**, not 10 — then run exactly that."*

Acting on that today means **discarding a completed 10-seed campaign to rebuild it smaller.**

**Task.** Add a dated **partially-superseded** banner to each, in the style `STAGE_B_REMAINING_TASKS.md` already uses correctly. The banner must be surgical — say precisely what is dead and what is live:

> **⚠️ Partially superseded, 2026-07-25.** Written at 01:04, before the Stage B campaign completed at 03:01 (`4459f53`). **Superseded:** every statement that the Stage B campaign is unrun, and §5 action 5 (cut to 5 seeds) — the 10-seed campaign is complete, verified, and committed. **Still live:** findings D1 (provenance), D2, D3, D4, D6, and the Stage C sequencing argument. See [`AUDIT_REMEDIATION_PLAN_2026-07-25.md`](AUDIT_REMEDIATION_PLAN_2026-07-25.md).

The precision matters: a blanket "superseded" would discard D1, which is still the project's most important open question.

#### W3c — `PAKDD_STATUS_REPORT.md` (R7)

At the **PAKDD root**, not in `docs/`. Dated 2026-07-23. Confirmed defects: 79/79 tests (line 39, 72); presents `η′=0.001` as the headline intervention (line 20) — the over-correction the project has since shown starves non-rare clients — rather than the validated combined fix; two broken links to `docs/fedgraa_divergence_diagnosis.md` (lines 11, 56); describes the divergence diagnosis as the frontier.

It is linked by `README.md` as a current-state pointer, and `latest_baseline.md`'s superseded banner points readers *to* it as authoritative. **A stale doc is currently the designated destination for readers leaving a correctly-bannered one.**

**Task.** Choose one and do it fully:
- **(a) Banner + redirect** *(recommended)* — mark superseded, point to `STAGE_A_MASTER_REPORT.md` / `STAGE_B_MASTER_REPORT.md`, fix the 2 links. ~15 min.
- **(b) Full refresh** to current state. Higher cost, and it duplicates the two capstones — which is how it went stale in the first place.

Then repoint `latest_baseline.md`'s banner at the capstones directly.

**Acceptance (W3)**
- No document in the tree claims Stage A or Stage B is outstanding.
- `grep -rn "79/79" --include=*.md .` returns only historical/archival contexts.
- Both strategy docs carry dated, scoped banners naming what remains live.

---

### W4 — Fix all 34 broken links and make the class impossible (R6, R10)

Independently re-measured this pass: **398 relative links checked, 34 broken** across 9 files. (The audit reported 37/347; the delta is the reorganization progressing plus a percent-decoding difference in the checker — the same underlying defects.)

| Root cause | Count | Files |
|---|---|---|
| `docs/paper/FedGRAA_Simulated_Peer_Review.md` links to `docs/current/` docs as siblings | **15** | that file alone |
| `docs/research_c/` docs' outbound links now resolve inside `research_c/` | **11** | `STAGE_C_LITERATURE_MASTER.md` (7), `STAGE_C_DEEP_RESEARCH_2026-07-25.md` (3), `privacy_compliant_fedgraa_redesign.md` (1) |
| `PAKDD_Final_Version_Submit (3).pdf` — unescaped `)` truncates the target at the first paren | **5** | `REPRODUCTION_AUDIT.md`, `PAPER_AMBIGUITY_REPORT.md`, `FedGRAA_Simulated_Peer_Review.md`, `FedGRAA_Deep_Research_Compendium.md`, `STAGE_C_LITERATURE_MASTER.md` |
| `PAKDD_STATUS_REPORT.md` → pre-reorganization `docs/` paths | 2 | (also W3c) |
| `scripts/archive/gpu/README.md` → pre-reorganization paths | 2 | archived, low priority — fix anyway |

**Tasks**

1. **Rename the PDF** to `PAKDD_Final_Version_Submit_v3.pdf` and update all references. This is the permanent solve — escaping to `%20%283%29` fixes rendering but leaves a filename that will keep breaking every future link, shell command, and CI path. Rename the sibling `.md` identically.
2. **Repoint the 26 reorganization-fallout links** to correct relative paths.
3. **Fix the 2 archived-script links.**
4. **Build `scripts/check_doc_links.py`** *(new file)* — ~40 lines: walk every `.md` outside `.venv313`/`__pycache__`, regex-match markdown link and image targets, skip `http(s):`/`mailto:`/anchor-only, percent-decode, resolve against the file's parent, report non-existent targets with file and target, exit 1 on any.
5. **Wire it into both completeness gates** so a doc move can never again silently break the tree. Add a unit test with a deliberately-broken fixture link — negative-control it, per the Stage B audit's own §2.1 lesson that an unexercised failure path is where the bugs live.

**Acceptance**
- `python scripts/check_doc_links.py` → exit 0, "398/398 resolve".
- Both `check_stage_*_complete.py` invoke it and fail when it fails.
- `pytest tests` includes a test that the checker *detects* an injected broken link.

---

### W5 — Close the two gate gaps (R3, R12)

The Stage B correctness audit's own §6 lesson: *"when a result is summarized, the summary is a claim and needs its own verification."* Two places where that is not yet enforced.

#### W5a — Gate the dual-metric family (R3)

`build_ablation_table.py` now emits `comparisons_all_acc`, `render_ablation.py` consumes it, and `audit_stage_b.py` check F enforces disclosure. But `check_stage_b_complete.py` — the *completeness* gate — validates only `{"rare_acc", "all_acc", "blowup"}` per cell (line 38) and has **zero** references to the `comparisons_all_acc` block.

So the fix for the audits' highest-severity scientific finding (F2/A4) is guarded by the correctness gate but invisible to the completeness gate. A builder regression that silently dropped `comparisons_all_acc` would leave `check_stage_b_complete.py` green.

**Task.** Extend it to require `comparisons_all_acc` with all 5 comparisons per γ per dataset, each carrying its `p_holm`, `d`, and divergence flag — mirroring exactly what it already demands of the rare-acc family.

#### W5b — Extend the correctness-audit method to Stage A (R12)

Flagged by the Stage B audit itself as "the obvious next hardening step." Stage A's ~90 reported values were verified **by hand, twice** — and by nothing re-runnable. `check_stage_a_complete.py` is a completeness gate: it checks presence, shape, and renders figures. It is structurally blind to exactly the defect class that produced A1, A2, and A3.

This is not hypothetical for Stage A: **R8 and R9 are both live instances of it** — a false "byte-identical" claim and stale seed-0 figures, sitting in Stage A's capstone right now, passing a green gate.

**Task.** Build `scripts/audit_stage_a.py` on `audit_stage_b.py`'s proven pattern:
- **Check A** — SHA-256 + config-vs-filename on all base-sweep JSONs.
- **Check D** — parse every number transcribed into `STAGE_A_MASTER_REPORT.md` §5–§8 and `results/README.md`, re-derive from `results/*.json`. This is the high-value check: it is what would have caught R9 automatically.
- **Check E** — flag any Stage A summary statistic whose mean/median diverge >3× (the A1 pattern).
- **Round-0 convention check** — `all_acc[0] == last_n_acc[0] == 0.0` across all 30 curves.
- **Negative-control it before trusting it** — inject a changed number, a corrupted JSON value, and a duplicated run file; confirm all three are detected *and that the failure path prints readably*. A5 was a crash in the failure path, found only because someone tried to make the tool fail.

**Acceptance**
- `python scripts/audit_stage_a.py` → exit 0, reporting the count of values re-derived.
- Negative-control results recorded in the script's docstring or a companion doc.
- `check_stage_b_complete.py` fails on a table with `comparisons_all_acc` removed.

---

### W6 — Cross-document consistency (R8, R9)

Both are small, both are in Stage A's capstone, and both are the kind of thing a reviewer notices.

#### W6a — "byte-identical" (R8)

At [`results/README.md:53`](../../results/README.md) and [`STAGE_A_MASTER_REPORT.md:624`](../current/STAGE_A_MASTER_REPORT.md).

The **scientific claim is correct and important** — every metric curve is exactly identical, all 5 methods, max elementwise difference exactly `0`, because `rare_client_ids(7)` truncates to the same 5 classes `rare_client_ids(5)` returns. Only the word is over-strong: the JSON objects differ in `history.n_rare` (`5` vs `7`), which is correct metadata behaviour.

**Task.** Replace with wording that is both true and *more* informative:

> **metrically identical** — every curve exact to the bit; only the recorded `n_rare` label differs, which is precisely what confirms the truncation happened at the intended layer.

Also review `results/README.md:56` and `STAGE_A_MASTER_REPORT.md:820`, which use "byte-identical" for the combined-fix regression check — a different comparison; verify it before rewording.

#### W6b — Stage A → Stage B forward reference (R9)

`STAGE_A_MASTER_REPORT.md` frames severity as *"CIC's worst was 193×; UNSW's worst 7.6×… an order of magnitude smaller"* at lines **34, 268, 385, 409, 447**. Those are correct **seed-0** figures and Stage A is honestly scoped to 3 seeds — this is not an error in context.

But `fedgraa_convergence_defect.md` §6 now reports the 10-seed reading: worst case **1.86×10⁶×** (CIC γ=0) vs **46.85×** (UNSW γ=1) — four orders apart on the tail — while the **medians are much closer** (CIC 2.71–14.19× vs UNSW 1.50–8.26×), and at γ=0 the CIC median (3.70×) is actually **lower** than UNSW's (5.10×).

A reader going A → B hits two different headline framings of the same phenomenon, and the capstones are explicitly meant to be read together.

**Task.** Add one forward-reference line at §5.8 and the executive summary (line 34):

> *Seed-0 figures, scoped to Stage A's 3-seed regime. Superseded at N=10 by [`fedgraa_convergence_defect.md` §6](../current/fedgraa_convergence_defect.md): the tail is far worse (1.86×10⁶× vs 46.85×) while the medians are close, and at γ=0 CIC's median is lower than UNSW's.*

Do **not** rewrite Stage A's numbers — they are correct for their stated regime, and rewriting them would destroy the honest 3-seed→10-seed provenance trail.

---

### W7 — Document the environment (R11)

Every gate and the full test suite fail under the default `python` on this machine with `ModuleNotFoundError: No module named 'numpy'`. The working interpreter is `PAKDD/.venv313/Scripts/python.exe`, and no document says so.

**Task.** Add a "Running the gates" block to `README.md` recording the interpreter explicitly:

```bash
# All commands below assume the project venv:
PY=.venv313/Scripts/python        # Windows: .venv313\Scripts\python.exe

$PY -m pytest tests                       # 142 passed
$PY scripts/check_stage_a_complete.py     # Stage A completeness
$PY scripts/check_stage_b_complete.py     # Stage B completeness
$PY scripts/audit_stage_b.py              # Stage B correctness
$PY scripts/check_doc_links.py            # doc-link integrity  (W4)
$PY scripts/audit_stage_a.py              # Stage A correctness (W5b)
```

Note the expected test count so a future drift is visible, and state that `requirements.txt` is what `.venv313` was built from.

---

## 3. Execution order

W1 gates everything. W2.1–W2.3, W3, W4, W6, W7 are then independent and parallelizable. W5 is the largest single item.

| Order | Workstream | Items | Est. | Blocks |
|---|---|---|---|---|
| **1** | **W1** Commit the doc layer | R1 | 30 min | everything |
| **2** | **W3a** Refresh the roadmap | R4 | 30 min | — |
| **3** | **W3b** Banner the strategy docs | R5 | 20 min | — |
| **4** | **W4.1–3** Fix the 34 links + rename PDF | R6 | 45 min | — |
| **5** | **W2.1–2.3** Retitle + provenance bullet + DBLP/S2 | R2 | 30 min | — |
| **6** | **W6** byte-identical + forward-reference | R8, R9 | 20 min | — |
| **7** | **W3c** `PAKDD_STATUS_REPORT.md` | R7 | 15 min | — |
| **8** | **W7** Document the venv | R11 | 10 min | — |
| **9** | **W4.4–5** Build + wire the link gate | R10 | 45 min | W4.1–3 |
| **10** | **W5a** Gate the dual-metric family | R3 | 30 min | — |
| **11** | **W5b** Build `audit_stage_a.py` | R12 | 2–3 h | W6 |
| **—** | **W2.4** Mentor conversation | R2 | 1 conv. | *(async — do not block on it)* |

**≈ 4 hours of mechanical work plus one conversation**, of which items 1–8 (≈3 h) close every reader-facing defect. Items 9–11 are the permanent guards that stop the whole class from recurring.

W5b is deliberately last: it should be written **after** W6 fixes R8/R9, so its first run on a corrected tree is a genuine green rather than a green achieved by writing the checker around known-bad values.

---

## 4. Definition of done

The remediation is complete when **all** of the following hold on a clean checkout of `HEAD`:

```bash
PY=.venv313/Scripts/python
git status --porcelain                    # -> empty
$PY -m pytest tests                       # -> 151 passed (was 142+; grew with test_check_doc_links.py)
$PY scripts/check_stage_a_complete.py     # -> exit 0
$PY scripts/check_stage_b_complete.py     # -> exit 0  (now incl. comparisons_all_acc)
$PY scripts/audit_stage_b.py              # -> exit 0
$PY scripts/audit_stage_a.py              # -> exit 0  (new)
$PY scripts/check_doc_links.py            # -> exit 0, all relative links resolve
```

And by inspection — **all nine re-verified 2026-07-26, all hold:**

- [x] Every doc cited as authoritative is **committed** — above all `STAGE_A_MASTER_REPORT.md`
- [x] No document claims Stage A or Stage B is outstanding
- [x] No unqualified "published" claim about the target manuscript anywhere in the tree
- [x] Both overtaken strategy docs carry dated banners that name what is still live
- [x] Zero broken relative links, enforced by a gate, with a negative-control test (`tests/test_check_doc_links.py`)
- [x] "byte-identical" replaced with "metrically identical" where the strong word is false
- [x] Stage A's seed-0 severity framing forward-references the 10-seed correction
- [x] `README.md` documents the `.venv313` interpreter and expected test count
- [x] Every new checker has been **negative-controlled** — deliberately broken, and confirmed to fail *readably*

**Not covered by this checklist** (out of this plan's original scope, raised fresh by `INDEPENDENT_FULL_AUDIT_2026-07-26.md`): the gates above are still not enforced anywhere until this pass's CI wiring (`.github/workflows/docs-check.yml` `stage-gates` job); several real code bugs in `fedgraa/aggregation.py`/`metrics.py`; and the paper draft's own numeric claims. Those are tracked in the new audit doc, not here.

---

## 5. Out of scope, and the risks that stay open

Named explicitly so a green result at the end of this plan is not over-read.

- **The provenance answer itself (W2.4).** This plan makes the *disclosure* honest everywhere. It cannot make the *question* resolved — that needs the mentor. It remains the highest-leverage open item in the project, and every downstream decision about how Stage B is framed, titled, and submitted branches on it.
- **The corrected convergence proposition as mathematics.** Its empirical predictions are verified (feasibility 60/60, defect 60/60) and `project_to_simplex` is validated to float32 against SLSQP over 300 vectors. The `O(1/T)` gradient-mapping rate and the Clarke-subdifferential treatment of `min(L_h, L_m)` are **not** independently re-derived. §4's own rigor flag scopes these for the mentor, correctly — that flag must survive all editing in W2.
- **The Stage C literature corpus.** Its own re-audit discloses that **~140 carried-over bibliography entries were never independently re-verified.** That is a live exposure, out of Stage A/B scope, and it should get its own plan before Stage C's Related Work is written on top of it.
- **Statistical power.** Everything here confirms the tests were applied correctly. Nothing here makes n=10 a larger sample. The Wilcoxon floor at `0.001953` = 2/2¹⁰ is a genuine resolution limit, correctly disclosed in the theory doc's §7.
- **Re-running any training.** All values are verified against persisted artifacts. Determinism across a re-run is asserted by pinned seeds, not tested.

---

## 6. The lesson worth keeping

The two audits agree on a diagnosis that this plan is shaped around: **the science is right and the reporting layer is where the defects live.** Zero numerical errors were found across roughly 400 independent checks in one audit and a full independent recomputation in the other. Every single defect — all 6 in Stage B's correctness audit, all 9 in the verification sweep — was in how correct data was summarized, described, linked, or committed.

That layer has historically been guarded by nothing. The completeness gates check presence and shape; they are structurally blind to a misleading mean, a stale figure, a broken link, or an uncommitted capstone. `audit_stage_b.py` closed that gap for Stage B's numbers, and it immediately earned its keep — check **D** caught a malformed significance marker within minutes of being pointed at freshly-written prose, and exposed a bug in its own regex doing it.

**W4 and W5 exist to finish that job**: a link gate, a dual-metric completeness check, and a Stage A correctness gate. Once those are in, every defect class either audit found is machine-detectable, and this document can be closed rather than repeated.

The corollary the Stage B audit stated, and the one to carry into Stage C: **a summary is a claim, and it needs its own verification.** The per-seed array was right in the JSON the whole time, and the reported figure was still wrong by five orders of magnitude.

---

*Sources: [`STAGE_AB_VERIFICATION_AUDIT_2026-07-25.md`](../audit/STAGE_AB_VERIFICATION_AUDIT_2026-07-25.md) (F1–F9) · [`STAGE_B_CORRECTNESS_AUDIT_2026-07-25.md`](../audit/STAGE_B_CORRECTNESS_AUDIT_2026-07-25.md) (A1–A6). Every status in §1 was re-verified against the working tree at `HEAD = a6c4f16` on 2026-07-25 — `pytest` (142 passed), all three existing gates (exit 0), an independent 398-link resolution sweep, `git ls-files` tracking checks on 6 doc paths, and direct line-level inspection of all 9 open findings.*
