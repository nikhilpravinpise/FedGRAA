# PDF↔MD Fidelity Contamination — Remediation Plan

**Date:** 2026-08-13
**Status:** PLAN ONLY — nothing in this document has been executed. Companion to [`PDF_MD_FIDELITY_IMPACT_ON_STAGES_ABC_2026-08-13.md`](PDF_MD_FIDELITY_IMPACT_ON_STAGES_ABC_2026-08-13.md) (the impact assessment) and [`PDF_MD_FIDELITY_AND_CONTAMINATION_AUDIT_2026-08-13.md`](PDF_MD_FIDELITY_AND_CONTAMINATION_AUDIT_2026-08-13.md) (the original finding).
**Purpose:** a complete, exact, ready-to-execute inventory of every change this bug still requires — nothing deferred, nothing left as "consider doing this later." Every item below is either a fully-specified edit with exact before/after text, or an explicit decision point with a stated recommendation. Read it, and everything can be executed in one pass.

---

## 0. What this plan does NOT need to cover, and why

Stated up front so the scope reads as deliberate, not incomplete:

- **No code changes.** §2 of the impact assessment confirms zero pipeline consumption of the fabricated data — nothing in `fedgraa/`, `scripts/`, or `results/` needs touching.
- **No dataset changes.** `data_provenance_audit.md` already independently confirmed both datasets genuine; unrelated to this bug.
- **No changes to `STAGE_B_MASTER_REPORT.md` or `STAGE_C_MASTER_REPORT.md` content.** Both are confirmed independent of finding #9 (verified twice — once by direct read, once by an independent adversarial agent pass). They only need to keep working once §1's targets are edited (checked in §5).
- **No changes to `results/README.md` or `understanding/04_experiments_and_results.md`.** Re-checked directly this session, and re-verified independently: `understanding/04_experiments_and_results.md` (lines 49, 53) does use `"(Last n classes)"` column-header phrasing, referring to this project's own real result tables, not the paper's fabricated per-round data. `results/README.md` contains **zero** occurrences of that phrase at all (it uses different phrasing, e.g. "rare-acc") — the correct statement for that file is "doesn't cite the fabricated data" rather than "cites it but from a real source." Both are confirmed false leads either way; neither needs a fix.
- **No edit to `docs/paper/draft1.md`.** Carries the same fabricated-table signature, but (a) it's already banner-marked SUPERSEDED with an explicit "do not cite numbers from this version" and a 2026-08-03 correction note, (b) confirmed zero downstream citations anywhere in the repo. Touching a frozen historical artifact that nothing reads and that already warns readers off it would be scope creep, not remediation. Listed here as a considered-and-rejected action, not an oversight.
- **No edit to `docs/paper/PAKDD_Final_Version_Submit_v3.md` itself** (the original flawed conversion, including dormant defects #4 and #6 from the original audit). Its own banner already states it is "kept here, unedited, as the original draft the review and the correction both respond to — do not cite numbers from this version." Editing it would contradict its stated purpose as a frozen artifact. The corrected content lives separately in `docs/paper-files/` (see §2). This is a decision point, not silently skipped — see §4 for the alternative.

---

## 1. Group A — Retract the contaminated citations (3 sites)

These are the load-bearing fixes: three places in the tracked repo currently cite hallucinated data as if it were genuine paper evidence.

### 1.1 `docs/current/PAPER_AMBIGUITY_REPORT.md` — item #9 (the origin point, first committed 2026-07-10)

**Current text (lines 223–237):**
```
## 9. Figures 2–3's per-round tables show implausibly static values over long stretches — SUSPICIOUS DATA (weaker)

In Figure 1's full per-round table (γ=0/0.5/1, n=3), several columns are
**exactly** flat for 60–100+ consecutive rounds (e.g., CIC "Static (All
classes)" stays at exactly `0.44` from round 60 through round 200 in the γ=0
panel; "Static (Last n classes)" stays at exactly `0.88` for rounds 20–100).
Neural network accuracy on held-out validation data during continued training
essentially never freezes to the same 2-decimal value across dozens of
evaluation checkpoints with zero fluctuation, even after convergence,
because mini-batch stochasticity and the specific validation sample
composition should produce at least small round-to-round noise. This is a
milder version of finding 8 above (round two-decimal values, suspiciously
static plateaus) and reinforces the same underlying concern: **at least some
of the paper's reported numbers do not have the statistical texture of raw
measured training curves.**
```

**Replacement:**
```
## 9. [RETRACTED 2026-08-13] Per-round tables show implausibly static values — was based on a PDF→MD conversion hallucination

*Original finding, dated 2026-07-10, retracted 2026-08-13. The "per-round
table" this finding analyzed does not exist in the source PDF. It was
fabricated by the llamaparse PDF→MD conversion tool: ~870–936 invented
numeric cells across three HTML tables purporting to give exact per-round
accuracy values for Figures 1–3. The PDF's actual Figures 1–3 are plotted
raster line charts with only axis tick labels — no per-round numeric
values, real or otherwise, exist to freeze or not freeze. Full account:
`docs/audit/PDF_MD_FIDELITY_AND_CONTAMINATION_AUDIT_2026-08-13.md`.*

*Finding #8 above — the real, small 5-row comparison table's suspiciously
round repeating-decimal values (66.66/88.88/77.77/33.33) — is separate,
unaffected, confirmed present in the actual PDF, and stands unchanged.*
```

**Why this exact wording:** preserves the finding number and its history (visible retraction, not silent deletion — an editor reading the doc later can see what was claimed and why it was wrong, matching this repo's existing convention of dated correction notes, e.g. the banner already on `PAKDD_Final_Version_Submit_v3.md`). Explicitly protects finding #8 from guilt-by-association.

### 1.2 `docs/current/STAGE_A_MASTER_REPORT.md` — finding #9 (highest exposure; Stage B links to this document for "upstream context")

**Current text (line 319, in context of lines 301–319):**
```
**9. Per-round tables show implausibly static values over long stretches — SUSPICIOUS DATA (weaker).** E.g. CIC "Static (All classes)" stays exactly `0.44` from round 60 through round 200 in the γ=0 panel; "Static (Last n classes)" stays exactly `0.88` for rounds 20–100. Real training curves essentially never freeze to the same 2-decimal value across dozens of evaluation checkpoints.
```

**Replacement:**
```
**9. [RETRACTED 2026-08-13] Per-round tables show implausibly static values — was based on a PDF→MD conversion hallucination.** *The "per-round table" this finding analyzed does not exist in the source PDF — it was fabricated by the llamaparse conversion tool (~900 invented numeric cells across 3 tables that don't exist in the paper; Figures 1–3 are plotted line charts with no per-round numeric labels). Full account: `docs/audit/PDF_MD_FIDELITY_AND_CONTAMINATION_AUDIT_2026-08-13.md`. Finding #8 above, about the real comparison table's round numbers, is unaffected and stands.*
```

**Same-file consistency check — already performed, not just flagged:** grepped the full 93KB file for `finding.{0,4}9\b`, `implausibly static`, and `SUSPICIOUS DATA`. Only one hit besides the severity-legend definition and finding #8 (unaffected, separate) — the single entry at line 319 quoted above. **No summary table or table-of-contents elsewhere in the file references finding #9 separately.** The single-entry edit above is the complete fix for this file; no sweep needed at execution time.

### 1.3 `understanding/02_the_problem.md` — line 52 (low-stakes miscitation, correct conclusion, wrong source)

**Current text (line 52):**
```
**Consequence:** FedAvg gets near **0% accuracy on the rare attack classes** — you can see this in the result tables, where the "FedAvg (Last n classes)" column is `0.00` everywhere.
```

**Replacement:**
```
**Consequence:** FedAvg gets near **0% accuracy on the rare attack classes** — you can see this in the paper's real chart images (Figure 1): the FedAvg series sits at or near zero on the rare-class panels throughout training.
```

**Note:** this is a citation swap only, not a retraction — the underlying claim (FedAvg underperforms on rare classes) is independently true and doesn't need walking back, just re-pointing at real evidence.

---

## 2. Group B — Fix the two delivery defects in `docs/paper-files/`

The corrected replacement file (`PAKDD_Final_Version_Submit_v3_CORRECTED.md`) is content-accurate but currently unusable/unverifiable as delivered.

### 2.1 Broken image paths

**Problem:** `PAKDD_Final_Version_Submit_v3_CORRECTED.md` references `figures/Figure1.png`, `figures/Figure2.png`, `figures/Figure3.png` (confirmed at lines 386, 434, 442). The three PNGs currently sit flat in `docs/paper-files/` — no `figures/` subdirectory exists (confirmed: `ls docs/paper-files/figures/` → "No such file or directory").

**Fix:**
```
mkdir docs/paper-files/figures
mv docs/paper-files/Figure1.png docs/paper-files/Figure2.png docs/paper-files/Figure3.png docs/paper-files/figures/
```

**Test after fix:** open `PAKDD_Final_Version_Submit_v3_CORRECTED.md` in a markdown previewer (or just `ls docs/paper-files/figures/` and confirm all three files present) — all three images should render.

### 2.2 Unrunnable verification script

**Problem:** `verify_corrected_md.py` requires `PAKDD_Final_Version_Submit_v3_pdftotext_layout.txt` as ground truth for its 49-assertion suite. That file does not exist in the repo (confirmed by re-running the script directly: `FileNotFoundError: ... 'PAKDD_Final_Version_Submit_v3_pdftotext_layout.txt'`). The "49/49 pass" claim in `FedGRAA_PDF_vs_MD_Fidelity_Report_v3_FINAL.md` is therefore currently unverifiable in this repo, regardless of whether it was true in whatever environment originally produced it.

**Two fix options — pick one:**

- **Option 1 (regenerate):** run `pdftotext -layout docs/paper/PAKDD_Final_Version_Submit_v3.pdf docs/paper-files/PAKDD_Final_Version_Submit_v3_pdftotext_layout.txt`, then re-run `python verify_corrected_md.py` and confirm 49/49. **Status re-checked this session: `pdftotext` (poppler) is now available on this machine** (`/mingw64/bin/pdftotext`) — the original audit's "no local install" caveat no longer applies to `pdftotext` itself. Note `pdftoppm` (needed for page-image rendering, not for this text extraction) is still absent, but that's irrelevant to this specific script's dependency.
- **Option 2 (drop the dependency):** edit `verify_corrected_md.py` to remove its dependency on the external layout file — either inline the ground-truth text it checks against directly in the script (mirroring how `verify_references.py` already has no external file dependency and runs standalone, per the original audit), or rewrite the 49 assertions to check `PAKDD_Final_Version_Submit_v3_CORRECTED.md` directly against a small set of inline expected strings instead of a full external transcription diff.

**Recommendation:** Option 2, even though Option 1 is now mechanically available. Regenerating a `pdftotext -layout` extraction still reproduces the kind of automated-extraction fragility this whole bug started from (the original fidelity audit explicitly distrusted `pdftotext` for tables/figures and used direct screenshots instead) — Option 1 would make the test suite permanently depend on a tool the audit already flagged as untrustworthy for this specific document. Option 2 makes the suite self-contained and reproducible on any machine regardless of local poppler availability, matching `verify_references.py`'s already-working pattern.

**Test after fix:** `cd docs/paper-files && python verify_corrected_md.py` → must print `49/49 pass` (or the suite's actual current pass count if it's been modified) with exit code 0, run standalone with no external file dependency, from a clean checkout.

---

## 3. Group C — Decision points (need your call, each with a recommendation)

### 3.1 Should `docs/paper-files/`, `docs/meeting_prep/`, and the audit docs be committed to git?

**Current state:** re-verified via `git status --short` — five paths are untracked (`??`), not four: `docs/paper-files/`, `docs/meeting_prep/`, `docs/audit/PDF_MD_FIDELITY_AND_CONTAMINATION_AUDIT_2026-08-13.md`, `docs/audit/PDF_MD_FIDELITY_IMPACT_ON_STAGES_ABC_2026-08-13.md`, and this remediation plan itself (`docs/audit/PDF_MD_FIDELITY_REMEDIATION_PLAN_2026-08-13.md`) — all untracked since you reverted the `.gitignore` exclusion. All five are visible and safe from accidental loss, but not yet part of the repo's history.

**Recommendation:** commit them, after Groups A and B are applied — the corrected files and the audit trail are exactly the kind of material that should be part of the permanent record, especially since two of the three citation-fix targets (§1.1, §1.2) explicitly reference `docs/audit/PDF_MD_FIDELITY_AND_CONTAMINATION_AUDIT_2026-08-13.md` by path. If that file isn't tracked, the retraction notes point at something that only exists locally.

**Not doing this automatically:** per this session's standing rule, commits only happen when you explicitly ask. Flagged here as a required decision, not silently deferred.

### 3.2 Does the corrected paper need a more prominent home than `docs/paper-files/`?

**Current state:** `PAKDD_Final_Version_Submit_v3_CORRECTED.md` lives in `docs/paper-files/`, a directory whose name doesn't obviously read as "the fidelity-corrected reference copy" to someone browsing `docs/paper/` where the original and its banner-linked siblings live.

**Recommendation:** leave it where it is. §0 already established the original `PAKDD_Final_Version_Submit_v3.md` must stay frozen and unedited per its own banner; promoting the corrected copy into `docs/paper/` alongside it risks exactly the kind of "which one is authoritative" confusion this whole remediation is trying to eliminate. If you want it more discoverable, the lower-risk move is a one-line pointer added to the existing banner on `PAKDD_Final_Version_Submit_v3.md` (e.g., "A transcription-fidelity-corrected copy of this draft's *content* — figures restored, fabricated tables removed — is available at `../paper-files/PAKDD_Final_Version_Submit_v3_CORRECTED.md`; it does not resolve the peer-review concerns above"). This is optional and not required for the contamination fix itself.

### 3.3 Should dormant defects #4 (Algorithm 2 signature) and #6 (dropped assumption marker) be fixed anywhere?

**Current state:** confirmed dormant — nothing in the repo reads either line for meaning (verified last session). They already exist, fixed, in `PAKDD_Final_Version_Submit_v3_CORRECTED.md` (per the original fidelity report's defect table).

**Recommendation:** no action beyond what Group B already does. Once `PAKDD_Final_Version_Submit_v3_CORRECTED.md` is usable (§2), it already carries the fix. No separate task needed.

---

## 4. Order of operations

1. **Group A (§1.1–1.3)** — three doc edits, independent of each other, no ordering constraint between them. (§1.2's consistency check is already done — see below — so this step is just the three edits.)
2. **Group B (§2.1–2.2)** — file move + script fix, independent of Group A.
3. **Verification pass (§5 below)** — re-grep the whole repo for the fabricated fingerprints to confirm zero remaining uncorrected citations, and run both `docs/paper-files/verify_*.py` scripts to confirm they pass standalone.
4. **Decision points (§3.1–3.3)** — resolved by you; §3.1 (commit) naturally comes last since it should capture the finished state, not a partial one.

## 5. Verification / test plan (run after Groups A+B, before calling this closed)

- `grep -rn "Static (Last n classes)\|Adaptive (Last n classes)\|FedAvg (Last n classes)" docs/current/ understanding/` → should return **zero matches** (currently returns 3, at the exact lines fixed above).
- `grep -in "finding.\{0,3\}9\b\|implausibly static" docs/current/STAGE_A_MASTER_REPORT.md` → confirms both the primary entry and any secondary cross-reference (per §1.2's consistency-check note) were updated.
- `cd docs/paper-files && python verify_corrected_md.py && python verify_references.py` → both must exit 0, standalone, no external file dependency errors.
- `ls docs/paper-files/figures/` → `Figure1.png`, `Figure2.png`, `Figure3.png` all present.
- Re-open `PDF_MD_FIDELITY_IMPACT_ON_STAGES_ABC_2026-08-13.md` and update it: §3's table, §6's "not yet applied" framing, and §7's one-liner all currently describe the pre-fix state and will need a final pass marking everything CLOSED once this plan is executed — this doc's own existence obligates that follow-up edit; noted here so it isn't missed.

---

## 6. Summary checklist

| # | Item | Type | Blocked by | Recommendation |
|---|---|---|---|---|
| 1.1 | Retract `PAPER_AMBIGUITY_REPORT.md` #9 | Doc edit | — | Apply exact text above |
| 1.2 | Retract `STAGE_A_MASTER_REPORT.md` #9 (confirmed single occurrence, no secondary references) | Doc edit | — | Apply exact text above |
| 1.3 | Re-point `understanding/02_the_problem.md` line 52 | Doc edit | — | Apply exact text above |
| 2.1 | Move 3 PNGs into `docs/paper-files/figures/` | File move | — | Apply |
| 2.2 | Fix `verify_corrected_md.py`'s missing dependency | Script fix | Choice of Option 1 vs 2 | Option 2 (inline, self-contained) |
| 3.1 | Commit the 5 untracked paths | Git commit | Groups A+B done; your explicit go-ahead | Yes, after A+B |
| 3.2 | Add pointer from original banner to corrected copy | Doc edit | Your preference | Optional, low-risk if wanted |
| 3.3 | Separately fix dormant defects #4/#6 | — | — | No action needed — already covered by 2.2's file being made usable |
| 5 | Full verification sweep + update the impact-assessment doc to CLOSED | Verification | 1.x + 2.x done | Required last step |

Every row above is fully specified. Nothing here needs further research before execution — say the word and I'll run the whole table in one pass.

---

## 7. Verification record — 2026-08-13

This plan was independently re-verified against live repo state after being written, using a separate adversarial pass (not the same reasoning that produced it). Checked: every "current text" quote in §1.1–1.3 against the actual files (all exact, word-for-word, zero drift); the §1.2 "no secondary reference" claim (re-grepped the full `STAGE_A_MASTER_REPORT.md` independently — confirmed, only one hit); §2.1's image-path and file-location claims (confirmed); §2.2's script-failure claim (reproduced the exact `FileNotFoundError` independently, confirmed `verify_references.py` runs standalone); §0's scope-exclusion claims for `draft1.md` and the original `.md`'s banners (confirmed both still read "do not cite numbers from this version"); §3.1's untracked-file count; and, most importantly, a full-repo grep for the fabrication's fingerprint strings to check for a missed 4th contamination site — **none found**. The 3 Group-A targets plus the 2 explicitly-excluded frozen files (`draft1.md`, `PAKDD_Final_Version_Submit_v3.md`) account for every file in the repo that contains the fabricated table's signature; `PAKDD_Final_Version_Submit_v3_CORRECTED.md` is confirmed clean.

Two inaccuracies surfaced and were corrected in this doc: §0's claim that `results/README.md` uses `"(Last n classes)"` phrasing was wrong (it uses different wording entirely — "rare-acc" — and has zero occurrences of the flagged phrase; the underlying conclusion "no fix needed there" was still correct, only the stated reason was imprecise) and §2.2's Option 1 caveat about no local `pdftotext`/poppler install was stale (`pdftotext` is now present on this machine; the Option 2 recommendation is unchanged since it never depended on that caveat). §3.1's file count was corrected from "four" to "five" to include this plan document itself among the untracked audit paths. No other errors found; no gaps found. **Plan confirmed accurate and complete as of this verification pass — ready to execute.**
