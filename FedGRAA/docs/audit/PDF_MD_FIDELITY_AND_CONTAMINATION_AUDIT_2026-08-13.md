# PDF↔MD Fidelity Audit + Downstream Contamination Check — 2026-08-13

**Subject:** `docs/paper/PAKDD_Final_Version_Submit_v3.pdf` vs. `docs/paper/PAKDD_Final_Version_Submit_v3.md` (the llamaparse conversion), plus an independent review of `docs/paper-files/` (a fix for that conversion, produced in a separate Claude web session) and a repo-wide check for whether the conversion's defects leaked into other documents.

**Verdict, short version:** the prior session's core finding is correct and its fix is *substantively* good — but it overstated its own testing, shipped one broken thing, and (because it scoped itself narrowly to one file pair) missed the actual worst consequence: **the fabricated data it identified has already been cited as genuine evidence in three other repo documents, including a "SUSPICIOUS DATA" finding in the most recent Stage A audit report.** That's the headline. Everything else here is in service of that.

---

## 1. Method

The PDF has no local `pdftoppm`/poppler install, and `pdftotext` alone can't be trusted for tables/figures, so I asked you for direct screenshots of all 12 rendered PDF pages and used those as ground truth — not another automated extraction that could carry its own bugs. I read the full `.md` (1,739 lines) and compared it page-by-page against your screenshots, independently of (and before fully trusting) the prior session's report. I then read that report, the corrected `.md` it produced, and both of its verification scripts — and actually *ran* those scripts rather than taking the report's "49/49 pass" claim on faith. Finally, I grepped the whole repo for fingerprints of the fabricated data to check whether it had spread beyond the one file.

## 2. Original `.md` vs. PDF — independently confirmed

I found the same 7 defects the prior session found, verified each one myself against your screenshots, and found nothing additional. Severity is my own assessment, not carried over from the prior report:

| # | Defect | Location (original `.md`) | Severity |
|---|---|---|---|
| 1 | **~870–936 fabricated numeric data points** across three giant HTML tables purporting to give exact per-round accuracy values for every (round × series × γ) combination in Figures 1–3. **The PDF contains no such table anywhere** — Figures 1–3 are plotted raster line charts with only axis tick labels (0.0–0.8/0.5, 20–200), no per-point numeric labels. Every one of these ~900 cell values (e.g. "round 100, Adaptive All-classes = 0.54") was invented by the conversion, not read off the source. | lines ~371–929 (Fig 1), ~938–1390 (Fig 2), ~1396–1626 (Fig 3) | **Critical** |
| 2 | Zero images embedded anywhere in the file — the real chart panels (which do exist as embedded raster PNGs in the PDF, confirmed via `pdfimages -list`) are entirely replaced by defect #1's invented tables | whole document | **High** (consequence of #1) |
| 3 | Method-comparison table (the one *real*, small results table — FedAvg/Static/FedGRAA × CIC/UNSW) — all 10 accuracy values silently dropped their `%` suffix (`88.88` instead of `88.88%`) | lines 903–927 | Low — numbers correct, cosmetic |
| 4 | Algorithm 2, line 1: signature reads `FedGRAA($\theta_k^t$, ...)` — should be `$\theta_c^t$` (the *vector* of all clients' models, per Algorithm 1 line 9, which defines `θᵗ_c` and passes it to FedGRAA). Confirmed against your screenshot of page 6: PDF clearly shows `θᵗ_c`. This isn't cosmetic — it silently changes what the algorithm's input means. | line 159 | **Medium** |
| 5 | Eq. (8) uses `\quad (8)` instead of `\eqno(8)` like the other 9 numbered equations — internal markup inconsistency | line 137 | Low |
| 6 | Appendix, Lemma 1 proof: dropped the trailing inline marker "`3`" after "ensures convergence to $w^*$" — this paper's style uses these as informal inline references to numbered Assumptions (mirrors "β-smooth 1" = Assumption 1 two words earlier). Confirmed present in your page-11 screenshot, absent in the `.md`. | line 1685 | Low — loses a cross-reference, no content loss |
| 7 | Abstract, sentence 2: PDF has a comma — "the standard *FedAvg* approach, encounters" — `.md` drops it | line 29 | Trivial |

**Everything else checks out.** All prose (every paragraph, both Related Work subsections, all of Section 3's derivation), all 10 numbered equations' math content, Algorithm 1 in full, Tables 1 and 2 (all 25 rows of sample-distribution data, CIC-IDS-17 and UNSW-NB15, verified cell-by-cell against your screenshots), the real accuracy-comparison table's numbers (not its formatting — see #3), the Convergence Proof section including both Lemmas, and **all 14 references** are faithful transcriptions with correct arXiv/DOI links.

## 3. Assessment of `docs/paper-files/` (the prior session's fix)

### What's solid
- **The diagnosis is correct.** Confirmed above, independently.
- **The reference check is real and passes.** I ran `verify_references.py` myself: `14/14 exact match`, genuinely reproducible, no issues.
- **The reconstructed figures are legitimate.** I viewed `Figure1.png`, `Figure2.png`, and `Figure3.png` directly and compared curve shapes and color-series against your PDF screenshots of pages 8–9 — they match. This was a real fix, not a placeholder.
- The specific text-level fixes (items 3–7 above) are correctly applied in `PAKDD_Final_Version_Submit_v3_CORRECTED.md`.

### What's not solid — two concrete, verifiable problems in the delivery

**(a) The "49/49 pass, confirmed to run standalone" claim is false as delivered.** The report states: *"the verification scripts... use relative paths and were confirmed to run standalone from a clean `outputs/` directory with nothing else present."* I ran `python verify_corrected_md.py` from inside `docs/paper-files/` exactly as instructed and it crashed immediately:

```
FileNotFoundError: [Errno 2] No such file or directory: 'PAKDD_Final_Version_Submit_v3_pdftotext_layout.txt'
```

That file — the ground-truth text extraction the whole 49-assertion suite is checked against — was never included in what got saved into this repo. The suite cannot run, so its "49/49" result is unverifiable here regardless of whether it was true in the other session's own environment. This is the exact class of problem the report itself was proud of having caught once already (the "leftover scratch copy" bug it describes fixing in its own §"tested, not just assembled") — except this time it wasn't caught before delivery. `verify_references.py`, by contrast, has no external file dependency (its reference text is hardcoded inline) and does run correctly, which is why I could confirm that one.

**(b) The corrected `.md`'s embedded images are broken as delivered.** `PAKDD_Final_Version_Submit_v3_CORRECTED.md` references its images as `figures/Figure1.png`, `figures/Figure2.png`, `figures/Figure3.png` (lines 386, 434, 442). The actual delivered files are flat in `docs/paper-files/` — `Figure1.png` etc., no `figures/` subdirectory exists. Opening the corrected file as-is (in this repo, at this path) renders three broken image links. This is a real defect in exactly the fix that was the whole point of the exercise — restoring the missing figures.

Neither of these is hard to fix (create `docs/paper-files/figures/` and move the three PNGs into it; regenerate or hand-write the `pdftotext -layout` extraction), but as delivered, the two claims "the fix embeds real images" and "the fix was tested, 49/49" are each half-true: the content is right, the packaging that would let you verify or use it is not.

## 4. The actual big finding: the fabricated tables already contaminated live audit documents

The prior session scoped itself explicitly to "transcription fidelity only" and didn't check whether its finding mattered anywhere else. It does. I grepped the repo for the fabricated table's unique column-header fingerprints (`"Adaptive (Last n classes)"`, `"FedAvg (Last n classes)"`, `"Static (Last n classes)"` — strings that only exist inside the invented tables, never in real paper prose) and found them quoted as if they were genuine paper data in three places outside the two known conversion files:

**`docs/current/STAGE_A_MASTER_REPORT.md` (93KB, dated 2026-08-10 — the most recent Stage A deliverable), finding #9:**
> *"Per-round tables show implausibly static values over long stretches — SUSPICIOUS DATA (weaker). E.g. CIC "Static (All classes)" stays exactly `0.44` from round 60 through round 200 in the γ=0 panel; "Static (Last n classes)" stays exactly `0.88` for rounds 20–100. Real training curves essentially never freeze to the same 2-decimal value across dozens of evaluation checkpoints."*

This finding accuses the **paper's authors** of presenting suspicious/implausible data. But the "per-round table" it's analyzing doesn't exist in the paper — it's the llamaparse hallucination from §2 of this report. The PDF's Figure 1 is a line chart; nothing in it has 2-decimal per-round values to freeze or not freeze, because there are no per-round numeric labels at all, real or otherwise. This finding is analyzing a document-conversion artifact and drawing a conclusion about the source paper's research integrity from it. It needs to be retracted or explicitly re-scoped to "artifact of a bad PDF conversion, not a paper defect."

(Finding #8, immediately above it in the same file — about the *real* small comparison table's suspiciously round numbers, 66.66/88.88/77.77/33.33 — is a different, legitimate observation about genuine paper content. Confirmed present in your PDF screenshots. That one stands; only #9 is compromised.)

**`docs/current/PAPER_AMBIGUITY_REPORT.md` (2026-07-25), lines 226–228:**
> *"...**exactly** flat for 60–100+ consecutive rounds (e.g., CIC "Static (All classes)" stays at exactly `0.44` from round 60 through round 200 in the γ=0 panel; "Static (Last n classes)" stays at exactly `0.88` for rounds 20–100). Neural network accuracy on held-out validation data during continued training essentially never freezes to the same 2-decimal value..."*

Same fabricated numbers, same conclusion, earlier document — likely the actual origin point that finding #9 above was copied or re-derived from three weeks later.

**`understanding/02_the_problem.md`, line 52:**
> *"Consequence: FedAvg gets near 0% accuracy on the rare attack classes — you can see this in the result tables, where the "FedAvg (Last n classes)" column is `0.00` everywhere."*

Lower stakes than the other two — the underlying claim (FedAvg underperforms on rare classes) is very likely still true and is visually consistent with the real chart images (the red/FedAvg-adjacent series does sit near the bottom in most panels) — but the document is citing a table that isn't real as its evidence source. The claim should be re-pointed at the actual chart images or at the real comparison table, not retracted, but the citation is currently to a hallucination.

**Why this matters more than the file-pair problem itself:** a bad PDF→MD conversion is a nuisance, fixable in one pass, which is exactly what `docs/paper-files/` mostly did. A conversion artifact that gets treated as ground-truth *evidence* inside the project's own audit trail — and used to accuse the source paper of data fabrication — is a different category of problem. `STAGE_B_MASTER_REPORT.md` (same date, 2026-08-10) does not reference or correct finding #9, so as of the most recent Stage B pass this is still live and uncorrected.

I checked whether the fabrication had spread further (specific values, not just the column-header strings) into `results/README.md` and `understanding/04_experiments_and_results.md`, since those also matched a looser grep for the real comparison-table numbers (88.88, 77.77, 66.66, 33.33) — it hadn't. Those two cite the *real* 5-row comparison table and, in `results/README.md`'s case, the project's own independent experiment reruns. That grep hit was a false lead; worth ruling out explicitly rather than leaving it unchecked.

## 5. One more instance of the same pattern, lower priority

`docs/paper/draft1.md` — an earlier, already-superseded draft (superseded per its own lineage and the citation-integrity note on the v3 banner, which separately flags a fabricated "FedSND" citation in this same draft) — has the identical `"Communication Rounds"` fabricated-table signature. Not re-audited in full here since it's explicitly out of the citable lineage already, but flagged in case anything still reads from it.

## 6. Recommended actions, in priority order

1. **Correct or retract `STAGE_A_MASTER_REPORT.md` finding #9.** This is the one with the most exposure — it's in the most recent, most substantial Stage A deliverable and makes a direct data-integrity accusation against the source paper based on a document-conversion bug, not paper content. Re-scope it as "artifact of the PDF→MD conversion tool, not present in the source" or delete it.
2. **Fix the citation in `PAPER_AMBIGUITY_REPORT.md`** the same way — either retract the specific per-round-freezing claim or re-derive it from the actual chart images (I can pull the relevant panel crops from your screenshots if useful).
3. **Re-point `understanding/02_the_problem.md` line 52** away from the fabricated table to the real chart images or the real comparison table.
4. **Fix the two delivery defects in `docs/paper-files/`:** move the three PNGs into a `figures/` subdirectory (or fix the `.md`'s image paths to match where they actually are), and either regenerate `PAKDD_Final_Version_Submit_v3_pdftotext_layout.txt` or drop the dependency so `verify_corrected_md.py` can actually run and back up its "49/49" claim.
5. **Once 4 is done, `PAKDD_Final_Version_Submit_v3_CORRECTED.md` is a good replacement** for `docs/paper/PAKDD_Final_Version_Submit_v3.md` if a faithful reference copy of the v3 draft's *content* is needed going forward — it's more accurate than the current file on every count in §2. It should **not** be read as resolving anything about the paper's substance; the SUPERSEDED banner and the peer-review concerns it names are untouched and separate, exactly as its own note says.

## 7. What I did not do

I did not re-litigate the substantive peer-review concerns in the SUPERSEDED banner (simplex projection, convexity-vs-architecture contradiction, unseeded results, missing baselines) — those are a different, already-tracked problem with their own resolution path (`FedGRAA_v2_PAKDD_Submission.md`). This audit is scoped to: is the `.md` a faithful transcription of the `.pdf`, is the prior fix sound, and did the transcription's defects leak anywhere they'd be mistaken for real evidence. All three are now answered.
