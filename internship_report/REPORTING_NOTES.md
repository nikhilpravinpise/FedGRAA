# Reporting Notes - FedGRAA Internship Report

One-page guide to filling the placeholders, recompiling, and understanding the
layout decisions in `report.tex`.

## 1. Placeholders to fill

All unverified personal/institutional fields render as angle-bracketed italic
text via the `\ph{...}` macro (defined in the preamble). Replace the argument,
keep or drop the `\ph{}` wrapper - both compile.

| Field | Occurrences | Where |
|---|---|---|
| `\ph{Roll No.}` | 2 | title page, certificate |
| `\ph{Department}` | 7 | title page, certificate, signatures |
| `\ph{Home Institution}` | 4 | title page, certificate, acknowledgement, signatures |
| `\ph{Professor Name}` | 4 | title page, certificate, acknowledgement, signatures |
| `\ph{Mentor Name}` | 4 | title page, certificate, acknowledgement, signatures |
| `\ph{Place}` | 4 | title page, signature blocks, closing |
| `\ph{Date}` | 2 | signature blocks |
| `\ph{Start Date}` / `\ph{End Date}` | 1 each | certificate |
| `\ph{Month Year}` | 2 | title page, closing |

Everything else in the report - every number, table cell, figure, and citation -
is final and traceable to a repository artifact (see Appendix A of the report).

## 2. Recompiling

Local (used throughout this project):

```bash
mkdir -p build && tectonic --outdir build report.tex
```

`build/` is gitignored, so a fresh clone must create it first - tectonic does
not create the output directory itself. Tectonic resolves `references.bib`,
runs BibTeX internally, and downloads any missing packages on first build.
Output lands in `build/report.pdf`.

Overleaf fallback (zero local install): zip `report.tex`, `references.bib`, and
the `figures/` folder, upload to a new Overleaf project, compile with pdfLaTeX
(two passes for cross-references and the bibliography).

Expected result: **36 PDF pages total** - title page, roman folios i-viii,
arabic folios 1-25 (Chapter 1 through Bibliography), arabic 26-27 (Appendix A).
The Chapter-1-to-Bibliography span is 25 arabic folios, inside the plan's
20-25 target.

## 3. Layout decisions worth knowing

- **Near-single spacing** (`\setstretch{0.95}`), `\footnotesize` tables,
  subfigure merging, and a compressed bibliography keep the report inside
  the page budget without dropping content. Loosening any of these pushes
  past 25 arabic folios.
- The List of Figures and List of Tables are compact (folios vi and vii);
  the nomenclature table is `\footnotesize`-scoped.
- Paired plots live in shared floats: the two diagnosis plots
  (`fig:diagnosis`) and the frontier + classic-optimizer pair
  (`fig:stagec-frontier`) sit side by side; the Stage B ablation is one
  stacked table for both datasets.
- The bibliography uses `plain` style at `\scriptsize` with tightened item
  spacing; all 43 entries are cited in the text.

## 4. Honest-reporting caveats (deliberate, do not "fix")

- The original FedGRAA manuscript's publication status is **unresolved**; its
  printed 88.88%/21.09% figures are reported as *claims*, not measurements.
- UNSW `n=7` collapses to the `n=5` client set (only five non-capped rare
  classes exist) - reported as an honest divergence, not patched.
- Defense-induced rare-class collapse (30/30 CIC, 19/30 UNSW cells at 0%
  recall) and the single non-significant Holm comparison
  (`p_holm = 0.0625`) are stated exactly.
- The SecAgg module is an *ideal* interface model, not a full Bonawitz
  cryptographic implementation; plaintext parameters remain exposed to
  gradient-inversion attacks.
- FedGRAA-v2's guarantee is gradient-mapping stationarity `O(1/T)+O(zeta^2)`,
  weaker than the manuscript's global-optimality claim; PL, `beta_eff`, and
  the Clarke-subdifferential treatment remain theory-hardening items.

## 5. Known non-defects

- The compile is clean: zero `Overfull`/`Underfull` boxes that affect the
  text (one benign `badness 1348` underfull inside `report.bbl`, cosmetic).
- `scripts/archive/gpu/test_gpu.py` fails collection on this CPU-only machine
  (it expects a CUDA build); it is an archived probe outside the `tests/`
  suite, not a regression.
