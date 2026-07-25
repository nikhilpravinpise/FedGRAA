# FedGRAA — Strict Keras Reproduction, Divergence Diagnosis & Corrected Convergence Theory

A complete, from-scratch Keras reproduction of the paper **"Federated
Gradient-Refined Adaptive Averaging for Intrusion Detection Systems"
(FedGRAA)** — plus a measured, statistically-tested account of *why the
algorithm as specified does not converge on real intrusion-detection data,
and what minimal correction fixes it*. (The manuscript's publication status
is itself unresolved — see [`docs/current/STAGE_B_MASTER_REPORT.md`](docs/current/STAGE_B_MASTER_REPORT.md)
§21.)

The implementation uses **TensorFlow/Keras** for the network described in
Section 5: `input → Dense(200, tanh) → Dense(100, tanh) → Dense(50, tanh) →
Dense(n_classes, softmax)`, trained with sparse categorical cross-entropy and
Adam.  Every ambiguous design choice that was previously hidden behind a flag
has been removed; the remaining ambiguities and apparent paper errors are
documented in [docs/archive/strict_reproduction_report.md](docs/archive/strict_reproduction_report.md)
and [docs/current/PAPER_AMBIGUITY_REPORT.md](docs/current/PAPER_AMBIGUITY_REPORT.md).

---

## Project status

| Stage | Goal | Status | Capstone document |
|---|---|---|---|
| **A** | Faithful reproduction — every paper table/figure has a persisted, reproduced-or-honestly-diverging counterpart | ✅ **complete** | [docs/current/STAGE_A_MASTER_REPORT.md](docs/current/STAGE_A_MASTER_REPORT.md) |
| **B** | Package the divergence critique as a rigorous, citable component — formal theory + statistically-tested 10-seed ablation | ✅ **complete** | [docs/current/STAGE_B_MASTER_REPORT.md](docs/current/STAGE_B_MASTER_REPORT.md) |
| **C** | Build the novel extension for an A/A*-tier venue | ⏳ not started — gated on a mentor conversation | [docs/research_c/STAGE_C_LITERATURE_MASTER.md](docs/research_c/STAGE_C_LITERATURE_MASTER.md) |

Verification signals on the current tree:

```powershell
.venv313\Scripts\python.exe -m pytest tests --tb=short          # 151 passed
.venv313\Scripts\python.exe scripts\check_stage_a_complete.py   # exit 0 (completeness)
.venv313\Scripts\python.exe scripts\check_stage_b_complete.py   # exit 0 (completeness)
.venv313\Scripts\python.exe scripts\audit_stage_a.py            # exit 0 (correctness)
.venv313\Scripts\python.exe scripts\audit_stage_b.py            # exit 0 (correctness)
.venv313\Scripts\python.exe scripts\check_doc_links.py          # exit 0 (doc-link integrity)
```

The completeness gates ask "is every artifact present and shaped right?";
the correctness audits ask the harder question, "are the numbers actually
true?" — every reported value is re-derived from the raw persisted JSON and
independently recomputed, never read back from the document asserting it.
See [`docs/audit/AUDIT_REMEDIATION_PLAN_2026-07-25.md`](docs/audit/AUDIT_REMEDIATION_PLAN_2026-07-25.md).

---

## Environment

- **Python 3.9 – 3.13**.  TensorFlow 2.21.0 provides wheels for Python 3.13.
- **RAM: 4 GB minimum, 8 GB recommended** when loading the full CIC-IDS-2017
  dataset (~2.8 M rows).  Use `--cache-dir .cache` to avoid re-reading the CSVs.
- No GPU required.  Training is CPU-only (a GPU attempt measured 25 % *slower*
  at this model size — see [docs/archive/GPU_DECISION_AND_NEXT_STEPS.md](docs/archive/GPU_DECISION_AND_NEXT_STEPS.md)).

### Install

```powershell
# Create a Python 3.13 environment, then:
pip install -r requirements.txt
```

Dependencies: `tensorflow`, `numpy`, `pandas`, `scikit-learn`, `scipy`
(significance testing), `pytest`, and optional `matplotlib` for plotting.

> **This repository's own environment lives at `.venv313/`**, built from
> `requirements.txt` above. The commands in this README (`Verification
> signals`, `Tests`, below) all invoke `.venv313\Scripts\python.exe`
> explicitly, on purpose: neither the system `python` on `PATH` nor the
> Windows Store shim has these dependencies installed, so running any gate
> or the test suite under a different interpreter fails immediately with
> `ModuleNotFoundError: No module named 'numpy'`. If you create your own
> environment elsewhere, substitute its interpreter path consistently.

### Datasets

Raw CSVs are **not tracked in git** (`*.csv` is gitignored to keep the repo
lightweight — see [`.gitignore`](../.gitignore)). On a fresh clone, `data/`
will be empty; you need to download the CSVs yourself and place them as
below before running anything against real data.

| Dataset | Expected location | Source |
|---|---|---|
| **CIC-IDS-2017** | `data/CIC-IDS-2017/` (8 CSV files) | [official UNB CIC-IDS-2017 page](https://www.unb.ca/cic/datasets/ids-2017.html) |
| **UNSW-NB15** | `data/UNSW-NB15/` | [official UNSW-NB15 page](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |

The copies used to produce every result in this repo were forensically
verified against raw CSV SHA-256s and canonical published distributions —
see [docs/audit/data_provenance_audit.md](docs/audit/data_provenance_audit.md).
**Verdict: genuine, complete, correctly labelled.** The divergence documented
below is algorithmic, not a data-quality artifact — but you must supply your
own copy of the CSVs to reproduce it locally.

---

## Data setup

### CIC-IDS-2017

Download the 8 CSV files from the
[official UNB CIC-IDS-2017 page](https://www.unb.ca/cic/datasets/ids-2017.html)
and place them in `data/CIC-IDS-2017/`. The first run reads all rows (~95 s)
and saves an `.npy` cache; subsequent runs load the cache in under 3 s.

```powershell
# First run: reads CSVs, saves cache
python run_experiments.py --data-dir data/CIC-IDS-2017 --cache-dir .cache --datasets CIC-IDS-2017

# Subsequent runs: load cache
python run_experiments.py --data-dir data/CIC-IDS-2017 --cache-dir .cache --datasets CIC-IDS-2017
```

### UNSW-NB15

Place the official partitioned train/test CSVs in `data/UNSW-NB15/`. The
loader auto-detects the label column. If the directory is empty, the loader
can attempt to download the files automatically from the
[official UNSW page](https://research.unsw.edu.au/projects/unsw-nb15-dataset).

---

## Running experiments

### Choosing what to run

| Flag | What it controls |
|---|---|
| `--datasets CIC-IDS-2017` | Run only CIC (skip UNSW). Default: both. |
| `--sweep gamma` | Figure 1 only (5 method runs). |
| `--sweep n` | Figures 2-3 only (9 runs). |
| `--sweep both` | All 14 runs (default). |
| `--rounds N` | Training rounds. Paper uses 200. Use 5–30 for quick checks. |
| `--eval-every N` | Print metrics every N rounds. Default 20. |
| `--cache-dir DIR` | Cache/reload preprocessed arrays here. |
| `--round-zero-zero` | Force round-0 accuracies to 0.00 (default True). |
| `--baselines` | Include FedProx / FedOpt / FedSND (related work only, never run head-to-head). |
| `--parallel` | Run sweep configs concurrently via `ProcessPoolExecutor`. |
| `--results-dir DIR` | Persist raw sweep results to `<DIR>/<dataset>_gamma_sweep.json` / `_n_sweep.json`. |
| `--plot-dir DIR` | Save Figure 1 / 2-3 PNGs here. |
| `--checkpoint-dir DIR` | Save/resume training checkpoints. |

### Recommended workflows

**Quickest sanity check (~1–2 min per method):**
```powershell
python run_experiments.py --rounds 5 --eval-every 5 --datasets CIC-IDS-2017 --sweep gamma
```

**Full Figure 1 reproduction (CIC only, several hours on CPU):**
```powershell
python run_experiments.py --data-dir data/CIC-IDS-2017 --cache-dir .cache --datasets CIC-IDS-2017 --sweep gamma
```

**Single experiment via the module CLI:**
```powershell
python -m fedgraa run --dataset CIC-IDS-2017 --method fedgraa --gamma 1 --n 3 --rounds 50 --data-dir data/CIC-IDS-2017 --cache-dir .cache
```

**Reproduce the Stage B ablation campaign** (~5.5 h unattended, resumable):
```powershell
python scripts\run_ablation_seed_sweep.py --dataset CIC-IDS-2017 --seeds 0-9 --workers 3
python scripts\run_ablation_seed_sweep.py --dataset UNSW-NB15   --seeds 0-9 --workers 3
python scripts\build_ablation_table.py --dataset CIC-IDS-2017
python scripts\render_ablation.py      --dataset CIC-IDS-2017
```
The driver skips any `(arm, seed)` already on disk, so re-running it on a
complete tree is a no-op.  Full runbook: [docs/current/STAGE_B_MASTER_REPORT.md](docs/current/STAGE_B_MASTER_REPORT.md) §18.2.

---

## Actual measured results

### 1. Literal reproduction (real CIC-IDS-2017, 200 rounds, n=3)

```
Static Reparam.  >  FedAvg ≈ FedGRAA(any γ) ≈ 0%     (rare-class accuracy)
```

| Method | All-class acc | Rare-class acc |
|---|---|---|
| FedAvg | 61.0% | 0.0% |
| Static Reparam. | 52.3% | **91.7%** |
| FedGRAA (γ=0) | 62.5% | 0.0% |
| FedGRAA (γ=0.5) | 18.6% | 0.0% |
| FedGRAA (γ=1) | 17.4% | 0.0% |

This is **not** what the paper reports (it claims FedGRAA reaches ~88% on
minority classes, beating both baselines).

### 2. The diagnosed root cause

FedGRAA's aggregation weights `w^t` **leave the probability simplex in round 1
at every γ**, because Algorithm 2 as printed has no projection step.  Off the
simplex, the aggregated model `θ_g = Σ w_k θ_k` stops being a convex
combination of client models and becomes an unbounded affine extrapolation —
so the paper's own Lemma 1 (composite loss must be non-increasing) fails
immediately.

Measured across the full Stage B campaign — 10 seeds × 3 γ × 2 datasets ×
2 unprojected arms = **120 cells** — `first_simplex_violation_round = 1` in
**120 of 120**, zero exceptions.  Under the projected fix the violation round
is `None` in all 120 projected cells, with `Σw = 1` to float32 precision.

### 3. The minimal fix, and what it recovers

Simplex projection **plus** a 10× smaller `η′` (0.01 instead of 0.1, `E′`
unchanged), kept strictly diagnostic-only.  Rare-class accuracy, mean
[95 % bootstrap CI] over **10 seeds**, 60-round horizon:

| Arm | CIC-IDS-2017 (γ=1) | UNSW-NB15 (γ=1) |
|---|---|---|
| FedAvg | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| Static Reparam. | 84.4% [77.2, 91.4] | 48.3% [40.8, 56.0] |
| Literal (η′=0.1, no projection) | 10.0% [0.0, 30.0] | 27.6% [15.0, 39.9] |
| Projection alone | 51.0% [24.2, 77.1] | 67.2% [59.6, 73.6] |
| η′ cut alone | 88.9% [68.9, 100.0] | 36.4% [33.7, 39.8] |
| **Both (the fix)** | **89.3% [83.6, 94.6]** | **69.2% [63.4, 73.8]** |

Composite-loss blowup worst case: literal reaches **1.86×10⁶×** on CIC and
**46.85×** on UNSW; the fix never exceeds 1.02× / 1.87×.

**The honest, dataset-dependent finding:** which of the two levers drives the
*accuracy* recovery reverses between datasets — the η′ cut dominates on
CIC-IDS-2017, projection dominates on UNSW-NB15 — but only the **combination**
performs well on both at every γ.  Full 2×2 factorial with paired *t* /
Wilcoxon / Cohen's *d* / Holm–Bonferroni: [docs/current/STAGE_B_MASTER_REPORT.md](docs/current/STAGE_B_MASTER_REPORT.md) §8–12.

### 4. The corrected convergence theory

Replacing Algorithm 2 line 6 with `w^t = Π_Δ(w^{t−1} − η′ ∂L'(w^{t−1}))` turns
the weight learner into projected (sub)gradient descent on the compact convex
set `Δ`, giving two guarantees stated **explicitly weaker** than the paper's
Theorem 1: feasibility is *structural* (the round-1 blowup becomes provably
impossible), and the iterates reach a *stationary/KKT point* at rate `O(1/T)`
— not the global minimum, because the loss is non-convex.  Full statement,
proof sketch, and the three items flagged for referee-grade hardening:
[docs/current/fedgraa_convergence_defect.md](docs/current/fedgraa_convergence_defect.md).

**The fix is never merged into the reproduction.** `ProjectedFedGRAA` lives
solely in `scripts/diagnose_fedgraa.py`; `fedgraa/` implements Algorithm 2
literally and unprojected, on purpose, per the standing no-silent-patching
mandate.

---

## Tests

Run the full suite before pushing:

```powershell
.venv313\Scripts\python.exe -m pytest tests --tb=short      # 151 passed
```

The suite covers:

- **Gradient correctness** — Keras back-prop and the FedGRAA weight gradient
  agree with finite differences (to float32 precision).
- **Fidelity** — dataset specs match Tables 1 & 2; partitioning follows the
  paper (one attack per client, equal benign, 90/10 split, median-frequency
  validation split).
- **Aggregation** — FedAvg / Static weights are correct; `γ` extremes select the
  right objective.
- **End-to-end** — a short federated run shows FedGRAA lifting rare-class
  accuracy above FedAvg.
- **Statistics** (Stage B) — bootstrap CI, paired *t*, Wilcoxon, Cohen's *d*,
  Holm–Bonferroni, each with an explicit zero-variance/degenerate-cell contract
  tested against SciPy references.
- **Ablation pipeline** (Stage B) — driver enumeration/resumability, table
  builder wiring, renderer smoke tests.

Five standalone checks live outside `pytest` (so the unit suite stays
decoupled from large committed binaries), all of which must exit `0`:
`scripts/check_stage_a_complete.py` and `scripts/check_stage_b_complete.py`
(completeness — is every artifact present and shaped right?);
`scripts/audit_stage_a.py` and `scripts/audit_stage_b.py` (correctness — is
every reported number independently re-derived from the raw persisted JSON,
never just read back from the document asserting it?); and
`scripts/check_doc_links.py` (every relative markdown link resolves).

---

## How it maps to the paper

| Paper element | Code |
|---|---|
| Keras Sequential `200→100→50` tanh, softmax | [fedgraa/model.py](fedgraa/model.py) — `MLP` |
| Adam (`η=0.01` local, `η'=0.1` weights) | [fedgraa/model.py](fedgraa/model.py) — `Adam` |
| Tables 1 & 2, hyperparameters | [fedgraa/config.py](fedgraa/config.py) |
| Real loader + synthetic generator + cache | [fedgraa/data.py](fedgraa/data.py) |
| One-attack-per-client, 90/10, median split | [fedgraa/partition.py](fedgraa/partition.py) |
| FedAvg (eq. 5), Static (eq. 7), FedGRAA (eq. 8-10) | [fedgraa/aggregation.py](fedgraa/aggregation.py) |
| Algorithm 1 federated loop | [fedgraa/trainer.py](fedgraa/trainer.py) |
| All-class & last-n accuracy | [fedgraa/metrics.py](fedgraa/metrics.py) |
| Figure 1 / 2 / 3 sweeps | [fedgraa/experiment.py](fedgraa/experiment.py) |
| Appendix A.1 convergence proof (explained) | [understanding/05_convergence_proof.md](understanding/05_convergence_proof.md) |
| Ambiguities / paper errors (17 catalogued) | [docs/current/PAPER_AMBIGUITY_REPORT.md](docs/current/PAPER_AMBIGUITY_REPORT.md) |

### Core math

The aggregated parameters are `P = Σ_k w_k P_k`, so `∂P/∂w_k = P_k`.  For the
composite validation loss `L'(P) = (1−γ)·min(L_h, L_m) + γ·L_u`:

```
∂L'/∂w_k = ⟨ ∇_P L'(P), P_k ⟩
```

FedGRAA runs this gradient through Adam for `E'=5` epochs per round, with **no
simplex projection** (Algorithm 2 does not contain one — this is the defect).

---

## Project layout

```
fedgraa/
  config.py        # hyperparameters + Table 1/2 dataset specs
  model.py         # Keras Sequential MLP + persistent Adam
  keras_backend.py # compatibility alias re-exporting model.py
  data.py          # real CSV loader (with NPY cache) + synthetic generator
  partition.py     # non-IID partition + median-frequency split
  aggregation.py   # FedAvg / Static / FedGRAA
  trainer.py       # Algorithm 1 loop
  metrics.py       # all-class / last-n accuracy
  experiment.py    # Figure 1/2/3 sweeps
  baselines.py     # FedProx / FedOpt / FedSND (related work, implemented but unrun)
  stats.py         # NEW (Stage B): bootstrap CI, paired t, Wilcoxon, Cohen's d, Holm
  plot.py          # matplotlib figures
  __main__.py      # CLI: python -m fedgraa ...
data/
  CIC-IDS-2017/    # 8 CSV files
  UNSW-NB15/       # official train/test CSVs
.cache/            # auto-created NPY cache (git-ignored)
scripts/
  diagnose_fedgraa.py        # per-round divergence instrumentation + ProjectedFedGRAA ablation
  sensitivity_sweep.py       # eta'/E' sweep
  render_diagnostics.py      # figures + summary tables from diagnostic JSON
  run_seed_sweep.py          # seed-variance bands (FedAvg / Static)
  seed_sweep_combined_fix.py # combined-fix seed band (Stage A; frozen)
  real_reproduction.py       # canonical real-data gamma sweep
  run_ablation_seed_sweep.py # NEW: resumable, worker-capped Stage B campaign driver
  build_ablation_table.py    # NEW: per-seed JSONs -> consolidated table JSON
  render_ablation.py         # NEW: table JSON -> markdown + figure
  check_stage_a_complete.py  # Stage A completeness signal
  check_stage_b_complete.py  # NEW: Stage B completeness signal
  archive/gpu/               # explored-and-rejected GPU benchmark (25% slower than CPU)
results/
  *_gamma_sweep.json, *_n_sweep.json   # persisted sweep results (--results-dir)
  README.md                            # Stage A reproducibility manifest
  figures/                             # Figure 1-3 PNGs + diagnostic plots
  diagnostics/                         # 80 per-seed ablation runs + sensitivity + seed sweeps
  ablation/                            # NEW: consolidated Stage B tables, figures, baselines, logs
    {CIC_IDS_2017,UNSW_NB15}_ablation_table.{json,md}
    {CIC_IDS_2017,UNSW_NB15}_ablation.png
    seed_sweep_{DS}_{fedavg,static}_r60.json
    README.md                          # Stage B manifest (27 checked paths)
    _logs/                             # raw campaign stdout
tests/             # 142 unit + gradient + e2e + diagnostic + statistics tests
docs/
  current/         # the live documents
    STAGE_A_MASTER_REPORT.md           # Stage A capstone (exhaustive)
    STAGE_B_MASTER_REPORT.md           # Stage B capstone (exhaustive)
    fedgraa_convergence_defect.md      # the standalone theory+ablation section
    fedgraa_divergence_diagnosis.md    # the original root-cause analysis
    PAPER_AMBIGUITY_REPORT.md          # paper-only critique (17 issues)
    publication_roadmap.md             # router doc: Stage A/B/C sequencing
    stage_a_plan.md, stage_b_plan.md   # execution plans
    STAGE_B_REMAINING_TASKS.md         # the Stage B runbook as executed (superseded)
    STAGE_DIRECTION_ASSESSMENT.md      # where the project sits in the research->write pipeline
  audit/           # independent verification passes
    REPRODUCTION_AUDIT.md              # line-by-line paper<->code fidelity audit
    data_provenance_audit.md           # dataset genuineness verification
    full_audit_2026-07-23.md           # from-scratch independent re-audit
    SPEEDUP_REPRODUCTION_AUDIT.md      # confirms speedup work didn't break fidelity
    RESEARCH_DIRECTION_AUDIT_2026-07-25.md  # strategy/direction review (raises the provenance question)
  research_c/      # Stage C literature foundation
    STAGE_C_LITERATURE_MASTER.md       # consolidated, verified literature master
    STAGE_C_DEEP_RESEARCH_2026-07-25.md, STAGE_C_LITERATURE_REAUDIT_2026-07-25.md
    privacy_compliant_fedgraa_redesign.md   # unbuilt "FedGRAA-v2" redesign (Stage C candidate)
    literature_review_and_publication_strategy.md, research_followup.md,
    FedGRAA_Deep_Research_Compendium.md, sagitra_report.md
  paper/           # the target manuscript + a simulated peer-review panel
    PAKDD_Final_Version_Submit_v3.md / .pdf
    FedGRAA_Simulated_Peer_Review.md
  archive/         # superseded/historical docs — see archive/README.md, do not treat as current
understanding/     # didactic walkthrough series (00_START_HERE.md ... 10_SOTA...)
run_experiments.py # main entry point (--results-dir persists sweep JSON)
requirements.txt
```

---

## Where to start reading

| If you want… | Read |
|---|---|
| The whole project in one file | [docs/current/STAGE_A_MASTER_REPORT.md](docs/current/STAGE_A_MASTER_REPORT.md) then [docs/current/STAGE_B_MASTER_REPORT.md](docs/current/STAGE_B_MASTER_REPORT.md) |
| The paper-section-shaped critique | [docs/current/fedgraa_convergence_defect.md](docs/current/fedgraa_convergence_defect.md) |
| A gentle introduction to FL and this paper | [understanding/00_START_HERE.md](understanding/00_START_HERE.md) |
| What's wrong with the paper itself | [docs/current/PAPER_AMBIGUITY_REPORT.md](docs/current/PAPER_AMBIGUITY_REPORT.md) and [docs/paper/FedGRAA_Simulated_Peer_Review.md](docs/paper/FedGRAA_Simulated_Peer_Review.md) |
| What happens next | [docs/current/publication_roadmap.md](docs/current/publication_roadmap.md) and [docs/audit/RESEARCH_DIRECTION_AUDIT_2026-07-25.md](docs/audit/RESEARCH_DIRECTION_AUDIT_2026-07-25.md) |

> **One open question, recorded rather than assumed.** The target manuscript is
> an anonymized submission draft, and its publication status has not been
> established (see [docs/audit/RESEARCH_DIRECTION_AUDIT_2026-07-25.md](docs/audit/RESEARCH_DIRECTION_AUDIT_2026-07-25.md) D1
> and [docs/current/STAGE_B_MASTER_REPORT.md](docs/current/STAGE_B_MASTER_REPORT.md) §21).
> This affects how the critique should be *framed*, not any measurement in it.
