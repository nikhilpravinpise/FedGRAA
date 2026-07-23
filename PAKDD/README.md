# FedGRAA — Strict Keras Reproduction

A complete, from-scratch Keras reproduction of the paper **"Federated
Gradient-Refined Adaptive Averaging for Intrusion Detection Systems"
(FedGRAA)**.

The implementation now uses **TensorFlow/Keras** for the network described in
Section 5: `input → Dense(200, tanh) → Dense(100, tanh) → Dense(50, tanh) →
Dense(n_classes, softmax)`, trained with sparse categorical cross-entropy and
Adam.  Every ambiguous design choice that was previously hidden behind a flag
has been removed; the remaining ambiguities and apparent paper errors are
documented in [docs/strict_reproduction_report.md](docs/strict_reproduction_report.md).

---

## Environment

- **Python 3.9 – 3.13**.  TensorFlow 2.21.0 provides wheels for Python 3.13.
- **RAM: 4 GB minimum, 8 GB recommended** when loading the full CIC-IDS-2017
  dataset (~2.8 M rows).  Use `--cache-dir .cache` to avoid re-reading the CSVs.
- No GPU required.  Training is CPU-only.

### Install

```powershell
# Create a Python 3.13 environment, then:
pip install -r requirements.txt
```

Dependencies: `tensorflow`, `numpy`, `pandas`, `scikit-learn`, `pytest`, and
optional `matplotlib` for plotting.

### Datasets

| Dataset | Status | Location |
|---|---|---|
| **CIC-IDS-2017** | ✅ Already in repo | `data/CIC-IDS-2017/` (8 CSV files) |
| **UNSW-NB15** | ⬇️ Needs download | See [UNSW-NB15 setup](#unsw-nb15) below |

---

## Data setup

### CIC-IDS-2017 (already present)

The 8 CSV files are already in `data/CIC-IDS-2017/`.  The first run reads all
rows (~95 s) and saves an `.npy` cache; subsequent runs load the cache in under
3 s.

```powershell
# First run: reads CSVs, saves cache
python run_experiments.py --data-dir data/CIC-IDS-2017 --cache-dir .cache --datasets CIC-IDS-2017

# Subsequent runs: load cache
python run_experiments.py --data-dir data/CIC-IDS-2017 --cache-dir .cache --datasets CIC-IDS-2017
```

### UNSW-NB15

Download the CSV files from the
[official UNSW page](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
and place them in `data/UNSW-NB15/`.  The loader will auto-detect the label
column.  If the directory is empty, the loader can attempt to download the files
automatically.

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
| `--baselines` | Include FedProx / FedOpt / FedSND (related work only). |
| `--parallel` | Run sweep configs concurrently via `ProcessPoolExecutor`. |
| `--results-dir DIR` | Persist raw sweep results to `<DIR>/<dataset>_gamma_sweep.json` / `_n_sweep.json`. |
| `--plot-dir DIR` | Save Figure 1 / 2-3 PNGs here. |
| `--checkpoint-dir DIR` | Save/resume training checkpoints. |

### Recommended workflows

**Quickest sanity check (synthetic data, ~1–2 min per method):**
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

---

## Actual measured results (real CIC-IDS-2017, 200 rounds, n=3)

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
minority classes, beating both baselines). The gap has been diagnosed, not
just observed: FedGRAA's aggregation weights `w^t` leave the probability
simplex in round 1 at every `γ` because Algorithm 2 as printed has no
projection step, driving the composite loss to blow up 2.7×–193× and the
weight learner to destabilize the global model. See
[`docs/fedgraa_divergence_diagnosis.md`](docs/fedgraa_divergence_diagnosis.md)
for the full measurement-backed root-cause analysis, including a diagnostic
ablation and a real-data hyperparameter intervention that confirm the
mechanism. **FedAvg's 0% is structural and seed-independent** (verified
across 3 seeds); Static Reparam's 91.7% is real and reproducible (88.9% ± 3.9%
across 3 seeds), not a lucky single run.

---

## Tests

Run the full suite before pushing:

```powershell
.venv313\Scripts\python.exe -m pytest tests --tb=short
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
| Ambiguities / paper errors | [docs/strict_reproduction_report.md](docs/strict_reproduction_report.md) |

### Core math

The aggregated parameters are `P = Σ_k w_k P_k`, so `∂P/∂w_k = P_k`.  For the
composite validation loss `L'(P) = (1−γ)·min(L_h, L_m) + γ·L_u`:

```
∂L'/∂w_k = ⟨ ∇_P L'(P), P_k ⟩
```

FedGRAA runs this gradient through Adam for `E'=5` epochs per round, with no
simplex projection (Algorithm 2 does not contain one).

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
  baselines.py     # FedProx / FedOpt / FedSND (related work)
  plot.py          # matplotlib figures
  __main__.py      # CLI: python -m fedgraa ...
data/
  CIC-IDS-2017/    # 8 CSV files (already present)
  UNSW-NB15/       # downloaded CSVs (already present)
.cache/            # auto-created NPY cache (git-ignored)
scripts/
  diagnose_fedgraa.py    # per-round divergence instrumentation + simplex-projection ablation
  sensitivity_sweep.py   # eta'/E' sweep
  render_diagnostics.py  # figures + summary tables from diagnostic JSON
  run_seed_sweep.py      # seed-variance bands
  real_reproduction.py   # canonical real-data gamma sweep
  archive/gpu/           # explored-and-rejected GPU benchmark (25% slower than CPU)
results/
  *_gamma_sweep.json, *_n_sweep.json  # persisted sweep results (--results-dir)
  figures/               # Figure 1-3 PNGs + diagnostic plots
  diagnostics/           # raw per-round divergence/sensitivity/seed-sweep data
tests/             # 79 unit + gradient + e2e + diagnostic tests
docs/
  strict_reproduction_report.md
  fedgraa_divergence_diagnosis.md   # root-cause analysis of the FedGRAA reproduction gap
  GPU_DECISION_AND_NEXT_STEPS.md
  PAKDD_Final_Version_Submit (3).md
run_experiments.py # main entry point (--results-dir persists sweep JSON)
requirements.txt
```

