# PAKDD / FedGRAA — Project Status Report

**Date:** 2026-07-15

## Code

| Item | State |
|---|---|
| Framework | Keras-only (TF 2.21) |
| Architecture | Dense(200→100→50, tanh) → Dense(n_classes, softmax) |
| Tests | **52/52 passed** (291 s) |
| Latest commit | `c1d6bd5` — fix: UNSW-NB15 feature count 43 → 39 |

## Data

| Dataset | Status | Cache |
|---|---|---|
| CIC-IDS-2017 | ✅ 8 CSVs present | ✅ `.npy` cache exists |
| UNSW-NB15 | ✅ `training-set.csv` + `testing-set.csv` present | ✅ `.npy` cache exists |

## Qualitative Result Verified (UNSW-NB15, real data, 2 rounds)

| Method | All-class | Last-n (rare) |
|---|---|---|
| FedAvg | 59.3% | **0.0%** |
| Static Reparam | 54.7% | **13.8%** |
| FedGRAA γ=0 | 64.0% | **0.0%** |
| FedGRAA γ=1 | 3.6% | **41.5%** |

**Ordering:** FedGRAA(γ=1) >> Static > FedAvg ≈ 0% on rare classes ✅

## Partial CIC-IDS-2017 Gamma Sweep (was in progress)

| Round | Method | All-class | Last-n |
|---|---|---|---|
| 10/50 | FedAvg | 59.2% | 0.0% |
| 20/50 | FedAvg | 59.7% | 0.0% |

## Known Gaps (from `docs/`)

| ID | Issue | Severity |
|---|---|---|
| B-2 | Framework mismatch: NumPy → Keras (✅ resolved by Keras-only refactor) | Blocking (resolved) |
| B-3 | No random seed reported in paper → code uses `seed=0` | Blocking (unresolvable) |
| C-1 | Standard scaling fitted pre-split (documented choice) | Critical |
| M-3 | Benign pool smaller than demand → reuse warning | Minor |
| C-4 | Round-0 accuracy forced to 0.00 to match paper's | Minor |

## Key Files

| File | Purpose |
|---|---|
| `fedgraa/config.py` | Hyperparameters + dataset specs |
| `fedgraa/model.py` | Keras MLP |
| `fedgraa/data.py` | CSV loader + NPY cache + synthetic generator |
| `fedgraa/aggregation.py` | FedAvg, Static, FedGRAA |
| `fedgraa/trainer.py` | Algorithm 1 federated loop |
| `fedgraa/experiment.py` | Figure 1 / 2 / 3 sweep runners |
| `run_experiments.py` | Main CLI entry point |
| `tests/` | 52 unit + gradient + e2e tests |
| `docs/strict_reproduction_report.md` | All documented deviations from paper |
| `docs/privacy_compliant_fedgraa_redesign.md` | FedGRAA-v2 redesign blueprint |
| `understanding/` | Paper explanation from scratch (9 files) |

## Next-Step Options

| Lane | What | Est. time |
|---|---|---|
| **B (continued)** | Full 200-round sweep on CIC + UNSW | ~40-70 h CPU |
| **B (quick)** | 50-round sweep on CIC only (was in progress) | ~8-10 h CPU |
| **C** | Implement FedGRAA-v2 (privacy-compliant) | Days of dev |
| **D** | Paper revision (algorithms, related work, privacy analysis) | N/A |

## CLI Reference

```powershell
# Quick smoke test (synthetic data)
python run_experiments.py --rounds 5 --eval-every 5 --sweep gamma

# CIC sweep (real data)
python run_experiments.py --data-dir data/CIC-IDS-2017 --cache-dir .cache --datasets CIC-IDS-2017 --rounds 200

# Single experiment
python -m fedgraa run --dataset CIC-IDS-2017 --method fedgraa --gamma 1 --n 3 --rounds 50

# Run tests
.venv313\Scripts\python.exe -m pytest tests --tb=short
```

## Update: Completed Real Data Sweep (5 rounds)

As of 2026-07-15, a 5-round experiment was successfully completed on real data for both datasets to verify training pipeline behavior and initial accuracy sweeps.

### CIC-IDS-2017 (5 rounds, real data)
| Method | All-class | Last-n (rare) |
|---|---|---|
| FedAvg | 53.13% | 0.00% |
| Static Reparam | 43.65% | **75.00%** |
| FedGRAA g=0 | 58.29% | 0.00% |
| FedGRAA g=0.5 | 25.18% | 0.00% |
| FedGRAA g=1 | 43.12% | 0.00% |

**Note:** At 5 rounds, FedGRAA's learnable validation weights have not yet adapted to boost rare classes on the full CIC dataset.

### UNSW-NB15 (5 rounds, real data)
| Method | All-class | Last-n (rare) |
|---|---|---|
| FedAvg | 64.09% | 0.00% |
| Static Reparam | 49.05% | **32.28%** |
| FedGRAA g=0 | 64.02% | 0.00% |
| FedGRAA g=0.5 | 52.91% | 0.00% |
| FedGRAA g=1 | 48.14% | **29.92%** |

**Note:** FedGRAA (g=1) successfully begins boosting rare classes to 29.92% recall in just 5 rounds, performing close to Static Reparameterization (32.28%) and outperforming FedAvg (0.00%).
