# FedGRAA — Project Status Report

> **⚠️ SUPERSEDED — historical snapshot, not current state (2026-07-25).** Dated
> 2026-07-23, this document predates the Stage A and Stage B campaigns and is
> now two stages stale: it cites **79/79 tests** (now **142/142**), presents
> the `η′=0.001` over-correction as the headline intervention (superseded by
> the validated projection + η′=0.01 combined fix), and frames the divergence
> diagnosis as the project frontier. **For current state, read:**
> [`docs/current/STAGE_A_MASTER_REPORT.md`](docs/current/STAGE_A_MASTER_REPORT.md)
> and [`docs/current/STAGE_B_MASTER_REPORT.md`](docs/current/STAGE_B_MASTER_REPORT.md),
> or [`docs/current/publication_roadmap.md`](docs/current/publication_roadmap.md)
> for the single entry point. Kept for historical record. See
> [`docs/audit/AUDIT_REMEDIATION_PLAN_2026-07-25.md`](docs/audit/AUDIT_REMEDIATION_PLAN_2026-07-25.md)
> (R7).

---

**Date:** 2026-07-23 (Divergence diagnosis complete — see update below)

---

## Update 2026-07-23: FedGRAA's divergence is now diagnosed, not just observed

The "reproduction gap" noted below (FedGRAA getting 0% rare-class accuracy)
has been root-caused. Full writeup:
[`docs/fedgraa_divergence_diagnosis.md`](docs/current/fedgraa_divergence_diagnosis.md).

**Short version:** the aggregation-weight vector `w^t` leaves the probability
simplex in round 1, at every `γ`, because Algorithm 2 as printed has no
projection step (a deliberate strict-reproduction choice, not a bug — see
`docs/strict_reproduction_report.md` §2.2). This causes the composite loss
`L'(w^t)` to blow up 2.7×–193× depending on `γ`, directly violating the
paper's own Lemma 1. A diagnostic-only simplex-projection ablation confirms
the mechanism (blowup collapses to 1.0×–4.3×) and a real-data hyperparameter
intervention (`η'=0.001` vs. the paper's `0.1`) eliminates the blowup entirely
and recovers 75-100% rare-class accuracy for most of a 60-round run — at the
cost of collapsing overall accuracy instead, since `γ=1`'s objective has
nothing pushing back against overweighting the rare clients.
>
> **Correction, added 2026-07-26 (this sentence is preserved verbatim above as
> the historical record; do not edit it further):**
> [`docs/current/STAGE_A_MASTER_REPORT.md` §11.1](docs/current/STAGE_A_MASTER_REPORT.md)
> identified that this sentence conflates two distinct, later-disambiguated
> experiments: the single-lever `η′=0.001` intervention *alone* (real, but
> collapses overall accuracy toward 0.8%), and the **validated combined fix**
> — simplex projection *and* moderate `η′=0.01` together — which recovers
> 72–94% mean rare-accuracy *without* collapsing overall accuracy
> (`fedgraa_convergence_defect.md` §5.7 Step 3). The combined fix, not the
> `η′=0.001`-alone reading above, is the project's current recommendation.
> This correction was itself flagged by three prior audit passes and not
> applied until now — see
> [`docs/audit/INDEPENDENT_FULL_AUDIT_2026-07-26.md`](docs/audit/INDEPENDENT_FULL_AUDIT_2026-07-26.md) §4.3.

UNSW-NB15's and
CIC-IDS-2017's cached datasets were also independently verified to be
genuine CSV-derived data, not synthetic. Also new: canonical sweep results
are now persisted to [`results/`](results/) instead of existing only as
stdout, and seed-variance sweeps (`results/diagnostics/seed_sweep_*.json`)
show FedAvg's rare-class failure is exact and seed-independent (0.00% ± 0.00%
across 3 seeds) while Static Reparam's advantage is real (88.9% ± 3.9%).

---

## Code

| Item | State |
|---|---|
| Framework | Keras-only (TF 2.21) |
| Architecture | Dense(200→100→50, tanh) → Dense(n_classes, softmax) |
| Tests | **79/79 passed** — 14 new tests added for the divergence diagnosis + result persistence (2026-07-23) |
| Latest commit | `71839ee` — docs: audit data provenance — CIC-IDS-2017 & UNSW-NB15 verified genuine |
| V2 Shipped | Sweep parallelism, pre-computed indices, no `.numpy()` on unused loss, checkpoint/resume, TF threading tune |
| Bugs fixed (this audit) | 3: parallel sweep tuple unpacking, checkpoint TF 2.21 API incompatibility, resume_or_init dead code |

## Real Data Gamma Sweep — CIC-IDS-2017 (200 rounds, n=3)

Completed 2026-07-22. **Sequential: 68 min. Parallel (5 workers): 20 min (3.3×).**

| Method | All-class acc | Last-n (rare) acc |
|---|---|---|
| FedAvg | 61.00% | **0.00%** |
| Static Reparam | 52.29% | **91.67%** |
| FedGRAA γ=0 | 62.53% | **0.00%** |
| FedGRAA γ=0.5 | 18.56% | **0.00%** |
| FedGRAA γ=1.0 | 17.38% | **0.00%** |

**Key finding:** FedGRAA weight learning does not converge on CIC-IDS-2017 with default hyperparameters (eta_prime=0.1, weight_epochs=5). The algorithm oscillates and composite loss diverges. This is a reproduction gap, not a code bug. Static Reparam (inverse-size weighting) achieves 91.67% rare-class accuracy without any learning. **This is now fully root-caused, not just observed — see [`docs/fedgraa_divergence_diagnosis.md`](docs/current/fedgraa_divergence_diagnosis.md).** The aggregation weights leave the probability simplex in round 1 at every γ (measured directly), the composite loss blows up 2.7×-193× as a direct result, and a diagnostic ablation + a real-data hyperparameter intervention both confirm the mechanism.

## GPU Report: RTX 5050 via WSL2

| Metric | CPU (native) | GPU (WSL2) | Verdict |
|---|---|---|---|
| Per-step | **0.70 ms** | 0.87 ms | **GPU 25% slower** |
| XLA compile | **0.66 s** | 3.45 s (PTX JIT) | CPU faster |
| Tests passing | **65/65** | 62/65 (3 FP32 precision fails) | CPU deterministic |

**GPU rejected for this project.** Small model (78→200→100→50→15, batch=64) means kernel launch overhead dominates. CPU + oneDNN + XLA is the optimal path.

## Verification Results

| Claim | Documented | Actual | Status |
|---|---|---|---|
| Tests passing | 55/55 | **79/79** | ✅ More tests added |
| Per-step time | 1.50 ms | **0.70 ms** | ✅ Code faster than claimed |
| Parallel speedup | 3-5× | **3.3×** | ✅ Within range |
| GPU speedup | 5-10× | **0.8× (slowdown)** | ❌ GPU rejected |

## Key Files

| File | Purpose |
|---|---|
| `fedgraa/config.py` | Hyperparameters + dataset specs |
| `fedgraa/model.py` | Keras MLP + fused XLA step |
| `fedgraa/aggregation.py` | FedAvg, Static, FedGRAA |
| `fedgraa/trainer.py` | Algorithm 1 federated loop + checkpoint/resume |
| `fedgraa/experiment.py` | Figure 1 / 2 / 3 sweep runners + parallel executor |
| `run_experiments.py` | Main CLI entry point (`--results-dir` persists sweep JSON, `--plot-dir` saves figures) |
| `tests/` | 79 tests (unit + gradient + e2e + v2 features + divergence diagnosis) |
| `docs/speedup_plan_v2.md` | Final audited speedup plan |
| `docs/strict_reproduction_report.md` | All documented deviations from paper |
| `docs/fedgraa_divergence_diagnosis.md` | Root-cause diagnosis of FedGRAA's 0% rare-class accuracy |
| `scripts/diagnose_fedgraa.py` | Per-round divergence instrumentation + simplex-projection ablation |
| `scripts/sensitivity_sweep.py` | eta'/E' sensitivity sweep |
| `scripts/run_seed_sweep.py` | Seed-variance bands |
| `results/` | Persisted sweep JSON + figures + raw diagnostic data |
| `scripts/archive/gpu/` | WSL2 + GPU setup + benchmark (explored, rejected — 25% slower; kept as evidence) |

## CLI Reference

```powershell
# Full reproduction (CIC-IDS-2017 gamma sweep, parallel, persisted to results/)
python run_experiments.py --data-dir data --cache-dir .cache --parallel --rounds 200 --results-dir results --plot-dir results/figures

# Single experiment
python -m fedgraa run --dataset CIC-IDS-2017 --method fedgraa --gamma 1 --n 3 --rounds 200

# With checkpoint/resume
python run_experiments.py --data-dir data --cache-dir .cache --checkpoint-dir .checkpoint

# Divergence diagnosis (per-round composite loss + weight trajectory)
python scripts/diagnose_fedgraa.py --dataset CIC-IDS-2017 --gammas 0 0.5 1.0 --rounds 60 --eval-every 1

# Run tests
.venv313\Scripts\python.exe -m pytest tests --tb=short

# Quick smoke test (synthetic data)
python run_experiments.py --rounds 5 --eval-every 5 --sweep gamma
```
