# Results manifest — Stage A reproduction artifacts

This file maps every committed artifact under `results/` to the paper artifact
it reproduces, the exact command that generated it, the git SHA of the code
that ran, the seed(s) used, and any honest divergence from the paper. It is
the reproducibility ledger required by [`../docs/current/stage_a_plan.md`](../docs/current/stage_a_plan.md)
§6.5, and the first draft of the manuscript's reproducibility appendix
([`publication_roadmap.md`](../docs/current/publication_roadmap.md) §4).

Generated against baseline commit `be4ef03` (Stage A completion — this file
and the artifacts it describes are committed together).

## Environment snapshot

| Component | Version |
|---|---|
| Python | 3.13.14 |
| TensorFlow | 2.21.0 |
| NumPy | 2.5.1 |
| Keras | 3.15.0 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.3 |
| matplotlib | 3.11.1 |
| pytest | 9.1.1 |

Full `pip freeze` output (52 packages) is reproducible via
`.venv313/Scripts/python.exe -m pip freeze` against the committed
environment; not embedded in full here to keep this file focused on the
provenance table below.

## Round-0 convention (stage_a_plan.md §3)

All four base-sweep artifacts below (CIC γ-sweep, CIC n-sweep, UNSW γ-sweep,
UNSW n-sweep) use the **strict / ditto** convention: `--round-zero-zero`
forces round-0 accuracy to 0.00 for every method, matching the paper's
Figure-1 table and `fedgraa/config.py`'s `round_zero_zero=True` fidelity
default (gap C-4). This **replaces** the previously-committed CIC γ-sweep,
which had used real (~9%) round-0 accuracy and was self-contradictory with
the config default. The diagnostic runs (combined-fix seed bands) are
diagnostic-only, use their own 60-round/eval-every-1 convention, and are
**not** subject to this decision — they were never governed by
`round_zero_zero` in the first place.

## Reproduction manifest

| Paper artifact | Dataset | File(s) | Command | Seed(s) | Paper-claimed | Reproduced | Notes |
|---|---|---|---|---|---|---|---|
| Table 1 (class stats) | CIC | `fedgraa/config.py::CIC_IDS_2017` | — | — | — | — | See [`../docs/audit/data_provenance_audit.md`](../docs/audit/data_provenance_audit.md) |
| Table 2 (class stats) | UNSW | `fedgraa/config.py::UNSW_NB15` | — | — | — | — | See [`../docs/audit/data_provenance_audit.md`](../docs/audit/data_provenance_audit.md) |
| Figure 1 (γ-sweep, n=3) | CIC | `results/CIC-IDS-2017_gamma_sweep.json`, `results/figures/CIC-IDS-2017_figure1.png` | `python run_experiments.py --data-dir data --cache-dir .cache --datasets CIC-IDS-2017 --sweep gamma --parallel --workers 5 --round-zero-zero --rounds 200 --eval-every 20 --results-dir results --plot-dir results/figures` | 0 | FedGRAA γ=1 rare-acc = 88.88% | FedAvg 61.00%/0.00%, Static 52.29%/91.67%, FedGRAA γ=0 62.53%/0.00%, γ=0.5 18.56%/0.00%, γ=1 17.38%/0.00% (all/rare) | **Honest divergence.** Literal 200-round FedGRAA collapses (composite-loss blowup) — this is the instability documented in `docs/fedgraa_divergence_diagnosis.md`, not a bug in this run. The recovered numbers live in the combined-fix diagnostic band below. |
| Figure 1 (γ-sweep, n=3) | UNSW | `results/UNSW-NB15_gamma_sweep.json`, `results/figures/UNSW-NB15_figure1.png` | `python run_experiments.py --data-dir data --cache-dir .cache --datasets UNSW-NB15 --sweep gamma --parallel --round-zero-zero --rounds 200 --eval-every 20 --results-dir results --plot-dir results/figures` | 0 | FedGRAA γ=1 rare-acc = 21.09% | FedAvg 49.95%/0.00%, Static 54.93%/31.48%, FedGRAA γ=0 49.58%/0.00%, γ=0.5 43.30%/25.13%, γ=1 38.00%/33.33% (all/rare) | **Honest divergence**, milder than CIC (literal UNSW instability blowup 7.6× vs CIC's 193×, per diagnosis §9) — literal FedGRAA γ=1 here actually *exceeds* Static, unlike CIC. |
| Figures 2–3 (n-sweep) | CIC | `results/CIC-IDS-2017_n_sweep.json`, `results/figures/CIC-IDS-2017_figure2_3.png` | `python run_experiments.py --data-dir data --cache-dir .cache --datasets CIC-IDS-2017 --sweep n --parallel --workers 3 --round-zero-zero --rounds 200 --eval-every 20 --results-dir results --plot-dir results/figures` | 0 | — (paper doesn't disaggregate n-sweep headline numbers) | n=3: FedAvg 0.00%/Static 91.67%/g0 0.00%/g0.5 0.00%/g1 0.00% rare-acc. n=5: FedAvg 0.00%/Static 74.43%/g0 0.00%/g0.5 0.00%/g1 0.00%. n=7: FedAvg 0.00%/Static 53.16%/g0 0.00%/g0.5 15.64%/g1 19.02% | **Honest divergence.** As with Figure 1, literal 200-round FedGRAA collapses under the documented composite-loss instability at every n; Static Reparam. is the only baseline retaining rare-class accuracy at this horizon. Figure renders all 5 curves/panel (regression-tested against the §4 plot bug, `tests/test_plot.py::test_plot_n_sweep_includes_fedgraa_curves`). |
| Figures 2–3 (n-sweep) | UNSW | `results/UNSW-NB15_n_sweep.json`, `results/figures/UNSW-NB15_figure2_3.png` | `python run_experiments.py --data-dir data --cache-dir .cache --datasets UNSW-NB15 --sweep n --parallel --round-zero-zero --rounds 200 --eval-every 20 --results-dir results --plot-dir results/figures` | 0 | — | n=3: FedAvg 0.00%/Static 31.48%/g0 0.00%/g0.5 25.13%/g1 33.33% rare-acc. n=5 **and** n=7 (identical): FedAvg 0.00%/Static 18.89%/g0 2.49%/g0.5 7.56%/g1 20.95% rare-acc | **Honest divergence, documented, not a bug:** UNSW-NB15 has only 5 genuinely-rare attack classes (`rare_client_ids(7)` truncates to 5), so the n=7 panel is byte-identical to n=5 — confirmed above. |
| Divergence diagnosis (literal + projected + combined) | CIC | `results/diagnostics/` (files prefixed `CIC_IDS_2017_`) + figures | see `docs/fedgraa_divergence_diagnosis.md` §6–8 | 0 | — | — | Pre-existing, unchanged by Stage A. |
| Divergence diagnosis (literal) | UNSW | `results/diagnostics/UNSW_NB15_literal_r60.json` + figures | see `docs/fedgraa_divergence_diagnosis.md` §9 | 0 | — | — | Pre-existing, unchanged by Stage A. |
| Combined-fix seed band (proj + η′=0.01, E′=5) | CIC | `results/diagnostics/seed_sweep_CIC_IDS_2017_fedgraa_combined_fix.json` | `python scripts/seed_sweep_combined_fix.py --dataset CIC-IDS-2017` (aggregates pre-existing per-seed diagnostics) | 0,1,2 | FedGRAA γ=1 rare-acc = 88.88% | γ=0: 63.54%±1.35% / 72.22%±10.39% (all/rare). γ=0.5: 56.18%±0.63% / 91.67%±0.00%. γ=1: 47.67%±3.71% / 94.44%±3.93% | Pre-existing band; re-verified byte-identical after generalizing the aggregator script (§6.3 regression check). 60-round diagnostic horizon, not the 200-round base sweep — reported as a distinct, diagnostic-only result per the no-silent-patching mandate (§9). |
| Combined-fix seed band | UNSW | `results/diagnostics/seed_sweep_UNSW_NB15_fedgraa_combined_fix.json` | `python scripts/diagnose_fedgraa.py --dataset UNSW-NB15 --gammas 0 0.5 1.0 --rounds 60 --eval-every 1 --project --eta-prime 0.01 --weight-epochs 5 --seed {0,1,2} --tag combined_eta0.01_E5[_seed{1,2}]` then `python scripts/seed_sweep_combined_fix.py --dataset UNSW-NB15` | 0,1,2 | FedGRAA γ=1 rare-acc = 21.09% | γ=0: 53.59%±3.07% / 0.00%±0.00% (all/rare). γ=0.5: 52.16%±0.98% / 23.44%±9.60%. γ=1: 52.29%±0.18% / 74.79%±1.99% | **New in Stage A.** Combined fix drives blowup toward 1.0 and lifts rare-acc substantially, but the gain is milder than CIC's (γ=1: 74.79% vs CIC's 94.44%) — a legitimate, publishable finding that fix severity/efficacy is dataset-dependent, not a failure (per §5 Task B3 note). |
| Combined-fix figures | UNSW | `results/figures/diagnosis_unsw_combined_fix_composite_loss.png`, `_weights_g0p0/g0p5/g1p0.png`, `_summary.md` | `python scripts/render_diagnostics.py --in results/diagnostics/UNSW_NB15_projected_combined_eta0.01_E5_r60.json --out-prefix results/figures/diagnosis_unsw_combined_fix` | 0 | — | — | New in Stage A. |

## Definition of done (stage_a_plan.md §0)

- [x] Every paper artifact in the matrix has a committed counterpart.
- [x] Every figure renders all curves (n-sweep plot bug fixed + regression-tested).
- [x] Combined-fix seed-variance band exists for both datasets.
- [x] Every honest divergence documented in-file and here.
- [x] `python scripts/check_stage_a_complete.py` exits 0.
- [x] `pytest tests` green.
- [x] This manifest exists.
