# Stage B ablation manifest

Maps every Stage B artifact -> exact generating command, seed set, and the
claim it supports, following the same backticked-path convention as
[`../README.md`](../README.md) (Stage A's manifest). Written against
baseline commit `a17a443` (same tree `stage_b_plan.md` was written
against). `python scripts/check_stage_b_complete.py` validates every
backticked `results/`-prefixed path referenced below exists on disk and is
git-tracked, and independently validates the full 4-arm x 10-seed x
2-dataset grid and the consolidated table's internal structure.

**Status: complete.** The 10-seed, 4-arm, 2-dataset factorial campaign (T3)
has run to completion, the 4 baseline seed-sweeps have been regenerated at
the 60-round ablation horizon, and the consolidated tables/figures are
built and embedded in
[`../../docs/current/fedgraa_convergence_defect.md`](../../docs/current/fedgraa_convergence_defect.md)
§5. `python scripts/check_stage_b_complete.py` exits `0` on this tree. §3
below (formerly "PENDING") now lists the real, git-tracked paths.

---

## 1. Code (not under `results/`, not manifest-checked, listed for completeness)

| Artifact | Task | Tests |
|---|---|---|
| [`../../fedgraa/stats.py`](../../fedgraa/stats.py) | T4 | `tests/test_stats.py` (19 cases: bootstrap CI, paired t, Wilcoxon, Cohen's d, Holm-Bonferroni, all degenerate-cell edges) |
| [`../../scripts/run_ablation_seed_sweep.py`](../../scripts/run_ablation_seed_sweep.py) | T3 driver, §6.2 | `tests/test_run_ablation_seed_sweep.py` (17 cases: filename convention, enumeration, resumability, seed-range parsing) |
| [`../../scripts/build_ablation_table.py`](../../scripts/build_ablation_table.py) | T5, §6.3 | `tests/test_build_ablation_table.py` (13 cases: cell/CI wiring, degenerate comparisons, Holm wiring, factorial effects) |
| [`../../scripts/render_ablation.py`](../../scripts/render_ablation.py) | T6, §6.4 | `tests/test_render_ablation.py` (5 cases: markdown + figure smoke tests) |
| [`../../scripts/check_stage_b_complete.py`](../../scripts/check_stage_b_complete.py) | §7 | run directly (standalone, like Stage A's checker); not part of `pytest tests` — checks artifact *presence and shape* |
| [`../../scripts/audit_stage_b.py`](../../scripts/audit_stage_b.py) | verification | run directly — checks artifact *correctness*: config-vs-arm fidelity on all 80 runs, table-vs-raw per-seed equality, every p-value/effect-size/Holm/factorial figure recomputed against SciPy independently of `fedgraa/stats.py`, all 232 doc-transcribed numbers re-derived, and heavy-tailed-summary detection. Negative-control tested (deliberately injected errors are caught and named) |
| [`../../docs/current/fedgraa_convergence_defect.md`](../../docs/current/fedgraa_convergence_defect.md) | T1/T2/T8 | theory doc; §1-7 all complete, §5 embeds the tables/figures below |

## 2. Existing (Stage A-era) diagnostic JSONs reused as Stage B seed-0/1/2 data

These were generated during Stage A / the original divergence diagnosis and
satisfy specific `(arm, dataset, seed)` cells of the Stage B factorial
without being re-run — `run_ablation_seed_sweep.py`'s resumability logic
recognizes them by the same filename convention.

| Path | Arm | Dataset | Seed | Claim it supports |
|---|---|---|---|---|
| `results/diagnostics/CIC_IDS_2017_literal_r60.json` | literal | CIC-IDS-2017 | 0 | §2's round-1 simplex violation table; factorial "projection off, η′=0.1" corner, seed 0 |
| `results/diagnostics/UNSW_NB15_literal_r60.json` | literal | UNSW-NB15 | 0 | same, UNSW |
| `results/diagnostics/CIC_IDS_2017_projected_r60.json` | projection_alone | CIC-IDS-2017 | 0 | factorial "projection on, η′=0.1" corner, seed 0 (§5.1, §6 discussion) |
| `results/diagnostics/CIC_IDS_2017_projected_combined_eta0.01_E5_r60.json` | both | CIC-IDS-2017 | 0 | factorial "projection on, η′=0.01" corner (the fix), seed 0 |
| `results/diagnostics/CIC_IDS_2017_projected_combined_eta0.01_E5_seed1_r60.json` | both | CIC-IDS-2017 | 1 | same, seed 1 |
| `results/diagnostics/CIC_IDS_2017_projected_combined_eta0.01_E5_seed2_r60.json` | both | CIC-IDS-2017 | 2 | same, seed 2 |
| `results/diagnostics/UNSW_NB15_projected_combined_eta0.01_E5_r60.json` | both | UNSW-NB15 | 0 | same, UNSW seed 0 |
| `results/diagnostics/UNSW_NB15_projected_combined_eta0.01_E5_seed1_r60.json` | both | UNSW-NB15 | 1 | same, UNSW seed 1 |
| `results/diagnostics/UNSW_NB15_projected_combined_eta0.01_E5_seed2_r60.json` | both | UNSW-NB15 | 2 | same, UNSW seed 2 |
| `results/diagnostics/CIC_IDS_2017_literal_real_smalleta_r60.json` | *(secondary, non-factorial)* | CIC-IDS-2017 | 0 | §6's η′=0.001/E′=1 over-correction row — documented, not a factorial cell (stage_b_plan.md §2 caveat (a)) |
| `results/diagnostics/seed_sweep_CIC_IDS_2017_fedgraa_combined_fix.json` | *(Stage A aggregate, "both" only, 3 seeds)* | CIC-IDS-2017 | 0-2 | superseded by the Stage B `both`-arm 10-seed cell in `CIC_IDS_2017_ablation_table.json` (§3) |
| `results/diagnostics/seed_sweep_UNSW_NB15_fedgraa_combined_fix.json` | *(Stage A aggregate, "both" only, 3 seeds)* | UNSW-NB15 | 0-2 | same, UNSW |

**Not reusable as-is (documented, not silently ignored):** `results/diagnostics/seed_sweep_CIC_IDS_2017_fedavg.json` and `results/diagnostics/seed_sweep_CIC_IDS_2017_static.json` are genuine FedAvg/Static seed-variance bands (3 seeds) but at **100 rounds**, not the Stage B ablation's 60-round horizon — stage_b_plan.md §3 requires re-running these at `--rounds 60` into the ablation results directory (§3 below), which Task T3 does via `run_seed_sweep.py --out ...`. `results/diagnostics/CIC_IDS_2017_projected_combined_eta0.001_E1_r60.json`, `results/diagnostics/CIC_IDS_2017_projected_combined_eta0.001_E5_r60.json`, and `results/diagnostics/CIC_IDS_2017_projected_combined_eta0.01_E1_r60.json` are earlier hyperparameter-grid points from the original diagnosis (stage_b_plan.md §3's Step 1 grid) — informative context, not part of the 4-arm factorial or its file-naming convention.

## 3. The completed 10-seed factorial campaign (Task T3)

Every per-seed diagnostic-JSON row below is a **shorthand for a file
family** (plain text, not backticked, not manifest-checked) rather than
enumerating all 4 arms × 10 seeds × 2 datasets (~80 near-identical rows);
the arm-level completeness of that grid — every arm has exactly 10 seeds
on both datasets — is what `check_stage_b_complete.py`'s per-arm-seed
check (§7 of `stage_b_plan.md`) verifies directly instead. The 8
consolidated/reference artifacts (baselines, tables, figures) are listed
with their exact, git-tracked, backticked paths — these *are*
manifest-checked.

| Target (file family, plain text) | Command | Seeds | Claim it supports |
|---|---|---|---|
| results/diagnostics/CIC_IDS_2017_literal[_seed*N*]_r60.json | `python scripts/run_ablation_seed_sweep.py --dataset CIC-IDS-2017 --seeds 0-9 --workers 3` | 0-9 (all present) | literal arm, CIC, all 10 seeds |
| results/diagnostics/CIC_IDS_2017_projected[_seed*N*]_r60.json | same driver | 0-9 (all present) | projection-alone arm, CIC |
| results/diagnostics/CIC_IDS_2017_literal_etaonly_eta0.01_E5[_seed*N*]_r60.json | same driver | 0-9 (all present) | canonical η′-alone arm, CIC |
| results/diagnostics/CIC_IDS_2017_projected_combined_eta0.01_E5[_seed*N*]_r60.json | same driver | 0-9 (all present; 0-2 reused from §2) | both arm, CIC |
| results/diagnostics/UNSW_NB15_{literal,projected,literal_etaonly_eta0.01_E5}[_seed*N*]_r60.json | `python scripts/run_ablation_seed_sweep.py --dataset UNSW-NB15 --seeds 0-9 --workers 3` | 0-9 (all present) | literal/projection-alone/eta-alone arms, UNSW |
| results/diagnostics/UNSW_NB15_projected_combined_eta0.01_E5[_seed*N*]_r60.json | same driver | 0-9 (all present; 0-2 reused from §2) | both arm, UNSW |
| `results/ablation/seed_sweep_CIC_IDS_2017_fedavg_r60.json` | `python scripts/run_seed_sweep.py --dataset CIC-IDS-2017 --method fedavg --seeds 0 1 2 3 4 5 6 7 8 9 --rounds 60 --eval-every 1 --out results/ablation/seed_sweep_CIC_IDS_2017_fedavg_r60.json` | 0-9 | FedAvg reference, CIC, 60-round horizon (regenerated — the existing 100-round band was not reusable, §2) |
| `results/ablation/seed_sweep_CIC_IDS_2017_static_r60.json` | same, `--method static` | 0-9 | Static reference, CIC |
| `results/ablation/seed_sweep_UNSW_NB15_fedavg_r60.json` | same, dataset UNSW | 0-9 | FedAvg reference, UNSW |
| `results/ablation/seed_sweep_UNSW_NB15_static_r60.json` | same, dataset UNSW | 0-9 | Static reference, UNSW |
| `results/ablation/CIC_IDS_2017_ablation_table.json` | `python scripts/build_ablation_table.py --dataset CIC-IDS-2017` | 0-9 | the consolidated table: all arm x γ cells, 5 comparisons, factorial read — CIC (`fedgraa_convergence_defect.md` §5.1) |
| `results/ablation/UNSW_NB15_ablation_table.json` | `python scripts/build_ablation_table.py --dataset UNSW-NB15` | 0-9 | same, UNSW (§5.2) |
| `results/ablation/CIC_IDS_2017_ablation_table.md` | `python scripts/render_ablation.py --dataset CIC-IDS-2017` | — | rendered table, embedded in `fedgraa_convergence_defect.md` §5.1 |
| `results/ablation/CIC_IDS_2017_ablation.png` | same | — | grouped bar chart, rare-acc by arm x γ, 95% CI, embedded §5.1 |
| `results/ablation/UNSW_NB15_ablation_table.md` | `python scripts/render_ablation.py --dataset UNSW-NB15` | — | rendered table, embedded §5.2 |
| `results/ablation/UNSW_NB15_ablation.png` | same | — | rendered figure, embedded §5.2 |
