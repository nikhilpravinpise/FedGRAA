# Stage A Execution Plan — Close Out the Faithful Reproduction

**Parent:** [`publication_roadmap.md`](publication_roadmap.md) §2 (Stage A).
**Status:** ready to execute. All open questions in the roadmap's Stage A bullets are resolved below against the actual code — no design decisions remain.
**Baseline commit:** `9f78529` (this plan was written against this tree).

---

## 0. What "done" means (definition of done)

> **Stage-A goal (verbatim from the roadmap):** *"every table/figure in the original
> paper has a corresponding persisted, reproduced (or honestly-diverging)
> counterpart in `results/`."*

Concretely, Stage A is **done** when all of the following are true and committed:

1. Every paper artifact in the matrix in §2 has a **committed** counterpart file under [`../../results/`](../../results/) (JSON data + PNG figure).
2. Each figure renders **all** curves it is supposed to (this is not currently true — see the `plot_n_sweep` defect in §4).
3. The combined-fix seed-variance band exists for **both** datasets, not just CIC.
4. Every honest divergence (e.g. UNSW-NB15 has no genuine 7th rare class) is **documented in-file and in the manifest**, not silently truncated.
5. A single command — `python scripts/check_stage_a_complete.py` (new, §7) — exits `0`, proving the artifact set is complete on any clean checkout.
6. `pytest tests` is still green (**verified 79/79 at commit `9f78529`, 71 s — this audit ran it**), plus the new tests this plan adds (§6.2, §6.4).
7. A [`../../results/README.md`](../../results/) manifest maps every committed artifact → the exact command, git SHA, seed, and paper artifact it reproduces (§6.5).

Stage A does **not** include: statistical significance testing (t-tests/bootstrap CIs), the consolidated ablation table, the corrected convergence theory, or any new method. Those are Stage B/C and are explicitly out of scope here (§9).

---

## 1. Grounding — what this plan verified in the code

This plan is written against the code, not the roadmap prose. Key facts established by reading the tree:

| Question the roadmap left open | Answer (verified) | Evidence |
|---|---|---|
| Does the n=5/7 sweep need a flag or config change? | **No code change.** `run_experiments.py --sweep n` already runs `ns=(3,5,7) × gammas=(0,0.5,1)` and persists via `save_n_sweep_results`. | [`../../run_experiments.py`](../../run_experiments.py) L179–225; [`../../fedgraa/experiment.py`](../../fedgraa/experiment.py) `run_n_sweep` L323 |
| Are `results/` files committed or just on disk? | **Committed.** `.gitignore` explicitly keeps `results/*.json` and `results/figures/*.png`; only `.npy`/`.checkpoint/` are excluded. So "persist" ⇒ write **and** `git add`. | [`../../.gitignore`](../../.gitignore) L24–29 |
| Is the UNSW γ-sweep base reproduction runnable now? | **Yes**, unchanged CLI: `--datasets UNSW-NB15 --sweep gamma`. Only the *divergence diagnosis* has been run on UNSW so far, never the FedAvg/Static/FedGRAA base sweep. | `results/diagnostics/UNSW_NB15_literal_r60.json` exists; `results/UNSW-NB15_gamma_sweep.json` does not |
| Can the combined-fix seed sweep be reused for UNSW as-is? | **No.** `seed_sweep_combined_fix.py` hardcodes CIC filenames; it needs a one-time generalization (§6.3). | [`../../scripts/seed_sweep_combined_fix.py`](../../scripts/seed_sweep_combined_fix.py) L33–39 |
| Will the n-sweep figure render correctly? | **No — latent bug.** `plot_n_sweep` looks for a bare `"fedgraa"` key that `run_n_sweep` never produces; the figure silently drops all three FedGRAA curves (§4). | [`../../fedgraa/plot.py`](../../fedgraa/plot.py) L158 vs `run_n_sweep` keys `fedgraa_g{g}` |
| Are both datasets present and cached? | CIC: cached for seeds 0,1,2. UNSW: cached for **seed 0 only** — seeds 1,2 will auto-build (~30 s CSV load each) on first use during the UNSW combined-fix sweep. | `.cache/` listing |
| Round-0 convention of the existing CIC artifact? | Real round-0 accuracies (not forced to 0.00) — `--round-zero-zero` was **off**, which **deviates from the strict-repro fidelity default** `round_zero_zero=True`. §3 now treats this as an explicit decision, not a silent default. | `results/CIC-IDS-2017_gamma_sweep.json`: fedavg round-0 all-acc = 0.0896; [`../../fedgraa/config.py`](../../fedgraa/config.py) L67 |

---

## 2. The reproduction matrix — what exists, what's missing

Legend: ✅ done & committed · ❌ missing · ⚠️ done but defective.

| Paper artifact | Dataset | Persisted counterpart (expected path) | Status | Task |
|---|---|---|---|---|
| **Table 1** (CIC class stats) | CIC | `fedgraa/config.py::CIC_IDS_2017` + [`../audit/data_provenance_audit.md`](../audit/data_provenance_audit.md) | ✅ | — |
| **Table 2** (UNSW class stats) | UNSW | `fedgraa/config.py::UNSW_NB15` + data-provenance audit | ✅ | — |
| **Figure 1** (γ-sweep, n=3) | CIC | `results/CIC-IDS-2017_gamma_sweep.json` + `results/figures/CIC-IDS-2017_figure1.png` | ✅ | — |
| **Figure 1** (γ-sweep, n=3) | UNSW | `results/UNSW-NB15_gamma_sweep.json` + `results/figures/UNSW-NB15_figure1.png` | ❌ | **B1** |
| **Figures 2–3** (n-sweep) | CIC | `results/CIC-IDS-2017_n_sweep.json` + `results/figures/CIC-IDS-2017_figure2_3.png` | ❌ | **A1** |
| **Figures 2–3** (n-sweep) | UNSW | `results/UNSW-NB15_n_sweep.json` + `results/figures/UNSW-NB15_figure2_3.png` | ❌ | **B2** |
| n-sweep figure renders FedGRAA curves | both | (correct PNGs above) | ⚠️ blocked by plot bug | **C1** |
| Divergence diagnosis — literal + projected + combined | CIC | `results/diagnostics/CIC_IDS_2017_*` + figures | ✅ | — |
| Divergence diagnosis — literal | UNSW | `results/diagnostics/UNSW_NB15_literal_r60.json` + figure | ✅ | — |
| **Combined-fix seed band** (proj + η′=0.01, E′=5) | CIC | `results/diagnostics/seed_sweep_CIC_IDS_2017_fedgraa_combined_fix.json` | ✅ | — |
| **Combined-fix seed band** | UNSW | `results/diagnostics/seed_sweep_UNSW_NB15_fedgraa_combined_fix.json` | ❌ | **B3** |
| Combined-fix figures | UNSW | `results/figures/diagnosis_unsw_combined_fix_*` | ❌ | **B3** |

**Net gap: four missing data artifacts** (A1, B1, B2, B3), **one figure-rendering defect** (C1), **one script generalization** (§6.3), plus figures, a manifest, a completeness check, and tests.

> **Two matrix caveats.** (a) The paper's n-sweeps are almost certainly split **Figure 2 = CIC, Figure 3 = UNSW** (gap-analysis M-4 and the "Figure 1's UNSW panels, Figure 3" pairing tie Fig 3 to UNSW; the sub-figure captions are anonymized so this can't be nailed down further — `paper_vs_code…` §9.2–9.3). "Figures 2–3" here follows the repo's existing convention of treating the n-sweep as one pair. (b) The CIC **Figure 1** ✅ is marked done on *existence*, but if the strict round-0 convention is chosen (§3) it must be **re-run with `--round-zero-zero`** to match the paper's round-0 row; its current committed form uses real round-0.

---

## 3. Global run conventions (apply to every task)

- **Working dir:** `c:\Dev\Research\PAKDD`. **Interpreter:** `.venv313\Scripts\python.exe` (shown as `python` below).
- **Data/cache:** always pass `--data-dir data --cache-dir .cache`. First UNSW seed-1/2 runs pay a one-time ~30 s preprocessing cost, then cache.
- **Rounds/eval:** base sweeps use `--rounds 200 --eval-every 20` (paper horizon). Diagnostic runs use `--rounds 60 --eval-every 1` (per-round instrumentation, matching the existing CIC diagnostics).
- **Round-0 — an unresolved fidelity decision, to be surfaced, not chosen silently:** the paper's Figure-1 table prints `0.00` for *every* method at round 0, and the strict-"ditto" analysis (`../archive/paper_vs_code_line_by_line_gap_analysis.md` item #2; gap **C-4** in `../archive/full_reproduction_gap_analysis.md`) says a faithful reproduction must force this via `round_zero_zero=True` — which is in fact the **fidelity default** in [`../../fedgraa/config.py`](../../fedgraa/config.py) L67. But `run_experiments.py` makes `--round-zero-zero` opt-in (L108 defaults the field to **False**), so the already-committed CIC γ-sweep records *real* ~9% round-0 accuracy and does **not** match the paper's table. Stage A must pick one convention and apply it to all four base-sweep artifacts:
  - **(Recommended) Strict / ditto:** pass `--round-zero-zero` on every base sweep **and re-run the CIC γ-sweep with it** (~20 min), so all four artifacts match the paper's Figure-1 round-0 row and the config fidelity contract. This is the "faithful reproduction" reading and what C-4 demands.
  - **Honest:** omit the flag everywhere (real round-0) and add an explicit *deviation-from-C-4* note to the manifest (§6.5) and [`../archive/strict_reproduction_report.md`](../archive/strict_reproduction_report.md), reconciling it with the `True` fidelity default.

  Either is defensible; **silently omitting it — which an earlier draft of this plan did — is not**, because it leaves the repo self-contradictory (config says `True`, every committed artifact says `False`). This is a reproduction-philosophy call; raise it with the mentor if unsure. The §8 runbook and §2 matrix assume the **strict** choice; adjust if you pick honest.
- **Combined fix = projection + `η′=0.01` + `E′=5`.** Not `η′=0.001`. The `0.001` value was the *single-lever* intervention that collapsed overall accuracy (diagnosis §6.1); the validated *combined* fix is the 10× reduction `η′=0.01` with `E′` untouched (diagnosis §8 Step 2–3). Use exactly `--project --eta-prime 0.01 --weight-epochs 5`.
- **Parallelism & RAM:** `--parallel` cuts wall time ~3.3× (measured) but each worker loads its own copy of the dataset array. The CIC `X` cache is **1.46 GB**; the n-sweep default is `min(8, cpu//2)` workers ⇒ up to ~11 GB peak for CIC. If the machine has <16 GB free, run that task **without** `--parallel` (safe, slower) or apply the optional `--workers` flag (§6.6). UNSW arrays are ~47 MB — parallel is never a memory risk there.

---

## 4. Defect found: `plot_n_sweep` drops all FedGRAA curves (must fix before figures)

**Location:** [`../../fedgraa/plot.py`](../../fedgraa/plot.py) L157–167.

```python
for ax, n in zip(axes, ns):
    for key in ("fedavg", "static", "fedgraa"):   # <-- bug: bare "fedgraa"
        if key not in results[n]:
            continue
        ...
```

`run_n_sweep` produces per-`n` keys `fedavg`, `static`, `fedgraa_g0.0`, `fedgraa_g0.5`, `fedgraa_g1.0` (see [`../../fedgraa/experiment.py`](../../fedgraa/experiment.py) L281). There is **no** bare `"fedgraa"` key, so the loop matches only `fedavg`/`static` and **silently omits every FedGRAA line** from the n-sweep figure. `plot_gamma_sweep` (L87–89) already does this correctly with the gamma-keyed order; `plot_n_sweep` was never updated to match, and **no test calls `plot_n_sweep`** ([`../../tests/test_plot.py`](../../tests/test_plot.py) imports it but never invokes it). Left unfixed, Tasks A1/B2 would ship figures that look complete but are missing the entire method under study. **Runtime-confirmed during this audit:** `plot_n_sweep` called on live `run_n_sweep` output rendered **2** lines per panel (fedavg, static), not 5 — the FedGRAA curves are provably absent, not merely suspected. Fix in §6.1.

---

## 5. Task-by-task plan

Each task lists the exact command, the artifacts it must produce, an effort estimate (derived from the measured timings in [`../../PAKDD_STATUS_REPORT.md`](../../PAKDD_STATUS_REPORT.md) and [`../../latest_baseline.md`](../../latest_baseline.md); calibrate with a `--rounds 1` probe first), and its acceptance criteria.

### Task A1 — CIC-IDS-2017 n-sweep (paper Figures 2–3)

```bash
python run_experiments.py --data-dir data --cache-dir .cache \
    --datasets CIC-IDS-2017 --sweep n --parallel \
    --rounds 200 --eval-every 20 \
    --results-dir results --plot-dir results/figures
```

- **Produces:** `results/CIC-IDS-2017_n_sweep.json`, `results/figures/CIC-IDS-2017_figure2_3.png`.
- **Configs:** 15 (n∈{3,5,7} × {fedavg, static, fedgraa_g0/g0.5/g1}).
- **Effort:** ~14 min/config. Sequential ≈ 3.5 h; parallel (5 workers) ≈ **55–70 min**. Runs unattended.
- **Depends on:** C1 (plot fix) for a correct figure — but the JSON can be generated first; re-render the figure after C1 (§6, ordering).
- **Accept when:** JSON loads via `load_n_sweep_results` with outer keys `{"3","5","7"}` and each inner dict has all 5 method keys; figure shows 5 curves per panel × 3 panels; both files `git add`-ed.

### Task B1 — UNSW-NB15 γ-sweep (paper Figure 1, second dataset)

```bash
python run_experiments.py --data-dir data --cache-dir .cache \
    --datasets UNSW-NB15 --sweep gamma --parallel \
    --rounds 200 --eval-every 20 \
    --results-dir results --plot-dir results/figures
```

- **Produces:** `results/UNSW-NB15_gamma_sweep.json`, `results/figures/UNSW-NB15_figure1.png`.
- **Effort:** 5 configs, UNSW ~11 min/config; parallel ≈ **15–20 min**.
- **Accept when:** JSON loads via `load_results` with keys `fedavg, static, fedgraa_g0.0/g0.5/g1.0`; figure (`plot_dual_metric`) renders; committed. Expect the literal FedGRAA runs to under-perform Static (consistent with the divergence finding — this is an *honestly-diverging* counterpart, which the goal explicitly allows).

### Task B2 — UNSW-NB15 n-sweep (paper Figures 2–3, second dataset)

```bash
python run_experiments.py --data-dir data --cache-dir .cache \
    --datasets UNSW-NB15 --sweep n --parallel \
    --rounds 200 --eval-every 20 \
    --results-dir results --plot-dir results/figures
```

- **Produces:** `results/UNSW-NB15_n_sweep.json`, `results/figures/UNSW-NB15_figure2_3.png`.
- **Effort:** 15 configs, parallel ≈ **45–55 min**.
- **⚠️ Honest-divergence caveat (must document):** UNSW-NB15 has only **5** genuinely-rare attack classes, so `rare_client_ids(7)` warns and returns 5 ([`../../fedgraa/config.py`](../../fedgraa/config.py) L139–154; `PAPER_AMBIGUITY_REPORT.md` #10). Both the partition and the `last_n_class_accuracy` metric ([`../../fedgraa/metrics.py`](../../fedgraa/metrics.py) L117) then operate on the same 5 classes ⇒ **the n=7 panel is identical to n=5, not an independent data point.** This is expected and correct; it must be labeled "n=7 ≡ n=5 (dataset has 5 rare classes)" in the manifest and, ideally, in the figure caption. It is *not* a bug to fix — it is a property of the dataset the paper's n-sweep under-specifies.
- **Accept when:** JSON present with keys `{"3","5","7"}`; figure renders 5 curves/panel (after C1); n=7≡n=5 documented; committed.

### Task B3 — UNSW-NB15 combined-fix seed band (mirror the CIC treatment)

Three per-seed diagnostic runs, then aggregate. Mirrors diagnosis §8 Step 3 exactly.

```bash
# Seed 0/1/2 — projected combined fix, 60 rounds, per-round instrumentation
python scripts/diagnose_fedgraa.py --dataset UNSW-NB15 --gammas 0 0.5 1.0 \
    --rounds 60 --eval-every 1 --project --eta-prime 0.01 --weight-epochs 5 \
    --seed 0 --tag combined_eta0.01_E5
python scripts/diagnose_fedgraa.py --dataset UNSW-NB15 --gammas 0 0.5 1.0 \
    --rounds 60 --eval-every 1 --project --eta-prime 0.01 --weight-epochs 5 \
    --seed 1 --tag combined_eta0.01_E5_seed1
python scripts/diagnose_fedgraa.py --dataset UNSW-NB15 --gammas 0 0.5 1.0 \
    --rounds 60 --eval-every 1 --project --eta-prime 0.01 --weight-epochs 5 \
    --seed 2 --tag combined_eta0.01_E5_seed2

# Aggregate the three seeds into a variance band (needs the generalized script, §6.3)
python scripts/seed_sweep_combined_fix.py --dataset UNSW-NB15

# Render combined-fix figures for UNSW (mirrors the CIC diagnosis_combined_fix_* set)
python scripts/render_diagnostics.py \
    --in results/diagnostics/UNSW_NB15_projected_combined_eta0.01_E5_r60.json \
    --out-prefix results/figures/diagnosis_unsw_combined_fix
```

- **Produces:**
- `../../results/diagnostics/UNSW_NB15_projected_combined_eta0.01_E5_r60.json` (+ `_seed1`, `_seed2`)
- `../../results/diagnostics/seed_sweep_UNSW_NB15_fedgraa_combined_fix.json`
- `../../results/figures/diagnosis_unsw_combined_fix_composite_loss.png`, `_weights_g0p0/g0p5/g1p0.png`, `_summary.md`
- **Effort:** 3 seeds × (3 γ × 60 rounds, eval_every=1). ~12–18 min/seed ⇒ **~40–55 min** total, plus a one-time ~1 min UNSW cache build for seeds 1 and 2.
- **Note on expected result:** the diagnosis (§9) found UNSW's literal instability is *milder* than CIC's (blowup 7.6× vs 193×; γ=0.5 already reaches 29.6% rare-acc unfixed). The combined fix should still drive blowup to 1.0× and lift rare-acc; report whatever it actually shows — a milder gain than CIC is a legitimate, publishable finding ("severity is dataset-dependent"), not a failure.
- **Accept when:** the seed-band JSON matches the CIC schema (per-γ `all_acc_per_seed`, `last_n_acc_per_seed`, `blowup_ratio_per_seed`, `*_mean`, `*_std`); figures render; all committed.

---

## 6. Code changes required (small, deterministic, testable — no big compute)

These do **not** need the 200-round runs and should be done **first** so the figure/aggregation steps are correct on the first pass.

### 6.1 Fix `plot_n_sweep` (REQUIRED — blocks A1/B2 figures)

In [`../../fedgraa/plot.py`](../../fedgraa/plot.py), replace the hardcoded `("fedavg", "static", "fedgraa")` loop with the same gamma-aware order `plot_gamma_sweep` uses, iterating whatever methods are present:

```python
order = ["fedavg", "static", "fedgraa_g0.0", "fedgraa_g0.5", "fedgraa_g1.0"]
for ax, n in zip(axes, ns):
    for key in order:
        if key not in results[n]:
            continue
        label, color, ls, marker = _METHOD_STYLE[key]
        h = results[n][key].history
        ...
```

`_METHOD_STYLE` already defines all five keys, so no style changes are needed.

### 6.2 Add a real `plot_n_sweep` test (REQUIRED — the bug shipped because nothing tested it)

In [`../../tests/test_plot.py`](../../tests/test_plot.py), add a test that builds a tiny `run_n_sweep` result and asserts the figure contains one line per method per panel:

```python
def test_plot_n_sweep_includes_fedgraa_curves():
    import matplotlib; matplotlib.use("Agg")
    # tiny synthetic n-sweep over ns=(3,) with all 5 methods present
    ... build results via run_n_sweep on the PLOT_TOY spec ...
    fig = plot_n_sweep(results, metric="last_n")
    ax = fig.axes[0]
    # 5 methods (fedavg, static, 3× fedgraa) must each contribute a Line2D
    assert len([ln for ln in ax.get_lines()]) >= 5
```

This is the regression guard: it fails on the current code and passes after 6.1.

### 6.3 Generalize `seed_sweep_combined_fix.py` for any dataset (REQUIRED — blocks B3)

Currently the script hardcodes CIC filenames ([`../../scripts/seed_sweep_combined_fix.py`](../../scripts/seed_sweep_combined_fix.py) L33–39). Add a `--dataset` argument that derives the three per-seed filenames and the output path from the dataset name, defaulting to CIC for backward compatibility:

```python
ap.add_argument("--dataset", default="CIC-IDS-2017")
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
ap.add_argument("--eta-prime", default="0.01")
ap.add_argument("--weight-epochs", type=int, default=5)
ap.add_argument("--rounds", type=int, default=60)
# derive: {DS}_projected_combined_eta{eta}_E{E}{_seed{n} if n>0}_r{rounds}.json
# output: results/diagnostics/seed_sweep_{DS}_fedgraa_combined_fix.json
```

Keep the aggregation math (mean/std over `diagnosis.final_all_acc` / `final_last_n_acc` / `composite_loss_blowup_ratio`) identical, so the CIC band is byte-stable when re-run. **Re-run the CIC aggregation once after refactoring** to confirm the existing `seed_sweep_CIC_IDS_2017_fedgraa_combined_fix.json` is reproduced unchanged (regression check).

### 6.4 Add an aggregator unit test (REQUIRED with 6.3)

Feed the generalized aggregator two or three tiny hand-written diagnostic dicts in a `tmp_path` and assert the emitted mean/std match a NumPy reference. This makes the UNSW band trustworthy without re-running training.

### 6.5 Results manifest — `../../results/README.md` (REQUIRED — reproducibility record)

Create [`../../results/README.md`](../../results/) mapping every committed artifact to: the paper artifact it reproduces, the **exact command**, the **git SHA**, the date, seed(s), and any divergence note (e.g. UNSW n=7≡n=5). Add two columns a reproducibility appendix needs and the JSONs don't carry: **paper-claimed vs reproduced** (the paper's headline numbers — CIC FedGRAA γ=1 = 88.88%, UNSW γ=1 = 21.09%, per `../archive/paper_vs_code_line_by_line_gap_analysis.md` §9.4 — beside what we measured), and an **environment snapshot** (Python 3.13, TF 2.21, NumPy — capture `pip freeze` once). This is both the "did we finish?" ledger and the first draft of the manuscript's reproducibility appendix ([`publication_roadmap.md`](publication_roadmap.md) §4). The current JSONs store metrics but no provenance; this file is where provenance lives.

### 6.6 *(Optional, recommended)* `--workers` and `--ns` CLI flags on `run_experiments.py`

Two small robustness additions, each ~3 lines + a test:
- `--workers N` → thread through to `run_*_parallel(workers=N)`, so RAM-constrained machines can cap CIC parallelism instead of choosing all-or-nothing (§3).
- `--ns 3 5` → let the UNSW n-sweep skip the redundant n=7≡n=5 run instead of hardcoding `(3,5,7)` ([`../../run_experiments.py`](../../run_experiments.py) L185/198).

Optional because Stage A completes without them (run CIC sequentially; document UNSW n=7). Recommended because they remove the two sharpest operational edges.

---

## 7. Verification — make "is Stage A done?" a single green/red signal

Add `scripts/check_stage_a_complete.py` that asserts the full artifact set exists, is loadable, and is internally consistent, exiting non-zero on any gap:

- All four new JSONs load via `load_results` / `load_n_sweep_results` and have the expected keys/shapes.
- Every expected figure exists and is non-empty — `CIC-IDS-2017_figure2_3.png`, `UNSW-NB15_figure1.png`, `UNSW-NB15_figure2_3.png`, plus the five `diagnosis_unsw_combined_fix_*` files (the base sweeps ship **more** than "four PNGs").
- Each n-sweep figure actually contains ≥5 `Line2D`s per panel (the *rendered* guard against the §4 plot bug regressing, complementing the data-level check below).
- Both n-sweep JSONs contain all 5 method keys per `n` (this is what would have caught the plot bug at the *data* level).
- The UNSW combined-fix band matches the CIC band's schema.
- Every path listed in `../../results/README.md` exists on disk and is git-tracked (`git ls-files`).

Keep this as a standalone script (not part of `pytest tests`) so the unit suite stays decoupled from large committed binaries. Run it as the last step of Stage A and cite its `0` exit in the completion note.

**Also run the full suite** after the §6 code changes: `.venv313\Scripts\python.exe -m pytest tests --tb=short` must stay green (79/79 + the new tests from 6.2/6.4).

---

## 8. One-shot runbook (ordered; copy-paste)

```bash
cd /c/Dev/Research/PAKDD

# --- Phase 0: calibrate machine (optional, ~2 min) ---
python run_experiments.py --data-dir data --cache-dir .cache \
    --datasets CIC-IDS-2017 --sweep gamma --rounds 1 --eval-every 1   # time one round/config

# --- Phase 1: code fixes first (no big compute) ---
#  6.1 fix plot_n_sweep · 6.2 add plot test · 6.3 generalize aggregator · 6.4 aggregator test
.venv313/Scripts/python.exe -m pytest tests --tb=short                 # must stay green
python scripts/seed_sweep_combined_fix.py --dataset CIC-IDS-2017        # 6.3 regression: CIC band unchanged

# --- Phase 2: base sweeps (unattended; watch RAM on CIC n-sweep, see §3) ---
#  Using the strict/ditto round-0 convention (§3, recommended)? Append --round-zero-zero to each
#  command below AND re-run the CIC gamma sweep with it so all four base artifacts match the paper.
python run_experiments.py --data-dir data --cache-dir .cache --datasets CIC-IDS-2017 \
    --sweep n     --parallel --rounds 200 --eval-every 20 --results-dir results --plot-dir results/figures   # A1
python run_experiments.py --data-dir data --cache-dir .cache --datasets UNSW-NB15 \
    --sweep both  --parallel --rounds 200 --eval-every 20 --results-dir results --plot-dir results/figures   # B1 + B2

# --- Phase 3: UNSW combined-fix seed band ---
for s in 0 1 2; do
  tag=combined_eta0.01_E5$( [ $s -gt 0 ] && echo _seed$s )
  python scripts/diagnose_fedgraa.py --dataset UNSW-NB15 --gammas 0 0.5 1.0 \
      --rounds 60 --eval-every 1 --project --eta-prime 0.01 --weight-epochs 5 --seed $s --tag $tag
done
python scripts/seed_sweep_combined_fix.py --dataset UNSW-NB15
python scripts/render_diagnostics.py \
    --in results/diagnostics/UNSW_NB15_projected_combined_eta0.01_E5_r60.json \
    --out-prefix results/figures/diagnosis_unsw_combined_fix

# --- Phase 4: manifest + verify + commit ---
#  write results/README.md (6.5)
python scripts/check_stage_a_complete.py            # must exit 0
git add results/ fedgraa/plot.py scripts/ tests/ docs/current/stage_a_plan.md
git commit -m "Stage A: complete faithful reproduction (CIC n-sweep, UNSW gamma/n/combined-fix)"
```

*(The `for`/`tag` loop is Bash. The PowerShell equivalent is three explicit `diagnose_fedgraa.py` calls as written in Task B3.)*

---

## 9. Explicitly out of scope (deferred to Stage B/C)

To keep Stage A bounded, the following are **not** done here — they belong to later stages and are listed so nobody expands Stage A into them mid-flight:

- **Statistical significance** (paired t-tests, bootstrap CIs) beyond the 3-seed mean±std bands → Stage B ([`publication_roadmap.md`](publication_roadmap.md) §2 Stage B).
- **Consolidated ablation table** (projection-alone vs η′-alone vs both) → Stage B.
- **Corrected convergence theory** for the fixed algorithm → Stage B.
- **Baselines beyond FedAvg/Static** (FedProx/FedOpt/FedSND already implemented; AFL/q-FedAvg/FedAdp not) → Stage C.
- **Scalability (50–200 clients) and extreme non-IID** experiments → Stage C.
- **Changing the strict-reproduction default** in `fedgraa/` — the combined fix stays diagnostic-only (`ProjectedFedGRAA` lives only in `scripts/`), per the standing no-silent-patching mandate ([`../archive/strict_reproduction_report.md`](../archive/strict_reproduction_report.md)).

---

## 10. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| CIC n-sweep OOM under `--parallel` (8 × 1.46 GB) | Med (on <16 GB) | Task stalls/crashes | Run sequentially, or add/use `--workers 3` (§6.6); Phase-0 probe reveals worker count |
| Figures ship without FedGRAA curves | **High if unfixed** | Silent, embarrassing | C1 fix + 6.2 test done in Phase 1, *before* any figure render (bug runtime-confirmed in audit) |
| Round-0 convention left implicit → repro contradicts the paper's Fig-1 table *and* the `config.py` fidelity default | Med | Reviewer catches self-inconsistency between config and artifacts | Decide per §3 (recommend strict `--round-zero-zero` on all four sweeps + re-run CIC γ); record the choice in the manifest |
| UNSW n=7 mistaken for a real data point | Med | Misleading figure/claim | Document n=7≡n=5 in manifest + caption; optional `--ns 3 5` to skip it |
| Combined-fix run with wrong η′ (0.001 not 0.01) | Low | Wrong band, wasted ~45 min | Commands in §5 pin `--eta-prime 0.01 --weight-epochs 5`; §3 calls it out |
| Aggregator refactor silently changes CIC band | Low | Breaks a committed result | 6.3 regression check re-runs CIC and diffs the JSON |
| Long runs interrupted (laptop sleep/crash) | Med | Lost hours | `--checkpoint-dir` resume is supported ([`../../run_experiments.py`](../../run_experiments.py) L96); or run overnight on AC power |
| Non-determinism across a re-run | Low | Numbers shift slightly | Seeds pinned; manifest records SHA + seed so any drift is auditable |

---

## 11. Effort summary

| Phase | Work | Wall time |
|---|---|---|
| 1 | Code fixes (§6.1–6.4) + tests + CIC-band regression | ~45–60 min hands-on |
| 2 | CIC n-sweep + UNSW γ+n sweeps | ~2–2.5 h compute (parallel), unattended |
| 3 | UNSW combined-fix seed band + figures | ~45–55 min compute, unattended |
| 4 | Manifest (§6.5) + completeness check (§7) + commit | ~30–45 min hands-on |
| **Total** | | **~1.5–2 h hands-on + ~3–3.5 h unattended compute** |

Sequential-only (no `--parallel`) roughly triples the compute phases to ~8–10 h but removes the RAM risk entirely — a reasonable overnight option.

---

*When Phases 1–4 are complete and `check_stage_a_complete.py` exits 0, Stage A is closed. Per [`publication_roadmap.md`](publication_roadmap.md) §6.5, the next action is to draft the Stage B critique section and bring the combined Stage A + B package to the PhD mentor to jointly scope Stage C.*

---

## Appendix — audit trail (2026-07-23, commit `9f78529`)

This plan was audited against the live tree. Verified by execution (not just code-reading):

- **Test baseline:** `pytest tests` → **79 passed** in 71 s. The "still green" bar is real.
- **n-sweep path works end-to-end:** `run_n_sweep` on a tiny synthetic spec produced keys `fedavg, static, fedgraa_g0.0/g0.5/g1.0`; `save_n_sweep_results` → `load_n_sweep_results` round-tripped with outer key `"3"` and matching inner keys. The A1/B2 command path is sound.
- **The §4 plot defect is real:** `plot_n_sweep` on that live output rendered **2** `Line2D`s per panel, not 5 — FedGRAA curves provably dropped.
- **`n=7 ≡ n=5` (UNSW):** confirmed — [`partition.py`](../../fedgraa/partition.py#L191) and the metric both consume the single `rare_client_ids(n)`, which truncates to 5.
- **All line-number citations** re-checked against current files (accurate to ±1; the `L185/198` ref was corrected from `L186/199`).
- **Corrected during audit:** the round-0 convention (§3, §1 table, §2 caveat, risk register) was flipped from a silent "omit it" to an explicit, surfaced fidelity decision; §0.6/§6.6/§7 wording and the manifest scope (§6.5) were tightened.
