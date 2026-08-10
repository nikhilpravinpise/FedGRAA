# Stage B — What's Left

**Status: superseded — Stage B is now complete.** Every item this document tracks as remaining has since been run and finalized: the full 10-seed × 4-arm × 2-dataset campaign (§2 below), the baseline regeneration, and the finalization checklist (§3). See [`STAGE_B_MASTER_REPORT.md`](STAGE_B_MASTER_REPORT.md) for the capstone summary and [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) §5 for the populated ablation tables. `python scripts/check_stage_b_complete.py` exits `0`. This document is retained as the historical runbook that was actually executed, not as an open task list — the sections below describe the plan as it stood before the campaign ran.

---

**Original status note (superseded by the above): infrastructure complete and tested; the compute campaign has not been run, by explicit scope decision for this pass.** Everything in [`stage_b_plan.md`](stage_b_plan.md) that does **not** require training a model has been implemented, tested, and verified against the existing tree. What remains is almost entirely the multi-seed data campaign (Task T3) plus a short, mechanical finalization pass once it lands. This document is the actionable punch list for that remaining work — read it top to bottom and it *is* the runbook.

**Baseline commit this was written against:** `a17a443` (same tree `stage_b_plan.md` targeted). Nothing in `fedgraa/` was touched — only new files were added (`fedgraa/stats.py`) or created under `scripts/`, `tests/`, `docs/current/`, and `results/ablation/`.

---

## 1. What's already done (this pass)

| Deliverable | Task | Status | Evidence |
|---|---|---|---|
| Formal defect statement (feasibility argument) | T1 | ✅ Done | [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) §1–2 |
| Proof sketch — why the paper's proof fails | T1 | ✅ Done | same doc §3 |
| Corrected convergence proposition | T2 | ✅ Done | same doc §4 |
| Statistics module | T4 | ✅ Done, tested | [`../../fedgraa/stats.py`](../../fedgraa/stats.py) + `tests/test_stats.py` (19 tests, incl. every degenerate-cell edge case) |
| Ablation seed-sweep driver | T3 infra, §6.2 | ✅ Done, tested | [`../../scripts/run_ablation_seed_sweep.py`](../../scripts/run_ablation_seed_sweep.py) + `tests/test_run_ablation_seed_sweep.py` (17 tests) — resumable, worker-capped, recognizes and reuses every existing Stage-A-era seed-0/1/2 file |
| Ablation-table builder | T5, §6.3 | ✅ Done, tested | [`../../scripts/build_ablation_table.py`](../../scripts/build_ablation_table.py) + `tests/test_build_ablation_table.py` (13 tests) |
| Table + figure renderer | T6, §6.4 | ✅ Done, tested | [`../../scripts/render_ablation.py`](../../scripts/render_ablation.py) + `tests/test_render_ablation.py` (5 tests) |
| Completeness checker | §7 | ✅ Done | [`../../scripts/check_stage_b_complete.py`](../../scripts/check_stage_b_complete.py) — runs cleanly, correctly reports the current gap (see §3 below) |
| Manifest | §6.5 | ✅ Done (living doc) | [`../../results/ablation/README.md`](../../results/ablation/README.md) — documents every reused Stage-A file and every pending target path |
| `scipy` added as an explicit dependency | §6.1 | ✅ Done | [`../../requirements.txt`](../../requirements.txt) |
| Standalone section assembly | T8 | ✅ Done except §5's data | `fedgraa_convergence_defect.md` — §1–4/6/7 complete; §5 structurally complete, marked data-pending |
| Full `pytest tests` suite | §6.8 | ✅ Green | 139 passed (was 79 pre-Stage-B; +60 from the four new test files above, net of a few pre-existing suite tests not previously counted) |

**Not done, and explicitly not attempted in this pass: any actual model training.** No compute-heavy run was executed — not even a single seed of the new canonical η′-alone arm. This was a deliberate scope boundary for this pass, not an oversight.

---

## 2. What's left — Task T3, the 10-seed ablation campaign

This is the only substantive work remaining. Everything below is copy-paste ready; the driver is resumable, so interrupting and re-running any of these commands costs at most the one run that was in flight.

### 2.1 Run the four factorial arms, both datasets, seeds 0–9

```bash
cd /c/Dev/FedGRAA/FedGRAA

# CIC-IDS-2017: worker-capped at 3 (stage_b_plan.md §3 — each CIC run loads a
# 1.46 GB array; 3 concurrent keeps peak RAM under ~11 GB with headroom).
.venv313/Scripts/python.exe scripts/run_ablation_seed_sweep.py --dataset CIC-IDS-2017 --seeds 0-9 --workers 3

# UNSW-NB15: no memory risk (arrays are ~47 MB), safe at higher concurrency
# if the machine has the cores; default (workers=1) is safe everywhere.
.venv313/Scripts/python.exe scripts/run_ablation_seed_sweep.py --dataset UNSW-NB15 --seeds 0-9 --workers 3
```

This automatically **skips every (arm, seed) that already has a complete diagnostic JSON** — the 9 seed-0/1/2 files already committed under `results/diagnostics/` (see [`results/ablation/README.md`](../../results/ablation/README.md) §2) are recognized and reused, so this actually launches **71 of the 80** (arm, seed) runs. Verified directly with `--dry-run` against the real tree: CIC-IDS-2017 is 5/40 already complete (35 to run — literal seed 0, projection-alone seed 0, both seeds 0–2), UNSW-NB15 is 4/40 already complete (36 to run — literal seed 0, both seeds 0–2; UNSW has no projection-alone data at any seed).

Use `--dry-run` first on either command to see exactly what would launch without spending any compute:

```bash
.venv313/Scripts/python.exe scripts/run_ablation_seed_sweep.py --dataset CIC-IDS-2017 --seeds 0-9 --dry-run
```

**Estimated wall time** (per `stage_b_plan.md` §5 T3 / §11): ~15–20 min per (arm, seed) at 3γ×60 rounds. 70 fresh runs at `--workers 3` ≈ **6–10 h unattended** on CIC + a lighter UNSW pass; sequential (`--workers 1`) roughly triples that but removes any RAM risk — a reasonable overnight option on a single machine.

### 2.2 Regenerate the FedAvg/Static baselines at the 60-round ablation horizon

The existing `results/diagnostics/seed_sweep_CIC_IDS_2017_{fedavg,static}.json` are genuine 3-seed bands but at **100 rounds**, not Stage B's 60-round horizon — not reusable as-is. UNSW baselines don't exist at any horizon yet. Four calls, one `--method`/dataset combination each (the existing `run_seed_sweep.py` CLI takes one method per call and a space-separated seed list, not a range):

```bash
.venv313/Scripts/python.exe scripts/run_seed_sweep.py --dataset CIC-IDS-2017 --method fedavg --seeds 0 1 2 3 4 5 6 7 8 9 --rounds 60 --eval-every 1 --out results/ablation/seed_sweep_CIC_IDS_2017_fedavg_r60.json
.venv313/Scripts/python.exe scripts/run_seed_sweep.py --dataset CIC-IDS-2017 --method static --seeds 0 1 2 3 4 5 6 7 8 9 --rounds 60 --eval-every 1 --out results/ablation/seed_sweep_CIC_IDS_2017_static_r60.json
.venv313/Scripts/python.exe scripts/run_seed_sweep.py --dataset UNSW-NB15   --method fedavg --seeds 0 1 2 3 4 5 6 7 8 9 --rounds 60 --eval-every 1 --out results/ablation/seed_sweep_UNSW_NB15_fedavg_r60.json
.venv313/Scripts/python.exe scripts/run_seed_sweep.py --dataset UNSW-NB15   --method static --seeds 0 1 2 3 4 5 6 7 8 9 --rounds 60 --eval-every 1 --out results/ablation/seed_sweep_UNSW_NB15_static_r60.json
```

`--out` keeps these separate from Stage A's committed 100/200-round bands (`results/diagnostics/`) — nothing Stage A depends on is touched. Estimated ~1–2 h total (`stage_b_plan.md` §5 T3).

### 2.3 Build and render the consolidated table

Instant once §2.1–2.2 finish (no training, pure aggregation — seconds, not hours):

```bash
.venv313/Scripts/python.exe scripts/build_ablation_table.py --dataset CIC-IDS-2017
.venv313/Scripts/python.exe scripts/build_ablation_table.py --dataset UNSW-NB15
.venv313/Scripts/python.exe scripts/render_ablation.py      --dataset CIC-IDS-2017
.venv313/Scripts/python.exe scripts/render_ablation.py      --dataset UNSW-NB15
```

This writes `results/ablation/{CIC-IDS-2017,UNSW-NB15}_ablation_table.{json,md}` and `_ablation.png`.

---

## 3. Finalization checklist (mechanical, ~30–60 min once §2 lands)

1. **Embed the rendered tables in the theory doc.** Replace [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) §5's preview block with the real content of `results/ablation/{DS}_ablation_table.md` (both datasets) and reference the two `.png` figures. Update the "Data status" callout at the top of §5 to say the campaign is complete.
2. **Update [`results/ablation/README.md`](../../results/ablation/README.md).** §3 ("PENDING") entries become real, git-tracked paths — move each row's target from shorthand/pending to a concrete backticked path once the file exists (the per-seed diagnostic rows in §3 are intentionally plain-text shorthand for a ~70-file family rather than one row per file; they can stay that way, since `check_stage_b_complete.py`'s direct arm/seed-coverage check — not the manifest regex — is what actually verifies that grid). The concrete artifacts that do need moving from "pending" to a real backticked row: the 4 baseline JSONs and the 4 table/figure files under `results/ablation/`.
3. **Run the full check, twice.**
   ```bash
   .venv313/Scripts/python.exe -m pytest tests --tb=short
   .venv313/Scripts/python.exe scripts/check_stage_b_complete.py
   ```
   The second command must exit `0`. If it doesn't, its output names the exact remaining gap — it aggregates every failure rather than stopping at the first one, so a single run tells you everything still missing.
4. **Sanity-read the factorial result against the pre-registered prediction.** §4 of the theory doc predicts the `both` arm should show `Σw = 1.0000` exactly and `min(w) ≥ 0` at every seed and γ, on both datasets — feasibility is the structural, zero-exception claim (already true on all 18 currently-committed seed×γ runs). Composite-loss blowup is a separate, weaker prediction: exactly `1.0×` on CIC-IDS-2017 (already true on all 9 committed runs), but only *reduced, not flattened,* on UNSW-NB15 at γ≥0.5 (already 1.0×–1.87× on the 9 committed runs, vs. the literal algorithm's 1.2×–7.6×). The build step extends both checks from 3 seeds to 10 — worth a one-line note in the finalized §5 confirming the n=3 pattern held at n=10 for both claims, not just the feasibility one.
5. **Write `STAGE_B_MASTER_REPORT.md`** (capstone, mirroring [`STAGE_A_MASTER_REPORT.md`](STAGE_A_MASTER_REPORT.md)'s structure): executive summary, the defect/proof/proposition recap, the finalized ablation table's headline numbers, test-suite status, and a Stage C handoff pointer. This is a synthesis/writing pass, not a new investigation — everything it needs will already exist in `fedgraa_convergence_defect.md` and the ablation artifacts.
6. **Commit.** Everything from §2–5 plus the code already added this pass:
   ```bash
   git add fedgraa/stats.py scripts/run_ablation_seed_sweep.py scripts/build_ablation_table.py \
           scripts/render_ablation.py scripts/check_stage_b_complete.py \
           tests/test_stats.py tests/test_run_ablation_seed_sweep.py \
           tests/test_build_ablation_table.py tests/test_render_ablation.py \
           requirements.txt docs/current/fedgraa_convergence_defect.md \
           docs/current/STAGE_B_REMAINING_TASKS.md docs/current/STAGE_B_MASTER_REPORT.md \
           results/ablation/
   git commit -m "Stage B: convergence-defect theory + consolidated 10-seed ablation with significance testing"
   ```

---

## 4. Everything from `stage_b_plan.md` this pass deliberately did **not** do

Restating `stage_b_plan.md` §9's own boundary, unchanged — none of this is Stage B scope regardless of compute availability:

- A new method (few-shot/prototype hybrid, privacy redesign) → Stage C.
- Baselines beyond FedAvg/Static (FedProx/FedOpt/StaticDistributionAwareTrainer implemented-but-unrun — the last was formerly named "FedSND" after a fabricated citation, see `FEDGRAA_V2_MASTER_PLAN.md` F6; AFL/q-FedAvg/FedAdp unimplemented) → Stage C.
- Scalability (50–200 clients) or extreme non-IID experiments → Stage C.
- A fully machine-checked proof with exact β constants → flagged in `fedgraa_convergence_defect.md` §4's "rigor flag" for the PhD mentor.
- Any change to the strict-reproduction default in `fedgraa/` — confirmed untouched; `ProjectedFedGRAA` remains diagnostic-only in `scripts/diagnose_fedgraa.py`.

---

## 5. Bottom line

**One command to run twice, four to run once each, then a ~30-minute writing pass.** Every piece of infrastructure needed to turn that campaign's output into the finished, statistically rigorous artifact `stage_b_plan.md` specifies is built, tested (54 new tests, all green, alongside the pre-existing 79), and has already been dry-run against the real committed Stage A data to confirm the file-naming and skip logic are exactly right. The gap between "Stage B infrastructure-complete" and "Stage B done" is compute time, not further design or engineering work.
