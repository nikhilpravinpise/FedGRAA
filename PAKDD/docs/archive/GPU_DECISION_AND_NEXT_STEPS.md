# GPU: Keep or Kill? — And What Actually Matters Next

**Date:** 2026-07-23
**Question asked:** *"Do we really need to remove the GPU stuff? What should we do?"*
**Short answer:** **No — don't delete it. But stop pursuing it, fix the one doc that still oversells it, and put the effort where the project is actually stuck: the science, not the speed.**

---

## 1. TL;DR

| Question | Verdict |
|---|---|
| Is the GPU rejection correct? | **Yes.** Measured, sound, and re-derivable from first principles. The model is too small for a GPU to win. |
| Should we delete the GPU scripts? | **No.** They *are* the evidence for the negative result. Deleting them makes "we tried GPU, it was slower" unreproducible. Archive + label instead. |
| Is there GPU *code* to remove from `fedgraa/`? | **None.** The source is device-agnostic. There is literally nothing to rip out. |
| Is anything actually broken? | **Yes — one doc.** [`v2_completion.md`](v2_completion.md) still calls GPU "the single remaining action item" and promises **192–320×**. That directly contradicts the measured result in [`speedup_plan_v2.md`](speedup_plan_v2.md). Fix that contradiction. |
| What should we spend time on instead? | **Reproduction, not performance.** Speed is *solved* (full repro ≈ 2 h on CPU). FedGRAA getting **0% rare-class accuracy** — versus the paper's headline **88%** — is the real, unsolved problem. |

---

## 2. What "GPU stuff" actually is (inventory)

There is **no GPU-specific code in the model or training path**. A grep for
`gpu|cuda|device|physical_device` across [`../../fedgraa/`](../../fedgraa/) returns
**zero matches**. The fused step is `@tf.function(jit_compile=True)` and would
transparently use a GPU if TF saw one — nothing to add, nothing to remove.

"GPU stuff" is entirely **tooling + prose**:

The 6 GPU helper scripts have been **archived** (not deleted) to
[`../../scripts/archive/gpu/`](../../scripts/archive/gpu/) with a
[`README.md`](../../scripts/archive/gpu/README.md) marking them explored-and-rejected:

| Location (now under `scripts/archive/gpu/`) | What | Verdict |
|---|---|---|
| [`benchmark_gpu.py`](../../scripts/archive/gpu/benchmark_gpu.py) | CPU-vs-GPU per-step timing harness | **Kept** — the measurement that produced the 25%-slower finding |
| [`fix_gpu_import.py`](../../scripts/archive/gpu/fix_gpu_import.py) | ctypes `RTLD_GLOBAL` CUDA preload (the TF 2.21 WSL2 workaround) | **Kept** — non-obvious, hard-won, documents a real TF bug |
| [`test_gpu.py`](../../scripts/archive/gpu/test_gpu.py), [`run_tests_gpu.py`](../../scripts/archive/gpu/run_tests_gpu.py) | Run the suite under XLA:GPU | **Kept (archived)** |
| [`setup_wsl2_gpu.ps1`](../../scripts/archive/gpu/setup_wsl2_gpu.ps1), [`setup_gpu_wsl.sh`](../../scripts/archive/gpu/setup_gpu_wsl.sh) | WSL2 + CUDA + TF install automation | **Kept (archived)** |
| GPU section of [`speedup_plan_v2.md`](speedup_plan_v2.md) | The negative result, fully measured | **Keep — this is canonical** |
| GPU section of [`v2_completion.md`](v2_completion.md) | Optimistic "5–10× / 192–320×" projection + "remaining action item" | **Reconciled** — SUPERSEDED banner added (see §4) |
| GPU rows in [`../../PAKDD_STATUS_REPORT.md`](../../PAKDD_STATUS_REPORT.md) | "GPU rejected" summary | **Keep — correct** |

*(The 7th script in `scripts/`, `real_reproduction.py`, is the CPU reproduction
runner — not GPU — and stays in place.)*

**Net:** nothing deleted from the codebase; scripts archived with a README; the
one contradicting doc reconciled.

---

## 3. Why the GPU rejection is correct (not just an opinion)

The measured numbers in [`speedup_plan_v2.md`](speedup_plan_v2.md):

| Metric | CPU (Windows native) | GPU (RTX 5050, WSL2) | Winner |
|---|---|---|---|
| Per-step | **0.70 ms** | 0.87 ms | CPU (GPU 25% slower) |
| XLA first-compile | **0.66 s** | 3.45 s (PTX JIT for CC 12.0) | CPU |
| Per-round (14 clients) | **~2.5 s** | ~3.18 s | CPU |
| Tests passing | **65/65** | 62/65 (3 FP32 precision fails) | CPU |

This is exactly what theory predicts, so the measurement is trustworthy:

- **The model is tiny.** `78→200→100→50→15`, ~41.7 K params, batch 64 →
  ~16 MFLOPs/step. That's a rounding error of GPU throughput.
- **It's launch-bound, not compute-bound.** Each step is a handful of small
  matmuls. On a GPU, per-kernel launch overhead dominates; the arithmetic
  intensity is far too low to amortize it. CPU + oneDNN + XLA keeps everything
  in cache with no host↔device round-trips.
- **Blackwell CC 12.0 is brand new** → kernels JIT from PTX on first use, adding
  compile latency with no steady-state payoff at this size.
- **Precision drift breaks strict reproduction.** cuDNN vs oneDNN FP32 ordering
  flips 3 finite-difference/equivalence tests. For a *paper-faithful*
  reproduction you want the deterministic CPU path anyway.

GPUs win on large matmuls / large batches / conv stacks. This workload is the
opposite. **The negative result is real and worth keeping precisely because it
saves the next person from repeating the WSL2 setup.**

### The decisive point: even a GPU *win* wouldn't matter

Speed is already solved on CPU:

```
Full reproduction (both datasets, all sweeps)
Pre-v1 (original):            ~80 h
v1 (fused XLA, CPU):          ~6.4 h     →  12.5×
v2 (CPU + parallel sweeps):   ~2 h       →  40×
```

A gamma sweep is **20 minutes**. Nothing about the research is gated on going
faster than that. Chasing GPU is optimizing a non-bottleneck.

---

## 4. The one thing that is actually wrong: contradicting docs

[`v2_completion.md`](v2_completion.md) predates the GPU benchmark and still says:

- *"What's Left: Phase 19 — GPU via WSL2 … the **single remaining action item**"*
- Expected *"**~0.15–0.30 ms per step** (5–10× vs CPU)"*
- *"v2 + RTX 5050 GPU + parallel: ~15–25 minutes → **192–320×**"*

Every one of those is **falsified** by the measurements in
[`speedup_plan_v2.md`](speedup_plan_v2.md) (GPU is *slower*). Two docs in the
same folder now tell opposite stories. That's the only concrete defect the GPU
question surfaces.

**Fix:** add a "SUPERSEDED — see GPU benchmark" banner at the top of
`v2_completion.md`'s Phase 19 section (or delete that section and point to
`speedup_plan_v2.md`). Keep the rest of `v2_completion.md` — its shipped-phase
log is fine. *(Done — the banner was added and the file has since been moved to
`v2_completion.md` as part of a broader docs decluttering pass.)*

---

## 5. Where the effort should actually go — the reproduction is not done

> **Update (2026-07-23): P0 below is now answered.** See
> [`fedgraa_divergence_diagnosis.md`](../current/fedgraa_divergence_diagnosis.md) for the
> full measurement-backed diagnosis. Short version: FedGRAA's aggregation
> weights measurably leave the probability simplex in round 1 at every γ,
> causing the composite loss to blow up 2.7×–193× — a direct, confirmed
> violation of the paper's own Lemma 1 — and a diagnostic ablation confirms
> the mechanism. The subsections below are kept as-written for history.

The performance work is a finished, polished side-quest. The **paper itself is
not yet reproduced.** Ranked by importance:

### P0 — FedGRAA does not converge (the headline result is missing)

Real CIC-IDS-2017, 200 rounds, n=3 (from [`../../PAKDD_STATUS_REPORT.md`](../../PAKDD_STATUS_REPORT.md)):

| Method | All-class acc | Rare (last-n) acc |
|---|---|---|
| FedAvg | 61.0% | **0.0%** |
| Static Reparam | 52.3% | **91.7%** |
| FedGRAA γ=0 | 62.5% | **0.0%** |
| FedGRAA γ=0.5 | 18.6% | **0.0%** |
| FedGRAA γ=1.0 | 17.4% | **0.0%** |

The paper claims FedGRAA hits **~88% on minority classes** and beats FedAvg by
**up to 55%**. Our FedGRAA gets **0% on rare classes at every γ** and *degrades*
overall accuracy as γ rises. Static Reparam (a formula with no learning) is the
only thing that recovers rare classes. **This is the result the whole project
exists to produce, and right now it says the opposite of the paper.**

This is currently written off as "a reproduction gap, not a code bug." That
conclusion may be right — but it has not been *earned* yet. Before we can
publish "FedGRAA does not reproduce," we owe a real diagnosis:

- **Is the composite-loss weight learner actually diverging?**
  `TrainHistory.is_composite_loss_monotone` already tracks Lemma 1's
  monotonicity claim — dump it per round for the γ=1 run and see where L′ blows
  up. That's a 10-minute instrumentation change against existing state.
- **Hyperparameter sensitivity.** The paper's `η'=0.1, E'=5` are the prime
  suspects (§ status report). Sweep `η' ∈ {0.001, 0.01, 0.1}` × `E' ∈ {1,5}`
  on the *fast synthetic path first*, then confirm on real data. The
  no-projection choice (Algorithm 2 as literally printed) means Adam can push
  weights off the simplex — check whether `w` is leaving `[0,1]`/`Σ=1`.
- **Is γ=0 FedGRAA ≈ Static?** It should be close (both should protect rare
  classes), yet γ=0 gets 0% while Static gets 91.7%. That gap is a lead worth
  pulling — it suggests the weight learner is actively *hurting* even at γ=0.

### P1 — Persist results as artifacts (right now they exist only as prose)

There are **zero result JSON/PNG files** anywhere in the repo — the numbers in
the status report can't be regenerated from disk or plotted.
[`../../scripts/real_reproduction.py`](../../scripts/real_reproduction.py) already
calls `save_results(...)` and there's a `plot.py`; wire the canonical run to
emit `results/*.json` + Figure 1/2/3 PNGs and commit them. Without this, "we
reproduced X" is an unverifiable claim.

### P2 — Confirm the UNSW-NB15 cache is real, not synthetic

At audit time (2026-07-10) `data/UNSW-NB15/` was empty and every UNSW number was
synthetic. There is now a `.cache/UNSW_NB15_seed0_{X,y}.npy` — **someone loaded
UNSW data since.** Verify whether it came from real CSVs or the synthetic
generator (`make_synthetic_dataset`). If synthetic, every UNSW result is still
`🔴 BLOCKED` and must be labeled as such; real CSVs are still needed.

### P3 — Seed / paper-number caveats (structural, unfixable by us)

The paper discloses no seed and its Figure-1 numbers are suspected placeholders
(round-0 = 0.00 is a reporting artifact we already special-case). Even a perfect
implementation isn't expected to hit the exact table. Run
`run_seed_sweep` to report variance bands rather than a single point, and frame
the writeup around "faithful implementation + honest negative result" instead of
matching a number that may not be real.

---

## 6. Concrete action checklist

**GPU (DONE — 2026-07-23):**
- [x] Added a "SUPERSEDED by GPU benchmark, see `speedup_plan_v2.md`" banner to the Phase 19 section of `v2_completion.md` (before committing that untracked file).
- [x] Moved the 6 GPU scripts into [`../../scripts/archive/gpu/`](../../scripts/archive/gpu/) with a [`README.md`](../../scripts/archive/gpu/README.md) recording the negative result. `real_reproduction.py` (CPU) stays in `scripts/`.
- [x] Left `fedgraa/` untouched — it's device-agnostic; nothing to remove.
- [x] Updated the `scripts/setup_wsl2_gpu.ps1` reference in `PAKDD_STATUS_REPORT.md` to point at the archive.

**Science (DONE — 2026-07-23, see [`fedgraa_divergence_diagnosis.md`](../current/fedgraa_divergence_diagnosis.md) for the full writeup):**
- [x] Instrumented and dumped `is_composite_loss_monotone` + full `w` trajectory per round (not just per eval round) for γ=0/0.5/1 on real CIC-IDS-2017 via `scripts/diagnose_fedgraa.py` (zero changes to `fedgraa/trainer.py`/`aggregation.py` required — `TrainHistory` already tracked both). **Confirmed weight-learner divergence**: the aggregation weight vector leaves the probability simplex in **round 1, at every γ**, and the composite loss `L'(w^t)` blows up 2.7×–193× depending on γ, a direct, measured violation of the paper's own Lemma 1.
- [x] Ran a diagnostic-only simplex-projection ablation (`ProjectedFedGRAA` in `scripts/diagnose_fedgraa.py`, never imported by `fedgraa/`) that isolates the mechanism: constraining `w` to the simplex collapses the blowup ratio to 1.0×–4.3×, confirming unconstrained weight drift is the proximate cause — but does **not** by itself recover rare-class accuracy within 60 rounds, so this is reported as a partial, not a full, explanation.
- [x] Ran the `η' × E'` sensitivity sweep on synthetic (6 configs, γ=1) — smaller `η'` restores monotonicity and 100% rare-class accuracy on synthetic — then confirmed a reduced-`η'` config on real CIC data (`scripts/diagnose_fedgraa.py --eta-prime`) to check whether the same fix transfers; see the diagnosis doc for the real-data result, which differs from synthetic in an important, honest way.
- [x] Persisted the canonical 200-round gamma sweep to `results/CIC-IDS-2017_gamma_sweep.json` (new `run_experiments.py --results-dir` flag, wired through `save_results`/`save_n_sweep_results`, both now with test coverage) and rendered Figure 1 PNGs to `results/figures/`.
- [x] Verified UNSW-NB15 **and** CIC-IDS-2017 cache provenance directly: both are real, CSV-derived data (UNSW's attack-class counts hit the paper's exact Table 2 values while Benign sits at the true uncapped 93,000, not the paper's 56,691 — synthetic generation would hit 56,691 exactly). No result in this repo is synthetic-blocked.
- [x] Ran seed-variance sweeps (Static, FedAvg; 3 seeds, real CIC data) to report a variance band instead of a single point.
- [x] Wrote the honest reproduction verdict — [`fedgraa_divergence_diagnosis.md`](../current/fedgraa_divergence_diagnosis.md) — now that the divergence is diagnosed, not before.

---

## 7. Bottom line

You're not cooking a performance problem anymore — that's plated and it's good
(40× on CPU, 65/65 tests, three real bugs fixed). **Don't throw away the GPU
work; just stop feeding it and fix the one doc that lies about it.** The dish
that's still raw is the *reproduction itself*: FedGRAA converging to 0% rare-class
accuracy is either the paper's real weakness or our hyperparameter gap, and until
we prove which, the project's headline claim is unsettled. That — not the GPU —
is what deserves the "boil the ocean" energy.
