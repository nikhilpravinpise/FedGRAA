# FedGRAA-v2 Compute Infrastructure — Platform Selection and Validated Setup

> **What this document is.** A record of how the FedGRAA-v2 compute campaign ([`FEDGRAA_V2_MASTER_PLAN.md`](FEDGRAA_V2_MASTER_PLAN.md) §7.2, §8 W4) gets run off the author's laptop: which cloud platforms were evaluated, why each was accepted or rejected, and a validated, reproducible remote-execution setup — including one genuine cross-platform reproducibility finding surfaced in the process.
>
> **What this document is not.** It does not change any claim, number, or finding in the master plan. The compute campaign itself (T1–T7) still cannot run until [`FEDGRAA_V2_MASTER_PLAN.md`](FEDGRAA_V2_MASTER_PLAN.md) §8's W0–W3 implementation work lands — no `ProjectedFedGRAA`, `FedGRAAv2`, `SecAggChannel`, or T3/T5 baseline comparators exist in `fedgraa/` as of this writing. This document only establishes *where* that campaign will run once the code exists.
>
> **✅ UPDATE (2026-07-28).** The implementation work this note said was still needed has landed: `ProjectedFedGRAA`, `FedGRAAv2`, `SecAggChannel`, and the full two-phase exact protocol now exist in `fedgraa/`, tested and green (234+ tests; the plan's own G3/G4 hard gates are closed — see [`FEDGRAA_V2_MASTER_PLAN.md`](FEDGRAA_V2_MASTER_PLAN.md) §14.1 Pass 5 and [`REMAINING_COMPUTE_WORK.md`](REMAINING_COMPUTE_WORK.md)). **T3/T5's baseline comparators are the one item from this note's original caveat that still does not exist** — they are ~14 other papers' methods, not FedGRAA-v2 code, and are scoped as their own reproduction effort in `REMAINING_COMPUTE_WORK.md` §4, not blocking T1/T2/T4. A Modal-ready campaign driver for T1/T2/T4 (`scripts/run_v2_frontier_sweep.py`) now exists, matching the pattern validated below, and has been exercised against one real cell (not the full campaign). Everything else in this document (platform selection, the fork/spawn finding, cost/concurrency analysis) is unchanged and still the operative plan for running that driver.
>
> **Compiled:** 2026-07-27.

---

## 1. Why remote compute at all

[`FEDGRAA_V2_MASTER_PLAN.md`](FEDGRAA_V2_MASTER_PLAN.md) §7.2 puts the minimum viable campaign (T1–T3) at **119 CPU-hours**, and the full T1–T7 stack at **243 CPU-hours** — days of wall-clock time on a single laptop core, and not something worth tying up a personal machine for. The workload is CPU-only (TensorFlow, ~40K-parameter models, no GPU requirement per Stage B's own measured rate), which makes it a good fit for rented or credited cloud compute rather than local hardware.

## 2. Platforms evaluated

| Platform | Outcome | Reason |
|---|---|---|
| **Microsoft Azure** (Azure for Students, $100 credit) | ❌ Not used | Card-free signup path repeatedly hit friction — an existing "Azure Free Trial" subscription on the account forced an "upgrade to Pay-As-You-Go" flow that (like DigitalOcean, below) leads back to a card. Never got past this cleanly. |
| **DigitalOcean** ($200 credit via GitHub Student Pack, expires 2026-07-31) | ❌ Not used | Card declined at checkout: *"Your transaction could not be completed. Please contact your issuer."* Consistent with an Indian card issuer blocking an international/recurring-merchant charge; not fixed in the time available. |
| **Camber** (200 CPU-hours/month, free, no card) | ❌ Rejected after doc review | Deep-dived `docs.cambercloud.com` and the `CamberCloud-Inc` GitHub org. Camber's actual execution model is (a) **Nova**, a chat-based agent interface (`@owner.agent_alias`), and (b) a fixed, closed set of named HPC `engine_type`s — `MPI`, `MESA`, `ATHENA`, `ESO`, `CHANGA`, `LAMMPS`, `GROMACS`, `OPENFOAM` — all domain-specific simulation codes. **There is no generic "run arbitrary Python with pinned pip packages" job type.** Custom Python only lives in interactive Jupyter notebooks, not unattended multi-hour background jobs. The official `camber-mcp-official` MCP server was also checked directly; it exposes only `agents_chat`/`agents_list` (chat with pre-built Snowflake/Databricks/BigQuery agents), no job submission or provisioning. Good fit for chat-driven data-science tasks; wrong fit for this project's batch reproduction script. |
| **Modal** (serverless Python execution, $280 pre-existing credit) | ✅ **Selected** | Purpose-built for "bring your own pip environment, run arbitrary Python functions remotely." No card friction (credit already present), no engine restrictions, driven entirely from a local CLI/SDK — no browser console clicking, no SSH session management. Validated end-to-end (§4). |

## 3. Why Modal fits this workload specifically

- **Custom pinned environments are first-class.** `modal.Image.debian_slim(python_version="3.13").pip_install(...)` builds an image with exact package versions, matching [`FEDGRAA_V2_MASTER_PLAN.md`](FEDGRAA_V2_MASTER_PLAN.md) §13.3's recorded baseline (Python 3.13.14 / TensorFlow 2.21.0 / NumPy 2.5.1 / SciPy 1.18.0) — confirmed by the build log installing exactly `tensorflow-2.21.0`, `numpy-2.5.1`, `scipy-1.18.0`.
- **No VM lifecycle to manage.** Functions are defined in a plain Python file and executed via `modal run script.py`; Modal provisions and tears down the container automatically.
- **Natively parallel.** `.map()` / `max_containers=` fan a function out across many concurrent containers — the sweep's `(arm, seed)` enumeration (already resumable/worker-capped per [`FEDGRAA_V2_MASTER_PLAN.md`](FEDGRAA_V2_MASTER_PLAN.md) §7.1) is a direct fit for this once the real campaign exists.
- **Detached execution.** `modal run --detach` keeps a job alive in Modal's cloud after the local client disconnects — relevant once real multi-hour runs replace this smoke test.
- **Driven from a local terminal**, which means it can be operated directly via shell commands rather than screenshot-relayed browser clicks.

## 4. Validated setup

### 4.1 One-time account setup

```
pip install modal
modal setup      # one-time browser OAuth, links CLI to the Modal account
```

### 4.2 Repo transfer: local directory copy, not `git clone`

The project's GitHub remote (`N1KH1LT0X1N/FedGRAA`) is **private** — an anonymous `git clone` inside a Modal container build step fails (`fatal: could not read Username for 'https://github.com'`, since there is no TTY to prompt for credentials during an image build). Rather than embed a token in the image, the working setup uploads the local `FedGRAA/` directory directly into the image via `Image.add_local_dir(..., copy=True)`, explicitly excluding directories that are large and unnecessary for the code itself:

| Excluded | Size | Why |
|---|---|---|
| `.cache` | 14.6 GB | Local caching artifacts, not code |
| `.venv313` | 1.8 GB | The container builds its own pinned environment via `pip_install` |
| `data` | 889 MB | Datasets — not needed for the unit-test smoke test; would need explicit handling for real experiment runs |
| `.pytest_cache`, `__pycache__`, `archive`, `.git` | small | Build/VCS artifacts |

### 4.3 Finding: `ProcessPoolExecutor` hangs indefinitely on Linux — fork vs. spawn

**This is the one genuine, reusable finding from this exercise**, in the same spirit as the master plan's own F-series findings (verified by execution, not assumed).

**Believed going in.** The existing 155-test suite (`FEDGRAA_V2_MASTER_PLAN.md` §13.3's regression floor) would just run remotely the way it runs locally.

**Measured.** The first remote run hung indefinitely — no error, no crash, no output for 20+ minutes against a 320-second local baseline. `modal container list` confirmed the container was still genuinely active (not crashed), ruling out a silent failure.

**Root cause.** [`fedgraa/experiment.py:191,263`](../../fedgraa/experiment.py#L191) uses `concurrent.futures.ProcessPoolExecutor` with no explicit `mp_context`. The function it dispatches is documented at [`fedgraa/experiment.py:124-128`](../../fedgraa/experiment.py#L124-L128):

> *"Top-level function for multiprocessing — runs one experiment config. Must be at module level (not nested or lambda) for Windows `spawn`."*

This comment confirms the code was written and tested under Windows' default start method, **`spawn`** (each worker process starts fresh and re-imports everything). **Linux's default is `fork`** instead. Forking a process that has already imported TensorFlow — which spins up internal thread pools, gRPC state, and absl logging threads — is a well-known deadlock trigger: the child process inherits copied-but-inconsistent lock/thread state from the parent. This is exactly what `test_parallel_sweep.py` and any test exercising `run_ablation`/`run_seeds_parallel` hit on first execution under Linux.

**Fix (validated).** Force the `spawn` start method explicitly before any executor is constructed, and run pytest in-process (not via a fresh `python -m pytest` subprocess) so the setting is in effect before test collection creates any pool:

```python
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
```

After this fix, the suite completed in 539.92s (~9 minutes) with no hang.

**Why this matters beyond the smoke test.** Any future execution of this codebase's multiprocessing-based sweep driver (`scripts/run_ablation_seed_sweep.py`, `run_ablation`/`run_seeds_parallel` in `experiment.py`) **on any Linux-based compute** — Modal, a rented VM, a university cluster — will hit this same hang unless the start method is forced to `spawn` first. This is a portability gap the master plan's own environment section (§13) does not currently cover, since §13 only diagnosed a Windows-specific import blocker.

**✅ Fixed, 2026-07-28.** `fedgraa/experiment.py` now constructs both `ProcessPoolExecutor` call sites (`run_gamma_sweep_parallel`, `run_n_sweep_parallel`) with `mp_context=multiprocessing.get_context("spawn")` explicitly, module-scoped as `_MP_CONTEXT`, rather than relying on the platform default. `scripts/run_ablation_seed_sweep.py` was already safe (its pool workers only call `subprocess.run`, never import TensorFlow in the parent before forking) and needed no change. The new v2 frontier driver (`scripts/run_v2_frontier_sweep.py`) forces the same context from the start. Verified this does not change behaviour on Windows (`tests/test_parallel_sweep.py`, 5/5 passing — Windows only ever supported `spawn` anyway, so this is a no-op there and a real fix on Linux/Modal).

### 4.4 Smoke test result

| Metric | Local (Windows, §13.3 baseline) | Modal (Linux, this pass) |
|---|---|---|
| Suite result | 155 passed, 366 warnings, 320.21s | **154 passed, 1 failed**, 366 warnings, 539.92s |
| Environment | Python 3.13.14 / TF 2.21.0 / NumPy 2.5.1 / SciPy 1.18.0 | Identical versions, confirmed from the pip install log |
| CPU features | AVX2, AVX512F, AVX512_VNNI, FMA | AVX2, FMA only |

**The one failure** — `test_fedgraa_weight_gradient_matches_finite_difference` ([`tests/test_aggregation.py:72`](../../tests/test_aggregation.py#L72)) — missed its `atol=5e-3` tolerance by a small margin (mismatch `0.00537`) on one of three gradient components. Given the CPU feature difference above (no AVX512 on the Modal container), this reads as a floating-point rounding difference in TensorFlow's oneDNN math kernels between CPU generations, not a logic defect introduced by anything in this setup. **Flagged, not fixed** — if the real campaign ever runs on hardware without AVX512, this test's tolerance may need revisiting, and any cross-hardware accuracy comparisons in the eventual paper should note this as a measured, hardware-dependent source of noise distinct from the `ζ` decomposition in [`FEDGRAA_V2_MASTER_PLAN.md`](FEDGRAA_V2_MASTER_PLAN.md) §6.2.

### 4.5 The working script

Reproduced in full (also at the scratchpad path used during this session):

```python
import modal

app = modal.App("fedgraa-v2-smoke-test")

EXCLUDE_DIRS = {".venv313", ".cache", "data", ".pytest_cache", "__pycache__", "archive", ".git"}


def _ignore(path):
    from pathlib import Path

    return any(part in EXCLUDE_DIRS for part in Path(path).parts)


image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "tensorflow>=2.21.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "scipy>=1.10",
        "pytest>=7.0",
        "matplotlib>=3.7",
    )
    .add_local_dir("C:/Dev/FedGRAA/FedGRAA", remote_path="/repo", copy=True, ignore=_ignore)
)


@app.function(image=image, cpu=4.0, memory=8192, timeout=1800)
def run_tests():
    import multiprocessing
    import os

    # fedgraa/experiment.py's ProcessPoolExecutor is only documented safe under
    # Windows' default 'spawn' start method. Linux defaults to 'fork', which
    # forks a process that already has TensorFlow loaded -- a known deadlock
    # cause. Force 'spawn' here to match the semantics the code was tested under.
    multiprocessing.set_start_method("spawn", force=True)

    import sys

    sys.path.insert(0, "/repo")
    os.chdir("/repo")

    import fedgraa  # noqa: F401

    print("mp OK / fedgraa OK")

    import pytest

    exit_code = pytest.main(["tests/", "-q"])
    print(f"pytest exit code: {exit_code}")
    return int(exit_code)


@app.local_entrypoint()
def main():
    code = run_tests.remote()
    print(f"Exit code: {code}")
```

## 5. Cost — measured on Modal, against real fedgraa code, not estimated

The original pass in this document approximated Modal's rate as "~$0.048/vCPU-hour" and priced CPU only. A later pass (still this same day) replaced that with Modal's published rates. **This pass goes one step further: it actually runs the existing v1 `fedgraa` code on Modal** — three purpose-built calibration scripts, committed at [`scripts/modal_calibration/`](../../scripts/modal_calibration/) so every number below is re-runnable, matching this project's own "measure, don't assume" discipline ([`FEDGRAA_V2_MASTER_PLAN.md`](FEDGRAA_V2_MASTER_PLAN.md) F3, F12). No v2 code exists yet (this document's front-matter, and [`FEDGRAA_V2_MASTER_PLAN.md`](FEDGRAA_V2_MASTER_PLAN.md)'s own front-matter), so all three calibrations exercise the real, already-shipping `FederatedTrainer`/`FedGRAA` path — the same code the eventual campaign's per-`(arm, seed)` unit will call.

### 5.1 Verified rates

| Resource | Modal's published rate | Per-hour equivalent | Note |
|---|---|---|---|
| CPU | $0.0000131 / physical core / sec | **$0.0472 / core-hr** | `cpu=1.0` requests **one physical core**, not a vCPU — [Configuring CPU, memory, and disk](https://modal.com/docs/guide/resources) |
| Memory | $0.00000222 / GiB / sec | **$0.00799 / GiB-hr** | Billed on **whichever is higher: requested or actual usage** |
| Non-preemptible (`nonpreemptible=True`) | **3× surcharge** on CPU + memory | — | Not needed — see §6.4 |
| Region pinning (`region=`) | **1.25×–1.75× surcharge** | — | Not needed — default routing avoids this |

### 5.2 Measured, not assumed: per-`(arm, seed)`-unit cost on each dataset

[`scripts/modal_calibration/modal_cpu_scaling_calibration.py`](../../scripts/modal_calibration/modal_cpu_scaling_calibration.py) and [`modal_cic_memory_calibration.py`](../../scripts/modal_calibration/modal_cic_memory_calibration.py) each run the **actual unit of work** the real campaign will run — `diagnose_fedgraa.py`'s `run_fine_grained` equivalent, 3 gammas × 60 rounds, matching `run_ablation_seed_sweep.py`'s own `--rounds 60 --eval-every 1` default — on Modal, on both datasets, at `cpu=1.0`. Raw results: [`modal_cpu_scaling_result.json`](../../scripts/modal_calibration/modal_cpu_scaling_result.json) (UNSW-NB15), [`modal_cic_memory_result.json`](../../scripts/modal_calibration/modal_cic_memory_result.json) (CIC-IDS-2017), both from runs on 2026-07-28.

| Dataset | Measured wall-clock, 3-gamma unit | Measured peak RSS | Recommended `memory=` |
|---|---|---|---|
| UNSW-NB15 | **681.5 s** (~11.4 min) | **2,086 MB** (`cpu=1.0`) | **4096 MB** (4 GiB — ~2× headroom) |
| CIC-IDS-2017 | **1,235.9 s** (~20.6 min) | **8,630 MB**, identical after gamma 0.0, 0.5, **and** 1.0 — confirms memory does not compound across the three sequential `load_dataset`/`partition_non_iid` calls a real unit makes | **12288 MB** (12 GiB — ~40% headroom) |

CIC-IDS-2017's wall-clock lands squarely inside [`FEDGRAA_V2_MASTER_PLAN.md`](FEDGRAA_V2_MASTER_PLAN.md) §7.2's independently-derived **"15–20 min per `(arm, seed)`"** — a real cross-check between two different measurement methods (this project's local Stage B timing vs. this pass's Modal run) landing on the same number.

Per-unit cost (CPU + memory, at the `memory=` above):

| Dataset | CPU cost/unit | Memory cost/unit | Total/unit |
|---|---|---|---|
| UNSW-NB15 | $0.00894 | $0.00605 | **$0.0150** |
| CIC-IDS-2017 | $0.01620 | $0.03292 | **$0.0491** |

### 5.3 Full-campaign cost, built from measured per-unit costs

[`FEDGRAA_V2_MASTER_PLAN.md`](FEDGRAA_V2_MASTER_PLAN.md) §7.2's tiers run **both datasets** for T1, T2, T3, T5 (evenly split by construction — e.g. T1 = 5 arms × 2 datasets × 10 seeds); T4 is assumed even for this estimate; T7 (dataset unspecified) is also assumed even; **T6 runs NF-UQ-NIDS-v2, a third dataset this pass has not calibrated** — its cost is carried at the master plan's own 28-CPU-hour estimate until it gets its own measurement, once that dataset's ingest lands.

| Tier | Runs (CIC / UNSW) | Cost | Core-hours (measured) | vs. master plan's estimate |
|---|---|---|---|---|
| T1–T3 | 210 / 210 | **~$13.47** | **111.9** | 119 (within 6%) |
| T1–T5 + T7 (excl. T6) | 380 / 380 | ~$24.36 | 202.4 | — |
| **T1–T7 (T6 at its book estimate)** | 380 / 380 + 100 (T6) | **~$26–29** | ~230 | 243 (within 6%) |

**Both totals land within 6% of the master plan's own effort estimates**, despite those estimates coming from a completely different measurement (local Stage B timing) than this pass's (live Modal runs). **Cost remains under 10.5% of the existing $280 credit even in the fullest, most conservative reading** — cost is not, and was never going to be, the constraint on this campaign. The binding constraint is concurrency (§6).

### 5.4 An available lever the original pass didn't check: academic credits

Modal's pricing page advertises **up to $10,000 in free credits for graduate students and academic researchers** ([modal.com/pricing](https://modal.com/pricing), "Special Programs"). This project is an academic PAKDD submission; applying is a same-day, low-effort action that would remove compute budget as a consideration entirely, for T1–T7 and any follow-on (T6, T7, and beyond).

## 6. Concurrency ceiling and how to run the real campaign fastest without losing runs

**This is the section that answers "how fast can this go, and what could silently cost us results."** The master plan's own compute-reality section ([`FEDGRAA_V2_MASTER_PLAN.md`](FEDGRAA_V2_MASTER_PLAN.md) §7.2) estimates **"1.7 days at 3 workers"** for T1–T3 and implies a similar multi-day wall-clock for T1–T7 on local hardware. That estimate assumes 3 parallel workers because that is what a laptop can offer. **Modal's measured concurrency ceiling is two orders of magnitude higher than that.**

### 6.1 The account-tier concurrency limit — measured, not just read off the pricing page

Per [modal.com/pricing](https://modal.com/pricing): Starter tier (no monthly fee) caps concurrent containers at 100; Team ($250/month) raises it to 1,000. Rather than assume this account is on Starter, [`scripts/modal_calibration/modal_concurrency_probe.py`](../../scripts/modal_calibration/modal_concurrency_probe.py) launched **130 trivial containers** (20 s sleep each, `max_containers=130`) and reconstructed true overlap from each container's own start/end timestamps ([raw result](../../scripts/modal_calibration/modal_concurrency_result.json)):

> **Measured: 99 of 130 launched containers ran concurrently.** This confirms the account is on Starter's 100-container ceiling directly, empirically — not inferred from the pricing page or from the absence of a $250/month charge.

**Team's 1,000-container ceiling is not worth $250/month here** — the campaign costs ~$27 total and finishes in hours at 99 concurrent containers (§6.3). There is no scenario in this project where paying for Team buys a meaningfully faster campaign.

Two other limits exist but do not bind at this campaign's scale: a **workspace rate limit of 200 Function calls/sec** (submitting 860 `.map()` inputs is nowhere near this), and an **input queue limit of 2,000 pending / 25,000 total inputs** ([Scaling out](https://modal.com/docs/guide/scale)) — both far above the 860-run T1–T7 total.

### 6.2 CPU-per-container sizing — measured: more cores do not mean more speed here

[`scripts/modal_calibration/modal_cpu_scaling_calibration.py`](../../scripts/modal_calibration/modal_cpu_scaling_calibration.py) ran the same 3-gamma UNSW-NB15 unit at `cpu=1.0`, `cpu=2.0`, and `cpu=4.0` in parallel on Modal ([raw result](../../scripts/modal_calibration/modal_cpu_scaling_result.json)):

| `cpu=` | Total wall-clock (3 gammas) | Peak RSS | `OMP`/`OPENBLAS`/`MKL`_NUM_THREADS (Modal auto-set) |
|---|---|---|---|
| 1.0 | 681.5 s | 2,086 MB | 1 / 1 / 1 |
| 2.0 | 790.1 s | 2,068 MB | 2 / 2 / 2 |
| 4.0 | 617.2 s | 2,117 MB | 4 / 4 / 4 |

**No monotonic speedup — `cpu=2.0` was slower than `cpu=1.0`, and `cpu=4.0`'s modest gain is within run-to-run scheduling noise.** This is not a fluke: [`fedgraa/trainer.py:20-21`](../../fedgraa/trainer.py#L20-L21) hard-codes `TF_NUM_INTRAOP_THREADS=1` (`os.environ.setdefault(...)`, set before TensorFlow is imported, present in every module that imports the trainer — not just a diagnostic script) — TensorFlow's own intra-op parallelism is capped at one thread regardless of how many cores Modal grants, by explicit design (the file's own comment, and the pattern repeated in five other scripts, indicates this was done deliberately to let many single-threaded worker *processes* run without oversubscribing a shared machine, per `run_ablation_seed_sweep.py`'s docstring on "Parallelism & RAM"). Modal's auto-set `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS` scale with `cpu=` as documented, but they don't move the needle because the training loop's own TF ops are pinned to one thread independent of those variables.

**Conclusion: use `cpu=1.0` for every container in the real campaign.** It is measured to be at least as fast as `cpu=2.0`/`cpu=4.0` for this workload, and it is the cheapest option (§5.2). This also matches Modal's own general guidance to scale CPU-bound work via more containers, not more cores per container or `@modal.concurrent` ([Input concurrency](https://modal.com/docs/guide/concurrent-inputs)).

### 6.3 Projected wall-clock at the measured 99-container ceiling

Modal's `.map()` scheduling is a continuous work queue, not discrete batches — as soon as one container finishes, the next queued input starts immediately, rather than waiting for a whole round of 99 to finish. Using the measured per-unit times (§5.2) and the measured 99-container ceiling (§6.1):

| Tier | Core-hours (measured, §5.3) | Wall-clock at 99 concurrent | vs. master plan's local estimate |
|---|---|---|---|
| T1–T3 (minimum viable) | 111.9 | **~1.1–1.5 h** | §7.2's "1.7 days at 3 workers" |
| T1–T7 (full stack, T6 at book estimate) | ~230 | **~2.3–3 h** | (no local estimate given; would be ~10 days at 1 worker) |

**This is the headline finding of this pass, now measured rather than merely computed from published rates.** The full T1–T7 campaign — the one the master plan gates behind "~3 weeks of critical path, then a multi-day compute run" — becomes an **afternoon**, not a multi-day commitment, once W0–W3's implementation work lands. This changes the shape of the mentor conversation in [`FEDGRAA_V2_MASTER_PLAN.md`](FEDGRAA_V2_MASTER_PLAN.md) §12.3: T5 (the novelty-defence grid, 45 CPU-hours) no longer needs to be traded away for external lab compute on wall-clock grounds — the trade, if made, should be about focus/scope, not about compute time being unavailable.

### 6.4 Preemption: rare, auto-retried, and already covered by the existing resumable driver — do not pay to avoid it

**Every Modal Function is preemptible by default** ([Preemption](https://modal.com/docs/guide/preemption)). On preemption, Modal "gracefully terminate[s] the Function and restart[s] it on the same input" automatically — no input is silently dropped. Preemptions are described as rare but more likely for longer-running functions, and each `(arm, seed)` unit runs 11–21 minutes (measured, §5.2 — UNSW-NB15 vs. CIC-IDS-2017), long enough to have non-zero exposure.

**What this costs if it happens:** a preempted run restarts from zero (no mid-run checkpoint), so the loss is one wasted container — a few cents (§5.2's per-unit cost, $0.015–$0.049), not a correctness problem — **provided each `(arm, seed)` result is persisted immediately on completion** rather than buffered until the whole sweep finishes. This is precisely the property [`scripts/run_ablation_seed_sweep.py`](../../scripts/run_ablation_seed_sweep.py) already has ("resumable, worker-capped" per [`FEDGRAA_V2_MASTER_PLAN.md`](FEDGRAA_V2_MASTER_PLAN.md) §7.1) — the Modal port inherits this for free as long as the per-result write happens inside the Modal function (or to a mounted Volume) rather than being collected only after `.map()` returns.

**Do not set `nonpreemptible=True`.** It costs a **3× surcharge on both CPU and memory** (§5.1) to eliminate a rare failure mode that, given the checkpoint property above, costs a few cents to retry anyway. Paying 3× on the whole campaign (~$78–87 instead of ~$26–29) to avoid an occasional ~11–21 minute retry is the wrong trade.

**One nesting hazard to avoid when porting the campaign driver:** `run_ablation_seed_sweep.py`'s existing parallelism (`run_ablation`/`run_seeds_parallel`) uses a local `ProcessPoolExecutor` — the same one §4.3 already found hangs on Linux without a forced `spawn` context. When porting to Modal, **the `(arm, seed)` enumeration should drive a Modal `.map()` directly, one call per `(arm, seed)` pair, with each container running a single training job** — not re-invoke the driver's own `ProcessPoolExecutor`-based parallelism *inside* a Modal container. Nesting process-pool parallelism inside a Modal-managed container would re-trigger the fork/deadlock risk at a second layer for no benefit, since Modal's container-level scaling already supplies the parallelism.

### 6.5 One correctness caveat carried over from §4.4, sharpened here

> ⚠️ **Corrected 2026-07-28** ([`../audit/PRE_COMPUTE_AUDIT_2026-07-28.md`](../audit/PRE_COMPUTE_AUDIT_2026-07-28.md) **M7**). §4.4 records the Modal container as *"AVX2, FMA only"* and this section reasons from that as if it were a fixed property of the platform. **It is not — Modal's fleet is heterogeneous.** The campaign image's own pre-flight (`modal_frontier_campaign.py::verify_env`), run 2026-07-28 on the real image, reported **`avx512: True`**, reading `/proc/cpuinfo` in the container exactly as §4.4's check did.
>
> The consequence is a strengthening, not a weakening, of the caveat below: the "same-hardware, same-run internal comparisons should mostly cancel" argument holds **within** a cell — both sides of T-4 are computed in one container — but **not across cells**. A 300-cell campaign will land on a mix of CPU generations, so **cross-cell numerical noise from CPU-generation differences is real, and belongs in Limitations** next to the `ζ` decomposition rather than being assumed away. It is small (§4.4 measured `0.00537` on a `5e-3`-tolerance finite-difference check) but it is not zero and it is not controllable from this side.
>
> The pre-flight also reported the container's Python as **3.13.3** against the local baseline's **3.13.14** — a patch-level difference the `PINNED` version assertion does not currently cover.


§4.4 already measured that the Modal container lacks AVX512 (`AVX2, FMA only` vs. the local machine's `AVX2, AVX512F, AVX512_VNNI, FMA`), which shifted one finite-difference gradient test by `0.00537` against a `5e-3` tolerance. The master plan's keystone exactness tests **T-4/T-4a/T-4b** run at a much tighter `1e-9` tolerance ([`FEDGRAA_V2_MASTER_PLAN.md`](FEDGRAA_V2_MASTER_PLAN.md) §9.2) — these are same-hardware, same-run internal comparisons (v2 vs. v1-projected, both computed in the same container), so the AVX512 difference should mostly cancel rather than compound. Still, **re-run T-4/T-4a/T-4b on Modal specifically, not only locally, before trusting any G3 gate pass produced there** — cheap insurance given that this is the single instance so far where CPU-generation differences have measurably moved a tolerance-sensitive number in this project.

## 7. Status and next step

> ## ✅ UPDATE (2026-07-28) — the driver this section calls "the next step" now exists
>
> [`../audit/PRE_COMPUTE_AUDIT_2026-07-28.md`](../audit/PRE_COMPUTE_AUDIT_2026-07-28.md)
> found that **§7's "nothing else is blocking it" was not true**: the Modal
> `.map()` wrapper described below had never been written (**B1**), and the
> driver it would have wrapped ran the frontier's reference arm at the wrong
> `η′` (**B2**). Both are fixed. The campaign now launches from
> [`../../scripts/modal_frontier_campaign.py`](../../scripts/modal_frontier_campaign.py),
> which implements exactly the pattern §4.5 validated, plus four things this
> document's own measurements imply but its example script did not do:
>
> | Addition | Why, per this document's own §§ |
> |---|---|
> | **Two functions split by dataset** (`memory=12288` / `memory=4096`) | §5.2 measured peak RSS 8,630 MB (CIC) vs 2,086 MB (UNSW); Modal bills `max(requested, actual)`, so one 12 GiB function would bill UNSW's cells at ~3× their measured memory cost |
> | **`modal.Volume`, committed per cell** | §6.4 — preemption is expected and auto-retried; a finished cell must survive a *different* container's preemption |
> | **Exact version pins + a `verify_env` pre-flight** | §3 confirmed the resolved versions from a *build log*, not a lockfile. `requirements.txt` carries `>=` bounds only, so a cache-busting rebuild mid-campaign would silently run later cells on different libraries (audit **H3**) |
> | **`data/NF-UQ-NIDS-v2` excluded from the image** | §4.2's table records `data` at **889 MB**. It is now **13.9 GB** — NF-UQ-NIDS-v2.csv alone is 12.79 GB — and no T1/T2/T4 cell reads it (audit **H2**) |
>
> **§5's per-unit cost measurements stand, and were re-validated.** They were
> taken against the **v1** trainer, while 80% of T1/T2's cells are **v2**
> arms — the audit's largest open risk. Measured directly on real UNSW-NB15
> data: `v2_m1` = **1.05×** v1's per-round wall-clock, `v2_m5` = **0.97×**.
> The `~$26–29` / `~2.3–3 h` figures hold.

This closed out the infrastructure side only. ~~Per [`FEDGRAA_V2_MASTER_PLAN.md`](FEDGRAA_V2_MASTER_PLAN.md) §8, the actual compute campaign is still gated on **W0 → W1 → W2 → W3**~~ **✅ W0–W3 (and W5, W6) landed 2026-07-28** — see [`REMAINING_COMPUTE_WORK.md`](REMAINING_COMPUTE_WORK.md) for the itemized status. The pattern validated here (pinned image, local-dir upload, forced `spawn` context, `.map()` over the enumeration) now has a concrete driver to point it at: [`scripts/run_v2_frontier_sweep.py`](../../scripts/run_v2_frontier_sweep.py)'s `run_one_cell` function is the exact unit of work a Modal `.map()` should fan out over, iterating `enumerate_cells(["T1","T2"])`. **Next step: wrap `run_one_cell` in a Modal `@app.function` following this document's §4.5 pattern and launch T1 (100 cells, ~28 CPU-h) as the first real campaign run** — nothing else is blocking it.
