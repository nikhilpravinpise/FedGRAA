# FedGRAA — Faithful Reproduction

A complete, from-scratch, dependency-light reproduction of the paper
**"Federated Gradient-Refined Adaptive Averaging for Intrusion Detection
Systems" (FedGRAA)**, targeting PAKDD submission.

Everything is implemented **exactly as the paper specifies** — the tanh/softmax
network, the Adam optimizers, the median-frequency validation split, the
one-attack-per-client non-IID partition, and the
`η=0.01 / E=2 / B=64 / T=200 / η'=0.1 / E'=5` hyperparameters. The design
alternatives raised in [understanding/07_my_questions.md](understanding/07_my_questions.md)
(ReLU, SMOTE, SGD, stratified split, …) are deliberately **not** substituted.
See [docs/plan.md](docs/plan.md) for the full fidelity contract.

---

## For collaborators — read this first

### Environment

- **Python 3.9 – 3.12** (3.13 has no stable TensorFlow wheel; the implementation
  is pure NumPy for exactly this reason — it runs on any Python ≥ 3.9).
- **RAM: 4 GB minimum, 8 GB recommended** when loading the full CIC-IDS-2017
  dataset (2.8 M rows). See [Data setup](#data-setup) for the cache workflow that
  keeps peak RAM low on subsequent runs.
- No GPU required. Training is CPU-only pure NumPy.

### Install

```powershell
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `scikit-learn`, `pytest`. No TensorFlow, no PyTorch.

### Datasets you need

| Dataset | Status | Location |
|---|---|---|
| **CIC-IDS-2017** | ✅ Already in repo | `data/CIC-IDS-2017/` (8 CSV files) |
| **UNSW-NB15** | ⬇️ Needs download | See [UNSW-NB15 setup](#unsw-nb15) below |

---

## Data setup

### CIC-IDS-2017 (already present)

The 8 CSV files are already in `data/CIC-IDS-2017/`. The first time you run an
experiment the loader reads all 2.8 M rows (~95 s), then saves a small `.npy`
cache. Every subsequent run loads the cache in under 3 seconds.

**Always pass `--cache-dir .cache`** — this is the single most important flag
for local development:

```powershell
# First run: reads CSVs (~95 s), saves cache
python run_experiments.py --data-dir data/CIC-IDS-2017 --cache-dir .cache --datasets CIC-IDS-2017

# All subsequent runs: loads cache (~3 s)
python run_experiments.py --data-dir data/CIC-IDS-2017 --cache-dir .cache --datasets CIC-IDS-2017
```

The `.cache/` directory is git-ignored. Each collaborator builds their own local cache.

> **Why this matters:** Before the cache was added, the pipeline was running
> `apply(pd.to_numeric)` across 78 columns × 2.8 M rows, which triggered a NumPy
> `vstack` that tried to allocate 518 MB on top of the already-loaded DataFrame
> and crashed with `MemoryError` on machines with < 8 GB free RAM. This is fixed —
> the loader now uses `select_dtypes` + `np.place` — but the cache avoids the
> 95-second load entirely on re-runs.

### UNSW-NB15

Download the 4 CSV files from the
[official UNSW page](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
(~175 MB total). Create the directory and put them there:

```
data/
  UNSW-NB15/
    UNSW_NB15_training-set.csv
    UNSW_NB15_testing-set.csv
    ... (any *.csv files work)
```

Then run with `--datasets UNSW-NB15`. The loader auto-detects the `attack_cat`
label column. UNSW has only 43 features and ~100 K rows — it runs roughly
3× faster than CIC.

---

## Running experiments

### Choosing what to run

| Flag | What it controls |
|---|---|
| `--datasets CIC-IDS-2017` | Run only CIC (skip UNSW). Default: both. |
| `--sweep gamma` | Figure 1 only (5 method runs). |
| `--sweep n` | Figures 2-3 only (9 runs). |
| `--sweep both` | All 14 runs (default). |
| `--rounds N` | Training rounds. Paper uses 200. Use 30-50 for quick checks. |
| `--eval-every N` | Print metrics every N rounds. Default 20. |
| `--cache-dir DIR` | Cache/reload preprocessed arrays here. **Always use this.** |

### Time estimates (CPU-only, ~7 s/round on a typical laptop)

| Command | Runs | Est. time |
|---|---|---|
| `--rounds 30 --sweep gamma` | 5 | ~18 min |
| `--rounds 200 --sweep gamma` | 5 | ~2 h |
| `--rounds 200 --sweep both` | 14 | ~5.5 h |

Plan accordingly — the full suite is an overnight job on CPU. Use `--sweep gamma`
for validation; only run `--sweep both` for the final submission results.

### Recommended workflows

**Quickest sanity check (confirms the pipeline runs end-to-end):**
```powershell
python run_experiments.py `
  --data-dir data/CIC-IDS-2017 --cache-dir .cache `
  --datasets CIC-IDS-2017 --rounds 30 --eval-every 10 --sweep gamma
```

**Full Figure 1 reproduction (CIC only, ~2 h):**
```powershell
python run_experiments.py `
  --data-dir data/CIC-IDS-2017 --cache-dir .cache `
  --datasets CIC-IDS-2017 --sweep gamma
```

**All figures, both datasets (overnight run):**
```powershell
python run_experiments.py `
  --data-dir data/CIC-IDS-2017 --cache-dir .cache
```

**Single experiment via the module CLI:**
```powershell
python -m fedgraa run --dataset CIC-IDS-2017 --method fedgraa `
  --gamma 1 --n 3 --rounds 50 `
  --data-dir data/CIC-IDS-2017 --cache-dir .cache
```

**Gamma sweep via the module CLI:**
```powershell
python -m fedgraa gamma-sweep --dataset CIC-IDS-2017 --rounds 50 `
  --data-dir data/CIC-IDS-2017 --cache-dir .cache
```

---

## Expected results

The paper's qualitative result, on the rare ("last-n") classes:

```
FedGRAA(γ=1)  >  Static Reparam.  >  FedAvg ≈ 0%
```

- **FedAvg** consistently drives rare-class accuracy to near zero — it
  over-weights large clients.
- **Static Reparameterization** recovers rare-class accuracy (last-n ≈ 85-95%)
  but sacrifices overall accuracy.
- **FedGRAA(γ=1)** achieves the best rare-class accuracy while recovering
  most of the overall accuracy.
- Raising γ trades overall accuracy for higher rare-class accuracy; FedGRAA's
  advantage grows as more clients hold rare data (n = 5, 7).

---

## Tests

Run the full suite before pushing:

```powershell
python -m pytest tests -q
```

27 tests covering gradient correctness, fidelity to Tables 1 & 2, partitioning,
aggregation math, and an end-to-end federated run. All should pass in ~30 s on
any machine.

What the tests verify:
- **Gradient correctness** — back-prop and the FedGRAA weight gradient both
  match central finite differences.
- **Fidelity** — dataset specs and the synthetic generator reproduce the exact
  counts of Tables 1 & 2; the partition yields one attack per client with equal
  benign and a 90/10 split; the median-frequency split is correct.
- **Aggregation** — FedAvg and Static weights have the right values; the γ
  extremes select the right objective.
- **End-to-end** — a short federated run lifts rare-class accuracy with FedGRAA
  above FedAvg.

---

## How it maps to the paper

| Paper element | Code |
|---|---|
| Network `200→100→50` tanh, softmax | [fedgraa/model.py](fedgraa/model.py) — `MLP` |
| Adam (η=0.01 local, η'=0.1 weights) | [fedgraa/model.py](fedgraa/model.py) — `Adam` |
| Tables 1 & 2, hyperparameters | [fedgraa/config.py](fedgraa/config.py) |
| Real loader + synthetic generator + cache | [fedgraa/data.py](fedgraa/data.py) |
| One-attack-per-client, 90/10, median split (eq. 8) | [fedgraa/partition.py](fedgraa/partition.py) |
| FedAvg (eq. 5), Static (eq. 7), FedGRAA (eq. 8-10) | [fedgraa/aggregation.py](fedgraa/aggregation.py) |
| Algorithm 1 federated loop | [fedgraa/trainer.py](fedgraa/trainer.py) |
| All-class & last-n accuracy | [fedgraa/metrics.py](fedgraa/metrics.py) |
| Figure 1 / 2 / 3 sweeps | [fedgraa/experiment.py](fedgraa/experiment.py) |

### The core math (FedGRAA weight learning)

The global model is a linear blend of client models, `P = Σ_k w_k · P_k`, so
`∂P/∂w_k = P_k`. For any validation loss `L'(P)`:

```
∂L'/∂w_k = ⟨ ∇_P L'(P), P_k ⟩      (dot product over all parameters)
```

Each round, FedGRAA:

1. builds the temporary global model `θ'_g = Σ_k w_k θ_k` (eq. 8 setup),
2. computes `L_a = min(L_h, L_m)` on the frequent buckets and `L_u` on the rare
   bucket (eq. 8),
3. blends them: `L' = (1−γ)·L_a + γ·L_u` (eq. 9),
4. back-props `L'` through the aggregated parameters, dots with each client's
   params to get `∇_w L'`, and takes an Adam step on `w` (eq. 10), repeated
   `E'=5` times.

The sub-gradient of `min(L_h, L_m)` follows whichever branch is smaller.
Verified against finite differences in [tests/test_aggregation.py](tests/test_aggregation.py).

---

## Why pure NumPy?

The paper uses Keras. TensorFlow has no stable wheel for Python 3.13, and adding
a heavy framework dependency creates reproducibility friction for collaborators.
The **identical** network math is implemented directly in NumPy with hand-written
forward/backward passes and Adam. Architecture, activations, loss, optimizer, and
every hyperparameter match the paper — only the execution engine differs. This
keeps the project deterministic, easy to test, and installable with four lines in
`requirements.txt`.

The trade-off: training is single-threaded CPU. Adding multi-threading for the 14
client updates is not a speedup on this hardware because NumPy's OpenBLAS already
uses multiple threads internally — adding Python-level threading creates BLAS
contention and is ~30% *slower* than sequential. The cache + `--sweep gamma` flag
give the best practical workflow.

---

## Project layout

```
fedgraa/
  config.py        # hyperparameters + Table 1/2 dataset specs
  model.py         # MLP forward/backprop + Adam (from scratch)
  data.py          # real CSV loader (with NPY cache) + synthetic generator
  partition.py     # non-IID partition + median-frequency split
  aggregation.py   # FedAvg / Static / FedGRAA
  trainer.py       # Algorithm 1 loop (with ETA output)
  metrics.py       # all-class / last-n accuracy
  experiment.py    # Figure 1/2/3 sweeps
  __main__.py      # CLI: python -m fedgraa ...
data/
  CIC-IDS-2017/    # 8 CSV files (already present)
  UNSW-NB15/       # put downloaded CSVs here
.cache/            # auto-created NPY cache (git-ignored)
tests/             # 27 unit + gradient + e2e tests
docs/
  plan.md          # design + fidelity contract
  PAKDD_Final_Version_Submit (3).md
run_experiments.py # main entry point — reproduce all headline tables
requirements.txt
```


---

## Why a from-scratch NumPy implementation?

The paper uses the Keras Sequential API. TensorFlow has no stable wheel for the
installed interpreter (Python 3.13), so the **identical** network and math are
implemented directly in NumPy with hand-written forward / back-propagation and a
hand-written Adam optimizer. The architecture, activations, loss, optimizer, and
every hyperparameter match the paper byte-for-byte — only the execution engine
differs. This also keeps the project deterministic, fast to test, and free of
heavy dependencies.

---

## Install

```powershell
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `scikit-learn`, `pytest`.

---

## Quick start (synthetic data, no downloads)

The repo ships a synthetic data generator whose per-class counts reproduce the
"Used" column of Tables 1 and 2, so every experiment runs end-to-end with zero
downloads.

```powershell
# Reproduce Figure 1 (gamma sweep, n = 3)
python -m fedgraa gamma-sweep --dataset CIC-IDS-2017 --rounds 200

# Reproduce Figures 2-3 (robustness as n grows)
python -m fedgraa n-sweep --dataset CIC-IDS-2017 --gamma 1

# A single configuration
python -m fedgraa run --dataset UNSW-NB15 --method fedgraa --gamma 1 --n 3

# Reproduce all headline tables for both datasets
python run_experiments.py
```

For a fast smoke run, lower the round count: `--rounds 40 --eval-every 10`.

---

## Using the real datasets

Download the CIC-IDS-2017 / UNSW-NB15 CSV exports and point the loader at the
directory containing them:

```powershell
python -m fedgraa gamma-sweep --dataset CIC-IDS-2017 --data-dir path\to\CIC-IDS-2017
```

The loader cleans `inf`/`NaN`, label-encodes onto the paper's class order,
standard-scales the features, and caps frequent classes at 10,000 — exactly the
Section-5 pipeline. If `--data-dir` is omitted (or contains no CSVs), the
synthetic generator is used automatically.

---

## How it maps to the paper

| Paper element | Code |
|---|---|
| Network `200→100→50` tanh, softmax | [fedgraa/model.py](fedgraa/model.py) — `MLP` |
| Adam (η=0.01 local, η'=0.1 weights) | [fedgraa/model.py](fedgraa/model.py) — `Adam` |
| Tables 1 & 2, hyperparameters | [fedgraa/config.py](fedgraa/config.py) |
| Real loader + synthetic generator | [fedgraa/data.py](fedgraa/data.py) |
| One-attack-per-client, 90/10, median split (eq. 8) | [fedgraa/partition.py](fedgraa/partition.py) |
| FedAvg (eq. 5), Static (eq. 7), FedGRAA (eq. 8-10) | [fedgraa/aggregation.py](fedgraa/aggregation.py) |
| Algorithm 1 federated loop | [fedgraa/trainer.py](fedgraa/trainer.py) |
| All-class & last-n accuracy | [fedgraa/metrics.py](fedgraa/metrics.py) |
| Figure 1 / 2 / 3 sweeps | [fedgraa/experiment.py](fedgraa/experiment.py) |

### The core math (FedGRAA weight learning)

The global model is a linear blend of client models, `P = Σ_k w_k · P_k`, so
`∂P/∂w_k = P_k`. For any validation loss `L'(P)`:

```
∂L'/∂w_k = ⟨ ∇_P L'(P), P_k ⟩      (a dot product over all parameters)
```

Each round, FedGRAA:

1. builds the temporary global model `θ'_g = Σ_k w_k θ_k` (eq. 8 setup),
2. computes `L_a = min(L_h, L_m)` on the frequent buckets and `L_u` on the rare
   bucket (eq. 8),
3. blends them, `L' = (1−γ)L_a + γ L_u` (eq. 9),
4. back-props `L'` to the aggregated parameters, dots with each client's params
   to get `∇_w L'`, and takes an Adam step on `w` (eq. 10), repeated `E'=5`
   times.

The sub-gradient of `min(L_h, L_m)` follows whichever branch is smaller. This is
verified against finite differences in
[tests/test_aggregation.py](tests/test_aggregation.py).

---

## Tests

```powershell
python -m pytest tests -q
```

The suite covers:

- **Gradient correctness** — back-prop and the FedGRAA weight gradient both match
  central finite differences.
- **Fidelity** — dataset specs and the synthetic generator reproduce the exact
  counts of Tables 1 & 2; the partition yields one attack per client with equal
  benign and a 90/10 split; the median-frequency split is correct.
- **Aggregation** — FedAvg and Static weights have the right values and normalize
  to 1; the γ extremes select the right objective.
- **End-to-end** — a short federated run lifts rare-class accuracy with FedGRAA
  above FedAvg.

---

## Expected behaviour

Reproducing the paper's qualitative result, on the rare ("last-n") classes:

```
FedGRAA(γ=1)  >  Static Reparam.  >  FedAvg ≈ 0
```

Raising `γ` trades a small amount of overall accuracy for higher rare-class
accuracy; FedGRAA's advantage grows as more clients hold rare data (`n = 5, 7`).

---

## Project layout

```
fedgraa/
  config.py        # hyperparameters + Table 1/2 dataset specs
  model.py         # MLP forward/backprop + Adam (from scratch)
  data.py          # real CSV loader + synthetic generator
  partition.py     # non-IID partition + median-frequency split
  aggregation.py   # FedAvg / Static / FedGRAA
  trainer.py       # Algorithm 1 loop
  metrics.py       # all-class / last-n accuracy
  experiment.py    # Figure 1/2/3 sweeps
  __main__.py      # CLI: python -m fedgraa ...
tests/             # full unit + gradient + e2e suite
run_experiments.py # reproduce all headline tables
plan.md            # design + fidelity contract
requirements.txt
```
