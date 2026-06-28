# FedGRAA — Faithful Reproduction

A complete, from-scratch, dependency-light reproduction of the paper
**"Federated Gradient-Refined Adaptive Averaging for Intrusion Detection
Systems" (FedGRAA)**.

Everything is implemented **exactly as the paper specifies** — the tanh/softmax
network, the Adam optimizers, the median-frequency validation split, the
one-attack-per-client non-IID partition, and the
`η=0.01 / E=2 / B=64 / T=200 / η'=0.1 / E'=5` hyperparameters. The design
alternatives raised in [understanding/07_my_questions.md](understanding/07_my_questions.md)
(ReLU, SMOTE, SGD, stratified split, …) are deliberately **not** substituted.
See [plan.md](plan.md) for the full fidelity contract.

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
