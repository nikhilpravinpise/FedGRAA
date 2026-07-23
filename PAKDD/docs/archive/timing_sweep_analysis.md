# Timing Analysis: Why the Sweep Feels Slow

This note explains why the current real-data sweep still takes a long time on a newer laptop, even though the measured wall time is better than the older machine.

## What Was Measured

The most useful timing probe was a **real-data CIC-IDS-2017 gamma sweep with `--rounds 1` and cached preprocessing**. That run produced:

- `FedAvg`: about 150s
- `Static Reparam.`: about 152s
- `FedGRAA g=0.0`: about 153s
- `FedGRAA g=0.5`: about 144s
- `FedGRAA g=1.0`: about 144s
- Total sweep wall time: **748s**

So the headline result is:

- The new laptop is **faster** than the older one for this workload.
- The sweep still feels slow because one "run" is already expensive, and the sweep repeats that cost five times.

## The Actual Workload

The apparent slowdown is mostly a workload issue, not a raw-laptop issue.

For CIC-IDS-2017, the current pipeline does the following:

- Loads 14 federated clients after one-attack-per-client partitioning.
- Trains on roughly **115,022** client-side training examples total per run after the 90/10 split.
- Uses `batch_size = 64` and `local_epochs = 2`.
- That works out to about **3,612 minibatches per round**.

With `--rounds 1`, that is still a full pass through the entire federated setup, not a tiny smoke test.

## Why It Is Slow

### 1. The training loop is serial

The core loop in [fedgraa/trainer.py](../fedgraa/trainer.py) iterates over all clients in Python, then iterates over batches inside each client update. There is no parallel client execution here.

Relevant paths:

- [fedgraa/trainer.py](../fedgraa/trainer.py)
- [fedgraa/model.py](../fedgraa/model.py)

### 2. Every batch uses eager TensorFlow gradient computation

Each local step goes through `MLP.loss_and_grad(...)`, which uses `tf.GradientTape()` for a forward/backward pass on the current minibatch. That is correct for the reproduction, but it is not a cheap loop.

Relevant path:

- [fedgraa/model.py](../fedgraa/model.py)

### 3. FedGRAA adds extra validation optimization work

For FedGRAA, every round also runs a weight-update loop with `weight_epochs = 5`, recomputing the composite validation gradient multiple times.

Relevant paths:

- [fedgraa/config.py](../fedgraa/config.py)
- [fedgraa/aggregation.py](../fedgraa/aggregation.py)

### 4. The sweep multiplies the cost

The gamma sweep does not run one model and vary a flag inside it. It runs a full `FedAvg`, a full `Static` run, and then three full `FedGRAA` runs for `gamma = 0, 0.5, 1.0`.

Relevant path:

- [fedgraa/experiment.py](../fedgraa/experiment.py)

### 5. Caching only skips preprocessing, not training

The `.cache` directory removes the big CSV load and normalization step, but after that the model still has to train. So cache helps startup time, but it does not change the dominant cost of the sweep.

Relevant path:

- [fedgraa/data.py](../fedgraa/data.py)

## Why The New Laptop Still Looks "Slow"

If you compare the old and new machine directly, the current numbers actually show improvement:

- Old laptop: about 220s per run
- New laptop: about 150s per run

That is a real speedup.

The reason it still feels slow is that the pipeline uses the result of one run as if it were a lightweight unit of work, but each run contains:

- a full federated client loop,
- minibatch training across all client data,
- validation recomputation,
- and, for FedGRAA, extra weight-optimization steps.

## Practical Interpretation

This means the bottleneck is not a bad laptop. It is a combination of:

- large real datasets,
- serial Python orchestration,
- TensorFlow eager execution,
- and a sweep design that repeats the full training cost several times.

## If You Want Faster Timing Checks

For quick progress checks, use one of these instead of a full gamma sweep:

```powershell
python run_experiments.py --rounds 1 --eval-every 1 --datasets CIC-IDS-2017 --sweep gamma --data-dir data/CIC-IDS-2017 --cache-dir .cache
```

Or go even lighter with the synthetic path:

```powershell
python run_experiments.py --rounds 5 --eval-every 5 --datasets CIC-IDS-2017 --sweep gamma
```

## Bottom Line

The new laptop is not the problem. The current sweep is expensive by construction, and the code executes that expense mostly in serial. The measured 150s per run is already an improvement over 220s, but the sweep still feels slow because it is doing real federated training, not just a small benchmark.