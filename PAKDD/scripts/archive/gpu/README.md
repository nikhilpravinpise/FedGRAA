# Archived: GPU exploration (RTX 5050 via WSL2)

**Status: explored and rejected. Do NOT use these for reproduction.**

We evaluated running the FedGRAA fused XLA step on an NVIDIA RTX 5050 Laptop GPU
(Blackwell, CC 12.0) via WSL2 + `tensorflow[and-cuda]==2.21.0`.

**Result: the GPU was 25% *slower* than CPU for this workload** (0.87 ms/step
GPU vs 0.70 ms/step CPU), and 3 numerical-precision tests fail on GPU due to
cuDNN-vs-oneDNN FP32 differences. The model is far too small
(`78→200→100→50→15`, ~42K params, batch 64) for a GPU to overcome kernel-launch
overhead — it is launch-bound, not compute-bound. CPU + oneDNN + XLA is the
optimal path. Full reproduction already runs in ~2 h on CPU, so nothing about
the research is gated on GPU acceleration.

These files are kept as **evidence for that negative result** — deleting them
would make the "we tried GPU, it was slower" finding unreproducible.

| File | What it was |
|---|---|
| `benchmark_gpu.py` | CPU-vs-GPU per-step / per-round timing harness (produced the 25%-slower number) |
| `fix_gpu_import.py` | ctypes `RTLD_GLOBAL` CUDA preload — the workaround for TF 2.21's Hermetic-CUDA GPU-detection bug on WSL2 (symlinks + `LD_LIBRARY_PATH` did not work; only this did) |
| `test_gpu.py` | GPU detection / smoke check |
| `run_tests_gpu.py` | Run the pytest suite under XLA:GPU |
| `setup_wsl2_gpu.ps1` | Automated WSL2 + Ubuntu install (run as Administrator) |
| `setup_gpu_wsl.sh` | In-WSL2 setup: uv + Python 3.13 + `tensorflow[and-cuda]` + the preload fix |

**Canonical write-up of the measurement and decision:**
[`../../../docs/speedup_plan_v2.md`](../../../docs/speedup_plan_v2.md) (GPU Report section) and
[`../../../docs/GPU_DECISION_AND_NEXT_STEPS.md`](../../../docs/GPU_DECISION_AND_NEXT_STEPS.md).
