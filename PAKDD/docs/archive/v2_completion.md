# v2 Completion Report: What's Done, What's Left, How to Finish It

*Generated 2026-07-21 after systematic audit + implementation pass.*  
*Setup instructions verified against official Microsoft, NVIDIA, and TensorFlow docs.*

---

## The Current State

Every measurable phase from the speedup plan is **implemented, tested, compiling clean**. Four bug fixes were found and shipped during the audit implementation pass.

### ✅ Shipped (6/6 Phases)

| Phase | What | Impact | Files |
|-------|------|--------|-------|
| **Phase 9** | Sweep-level parallelism | ~3-5× on sweeps | `experiment.py`, `run_experiments.py`, `__main__.py`, `tests/test_parallel_sweep.py` |
| **Phase 12** | Pre-computed epoch indices | ~1.2 s/run saved | `trainer.py`, `baselines.py` |
| **Phase 13** | Remove `.numpy()` from unused loss | ~0.6 s/run saved | `trainer.py`, `baselines.py` |
| **Phase 14** | Checkpoint/resume | No wasted compute on crash | `trainer.py`, `experiment.py`, `run_experiments.py`, `__main__.py`, `tests/test_checkpoint.py` |
| **Phase 16** | TF threading tune (`INTRAOP=1`) | ~5-10% per-step | `trainer.py` (1 line) |
| **Phase 19 infra** | GPU-ready code (no code changes needed) | — | All source files |

### 🐛 Bugs Fixed During Audit

1. **`__main__.py` n-sweep broken** (`gamma=` vs `gammas=`) — `run_n_sweep()` was called with `gamma=args.gamma` (a float), but the function expects `gammas=...` (a sequence). At runtime this raises `TypeError`.
2. **Dead code in `resume_or_init`** — `global_params[:] = ckpt.global_params` replaced list elements with checkpoint references, then a redundant `p[...] = cp` loop copied the same data into the same objects (no-op).
3. **`__main__.py` missing `--parallel` and `--checkpoint-dir` flags** — the `python -m fedgraa` entry point didn't expose Phase 9 or Phase 14 functionality.
4. **`__main__.py` gamma-sweep had unused `--gamma` arg** and n-sweep output only showed one gamma column.

---

## What's Left: Phase 19 — GPU via WSL2

> ## ⛔ SUPERSEDED (2026-07-23) — GPU was tested and REJECTED
> 
> **This entire section predates the actual GPU benchmark and its projections
> are wrong.** The setup was completed on an RTX 5050 (Blackwell, CC 12.0) via
> WSL2 and measured — the GPU is **25% *slower* than CPU** for this model
> (0.87 ms/step GPU vs 0.70 ms/step CPU), and 3 numerical-precision tests fail
> on GPU. The model is too small (~42K params, batch 64) to overcome kernel-
> launch overhead; it is launch-bound, not compute-bound.
> 
> **Every performance claim below is falsified:** the projected "0.15–0.30 ms/step
> (5–10× vs CPU)" and "v2 + RTX 5050 GPU: ~15–25 minutes → 192–320×" did **not**
> materialize. GPU is *not* a remaining action item — it is a closed, rejected
> path. The setup steps below are retained only as a historical record of what
> was attempted (the scripts now live in [`../scripts/archive/gpu/`](../scripts/archive/gpu/)).
> 
> **Canonical, measured truth:** see the GPU Report in
> [`speedup_plan_v2.md`](speedup_plan_v2.md) and the decision write-up in
> [`GPU_DECISION_AND_NEXT_STEPS.md`](GPU_DECISION_AND_NEXT_STEPS.md). Speed is
> already solved on CPU (full reproduction ≈ 2 h); nothing is gated on GPU.

This is the **single remaining action item**. It's purely infrastructure — no code changes. The codebase is already GPU-ready: `@tf.function(jit_compile=True)` automatically uses the GPU when TF detects one.

### Prerequisites (Your Current State — 2026-07-21)

| Item | Status | Source |
|------|--------|--------|
| NVIDIA GeForce RTX 5050 Laptop GPU | ✅ Present (CC 12.0, Blackwell) | `nvidia-smi` |
| NVIDIA Windows Driver | ✅ Driver 592.82, CUDA 13.1 | `nvidia-smi` |
| Windows 11 | ✅ Present | `winver` |
| WSL2 | ❌ **Not installed** | `wsl --status` → "not installed" |
| Ubuntu in WSL2 | ❌ Not installed | WSL2 missing |
| `uv` in WSL2 | ❌ Not installed | WSL2 missing |
| Python 3.13 in WSL2 | ❌ Not installed | WSL2 missing |
| Project on WSL2 filesystem | ❌ Not copied | WSL2 missing |
| `tensorflow[and-cuda]` in WSL2 | ❌ Not installed | WSL2 missing |

### ⚠️ Verified Compatibility Notes

Cross-checked against official documentation (Microsoft, NVIDIA, TensorFlow):

| Concern | Official Source | Verdict |
|---------|---------------|---------|
| RTX 5050 (Blackwell, CC 12.0) support | NVIDIA: CC 12.0 listed; TF: "CC 3.5, 5.0, 6.0, 7.0, 7.5, 8.0 and higher" | ✅ Supported ("and higher" includes 12.0) |
| TF 2.21 GPU via `[and-cuda]` on WSL2 | TF pip guide: `pip install tensorflow[and-cuda]` for GPU | ✅ This is the documented approach |
| TF 2.21 Hermetic CUDA + WSL2 GPU detection | GitHub issue [#113541](https://github.com/tensorflow/tensorflow/issues/113541) | ⚠️ **Known bug**: Hermetic CUDA libs need explicit `LD_LIBRARY_PATH` (fix below) |
| CUDA toolkit installation in WSL2 | NVIDIA: "Do NOT install Linux GPU driver. Use WSL-Ubuntu repo. Install only `cuda-toolkit-12-x`, never `cuda` or `cuda-drivers`." | ⚠️ **Optional for TF 2.21** (Hermetic CUDA bundles its own; only needed for `nvcc` etc.) |
| Python version | TF 2.21 build config: Python 3.10–3.13, CUDA 12.5, cuDNN 9.3 | Ubuntu 22.04 ships Python 3.10 ✅ |

### Storage Strategy: `uv` + `.venv313`

This setup uses **`uv`** (Rust-based Python package manager by Astral) for maximum speed and minimal storage:

- **Global content-addressed cache** (`~/.cache/uv`): Every wheel is downloaded *once* and hardlinked into all virtualenvs. Five projects using the same `tensorflow` version = one copy on disk.
- **`.venv313`**: Named to indicate Python 3.13 (TF 2.21 supports 3.10–3.13; 3.13 is the best fit). The venv itself is lightweight — mostly hardlinks into uv's cache.
- **No pip/venv needed**: `uv` replaces pip, venv, and pyenv. It manages Python versions and dependencies in one tool.
- **To free space later**: `uv cache clean` removes all cached wheels; `uv cache prune` removes unused entries only.
- **To move cache off C: drive**: `export UV_CACHE_DIR=/mnt/d/uv-cache` before running `uv` commands.

### Step 1: Run the Automated Setup Script (Administrator PowerShell)

This is your starting point — WSL2 is **not installed**. Run the script or the commands below **as Administrator**:

```powershell
# Run in ADMINISTRATOR PowerShell

# Install WSL2 + VirtualMachinePlatform + Ubuntu (one command)
wsl --install

# Update WSL kernel
wsl --update
```

Reboot if prompted. After reboot, Ubuntu will finish installing on first launch — create your Linux username/password when asked.

### Step 2: Inside WSL2 — Install `uv` + Python + Project

After reboot, open PowerShell (regular, not admin) and enter WSL:

```powershell
wsl --distribution Ubuntu
```

Inside the Ubuntu terminal:

```bash
# === Inside WSL2 Ubuntu ===

# 2a. Update packages
sudo apt-get update
sudo apt-get upgrade -y

# 2b. Install build tools (curl for uv install)
sudo apt-get install -y curl build-essential

# 2c. Install uv (standalone — manages Python itself, no pip needed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version

# 2d. Install Python 3.13 via uv (does NOT touch system Python)
uv python install 3.13
uv python list  # confirm 3.13 is available

# 2e. Copy project to WSL2 filesystem (CRITICAL: NOT /mnt/c — that's slow!)
cp -r /mnt/c/Dev/Research/PAKDD ~/PAKDD
cd ~/PAKDD

# 2f. Create virtualenv with a name that says what's inside
uv venv .venv313 --python 3.13

# 2g. Install TF with GPU support (Hermetic CUDA — bundles CUDA 12.5 + cuDNN 9.3)
# uv's global cache means tensorflow is stored ONCE even across projects
uv pip install 'tensorflow[and-cuda]==2.21.0'

# 2h. Install remaining dependencies
uv pip install numpy pandas scikit-learn matplotlib pytest

# 2i. Optional: move uv cache to save C: drive space
# mkdir -p /mnt/d/uv-cache
# echo 'export UV_CACHE_DIR=/mnt/d/uv-cache' >> ~/.bashrc
# source ~/.bashrc
```

### Step 3: Fix TF 2.21 GPU Detection on WSL2

TF 2.21 uses Hermetic CUDA — CUDA libs are in `site-packages/nvidia/*/lib/`. TF's internal loader doesn't find them on WSL2 without preloading. The fix is to load the CUDA shared libraries with `RTLD_GLOBAL` **before** importing tensorflow.

Create `fix_gpu_import.py` in the project root:

```bash
cat > ~/PAKDD/fix_gpu_import.py << 'PYEOF'
import os, site, ctypes

sp = site.getsitepackages()[0]
libs = ['cuda_runtime', 'cublas', 'cudnn', 'cusolver', 'cusparse', 'curand', 'nvjitlink']
for lib in libs:
    lib_dir = os.path.join(sp, 'nvidia', lib, 'lib')
    if not os.path.isdir(lib_dir):
        continue
    for f in sorted(os.listdir(lib_dir)):
        if '.so' in f and not f.endswith('.so'):
            try:
                ctypes.CDLL(os.path.join(lib_dir, f), mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass
            break
PYEOF
```

Prepend this to any script that imports tensorflow:

```bash
# Instead of: python3 -c "import tensorflow as tf"
# Use:        python3 -c "exec(open('fix_gpu_import.py').read()); import tensorflow as tf"

# Verify GPU is detected
cd ~/PAKDD && source .venv313/bin/activate
python3 -c "exec(open('fix_gpu_import.py').read()); import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Expected: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

**Note:** The alternatives (symlinks in TF dir or `LD_LIBRARY_PATH`) were tested and did NOT work on this setup. Only the `ctypes.RTLD_GLOBAL` approach resolved the `Cannot dlopen some GPU libraries` error.

### Step 4: Run Tests

```bash
cd ~/PAKDD
uv run python -m pytest tests/ -v
```

All 38+ tests should pass. The fused XLA steps will automatically compile under XLA:GPU.

### Step 5: Run a Quick Benchmark

```bash
cd ~/PAKDD
uv run python3 -c "
import time, numpy as np, tensorflow as tf
from fedgraa.model import MLP, _fused_client_step, build_tf_optimizer
layer_sizes = (78, 200, 100, 50, 15)
model = MLP(layer_sizes, seed=0)
opt = build_tf_optimizer(0.01, model._model.trainable_variables)
X = np.random.randn(64, 78).astype(np.float32)
y = np.random.randint(0, 15, size=64).astype(np.int32)
_fused_client_step(tf.constant(X), tf.constant(y), model._model, opt, tf.constant(64))
t0 = time.time()
for _ in range(100):
    _fused_client_step(tf.constant(X), tf.constant(y), model._model, opt, tf.constant(64))
t1 = time.time()
per_step = (t1 - t0) / 100 * 1000
print(f'GPU per-step: {per_step:.3f} ms')
print(f'Speedup vs CPU (1.50 ms): {1.50 / per_step:.1f}x')
"
```

Expected: **~0.15-0.30 ms per step** (5-10× vs 1.50 ms CPU).

### Step 6: Run a Full Reproduction

```bash
cd ~/PAKDD
uv run python run_experiments.py --rounds 30 --eval-every 10 --parallel
uv run python run_experiments.py --parallel --checkpoint-dir .checkpoints/
```

Expected: Gamma sweep in **~5-8 minutes**, full reproduction in **~15-25 minutes**.

### Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `tf.config.list_physical_devices('GPU')` returns empty | Hermetic CUDA libs not preloaded | Run `exec(open('fix_gpu_import.py').read())` before `import tensorflow` |
| `Could not load dynamic library 'libcudart.so'` | CUDA runtime not found by loader | Same fix — ctypes RTLD_GLOBAL preloads it |
| `Cannot dlopen some GPU libraries` | NVIDIA libs not in dynamic linker path | Use the ctypes preload fix (symlinks + LD_LIBRARY_PATH were tested and didn't work) |
| `CUDA_ERROR_NO_DEVICE` | WSL2 kernel too old | `wsl --update` from Windows PowerShell |
| Training is slow on GPU | Data on `/mnt/c` (slow Windows mount) | Copy project to `~/PAKDD` on ext4 |
| `nvidia-smi` not found in WSL2 | Path issue | `/usr/lib/wsl/lib/nvidia-smi` |
| `uv: command not command` | uv not in PATH | `source ~/.bashrc` or restart terminal |
| Disk space concerns | uv cache growing large | `uv cache clean` (safe after install) or set `UV_CACHE_DIR` to another drive |

### uv Disk Space Management

```bash
# See where uv stores its cache
uv cache dir

# See cache size
du -sh $(uv cache dir)

# Remove ONLY unused cache entries (safe to run periodically)
uv cache prune

# Remove ALL cached wheels (re-download on next install)
uv cache clean

# Move cache to another drive (set before running uv)
export UV_CACHE_DIR=/mnt/d/uv-cache
# Or make it permanent:
echo 'export UV_CACHE_DIR=/mnt/d/uv-cache' >> ~/.bashrc

# The venv itself is small — most bytes are hardlinks to cache
du -sh .venv313

# To see what tensorflow + nvidia packages are in the venv:
uv pip list | grep -E 'tensorflow|nvidia'
```

---

## The Final Numbers

```
                              Full reproduction (both datasets, all sweeps)
                              ───────────────────────────────────────────────
Pre-v1 (original code):       ~80 hours
v1 (fused XLA, CPU only):     ~6.4 hours          → 12.5×
v2 CPU + parallel sweeps:     ~2 hours             → 40×  (3.2× on v1)   ← final, shipped
v2 + RTX 5050 GPU + parallel: ✗ REJECTED — GPU measured 25% SLOWER than CPU (see banner above)
```

*The GPU line was a pre-benchmark projection that did not hold. CPU + parallel
sweeps (~2 h, 40×) is the final path.*

---

## Summary of All Changes

### Code Changes (6 files)

| File | What |
|------|------|
| `fedgraa/__main__.py` | Fixed n-sweep bug, added `--parallel`/`--checkpoint-dir` flags |
| `fedgraa/trainer.py` | Phase 14 checkpoint/resume, Phase 12 pre-computed indices, Phase 13 `.numpy()` removal, Phase 16 INTRAOP=1, fixed resume_or_init dead code |
| `fedgraa/experiment.py` | Phase 9 sweep parallelism, Phase 14 checkpoint wiring |
| `fedgraa/baselines.py` | Phase 12 pre-computed indices, Phase 13 `.numpy()` removal |
| `run_experiments.py` | `--parallel` and `--checkpoint-dir` CLI flags |
| `docs/speedup_plan_v2.md` | Final audited plan with shipped/rejected/remaining status |

### New Files (2)

| File | What |
|------|------|
| `tests/test_checkpoint.py` | 4 tests for Phase 14 save/load/atomicity |
| `tests/test_parallel_sweep.py` | 2 tests for Phase 9 picklability |

### Infrastructure Setup (explored, then rejected)

| File | What |
|------|------|
| `scripts/archive/gpu/setup_wsl2_gpu.ps1` | Automated WSL2 installation — **archived; GPU was tested and rejected (25% slower than CPU), see banner in Phase 19 above** |

All 25 Python files compile clean. Zero `TODO`/`FIXME`/`HACK` comments exist.

---

## Sources Verified

| Source | URL | What It Confirms |
|--------|-----|-------------------|
| Microsoft WSL Install Guide | `https://learn.microsoft.com/en-us/windows/wsl/install` | `wsl --install` is the correct command; Ubuntu is default |
| NVIDIA CUDA on WSL Guide | `https://docs.nvidia.com/cuda/wsl-user-guide/index.html` | Do NOT install Linux GPU driver; use WSL-Ubuntu repo; install `cuda-toolkit-12-x` only |
| TensorFlow pip install guide | `https://www.tensorflow.org/install/pip` | `pip install tensorflow[and-cuda]` is correct for GPU; TF 2.21 supports Python 3.10-3.13 |
| TensorFlow build config | `https://www.tensorflow.org/install/source` | TF 2.21 uses CUDA 12.5, cuDNN 9.3 |
| TF 2.21 PyPI page | `https://pypi.org/project/tensorflow/2.21.0/` | Hermetic CUDA packages: `nvidia-cublas-cu12`, `nvidia-cuda-runtime-cu12`, etc. |
| NVIDIA GPU compute capability | `https://developer.nvidia.com/cuda-gpus` | RTX 5050 has Compute Capability 12.0 (Blackwell) |
| TF 2.21 GPU detection on WSL2 | Tested on this machine 2026-07-21 | Symlinks + LD_LIBRARY_PATH did NOT work; `ctypes.RTLD_GLOBAL` preload is required |

---

*End of v2 completion report. All code changes are landed, tested, and verified.*

*Update (2026-07-23): the "remaining action" — GPU via WSL2 — was completed and
**rejected**: the RTX 5050 measured 25% slower than CPU for this model. There is
no remaining action; the CPU + parallel-sweep path (~2 h full reproduction, 40×)
is final. See [`speedup_plan_v2.md`](speedup_plan_v2.md) and
[`GPU_DECISION_AND_NEXT_STEPS.md`](GPU_DECISION_AND_NEXT_STEPS.md). The WSL2/GPU
setup scripts are archived in [`../scripts/archive/gpu/`](../scripts/archive/gpu/).*
