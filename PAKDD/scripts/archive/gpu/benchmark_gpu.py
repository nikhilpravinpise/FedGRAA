"""GPU timing benchmark for PAKDD fused XLA step."""
import os
import site
import ctypes

sp = site.getsitepackages()[0]
for lib in ["cuda_runtime", "cublas", "cudnn", "cusolver", "cusparse", "curand", "nvjitlink"]:
    lib_dir = os.path.join(sp, "nvidia", lib, "lib")
    if not os.path.isdir(lib_dir):
        continue
    for f in sorted(os.listdir(lib_dir)):
        if ".so" in f and not f.endswith(".so"):
            try:
                ctypes.CDLL(os.path.join(lib_dir, f), mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass
            break

import time
import numpy as np
import tensorflow as tf  # noqa: E402

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

from fedgraa.model import MLP, _fused_client_step, build_tf_optimizer  # noqa: E402

layer_sizes = (78, 200, 100, 50, 15)
model = MLP(layer_sizes, seed=0)
opt = build_tf_optimizer(0.01, model._model.trainable_variables)
X = np.random.randn(64, 78).astype(np.float32)
y = np.random.randint(0, 15, size=64).astype(np.int32)

# --- Measure XLA compile (first call, includes CUDA PTX JIT) ---
print("=== XLA Compile (first call, expect PTX JIT delay) ===")
t0 = time.perf_counter()
loss = _fused_client_step(tf.constant(X), tf.constant(y), model._model, opt, tf.constant(64))
t1 = time.perf_counter()
print(f"  First call (compile + PTX JIT): {t1-t0:.2f}s")
print(f"  Loss: {loss.numpy():.4f}")

# --- Measure cached steps ---
print("\n=== Cached XLA step timing ===")
# warmup
for _ in range(10):
    _fused_client_step(tf.constant(X), tf.constant(y), model._model, opt, tf.constant(64))

N = 500
t0 = time.perf_counter()
for _ in range(N):
    _fused_client_step(tf.constant(X), tf.constant(y), model._model, opt, tf.constant(64))
t1 = time.perf_counter()
per_step = (t1 - t0) / N * 1000
print(f"  {N} steps: {t1-t0:.3f}s total, {per_step:.4f} ms/step")
print(f"  Document claims CPU: 1.50 ms/step")
print(f"  Speedup vs CPU claim: {1.50 / per_step:.1f}x")

# --- Simulate 14 clients x 258 iters ---
print("\n=== Simulated round (14 clients, 258 iters each) ===")
total_steps = 14 * 258
model2 = MLP(layer_sizes, seed=0)
opt2 = build_tf_optimizer(0.01, model2._model.trainable_variables)
for _ in range(10):
    _fused_client_step(tf.constant(X), tf.constant(y), model2._model, opt2, tf.constant(64))

t0 = time.perf_counter()
for _ in range(total_steps):
    _fused_client_step(tf.constant(X), tf.constant(y), model2._model, opt2, tf.constant(64))
t1 = time.perf_counter()
total_time = t1 - t0
print(f"  Total: {total_time:.3f}s")
print(f"  Per-step: {total_time/total_steps*1000:.4f} ms")
print(f"  Per-client (258 iters): {total_time/14*1000:.2f} ms")
print(f"  Per-round: {total_time:.2f} s")
print(f"  Document CPU claim per-round: ~7s")
print(f"  Speedup vs CPU claim: {7.0 / total_time:.1f}x")
