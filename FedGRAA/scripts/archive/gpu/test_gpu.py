"""GPU detection test for WSL2."""
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

import tensorflow as tf  # noqa: E402

gpu = tf.config.list_physical_devices("GPU")
print("GPU devices:", gpu)
print("TF version:", tf.__version__)
print("Built with CUDA:", tf.sysconfig.get_build_info()["is_cuda_build"])
print("CUDA version:", tf.sysconfig.get_build_info()["cuda_version"])
print("cuDNN version:", tf.sysconfig.get_build_info()["cudnn_version"])
