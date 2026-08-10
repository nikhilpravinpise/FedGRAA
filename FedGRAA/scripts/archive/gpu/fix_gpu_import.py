"""Run this before importing tensorflow to ensure GPU is found on WSL2.

Usage:
  python3 -c "exec(open('fix_gpu_import.py').read())"
  # then import tensorflow as usual

Or just: import fix_gpu_import  (when installed as a package)
"""
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
