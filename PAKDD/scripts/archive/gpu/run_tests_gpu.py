import os, site, ctypes, sys

# Preload CUDA libs before anything imports tensorflow
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Now import and run tests
import pytest
sys.exit(pytest.main(sys.argv[1:] or ['tests/']))
