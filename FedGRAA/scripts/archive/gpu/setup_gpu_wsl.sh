#!/usr/bin/env bash
set -euo pipefail

echo '========================================'
echo '  FedGRAA GPU Setup — run inside WSL'
echo '========================================'

echo '[1/8] Updating packages...'
sudo apt-get update -qq
sudo apt-get upgrade -y -qq
sudo apt-get install -y -qq curl build-essential

echo '[2/8] Installing uv...'
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version

echo '[3/8] Installing Python 3.13 via uv...'
uv python install 3.13

echo '[4/8] Copying project to WSL2...'
cp -r /mnt/c/Dev/FedGRAA/FedGRAA ~/FedGRAA
cd ~/FedGRAA

echo '[5/8] Creating .venv313...'
uv venv .venv313 --python 3.13

echo '[6/8] Installing tensorflow[and-cuda] (~3.5 GB download, 10-20 min)...'
uv pip install tensorflow[and-cuda]==2.21.0

echo '[7/8] Installing remaining deps...'
uv pip install numpy pandas scikit-learn matplotlib pytest

echo '[8/8] Creating GPU fix script...'
source .venv313/bin/activate
cat > fix_gpu_import.py << 'PYEOF'
import os, site, ctypes
sp = site.getsitepackages()[0]
libs = ['cuda_runtime','cublas','cudnn','cusolver','cusparse','curand','nvjitlink']
for lib in libs:
    lib_dir = os.path.join(sp, 'nvidia', lib, 'lib')
    if not os.path.isdir(lib_dir): continue
    for f in sorted(os.listdir(lib_dir)):
        if '.so' in f and not f.endswith('.so'):
            try: ctypes.CDLL(os.path.join(lib_dir, f), mode=ctypes.RTLD_GLOBAL)
            except: pass
            break
PYEOF

echo ''
echo '========================================'
echo '  Setup complete! Verify with:'
echo '========================================'
echo ''
echo '  cd ~/FedGRAA'
echo '  source .venv313/bin/activate'
echo '  python3 -c "exec(open(\"fix_gpu_import.py\").read()); import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))"'
