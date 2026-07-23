<#
.SYNOPSIS
    One-time WSL2 + GPU setup for PAKDD reproduction.
.DESCRIPTION
    Installs WSL2, updates kernel, installs Ubuntu, and guides you through
    setting up TensorFlow with GPU support using uv for storage isolation.
    
    Uses `uv` (Rust-based Python package manager) for:
      - Global content-addressed cache (wheels downloaded once, hardlinked everywhere)
      - Named venv (.venv313) — lightweight, most bytes are hardlinks to cache
      - No pip/venv/pyenv needed — uv replaces all three
    
    Verified against:
      - Microsoft WSL install guide (learn.microsoft.com/en-us/windows/wsl/install)
      - NVIDIA CUDA on WSL guide (docs.nvidia.com/cuda/wsl-user-guide)
      - TF pip install guide (tensorflow.org/install/pip)
      - uv docs (docs.astral.sh/uv)
    
    Run this in an ADMINISTRATOR PowerShell prompt.
.NOTES
    A reboot may be required after this script completes.
    After reboot, continue with the Ubuntu setup steps printed at the end.
#>

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  PAKDD WSL2 + GPU Setup (uv)" -ForegroundColor Cyan
Write-Host "  Verified against official docs" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# === Step 1: Install WSL2 ===
Write-Host "[1/3] Installing WSL2..." -ForegroundColor Yellow
try {
    wsl --install
    Write-Host "  WSL2 installed successfully." -ForegroundColor Green
} catch {
    Write-Host "  wsl --install failed. Trying manual feature enable..." -ForegroundColor Yellow
    dism /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
    dism /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
    wsl --set-default-version 2
}

# === Step 2: Update WSL kernel ===
Write-Host "[2/3] Updating WSL kernel..." -ForegroundColor Yellow
try {
    wsl --update
} catch {
    Write-Host "  Could not update WSL kernel (may not be needed)." -ForegroundColor Yellow
}

# === Step 3: Verify ===
Write-Host "[3/3] Verifying installation..." -ForegroundColor Yellow
try {
    wsl --status | Out-Null
    Write-Host "  WSL status: OK" -ForegroundColor Green
} catch {
    Write-Host "  WSL may need a reboot to complete installation." -ForegroundColor Yellow
}

Write-Host "========================================" -ForegroundColor Green
Write-Host "  WSL2 installed. Ubuntu will launch on first boot." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "NEXT STEPS (Ubuntu inside WSL2):" -ForegroundColor Cyan
Write-Host "1. REBOOT if prompted." -ForegroundColor Yellow
Write-Host "2. Ubuntu finishes installing on first launch (create user/password)." -ForegroundColor Yellow
Write-Host "3. Inside Ubuntu, run these commands:" -ForegroundColor Yellow
Write-Host ""
Write-Host "   # Update + install curl (for uv install)" -ForegroundColor White
Write-Host "   sudo apt-get update && sudo apt-get upgrade -y"
Write-Host "   sudo apt-get install -y curl build-essential"
Write-Host ""
Write-Host "   # Install uv (manages Python + venv + pip in one tool)" -ForegroundColor White
Write-Host "   curl -LsSf https://astral.sh/uv/install.sh | sh"
Write-Host "   source ~/.bashrc"
Write-Host "   uv python install 3.13"
Write-Host ""
Write-Host "   # Copy project (CRITICAL: NOT /mnt/c — that's slow!)" -ForegroundColor White
Write-Host '   cp -r /mnt/c/Dev/Research/PAKDD ~/PAKDD'
Write-Host "   cd ~/PAKDD"
Write-Host ""
Write-Host "   # Create isolated venv (named .venv313)" -ForegroundColor White
Write-Host "   uv venv .venv313 --python 3.13"
Write-Host ""
Write-Host "   # Install TF with GPU (uv caches globally — one copy on disk)" -ForegroundColor White
Write-Host "   uv pip install 'tensorflow[and-cuda]==2.21.0'"
Write-Host "   uv pip install numpy pandas scikit-learn matplotlib pytest"
Write-Host ""
Write-Host "   # Fix TF GPU detection (ctypes RTLD_GLOBAL preload)" -ForegroundColor White
Write-Host "   source .venv313/bin/activate" -ForegroundColor White
Write-Host '   cat > fix_gpu_import.py << PYEOF' -ForegroundColor White
Write-Host '   import os, site, ctypes' -ForegroundColor White
Write-Host "   sp = site.getsitepackages()[0]" -ForegroundColor White
Write-Host '   libs = ["cuda_runtime","cublas","cudnn","cusolver","cusparse","curand","nvjitlink"]' -ForegroundColor White
Write-Host "   for lib in libs:" -ForegroundColor White
Write-Host "       lib_dir = os.path.join(sp, 'nvidia', lib, 'lib')" -ForegroundColor White
Write-Host "       if not os.path.isdir(lib_dir): continue" -ForegroundColor White
Write-Host "       for f in sorted(os.listdir(lib_dir)):" -ForegroundColor White
Write-Host "           if '.so' in f and not f.endswith('.so'):" -ForegroundColor White
Write-Host "               try: ctypes.CDLL(os.path.join(lib_dir, f), mode=ctypes.RTLD_GLOBAL)" -ForegroundColor White
Write-Host "               except: pass; break" -ForegroundColor White
Write-Host "   PYEOF" -ForegroundColor White
Write-Host ""
Write-Host "   # Verify GPU (run fix before import)" -ForegroundColor White
Write-Host '   python3 -c "exec(open('"'"'fix_gpu_import.py'"'"').read()); import tensorflow as tf; print(tf.config.list_physical_devices('"'"'GPU'"'"'))"' -ForegroundColor White
Write-Host ""
Write-Host "   # Run tests" -ForegroundColor White
Write-Host "   uv run python -m pytest tests/ -v"
Write-Host ""
Write-Host "STORAGE NOTES:" -ForegroundColor Magenta
Write-Host "- uv stores ONE copy of each wheel in ~/.cache/uv" -ForegroundColor Magenta
Write-Host "- The .venv313 is mostly hardlinks — almost no extra disk used" -ForegroundColor Magenta
Write-Host "- To move cache off C: drive: export UV_CACHE_DIR=/mnt/d/uv-cache" -ForegroundColor Magenta
Write-Host "- To free space: uv cache prune (safe), uv cache clean (nuclear)" -ForegroundColor Magenta
Write-Host ""
Write-Host "See docs/v2_completion.md for troubleshooting and benchmarks."
