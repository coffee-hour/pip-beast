#!/usr/bin/env bash
# One-shot setup for Pip-Beast on an NVIDIA RTX 5090 (or any CUDA GPU). Safe to re-run.
# Creates .venv, installs a CUDA 12.8 build of PyTorch (needed for Blackwell sm_120),
# installs requirements, and verifies the GPU is visible.
set -euo pipefail

cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
CUDA_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "error: $PY not found. Install Python 3.10+ and retry." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi not found. Install the NVIDIA driver (580+ for RTX 50-series) and retry." >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo "creating .venv"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip wheel >/dev/null
python -m pip install --index-url "$CUDA_INDEX" torch
python -m pip install -r requirements.txt

python - <<'EOF'
import torch
assert torch.cuda.is_available(), 'CUDA not available to PyTorch'
p = torch.cuda.get_device_properties(0)
print(f'GPU ok: {p.name}, {p.total_memory / 2**30:.1f} GiB, sm_{p.major}{p.minor}, torch {torch.__version__}')
EOF

echo
echo "setup complete. Next: ./run.sh prepare, ./run.sh train, ./run.sh chat"
