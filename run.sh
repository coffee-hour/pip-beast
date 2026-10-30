#!/usr/bin/env bash
# Pip-Beast runner. Usage:
#   ./run.sh prepare [input.txt]              build data/tokenizer.json, train.bin, val.bin
#   ./run.sh train [--preset fast|beast] ...  train (auto-resumes from checkpoints/)
#   ./run.sh ui [--host H] [--port P]         web UI on http://localhost:8000
#   ./run.sh chat [--core]                    terminal chat (satellites on unless --core)
#   ./run.sh test                             quick model sanity check (parameter count, forward pass)
#   ./run.sh presets                          print preset architectures and parameter counts
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "no .venv found; running setup.sh first"
  ./setup.sh
fi
# shellcheck disable=SC1091
source .venv/bin/activate

cmd="${1:-help}"
shift || true

case "$cmd" in
  prepare)
    input="${1:-input.txt}"
    if [ ! -f "$input" ]; then
      echo "error: $input not found. Pass a UTF-8 text file: ./run.sh prepare path/to/text.txt" >&2
      exit 1
    fi
    mkdir -p data
    python prepare_data.py --input "$input" --out_dir data
    ;;
  train)
    exec python train_5090.py "$@"
    ;;
  ui)
    exec python web_ui.py "$@"
    ;;
  chat)
    exec python chat.py "$@"
    ;;
  presets)
    python model.py
    ;;
  test)
    python model.py
    python -c "import torch; from model import BeastConfig, PipBeast; m=PipBeast(BeastConfig(n_layers=2)).cuda(); x=torch.randint(0,32000,(2,64)).cuda(); l,_=m(x,x); print('forward ok', tuple(l.shape)); lc,_=m.forward_cached(x[:, :63]); lc2,_=m.forward_cached(x[:, 63:], *[None])" || true
    ;;
  *)
    echo "usage: ./run.sh {prepare|train|ui|chat|presets|test} [args]"
    echo "  train presets: ./run.sh train --preset fast   (about 222M params)"
    echo "                 ./run.sh train --preset beast  (about 1.04B params, default)"
    exit 1
    ;;
esac
