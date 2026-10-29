#!/usr/bin/env bash
# Pip-Beast runner. Usage:
#   ./run.sh prepare [input.txt]   build data/tokenizer.json, train.bin, val.bin
#   ./run.sh train [extra args]    train (auto-resumes from checkpoints/)
#   ./run.sh chat [--core]         terminal chat (satellites on unless --core)
#   ./run.sh test                  quick model sanity check (parameter count, forward pass)
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
  chat)
    exec python chat.py "$@"
    ;;
  test)
    python model.py
    python -c "import torch; from model import BeastConfig, PipBeast; m=PipBeast(BeastConfig(n_layers=2)).cuda(); x=torch.randint(0,32000,(2,64)).cuda(); l,_=m(x,x); print('forward ok', tuple(l.shape))"
    ;;
  *)
    echo "usage: ./run.sh {prepare|train|chat|test} [args]"
    exit 1
    ;;
esac
