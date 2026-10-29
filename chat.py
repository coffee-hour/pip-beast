"""Pip-Beast terminal chat. Runs in any terminal, with or without the satellite layer.

    python chat.py                      # satellites on (default)
    python chat.py --core               # pure Pip-Beast, no satellites
    python chat.py --ckpt checkpoints/best.pt --tokenizer data/tokenizer.json

In chat: /quit to exit, /reset to clear the hormone state, /core and /sat to toggle the layer.
"""
import argparse
import os
import sys

import torch

from generate import complete, load_checkpoint
from tokenizer import BeastTokenizer

AMBER = '\033[38;5;214m'
DIM = '\033[38;5;94m'
RESET = '\033[0m'


def color_ok() -> bool:
    return sys.stdout.isatty() and os.environ.get('NO_COLOR') is None


def paint(text: str, code: str) -> str:
    return f'{code}{text}{RESET}' if color_ok() else text


def banner(use_sat: bool, device: str):
    mode = 'satellites on' if use_sat else 'core only'
    print(paint('PIP-BEAST', AMBER) + paint(f'  |  {device}  |  {mode}', DIM))
    print(paint('/quit to exit  /reset  /core  /sat', DIM))
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='checkpoints/best.pt')
    p.add_argument('--tokenizer', default='data/tokenizer.json')
    p.add_argument('--memory', default='checkpoints/episodes.json')
    p.add_argument('--core', action='store_true', help='disable satellites')
    p.add_argument('--max_new_tokens', type=int, default=200)
    a = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not os.path.exists(a.ckpt):
        sys.exit(f'No checkpoint at {a.ckpt}. Train first: ./run.sh train')
    model, _ = load_checkpoint(a.ckpt, device)
    tok = BeastTokenizer(a.tokenizer)

    orch = None
    try:
        from satellites.orchestrator import Orchestrator
        orch = Orchestrator(model, tok, memory_path=a.memory, device=device,
                            max_new_tokens=a.max_new_tokens)
    except Exception as e:  # satellites are optional; fall back to core
        print(paint(f'satellites unavailable ({e}); running core only', DIM))

    use_sat = orch is not None and not a.core
    banner(use_sat, device)
    history = ''

    while True:
        try:
            line = input(paint('you > ', AMBER)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == '/quit':
            break
        if line == '/reset':
            history = ''
            if orch:
                orch.endo.levels = dict(orch.endo.levels)
                from satellites.endocrine import BASELINE
                orch.endo.levels = dict(BASELINE)
            print(paint('state cleared', DIM))
            continue
        if line == '/core':
            use_sat = False
            print(paint('satellites off', DIM))
            continue
        if line == '/sat':
            if orch is None:
                print(paint('satellites unavailable', DIM))
            else:
                use_sat = True
                print(paint('satellites on', DIM))
            continue

        if use_sat:
            reply = orch.respond(line)
        else:
            prompt = history + f'User: {line}\nPip:'
            reply = complete(model, tok, prompt, a.max_new_tokens, 0.8, 40, device).strip()
            history = prompt + ' ' + reply + '\n'
        print(paint('pip > ', AMBER) + reply)
        print()


if __name__ == '__main__':
    main()
