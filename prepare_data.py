"""Build the tokenizer and memmap token files for train_5090.py.

Writes data/tokenizer.json, data/train.bin, data/val.bin (uint16 token ids). Pure Python BPE
with the same algorithm as Pip. Note: BPE training is pure Python and can be slow on very large
inputs; keep the input to a few hundred MB or less for a first run.

    python prepare_data.py --input input.txt --out_dir data --vocab_size 32000
"""
import argparse
import os

import numpy as np

from tokenizer import BPETokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out_dir', default='data')
    p.add_argument('--vocab_size', type=int, default=32000)
    p.add_argument('--val_fraction', type=float, default=0.01)
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    with open(a.input, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    print(f'training BPE: {len(text):,} characters, vocab {a.vocab_size}')
    tok = BPETokenizer.train(text, a.vocab_size)
    tok.save(os.path.join(a.out_dir, 'tokenizer.json'))

    ids = np.array(tok.encode(text), dtype=np.uint32)
    assert ids.max() < 65536, 'vocab must fit uint16 for these bins'
    n_val = max(1, int(len(ids) * a.val_fraction))
    ids.astype(np.uint16)[:-n_val].tofile(os.path.join(a.out_dir, 'train.bin'))
    ids.astype(np.uint16)[-n_val:].tofile(os.path.join(a.out_dir, 'val.bin'))
    print(f'tokens: total {len(ids):,}, train {len(ids) - n_val:,}, val {n_val:,}')


if __name__ == '__main__':
    main()
