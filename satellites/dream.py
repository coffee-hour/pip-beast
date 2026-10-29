"""Dream: offline consolidation for hippocampus episodes. Run between sessions, not during chat.

Drops near-duplicate episodes (cosine at or above a threshold), keeps the most salient survivors
up to a budget, and rewrites the local episode file. Nothing leaves the machine.

    python -m satellites.dream checkpoints/episodes.json --keep 500
"""
import argparse

import torch

from satellites.hippocampus import Hippocampus


def consolidate(hip: Hippocampus, keep: int = 500, dup_threshold: float = 0.95) -> dict:
    before = len(hip.episodes)
    ranked = sorted(hip.episodes, key=lambda e: e['salience'], reverse=True)
    kept, kept_vecs = [], []
    for ep in ranked:
        v = torch.tensor(ep['vec'])
        if kept_vecs and float(torch.stack(kept_vecs).matmul(v).max()) >= dup_threshold:
            continue
        kept.append(ep)
        kept_vecs.append(v)
        if len(kept) >= keep:
            break
    kept.sort(key=lambda e: e['t'])
    hip.episodes = kept
    hip.save()
    return {'before': before, 'after': len(kept), 'dropped': before - len(kept)}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('path')
    p.add_argument('--keep', type=int, default=500)
    p.add_argument('--dup', type=float, default=0.95)
    a = p.parse_args()
    print(consolidate(Hippocampus(a.path), a.keep, a.dup))
