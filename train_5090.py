"""Pip-Beast training loop tuned for a single 32GB RTX 5090 (Blackwell, sm_120).

Features: bf16 autocast, optional torch.compile, fused AdamW, cosine LR with warmup, gradient
accumulation, gradient checkpointing toggle, memmap token loader (uint16 .bin), MFU and VRAM
telemetry, optional wandb, safetensors-free torch checkpoints with top-k retention and auto-resume.

Data: expects data/train.bin and data/val.bin (uint16 token ids, e.g. from prepare_data.py).

Examples:
    python train_5090.py                       # defaults, auto-resume if checkpoints exist
    python train_5090.py --batch 8 --accum 8   # effective 64 x seq_len tokens per step
    python train_5090.py --grad_ckpt --seq_len 4096
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from model import BeastConfig, PipBeast

# RTX 5090 dense BF16 tensor throughput (no sparsity), TFLOPS. Used for MFU.
PEAK_BF16_TFLOPS_5090 = 209.5


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data')
    p.add_argument('--out_dir', default='checkpoints')
    p.add_argument('--vocab_size', type=int, default=32000)
    p.add_argument('--seq_len', type=int, default=2048)
    p.add_argument('--batch', type=int, default=8, help='micro-batch size')
    p.add_argument('--accum', type=int, default=8, help='gradient accumulation steps')
    p.add_argument('--max_steps', type=int, default=100000)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--min_lr_ratio', type=float, default=0.1)
    p.add_argument('--warmup_steps', type=int, default=1000)
    p.add_argument('--weight_decay', type=float, default=0.1)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--eval_interval', type=int, default=500)
    p.add_argument('--eval_iters', type=int, default=20)
    p.add_argument('--log_interval', type=int, default=10)
    p.add_argument('--save_interval', type=int, default=1000)
    p.add_argument('--keep_checkpoints', type=int, default=3)
    p.add_argument('--compile', action='store_true', default=True)
    p.add_argument('--no_compile', dest='compile', action='store_false')
    p.add_argument('--grad_ckpt', action='store_true', help='activation checkpointing')
    p.add_argument('--use_8bit_adam', action='store_true', help='bitsandbytes AdamW8bit')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--seed', type=int, default=1337)
    return p.parse_args()


def get_lr(step, a):
    if step < a.warmup_steps:
        return a.lr * (step + 1) / a.warmup_steps
    prog = min(1.0, (step - a.warmup_steps) / max(1, a.max_steps - a.warmup_steps))
    min_lr = a.lr * a.min_lr_ratio
    return min_lr + 0.5 * (a.lr - min_lr) * (1 + math.cos(math.pi * prog))


def load_bin(path):
    return np.memmap(path, dtype=np.uint16, mode='r')


def get_batch(data, batch, seq_len, device):
    ix = torch.randint(len(data) - seq_len - 1, (batch,))
    x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + seq_len].astype(np.int64)) for i in ix])
    return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)


def build_optimizer(model, a, device):
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{'params': decay, 'weight_decay': a.weight_decay},
              {'params': no_decay, 'weight_decay': 0.0}]
    if a.use_8bit_adam:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(groups, lr=a.lr, betas=(0.9, 0.95))
    fused = device.startswith('cuda')
    return torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95), fused=fused)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, a, device):
    model.eval()
    out = {}
    for name, data in (('train', train_data), ('val', val_data)):
        losses = torch.zeros(a.eval_iters)
        for k in range(a.eval_iters):
            x, y = get_batch(data, a.batch, a.seq_len, device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(out_dir, step, model, optimizer, cfg, a, best_val):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    state = {'step': step, 'model': raw.state_dict(), 'optimizer': optimizer.state_dict(),
             'config': cfg.__dict__, 'best_val': best_val, 'torch_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    tmp = out_dir / f'step_{step:07d}.pt.tmp'
    torch.save(state, tmp)
    os.replace(tmp, out_dir / f'step_{step:07d}.pt')
    torch.save({'config': cfg.__dict__, 'model': raw.state_dict()}, out_dir / 'best.pt')
    ckpts = sorted(out_dir.glob('step_*.pt'))
    for old in ckpts[:-a.keep_checkpoints]:
        old.unlink()


def latest_checkpoint(out_dir):
    ckpts = sorted(Path(out_dir).glob('step_*.pt'))
    return ckpts[-1] if ckpts else None


def main():
    a = get_args()
    assert torch.cuda.is_available(), 'CUDA GPU required. Run setup.sh to check the environment.'
    device = 'cuda'
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name} | {props.total_memory / 2**30:.1f} GiB | sm_{props.major}{props.minor}')

    train_data = load_bin(Path(a.data_dir) / 'train.bin')
    val_path = Path(a.data_dir) / 'val.bin'
    val_data = load_bin(val_path) if val_path.exists() else train_data

    cfg = BeastConfig(vocab_size=a.vocab_size, max_seq_len=a.seq_len, grad_checkpointing=a.grad_ckpt)
    model = PipBeast(cfg).to(device)
    n_params = model.num_params()
    print(f'Model: {n_params / 1e9:.3f}B parameters')

    optimizer = build_optimizer(model, a, device)
    step, best_val = 0, float('inf')
    resume = latest_checkpoint(a.out_dir)
    if resume is not None:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        step, best_val = ckpt['step'], ckpt.get('best_val', float('inf'))
        torch.set_rng_state(ckpt['torch_rng'])
        if ckpt.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
        print(f'Resumed from {resume.name} at step {step}')

    run = None
    if a.wandb:
        import wandb
        run = wandb.init(project='pip-beast', config=vars(a))

    fwd_model = torch.compile(model) if a.compile else model
    tokens_per_step = a.batch * a.accum * a.seq_len
    flops_per_token = 6 * n_params  # forward + backward approximation
    peak_flops = PEAK_BF16_TFLOPS_5090 * 1e12
    torch.cuda.reset_peak_memory_stats()

    print(f'Tokens per optimizer step: {tokens_per_step:,}')
    model.train()
    t_last = time.time()
    while step < a.max_steps:
        lr = get_lr(step, a)
        for g in optimizer.param_groups:
            g['lr'] = lr

        for _ in range(a.accum):
            x, y = get_batch(train_data, a.batch, a.seq_len, device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, loss = fwd_model(x, y)
            (loss / a.accum).backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % a.log_interval == 0:
            torch.cuda.synchronize()
            dt = (time.time() - t_last) / a.log_interval
            t_last = time.time()
            tok_s = tokens_per_step / dt
            mfu = 100.0 * flops_per_token * tok_s / peak_flops
            vram = torch.cuda.max_memory_allocated() / 2**30
            print(f'step {step:7d} | loss {loss.item():.4f} | lr {lr:.2e} | gnorm {gnorm:.2f} | '
                  f'{tok_s:,.0f} tok/s | MFU {mfu:.1f}% | VRAM peak {vram:.1f} GiB')
            if run:
                run.log({'loss': loss.item(), 'lr': lr, 'tok_s': tok_s, 'mfu': mfu, 'vram_gib': vram}, step=step)

        if step % a.eval_interval == 0 or step == a.max_steps:
            losses = estimate_loss(fwd_model, train_data, val_data, a, device)
            print(f'eval step {step} | train {losses["train"]:.4f} | val {losses["val"]:.4f}')
            if run:
                run.log({'eval/train': losses['train'], 'eval/val': losses['val']}, step=step)
            if losses['val'] < best_val:
                best_val = losses['val']

        if step % a.save_interval == 0 or step == a.max_steps:
            save_checkpoint(a.out_dir, step, model, optimizer, cfg, a, best_val)
            print(f'saved checkpoint at step {step} (best val {best_val:.4f})')

    if run:
        run.finish()
    print('done')


if __name__ == '__main__':
    main()
