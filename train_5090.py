"""Pip-Beast training loop tuned for a single 32GB RTX 5090 (Blackwell, sm_120).

Presets (model shape plus training defaults):
    --preset fast    ~222M params: 16 layers, dim 1024, 8 heads / 4 KV, hidden 2816
    --preset beast   ~1.04B params: 22 layers, dim 2048, 16 heads / 4 KV, hidden 5504 (default)

Auto batch: unless --batch is given, the largest power-of-two micro-batch that fits is probed
(budget is 92% of VRAM, minus optimizer state when no optimizer state exists yet). Gradient
accumulation is then computed so each optimizer step covers about --target_tokens tokens.

Features: bf16 autocast, optional torch.compile, fused AdamW, cosine LR with warmup (default
warmup max(100, 1% of max_steps)), gradient accumulation, gradient checkpointing toggle,
memmap token loader (uint16 .bin), MFU and VRAM telemetry, optional wandb, torch checkpoints
with top-k retention and auto-resume.

Data: expects data/train.bin and data/val.bin (uint16 token ids, e.g. from prepare_data.py).

Examples:
    python train_5090.py --preset fast
    python train_5090.py --preset beast --target_tokens 1048576
    python train_5090.py --batch 8 --accum 8 --no_auto_batch --grad_ckpt --seq_len 4096
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from model import PRESETS, BeastConfig, PipBeast

# RTX 5090 dense BF16 tensor throughput (no sparsity), TFLOPS. Used for MFU.
PEAK_BF16_TFLOPS_5090 = 209.5

# Per-preset training defaults. Explicit CLI flags always win.
TRAIN_PRESETS = {
    'fast': dict(lr=6e-4, target_tokens=524288),
    'beast': dict(lr=3e-4, target_tokens=524288),
}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--preset', choices=sorted(PRESETS), default='beast')
    p.add_argument('--data_dir', default='data')
    p.add_argument('--out_dir', default='checkpoints')
    p.add_argument('--vocab_size', type=int, default=32000)
    p.add_argument('--seq_len', type=int, default=2048)
    p.add_argument('--batch', type=int, default=None, help='micro-batch size (default: auto-probe)')
    p.add_argument('--no_auto_batch', dest='auto_batch', action='store_false',
                   help='skip the VRAM probe and use --batch (default 8)')
    p.add_argument('--accum', type=int, default=None,
                   help='gradient accumulation steps (default: computed from --target_tokens)')
    p.add_argument('--target_tokens', type=int, default=None,
                   help='tokens per optimizer step (default per preset, 524288)')
    p.add_argument('--max_steps', type=int, default=100000)
    p.add_argument('--lr', type=float, default=None, help='peak LR (default per preset)')
    p.add_argument('--min_lr_ratio', type=float, default=0.1)
    p.add_argument('--warmup_steps', type=int, default=None,
                   help='default max(100, 1%% of max_steps)')
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


def probe_micro_batch(model, seq_len, device, n_params, optimizer_built, max_batch=64):
    """Largest power-of-two micro-batch whose forward+backward peak fits the VRAM budget.

    Budget is 92% of total VRAM. If no optimizer state exists yet, reserve 8 bytes per parameter
    for AdamW's first and second moments, which are allocated on the first step.
    """
    total = torch.cuda.get_device_properties(0).total_memory
    reserve = 0 if optimizer_built else 8 * n_params
    budget = int(total * 0.92) - reserve
    best, b = 0, 1
    torch.cuda.reset_peak_memory_stats()
    while b <= max_batch:
        try:
            x = torch.randint(0, model.cfg.vocab_size, (b, seq_len), device=device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, loss = model(x, x)
            loss.backward()
            peak = torch.cuda.max_memory_allocated()
            model.zero_grad(set_to_none=True)
            del x, loss
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            break
        if peak > budget:
            break
        best = b
        b *= 2
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    if best == 0:
        raise RuntimeError(f'batch 1 at seq_len {seq_len} does not fit; try --grad_ckpt or a smaller --seq_len')
    return best


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

    # Resolve preset defaults for anything the user did not set explicitly.
    tp = TRAIN_PRESETS[a.preset]
    a.lr = a.lr if a.lr is not None else tp['lr']
    a.warmup_steps = a.warmup_steps if a.warmup_steps is not None else max(100, a.max_steps // 100)
    target_tokens = a.target_tokens if a.target_tokens is not None else tp['target_tokens']

    train_data = load_bin(Path(a.data_dir) / 'train.bin')
    val_path = Path(a.data_dir) / 'val.bin'
    val_data = load_bin(val_path) if val_path.exists() else train_data

    cfg = BeastConfig(vocab_size=a.vocab_size, max_seq_len=a.seq_len,
                      grad_checkpointing=a.grad_ckpt, **PRESETS[a.preset])
    model = PipBeast(cfg).to(device)
    n_params = model.num_params()
    print(f'Preset {a.preset} | Model: {n_params / 1e9:.3f}B parameters '
          f'({cfg.n_layers}L x {cfg.dim}d, {cfg.n_heads}Q/{cfg.n_kv_heads}KV heads)')

    optimizer = build_optimizer(model, a, device)
    step, best_val = 0, float('inf')
    resume = latest_checkpoint(a.out_dir)
    if resume is not None:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        saved = ckpt['config']
        if saved['dim'] != cfg.dim or saved['n_layers'] != cfg.n_layers:
            sys.exit(f'checkpoint in {a.out_dir} is {saved["dim"]}d x {saved["n_layers"]}L but '
                     f'preset {a.preset} is {cfg.dim}d x {cfg.n_layers}L. Use the matching '
                     f'--preset or a different --out_dir.')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        step, best_val = ckpt['step'], ckpt.get('best_val', float('inf'))
        torch.set_rng_state(ckpt['torch_rng'])
        if ckpt.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
        print(f'Resumed from {resume.name} at step {step}')

    # Micro-batch and gradient accumulation.
    if a.batch is None:
        if a.auto_batch:
            a.batch = probe_micro_batch(model, a.seq_len, device, n_params,
                                        optimizer_built=resume is not None)
        else:
            a.batch = 8
    if a.accum is None:
        a.accum = max(1, round(target_tokens / (a.batch * a.seq_len)))
    tokens_per_step = a.batch * a.accum * a.seq_len
    print(f'Batch plan: micro-batch {a.batch} x accum {a.accum} x seq {a.seq_len} = '
          f'{tokens_per_step:,} tokens per optimizer step | lr {a.lr:.2e} | '
          f'warmup {a.warmup_steps} steps')

    run = None
    if a.wandb:
        import wandb
        run = wandb.init(project='pip-beast', config=vars(a))

    fwd_model = torch.compile(model) if a.compile else model
    flops_per_token = 6 * n_params  # forward + backward approximation
    peak_flops = PEAK_BF16_TFLOPS_5090 * 1e12
    torch.cuda.reset_peak_memory_stats()

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
