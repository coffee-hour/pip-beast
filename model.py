"""Pip-Beast: standalone decoder-only transformer. Pure PyTorch.

RMSNorm, RoPE, grouped-query attention via scaled_dot_product_attention (FlashAttention
kernels are used automatically on supported GPUs), SwiGLU FFN, tied input/output embeddings.

Presets (see PRESETS):
    fast   ~222M params: 16 layers, dim 1024, 8 query heads, 4 KV heads, SwiGLU hidden 2816
    beast  ~1.04B params: 22 layers, dim 2048, 16 query heads, 4 KV heads, SwiGLU hidden 5504

KV cache: forward_cached() decodes one token at a time against cached keys/values. The
uncached forward() used by training is unchanged.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class BeastConfig:
    vocab_size: int = 32000
    dim: int = 2048
    n_layers: int = 22
    n_heads: int = 16
    n_kv_heads: int = 4
    hidden_dim: int = 5504
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    grad_checkpointing: bool = False

    def __post_init__(self):
        assert self.dim % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


PRESETS = {
    'fast': dict(dim=1024, n_layers=16, n_heads=8, n_kv_heads=4, hidden_dim=2816),
    'beast': dict(dim=2048, n_layers=22, n_heads=16, n_kv_heads=4, hidden_dim=5504),
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


def rope_tables(head_dim: int, seq_len: int, theta: float):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len).float(), freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin, pos: int = 0):
    T = x.shape[2]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos[pos:pos + T].to(x.dtype)[None, None]
    s = sin[pos:pos + T].to(x.dtype)[None, None]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: BeastConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.hd = cfg.head_dim
        self.dropout = cfg.dropout
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.hd, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.hd, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.hd, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.hd, cfg.dim, bias=False)

    def _qkv(self, x, cos, sin, pos: int):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        return apply_rope(q, cos, sin, pos), apply_rope(k, cos, sin, pos), v

    def _expand(self, t):
        rep = self.n_heads // self.n_kv
        return t.repeat_interleave(rep, dim=1) if rep > 1 else t

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q, k, v = self._qkv(x, cos, sin, 0)
        y = F.scaled_dot_product_attention(
            q, self._expand(k), self._expand(v), is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))

    def forward_cached(self, x, cos, sin, cache, pos: int):
        """cache is None or (k, v) with shape (B, n_kv, S_past, hd). Returns (out, new_cache)."""
        B, T, _ = x.shape
        q, k, v = self._qkv(x, cos, sin, pos)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        S = k.shape[2]
        mask = None
        if T > 1:
            # query i (absolute position S - T + i) may attend to keys 0..S - T + i
            mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S - T)
        y = F.scaled_dot_product_attention(q, self._expand(k), self._expand(v), attn_mask=mask)
        return self.wo(y.transpose(1, 2).reshape(B, T, -1)), (k, v)


class SwiGLU(nn.Module):
    def __init__(self, cfg: BeastConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)  # gate
        self.w3 = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)  # up
        self.w2 = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)  # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: BeastConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin):
        x = x + self.drop(self.attn(self.attn_norm(x), cos, sin))
        x = x + self.drop(self.ffn(self.ffn_norm(x)))
        return x

    def forward_cached(self, x, cos, sin, cache, pos: int):
        a, cache = self.attn.forward_cached(self.attn_norm(x), cos, sin, cache, pos)
        x = x + a
        x = x + self.ffn(self.ffn_norm(x))
        return x, cache


class PipBeast(nn.Module):
    def __init__(self, cfg: BeastConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)
        self._init_weights()

    def _init_weights(self):
        std = 0.02
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, std=std)
        scaled = std / math.sqrt(2 * self.cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith('attn.wo.weight') or name.endswith('ffn.w2.weight'):
                nn.init.normal_(p, std=scaled)

    def forward(self, idx, targets=None):
        assert idx.size(1) <= self.cfg.max_seq_len, 'sequence longer than max_seq_len'
        x = self.tok_emb(idx)
        for layer in self.layers:
            if self.cfg.grad_checkpointing and self.training:
                x = checkpoint(layer, x, self.rope_cos, self.rope_sin, use_reentrant=False)
            else:
                x = layer(x, self.rope_cos, self.rope_sin)
        x = self.norm(x)
        if targets is None:
            return self.lm_head(x[:, -1:, :]), None
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                               targets.reshape(-1), ignore_index=-1)
        return logits, loss

    def forward_cached(self, idx, caches=None, pos: int = 0):
        """Incremental forward with a KV cache. idx: (B, T) new tokens starting at absolute
        position pos. Returns (logits for the last new token as (B, 1, V), list of per-layer caches).
        Pass caches=None and pos=0 to prefill. Caller must keep pos + T <= max_seq_len."""
        assert pos + idx.size(1) <= self.cfg.max_seq_len, 'cache overflow: re-prime the cache'
        if caches is None:
            caches = [None] * len(self.layers)
        x = self.tok_emb(idx)
        new_caches = []
        for layer, c in zip(self.layers, caches):
            x, c = layer.forward_cached(x, self.rope_cos, self.rope_sin, c, pos)
            new_caches.append(c)
        x = self.norm(x[:, -1:, :])
        return self.lm_head(x), new_caches

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
    for name, arch in PRESETS.items():
        n = PipBeast(BeastConfig(**arch)).num_params()
        print(f'{name}: {n:,} parameters ({n / 1e9:.3f}B), tied embeddings')
