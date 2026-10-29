"""Pip-Beast: standalone decoder-only transformer (~1.11B params). Pure PyTorch.

RMSNorm, RoPE, grouped-query attention via scaled_dot_product_attention (FlashAttention
kernels are used automatically on supported GPUs), SwiGLU FFN, tied input/output embeddings.
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


def apply_rope(x, cos, sin):
    T = x.shape[2]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos[:T].to(x.dtype)[None, None]
    s = sin[:T].to(x.dtype)[None, None]
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

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        rep = self.n_heads // self.n_kv
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))


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

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
    cfg = BeastConfig()
    n = PipBeast(cfg).num_params()
    print(f'Pip-Beast parameters: {n:,} ({n / 1e9:.3f}B)')
