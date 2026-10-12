"""INFINITY NEURAL CORE — original decoder-only transformer.

Architecture (all layers implemented here; PyTorch supplies tensors/autograd):

    tokens -> embedding
           -> N x [ x + Attn(RMSNorm(x)) ; x + SwiGLU(RMSNorm(x)) ]
           -> RMSNorm -> linear head (tied to the embedding) -> logits

- Positional information: rotary position embeddings (RoPE) applied to q/k.
- Attention: multi-head causal self-attention. The explicit math path is the
  reference implementation; ``use_fused_attention`` switches to PyTorch's
  fused kernel for the same computation (tests check they agree).
- Generation uses a per-layer key/value cache.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def rope_frequencies(head_dim: int, max_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D); cos/sin: (T, D/2). Rotate interleaved pairs.
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


@dataclass
class LayerCache:
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.kv_heads
        self.head_dim = cfg.head_dim
        # Fused projection. With n_kv == n_heads this is exactly the original 3*d_model layout.
        self.qkv = nn.Linear(cfg.d_model, (cfg.n_heads + 2 * self.n_kv) * cfg.head_dim, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        self.use_fused_attention = True

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: LayerCache | None = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        hd = self.head_dim
        q, k, v = self.qkv(x).split([self.n_heads * hd, self.n_kv * hd, self.n_kv * hd], dim=2)
        q = q.view(B, T, self.n_heads, hd).transpose(1, 2)
        k = k.view(B, T, self.n_kv, hd).transpose(1, 2)
        v = v.view(B, T, self.n_kv, hd).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        past = 0
        if cache is not None:
            if cache.k is not None:
                past = cache.k.size(2)
                k = torch.cat([cache.k, k], dim=2)
                v = torch.cat([cache.v, v], dim=2)
            cache.k, cache.v = k, v

        if self.n_kv != self.n_heads:  # grouped-query attention: share each K/V head across a group of query heads
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        S = k.size(2)
        drop = self.dropout if self.training else 0.0
        if self.use_fused_attention:
            if past == 0:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
            else:
                mask = self._causal_mask(T, S, past, x.device)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = self._causal_mask(T, S, past, x.device)
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att.float(), dim=-1).to(q.dtype)
            if drop:
                att = F.dropout(att, p=drop)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

    @staticmethod
    def _causal_mask(T: int, S: int, past: int, device) -> torch.Tensor:
        # Query i (absolute position past+i) may attend to keys 0..past+i.
        q_pos = torch.arange(past, past + T, device=device)[:, None]
        k_pos = torch.arange(S, device=device)[None, :]
        return k_pos <= q_pos


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.grad_checkpoint = cfg.grad_checkpoint

    def forward(self, x, cos, sin, cache=None):
        if self.grad_checkpoint and self.training and cache is None and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            return checkpoint(self._forward, x, cos, sin, None, use_reentrant=False)
        return self._forward(x, cos, sin, cache)

    def _forward(self, x, cos, sin, cache):
        x = x + self.resid_dropout(self.attn(self.norm1(x), cos, sin, cache))
        x = x + self.ffn(self.norm2(x))
        return x


class InfinityNeuralCore(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight
        cos, sin = rope_frequencies(cfg.head_dim, cfg.context_length, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        # Scale residual output projections by depth (GPT-2 style).
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def set_fused_attention(self, enabled: bool) -> None:
        for b in self.blocks:
            b.attn.use_fused_attention = enabled

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.embed.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        caches: list[LayerCache] | None = None,
        ignore_index: int = -100,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        past = 0
        if caches is not None and caches[0].k is not None:
            past = caches[0].k.size(2)
        if past + T > self.cfg.context_length:
            raise ValueError(f"sequence length {past + T} exceeds context_length {self.cfg.context_length}")
        cos = self.rope_cos[past : past + T]
        sin = self.rope_sin[past : past + T]
        x = self.embed(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, caches[i] if caches is not None else None)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignore_index
            )
        return logits, loss

    def new_caches(self) -> list[LayerCache]:
        return [LayerCache() for _ in self.blocks]
