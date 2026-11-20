"""Cortex transformer backbone: decoder-only, pre-LN, RoPE, tied embeddings.

Parameter accounting is exact: `theoretical_param_count` in ModelConfig is
asserted (== by test) against a live instantiated model.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cortex.interfaces import ModelConfig


def rope_cache(seq_len: int, head_dim: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)                     # [T, D/2]
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, Dh]; cos/sin: [T, Dh/2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = out1
    out[..., 1::2] = out2
    return out


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                key_pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        att = att.masked_fill(causal, float("-inf"))
        if key_pad_mask is not None:
            att = att.masked_fill(key_pad_mask[:, None, None, :], float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, d_ff: int):
        super().__init__()
        self.fc = nn.Linear(dim, d_ff)
        self.proj = nn.Linear(d_ff, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.attn = CausalSelfAttention(cfg.dim, cfg.n_heads)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.mlp = MLP(cfg.dim, cfg.d_ff)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                key_pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin, key_pad_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class CortexTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.dim)
        if cfg.tie_word_embeddings:
            self.head: nn.Module = None  # type: ignore[assignment]
        else:
            self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None,
                pad_mask: torch.Tensor | None = None):
        """tokens/pad_mask: [B, T]; pad_mask True = mask out as attention key."""
        B, T = tokens.shape
        device = tokens.device
        head_dim = self.cfg.dim // self.cfg.n_heads
        cos, sin = rope_cache(T, head_dim, device, torch.float32)
        x = self.tok_emb(tokens)
        for block in self.blocks:
            x = block(x, cos, sin, pad_mask)
        x = self.ln_f(x)
        logits = x @ self.tok_emb.weight.T if self.cfg.tie_word_embeddings else self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.inference_mode()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = 50,
                 eot_id: int | None = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            ctx = tokens[:, -self.cfg.max_seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, nxt], dim=1)
            if eot_id is not None and int(nxt) == eot_id:
                break
        return tokens

    def actual_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
