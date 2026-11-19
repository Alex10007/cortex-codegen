"""Virtual slot expert + sparse virtual optimizer (stage 7).

A *slot* is one LoRA-shaped trainable expert (A [dim, rank], B [rank, dim])
carved from a row of a virtual parameter bank. Slot parameters are real
leaf tensors: they receive genuine gradients from the joint loss and are
updated by a genuine AdamW implementation (mathematically the standard
algorithm, verified against an independent recomputation in tests).
Only slots that received a gradient and took an optimizer step count as
trained; paging a slot into memory does not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class SlotState:
    bank: int
    row: int
    A: torch.nn.Parameter
    B: torch.nn.Parameter
    step_count: int = 0
    m: dict[str, torch.Tensor] = field(default_factory=dict)
    v: dict[str, torch.Tensor] = field(default_factory=dict)
    grad_norm_last: float | None = None

    @property
    def params(self) -> dict[str, torch.nn.Parameter]:
        return {"A": self.A, "B": self.B}

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Differentiable expert application: y = ((x @ A) @ B) * gate.
        x: [B, T, dim]; returns residual contribution [B, T, dim]."""
        return ((x @ self.A) @ self.B) * gate


def slot_adam_states(st: SlotState) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if st.step_count == 0:
        return out
    for name, p in st.params.items():
        out[f"{name}_m"] = st.m[name].detach().numpy()
        out[f"{name}_v"] = st.v[name].detach().numpy()
    out["step_count"] = st.step_count
    return out


class VirtualAdamW:
    """Standard AdamW over active virtual slots (per-slot bias correction,
    decoupled weight decay). State lives with the slot so it can be
    checkpointed and recovered."""

    def __init__(self, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.98),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        self.lr, self.betas, self.eps, self.wd = lr, betas, eps, weight_decay
        self.updates: dict[tuple[int, int], int] = {}

    @torch.no_grad()
    def step(self, st: SlotState) -> dict[str, float]:
        if st.A.grad is None and st.B.grad is None:
            return {}
        if not st.m:
            for name, p in st.params.items():
                st.m[name] = torch.zeros_like(p)
                st.v[name] = torch.zeros_like(p)
        st.step_count += 1
        t = st.step_count
        b1, b2 = self.betas
        total_norm = 0.0
        for name, p in st.params.items():
            g = p.grad
            if g is None:
                continue
            st.m[name].mul_(b1).add_(g, alpha=1 - b1)
            st.v[name].mul_(b2).addcmul_(g, g, value=1 - b2)
            mhat = st.m[name] / (1 - b1 ** t)
            vhat = st.v[name] / (1 - b2 ** t)
            p.addcdiv_(mhat, vhat.sqrt().add_(self.eps), value=-self.lr)
            p.mul_(1 - self.lr * self.wd)
            total_norm += float(g.float().pow(2).sum())
        self.updates[(st.bank, st.row)] = t
        st.grad_norm_last = math.sqrt(total_norm)
        return {"bank": st.bank, "row": st.row, "step": t,
                "grad_norm": st.grad_norm_last}
