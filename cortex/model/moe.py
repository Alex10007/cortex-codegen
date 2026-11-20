"""Mixture-of-Experts extension of the Cortex transformer (stage 7).

The dense `CortexTransformer` is untouched and remains the experimental
control; this subclass swaps each block's dense MLP for an MoE feed-forward
that combines:

- a small number of physical expert FFNs (trained with the physical AdamW), and
- virtual expert routes executed by externally resident trainable slots
  (LoRA pairs A[dim, rank], B[rank, dim] carved from 500B-parameter bank
  rows, see cortex/virtual/store.py), each producing a real autograd
  contribution through the same joint completion-masked CE loss.

Routing: combined physical+virtual router logits, softmax, top-k selection,
renormalized gate over selected experts (Switch Transformer style). Hard
top-k assignment is non-differentiable; gradient flows through the softmax
gate weights and the selected experts' outputs, which is the standard,
mathematically valid treatment. Router load-balancing (aux) and z-loss
terms are defined in docs/OBJECTIVE_502B.md.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cortex.interfaces import ModelConfig
from cortex.model.transformer import CortexTransformer

PHYS_EXPERTS = 4
VIRTUAL_ROUTES = 8
TOP_K = 2
AUX_COEF = 0.01
Z_COEF = 0.001


class MoEFFN(nn.Module):
    """Physical experts + virtual routes behind one shared top-k router."""

    def __init__(self, dim: int, n_physical: int = PHYS_EXPERTS,
                 n_virtual: int = VIRTUAL_ROUTES, top_k: int = TOP_K,
                 expert_ffn: int | None = None):
        super().__init__()
        self.n_physical = n_physical
        self.n_virtual = n_virtual
        self.top_k = top_k
        self.n_experts = n_physical + n_virtual
        if self.top_k > self.n_experts:
            raise ValueError("top_k exceeds expert count")
        ff = expert_ffn or max(dim // 2, 32)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, ff), nn.GELU(), nn.Linear(ff, dim))
            for _ in range(n_physical))
        self.router = nn.Linear(dim, self.n_experts, bias=False)
        self.virtual_slots: dict[int, object] = {}
        self.last_counts: torch.Tensor | None = None
        self.last_aux: torch.Tensor | None = None
        self.last_z: torch.Tensor | None = None

    def attach_virtual_route(self, route_index: int, slot) -> None:
        """Bind a virtual route to an externally resident SlotState
        (real nn.Parameter leaves from the virtual-parameter store)."""
        if not 0 <= route_index < self.n_virtual:
            raise IndexError("route index out of range")
        self.virtual_slots[int(route_index)] = slot

    def clear_virtual_routes(self) -> None:
        self.virtual_slots.clear()

    def _virtual_apply(self, route: int, x: torch.Tensor) -> torch.Tensor:
        slot = self.virtual_slots[route]
        one = torch.ones((), dtype=x.dtype, device=x.device)
        return slot.forward(x, one)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = x.reshape(-1, D)
        logits = self.router(h)
        # Virtual routes are optional until their external SlotState objects
        # have been attached. Never allow an unbound route to win top-k and
        # fail later with a KeyError in _virtual_apply.
        active = torch.ones(self.n_experts, dtype=torch.bool, device=h.device)
        for route in range(self.n_virtual):
            if route not in self.virtual_slots:
                active[self.n_physical + route] = False
        n_active = int(active.sum())
        if n_active < self.top_k:
            raise RuntimeError(
                f"MoE has {n_active} active experts but top_k={self.top_k}; "
                "attach the required virtual slots or reduce top_k")
        logits = logits.masked_fill(~active.unsqueeze(0),
                                    torch.finfo(logits.dtype).min)
        probs = F.softmax(logits, dim=-1)
        top_vals, top_idx = torch.topk(probs, self.top_k, dim=-1)
        gate = top_vals / top_vals.sum(-1, keepdim=True).clamp_min(1e-9)

        y = torch.zeros_like(h)
        counts = torch.zeros(self.n_experts, dtype=torch.long, device=x.device)
        for e in range(self.n_experts):
            sel = top_idx == e                                  # [N, k]
            w = (gate * sel).sum(-1)                            # [N]
            rows = w > 0
            if not bool(rows.any()):
                continue
            counts[e] = int(rows.sum())
            xe = h[rows]
            if e < self.n_physical:
                out = self.experts[e](xe)
            else:
                out = self._virtual_apply(e - self.n_physical, xe)
            y.index_add_(0, rows.nonzero(as_tuple=True)[0], w[rows, None] * out)

        n = h.size(0)
        # With top-k dispatch, each token contributes k assignments. Normalize
        # by N*k so expert fractions sum to one, matching the load-balancing
        # objective's probability mass convention.
        f = counts.to(probs.dtype) / max(n * self.top_k, 1)
        p_mean = probs.mean(0)
        self.last_counts = counts.detach()
        self.last_probs = p_mean.detach()
        self.last_active_mask = active.detach()
        self.last_aux = n_active * torch.sum(f * p_mean)
        self.last_z = torch.mean(
            torch.logsumexp(logits, dim=-1) ** 2).clamp_min(0.0)
        return y.view(B, T, D)

    def routing_diagnostics(self, collapse_threshold: float = 0.9) -> dict:
        assert self.last_counts is not None
        total = int(self.last_counts.sum())
        counts = [int(c) for c in self.last_counts.tolist()]
        max_frac = (max(counts) / total) if total else 0.0
        return {
            "counts": counts,
            "tokens_routed": total,
            "utilization_fraction": (
                sum(1 for i, c in enumerate(counts)
                    if c > 0 and bool(self.last_active_mask[i]))
                / max(1, int(self.last_active_mask.sum()))),
            "active_experts": int(self.last_active_mask.sum()),
            "max_expert_fraction": round(max_frac, 4),
            "collapsed": max_frac > collapse_threshold,
            "physical": counts[:self.n_physical],
            "virtual": counts[self.n_physical:],
        }


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> float:
    """Jensen-Shannon divergence (bits, base 2): 0 identical, 1 disjoint."""
    p = p / p.sum().clamp_min(eps)
    q = q / q.sum().clamp_min(eps)
    m = 0.5 * (p + q)

    def kl(a: torch.Tensor) -> float:
        mask = a > 0
        if not bool(mask.any()):
            return 0.0
        return float((a[mask] * ((a[mask] / m[mask].clamp_min(eps)).log2())).sum())

    return 0.5 * kl(p) + 0.5 * kl(q)


class CortexMoETransformer(CortexTransformer):
    """CortexTransformer whose MLP sublayers are MoE feed-forwards.

    Virtual routes are inactive (physical-expert only) until the training
    coordinator attaches externally resident slots via
    `attach_virtual_slots`. The dense base class remains the control.
    """

    def __init__(self, cfg: ModelConfig, n_physical: int = PHYS_EXPERTS,
                 n_virtual: int = VIRTUAL_ROUTES, top_k: int = TOP_K,
                 aux_coef: float = AUX_COEF, z_coef: float = Z_COEF,
                 expert_ffn: int | None = None, expert_every: int = 1):
        super().__init__(cfg)
        expert_ffn = expert_ffn or cfg.d_ff
        self.aux_coef = aux_coef
        self.z_coef = z_coef
        self.n_virtual = n_virtual
        self.moe_layers: list[MoEFFN] = []
        for i, block in enumerate(self.blocks):
            if i % expert_every:
                continue
            moe = MoEFFN(cfg.dim, n_physical=n_physical,
                         n_virtual=n_virtual, top_k=top_k,
                         expert_ffn=expert_ffn)
            block.mlp = moe
            self.moe_layers.append(moe)
        if not self.moe_layers:
            raise ValueError("expert_every left no MoE layers")

    # -- virtual route wiring ------------------------------------------------

    def attach_virtual_slots(self, slots: dict[int, object]) -> None:
        """slots: {route_index: SlotState} applied to every MoE layer; the
        same slot (bank row) is bound at each MoE layer for a given route."""
        missing = [r for r in range(self.n_virtual) if r not in slots]
        if missing:
            raise KeyError(f"missing virtual routes: {missing}")
        for layer in self.moe_layers:
            for r, slot in slots.items():
                layer.attach_virtual_route(r, slot)

    def clear_virtual_slots(self) -> None:
        for layer in self.moe_layers:
            layer.clear_virtual_routes()

    def virtual_param_count(self) -> int:
        """Count unique attached virtual scalars, including both LoRA factors."""
        slots: dict[tuple[int, int], object] = {}
        for layer in self.moe_layers:
            for slot in layer.virtual_slots.values():
                key = (int(slot.bank), int(slot.row))
                prior = slots.get(key)
                if prior is not None and prior is not slot:
                    raise RuntimeError(
                        f"multiple SlotState objects own virtual slot {key}")
                slots[key] = slot
        return sum(slot.A.numel() + slot.B.numel()
                   for slot in slots.values())

    def moe_param_count(self) -> dict[str, int]:
        """Report dense reference, physical MoE components and virtual slots
        as separate ledgers; never label the MoE model as the dense 2B model."""
        physical_experts = sum(
            p.numel() for layer in self.moe_layers
            for p in layer.experts.parameters())
        routers = sum(layer.router.weight.numel() for layer in self.moe_layers)
        physical_total = self.actual_param_count()
        backbone = physical_total - physical_experts - routers
        virtual = self.virtual_param_count()
        return {
            "dense_reference_parameters": self.cfg.theoretical_param_count(),
            "backbone_parameters": backbone,
            "physical_expert_parameters": physical_experts,
            "router_parameters": routers,
            "physical_total_parameters": physical_total,
            "virtual_slot_parameters": virtual,
            "total_with_virtual": physical_total + virtual,
            # Backward-compatible field names retained for existing reports.
            "total_actual": physical_total,
            "physical_experts_routers": physical_experts + routers,
            "virtual_parameters": virtual,
            "backbone_and_dense_mlp": backbone,
        }

    # -- loss ----------------------------------------------------------------

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None,
                pad_mask: torch.Tensor | None = None):
        logits, loss = super().forward(tokens, targets, pad_mask)
        if targets is not None:
            aux_t = sum(l.last_aux for l in self.moe_layers)           # type: ignore[operator]
            z_t = sum(l.last_z for l in self.moe_layers)               # type: ignore[operator]
            loss = loss + self.aux_coef * aux_t + self.z_coef * z_t
            self.last_aux_total = float(aux_t.detach())
            self.last_z_total = float(z_t.detach())
        return logits, loss
