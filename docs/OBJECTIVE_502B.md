# Cortex Codegen — Unified 502B Training Objective (Stage 7)

Owner: Fibonera. Versioned extension of the Stage-5/6 corrected objective;
the dense objective (`cortex/training/sft.py` completion-masked next-token CE)
is unchanged and the dense model remains the experimental control.

## 1. Parameter systems

| System | Parameters | Storage | Optimizer |
|---|---|---|---|
| Physical (dense/MoE backbone) | 2,056,012,800 target; 8.07M dense demo / 7,491,456 MoE demo actually trained here | torch checkpoints (FSDP-sharded at 2B scale) | AdamW (decoupled wd 0.01), grad clip 1.0 |
| Virtual experts | exactly 500,000,000,000 allocated float32 scalars (5,000 banks x [2,000 x 50,000]) | one sparse file per bank, paged on demand, sha256-digested rows, atomic manifest | `VirtualAdamW` (standard AdamW, per-slot bias correction) |

A slot is a LoRA pair `A ∈ R^{dim×rank}`, `B ∈ R^{rank×dim}` carved from one
bank row (2·dim·rank of 50,000 values; demo rank 16–64, 2B-capable rank ≤ 9).

## 2. Unified objective

For a batch of verified-pair records with completion-only masking
(labels `-100` on instruction and padding, causal next-token shift, EOT
appended, pad keys masked −inf pre-softmax):

```
L(theta, phi) = L_CE(theta, phi) + lambda_aux * sum_l L_aux^(l)(phi_route) + lambda_z * sum_l L_z^(l)(phi_route)
```

where `theta` = physical parameters, `phi` = routed-expert parameters
(physical experts + virtual slots reachable through the router), and
`l` ranges over MoE layers.

1. `L_CE` — completion-token mean cross-entropy over next-token predictions
   (purpose: causal language modelling / code generation; the ONLY term with
   access to token targets).
2. `L_aux = E_l · Σ_i f_i · P_i` (Switch load-balancing):
   `f_i` = fraction of tokens routed to expert i (non-differentiable count),
   `P_i` = mean router softmax prob for expert i (differentiable).
   Purpose: prevent routing collapse so all experts participate.
   `lambda_aux = 0.01`.
3. `L_z = mean_x (logsumexp(router_logits(x))^2)`, `lambda_z = 0.001`.
   Purpose: keep router logits from growing unboundedly, stabilizing routing.

Weights are not arbitrary: L_CE is the task objective; aux and z are
router-health regularizers with standard Switch-Transformer scalings
(`E_l` multiplier matches Fedus et al.'s formulation), one order of magnitude
below the CE term so they shape routing without dominating learning.

## 3. Gradient pathways

- Physical: full autograd. `L_CE` and both regularization terms backprop
  through attention, experts, and routers.
- Virtual: hard top-k assignment is non-differentiable; the alternative
  optimization signal is the standard switch gate — a selected expert's
  virtual slot output is scaled by its router softmax weight, so
  `∂L/∂A`, `∂L/∂B` are real autograd gradients. `VirtualAdamW` then applies
  per-slot updates (m, v, bias-corrected, decoupled wd). Slots that receive
  no gradient take no step and are not counted as trained.
- Verified outcomes: execution-verified pairs are the data gate (harvester +
  double execution); they select what is trained, they are not a separate
  loss term.

## 4. Update and commit protocol

1. Step: CE+aux+z backward → physical clip + AdamW step → per-slot
   VirtualAdamW step → participation log (which slots stepped, grad norms).
2. Checkpoint at step S: physical `torch.save` + sha256; store commit of all
   dirty slots (row write → flush → sha256 → npz optimizer state → atomic
   manifest update). Physical step S and virtual step S must match before a
   lineage record is written; mismatch or digest failure aborts the commit.

## 5. Evaluation

- Immutable benchmark suites (fingerprint-protected) run free-running
  autoregressive generation; pass@1 by sandboxed pytest execution.
- Divergent reasoning: N hypotheses (temperature sweep) per task, per-hypothesis
  execution grading, contradiction/convergence metrics, verified-only
  selection.
- Promotion: only with reproducible improvement on these suites; allocated
  capacity is never reported as trained capacity.
