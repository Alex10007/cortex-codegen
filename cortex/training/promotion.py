"""Checkpoint promotion gate.

A checkpoint may only be promoted (deployed pointer / status update) if it
demonstrates a measurable improvement on the independent execution-graded
evaluations, with no regression on validation loss, and with benchmark
integrity intact. Nothing else may move the promotion state.
"""
from __future__ import annotations

from typing import Any, Optional

PASS_AT_1_MIN_IMPROVEMENT = 0.1


def promotion_gate(baseline: dict[str, Any], candidate: dict[str, Any],
                   fingerprint_ok: bool = True,
                   min_pass_at_1: float = PASS_AT_1_MIN_IMPROVEMENT) -> dict[str, Any]:
    """Decide promotion from measured metrics only."""
    base_pass = baseline.get("pass_at_1", 0.0)
    cand_pass = candidate.get("pass_at_1", 0.0)
    reasons: list[str] = []
    promoted = True

    if not fingerprint_ok:
        promoted = False
        reasons.append("benchmark fingerprint mismatch — results not comparable")
    base_val = baseline.get("val_loss")
    cand_val = candidate.get("val_loss")
    if cand_val is None or base_val is None or cand_val > base_val:
        promoted = False
        reasons.append(
            f"validation loss not improved (base={base_val}, cand={cand_val})")
    if cand_pass < base_pass + min_pass_at_1:
        promoted = False
        reasons.append(
            f"pass@1 did not improve by >= {min_pass_at_1} "
            f"(base={base_pass}, cand={cand_pass})")
    return {
        "promoted": promoted,
        "reasons": reasons or ["measurable improvement on independent evals"],
        "baseline": baseline,
        "candidate": candidate,
    }


def apply_promotion(gate: dict[str, Any], checkpoint_path: str,
                    status_ledger: dict[str, Any], config_name: str) -> bool:
    if not gate["promoted"]:
        return False
    status_ledger.setdefault(config_name, {})["promoted_checkpoint"] = checkpoint_path
    return True
