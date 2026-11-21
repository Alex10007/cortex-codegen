"""Stage-10 capability maturity classification (workstream I).

A capability's maturity level is derived from evidence checks, never from
intent. Level requirements are cumulative - each level's evidence gate
includes every lower level's gate.

Levels (per the Stage 10 directive):
    0 conceptual / specified
    1 implemented
    2 executes correctly
    3 learns from actual training
    4 improves on immutable held-out evaluation
    5 improvement independently reproduces
    6 outperforms a relevant baseline
    7 transfers reliably to unfamiliar tasks
"""
from __future__ import annotations

from typing import Any

LEVEL_NAMES = {
    0: "conceptual-or-specified",
    1: "implemented",
    2: "executes-correctly",
    3: "learns-from-training",
    4: "improves-on-heldout",
    5: "independently-reproduces",
    6: "outperforms-baseline",
    7: "transfers-to-unfamiliar",
}


def classify_capability(evidence: dict[str, Any]) -> int:
    """Highest level whose full evidence chain holds.

    Expected keys (each gate is a boolean; the chain is cumulative):
        implemented, executes, learns, improves_heldout,
        independent_reproduction, outperforms_baseline, transfers

    'blocked' does not erase already-earned gates: the level reflects the
    highest satisfied evidence chain, and build_report marks the capability
    as blocked with the explicit reason so no progress is silently claimed.
    """
    level = 0
    gates = [
        bool(evidence.get("implemented")),
        bool(evidence.get("executes")),
        bool(evidence.get("learns")),
        bool(evidence.get("improves_heldout")),
        bool(evidence.get("independent_reproduction")),
        bool(evidence.get("outperforms_baseline")),
        bool(evidence.get("transfers")),
    ]
    for gate in gates:
        if not gate:
            break
        level += 1
    return level


def cap_at_zero(level: int, reason: str) -> dict[str, Any]:
    """Notation helper: an explicitly downgraded capability with its reason,
    preventing silent level inflation."""
    if level and not reason:
        raise ValueError("a nonzero capability cannot be capped without a reason")
    return {"capped_level": 0 if reason else level, "reason": reason}


def build_report(capabilities: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """capabilities: name -> evidence dict. Returns the full maturity
    matrix with level names and gap-to-next-level requirements."""
    rows = []
    for name, ev in sorted(capabilities.items()):
        level = classify_capability(ev)
        if ev.get("blocked"):
            gate_status = "blocked: " + ev["blocked"]
        else:
            gates = ("implemented", "executes", "learns", "improves_heldout",
                     "independent_reproduction", "outperforms_baseline",
                     "transfers")
            missing = next((g for g in gates if not ev.get(g)), None)
            gate_status = f"next gate unmet: {missing}" if missing \
                else "max level"
        rows.append({
            "capability": name,
            "level": level,
            "level_name": LEVEL_NAMES[level],
            "status": gate_status,
            "evidence_keys": sorted(k for k, v in ev.items() if v),
        })
    return {"kind": "cortex-capability-maturity", "owned_by": "Fibonera",
            "levels": LEVEL_NAMES, "capabilities": rows}
