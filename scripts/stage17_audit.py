"""Static integration audit for Stage 17 capability foundations.

This checks source ownership and declared implementation status. It does not
execute tests, benchmark model quality, or certify capabilities marked partial.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

CAPABILITIES = [
    {"id": "recursive_reasoning", "owner": "cortex/evaluation/divergent.py",
     "status": "partial", "gap": "No general-purpose recursive reasoning controller."},
    {"id": "engineering_world_model", "owner": "cortex/knowledge/graph.py",
     "status": "partial", "gap": "Graph exists; cross-language dependency and causal coverage incomplete."},
    {"id": "adaptive_divergent_reasoning", "owner": "cortex/evaluation/divergent.py",
     "status": "partial", "gap": "Candidate count and depth are not yet driven by task assessment."},
    {"id": "metacognitive_control", "owner": "cortex/planning/engine.py",
     "status": "foundation_implemented", "gap": "Deterministic heuristic policy only; no learned calibration."},
    {"id": "whole_repository_intelligence", "owner": "cortex/planning/engine.py",
     "status": "partial", "gap": "Repository map is incomplete for cross-language symbols and call graphs."},
    {"id": "autonomous_engineering", "owner": "cortex/pipeline/engineer.py",
     "status": "partial", "gap": "Patcher is injected; end-to-end model-directed patch generation is not established."},
    {"id": "execution_grounded_learning", "owner": "cortex/knowledge/learning.py",
     "status": "partial", "gap": "Verified outcome ingestion exists; online policy/model learning is not established."},
    {"id": "formal_semantic_verification", "owner": "cortex/testing/engine.py",
     "status": "partial", "gap": "Tests and static analysis exist; symbolic and formal proof integration is absent."},
    {"id": "continual_learning", "owner": "cortex/training/engine.py",
     "status": "not_implemented", "gap": "No validated continual-learning/anti-forgetting protocol."},
    {"id": "hierarchical_persistent_memory", "owner": "cortex/memory/engine.py",
     "status": "foundation_implemented", "gap": "Tiered events and contradiction history added; consolidation and decay remain."},
    {"id": "curriculum_data_intelligence", "owner": "cortex/training/dataset_v2.py",
     "status": "partial", "gap": "Dataset audits exist; autonomous quality-driven curriculum is not established."},
    {"id": "adaptive_compute_allocation", "owner": "cortex/planning/engine.py",
     "status": "partial", "gap": "Assessment recommends depth; dynamic model compute allocation remains."},
    {"id": "mathematical_scientific_reasoning", "owner": "cortex/evaluation",
     "status": "not_implemented", "gap": "No dedicated independently scored math/science benchmark suite."},
    {"id": "multimodal_engineering_design", "owner": "cortex",
     "status": "not_implemented", "gap": "Current model path is text-only."},
    {"id": "game_simulation_engineering", "owner": "cortex/pipeline/engineer.py",
     "status": "not_implemented", "gap": "No verified game/simulation-specific toolchain or benchmark suite."},
]

REQUIRED = {
    "cortex/planning/engine.py": ("assess_objective",),
    "cortex/memory/engine.py": ("MEMORY_TIERS", "fact_history", "memory_summary"),
    "cortex/knowledge/graph.py": ("class KnowledgeGraph",),
    "cortex/knowledge/indexer.py": ("imports_local", "calls", "inherits"),
    "cortex/evaluation/divergent.py": ("def divergent_eval", "adaptive_budget: bool = False"),
    "cortex/core.py": ("retrieval_events", "adaptive_compute"),
    "cortex/api/app.py": ("adaptive_compute: bool = False",),
    "cortex/pipeline/engineer.py": ("def run_engineering_pipeline",),
    "cortex/execution/sandbox.py": ("def run",),
    "cortex/training/engine.py": ("def train",),
}


def build_report() -> dict[str, Any]:
    checks = []
    for relative, symbols in REQUIRED.items():
        path = ROOT / relative
        source = path.read_text(encoding="utf-8") if path.is_file() else ""
        missing = [symbol for symbol in symbols if symbol not in source]
        checks.append({"path": relative, "exists": path.is_file(),
                       "required_symbols": list(symbols), "missing_symbols": missing,
                       "status": "PASS" if path.is_file() and not missing else "FAIL"})
    return {
        "kind": "cortex-stage17-capability-inventory",
        "owner": "Fibonera",
        "stage": 17,
        "schema_version": "1.0",
        "audit_type": "static_source_inventory_not_runtime_qualification",
        "capability_count": len(CAPABILITIES),
        "foundation_implemented": sum(
            row["status"] == "foundation_implemented" for row in CAPABILITIES),
        "partial": sum(row["status"] == "partial" for row in CAPABILITIES),
        "not_implemented": sum(row["status"] == "not_implemented" for row in CAPABILITIES),
        "capabilities": CAPABILITIES,
        "canonical_subsystem_checks": checks,
        "duplicate_subsystem_policy": "Extend canonical owner modules; do not create parallel memory, planner, MoE, execution, or evaluation engines.",
        "stage12_champion_unchanged": True,
        "model_checkpoint_modified": False,
        "training_performed": False,
        "tests_executed": False,
        "runtime_qualified": False,
    }


def main() -> int:
    report = build_report()
    output = ROOT / "artifacts" / "stage17_capability_inventory.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all(c["status"] == "PASS" for c in report["canonical_subsystem_checks"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
