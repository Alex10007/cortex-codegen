"""Stage 17 tests for integrated metacognitive planning and hierarchical memory."""
import sqlite3

import pytest

from cortex.memory.engine import MemoryEngine
from cortex.planning.engine import assess_objective, plan
from cortex.knowledge.graph import KnowledgeGraph
from cortex.knowledge.indexer import index_code_root
from cortex.knowledge.vector_store import VectorStore
from cortex.interfaces import Provenance


def _repo_map():
    return {
        "root": "/tmp/repo",
        "file_count": 12,
        "languages": {"python": 8},
        "manifests": ["pyproject.toml"],
        "test_files": ["tests/test_core.py"],
    }


def test_assessment_is_explicitly_heuristic_not_calibrated_confidence():
    assessment = assess_objective("add retry handling to api/client.py", _repo_map())
    assert assessment["assessment_mode"] == "deterministic_heuristic_not_model_confidence"
    assert 0.0 <= assessment["ambiguity_score"] <= 1.0
    assert 0.0 <= assessment["complexity_score"] <= 1.0
    assert assessment["risk_level"] in {"low", "medium", "high", "critical"}
    assert "targeted_tests" in assessment["required_evidence"]


def test_ambiguous_objective_recommends_clarification_and_deeper_verification():
    vague = assess_objective("make it better", _repo_map())
    specific = assess_objective("fix parser indentation in cortex/tokenizer/parser.py",
                                _repo_map())
    assert vague["ambiguity_score"] > specific["ambiguity_score"]
    assert vague["clarification_recommended"] is True
    assert vague["reasoning_depth"] in {"expanded", "deep"}


def test_destructive_production_objective_requires_human_authorization():
    assessment = assess_objective("delete production database", _repo_map())
    assert assessment["risk_level"] == "critical"
    assert assessment["requires_human_authorization"] is True
    assert "explicit_human_authorization" in assessment["required_evidence"]


def test_plan_exposes_assessment_without_changing_core_step_sequence():
    result = plan("add retry handling to api/client.py", "/tmp/repo", _repo_map())
    payload = result.to_dict()
    assert payload["assessment"]["policy_version"] == "cortex-task-assessment-v1"
    assert [step["action"] for step in payload["steps"]] == [
        "analyze", "implement", "test-command", "security-scan", "validate"
    ]


def test_memory_tiers_are_persisted_and_queryable(tmp_path):
    memory = MemoryEngine(str(tmp_path / "memory.sqlite"))
    memory.record("p", "observation", {"x": 1}, verified=False,
                   tier="episodic", confidence=0.6, source="runtime")
    memory.record("p", "algorithm", {"name": "binary-search"}, verified=True,
                   tier="procedural", confidence=1.0, source="test-suite")
    assert memory.events("p", tier="procedural")[0]["kind"] == "algorithm"
    summary = memory.memory_summary("p")
    assert summary["events_by_tier"] == {"episodic": 1, "procedural": 1}


def test_unverified_contradiction_cannot_overwrite_verified_fact(tmp_path):
    memory = MemoryEngine(str(tmp_path / "memory.sqlite"))
    memory.put_fact("p", "database", "PostgreSQL", verified=True,
                    confidence=1.0, source="verified-config")
    memory.put_fact("p", "database", "SQLite", verified=False,
                    confidence=0.4, source="unverified-chat")
    assert memory.get_fact("p", "database") == "PostgreSQL"
    history = memory.fact_history("p", "database")
    assert history[0]["relation"] == "contradiction"
    assert history[0]["disposition"] == "contested"
    assert history[1]["disposition"] == "active"


def test_verified_fact_revision_preserves_old_value_in_history(tmp_path):
    memory = MemoryEngine(str(tmp_path / "memory.sqlite"))
    memory.put_fact("p", "runtime", "Python 3.11", verified=True,
                    source="old-runtime-check")
    memory.put_fact("p", "runtime", "Python 3.12", verified=True,
                    source="new-runtime-check")
    assert memory.get_fact("p", "runtime") == "Python 3.12"
    history = memory.fact_history("p", "runtime")
    assert history[0]["disposition"] == "active"
    assert history[0]["relation"] == "contradiction"
    assert history[1]["disposition"] == "superseded"


def test_memory_rejects_invalid_tier_and_confidence(tmp_path):
    memory = MemoryEngine(str(tmp_path / "memory.sqlite"))
    with pytest.raises(ValueError, match="tier"):
        memory.record("p", "event", {}, verified=False, tier="imaginary")
    with pytest.raises(ValueError, match="confidence"):
        memory.put_fact("p", "x", "y", confidence=1.5)


def test_legacy_memory_schema_migrates_additively(tmp_path):
    db = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, project TEXT NOT NULL,
        kind TEXT NOT NULL, verified INTEGER NOT NULL DEFAULT 0,
        payload TEXT NOT NULL, created_at TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT, project TEXT NOT NULL,
        key TEXT NOT NULL, value TEXT NOT NULL, verified INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL, UNIQUE(project,key))""")
    conn.execute("INSERT INTO facts(project,key,value,verified,updated_at) "
                 "VALUES('p','existing','preserved',1,'2026-01-01T00:00:00+00:00')")
    conn.commit()
    conn.close()

    memory = MemoryEngine(str(db))
    assert memory.get_fact("p", "existing") == "preserved"
    assert memory.events("p") == []
    columns = {row[1] for row in memory._conn.execute("PRAGMA table_info(events)")}
    assert {"memory_tier", "confidence", "source"}.issubset(columns)



def test_indexer_adds_local_import_call_and_inheritance_edges(tmp_path):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "base.py").write_text(
        "class Base:\n    pass\n\ndef helper():\n    return 1\n")
    (root / "pkg" / "child.py").write_text(
        "from .base import helper\n\n"
        "class Child(Base):\n    pass\n\n"
        "def run():\n    return helper()\n")
    store = VectorStore(str(tmp_path / "kg-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "kg-graph.db"))
    counts = index_code_root(str(root), store, graph, version="stage17-test")
    module_child = f"py:{root.name}/pkg/child.py:stage17-test"
    module_base = f"py:{root.name}/pkg/base.py:stage17-test"
    run_id = module_child + ":func:run"
    helper_id = module_base + ":func:helper"
    child_id = module_child + ":class:Child"
    base_id = module_base + ":class:Base"
    assert any(edge["node"] == module_base for edge in
               graph.neighbors(module_child, etype="imports_local"))
    assert any(edge["node"] == helper_id for edge in
               graph.neighbors(run_id, etype="calls"))
    assert any(edge["node"] == base_id for edge in
               graph.neighbors(child_id, etype="inherits"))
    assert counts["local_imports"] >= 1
    assert counts["calls"] >= 1
    assert counts["inherits"] >= 1



def test_adaptive_divergent_budget_uses_existing_task_assessment(monkeypatch):
    from types import SimpleNamespace
    from cortex.evaluation import divergent

    dispatched = []

    def fake_sample(_agent, task, temperatures, _max_tokens, _prefix, tag):
        dispatched.append((task.task_id, tuple(temperatures), tag))
        return [{
            "round": tag,
            "temperature": temperature,
            "sha1": f"{task.task_id}-{temperature}",
            "passed": False,
            "grade": {"exit_code": 1, "stderr_tail": ""},
        } for temperature in temperatures]

    monkeypatch.setattr(divergent, "_sample_and_grade", fake_sample)
    tasks = [
        SimpleNamespace(task_id="focused", instruction=
                        "fix parser indentation in cortex/parser.py"),
        SimpleNamespace(task_id="critical", instruction=
                        "delete production database"),
    ]
    report = divergent.divergent_eval(
        agent=None, tasks=tasks, temperatures=(0.2, 0.7, 1.0),
        repair_round=False, adaptive_budget=True)
    counts = {row["task_id"]: row["n_hypotheses"] for row in report["per_task"]}
    assert counts["focused"] == 1
    assert counts["critical"] == 2
    assert report["aggregate"]["adaptive_budget"] is True
    assert report["aggregate"]["hypotheses_dispatched_total"] == 3
    assert all(row["adaptive_assessment"]["assessment_mode"] ==
               "deterministic_heuristic_not_model_confidence"
               for row in report["per_task"])



def test_output_budget_is_bounded_and_policy_driven():
    from cortex.planning.engine import allocate_output_budget

    assert allocate_output_budget(100, {"compute_budget_fraction": 0.5}) == 50
    assert allocate_output_budget(100, {"compute_budget_fraction": 0.75}) == 75
    assert allocate_output_budget(100, {"compute_budget_fraction": 1.0}) == 100
    assert allocate_output_budget(10, {"compute_budget_fraction": 0.5}) >= 1


def test_retrieval_events_prioritize_verified_procedural_memory(tmp_path):
    memory = MemoryEngine(str(tmp_path / "memory.sqlite"))
    memory.record("p", "recent-observation", {"n": 1}, verified=True,
                  tier="episodic", confidence=1.0, source="runtime")
    memory.record("p", "tested-procedure", {"n": 2}, verified=True,
                  tier="procedural", confidence=0.7, source="tests")
    memory.record("p", "unverified-procedure", {"n": 3}, verified=False,
                  tier="procedural", confidence=1.0, source="untrusted")
    rows = memory.retrieval_events("p")
    assert [row["kind"] for row in rows] == ["tested-procedure", "recent-observation"]
    assert all(row["verified"] for row in rows)


def test_verified_fact_records_expose_source_and_revision(tmp_path):
    memory = MemoryEngine(str(tmp_path / "memory.sqlite"))
    memory.put_fact("p", "framework", "FastAPI", verified=True,
                    confidence=0.95, source="dependency-manifest")
    rows = memory.verified_fact_records("p")
    assert rows == [{
        "key": "framework", "value": "FastAPI", "confidence": 0.95,
        "source": "dependency-manifest", "revision": 1,
        "updated_at": rows[0]["updated_at"],
    }]



def test_verified_pipeline_learning_requires_strict_execution_evidence(tmp_path):
    from cortex.interfaces import Provenance
    from cortex.knowledge.learning import LearningEngine, UnverifiedOutcomeError
    from cortex.knowledge.vector_store import VectorStore
    import hashlib

    store = VectorStore(str(tmp_path / "knowledge.sqlite"))
    learner = LearningEngine(store)
    code = "def verified_add(a, b):\n    return a + b\n"
    provenance = Provenance(
        source_uri="file:///verified_add.py", license="project-local",
        version="test", content_sha256=hashlib.sha256(code.encode()).hexdigest(),
        collected_at="2026-11-21T00:00:00+00:00", collector="stage17-test")
    with pytest.raises(UnverifiedOutcomeError, match="strict sandbox"):
        learner.ingest_verified_pipeline_outcome(
            code, {"status": "verified", "verified": True}, provenance)

    report = {
        "status": "verified", "verified": True,
        "implementation_performed": True, "validation_passed": True,
        "objective_verified": True,
        "tests": {
            "ok": True, "exit_code": 0, "timed_out": False,
            "isolation": "bwrap+user-pid-net-ipc-uts+rlimits+cgroup-v2-cpu-memory-pids+readonly-runtime+workspace",
            "limits": {
                "cgroup_v2_enforced": True,
                "cgroup_empty_after_cleanup": True,
                "aggregate_cpu_limit": True,
                "aggregate_memory_limit": True,
                "aggregate_process_limit": True,
            },
        },
        "security": {"summary": {"critical": 0}},
    }
    item = learner.ingest_verified_pipeline_outcome(code, report, provenance)
    assert learner.is_known_pattern(code)
    assert store.get(item.item_id)["verified"] == 1



def test_change_impact_traverses_call_import_and_inheritance_dependents(tmp_path):
    import hashlib
    from pathlib import Path

    root = tmp_path / "repo"
    pkg = root / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "base.py").write_text(
        "class Base:\n    pass\n\ndef helper():\n    return 1\n")
    (pkg / "child.py").write_text(
        "from .base import helper\n\n"
        "class Child(Base):\n    pass\n\n"
        "def run():\n    return helper()\n")
    vector = VectorStore(str(tmp_path / "impact-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "impact-graph.db"))
    index_code_root(str(root), vector, graph, version="impact-test")

    report = graph.impact_analysis(["pkg/base.py"], str(root))
    impacted = report["impacted_nodes"]
    impacted_paths = {row["path"] for row in impacted}
    assert report["unresolved_paths"] == []
    assert "pkg/child.py" in impacted_paths
    assert any(row["relationship"] == "calls" for row in impacted)
    assert any(row["relationship"] == "inherits" for row in impacted)
    assert report["impacted_path_count"] >= 1


def test_change_impact_rejects_path_escape_and_bounds(tmp_path):
    graph = KnowledgeGraph(str(tmp_path / "impact-graph.db"))
    root = tmp_path / "repo"
    root.mkdir()
    with pytest.raises(ValueError, match="escapes"):
        graph.impact_analysis(["../outside.py"], str(root))
    with pytest.raises(ValueError, match="bounds"):
        graph.impact_analysis(["inside.py"], str(root), max_depth=0)



def test_impact_request_enforces_bounded_input():
    from pydantic import ValidationError
    from cortex.api.app import ImpactRequest

    request = ImpactRequest(repo_path="/tmp/repo", changed_paths=["src/main.py"])
    assert request.max_depth == 3
    assert request.max_nodes == 500
    with pytest.raises(ValidationError):
        ImpactRequest(repo_path="/tmp/repo", changed_paths=[], max_depth=3)
    with pytest.raises(ValidationError):
        ImpactRequest(repo_path="/tmp/repo", changed_paths=["src/main.py"],
                      max_depth=11)



def test_regression_selector_targets_importing_tests_and_reports_limitations(tmp_path):
    from cortex.testing.engine import select_regression_tests

    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pkg" / "base.py").write_text("def helper():\n    return 1\n")
    (root / "tests" / "test_base.py").write_text(
        "from pkg.base import helper\n\ndef test_helper():\n    assert helper() == 1\n")
    (root / "tests" / "test_unrelated.py").write_text(
        "def test_other():\n    assert True\n")
    report = {
        "changed_paths": ["pkg/base.py"],
        "impacted_nodes": [],
    }
    selected = select_regression_tests(str(root), report)
    assert selected["mode"] == "targeted_static_recommendation"
    assert selected["targets"] == ["tests/test_base.py"]
    assert selected["fallback_to_full_suite"] is False
    assert selected["limitations"]


def test_regression_selector_falls_back_to_full_suite_when_unresolved(tmp_path):
    from cortex.testing.engine import select_regression_tests

    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "test_one.py").write_text("def test_one():\n    assert True\n")
    selected = select_regression_tests(
        str(root), {"changed_paths": ["src/no_match.py"], "impacted_nodes": []})
    assert selected["mode"] == "full_suite_fallback"
    assert selected["targets"] == ["tests/test_one.py"]
    assert selected["fallback_to_full_suite"] is True
