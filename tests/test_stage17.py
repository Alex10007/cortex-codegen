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



def test_regression_selector_never_returns_partial_targeted_suite_when_over_limit(tmp_path):
    from cortex.testing.engine import select_regression_tests

    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pkg" / "module.py").write_text("def work():\n    return 1\n")
    for name in ("test_one.py", "test_two.py"):
        (root / "tests" / name).write_text(
            "from pkg.module import work\ndef test_work():\n    assert work() == 1\n")
    selected = select_regression_tests(
        str(root), {"changed_paths": ["pkg/module.py"], "impacted_nodes": []},
        max_tests=1)
    assert selected["mode"] == "full_suite_fallback"
    assert selected["targets"] == []
    assert selected["truncated"] is True
    assert selected["full_suite_command"] == ["python", "-m", "pytest", "-q"]


def test_pipeline_attaches_change_impact_without_skipping_full_suite(monkeypatch):
    from cortex.core import CortexAgent
    from types import SimpleNamespace

    audit_events = []

    class Audit:
        def record(self, event, payload):
            audit_events.append((event, payload))

    class Memory:
        def record(self, *args, **kwargs):
            return None

    agent = CortexAgent.__new__(CortexAgent)
    agent.audit = Audit()
    agent.memory = Memory()
    impact = {
        "kind": "cortex-change-impact",
        "impacted_nodes": [{"path": "src/consumer.py"}],
        "unresolved_paths": [],
        "regression_test_selection": {
            "mode": "targeted_static_recommendation",
            "targets": ["tests/test_consumer.py"],
        },
    }
    calls = []

    def analyze_change_impact(paths, root):
        calls.append((paths, root))
        return impact

    agent.analyze_change_impact = analyze_change_impact

    def fake_pipeline(objective, repo_path, patcher=None, **kwargs):
        edits = patcher(SimpleNamespace()) if patcher else {}
        return {
            "status": "changes-tested-not-objective-verified",
            "files_changed": sorted(edits),
            "tests": {"ok": True, "command": ["python", "-m", "pytest", "-q"]},
            "verified": False,
        }

    monkeypatch.setattr("cortex.pipeline.engineer.run_engineering_pipeline",
                        fake_pipeline)
    monkeypatch.setattr("cortex.core.record_project", lambda _path: "project")

    report = agent.run_pipeline(
        "change a module", "/repo",
        patcher=lambda _plan: {"src/consumer.py": "value = 1\n"})

    assert calls == [(["src/consumer.py"], "/repo")]
    assert report["change_impact"] == impact
    assert report["tests"]["command"] == ["python", "-m", "pytest", "-q"]
    event = next(payload for name, payload in audit_events
                 if name == "pipeline.change_impact")
    assert event["execution_policy"] == "full_configured_suite_unchanged"


def test_pipeline_keeps_going_when_advisory_impact_analysis_fails(monkeypatch):
    from cortex.core import CortexAgent
    from types import SimpleNamespace

    class Audit:
        def record(self, event, payload):
            return None

    class Memory:
        def record(self, *args, **kwargs):
            return None

    agent = CortexAgent.__new__(CortexAgent)
    agent.audit = Audit()
    agent.memory = Memory()

    def fail_impact(_paths, _root):
        raise RuntimeError("graph unavailable")

    agent.analyze_change_impact = fail_impact
    observed = {}

    def fake_pipeline(objective, repo_path, patcher=None, **kwargs):
        observed["edits"] = patcher(SimpleNamespace()) if patcher else {}
        return {"status": "tests-failed", "files_changed": [],
                "tests": {"ok": False}, "verified": False}

    monkeypatch.setattr("cortex.pipeline.engineer.run_engineering_pipeline",
                        fake_pipeline)
    monkeypatch.setattr("cortex.core.record_project", lambda _path: "project")

    report = agent.run_pipeline(
        "change a module", "/repo",
        patcher=lambda _plan: {"src/a.py": "value = 2\n"})

    assert observed["edits"] == {"src/a.py": "value = 2\n"}
    assert report["change_impact"]["status"] == "unavailable"
    assert report["change_impact"]["verification_policy"] == (
        "full_configured_suite_unchanged")


def test_cross_language_indexer_adds_symbols_to_canonical_graph(tmp_path):
    from cortex.knowledge.graph import KnowledgeGraph
    from cortex.knowledge.indexer import index_code_root
    from cortex.knowledge.vector_store import VectorStore

    root = tmp_path / "multi"
    (root / "src").mkdir(parents=True)
    (root / "src" / "util.ts").write_text(
        "export interface User {}\nexport function normalizeUser() {}\n")
    (root / "src" / "main.ts").write_text(
        "import { normalizeUser } from './util';\n"
        "export class UserService {}\n")
    (root / "src" / "lib.rs").write_text(
        "pub struct Engine;\npub fn execute() {}\n")
    (root / "src" / "main.go").write_text(
        "package main\nfunc Run() {}\ntype Runner struct {}\n")
    (root / "src" / "Main.java").write_text(
        "public class Main {}\npublic interface Service {}\n")

    store = VectorStore(str(tmp_path / "cross-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "cross-graph.db"))
    counts = index_code_root(str(root), store, graph, version="cross-test")

    assert counts["foreign_files"] == 5
    assert counts["foreign_symbols"] >= 8
    assert counts["foreign_local_imports"] == 1

    module_id = "typescript:multi/src/main.ts:cross-test"
    outgoing = graph.neighbors(module_id, etype="imports_local")
    assert len(outgoing) == 1
    assert outgoing[0]["node"] == "typescript:multi/src/util.ts:cross-test"

    rust_module = "rust:multi/src/lib.rs:cross-test"
    assert graph.neighbors(rust_module, etype="defines")
    go_module = "go:multi/src/main.go:cross-test"
    assert graph.neighbors(go_module, etype="defines")
    java_module = "java:multi/src/Main.java:cross-test"
    assert graph.neighbors(java_module, etype="defines")


def test_cross_language_indexer_reports_unresolved_relative_imports(tmp_path):
    from cortex.knowledge.graph import KnowledgeGraph
    from cortex.knowledge.indexer import index_code_root
    from cortex.knowledge.vector_store import VectorStore

    root = tmp_path / "unresolved"
    root.mkdir()
    (root / "main.ts").write_text("import { missing } from './not-present';\n")
    store = VectorStore(str(tmp_path / "unresolved-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "unresolved-graph.db"))
    counts = index_code_root(str(root), store, graph, version="unresolved-test")
    assert counts["foreign_files"] == 1
    assert counts["foreign_local_imports"] == 0
    assert counts["foreign_unresolved_imports"] == 1



def test_dependency_cycle_analysis_reports_strongly_connected_components(tmp_path):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    files = [root / "pkg" / name for name in ("a.py", "b.py", "c.py", "outside.py")]
    for path in files:
        path.write_text("# indexed fixture\n")
    graph = KnowledgeGraph(str(tmp_path / "cycles.sqlite"))
    provenance = lambda path: Provenance(
        source_uri=path.as_uri(), license="project-local", version="test",
        content_sha256="fixture", collected_at="2026-11-21T00:00:00+00:00",
        collector="stage17k-test")
    ids = {path.name: "node:" + path.name for path in files}
    for path in files:
        graph.upsert_node(ids[path.name], "module", path.name, provenance(path))
    graph.upsert_edge(ids["a.py"], ids["b.py"], "imports_local", "test")
    graph.upsert_edge(ids["b.py"], ids["c.py"], "imports_local", "test")
    graph.upsert_edge(ids["c.py"], ids["a.py"], "imports_local", "test")
    graph.upsert_edge(ids["outside.py"], ids["a.py"], "imports_local", "test")
    graph.commit()
    report = graph.dependency_cycles(str(root))
    assert report["cycle_count"] == 1
    assert set(report["cycles"][0]["paths"]) == {
        "pkg/a.py", "pkg/b.py", "pkg/c.py"}
    assert report["truncated"] is False


def test_dependency_cycle_analysis_handles_self_loop_and_bounds(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "self.py"
    source.write_text("# self cycle\n")
    graph = KnowledgeGraph(str(tmp_path / "self-cycle.sqlite"))
    provenance = Provenance(
        source_uri=source.as_uri(), license="project-local", version="test",
        content_sha256="fixture", collected_at="2026-11-21T00:00:00+00:00",
        collector="stage17k-test")
    graph.upsert_node("self", "module", "self.py", provenance)
    graph.upsert_edge("self", "self", "imports_local", "test")
    graph.commit()
    assert graph.dependency_cycles(str(root))["cycle_count"] == 1
    with pytest.raises(ValueError, match="max_nodes"):
        graph.dependency_cycles(str(root), max_nodes=0)


def test_architecture_boundary_rules_report_only_explicit_prefix_matches(tmp_path):
    root = tmp_path / "repo"
    (root / "presentation").mkdir(parents=True)
    (root / "domain").mkdir()
    (root / "domain2").mkdir()
    paths = [root / "presentation" / "view.py",
             root / "domain" / "model.py",
             root / "domain2" / "other.py"]
    for path in paths:
        path.write_text("# architecture fixture\n")
    graph = KnowledgeGraph(str(tmp_path / "boundaries.sqlite"))
    provenance = lambda path: Provenance(
        source_uri=path.as_uri(), license="project-local", version="test",
        content_sha256="fixture", collected_at="2026-11-21T00:00:00+00:00",
        collector="stage17k-test")
    for index, path in enumerate(paths):
        graph.upsert_node(str(index), "module", path.name, provenance(path))
    graph.upsert_edge("0", "1", "imports_local", "test")
    graph.upsert_edge("0", "2", "imports_local", "test")
    graph.commit()
    report = graph.architecture_boundary_violations(
        str(root), [{"source_prefix": "presentation",
                     "forbidden_target_prefix": "domain"}])
    assert report["violation_count"] == 1
    assert report["violations"][0]["target_path"] == "domain/model.py"
    assert report["truncated"] is False


def test_architecture_boundary_rules_reject_path_traversal(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    graph = KnowledgeGraph(str(tmp_path / "invalid-boundaries.sqlite"))
    with pytest.raises(ValueError, match="repository-relative"):
        graph.architecture_boundary_violations(
            str(root), [{"source_prefix": "../src",
                         "forbidden_target_prefix": "vendor"}])



def test_scope_aware_indexer_resolves_nested_and_instance_method_calls(tmp_path):
    root = tmp_path / "scope-repo"
    root.mkdir()
    (root / "mod.py").write_text(
        "def outer():\n"
        "    def helper():\n"
        "        return 1\n"
        "    def inner():\n"
        "        return helper()\n"
        "    return inner()\n"
        "class Worker:\n"
        "    def helper(self):\n"
        "        return 2\n"
        "    def run(self):\n"
        "        return self.helper()\n"
    )
    store = VectorStore(str(tmp_path / "scope-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "scope-graph.db"))
    counts = index_code_root(str(root), store, graph, version="scope-test")
    module = f"py:{root.name}/mod.py:scope-test"
    inner = module + ":func:outer.inner"
    helper = module + ":func:outer.helper"
    worker_run = module + ":func:Worker.run"
    worker_helper = module + ":func:Worker.helper"
    assert graph.neighbors(inner, etype="calls")[0]["node"] == helper
    assert graph.neighbors(worker_run, etype="calls")[0]["node"] == worker_helper
    assert counts["calls"] >= 3


def test_scope_aware_indexer_does_not_resolve_shadowed_or_ambiguous_calls(tmp_path):
    root = tmp_path / "shadow-repo"
    root.mkdir()
    (root / "mod.py").write_text(
        "def target():\n"
        "    return 1\n"
        "def caller(target):\n"
        "    return target()\n"
        "def duplicate():\n"
        "    return 2\n"
        "def duplicate():\n"
        "    return 3\n"
        "def ambiguous_caller():\n"
        "    return duplicate()\n"
    )
    store = VectorStore(str(tmp_path / "shadow-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "shadow-graph.db"))
    counts = index_code_root(str(root), store, graph, version="shadow-test")
    module = f"py:{root.name}/mod.py:shadow-test"
    caller = module + ":func:caller"
    ambiguous_caller = module + ":func:ambiguous_caller"
    assert graph.neighbors(caller, etype="calls") == []
    assert graph.neighbors(ambiguous_caller, etype="calls") == []
    assert counts["ambiguous_calls"] == 1
    assert counts["unresolved_calls"] >= 2


def test_scope_aware_indexer_links_python_type_annotations(tmp_path):
    root = tmp_path / "types-repo"
    root.mkdir()
    (root / "models.py").write_text(
        "class Model:\n"
        "    pass\n"
        "def consume(value: Model) -> Model:\n"
        "    return value\n"
    )
    store = VectorStore(str(tmp_path / "types-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "types-graph.db"))
    counts = index_code_root(str(root), store, graph, version="types-test")
    module = f"py:{root.name}/models.py:types-test"
    consume = module + ":func:consume"
    model = module + ":class:Model"
    references = graph.neighbors(consume, etype="type_uses")
    assert {item["node"] for item in references} == {model}
    assert counts["type_references"] == 2


def test_cross_module_python_import_aliases_resolve_calls_and_types(tmp_path):
    root = tmp_path / "cross-module"
    package = root / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "service.py").write_text(
        "class Model:\n"
        "    pass\n"
        "def execute(value):\n"
        "    return value\n"
    )
    (root / "consumer.py").write_text(
        "from pkg.service import execute as run, Model as Data\n"
        "def consume(value: Data) -> Data:\n"
        "    return run(value)\n"
        "class Child(Data):\n"
        "    pass\n"
    )
    store = VectorStore(str(tmp_path / "cross-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "cross-graph.db"))
    counts = index_code_root(str(root), store, graph, version="cross-test")
    service = f"py:{root.name}/pkg/service.py:cross-test"
    consumer = f"py:{root.name}/consumer.py:cross-test"
    execute = service + ":func:execute"
    model = service + ":class:Model"
    consume = consumer + ":func:consume"
    child = consumer + ":class:Child"
    assert graph.neighbors(consume, etype="calls")[0]["node"] == execute
    assert graph.neighbors(consume, etype="type_uses")[0]["node"] == model
    assert graph.neighbors(child, etype="inherits")[0]["node"] == model
    assert counts["resolved_imports"] >= 2
    assert counts["cross_module_calls"] == 1


def test_relative_python_imports_and_module_alias_calls_resolve(tmp_path):
    root = tmp_path / "relative-imports"
    package = root / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "shared.py").write_text("def work():\n    return 1\n")
    (package / "consumer.py").write_text(
        "from .shared import work\n"
        "import pkg.shared as shared\n"
        "def first():\n"
        "    return work()\n"
        "def second():\n"
        "    return shared.work()\n"
    )
    store = VectorStore(str(tmp_path / "relative-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "relative-graph.db"))
    counts = index_code_root(str(root), store, graph, version="relative-test")
    module = f"py:{root.name}/pkg/consumer.py:relative-test"
    work = f"py:{root.name}/pkg/shared.py:relative-test:func:work"
    assert graph.neighbors(module + ":func:first", etype="calls")[0]["node"] == work
    assert graph.neighbors(module + ":func:second", etype="calls")[0]["node"] == work
    assert counts["cross_module_calls"] == 2


def test_python_import_resolution_reports_unresolved_and_ambiguous_without_guessing(tmp_path):
    root = tmp_path / "ambiguous-imports"
    package = root / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "dupe.py").write_text(
        "def target():\n    return 1\n"
        "def target():\n    return 2\n"
    )
    (root / "consumer.py").write_text(
        "from pkg.dupe import target\n"
        "import third_party_library\n"
        "def call():\n    return target()\n"
    )
    store = VectorStore(str(tmp_path / "ambiguous-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "ambiguous-graph.db"))
    counts = index_code_root(str(root), store, graph, version="ambiguous-test")
    consumer = f"py:{root.name}/consumer.py:ambiguous-test:func:call"
    assert graph.neighbors(consumer, etype="calls") == []
    assert counts["ambiguous_imports"] == 1
    assert counts["unresolved_imports"] >= 1


def test_stage17n_resolves_function_local_import_without_leaking_scope(tmp_path):
    root = tmp_path / "local-import"
    root.mkdir()
    (root / "service.py").write_text("def run():\n    return 1\n")
    (root / "consumer.py").write_text(
        "def inside():\n"
        "    from service import run as local_run\n"
        "    return local_run()\n"
        "def outside():\n"
        "    return local_run()\n"
    )
    store = VectorStore(str(tmp_path / "local-import-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "local-import-graph.db"))
    counts = index_code_root(str(root), store, graph, version="stage17n-local")
    consumer = f"py:{root.name}/consumer.py:stage17n-local"
    service_run = f"py:{root.name}/service.py:stage17n-local:func:run"
    assert graph.neighbors(consumer + ":func:inside", etype="calls")[0]["node"] == service_run
    assert graph.neighbors(consumer + ":func:outside", etype="calls") == []
    assert counts["cross_module_calls"] == 1


def test_stage17n_nonlocal_import_binding_resolves_in_enclosing_function(tmp_path):
    root = tmp_path / "nonlocal-import"
    root.mkdir()
    (root / "service.py").write_text("def run():\n    return 1\n")
    (root / "consumer.py").write_text(
        "def outer():\n"
        "    from service import run as selected\n"
        "    def inner():\n"
        "        nonlocal selected\n"
        "        return selected()\n"
        "    return inner()\n"
    )
    store = VectorStore(str(tmp_path / "nonlocal-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "nonlocal-graph.db"))
    index_code_root(str(root), store, graph, version="stage17n-nonlocal")
    consumer = f"py:{root.name}/consumer.py:stage17n-nonlocal"
    service_run = f"py:{root.name}/service.py:stage17n-nonlocal:func:run"
    assert graph.neighbors(consumer + ":func:outer.inner", etype="calls")[0]["node"] == service_run


def test_stage17n_package_reexports_resolve_consumer_import(tmp_path):
    root = tmp_path / "reexport-repo"
    package = root / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("from .service import run as public_run\n")
    (package / "service.py").write_text("def run():\n    return 1\n")
    (root / "consumer.py").write_text(
        "from pkg import public_run\n"
        "def invoke():\n"
        "    return public_run()\n"
    )
    store = VectorStore(str(tmp_path / "reexport-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "reexport-graph.db"))
    counts = index_code_root(str(root), store, graph, version="stage17n-reexport")
    consumer = f"py:{root.name}/consumer.py:stage17n-reexport:func:invoke"
    service_run = f"py:{root.name}/pkg/service.py:stage17n-reexport:func:run"
    assert graph.neighbors(consumer, etype="calls")[0]["node"] == service_run
    assert counts["resolved_reexports"] >= 1
    consumer_module = f"py:{root.name}/consumer.py:stage17n-reexport"
    package_module = f"py:{root.name}/pkg/__init__.py:stage17n-reexport"
    assert graph.neighbors(consumer_module, etype="imports_module")[0]["node"] == package_module


def test_stage17n_conditional_import_is_distinguished(tmp_path):
    root = tmp_path / "conditional-import"
    root.mkdir()
    (root / "service.py").write_text("def run():\n    return 1\n")
    (root / "consumer.py").write_text(
        "if True:\n"
        "    from service import run\n"
        "def invoke():\n"
        "    return run()\n"
    )
    store = VectorStore(str(tmp_path / "conditional-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "conditional-graph.db"))
    counts = index_code_root(str(root), store, graph, version="stage17n-conditional")
    consumer_module = f"py:{root.name}/consumer.py:stage17n-conditional"
    consumer_call = consumer_module + ":func:invoke"
    service_run = f"py:{root.name}/service.py:stage17n-conditional:func:run"
    assert graph.neighbors(consumer_call, etype="calls")[0]["node"] == service_run
    assert counts["conditional_imports"] == 1
    assert counts["resolved_imports"] >= 1


def test_stage17n_global_import_binding_resolves_at_module_scope(tmp_path):
    root = tmp_path / "global-import"
    root.mkdir()
    (root / "service.py").write_text("def run():\n    return 1\n")
    (root / "consumer.py").write_text(
        "def invoke():\n"
        "    global global_run\n"
        "    from service import run as global_run\n"
        "    return global_run()\n"
    )
    store = VectorStore(str(tmp_path / "global-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "global-graph.db"))
    index_code_root(str(root), store, graph, version="stage17n-global")
    consumer = f"py:{root.name}/consumer.py:stage17n-global:func:invoke"
    service_run = f"py:{root.name}/service.py:stage17n-global:func:run"
    assert graph.neighbors(consumer, etype="calls")[0]["node"] == service_run


def test_stage17n_class_body_import_does_not_leak_into_method_scope(tmp_path):
    root = tmp_path / "class-local-import"
    root.mkdir()
    (root / "service.py").write_text("def run():\n    return 1\n")
    (root / "consumer.py").write_text(
        "class Container:\n"
        "    from service import run as class_run\n"
        "    def method(self):\n"
        "        return class_run()\n"
    )
    store = VectorStore(str(tmp_path / "class-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "class-graph.db"))
    index_code_root(str(root), store, graph, version="stage17n-class")
    consumer = f"py:{root.name}/consumer.py:stage17n-class:func:Container.method"
    assert graph.neighbors(consumer, etype="calls") == []

def test_compiler_scope_graph_captures_nested_free_global_and_special_scopes(tmp_path):
    root = tmp_path / "compiler-scopes"
    root.mkdir()
    (root / "scope_cases.py").write_text(
        "module_global = 0\n"
        "def outer(argument):\n"
        "    captured = 1\n"
        "    def inner():\n"
        "        nonlocal captured\n"
        "        global module_global\n"
        "        module_global = 2\n"
        "        return argument + captured\n"
        "    return inner\n"
        "def with_lambda(values):\n"
        "    return list(map(lambda entry: entry + 1, values))\n"
        "def with_comprehension(values):\n"
        "    return [entry for entry in values]\n"
        "def with_exception_target():\n"
        "    try:\n"
        "        raise ValueError('fixture')\n"
        "    except ValueError as error:\n"
        "        return str(error)\n"
    )
    store = VectorStore(str(tmp_path / "compiler-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "compiler-graph.db"))
    counts = index_code_root(str(root), store, graph, version="compiler-scope-test")

    module = f"py:{root.name}/scope_cases.py:compiler-scope-test"
    scopes = graph._conn.execute(
        "SELECT id, label FROM nodes WHERE type='python_compiler_scope'"
    ).fetchall()
    labels = {label for _node_id, label in scopes}
    assert any("function:outer@" in label for label in labels)
    assert any("function:inner@" in label for label in labels)
    assert any("function:lambda@" in label for label in labels)
    # Python 3.12+ may inline comprehensions (PEP 709); earlier interpreters
    # expose a dedicated listcomp compiler scope.
    has_dedicated_comprehension = any("listcomp" in label for label in labels)
    assert has_dedicated_comprehension or any(
        "function:with_comprehension@" in label for label in labels)

    def scope_id(fragment):
        matches = [node_id for node_id, label in scopes if fragment in label]
        assert len(matches) == 1
        return matches[0]

    inner = scope_id("function:inner@")
    inner_nonlocals = graph.neighbors(inner, etype="compiler_nonlocal")
    inner_globals = graph.neighbors(inner, etype="compiler_global")
    inner_declared_globals = graph.neighbors(inner, etype="compiler_declared_global")
    binding_labels = {
        row[0]: row[1] for row in graph._conn.execute(
            "SELECT id, label FROM nodes WHERE type='python_compiler_symbol'"
        ).fetchall()
    }
    assert any(binding_labels[row["node"]].endswith("::captured")
               for row in inner_nonlocals)
    assert any(binding_labels[row["node"]].endswith("::module_global")
               for row in inner_globals)
    assert any(binding_labels[row["node"]].endswith("::module_global")
               for row in inner_declared_globals)

    comprehension = (scope_id("listcomp") if has_dedicated_comprehension
                    else scope_id("function:with_comprehension@"))
    comprehension_locals = graph.neighbors(comprehension, etype="compiler_local")
    assert any(binding_labels[row["node"]].endswith("::entry")
               for row in comprehension_locals)

    handler = scope_id("function:with_exception_target@")
    handler_locals = graph.neighbors(handler, etype="compiler_local")
    assert any(binding_labels[row["node"]].endswith("::error")
               for row in handler_locals)
    assert graph.neighbors(module, etype="has_compiler_scope")
    assert counts["compiler_scopes"] == len(scopes)
    assert counts["compiler_symbols"] > counts["compiler_scopes"]
    assert counts["compiler_free_symbols"] >= 1
    assert counts["compiler_nonlocal_symbols"] >= 1
    assert counts["compiler_global_symbols"] >= 1
    assert counts["compiler_scope_parse_errors"] == 0

def test_stage17p_compiler_global_declaration_bypasses_enclosing_shadow(tmp_path):
    root = tmp_path / "compiler-global-resolution"
    root.mkdir()
    (root / "mod.py").write_text(
        "def selected():\n"
        "    return 'module'\n"
        "def outer():\n"
        "    def selected():\n"
        "        return 'outer'\n"
        "    def inner():\n"
        "        global selected\n"
        "        return selected()\n"
        "    return inner()\n"
    )
    store = VectorStore(str(tmp_path / "global-resolution-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "global-resolution-graph.db"))
    counts = index_code_root(str(root), store, graph, version="stage17p-global")
    module = f"py:{root.name}/mod.py:stage17p-global"
    inner = module + ":func:outer.inner"
    module_selected = module + ":func:selected"
    outer_selected = module + ":func:outer.selected"
    targets = {item["node"] for item in graph.neighbors(inner, etype="calls")}
    assert targets == {module_selected}
    assert outer_selected not in targets
    assert counts["compiler_global_symbols"] >= 1


def test_stage17p_compiler_nonlocal_declaration_selects_nearest_enclosing_binding(tmp_path):
    root = tmp_path / "compiler-nonlocal-resolution"
    root.mkdir()
    (root / "mod.py").write_text(
        "def selected():\n"
        "    return 'module'\n"
        "def outer():\n"
        "    def selected():\n"
        "        return 'outer'\n"
        "    def middle():\n"
        "        def inner():\n"
        "            nonlocal selected\n"
        "            return selected()\n"
        "        return inner()\n"
        "    return middle()\n"
    )
    store = VectorStore(str(tmp_path / "nonlocal-resolution-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "nonlocal-resolution-graph.db"))
    counts = index_code_root(str(root), store, graph, version="stage17p-nonlocal")
    module = f"py:{root.name}/mod.py:stage17p-nonlocal"
    inner = module + ":func:outer.middle.inner"
    outer_selected = module + ":func:outer.selected"
    module_selected = module + ":func:selected"
    targets = {item["node"] for item in graph.neighbors(inner, etype="calls")}
    assert targets == {outer_selected}
    assert module_selected not in targets
    assert counts["compiler_nonlocal_symbols"] >= 1


def test_stage17q_resolves_nested_package_module_attribute_call_chains(tmp_path):
    root = tmp_path / "nested-module-chain"
    package = root / "pkg"
    services = package / "services"
    services.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (services / "__init__.py").write_text("")
    (services / "worker.py").write_text(
        "def run():\n"
        "    return 'ok'\n"
    )
    (root / "consumer.py").write_text(
        "import pkg as p\n"
        "def invoke():\n"
        "    return p.services.worker.run()\n"
    )
    store = VectorStore(str(tmp_path / "nested-chain-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "nested-chain-graph.db"))
    counts = index_code_root(str(root), store, graph, version="stage17q-chain")
    consumer = f"py:{root.name}/consumer.py:stage17q-chain:func:invoke"
    worker_run = (
        f"py:{root.name}/pkg/services/worker.py:stage17q-chain:func:run"
    )
    targets = {item["node"] for item in graph.neighbors(consumer, etype="calls")}
    assert targets == {worker_run}
    assert counts["cross_module_calls"] == 1


def test_stage17q_resolves_scoped_nested_module_alias_without_leaking(tmp_path):
    root = tmp_path / "scoped-module-chain"
    root.mkdir()
    (root / "service.py").write_text(
        "def run():\n"
        "    return 1\n"
    )
    (root / "consumer.py").write_text(
        "def inside():\n"
        "    import service as selected\n"
        "    return selected.run()\n"
        "def outside():\n"
        "    return selected.run()\n"
    )
    store = VectorStore(str(tmp_path / "scoped-chain-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "scoped-chain-graph.db"))
    counts = index_code_root(str(root), store, graph, version="stage17q-scoped")
    consumer = f"py:{root.name}/consumer.py:stage17q-scoped"
    service_run = f"py:{root.name}/service.py:stage17q-scoped:func:run"
    inside_targets = {
        item["node"] for item in graph.neighbors(
            consumer + ":func:inside", etype="calls")
    }
    outside_targets = graph.neighbors(
        consumer + ":func:outside", etype="calls")
    assert inside_targets == {service_run}
    assert outside_targets == []
    assert counts["cross_module_calls"] == 1


def test_stage17r_resolves_unique_callable_alias_to_local_function(tmp_path):
    root = tmp_path / "callable-alias"
    root.mkdir()
    (root / "mod.py").write_text(
        "def implementation():\n"
        "    return 7\n"
        "def invoke():\n"
        "    selected = implementation\n"
        "    return selected()\n"
    )
    store = VectorStore(str(tmp_path / "callable-alias-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "callable-alias-graph.db"))
    counts = index_code_root(str(root), store, graph, version="stage17r-alias")
    module = f"py:{root.name}/mod.py:stage17r-alias"
    invoke = module + ":func:invoke"
    implementation = module + ":func:implementation"
    targets = {item["node"] for item in graph.neighbors(invoke, etype="calls")}
    assert targets == {implementation}
    assert counts["calls"] >= 1


def test_stage17r_rejects_reassigned_callable_alias(tmp_path):
    root = tmp_path / "reassigned-callable-alias"
    root.mkdir()
    (root / "mod.py").write_text(
        "def first():\n"
        "    return 1\n"
        "def second():\n"
        "    return 2\n"
        "def invoke():\n"
        "    selected = first\n"
        "    selected = second\n"
        "    return selected()\n"
    )
    store = VectorStore(str(tmp_path / "reassigned-alias-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "reassigned-alias-graph.db"))
    counts = index_code_root(str(root), store, graph, version="stage17r-reassigned")
    module = f"py:{root.name}/mod.py:stage17r-reassigned"
    invoke = module + ":func:invoke"
    assert graph.neighbors(invoke, etype="calls") == []
    assert counts["unresolved_calls"] >= 1


def test_stage17r_resolves_callable_alias_of_imported_function(tmp_path):
    root = tmp_path / "imported-callable-alias"
    root.mkdir()
    (root / "service.py").write_text(
        "def execute():\n"
        "    return 9\n"
    )
    (root / "consumer.py").write_text(
        "from service import execute\n"
        "def invoke():\n"
        "    selected = execute\n"
        "    return selected()\n"
    )
    store = VectorStore(str(tmp_path / "imported-alias-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "imported-alias-graph.db"))
    counts = index_code_root(str(root), store, graph, version="stage17r-imported")
    consumer = f"py:{root.name}/consumer.py:stage17r-imported:func:invoke"
    execute = f"py:{root.name}/service.py:stage17r-imported:func:execute"
    targets = {item["node"] for item in graph.neighbors(consumer, etype="calls")}
    assert targets == {execute}
    assert counts["cross_module_calls"] == 1


def test_stage17s_resolves_inherited_method_and_preserves_override(tmp_path):
    root = tmp_path / "inherited-methods"
    root.mkdir()
    (root / "mod.py").write_text(
        "class Base:\n"
        "    def run(self):\n"
        "        return 1\n"
        "class Child(Base):\n"
        "    def invoke(self):\n"
        "        return self.run()\n"
        "class Override(Base):\n"
        "    def run(self):\n"
        "        return 2\n"
        "    def invoke(self):\n"
        "        return self.run()\n"
    )
    store = VectorStore(str(tmp_path / "stage17s-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "stage17s-graph.db"))
    index_code_root(str(root), store, graph, version="stage17s-inherit")
    module = f"py:{root.name}/mod.py:stage17s-inherit"
    base_run = module + ":func:Base.run"
    child_invoke = module + ":func:Child.invoke"
    override_run = module + ":func:Override.run"
    override_invoke = module + ":func:Override.invoke"
    assert {row["node"] for row in graph.neighbors(child_invoke, etype="calls")} == {base_run}
    assert {row["node"] for row in graph.neighbors(override_invoke, etype="calls")} == {override_run}


def test_stage17s_rejects_conditional_callable_alias(tmp_path):
    root = tmp_path / "conditional-alias"
    root.mkdir()
    (root / "mod.py").write_text(
        "def implementation():\n"
        "    return 1\n"
        "def invoke(flag):\n"
        "    if flag:\n"
        "        selected = implementation\n"
        "    return selected()\n"
    )
    store = VectorStore(str(tmp_path / "conditional-alias-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "conditional-alias-graph.db"))
    counts = index_code_root(str(root), store, graph, version="stage17s-conditional")
    module = f"py:{root.name}/mod.py:stage17s-conditional"
    invoke = module + ":func:invoke"
    assert graph.neighbors(invoke, etype="calls") == []
    assert counts["unresolved_calls"] >= 1


def test_stage17s_rejects_ambiguous_multiple_inheritance_dispatch(tmp_path):
    root = tmp_path / "ambiguous-inheritance"
    root.mkdir()
    (root / "mod.py").write_text(
        "class Left:\n"
        "    def run(self):\n"
        "        return 1\n"
        "class Right:\n"
        "    def run(self):\n"
        "        return 2\n"
        "class Child(Left, Right):\n"
        "    def invoke(self):\n"
        "        return self.run()\n"
    )
    store = VectorStore(str(tmp_path / "ambiguous-inheritance-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "ambiguous-inheritance-graph.db"))
    counts = index_code_root(str(root), store, graph, version="stage17s-ambiguous")
    module = f"py:{root.name}/mod.py:stage17s-ambiguous"
    invoke = module + ":func:Child.invoke"
    assert graph.neighbors(invoke, etype="calls") == []
    assert counts["ambiguous_calls"] >= 1


def test_stage17t_resolves_constructor_typed_instance_method(tmp_path):
    root = tmp_path / "typed-instance"
    root.mkdir()
    (root / "mod.py").write_text(
        "class Service:\n"
        "    def execute(self):\n"
        "        return 1\n"
        "def invoke():\n"
        "    service = Service()\n"
        "    return service.execute()\n"
    )
    store = VectorStore(str(tmp_path / "stage17t-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "stage17t-graph.db"))
    index_code_root(str(root), store, graph, version="stage17t-instance")
    module = f"py:{root.name}/mod.py:stage17t-instance"
    invoke = module + ":func:invoke"
    execute = module + ":func:Service.execute"
    assert {row["node"] for row in graph.neighbors(invoke, etype="calls")} == {execute}


def test_stage17t_resolves_self_attribute_constructor_type(tmp_path):
    root = tmp_path / "self-attribute-type"
    root.mkdir()
    (root / "mod.py").write_text(
        "class Worker:\n"
        "    def run(self):\n"
        "        return 1\n"
        "class Controller:\n"
        "    def __init__(self):\n"
        "        self.worker = Worker()\n"
        "    def invoke(self):\n"
        "        return self.worker.run()\n"
    )
    store = VectorStore(str(tmp_path / "stage17t-self-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "stage17t-self-graph.db"))
    index_code_root(str(root), store, graph, version="stage17t-self")
    module = f"py:{root.name}/mod.py:stage17t-self"
    invoke = module + ":func:Controller.invoke"
    run = module + ":func:Worker.run"
    assert {row["node"] for row in graph.neighbors(invoke, etype="calls")} == {run}


def test_stage17t_uses_annotation_and_rejects_conflicting_reassignment(tmp_path):
    root = tmp_path / "annotated-instance"
    root.mkdir()
    (root / "mod.py").write_text(
        "class First:\n"
        "    def run(self):\n"
        "        return 1\n"
        "class Second:\n"
        "    def run(self):\n"
        "        return 2\n"
        "def annotated(service: First):\n"
        "    return service.run()\n"
        "def conflicting():\n"
        "    service = First()\n"
        "    service = Second()\n"
        "    return service.run()\n"
    )
    store = VectorStore(str(tmp_path / "stage17t-annotated-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "stage17t-annotated-graph.db"))
    counts = index_code_root(str(root), store, graph, version="stage17t-annotated")
    module = f"py:{root.name}/mod.py:stage17t-annotated"
    annotated = module + ":func:annotated"
    first_run = module + ":func:First.run"
    conflicting = module + ":func:conflicting"
    assert {row["node"] for row in graph.neighbors(annotated, etype="calls")} == {first_run}
    assert graph.neighbors(conflicting, etype="calls") == []
    assert counts["unresolved_calls"] >= 1


def test_stage17u_resolves_direct_factory_return_method_call(tmp_path):
    root = tmp_path / "factory-return"
    root.mkdir()
    (root / "mod.py").write_text(
        "class Service:\n"
        "    def execute(self):\n"
        "        return 1\n"
        "def create_service():\n"
        "    return Service()\n"
        "def invoke():\n"
        "    return create_service().execute()\n"
    )
    store = VectorStore(str(tmp_path / "stage17u-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "stage17u-graph.db"))
    counts = index_code_root(str(root), store, graph, version="stage17u-direct")
    module = f"py:{root.name}/mod.py:stage17u-direct"
    invoke = module + ":func:invoke"
    execute = module + ":func:Service.execute"
    assert {row["node"] for row in graph.neighbors(invoke, etype="calls")} == {execute}
    assert counts["factory_return_resolutions"] == 1


def test_stage17u_resolves_unique_annotated_factory_return(tmp_path):
    root = tmp_path / "annotated-factory-return"
    root.mkdir()
    (root / "mod.py").write_text(
        "class Service:\n"
        "    def execute(self):\n"
        "        return 1\n"
        "def create_service() -> Service:\n"
        "    raise NotImplementedError\n"
        "def invoke():\n"
        "    return create_service().execute()\n"
    )
    store = VectorStore(str(tmp_path / "stage17u-annotated-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "stage17u-annotated-graph.db"))
    counts = index_code_root(str(root), store, graph, version="stage17u-annotated")
    module = f"py:{root.name}/mod.py:stage17u-annotated"
    invoke = module + ":func:invoke"
    execute = module + ":func:Service.execute"
    assert {row["node"] for row in graph.neighbors(invoke, etype="calls")} == {execute}
    assert counts["factory_return_resolutions"] == 1


def test_stage17u_rejects_ambiguous_and_conditional_factory_returns(tmp_path):
    root = tmp_path / "ambiguous-factory-return"
    root.mkdir()
    (root / "mod.py").write_text(
        "class First:\n"
        "    def execute(self):\n"
        "        return 1\n"
        "class Second:\n"
        "    def execute(self):\n"
        "        return 2\n"
        "def ambiguous(flag):\n"
        "    if flag:\n"
        "        return First()\n"
        "    return Second()\n"
        "def conditional(flag):\n"
        "    if flag:\n"
        "        return First()\n"
        "def invoke_ambiguous():\n"
        "    return ambiguous(True).execute()\n"
        "def invoke_conditional():\n"
        "    return conditional(True).execute()\n"
    )
    store = VectorStore(str(tmp_path / "stage17u-ambiguous-vector.db"))
    graph = KnowledgeGraph(str(tmp_path / "stage17u-ambiguous-graph.db"))
    counts = index_code_root(str(root), store, graph, version="stage17u-ambiguous")
    module = f"py:{root.name}/mod.py:stage17u-ambiguous"
    ambiguous_targets = {
        row["node"] for row in graph.neighbors(
            module + ":func:invoke_ambiguous", etype="calls")
    }
    conditional_targets = {
        row["node"] for row in graph.neighbors(
            module + ":func:invoke_conditional", etype="calls")
    }
    assert module + ":func:First.execute" not in ambiguous_targets
    assert module + ":func:Second.execute" not in ambiguous_targets
    assert module + ":func:First.execute" not in conditional_targets
    assert counts["ambiguous_factory_returns"] >= 1
    assert counts["factory_return_resolutions"] == 0
