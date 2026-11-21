"""Stage 17 tests for integrated metacognitive planning and hierarchical memory."""
import sqlite3

import pytest

from cortex.memory.engine import MemoryEngine
from cortex.planning.engine import assess_objective, plan


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
