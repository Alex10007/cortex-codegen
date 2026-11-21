import json

from cortex.memory.engine import MemoryEngine
from cortex.monitoring.audit import AuditLog


def test_memory_roundtrip(tmp_path):
    mem = MemoryEngine(str(tmp_path / "mem.db"))
    mem.record("proj-a", "outcome", {"ok": True}, verified=True)
    mem.record("proj-a", "failure", {"err": "boom"}, verified=False)
    mem.put_fact("proj-a", "arch", "monolith", verified=True)
    ev = mem.events("proj-a")
    assert len(ev) == 2
    assert mem.verified_outcomes("proj-a")[0]["payload"]["ok"] is True
    assert mem.get_fact("proj-a", "arch") == "monolith"
    mem.put_fact("proj-a", "arch", "modular", verified=True)
    assert mem.get_fact("proj-a", "arch") == "modular"


def test_audit_append_only(tmp_path):
    log = AuditLog(str(tmp_path / "audit.jsonl"))
    log.record("op1", {"n": 1})
    log.record("op2", {"n": 2})
    entries = log.tail(10)
    assert [e["event"] for e in entries] == ["op1", "op2"]
    assert all("ts" in e for e in entries)
    raw = (tmp_path / "audit.jsonl").read_text().strip().splitlines()
    assert all(json.loads(l) for l in raw)
