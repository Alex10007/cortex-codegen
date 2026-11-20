"""Tests for behavioral sandbox qualification and fail-closed API gating."""
import json
from types import SimpleNamespace

import pytest

from cortex.execution import qualification


def test_qualification_reports_each_behavioral_boundary(monkeypatch):
    observed = {
        "workspace_writable": True,
        "host_path_inaccessible": True,
        "runtime_readonly": True,
        "proc_dev_available": True,
        "network_unreachable": True,
    }
    monkeypatch.setattr(qualification, "_supports_bwrap", lambda: True)
    monkeypatch.setattr(qualification, "run", lambda **kwargs: SimpleNamespace(
        ok=True, stdout=json.dumps(observed)))
    report = qualification.qualify_sandbox()
    assert report["passed"] is True
    assert set(report["checks"].values()) == {"PASS"}
    assert any("cgroup" in item for item in report["limitations"])


def test_qualification_fails_when_host_path_is_visible(monkeypatch):
    observed = {
        "workspace_writable": True, "host_path_inaccessible": False,
        "runtime_readonly": True, "proc_dev_available": True,
        "network_unreachable": True,
    }
    monkeypatch.setattr(qualification, "_supports_bwrap", lambda: True)
    monkeypatch.setattr(qualification, "run", lambda **kwargs: SimpleNamespace(
        ok=True, stdout=json.dumps(observed)))
    report = qualification.qualify_sandbox()
    assert report["passed"] is False
    assert report["checks"]["host_path_inaccessible"] == "FAIL"


def test_bwrap_presence_alone_is_not_qualification(monkeypatch):
    monkeypatch.setattr(qualification, "_supports_bwrap", lambda: False)
    report = qualification.qualify_sandbox()
    assert report["passed"] is False
    assert report["checks"]["bwrap_available"] == "FAIL"
    assert report["checks"]["isolated_execution"] == "NOT_RUN"


def test_probe_setup_failure_leaves_behavioral_checks_not_run(monkeypatch):
    from cortex.execution.sandbox import NetworkIsolationUnavailable
    monkeypatch.setattr(qualification, "_supports_bwrap", lambda: True)
    def fail(**kwargs):
        raise NetworkIsolationUnavailable("namespace setup denied")
    monkeypatch.setattr(qualification, "run", fail)
    report = qualification.qualify_sandbox()
    assert report["passed"] is False
    assert report["checks"]["isolated_execution"] == "FAIL"
    assert report["checks"]["workspace_writable"] == "NOT_RUN"


def test_api_gate_requires_behavioral_qualification(monkeypatch):
    from fastapi import HTTPException
    from cortex.api import app as api_module
    from cortex.execution import qualification, sandbox
    monkeypatch.setattr(sandbox, "_supports_bwrap", lambda: True)
    monkeypatch.setattr(qualification, "qualify_sandbox", lambda: {
        "passed": False, "checks": {"host_path_inaccessible": "FAIL"}})
    with pytest.raises(HTTPException) as exc:
        api_module._require_full_sandbox()
    assert exc.value.status_code == 503
    assert "qualification" in exc.value.detail.lower()
