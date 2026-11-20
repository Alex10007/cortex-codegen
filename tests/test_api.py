import pytest

pytest.importorskip("fastapi")

from cortex.api.app import create_app
from cortex.core import CortexAgent


@pytest.fixture(scope="module")
def app():
    agent = CortexAgent(state_dir=None)  # default .cortex-state in repo dir
    return create_app(agent, api_token="test-token")


def test_health(app):
    client = _client(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_profile_and_services(app):
    client = _client(app)
    prof = client.get("/system/profile").json()
    assert prof["cpu_cores"] >= 1
    services = client.get("/system/services").json()
    engines = {s["engine"] for s in services}
    assert {"HardwareProfiler", "ModelCore", "KnowledgeEngine"} <= engines


def test_sandbox_and_knowledge_endpoints(app):
    client = _client(app)
    r = client.post("/sandbox/run", json={"code": "print('api-sandbox-ok')"})
    if r.status_code == 503:
        assert "bubblewrap" in r.json()["detail"].lower()
    else:
        assert r.status_code == 200
        assert r.json()["limits"]["filesystem_isolated"] is True
        assert "api-sandbox-ok" in r.json()["stdout"]

    r = client.post("/knowledge/query", json={"query": "anything", "k": 3})
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def _client(app):
    from fastapi.testclient import TestClient

    return TestClient(app, headers={"Authorization": "Bearer test-token"})



def test_protected_endpoint_rejects_missing_token(app):
    from fastapi.testclient import TestClient

    client = TestClient(app)
    r = client.get("/system/profile")
    assert r.status_code == 401


def test_eval_requires_candidate_unless_reference(app):
    client = _client(app)
    r = client.post("/eval/run", json={"reference": False})
    assert r.status_code == 422


def test_repo_endpoint_requires_configured_allowlist(app, monkeypatch):
    client = _client(app)
    monkeypatch.delenv("CORTEX_ALLOWED_REPO_ROOTS", raising=False)
    r = client.post("/repo/analyze", json={"repo_path": "."})
    assert r.status_code == 503
