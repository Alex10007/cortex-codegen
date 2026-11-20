from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from cortex.api.app import create_app
from cortex.api.rate_limit import SQLiteRateLimiter


class _FakeAgent:
    hardware = SimpleNamespace(cpu_cores=4)

    def generate_with_metadata(self, **kwargs):
        return {"text": "ok"}


def test_rate_limit_is_shared_between_limiter_instances(tmp_path):
    database = tmp_path / "shared.sqlite3"
    first = SQLiteRateLimiter(database)
    second = SQLiteRateLimiter(database)

    assert first.consume("principal-a", 2)[0] is True
    assert second.consume("principal-a", 2)[0] is True
    allowed, retry_after = first.consume("principal-a", 2)
    assert allowed is False
    assert retry_after >= 1


def test_concurrent_consumption_never_exceeds_limit(tmp_path):
    database = tmp_path / "concurrent.sqlite3"
    limiters = [SQLiteRateLimiter(database) for _ in range(4)]

    def consume(index):
        return limiters[index % len(limiters)].consume("same-principal", 5)[0]

    with ThreadPoolExecutor(max_workers=12) as pool:
        outcomes = list(pool.map(consume, range(36)))

    assert sum(outcomes) == 5


def test_api_authenticated_requests_are_rate_limited_and_health_is_public(tmp_path):
    app = create_app(
        agent=_FakeAgent(), api_token="secret-token",
        rate_limit_db=str(tmp_path / "api.sqlite3"),
        authenticated_requests_per_minute=2)
    client = TestClient(app, headers={"Authorization": "Bearer secret-token"})

    assert client.get("/health").status_code == 200
    assert client.get("/system/profile").status_code == 200
    assert client.get("/system/profile").status_code == 200
    limited = client.get("/system/profile")
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) >= 1


def test_invalid_authentication_attempts_are_rate_limited(tmp_path):
    app = create_app(
        agent=_FakeAgent(), api_token="secret-token",
        rate_limit_db=str(tmp_path / "auth.sqlite3"),
        auth_failures_per_minute=2)
    client = TestClient(app, headers={"Authorization": "Bearer wrong"})

    assert client.get("/system/profile").status_code == 401
    assert client.get("/system/profile").status_code == 401
    limited = client.get("/system/profile")
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) >= 1


def test_expensive_endpoints_have_a_separate_lower_limit(tmp_path):
    app = create_app(
        agent=_FakeAgent(), api_token="secret-token",
        rate_limit_db=str(tmp_path / "expensive.sqlite3"),
        authenticated_requests_per_minute=10,
        expensive_requests_per_minute=1)
    client = TestClient(app, headers={"Authorization": "Bearer secret-token"})

    payload = {"prompt": "hello", "max_new_tokens": 1}
    assert client.post("/generate", json=payload).status_code == 200
    limited = client.post("/generate", json=payload)
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) >= 1


def test_rate_limit_database_does_not_store_raw_bucket_identity(tmp_path):
    import sqlite3

    database = Path(tmp_path / "privacy.sqlite3")
    limiter = SQLiteRateLimiter(database)
    limiter.consume("authenticated:very-secret-token", 3)
    with sqlite3.connect(database) as connection:
        stored = connection.execute(
            "SELECT bucket_key FROM rate_limit_events").fetchone()[0]
    assert stored != "authenticated:very-secret-token"
    assert len(stored) == 64
