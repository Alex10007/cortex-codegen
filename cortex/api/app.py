"""Authenticated API surface for the Fibonera Cortex engines.

Only /health is public. Every other endpoint requires a configured bearer
token. Filesystem operations are restricted to explicitly configured roots,
and API code execution requires the full bubblewrap sandbox.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
from pathlib import Path
from typing import Any
from functools import lru_cache

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from cortex.api.rate_limit import SQLiteRateLimiter


class SandboxRequest(BaseModel):
    code: str = Field(..., max_length=200_000)
    timeout_s: float = Field(default=15.0, ge=0.1, le=20.0)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10_000)
    k: int = Field(default=5, ge=1, le=100)


class RepoRequest(BaseModel):
    repo_path: str = Field(..., min_length=1, max_length=4096)


class PipelineRequest(RepoRequest):
    objective: str = Field(..., min_length=1, max_length=20_000)
    authorize_destructive: bool = False


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=20_000)
    project: str | None = Field(default=None, max_length=256)
    max_new_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class EvalRequest(BaseModel):
    candidate: str | None = Field(default=None, max_length=200_000)
    reference: bool = False


def _resolve_allowed_repo_path(repo_path: str) -> str:
    configured = os.environ.get("CORTEX_ALLOWED_REPO_ROOTS", "")
    roots = [Path(item).expanduser().resolve()
             for item in configured.split(os.pathsep) if item.strip()]
    if not roots:
        raise HTTPException(
            status_code=503,
            detail="repository operations disabled: configure "
                   "CORTEX_ALLOWED_REPO_ROOTS")
    try:
        candidate = Path(repo_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="repo_path does not exist")
    if not candidate.is_dir():
        raise HTTPException(status_code=400, detail="repo_path is not a directory")
    if not any(candidate == root or root in candidate.parents for root in roots):
        raise HTTPException(status_code=403, detail="repo_path is outside allowed roots")
    return str(candidate)


@lru_cache(maxsize=1)
def _cached_sandbox_qualification() -> dict[str, Any]:
    from cortex.execution.qualification import qualify_sandbox

    return qualify_sandbox()


def _require_full_sandbox() -> None:
    from cortex.execution.sandbox import _supports_bwrap

    if not _supports_bwrap():
        raise HTTPException(
            status_code=503,
            detail="code execution disabled: working bubblewrap user/pid/"
                   "network namespace isolation is unavailable")
    try:
        report = _cached_sandbox_qualification()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"sandbox behavioral qualification could not complete: {type(exc).__name__}") from exc
    if not report.get("passed"):
        failed = [name for name, status in report.get("checks", {}).items()
                  if status != "PASS"]
        raise HTTPException(
            status_code=503,
            detail="sandbox behavioral qualification failed: " + ", ".join(failed))

def create_app(agent=None, api_token: str | None = None, *,
               rate_limit_db: str | None = None,
               authenticated_requests_per_minute: int = 120,
               auth_failures_per_minute: int = 20,
               expensive_requests_per_minute: int = 20) -> FastAPI:
    from cortex.core import CortexAgent

    agent = agent or CortexAgent()
    configured_token = api_token or os.environ.get("CORTEX_API_TOKEN")
    database = (rate_limit_db or os.environ.get("CORTEX_RATE_LIMIT_DB")
                or ".cortex-state/api-rate-limits.sqlite3")
    limiter = SQLiteRateLimiter(database)
    app = FastAPI(title="Fibonera Cortex Codegen API", version="0.2.0")

    def consume_or_reject(bucket: str, limit: int) -> None:
        try:
            allowed, retry_after = limiter.consume(bucket, limit, 60)
        except (sqlite3.Error, OSError) as exc:
            # Fail closed: a broken limiter must not silently disable controls.
            raise HTTPException(
                status_code=503,
                detail="API rate-limit storage is unavailable") from exc
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="rate limit exceeded; retry later",
                headers={"Retry-After": str(retry_after)})

    def require_auth(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> None:
        if not configured_token:
            raise HTTPException(
                status_code=503,
                detail="API is locked: configure CORTEX_API_TOKEN")
        expected = f"Bearer {configured_token}"
        if authorization is None or not hmac.compare_digest(
                authorization, expected):
            peer = request.client.host if request.client else "unknown"
            consume_or_reject("auth-failure:" + peer, auth_failures_per_minute)
            raise HTTPException(status_code=401, detail="invalid bearer token")

        principal = hashlib.sha256(
            configured_token.encode("utf-8")).hexdigest()
        consume_or_reject("authenticated:" + principal,
                          authenticated_requests_per_minute)
        request.state.rate_limit_principal = principal

    def rate_limit_expensive(request: Request) -> None:
        principal = getattr(request.state, "rate_limit_principal", None)
        if not principal:
            raise HTTPException(status_code=401, detail="authentication required")
        consume_or_reject("expensive:" + principal,
                          expensive_requests_per_minute)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "fibonera-cortex"}

    @app.get("/system/profile", dependencies=[Depends(require_auth)])
    def system_profile() -> dict[str, Any]:
        return agent.hardware.__dict__

    @app.get("/system/services", dependencies=[Depends(require_auth)])
    def services() -> list[dict[str, Any]]:
        return agent.service_reports()

    @app.post("/sandbox/run",
              dependencies=[Depends(require_auth), Depends(rate_limit_expensive)])
    def sandbox_run(req: SandboxRequest) -> dict[str, Any]:
        from cortex.execution.sandbox import NetworkIsolationUnavailable

        _require_full_sandbox()
        try:
            return agent.sandbox_run(
                req.code, timeout_s=req.timeout_s, require_full_isolation=True)
        except NetworkIsolationUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    @app.post("/knowledge/query", dependencies=[Depends(require_auth)])
    def knowledge_query(req: QueryRequest) -> list[dict[str, Any]]:
        return agent.knowledge_query(req.query, k=req.k)

    @app.post("/generate",
              dependencies=[Depends(require_auth), Depends(rate_limit_expensive)])
    def generate(req: GenerateRequest) -> dict[str, Any]:
        try:
            output = agent.generate_with_metadata(
                req.prompt, max_new_tokens=req.max_new_tokens,
                temperature=req.temperature, project=req.project)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=413, detail=str(exc))
        return output

    @app.get("/knowledge/capacity", dependencies=[Depends(require_auth)])
    def knowledge_capacity() -> dict[str, Any]:
        return agent.knowledge_capacity()

    @app.post("/repo/analyze",
              dependencies=[Depends(require_auth), Depends(rate_limit_expensive)])
    def repo_analyze(req: RepoRequest) -> dict[str, Any]:
        return agent.analyze_repo(_resolve_allowed_repo_path(req.repo_path))

    @app.post("/security/scan",
              dependencies=[Depends(require_auth), Depends(rate_limit_expensive)])
    def security_scan(req: RepoRequest) -> dict[str, Any]:
        return agent.security_scan(_resolve_allowed_repo_path(req.repo_path))

    @app.post("/pipeline/run",
              dependencies=[Depends(require_auth), Depends(rate_limit_expensive)])
    def pipeline_run(req: PipelineRequest) -> dict[str, Any]:
        _require_full_sandbox()
        return agent.run_pipeline(
            req.objective, _resolve_allowed_repo_path(req.repo_path),
            authorize_destructive=req.authorize_destructive)

    @app.post("/eval/run",
              dependencies=[Depends(require_auth), Depends(rate_limit_expensive)])
    def eval_run(req: EvalRequest) -> dict[str, Any]:
        if not req.reference and req.candidate is None:
            raise HTTPException(
                status_code=422,
                detail="candidate is required unless reference=true")
        _require_full_sandbox()
        return agent.eval_run(candidate=req.candidate, reference=req.reference)

    return app


app = None


def get_app() -> FastAPI:
    global app
    if app is None:
        app = create_app()
    return app
