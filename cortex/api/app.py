"""Authenticated API surface for the Fibonera Cortex engines.

Only /health is public. Every other endpoint requires a configured bearer
token. Filesystem operations are restricted to explicitly configured roots,
and API code execution requires the full bubblewrap sandbox.
"""
from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


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


def _require_full_sandbox() -> None:
    from cortex.execution.sandbox import _supports_bwrap

    if not _supports_bwrap():
        raise HTTPException(
            status_code=503,
            detail="code execution disabled: working bubblewrap user/pid/"
                   "network namespace isolation is unavailable")


def create_app(agent=None, api_token: str | None = None) -> FastAPI:
    from cortex.core import CortexAgent

    agent = agent or CortexAgent()
    configured_token = api_token or os.environ.get("CORTEX_API_TOKEN")
    app = FastAPI(title="Fibonera Cortex Codegen API", version="0.2.0")

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        if not configured_token:
            raise HTTPException(
                status_code=503,
                detail="API is locked: configure CORTEX_API_TOKEN")
        expected = f"Bearer {configured_token}"
        if authorization is None or not hmac.compare_digest(
                authorization, expected):
            raise HTTPException(status_code=401, detail="invalid bearer token")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "fibonera-cortex"}

    @app.get("/system/profile", dependencies=[Depends(require_auth)])
    def system_profile() -> dict[str, Any]:
        return agent.hardware.__dict__

    @app.get("/system/services", dependencies=[Depends(require_auth)])
    def services() -> list[dict[str, Any]]:
        return agent.service_reports()

    @app.post("/sandbox/run", dependencies=[Depends(require_auth)])
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

    @app.post("/generate", dependencies=[Depends(require_auth)])
    def generate(req: GenerateRequest) -> dict[str, Any]:
        try:
            output = agent.generate(
                req.prompt, max_new_tokens=req.max_new_tokens,
                temperature=req.temperature, project=req.project)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=413, detail=str(exc))
        return {"text": output, "project_context_used": bool(req.project)}

    @app.get("/knowledge/capacity", dependencies=[Depends(require_auth)])
    def knowledge_capacity() -> dict[str, Any]:
        return agent.knowledge_capacity()

    @app.post("/repo/analyze", dependencies=[Depends(require_auth)])
    def repo_analyze(req: RepoRequest) -> dict[str, Any]:
        return agent.analyze_repo(_resolve_allowed_repo_path(req.repo_path))

    @app.post("/security/scan", dependencies=[Depends(require_auth)])
    def security_scan(req: RepoRequest) -> dict[str, Any]:
        return agent.security_scan(_resolve_allowed_repo_path(req.repo_path))

    @app.post("/pipeline/run", dependencies=[Depends(require_auth)])
    def pipeline_run(req: PipelineRequest) -> dict[str, Any]:
        _require_full_sandbox()
        return agent.run_pipeline(
            req.objective, _resolve_allowed_repo_path(req.repo_path),
            authorize_destructive=req.authorize_destructive)

    @app.post("/eval/run", dependencies=[Depends(require_auth)])
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
