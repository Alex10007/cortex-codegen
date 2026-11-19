"""API Gateway: FastAPI surface over the Cortex engines."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class SandboxRequest(BaseModel):
    code: str = Field(..., max_length=1_000_000)
    timeout_s: float = 15.0


class QueryRequest(BaseModel):
    query: str
    k: int = 5


class PipelineRequest(BaseModel):
    objective: str
    repo_path: str
    authorize_destructive: bool = False


def create_app(agent=None) -> FastAPI:
    from cortex.core import CortexAgent

    agent = agent or CortexAgent()
    app = FastAPI(title="Cortex Codegen API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "cortex-codegen"}

    @app.get("/system/profile")
    def system_profile() -> dict[str, Any]:
        return agent.hardware.__dict__

    @app.get("/system/services")
    def services() -> list[dict[str, Any]]:
        return agent.service_reports()

    @app.post("/sandbox/run")
    def sandbox_run(req: SandboxRequest) -> dict[str, Any]:
        return agent.sandbox_run(req.code, timeout_s=req.timeout_s)

    @app.post("/knowledge/query")
    def knowledge_query(req: QueryRequest) -> list[dict[str, Any]]:
        return agent.knowledge_query(req.query, k=req.k)

    @app.get("/knowledge/capacity")
    def knowledge_capacity() -> dict[str, Any]:
        return agent.knowledge_capacity()

    @app.post("/repo/analyze")
    def repo_analyze(repo_path: str) -> dict[str, Any]:
        import os

        if not os.path.isdir(repo_path):
            raise HTTPException(status_code=400, detail="repo_path is not a directory")
        return agent.analyze_repo(repo_path)

    @app.post("/security/scan")
    def security_scan(repo_path: str) -> dict[str, Any]:
        return agent.security_scan(repo_path)

    @app.post("/pipeline/run")
    def pipeline_run(req: PipelineRequest) -> dict[str, Any]:
        return agent.run_pipeline(req.objective, req.repo_path,
                                  authorize_destructive=req.authorize_destructive)

    @app.post("/eval/run")
    def eval_run(reference: bool = False) -> dict[str, Any]:
        return agent.eval_run(reference=reference)

    return app


app = None


def get_app() -> FastAPI:
    global app
    if app is None:
        app = create_app()
    return app
