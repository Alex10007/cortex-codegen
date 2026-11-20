"""CortexAgent: top-level facade wiring every engine behind one interface.

Model status and knowledge capacity are reported from real state (files on
disk, measured database statistics) — never asserted optimistically.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

_STATE_ROOT = os.environ.get("CORTEX_STATE_DIR", ".cortex-state")


class CortexAgent:
    def __init__(self, state_dir: Optional[str] = None,
                 tokenizer_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None):
        from cortex.hardware.profiler import profile_hardware
        from cortex.knowledge.graph import KnowledgeGraph
        from cortex.knowledge.vector_store import VectorStore
        from cortex.memory.engine import MemoryEngine
        from cortex.monitoring.audit import AuditLog
        from cortex.knowledge.learning import LearningEngine

        self.state_dir = Path(state_dir or _STATE_ROOT)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.audit = AuditLog(str(self.state_dir / "audit.jsonl"))

        self.hardware = profile_hardware()
        self.store = VectorStore(str(self.state_dir / "knowledge.db"))
        self.graph = KnowledgeGraph(str(self.state_dir / "knowledge-graph.db"))
        self.memory = MemoryEngine(str(self.state_dir / "memory.db"))
        self.learning = LearningEngine(self.store)

        self.tokenizer_path = tokenizer_path
        self.checkpoint_path = checkpoint_path
        self._model = None
        self._tokenizer = None

    # -- model -------------------------------------------------------------
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            if not self.tokenizer_path or not Path(self.tokenizer_path).exists():
                raise FileNotFoundError(
                    "tokenizer not trained yet; run scripts/build_and_verify.py")
            from cortex.tokenizer.cortex_tokenizer import load_tokenizer
            self._tokenizer = load_tokenizer(self.tokenizer_path)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            if not self.checkpoint_path or not Path(self.checkpoint_path).exists():
                raise FileNotFoundError(
                    "no trained checkpoint; run scripts/build_and_verify.py")
            import torch

            from cortex.model.configs import MODEL_STATUS, get_config
            from cortex.model.transformer import CortexTransformer
            ck = torch.load(self.checkpoint_path, map_location="cpu",
                            weights_only=True)
            cfg_name = ck["model"]
            model = CortexTransformer(get_config(cfg_name))
            model.load_state_dict(ck["model_state"])
            model.eval()
            self._model = model
            self.checkpoint_cfg = cfg_name
            _ = MODEL_STATUS
        return self._model

    def load_checkpoint(self, checkpoint_path: str) -> str:
        """Point the agent at a checkpoint and force a model reload."""
        self.checkpoint_path = checkpoint_path
        self._model = None
        return self.model.__class__.__name__

    def generate(self, prompt: str, max_new_tokens: int = 64,
                 temperature: float = 0.8) -> str:
        import torch

        ids = torch.tensor([self.tokenizer.encode(self.tokenizer.EOT + "\n" + prompt)],
                           dtype=torch.long)
        out = self.model.generate(ids, max_new_tokens=max_new_tokens,
                                  temperature=temperature,
                                  eot_id=self.tokenizer.eot_id)
        text = self.tokenizer.decode(out[0].tolist())
        return text

    # -- knowledge ---------------------------------------------------------
    def ingest_corpus(self, roots: list[str]) -> dict[str, int]:
        from cortex.knowledge.indexer import index_code_root

        totals = {"files": 0, "modules": 0, "classes": 0, "functions": 0,
                  "docs": 0, "edges": 0, "skipped": 0}
        for root in roots:
            counts = index_code_root(root, self.store, self.graph)
            for key in totals:
                totals[key] += counts.get(key, 0)
        self.audit.record("knowledge.ingest", {"roots": roots, "counts": totals})
        return totals

    def knowledge_query(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        return self.store.search(query, k=k)

    def knowledge_capacity(self, probe_n: int = 200, k: int = 5) -> dict[str, Any]:
        from cortex.knowledge.capacity import measure_capacity

        return measure_capacity(self.store, probe_n=probe_n, k=k,
                                seed=sum(ord(c) for c in "cortex-capacity"))

    # -- engineering -------------------------------------------------------
    def analyze_repo(self, repo_path: str) -> dict[str, Any]:
        from cortex.planning.engine import analyze_repository

        return analyze_repository(repo_path)

    def plan(self, objective: str, repo_path: str) -> dict[str, Any]:
        from cortex.planning.engine import analyze_repository, plan

        repo_map = analyze_repository(repo_path)
        return plan(objective, repo_path, repo_map).to_dict()

    def run_pipeline(self, objective: str, repo_path: str,
                     patcher=None, authorize_destructive: bool = False) -> dict[str, Any]:
        from cortex.pipeline.engineer import run_engineering_pipeline

        report = run_engineering_pipeline(
            objective, repo_path, patcher=patcher,
            authorize_destructive=authorize_destructive, audit=self.audit)
        self.memory.record(record_project(repo_path), "pipeline", {
            "objective": objective, "status": report["status"],
            "verified": report.get("verified", False),
        }, verified=bool(report.get("verified")))
        return report

    def security_scan(self, root: str) -> dict[str, Any]:
        from cortex.security.analyzer import analyze_tree, summarize

        findings = analyze_tree(root)
        return {"summary": summarize(findings),
                "findings": [f.__dict__ for f in findings]}

    def sandbox_run(self, code: str, timeout_s: float = 20.0,
                    require_full_isolation: bool = False) -> dict[str, Any]:
        from cortex.execution.sandbox import run

        result = run(code=code, timeout_s=timeout_s, audit=self.audit.record,
                     require_full_isolation=require_full_isolation)
        return result.__dict__

    def eval_run(self, candidate: Optional[str] = None,
                 reference: bool = False) -> dict[str, Any]:
        from cortex.evaluation.runner import evaluate

        return evaluate(candidate, reference=reference, audit=self.audit.record)

    # -- status -------------------------------------------------------------
    def service_reports(self) -> list[dict[str, Any]]:
        from cortex.model.configs import MODEL_STATUS

        reports: list[dict[str, Any]] = []
        reports.append({
            "engine": "HardwareProfiler", "status": "operational",
            "evidence": self.hardware.summary(),
        })
        tok_status = ("operational" if self.tokenizer_path and
                      Path(self.tokenizer_path).exists() else "not-built")
        reports.append({"engine": "Tokenizer", "status": tok_status,
                        "evidence": str(self.tokenizer_path)})
        model_status = MODEL_STATUS.get(
            self.checkpoint_cfg if getattr(self, "checkpoint_cfg", None) else "cortex-demo",
            {})
        ckpt_exists = self.checkpoint_path and Path(self.checkpoint_path).exists()
        reports.append({
            "engine": "ModelCore",
            "status": ("operational" if ckpt_exists else
                       "declared-only (2B-class) / demo-untrained"),
            "evidence": f"checkpoint={self.checkpoint_path} exists={bool(ckpt_exists)}",
        })
        if self.store.counts()["stored"]:
            reports.append({"engine": "KnowledgeEngine", "status": "operational",
                            "evidence": str(self.knowledge_capacity(probe_n=50))})
        else:
            reports.append({"engine": "KnowledgeEngine", "status": "empty",
                            "evidence": "no items ingested yet"})
        reports.append({"engine": "MemoryEngine", "status": "operational",
                        "evidence": str(self.state_dir / "memory.db")})
        reports.append({"engine": "Sandbox", "status": "operational",
                        "evidence": "unshare -rn supported on this kernel: "
                        + str(_unshare_supported())})
        return reports


def record_project(repo_path: str) -> str:
    return Path(repo_path).name


def _unshare_supported() -> bool:
    from cortex.execution.sandbox import _supports_unshare

    return _supports_unshare()
