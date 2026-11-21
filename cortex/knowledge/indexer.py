"""Repository/code indexer: builds provenance-tracked knowledge items and a
versioned knowledge graph from real source trees (AST-based for Python)."""
from __future__ import annotations

import ast
import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cortex.interfaces import KnowledgeItem, Provenance
from cortex.knowledge.graph import KnowledgeGraph
from cortex.knowledge.vector_store import VectorStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _provenance(path: Path, version: str, license_tag: str,
                collector: str) -> Provenance:
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        digest = "unavailable"
    return Provenance(
        source_uri=f"file://{path}",
        license=license_tag,
        version=version,
        content_sha256=digest,
        collected_at=_now(),
        collector=collector,
    )


def _parse_python(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return None


def index_code_root(root: str, store: VectorStore, graph: KnowledgeGraph,
                    language: str = "python",
                    version: str | None = None,
                    license_tag: str = "project-local",
                    max_file_bytes: int = 300_000) -> dict[str, int]:
    """Index a real source tree into the vector store and knowledge graph."""
    version = version or platform.python_version()
    root_path = Path(root).resolve()
    counts = {"files": 0, "modules": 0, "classes": 0, "functions": 0, "docs": 0,
              "edges": 0, "skipped": 0}

    py_files = [p for p in root_path.rglob("*.py")
                if p.stat().st_size <= max_file_bytes]
    for p in py_files:
        tree = _parse_python(p)
        if tree is None:
            counts["skipped"] += 1
            continue
        rel = p.relative_to(root_path).as_posix()
        counts["files"] += 1
        mid = f"py:{root_path.name}/{rel}:{version}"
        prov = _provenance(p, version, license_tag, "cortex.knowledge.indexer")

        doc = ast.get_docstring(tree) or ""
        store.add(KnowledgeItem(
            item_id=mid, kind="module", language=language, text=doc or rel,
            provenance=prov, tags={"path": rel, "hash": prov.content_sha256},
        ), defer_commit=True)
        graph.upsert_node(mid, "module", rel, prov)
        counts["modules"] += 1
        if doc:
            store.add(KnowledgeItem(
                item_id=mid + ":doc", kind="doc", language=language, text=doc,
                provenance=prov, tags={"path": rel},
            ), defer_commit=True)
            counts["docs"] += 1

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                cid = mid + ":class:" + node.name
                cdoc = ast.get_docstring(node) or node.name
                store.add(KnowledgeItem(
                    item_id=cid, kind="class", language=language, text=cdoc,
                    provenance=prov, tags={"path": rel, "name": node.name},
                ), defer_commit=True)
                graph.upsert_node(cid, "class", f"{rel}::{node.name}", prov)
                graph.upsert_edge(mid, cid, "defines", version)
                counts["classes"] += 1
                counts["edges"] += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fid = mid + ":func:" + node.name
                fdoc = ast.get_docstring(node) or node.name
                args = [a.arg for a in node.args.args]
                store.add(KnowledgeItem(
                    item_id=fid, kind="function", language=language, text=fdoc,
                    provenance=prov,
                    tags={"path": rel, "name": node.name, "args": args},
                ), defer_commit=True)
                graph.upsert_node(fid, "function", f"{rel}::{node.name}", prov)
                graph.upsert_edge(mid, fid, "defines", version)
                counts["functions"] += 1
                counts["edges"] += 1
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    graph.upsert_edge(mid, "pymod:" + alias.name, "imports", version)
                    graph.upsert_node("pymod:" + alias.name, "external_module",
                                      alias.name, prov)
                    counts["edges"] += 1
            elif isinstance(node, ast.ImportFrom) and node.module:
                graph.upsert_edge(mid, "pymod:" + node.module, "imports", version)
                graph.upsert_node("pymod:" + node.module, "external_module",
                                  node.module, prov)
                counts["edges"] += 1

    store.flush()
    graph.commit()
    return counts
