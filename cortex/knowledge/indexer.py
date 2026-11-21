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
              "edges": 0, "skipped": 0, "local_imports": 0, "calls": 0,
              "unresolved_calls": 0, "inherits": 0}

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

        class ScopedIndexer(ast.NodeVisitor):
            def __init__(self):
                self.scope: list[str] = []
                self.parents: list[str] = [mid]

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                qual = ".".join([*self.scope, node.name])
                cid = mid + ":class:" + qual
                cdoc = ast.get_docstring(node) or qual
                store.add(KnowledgeItem(
                    item_id=cid, kind="class", language=language, text=cdoc,
                    provenance=prov,
                    tags={"path": rel, "name": qual,
                          "short_name": node.name, "scope": list(self.scope)},
                ), defer_commit=True)
                graph.upsert_node(cid, "class", f"{rel}::{qual}", prov)
                graph.upsert_edge(self.parents[-1], cid, "defines", version)
                counts["classes"] += 1
                counts["edges"] += 1
                self.scope.append(node.name)
                self.parents.append(cid)
                self.generic_visit(node)
                self.parents.pop()
                self.scope.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self._visit_function(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self._visit_function(node)

            def _visit_function(self, node) -> None:
                qual = ".".join([*self.scope, node.name])
                fid = mid + ":func:" + qual
                fdoc = ast.get_docstring(node) or qual
                args = [a.arg for a in node.args.args]
                store.add(KnowledgeItem(
                    item_id=fid, kind="function", language=language, text=fdoc,
                    provenance=prov,
                    tags={"path": rel, "name": qual,
                          "short_name": node.name, "scope": list(self.scope),
                          "args": args},
                ), defer_commit=True)
                graph.upsert_node(fid, "function", f"{rel}::{qual}", prov)
                graph.upsert_edge(self.parents[-1], fid, "defines", version)
                counts["functions"] += 1
                counts["edges"] += 1
                self.scope.append(node.name)
                self.parents.append(fid)
                self.generic_visit(node)
                self.parents.pop()
                self.scope.pop()

            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    graph.upsert_edge(mid, "pymod:" + alias.name, "imports", version)
                    graph.upsert_node("pymod:" + alias.name, "external_module",
                                      alias.name, prov)
                    counts["edges"] += 1

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                if node.module:
                    graph.upsert_edge(mid, "pymod:" + node.module, "imports", version)
                    graph.upsert_node("pymod:" + node.module, "external_module",
                                      node.module, prov)
                    counts["edges"] += 1

        ScopedIndexer().visit(tree)


    # Enrich the canonical knowledge graph with repository-local imports,
    # uniquely-resolved calls, and class inheritance. No parallel index is made.
    def module_key(relative_path: str) -> str:
        parts = list(Path(relative_path).with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)

    module_ids: dict[str, str] = {}
    for source_path in py_files:
        relative = source_path.relative_to(root_path).as_posix()
        module_ids[module_key(relative)] = (
            f"py:{root_path.name}/{relative}:{version}")

    for source_path in py_files:
        relative = source_path.relative_to(root_path).as_posix()
        tree = _parse_python(source_path)
        if tree is None:
            continue
        source_module = module_key(relative)
        source_id = module_ids[source_module]
        package_parts = (source_module.split(".") if
                         relative.endswith("/__init__.py") or relative == "__init__.py"
                         else source_module.split(".")[:-1])
        for node in ast.walk(tree):
            candidates: list[str] = []
            if isinstance(node, ast.Import):
                candidates.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    trim = max(0, node.level - 1)
                    base_parts = package_parts[:max(0, len(package_parts) - trim)]
                else:
                    base_parts = []
                if node.module:
                    base_parts.extend(node.module.split("."))
                    candidates.append(".".join(base_parts))
                else:
                    candidates.extend(
                        ".".join([*base_parts, alias.name])
                        for alias in node.names if alias.name != "*")
            for candidate in candidates:
                target_id = module_ids.get(candidate)
                if target_id and target_id != source_id:
                    graph.upsert_edge(source_id, target_id, "imports_local", version)
                    counts["local_imports"] += 1

    symbol_index: dict[str, list[str]] = {}
    pending_calls: list[tuple[str, str]] = []
    pending_inheritance: list[tuple[str, str]] = []
    for source_path in py_files:
        relative = source_path.relative_to(root_path).as_posix()
        tree = _parse_python(source_path)
        if tree is None:
            continue
        source_id = module_ids[module_key(relative)]

        class SymbolGraphVisitor(ast.NodeVisitor):
            def __init__(self):
                self.scope_names: list[str] = []
                self.scope_ids = [source_id]
                self.symbol_ids: dict[str, list[str]] = {}
                self.calls: list[tuple[str, str]] = []
                self.inheritance: list[tuple[str, str]] = []

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                qualified = ".".join([*self.scope_names, node.name])
                symbol_id = source_id + ":class:" + qualified
                self.symbol_ids.setdefault(node.name, []).append(symbol_id)
                for base in node.bases:
                    base_name = (base.id if isinstance(base, ast.Name) else
                                 base.attr if isinstance(base, ast.Attribute) else None)
                    if base_name:
                        self.inheritance.append((symbol_id, base_name))
                self.scope_names.append(node.name)
                self.scope_ids.append(symbol_id)
                self.generic_visit(node)
                self.scope_ids.pop()
                self.scope_names.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self._visit_function(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self._visit_function(node)

            def _visit_function(self, node) -> None:
                qualified = ".".join([*self.scope_names, node.name])
                symbol_id = source_id + ":func:" + qualified
                self.symbol_ids.setdefault(node.name, []).append(symbol_id)
                self.scope_names.append(node.name)
                self.scope_ids.append(symbol_id)
                self.generic_visit(node)
                self.scope_ids.pop()
                self.scope_names.pop()

            def visit_Call(self, node: ast.Call) -> None:
                target_name = (node.func.id if isinstance(node.func, ast.Name)
                               else node.func.attr if isinstance(node.func, ast.Attribute)
                               else None)
                if target_name:
                    self.calls.append((self.scope_ids[-1], target_name))
                self.generic_visit(node)

        visitor = SymbolGraphVisitor()
        visitor.visit(tree)
        for name, ids in visitor.symbol_ids.items():
            symbol_index.setdefault(name, []).extend(ids)
        pending_calls.extend(visitor.calls)
        pending_inheritance.extend(visitor.inheritance)

    for caller_id, target_name in pending_calls:
        targets = sorted(set(symbol_index.get(target_name, [])))
        if len(targets) == 1 and targets[0] != caller_id:
            graph.upsert_edge(caller_id, targets[0], "calls", version)
            counts["calls"] += 1
        else:
            counts["unresolved_calls"] += 1

    for child_id, base_name in pending_inheritance:
        targets = sorted(set(symbol_index.get(base_name, [])))
        if len(targets) == 1 and targets[0] != child_id:
            graph.upsert_edge(child_id, targets[0], "inherits", version)
            counts["inherits"] += 1

    store.flush()
    graph.commit()
    return counts
