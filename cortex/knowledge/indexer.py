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
              "unresolved_calls": 0, "ambiguous_calls": 0, "inherits": 0,
              "ambiguous_inheritance": 0, "type_references": 0,
              "ambiguous_type_references": 0}

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

    # Scope-aware Python symbol resolution. Only lexical declarations and
    # explicit self/cls member references are resolved; arbitrary same-name
    # matches elsewhere in the repository are never treated as evidence.
    symbol_definitions: dict[tuple[str, tuple[str, ...], str], list[tuple[str, str]]] = {}
    scope_bindings: dict[tuple[str, tuple[str, ...]], set[str]] = {}
    pending_calls: list[tuple[str, str, tuple[str, ...], str, str | None]] = []
    pending_inheritance: list[tuple[str, tuple[str, ...], str]] = []
    pending_type_refs: list[tuple[str, tuple[str, ...], str]] = []

    class _BindingCollector(ast.NodeVisitor):
        def __init__(self, root_node: ast.AST):
            self.root_node = root_node
            self.names: set[str] = set()
            self.global_names: set[str] = set()
            self.nonlocal_names: set[str] = set()

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                self.names.add(node.id)

        def visit_Global(self, node: ast.Global) -> None:
            self.global_names.update(node.names)

        def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
            self.nonlocal_names.update(node.names)

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                self.names.add(alias.asname or alias.name.split(".")[0])

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            for alias in node.names:
                if alias.name != "*":
                    self.names.add(alias.asname or alias.name)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is not self.root_node:
                self.names.add(node.name)
                return
            self._visit_function_body(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if node is not self.root_node:
                self.names.add(node.name)
                return
            self._visit_function_body(node)

        def _visit_function_body(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                self.names.add(arg.arg)
            if node.args.vararg:
                self.names.add(node.args.vararg.arg)
            if node.args.kwarg:
                self.names.add(node.args.kwarg.arg)
            for statement in node.body:
                self.visit(statement)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            if node is not self.root_node:
                self.names.add(node.name)
                return
            for statement in node.body:
                self.visit(statement)

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_ListComp(self, node: ast.ListComp) -> None:
            return

        visit_SetComp = visit_ListComp
        visit_DictComp = visit_ListComp
        visit_GeneratorExp = visit_ListComp

    def _bindings_for(node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
        collector = _BindingCollector(node)
        if isinstance(node, ast.Module):
            for statement in node.body:
                collector.visit(statement)
        else:
            collector.visit(node)
        return collector.names - collector.global_names - collector.nonlocal_names

    scope_global_names: dict[tuple[str, tuple[str, ...]], set[str]] = {}
    scope_nonlocal_names: dict[tuple[str, tuple[str, ...]], set[str]] = {}

    def _record_scope_declarations(source_id: str, scope: tuple[str, ...],
                                   node: ast.AST) -> None:
        collector = _BindingCollector(node)
        if isinstance(node, ast.Module):
            for statement in node.body:
                collector.visit(statement)
        else:
            collector.visit(node)
        scope_global_names[(source_id, scope)] = collector.global_names
        scope_nonlocal_names[(source_id, scope)] = collector.nonlocal_names

    for source_path in py_files:
        relative = source_path.relative_to(root_path).as_posix()
        tree = _parse_python(source_path)
        if tree is None:
            continue
        source_id = module_ids[module_key(relative)]
        scope_bindings[(source_id, ())] = _bindings_for(tree)
        _record_scope_declarations(source_id, (), tree)

        class ScopedSymbolVisitor(ast.NodeVisitor):
            def __init__(self):
                self.scope_names: list[str] = []
                self.scope_ids = [source_id]

            def _record(self, name: str, kind: str, qualified: str) -> str:
                symbol_id = source_id + ":" + kind + ":" + qualified
                key = (source_id, tuple(self.scope_names), name)
                symbol_definitions.setdefault(key, []).append((symbol_id, kind))
                return symbol_id

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                qualified = ".".join([*self.scope_names, node.name])
                class_id = self._record(node.name, "class", qualified)
                for base in node.bases:
                    base_name = (base.id if isinstance(base, ast.Name) else
                                 base.attr if isinstance(base, ast.Attribute) else None)
                    if base_name:
                        pending_inheritance.append(
                            (class_id, tuple(self.scope_names), base_name))
                # Decorators and base expressions execute in the enclosing scope.
                for expression in [*node.decorator_list, *node.bases]:
                    self.visit(expression)
                class_scope = tuple([*self.scope_names, node.name])
                scope_bindings[(source_id, class_scope)] = _bindings_for(node)
                _record_scope_declarations(source_id, class_scope, node)
                self.scope_names.append(node.name)
                self.scope_ids.append(class_id)
                for statement in node.body:
                    self.visit(statement)
                self.scope_ids.pop()
                self.scope_names.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self._visit_function(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self._visit_function(node)

            def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
                qualified = ".".join([*self.scope_names, node.name])
                function_id = self._record(node.name, "func", qualified)
                # Defaults, decorators and annotations are evaluated in the parent scope.
                for expression in [*node.decorator_list, *node.args.defaults,
                                   *[item for item in node.args.kw_defaults if item is not None]]:
                    self.visit(expression)
                arguments = [*node.args.posonlyargs, *node.args.args,
                             *node.args.kwonlyargs]
                if node.args.vararg:
                    arguments.append(node.args.vararg)
                if node.args.kwarg:
                    arguments.append(node.args.kwarg)
                for argument in arguments:
                    if argument.annotation:
                        self._record_type_names(argument.annotation, function_id)
                if node.returns:
                    self._record_type_names(node.returns, function_id)
                function_scope = tuple([*self.scope_names, node.name])
                scope_bindings[(source_id, function_scope)] = _bindings_for(node)
                _record_scope_declarations(source_id, function_scope, node)
                self.scope_names.append(node.name)
                self.scope_ids.append(function_id)
                for statement in node.body:
                    self.visit(statement)
                self.scope_ids.pop()
                self.scope_names.pop()

            def _record_type_names(self, expression: ast.AST,
                                   owner_id: str | None = None) -> None:
                for item in ast.walk(expression):
                    if isinstance(item, ast.Name):
                        pending_type_refs.append(
                            (owner_id or self.scope_ids[-1],
                             tuple(self.scope_names), item.id))

            def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                self._record_type_names(node.annotation)
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                target_name = None
                receiver = None
                if isinstance(node.func, ast.Name):
                    target_name = node.func.id
                elif (isinstance(node.func, ast.Attribute) and
                      isinstance(node.func.value, ast.Name) and
                      node.func.value.id in {"self", "cls"}):
                    target_name = node.func.attr
                    receiver = node.func.value.id
                if target_name:
                    pending_calls.append((
                        self.scope_ids[-1], source_id, tuple(self.scope_names),
                        target_name, receiver))
                self.generic_visit(node)

        ScopedSymbolVisitor().visit(tree)

    # Scope-aware import resolution over the existing Python AST index.
    # Imports are keyed by lexical scope and only unique repository-local targets
    # are resolved. Conditional imports retain distinct relationship types.
    counts.update({"resolved_imports": 0, "unresolved_imports": 0,
                   "ambiguous_imports": 0, "cross_module_calls": 0,
                   "conditional_imports": 0, "resolved_reexports": 0})
    imported_symbols: dict[tuple[str, tuple[str, ...], str], list[tuple[str, str]]] = {}
    imported_modules: dict[tuple[str, tuple[str, ...], str], list[str]] = {}
    import_records: list[tuple[str, tuple[str, ...], str, str, str, bool]] = []
    pending_module_attribute_calls: list[tuple[str, tuple[str, ...], str, str, str]] = []
    class_scopes: set[tuple[str, tuple[str, ...]]] = set()

    for module_id, scope in scope_bindings:
        for index in range(len(scope)):
            parent = scope[:index]
            declaration = symbol_definitions.get((module_id, parent, scope[index]), [])
            if any(kind == "class" for _, kind in declaration):
                class_scopes.add((module_id, scope[:index + 1]))

    def _python_import_target(relative: str, node: ast.ImportFrom) -> str | None:
        source_key = module_key(relative)
        is_package = relative == "__init__.py" or relative.endswith("/__init__.py")
        package_parts = source_key.split(".") if is_package else source_key.split(".")[:-1]
        if node.level:
            trim = node.level - 1
            if trim > len(package_parts):
                return None
            base = package_parts[:len(package_parts) - trim]
        else:
            base = []
        if node.module:
            base.extend(node.module.split("."))
        candidate = ".".join(base)
        return candidate if candidate in module_ids else None

    class ImportBindingVisitor(ast.NodeVisitor):
        def __init__(self, source_id: str, relative: str):
            self.source_id = source_id
            self.relative = relative
            self.scope: list[str] = []
            self.conditional_depth = 0

        def _visit_conditional(self, statements: list[ast.stmt]) -> None:
            self.conditional_depth += 1
            for statement in statements:
                self.visit(statement)
            self.conditional_depth -= 1

        def visit_If(self, node: ast.If) -> None:
            self.visit(node.test)
            self._visit_conditional(node.body)
            self._visit_conditional(node.orelse)

        def visit_While(self, node: ast.While) -> None:
            self.visit(node.test)
            self._visit_conditional(node.body)
            self._visit_conditional(node.orelse)

        def visit_For(self, node: ast.For) -> None:
            self.visit(node.iter)
            self._visit_conditional(node.body)
            self._visit_conditional(node.orelse)

        visit_AsyncFor = visit_For

        def visit_Try(self, node: ast.Try) -> None:
            self._visit_conditional(node.body)
            for handler in node.handlers:
                self._visit_conditional(handler.body)
            self._visit_conditional(node.orelse)
            self._visit_conditional(node.finalbody)

        def visit_TryStar(self, node: ast.Try) -> None:
            self.visit_Try(node)

        def visit_Match(self, node: ast.Match) -> None:
            self.visit(node.subject)
            for case in node.cases:
                self._visit_conditional(case.body)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            for expression in [*node.decorator_list, *node.bases, *node.keywords]:
                self.visit(expression)
            self.scope.append(node.name)
            for statement in node.body:
                self.visit(statement)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_function(node)

        def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            for expression in [*node.decorator_list, *node.args.defaults,
                               *[item for item in node.args.kw_defaults if item is not None]]:
                self.visit(expression)
            self.scope.append(node.name)
            for statement in node.body:
                self.visit(statement)
            self.scope.pop()

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                bound_name = alias.asname or alias.name.split(".")[0]
                target_key = alias.name if alias.asname else alias.name.split(".")[0]
                import_records.append((self.source_id, tuple(self.scope), bound_name,
                                       target_key, "module", self.conditional_depth > 0))

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            target_key = _python_import_target(self.relative, node)
            for alias in node.names:
                import_records.append((self.source_id, tuple(self.scope),
                                       alias.asname or alias.name,
                                       target_key or "", alias.name,
                                       self.conditional_depth > 0))

    for source_path in py_files:
        relative = source_path.relative_to(root_path).as_posix()
        tree = _parse_python(source_path)
        if tree is not None:
            ImportBindingVisitor(module_ids[module_key(relative)], relative).visit(tree)

    def _effective_import_scope(module_id: str, scope: tuple[str, ...],
                                name: str) -> tuple[str, ...]:
        if name in scope_global_names.get((module_id, scope), set()):
            return ()
        if name in scope_nonlocal_names.get((module_id, scope), set()):
            for depth in range(len(scope) - 1, -1, -1):
                candidate = scope[:depth]
                if candidate and (module_id, candidate) not in class_scopes:
                    if name in scope_bindings.get((module_id, candidate), set()):
                        return candidate
            return scope
        return scope

    unresolved_records = list(import_records)
    max_passes = max(1, len(module_ids) + 1)
    for _pass in range(max_passes):
        next_unresolved = []
        progress = False
        for source_id, lexical_scope, bound_name, target_key, imported_name, conditional in unresolved_records:
            effective_scope = _effective_import_scope(source_id, lexical_scope, bound_name)
            if imported_name == "module":
                target_id = module_ids.get(target_key)
                if target_id and target_id != source_id:
                    key = (source_id, effective_scope, bound_name)
                    values = imported_modules.setdefault(key, [])
                    if target_id not in values:
                        values.append(target_id)
                        progress = True
                    relation = "conditional_imports_module" if conditional else "imports_module"
                    graph.upsert_edge(source_id, target_id, relation, version)
                    counts["resolved_imports"] += 1
                    if conditional:
                        counts["conditional_imports"] += 1
                else:
                    counts["unresolved_imports"] += 1
                continue

            target_id = module_ids.get(target_key) if target_key else None
            if not target_id:
                child_key = ".".join(part for part in (target_key, imported_name) if part)
                target_id = module_ids.get(child_key)
                if target_id:
                    key = (source_id, effective_scope, bound_name)
                    imported_modules.setdefault(key, []).append(target_id)
                    relation = "conditional_imports_module" if conditional else "imports_module"
                    graph.upsert_edge(source_id, target_id, relation, version)
                    counts["resolved_imports"] += 1
                    if conditional:
                        counts["conditional_imports"] += 1
                    progress = True
                else:
                    counts["unresolved_imports"] += 1
                continue

            if imported_name == "*":
                counts["unresolved_imports"] += 1
                continue
            definitions = symbol_definitions.get((target_id, (), imported_name), [])
            if not definitions:
                definitions = imported_symbols.get((target_id, (), imported_name), [])
            unique = sorted({item[0]: item for item in definitions}.values())
            if len(definitions) == 1 and len(unique) == 1:
                key = (source_id, effective_scope, bound_name)
                values = imported_symbols.setdefault(key, [])
                if unique[0] not in values:
                    values.append(unique[0])
                    progress = True
                relation = "conditional_imports_symbol" if conditional else "imports_symbol"
                graph.upsert_edge(source_id, unique[0][0], relation, version)
                if source_id.endswith("/__init__.py:" + version):
                    graph.upsert_edge(source_id, unique[0][0],
                                      "conditional_reexports" if conditional else "reexports",
                                      version)
                    counts["resolved_reexports"] += 1
                counts["resolved_imports"] += 1
                if conditional:
                    counts["conditional_imports"] += 1
            elif len(definitions) > 1:
                counts["ambiguous_imports"] += 1
            else:
                child_key = ".".join(part for part in (target_key, imported_name) if part)
                child_id = module_ids.get(child_key)
                if child_id and child_id != source_id:
                    key = (source_id, effective_scope, bound_name)
                    imported_modules.setdefault(key, []).append(child_id)
                    relation = "conditional_imports_module" if conditional else "imports_module"
                    graph.upsert_edge(source_id, child_id, relation, version)
                    counts["resolved_imports"] += 1
                    if conditional:
                        counts["conditional_imports"] += 1
                    progress = True
                else:
                    next_unresolved.append((source_id, lexical_scope, bound_name,
                                             target_key, imported_name, conditional))
        if not progress:
            unresolved_records = next_unresolved
            break
        unresolved_records = next_unresolved

    counts["unresolved_imports"] += len(unresolved_records)

    class ModuleAliasCallVisitor(ast.NodeVisitor):
        def __init__(self, source_id: str):
            self.source_id = source_id
            self.scope: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.scope.append(node.name)
            for statement in node.body:
                self.visit(statement)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_function(node)

        def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            self.scope.append(node.name)
            for statement in node.body:
                self.visit(statement)
            self.scope.pop()

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                pending_module_attribute_calls.append(
                    (self.source_id, tuple(self.scope), func.value.id,
                     func.attr, self.source_id))
            self.generic_visit(node)

    for source_path in py_files:
        relative = source_path.relative_to(root_path).as_posix()
        tree = _parse_python(source_path)
        if tree is not None:
            ModuleAliasCallVisitor(module_ids[module_key(relative)]).visit(tree)

    def _resolve_lexical(module_id: str, scope: tuple[str, ...],
                         name: str) -> tuple[list[tuple[str, str]], bool]:
        for depth in range(len(scope), -1, -1):
            current_scope = scope[:depth]
            if current_scope and (module_id, current_scope) in class_scopes and depth < len(scope):
                continue
            key = (module_id, current_scope, name)
            definitions = symbol_definitions.get(key, [])
            if definitions:
                return definitions, False
            imported = imported_symbols.get((module_id, current_scope, name), [])
            if imported:
                return imported, False
            if name in scope_bindings.get((module_id, current_scope), set()):
                return [], False
        return [], False

    def _resolve_imported_module(module_id: str, scope: tuple[str, ...],
                                 name: str) -> list[str]:
        for depth in range(len(scope), -1, -1):
            current_scope = scope[:depth]
            if current_scope and (module_id, current_scope) in class_scopes and depth < len(scope):
                continue
            targets = imported_modules.get((module_id, current_scope, name), [])
            if targets:
                return sorted(set(targets))
            if name in scope_bindings.get((module_id, current_scope), set()):
                return []
        return []

    for caller_id, module_id, scope, target_name, receiver in pending_calls:
        if receiver in {"self", "cls"}:
            class_scope = next(
                (scope[:index + 1] for index in range(len(scope) - 2, -1, -1)
                 if any(kind == "class" for _, kind in symbol_definitions.get(
                     (module_id, scope[:index], scope[index]), []))),
                None)
            if class_scope is None:
                counts["unresolved_calls"] += 1
                continue
            definitions = symbol_definitions.get((module_id, class_scope, target_name), [])
        else:
            definitions, _ = _resolve_lexical(module_id, scope, target_name)
        unique = sorted({item[0] for item in definitions})
        if len(definitions) == 1 and len(unique) == 1 and unique[0] != caller_id:
            graph.upsert_edge(caller_id, unique[0], "calls", version)
            counts["calls"] += 1
            if unique[0].split(":class:", 1)[0].split(":func:", 1)[0] != module_id:
                counts["cross_module_calls"] += 1
        elif len(definitions) > 1:
            counts["ambiguous_calls"] += 1
            counts["unresolved_calls"] += 1
        else:
            counts["unresolved_calls"] += 1

    for caller_id, scope, alias_name, member_name, module_id in pending_module_attribute_calls:
        targets = _resolve_imported_module(module_id, scope, alias_name)
        if not targets:
            # Most attribute calls are ordinary object/member calls, not
            # imported-module calls. Do not misclassify them as import failures.
            continue
        if len(targets) != 1:
            counts["ambiguous_calls"] += 1
            counts["unresolved_calls"] += 1
            continue
        definitions = symbol_definitions.get((targets[0], (), member_name), [])
        unique_defs = sorted({item[0]: item for item in definitions}.values())
        if len(definitions) == 1 and len(unique_defs) == 1 and unique_defs[0][0] != caller_id:
            graph.upsert_edge(caller_id, unique_defs[0][0], "calls", version)
            counts["calls"] += 1
            counts["cross_module_calls"] += 1
        elif len(unique_defs) > 1:
            counts["ambiguous_calls"] += 1
            counts["unresolved_calls"] += 1
        else:
            counts["unresolved_calls"] += 1

    for child_id, scope, base_name in pending_inheritance:
        module_id = child_id.split(":class:", 1)[0]
        definitions, _ = _resolve_lexical(module_id, scope, base_name)
        class_definitions = [item for item in definitions if item[1] == "class"]
        targets = sorted({item[0] for item in class_definitions})
        if len(class_definitions) == 1 and len(targets) == 1 and targets[0] != child_id:
            graph.upsert_edge(child_id, targets[0], "inherits", version)
            counts["inherits"] += 1
        elif len(class_definitions) > 1:
            counts["ambiguous_inheritance"] += 1

    for source_id, scope, type_name in pending_type_refs:
        module_id = source_id.split(":func:", 1)[0].split(":class:", 1)[0]
        definitions, _ = _resolve_lexical(module_id, scope, type_name)
        type_definitions = [item for item in definitions
                            if item[1] in {"class", "type"}]
        targets = sorted({item[0] for item in type_definitions})
        if len(type_definitions) == 1 and len(targets) == 1 and targets[0] != source_id:
            graph.upsert_edge(source_id, targets[0], "type_uses", version)
            counts["type_references"] += 1
        elif len(type_definitions) > 1:
            counts["ambiguous_type_references"] += 1

    # Cross-language declaration inventory on the canonical graph. This
    # intentionally records syntax-level declarations and only resolves
    # unambiguous relative module paths; it does not claim semantic parsing.
    import re

    language_specs = {
        ".js": ("javascript", [
            (r"\b(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)", "function"),
            (r"\b(?:export\s+)?class\s+([A-Za-z_$][\w$]*)", "class"),
            (r"\b(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=", "value"),
        ]),
        ".jsx": ("javascript", [
            (r"\b(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)", "function"),
            (r"\b(?:export\s+)?class\s+([A-Za-z_$][\w$]*)", "class"),
        ]),
        ".ts": ("typescript", [
            (r"\b(?:export\s+)?(?:declare\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)", "class"),
            (r"\b(?:export\s+)?interface\s+([A-Za-z_$][\w$]*)", "interface"),
            (r"\b(?:export\s+)?type\s+([A-Za-z_$][\w$]*)", "type"),
            (r"\b(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)", "function"),
            (r"\b(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=", "value"),
        ]),
        ".tsx": ("typescript", [
            (r"\b(?:export\s+)?(?:declare\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)", "class"),
            (r"\b(?:export\s+)?interface\s+([A-Za-z_$][\w$]*)", "interface"),
            (r"\b(?:export\s+)?type\s+([A-Za-z_$][\w$]*)", "type"),
            (r"\b(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)", "function"),
        ]),
        ".rs": ("rust", [
            (r"\b(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_][\w]*)", "function"),
            (r"\b(?:pub\s+)?struct\s+([A-Za-z_][\w]*)", "struct"),
            (r"\b(?:pub\s+)?enum\s+([A-Za-z_][\w]*)", "enum"),
            (r"\b(?:pub\s+)?trait\s+([A-Za-z_][\w]*)", "trait"),
        ]),
        ".go": ("go", [
            (r"\bfunc\s+(?:\([^)]*\)\s*)?([A-Za-z_][\w]*)\s*\(", "function"),
            (r"\btype\s+([A-Za-z_][\w]*)\s+(?:struct|interface)\b", "type"),
        ]),
        ".java": ("java", [
            (r"\b(?:public|protected|private)?\s*(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)", "class"),
            (r"\b(?:public|protected|private)?\s*interface\s+([A-Za-z_$][\w$]*)", "interface"),
            (r"\b(?:public|protected|private)?\s*enum\s+([A-Za-z_$][\w$]*)", "enum"),
        ]),
    }
    ignored_dirs = {".git", ".venv", "venv", "node_modules", "vendor",
                    "dist", "build", "__pycache__", ".cortex-state"}
    foreign_files = [
        path for path in root_path.rglob("*")
        if path.is_file() and path.suffix in language_specs
        and not any(part in ignored_dirs for part in path.relative_to(root_path).parts)
        and path.stat().st_size <= max_file_bytes
    ]
    counts.update({"foreign_files": 0, "foreign_symbols": 0,
                   "foreign_local_imports": 0, "foreign_unresolved_imports": 0})
    foreign_modules: dict[str, str] = {}
    foreign_records: list[tuple[Path, str, str, str]] = []
    for source_path in foreign_files:
        rel = source_path.relative_to(root_path).as_posix()
        lang, _patterns = language_specs[source_path.suffix]
        module_id = f"{lang}:{root_path.name}/{rel}:{version}"
        foreign_modules[rel] = module_id
        foreign_records.append((source_path, rel, lang, module_id))
        prov = _provenance(source_path, version, license_tag,
                           "cortex.knowledge.indexer")
        graph.upsert_node(module_id, "module", rel, prov)
        counts["foreign_files"] += 1

    foreign_by_stem: dict[str, list[str]] = {}
    for _path, rel, _lang, module_id in foreign_records:
        stem = str(Path(rel).with_suffix(""))
        foreign_by_stem.setdefault(stem, []).append(module_id)
        foreign_by_stem.setdefault(Path(stem).name, []).append(module_id)

    for source_path, rel, lang, module_id in foreign_records:
        try:
            source = source_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            counts["skipped"] += 1
            continue
        prov = _provenance(source_path, version, license_tag,
                           "cortex.knowledge.indexer")
        _lang, patterns = language_specs[source_path.suffix]
        seen_symbols: set[tuple[str, str]] = set()
        for pattern, symbol_type in patterns:
            for match in re.finditer(pattern, source):
                name = match.group(1)
                if (symbol_type, name) in seen_symbols:
                    continue
                seen_symbols.add((symbol_type, name))
                symbol_id = f"{module_id}:{symbol_type}:{name}"
                graph.upsert_node(symbol_id, symbol_type, f"{rel}::{name}", prov)
                graph.upsert_edge(module_id, symbol_id, "defines", version)
                counts["foreign_symbols"] += 1

        # Resolve only relative JS/TS imports and Rust path modules when the
        # destination maps to exactly one indexed source file.
        import_specs = []
        if lang in {"javascript", "typescript"}:
            import_specs = re.findall(
                r"""\b(?:import|export)\s+(?:[^;]*?\s+from\s+)?["'](\.[^"']+)["']""",
                source)
            import_specs += re.findall(r"""\bimport\s*\(\s*["'](\.[^"']+)["']""", source)
        elif lang == "rust":
            import_specs = ["::".join(parts.split("::")[:-1])
                            for parts in re.findall(r"\buse\s+([A-Za-z_][\w:]*)", source)
                            if "::" in parts]
            import_specs += re.findall(r"\bmod\s+([A-Za-z_][\w]*)\s*;", source)
        for spec in sorted(set(import_specs)):
            if spec.startswith("."):
                base = (source_path.parent / spec).relative_to(root_path).as_posix()
                candidates = [base + ext for ext in (".ts", ".tsx", ".js", ".jsx")]
                candidates += [base + "/index" + ext for ext in (".ts", ".tsx", ".js", ".jsx")]
                candidates += [base]
            else:
                candidates = [str(source_path.parent.relative_to(root_path) / spec / "mod.rs"),
                              str(source_path.parent.relative_to(root_path) / (spec + ".rs"))]
            matches = sorted({foreign_modules[c] for c in candidates if c in foreign_modules})
            if len(matches) == 1 and matches[0] != module_id:
                graph.upsert_edge(module_id, matches[0], "imports_local", version)
                counts["foreign_local_imports"] += 1
            else:
                counts["foreign_unresolved_imports"] += 1


    store.flush()
    graph.commit()
    return counts
