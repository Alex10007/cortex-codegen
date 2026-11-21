"""SQLite-backed knowledge graph: versioned nodes and typed edges."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from urllib.parse import unquote, urlparse
from typing import Any

from cortex.interfaces import Provenance


class KnowledgeGraph:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                label TEXT NOT NULL,
                version TEXT NOT NULL,
                provenance TEXT NOT NULL
            )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS edges (
                src TEXT NOT NULL,
                dst TEXT NOT NULL,
                type TEXT NOT NULL,
                version TEXT NOT NULL,
                PRIMARY KEY (src, dst, type, version)
            )"""
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src, type)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst, type)")
        self._conn.commit()

    def upsert_node(self, node_id: str, ntype: str, label: str,
                    provenance: Provenance) -> None:
        self._conn.execute(
            """INSERT INTO nodes (id, type, label, version, provenance)
               VALUES (?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 type=excluded.type, label=excluded.label,
                 version=excluded.version, provenance=excluded.provenance""",
            (node_id, ntype, label, provenance.version, json.dumps(provenance.__dict__)),
        )

    def upsert_edge(self, src: str, dst: str, etype: str, version: str) -> None:
        self._conn.execute(
            """INSERT OR IGNORE INTO edges (src, dst, type, version)
               VALUES (?,?,?,?)""",
            (src, dst, etype, version),
        )

    def commit(self) -> None:
        self._conn.commit()

    def neighbors(self, node_id: str, etype: str | None = None,
                  direction: str = "out") -> list[dict[str, Any]]:
        if direction == "out":
            rows = self._conn.execute(
                "SELECT dst, type, version FROM edges WHERE src=?", (node_id,)
            ).fetchall()
        elif direction == "in":
            rows = self._conn.execute(
                "SELECT src, type, version FROM edges WHERE dst=?", (node_id,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT dst, type, version FROM edges WHERE src=? "
                "UNION ALL SELECT src, type, version FROM edges WHERE dst=?",
                (node_id, node_id),
            ).fetchall()
        return [{"node": r[0], "type": r[1], "version": r[2]} for r in rows
                if etype is None or r[1] == etype]


    def impact_analysis(self, changed_paths: list[str], repo_root: str,
                        max_depth: int = 3, max_nodes: int = 500) -> dict[str, Any]:
        """Find repository-local dependents of changed files using this graph.

        Edges point from dependent to dependency. The traversal therefore
        walks incoming imports_local/calls/inherits edges. Results are bounded
        and include provenance-backed paths; unresolved paths are reported.
        """
        if not isinstance(changed_paths, list) or not changed_paths:
            raise ValueError("changed_paths must be a non-empty list")
        if not 1 <= max_depth <= 10 or not 1 <= max_nodes <= 10000:
            raise ValueError("impact bounds are outside supported limits")
        root = Path(repo_root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("repo_root must be an existing directory")
        targets: set[Path] = set()
        for raw in changed_paths:
            if not isinstance(raw, str) or not raw:
                raise ValueError("each changed path must be a non-empty string")
            candidate = Path(raw)
            candidate = candidate if candidate.is_absolute() else root / candidate
            resolved = candidate.resolve(strict=False)
            if resolved != root and root not in resolved.parents:
                raise ValueError(f"changed path escapes repository: {raw}")
            targets.add(resolved)

        rows = self._conn.execute(
            "SELECT id,type,label,provenance FROM nodes").fetchall()
        path_by_id: dict[str, str] = {}
        type_by_id: dict[str, str] = {}
        seeds: set[str] = set()
        for node_id, ntype, label, provenance_json in rows:
            try:
                provenance = json.loads(provenance_json)
                source_uri = provenance.get("source_uri", "")
                parsed = urlparse(source_uri)
                source_path = (
                    Path(unquote(parsed.path)).resolve(strict=False)
                    if parsed.scheme == "file" else None
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                source_path = None
            if source_path is not None:
                try:
                    rel = source_path.relative_to(root).as_posix()
                except ValueError:
                    continue
                path_by_id[node_id] = rel
                type_by_id[node_id] = ntype
                if source_path in targets:
                    seeds.add(node_id)

        if not seeds:
            return {
                "kind": "cortex-change-impact",
                "repo_root": str(root),
                "changed_paths": sorted(p.relative_to(root).as_posix() for p in targets),
                "resolved_changed_nodes": [],
                "impacted_nodes": [],
                "unresolved_paths": sorted(
                    p.relative_to(root).as_posix() for p in targets),
                "truncated": False,
                "max_depth": max_depth,
            }

        # A changed module also changes the symbols it defines.
        expanded_seeds = set(seeds)
        for node_id in list(seeds):
            rows = self._conn.execute(
                "SELECT dst FROM edges WHERE src=? AND type='defines'",
                (node_id,)).fetchall()
            expanded_seeds.update(row[0] for row in rows)

        seen = set(expanded_seeds)
        frontier = set(expanded_seeds)
        impacted: dict[str, dict[str, Any]] = {}
        traversable = ("imports_local", "calls", "inherits")
        for depth in range(1, max_depth + 1):
            next_frontier: set[str] = set()
            for dependency in sorted(frontier):
                placeholders = ",".join("?" for _ in traversable)
                rows = self._conn.execute(
                    f"SELECT src,type FROM edges WHERE dst=? AND type IN ({placeholders})",
                    (dependency, *traversable)).fetchall()
                for dependent, edge_type in rows:
                    if dependent in seen:
                        continue
                    seen.add(dependent)
                    impacted[dependent] = {
                        "node_id": dependent,
                        "node_type": type_by_id.get(dependent, "unknown"),
                        "path": path_by_id.get(dependent),
                        "relationship": edge_type,
                        "depth": depth,
                        "depends_on": dependency,
                    }
                    next_frontier.add(dependent)
                    if len(impacted) >= max_nodes:
                        break
                if len(impacted) >= max_nodes:
                    break
            if len(impacted) >= max_nodes or not next_frontier:
                break
            frontier = next_frontier

        resolved_paths = {path for node_id, path in path_by_id.items()
                          if node_id in seeds}
        unresolved = sorted(
            p.relative_to(root).as_posix() for p in targets
            if p.relative_to(root).as_posix() not in resolved_paths)
        return {
            "kind": "cortex-change-impact",
            "repo_root": str(root),
            "changed_paths": sorted(p.relative_to(root).as_posix() for p in targets),
            "resolved_changed_nodes": sorted(seeds),
            "impacted_nodes": sorted(impacted.values(),
                                     key=lambda row: (row["depth"], row["path"] or "",
                                                      row["node_id"])),
            "impacted_path_count": len({
                row["path"] for row in impacted.values() if row["path"]}),
            "unresolved_paths": unresolved,
            "truncated": len(impacted) >= max_nodes,
            "max_depth": max_depth,
        }

    def dependency_cycles(self, repo_root: str, relationship_types: tuple[str, ...] = ("imports_local",), max_nodes: int = 10000) -> dict[str, Any]:
        """Return strongly connected dependency components from canonical graph edges.

        Edges are interpreted as source/dependent -> destination/dependency.
        Only provenance-backed nodes inside repo_root are considered. This is
        bounded graph analysis, not a complete language-level dependency proof.
        """
        if not relationship_types or any(
                not isinstance(item, str) or not item for item in relationship_types):
            raise ValueError("relationship_types must contain non-empty strings")
        if not 1 <= max_nodes <= 10000:
            raise ValueError("max_nodes must be between 1 and 10000")
        root = Path(repo_root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("repo_root must be an existing directory")
        rows = self._conn.execute(
            "SELECT id,provenance FROM nodes").fetchall()
        path_by_id: dict[str, str] = {}
        for node_id, provenance_json in rows:
            try:
                provenance = json.loads(provenance_json)
                parsed = urlparse(provenance.get("source_uri", ""))
                source_path = (Path(unquote(parsed.path)).resolve(strict=False)
                               if parsed.scheme == "file" else None)
                relative = source_path.relative_to(root).as_posix() if source_path else None
            except (TypeError, ValueError, json.JSONDecodeError):
                relative = None
            if relative is not None:
                path_by_id[node_id] = relative
        truncated = len(path_by_id) > max_nodes
        allowed = set(sorted(path_by_id)[:max_nodes])
        placeholders = ",".join("?" for _ in relationship_types)
        edge_rows = self._conn.execute(
            f"SELECT src,dst,type FROM edges WHERE type IN ({placeholders})",
            relationship_types).fetchall()
        adjacency: dict[str, list[str]] = {node: [] for node in allowed}
        for src, dst, _kind in edge_rows:
            if src in allowed and dst in allowed:
                adjacency[src].append(dst)
        for node in adjacency:
            adjacency[node] = sorted(set(adjacency[node]))

        # Iterative Kosaraju avoids recursion-depth failures on large repositories.
        visited: set[str] = set()
        finish: list[str] = []
        for start in sorted(adjacency):
            if start in visited:
                continue
            stack: list[tuple[str, bool]] = [(start, False)]
            while stack:
                node, exiting = stack.pop()
                if exiting:
                    finish.append(node)
                    continue
                if node in visited:
                    continue
                visited.add(node)
                stack.append((node, True))
                for neighbor in reversed(adjacency[node]):
                    if neighbor not in visited:
                        stack.append((neighbor, False))
        reverse: dict[str, list[str]] = {node: [] for node in allowed}
        for src, targets in adjacency.items():
            for dst in targets:
                reverse[dst].append(src)
        assigned: set[str] = set()
        components: list[list[str]] = []
        for start in reversed(finish):
            if start in assigned:
                continue
            component: list[str] = []
            stack = [start]
            assigned.add(start)
            while stack:
                node = stack.pop()
                component.append(node)
                for neighbor in sorted(reverse[node], reverse=True):
                    if neighbor not in assigned:
                        assigned.add(neighbor)
                        stack.append(neighbor)
            component.sort()
            if len(component) > 1 or (component and component[0] in adjacency[component[0]]):
                components.append(component)
        components.sort(key=lambda item: (path_by_id.get(item[0], ""), item[0]))
        return {
            "kind": "cortex-dependency-cycles",
            "repo_root": str(root),
            "relationship_types": list(relationship_types),
            "cycle_count": len(components),
            "cycles": [{
                "node_ids": component,
                "paths": sorted({path_by_id[node] for node in component}),
            } for component in components],
            "indexed_node_count": len(allowed),
            "truncated": truncated,
            "limitations": [
                "Only selected graph relationships are analyzed.",
                "Unresolved imports and dependencies absent from the index are not inferred.",
                "A strongly connected component may contain more than one distinct simple cycle.",
            ],
        }

    def architecture_boundary_violations(
            self, repo_root: str, rules: list[dict[str, str]],
            relationship_types: tuple[str, ...] = ("imports_local",),
            max_nodes: int = 10000) -> dict[str, Any]:
        """Evaluate explicit path-prefix dependency rules against graph edges.

        Each rule has source_prefix and forbidden_target_prefix. Prefixes are
        repository-relative directory/file prefixes, not glob or regex syntax.
        """
        if not isinstance(rules, list) or not rules:
            raise ValueError("rules must be a non-empty list")
        normalized: list[dict[str, str]] = []
        for rule in rules:
            if not isinstance(rule, dict):
                raise ValueError("each rule must be an object")
            source = rule.get("source_prefix")
            target = rule.get("forbidden_target_prefix")
            if not isinstance(source, str) or not source or not isinstance(target, str) or not target:
                raise ValueError("rules require source_prefix and forbidden_target_prefix")
            for prefix in (source, target):
                candidate = Path(prefix)
                if candidate.is_absolute() or ".." in candidate.parts:
                    raise ValueError("rule prefixes must be repository-relative")
            normalized.append({
                "source_prefix": Path(source).as_posix().strip("./"),
                "forbidden_target_prefix": Path(target).as_posix().strip("./"),
            })
        if not 1 <= max_nodes <= 10000:
            raise ValueError("max_nodes must be between 1 and 10000")
        if not relationship_types or any(
                not isinstance(item, str) or not item for item in relationship_types):
            raise ValueError("relationship_types must contain non-empty strings")
        root = Path(repo_root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("repo_root must be an existing directory")
        rows = self._conn.execute(
            "SELECT id,provenance FROM nodes").fetchall()
        path_by_id: dict[str, str] = {}
        for node_id, provenance_json in rows:
            try:
                provenance = json.loads(provenance_json)
                parsed = urlparse(provenance.get("source_uri", ""))
                source_path = (Path(unquote(parsed.path)).resolve(strict=False)
                               if parsed.scheme == "file" else None)
                relative = source_path.relative_to(root).as_posix() if source_path else None
            except (TypeError, ValueError, json.JSONDecodeError):
                relative = None
            if relative is not None:
                path_by_id[node_id] = relative
        truncated = len(path_by_id) > max_nodes
        allowed = set(sorted(path_by_id)[:max_nodes])
        placeholders = ",".join("?" for _ in relationship_types)
        edges = self._conn.execute(
            f"SELECT src,dst,type FROM edges WHERE type IN ({placeholders})",
            relationship_types).fetchall()
        violations = []
        for src, dst, edge_type in edges:
            if src not in allowed or dst not in allowed:
                continue
            source_path, target_path = path_by_id[src], path_by_id[dst]
            for rule_index, rule in enumerate(normalized):
                source_prefix = rule["source_prefix"]
                target_prefix = rule["forbidden_target_prefix"]
                source_matches = (source_path == source_prefix or
                                  source_path.startswith(source_prefix.rstrip("/") + "/"))
                target_matches = (target_path == target_prefix or
                                  target_path.startswith(target_prefix.rstrip("/") + "/"))
                if source_matches and target_matches:
                    violations.append({
                        "rule_index": rule_index,
                        "source_node": src,
                        "target_node": dst,
                        "source_path": source_path,
                        "target_path": target_path,
                        "relationship": edge_type,
                    })
        violations.sort(key=lambda row: (
            row["rule_index"], row["source_path"], row["target_path"],
            row["source_node"], row["target_node"]))
        return {
            "kind": "cortex-architecture-boundaries",
            "repo_root": str(root),
            "rules": normalized,
            "violation_count": len(violations),
            "violations": violations,
            "indexed_node_count": len(allowed),
            "truncated": truncated,
            "limitations": [
                "Rules are caller-supplied path-prefix constraints, not inferred architecture policy.",
                "Only selected graph relationships with repository-local provenance are checked.",
                "Missing, unresolved, or unindexed dependencies cannot be reported.",
            ],
        }

    def nodes_by_type(self, ntype: str, limit: int = 1000) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id,label,version FROM nodes WHERE type=? LIMIT ?", (ntype, limit)
        ).fetchall()
        return [{"id": r[0], "label": r[1], "version": r[2]} for r in rows]

    def stats(self) -> dict[str, int]:
        nodes = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edges = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        types = dict(self._conn.execute(
            "SELECT type, COUNT(*) FROM nodes GROUP BY type").fetchall())
        return {"nodes": nodes, "edges": edges,
                "node_types": {k: int(v) for k, v in types.items()}}

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()
