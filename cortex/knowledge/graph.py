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
                source_path = Path(unquote(parsed.path)).resolve(strict=False)                     if parsed.scheme == "file" else None
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
