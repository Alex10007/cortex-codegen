"""Structural training audit (stage 12 workstream B).

Measures, from parsed dataset records and the actual chunked stream,
whether the model is trained on complete program structures and whether
chunk-window boundaries split syntactic units. Uses the same parser and
delimiter scanner as the failure diagnostics so measurements are
comparable with generation outcomes.
"""
from __future__ import annotations

import ast
import hashlib
from typing import Any, Iterable

from cortex.evaluation.autoregressive_diagnostics import structure_scan
from cortex.training.longcontext import record_ids


def structure_metrics(code: str) -> dict[str, Any]:
    """Parse one code string and measure complete program structures."""
    fn_lens: list[int] = []
    max_depth = 0
    n_funcs = n_classes = n_imports = n_returns = 0
    has_docstring = 0
    decorators = 0
    annotations = 0
    try:
        tree = ast.parse(code)
        parses = True
    except (SyntaxError, ValueError):
        tree = None
        parses = False
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                n_funcs += 1
                fn_lens.append(len(ast.unparse(node).splitlines()))
                if node.returns is not None or any(
                        a.annotation for a in node.args.args):
                    annotations += 1
                if node.decorator_list:
                    decorators += 1
                if node.body and isinstance(node.body[0],
                                            ast.Expr) and isinstance(
                        node.body[0].value, ast.Constant) and isinstance(
                        node.body[0].value.value, str):
                    has_docstring += 1
            elif isinstance(node, ast.ClassDef):
                n_classes += 1
            elif isinstance(node, ast.Import):
                n_imports += 1
            elif isinstance(node, ast.ImportFrom):
                n_imports += 1
            elif isinstance(node, ast.Return):
                n_returns += 1

        def depth(node: ast.AST) -> int:
            kids = list(ast.iter_child_nodes(node))
            local = 1 if isinstance(node, (ast.FunctionDef, ast.ClassDef,
                                           ast.If, ast.For, ast.While,
                                           ast.Try, ast.With)) else 0
            return local + max((depth(k) for k in kids), default=0)

        max_depth = depth(tree)
    lines = code.splitlines()
    return {
        "parses": parses,
        "complete_functions": n_funcs,
        "complete_classes": n_classes,
        "imports": n_imports,
        "returns": n_returns,
        "functions_with_docstring": has_docstring,
        "functions_with_decorators": decorators,
        "functions_with_annotations": annotations,
        "max_function_len_astnodes": max(fn_lens) if fn_lens else 0,
        "code_lines": len(lines),
        "max_nesting_depth": max_depth,
        "delimiters_balanced": not structure_scan(code)["unbalanced"],
        "complete_program": bool(
            parses and n_funcs > 0 and code.strip() != ""),
    }


def body_fingerprint(code: str) -> dict[str, str]:
    """Stable fingerprints of top-level def/class bodies for duplication
    analysis across records and splits."""
    out: dict[str, str] = {}
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return out
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            try:
                body = ast.unparse(node)
            except Exception:
                continue
            out[f"{type(node).__name__}:{node.name}"] = hashlib.sha256(
                body.encode("utf-8")).hexdigest()
    return out


def audit_dataset_structure(recs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate structural coverage of the training dataset."""
    n = 0
    agg = {"parses": 0, "complete_program": 0, "with_class": 0,
           "with_imports": 0, "balanced": 0}
    fn_lens: list[int] = []
    depths: list[int] = []
    with_doc = 0
    for rec in recs:
        m = structure_metrics(rec["code"])
        n += 1
        agg["parses"] += int(m["parses"])
        agg["complete_program"] += int(m["complete_program"])
        agg["with_class"] += int(m["complete_classes"] > 0)
        agg["with_imports"] += int(m["imports"] > 0)
        agg["balanced"] += int(m["delimiters_balanced"])
        with_doc += int(m["functions_with_docstring"] > 0)
        if m["max_function_len_astnodes"]:
            fn_lens.append(m["max_function_len_astnodes"])
        depths.append(m["max_nesting_depth"])
    fn_lens.sort()
    pct = lambda k: round(agg[k] / n, 4) if n else None  # noqa: E731
    return {
        "n_records": n,
        "parses_rate": pct("parses"),
        "complete_program_rate": pct("complete_program"),
        "with_class_rate": pct("with_class"),
        "with_imports_rate": pct("with_imports"),
        "delimiters_balanced_rate": pct("balanced"),
        "docstring_records_rate": round(with_doc / n, 4) if n else None,
        "function_len_astnodes_avg": round(
            sum(fn_lens) / len(fn_lens), 1) if fn_lens else None,
        "function_len_astnodes_max": max(fn_lens) if fn_lens else None,
        "nesting_depth_avg": round(sum(depths) / len(depths), 2) if depths
        else None,
        "nesting_depth_max": max(depths) if depths else None,
    }


def chunk_boundary_audit(recs: list[dict[str, Any]], tok,
                         seq_len: int,
                         prompt_style: str = "legacy") -> dict[str, Any]:
    """For every chunked-stream window: does the window boundary fall
    inside an open syntactic construct (unbalanced delimiters or an open
    quote), and does the supervised target of each row END inside an open
    construct? Rows are windows exactly as make_batch_chunked builds them,
    decoded back through the same tokenizer."""
    from cortex.training.longcontext import chunk_record

    n_rows = 0
    n_input_end_open = 0
    per_split_kind = {"open_delimiter": 0, "open_quote": 0,
                      "clean": 0}
    for rec in recs:
        windows = chunk_record(rec, tok, seq_len, prompt_style)
        if not windows:
            continue
        for win_ids, _ in windows:
            text = tok.decode(win_ids)
            n_rows += 1
            scan = structure_scan(text)
            ends_open = scan["unbalanced"]
            n_input_end_open += int(ends_open)
            if ends_open and scan["open_quote"] is not None:
                per_split_kind["open_quote"] += 1
            elif ends_open:
                per_split_kind["open_delimiter"] += 1
            else:
                per_split_kind["clean"] += 1
    return {
        "n_windows": n_rows,
        "pct_window_end_inside_open_construct": round(
            n_input_end_open / n_rows, 4) if n_rows else None,
        "pct_window_end_clean": 1.0 - round(
            n_input_end_open / n_rows, 4) if n_rows else None,
        "kinds": per_split_kind,
        "n_supervised_targets_snapshot_note":
            "target-end construct state measured via the SAME window text "
            "(input row) - the streamed completion token layout is shared",
    }


def prompt_style_context_stats(recs: list[dict[str, Any]], tok,
                               seq_len: int) -> dict[str, Any]:
    """Measured EOT-in-context accounting for both prompt styles (the
    stage-12 sequence-boundary finding's root cause evidence)."""
    eot = tok.eot_id
    out: dict[str, Any] = {}
    for style in ("legacy", "eval_aligned"):
        rows_n = 0
        rows_with_eot_in_ctx = 0
        eot_in_ctx_total = 0
        prompt_len_sum = 0
        from cortex.training.longcontext import chunk_record

        for rec in recs:
            windows = chunk_record(rec, tok, seq_len, style)
            if not windows:
                continue
            for win_ids, _ in windows:
                # x row for this window is ids[:-1] (shifted like the
                # trainer does); targets start at index len(prompt)-1
                x = win_ids[:-1]
                rows_n += 1
                c = sum(1 for t in x if t == eot)
                eot_in_ctx_total += c
                rows_with_eot_in_ctx += int(c > 0)
                prompt, _ = record_ids(rec, tok, style)
                prompt_len_sum += len(prompt)
        out[style] = {
            "rows": rows_n,
            "rows_with_eot_in_context": rows_with_eot_in_ctx,
            "eot_in_context_total": eot_in_ctx_total,
            "avg_prompt_tokens": round(
                prompt_len_sum / max(rows_n, 1), 1),
        }
    return out
