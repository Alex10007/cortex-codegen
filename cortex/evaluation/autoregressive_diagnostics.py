"""Token-by-token autoregressive failure diagnostics (stage 12).

Instruments the existing generation pipeline without changing it: for a
free-running generation we capture the sampled token stream, the incremental
decoded text, per-step top-k distributions, entropy, repetition signals and
delimiter/quote state, and we validate incremental prefixes against the
Python parser to locate the FIRST structurally invalid token. A prefix
ending in an open construct (`"(" was never closed`, unterminated
triple-quoted string, unexpected EOF while parsing) is an OPEN state — an
expected condition mid-generation — and is distinguished from a HARD syntax
error (a prefix that can never become valid by continuing, e.g. unexpected
indent from a leading docstring, `return` outside a block, missing colon).

The analyzer has a self-test over known valid and invalid programs;
`analyzer_selftest()` runs it and the independent audit re-executes it.
"""
from __future__ import annotations

import ast
from typing import Any

import torch
import torch.nn.functional as F

# syntax-error message fragments that indicate a still-OPEN construct
_OPEN_STATE_FRAGMENTS = (
    "was never closed",
    "unexpected EOF while parsing",
    "unterminated triple-quoted string literal",
    "unterminated string literal",
    "expected an indented block",
)


def _err_state(exc: SyntaxError) -> str:
    msg = (exc.msg or "").lower()
    if any(frag in msg for frag in _OPEN_STATE_FRAGMENTS):
        return "open_state"
    return "hard_error"


def parse_state(text: str) -> dict[str, Any]:
    """Classify a text prefix as ok / open_state / hard_error."""
    try:
        ast.parse(text)
        return {"state": "ok", "message": None, "line": None,
                "column": None, "raw": None}
    except (SyntaxError, IndentationError, ValueError) as exc:
        base = getattr(exc, "msg", None)
        state = _err_state(exc) if isinstance(exc, SyntaxError) \
            else "hard_error"
        return {
            "state": state,
            "message": (base or None) if isinstance(exc, SyntaxError)
            else str(exc),
            "line": getattr(exc, "lineno", None),
            "column": getattr(exc, "offset", None),
            "raw": str(exc),
        }


def structure_scan(text: str) -> dict[str, Any]:
    """Char-level delimiter/quote state of a text prefix (outside strings).

    Tracks (), [], {} depth and whether a quoted string is currently open.
    Shared primitive for first-invalid analysis and EOT-gated constrained
    decoding.
    """
    depth = {"(": 0, "[": 0, "{": 0}
    close = {")": "(", "]": "[", "}": "{"}
    quote = None  # None | '"' | "'" | '"""' | "'''"
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if quote is not None:
            if len(quote) == 3 and text.startswith(quote, i):
                i += 3
                quote = None
                continue
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == quote[0]:
                if len(quote) == 1:
                    quote = None
                # triple-quoted string keeps accumulating plain quotes
            i += 1
            continue
        if text.startswith('"""', i) or text.startswith("'''", i):
            quote = text[i:i + 3]
            i += 3
            continue
        if ch in ('"', "'"):
            quote = ch
            i += 1
            continue
        if ch == "#":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch in depth:
            depth[ch] += 1
        elif ch in close:
            depth[close[ch]] = max(0, depth[close[ch]] - 1)
        i += 1
    return {
        "paren_depth": depth["("], "bracket_depth": depth["["],
        "brace_depth": depth["{"], "open_quote": quote,
        "unbalanced": quote is not None or any(k > 0 for k in depth.values()),
    }


def lexical_state(text: str) -> dict[str, Any]:
    """Fine-grained lexical state at the end of `text` for stage-13
    structural analysis: current indentation depth of the last line,
    whether the text sits mid-line, open-string state and delimiter
    depths. Extends structure_scan (kept unchanged)."""
    base = structure_scan(text)
    tail = text.split("\n")[-1]
    return {
        "line_indent_spaces": len(tail) - len(tail.lstrip(" ")), 
        "mid_line": tail != "" and not text.endswith("\n"),
        "last_line_blank_or_comment": tail.strip() == "" 
        or tail.strip().startswith("#"),
        **base,
    }


def classify_structural_failure(traj: dict[str, Any]) -> dict[str, Any]:
    """Stage-13 structural taxonomy: combine the incrementally-replayed
    first-invalid analysis with the per-step lexical history to attribute
    the defect (indent increase/decrease, missing indent, missing colon,
    unclosed delimiter/quote, premature/complete EOT, truncation, ...).
    `first_invalid_index` i refers to prefix i (i tokens); the offending
    token is therefore generated token i, i.e. steps[i-1]."""
    fi = traj["first_invalid"]
    idx = fi.get("first_invalid_index")
    if idx is None:
        if fi.get("failure_class") == "valid_program":
            return {"taxonomy": "complete_program", "evidence": fi}
        stopped = traj.get("stopped") or ""
        if stopped.startswith("repetition"):
            tax = "repetition_degeneration_truncation"
        elif stopped.startswith("max_new_tokens"):
            tax = "truncated_generation_max_new_tokens"
        else:
            tax = "premature_eot_open_construct"
        return {"taxonomy": tax,
                "evidence": {**fi, "stopped": stopped,
                             "n_tokens": traj["completion_length_tokens"]}}
    steps = traj["steps"]
    bad = steps[max(idx - 1, 0)] if steps else {}
    lex = bad.get("lex_before") or {}
    token = bad.get("token_text", "")
    cat = fi.get("category")
    common = {"step": idx, "token_text": token, "token_id":
              bad.get("token_id"), "lex_before": {
        k: lex.get(k) for k in ("line_indent_spaces", "mid_line",
                                "open_quote", "paren_depth",
                                "brace_depth")},
        "parser_message": fi.get("parser_message")}
    if cat == "premature_stop_open_construct":
        return {"taxonomy": "premature_eot_open_construct",
                "evidence": common}
    if cat == "missing_colon":
        return {"taxonomy": "missing_colon", "evidence": common}
    if cat in ("missing_closing_delimiter", "invalid_string_or_quote"):
        return {"taxonomy": "unclosed_delimiter_or_string",
                "evidence": common}
    if cat in ("invalid_indentation_transition", "incorrect_indentation"):
        indent = -1
        if token and "\n" in token:
            after_nl = token.split("\n")[-1]
            indent = len(after_nl) - len(after_nl.lstrip(" "))
        prev_indent = lex.get("line_indent_spaces") or 0
        msg = (fi.get("parser_message") or "").lower()
        if "unindent" in msg:
            tax = "incorrect_indent_decrease"
        elif indent >= 0 and indent < prev_indent:
            tax = "incorrect_indent_decrease"
        elif indent > prev_indent and cat == "incorrect_indentation":
            tax = "unexpected_indentation"
        elif indent > prev_indent:
            tax = "incorrect_indent_increase"
        else:
            tax = "unexpected_indentation_transition"
        common = dict(common, token_line_indent=indent)
        return {"taxonomy": tax, "evidence": common}
    if cat == "unexpected_end_of_sequence":
        return {"taxonomy": "incomplete_block_missing_indent",
                "evidence": common}
    return {"taxonomy": str(cat or "other"), "evidence": common}


def classify_hard_error(err: dict[str, Any]) -> str:
    """Map a hard SyntaxError record to the stage-12 failure taxonomy."""
    msg = (err.get("message") or "").lower()
    if "expected ':'" in msg:
        return "missing_colon"
    if "never closed" in msg:
        return "missing_closing_delimiter"
    if "unmatched" in msg and any(c in msg for c in ")]}"):
        return "missing_opening_delimiter"
    if "unindent does not match" in msg:
        return "invalid_indentation_transition"
    if "unexpected indent" in msg or "indentation" in msg:
        return "incorrect_indentation"
    if "invalid character" in msg or "invalid decimal" in msg:
        return "incorrect_token_boundary"
    if "unterminated" in msg:
        return "invalid_string_or_quote"
    if "is a keyword" in msg or "cannot assign" in msg \
            or "invalid syntax. perhaps" in msg:
        return "invalid_token"
    if "unexpected eof" in msg or "was never closed" in msg \
            or "expected an indented block" in msg:
        return "unexpected_end_of_sequence"
    return "other_parser_confirmed_syntax_error"


def first_invalid_token_incremental(prefixes: list[str]) -> dict[str, Any]:
    """Walk incremental decoded prefixes and locate the defect.

    Semantics (documented, self-tested):
    - final prefix parses clean -> valid_program;
    - some prefix is HARD-invalid and every later prefix stays HARD-invalid
      (the defect persists to the end of the output) -> the first such
      prefix is the structural failure point. Transient hard errors that
      later return to OK (e.g. the bare keyword `def` mid-typing) are NOT
      failures - only errors that persist count;
    - no persisting hard error and the final prefix is OPEN STATE
      (unclosed delimiter / unterminated string / EOF-while-parsing) ->
      incomplete_only_never_invalid: the model never emitted invalid
      syntax, it stopped prematurely - a truncation/premature-EOT outcome,
      not a syntax defect.

    `directly_introduced_by_newest_token` is True when the persisting-hard
    prefix's immediate predecessor prefix was fully parseable (the newest
    token breaks the parse); False when the predecessor already failed,
    i.e. the inconsistency was inherited from an earlier token."""
    if not prefixes:
        prefixes = [""]
    states = [parse_state(p) for p in prefixes]
    n = len(states)
    for i, st in enumerate(states):
        if st["state"] != "hard_error":
            continue
        if all(s["state"] == "hard_error" for s in states[i:]):
            last_valid = max((j for j, s in enumerate(states[:i])
                              if s["state"] == "ok"), default=-1)
            frac = last_valid / max(n - 1, 1)
            position = "beginning" if frac < 0.2 else (
                "end" if frac > 0.8 else "middle")
            prev_ok = (states[i - 1]["state"] == "ok") if i > 0 else True
            return {
                "first_invalid_index": i,
                "position": position,
                "directly_introduced_by_newest_token": bool(prev_ok),
                "failure_class": "invalid_token_direct" if prev_ok
                else "structural_inconsistency",
                "category": classify_hard_error(
                    {"message": st["message"], "raw": st["raw"]}),
                "parser_message": st["message"],
                "error_line": st["line"],
                "error_column": st["column"],
            }
    final = states[-1]
    return {
        "first_invalid_index": None,
        "position": None,
        "directly_introduced_by_newest_token": None,
        "failure_class": "incomplete_only_never_invalid"
        if final["state"] == "open_state" else "valid_program",
        "category": "premature_stop_open_construct" if final[
            "state"] == "open_state" else None,
        "parser_message": final["message"],
        "error_line": final["line"],
        "error_column": final["column"],
    }


def analyzer_selftest() -> dict[str, Any]:
    """Known valid and invalid programs verifying first-invalid analysis,
    structure_scan and classification (workstream I requirement)."""
    cases = [
        ("valid_program", "def f(x):\n    return x + 1\n", "valid"),
        ("valid_empty", "", "valid"),
        ("open_paren_truncation", "def f(x):\n    y = (x + 1", "open_only"),
        ("open_triple_quote_continuation",
         'def f(x):\n    """doc\n    >>> f(1)\n', "open_only"),
        ("closed_triple_quote_ok",
         'def f(x):\n    """doc\n    >>> f(1)\n    """\n    return x\n',
         "valid"),
        ("never_valid_leading_indent",
         "\n\n    \"\"\"\n    return x\n", "hard"),
        ("never_valid_unexpected_indent", "\n    x = 1\n", "hard"),
        ("never_valid_missing_colon", "def f()\n    return 1\n", "hard"),
        ("never_valid_unmatched_close", "x = 1\n)\n", "hard"),
        ("never_valid_double_return", "def f(x):\n    return return x\n",
         "hard"),
    ]
    results = []
    ok = True
    for name, program, expected in cases:
        prefixes = [program[:i] for i in range(0, len(program) + 1)]
        fi = first_invalid_token_incremental(prefixes)
        st = parse_state(program)
        scan = structure_scan(program)
        if expected == "valid":
            agrees = fi["failure_class"] == "valid_program" \
                and st["state"] == "ok" and not scan["unbalanced"]
        elif expected == "open_only":
            agrees = fi["failure_class"] == "incomplete_only_never_invalid" \
                and st["state"] == "open_state" and scan["unbalanced"]
        else:
            agrees = fi["first_invalid_index"] is not None \
                and st["state"] == "hard_error"

            if name == "never_valid_missing_colon":
                agrees = agrees and fi["category"] == "missing_colon"
        ok = ok and agrees
        results.append({"case": name, "expected": expected,
                        "analyzer_agrees": bool(agrees),
                        "failure_class": fi["failure_class"],
                        "category": fi["category"],
                        "first_invalid_index": fi["first_invalid_index"],
                        "parse_state": st["state"],
                        "parser_message": fi["parser_message"],})
    return {
        "self_test_ok": ok,
        "n_cases": len(results),
        "cases": results,
        "definition": "hard prefixes cannot become valid by further "
                      "generation; open_state prefixes can; increments "
                      "validated per prefix, not from the final error",
    }


@torch.no_grad()
def trajectory_generate(model, tok, prompt_text: str, decoding: dict,
                        seq_crop: int = 200,
                        eot_gated: bool = False) -> dict[str, Any]:
    """Mirror of CortexTransformer.generate's sampling loop (same ctx crop,
    temperature/top-k math, multinomial and EOT stop) extended with per-step
    instrumentation. Same-recipe equivalence with model.generate is self-
    tested deterministically (per-step top-5 vs raw forward math); sampled-
    token equality is not asserted because CPU torch.multinomial draws were
    observed to differ across process launches under identical seeds.
    With `eot_gated` (an EXPLORATORY decoding arm, never used to claim
    model learning), the EOT token is masked out while the completion sits
    inside an open syntactic construct."""
    ids = tok.encode(prompt_text)[-seq_crop:]
    tokens = torch.tensor([ids], dtype=torch.long)
    eot_id = tok.eot_id
    temperature = decoding["temperature"]
    top_k = decoding["top_k"]
    max_new_tokens = decoding["max_new_tokens"]
    steps = []
    generated_ids: list[int] = []
    stopped = "max_new_tokens"
    eot_step = None
    for i in range(max_new_tokens):
        ctx = tokens[:, -model.cfg.max_seq_len:]
        logits, _ = model(ctx)
        last = logits[0, -1, :] / max(temperature, 1e-6)
        probs_full = F.softmax(last, dim=-1)
        entropy = float(-(probs_full * probs_full.clamp_min(1e-12).log())
                        .sum())
        step_logits = last.clone()
        eot_blocked = False
        if eot_gated and generated_ids and structure_scan(
                tok.decode(generated_ids))["unbalanced"]:
            eot_blocked = True
        if top_k > 0:
            kth = torch.topk(step_logits,
                             min(top_k, step_logits.size(-1)))[0][-1:]
            step_logits = step_logits.masked_fill(
                step_logits < kth, float("-inf"))
        if eot_gated and eot_blocked:
            step_logits[eot_id] = float("-inf")
            if not torch.isfinite(step_logits).any():
                step_logits[eot_id] = 0.0
        probs = F.softmax(step_logits, dim=-1)
        top = torch.topk(probs, k=min(5, probs.size(-1)))
        nxt = torch.multinomial(probs, num_samples=1)
        nid = int(nxt)
        lex_before = lexical_state(tok.decode(generated_ids)) \
            if generated_ids else lexical_state("")
        generated_ids.append(nid)
        nxt = nxt.unsqueeze(0)
        comp = tok.decode(generated_ids)
        steps.append({
            "step": i + 1,
            "lex_before": {
                "line_indent_spaces": lex_before["line_indent_spaces"],
                "mid_line": lex_before["mid_line"],
                "open_quote": lex_before["open_quote"],
                "paren_depth": lex_before["paren_depth"],
                "brace_depth": lex_before["brace_depth"],
                "bracket_depth": lex_before["bracket_depth"]},
            "token_id": nid,
            "token_text": tok.decode([nid]),
            "top5": [{"id": int(t), "p": round(float(p), 6)}
                     for t, p in zip(top.indices, top.values)],
            "entropy": round(entropy, 4),
            "incremental_parse_state": parse_state(comp)["state"],
            "structure": structure_scan(comp),
            "eot_masked_this_step": eot_blocked,
        })
        tokens = torch.cat([tokens, nxt], dim=1)
        if nid == eot_id:
            stopped = "eot"
            eot_step = i + 1
            break
        if len(generated_ids) >= 8 and len(set(generated_ids[-4:])) == 1:
            stopped = f"repetition_degeneracy_at_step_{i + 1}"
            break
    analysis = first_invalid_token_incremental(
        [tok.decode(generated_ids[:j]) for j in
         range(0, len(generated_ids) + 1)])
    return {
        "prompt_token_ids": ids,
        "generated_token_ids": generated_ids,
        "text": tok.decode(generated_ids),
        "steps": steps,
        "stopped": stopped,
        "eot_step": eot_step,
        "completion_length_tokens": len(generated_ids),
        "first_invalid": analysis,
    }


@torch.no_grad()
def reference_token_probability(model, tok, ids: list[int],
                                 n_prompt: int,
                                 max_scored: int = 64) -> dict[str, Any]:
    """Teacher-forced probability of each reference completion token under a
    chosen prompt (workstream A2 'probability assigned to the reference
    token'), computed for eval-style vs train-style prompts on the SAME
    reference completion."""
    T = model.cfg.max_seq_len
    kept = ids[-T:]
    offset = len(ids) - len(kept)
    x = torch.tensor([kept[:-1]], dtype=torch.long)
    logits, _ = model(x)
    logp = F.log_softmax(logits[0], dim=-1)
    probs = []
    for j in range(n_prompt, min(len(ids), n_prompt + max_scored)):
        p = j - offset - 1
        if 0 <= p < x.size(1):
            probs.append(float(logp[p, ids[j]].item()))
    t = torch.tensor(probs) if probs else torch.tensor([0.0])
    return {
        "n_scored": len(probs),
        "mean_ref_token_logprob": round(float(t.mean()), 6),
        "mean_ref_token_prob": round(float(t.exp().mean()), 6)
        if probs else None,
        "min_ref_token_logprob": round(float(t.min()), 6)
        if probs else None,
    }
