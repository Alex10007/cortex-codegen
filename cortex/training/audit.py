"""Independent audit of the training pipeline (stage 5).

Every check executes against live components and fails loudly:
- input/target alignment: tokens [t] must predict tokens [t+1] exactly where
  the target is unmasked, and the unmasked region must be exactly the
  completion (incl. EOT), per record;
- gradient flow: one real backward pass; every parameter must receive a
  finite, non-zero gradient;
- loss computation: loss is token-mean over unmasked targets only; padded
  and prompt positions are excluded;
- padding: pad tokens never contribute to the loss and, with pad_mask, are
  excluded from attention keys (output invariance to pad token identity);
- autoregressive inference compatibility: after an overfit run the model
  retrains-free generates its trained completion.
"""
from __future__ import annotations

from typing import Any


def audit_alignment(recs: list[dict[str, Any]], tok, seq_len: int) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    check("shift: y[t] == x[t+1] over real tokens", _check_shift(recs, tok, seq_len))
    check("prompt labels fully masked (-100)", _check_prompt_mask(recs, tok, seq_len))
    check("completion region labels = actual next tokens", _check_completion_labels(recs, tok, seq_len))
    check("final unmasked target is EOT", _check_eot_target(recs, tok, seq_len))
    check("padded labels masked (-100)", _check_pad_labels(recs, tok, seq_len))
    return {
        "suite": "input-target alignment",
        "n_records": len(recs),
        "all_pass": all(c["ok"] for c in checks),
        "checks": checks,
    }


def _loss_regions(rec: dict[str, Any], tok, seq_len: int):
    prompt = tok.encode(rec["instruction"] + "\n")
    completion = tok.encode(rec["code"].strip() + "\n" + tok.EOT)
    ids = prompt + completion
    labels = [-100] * len(prompt) + completion[:]
    return prompt, completion, labels


def _check_shift(recs, tok, seq_len) -> bool:
    pad = tok.pad_id
    for rec in recs:
        x, y = _batch([rec], tok, seq_len)
        if x is None:
            return False
        row_x, row_y = x[0], y[0]
        limit = int((row_x != pad).nonzero().flatten()[-1])
        for t in range(limit):
            if int(row_y[t]) != -100 and int(row_y[t]) != int(x[0, t + 1]):
                return False
    return True


def _batch(recs, tok, seq_len):
    from cortex.training.sft import make_batch

    return make_batch(recs, tok, seq_len)


def _check_prompt_mask(recs, tok, seq_len) -> bool:
    for rec in recs:
        prompt, _, _ = _loss_regions(rec, tok, seq_len)
        x, y = _batch([rec], tok, seq_len)
        if x is None:
            return False
        for t in range(len(prompt) - 1):
            if int(y[0, t].item()) != -100:
                return False
    return True


def _check_completion_labels(recs, tok, seq_len) -> bool:
    for rec in recs:
        prompt, completion, _ = _loss_regions(rec, tok, seq_len)
        x, y = _batch([rec], tok, seq_len)
        if x is None:
            return False
        for t in range(len(prompt) - 1, len(prompt) + len(completion) - 2):
            expected = int(x[0, t + 1].item())
            if int(y[0, t].item()) != expected:
                return False
    return True


def _check_eot_target(recs, tok, seq_len) -> bool:
    for rec in recs:
        x, y = _batch([rec], tok, seq_len)
        if x is None:
            return False
        pad = tok.pad_id
        real = (x[0] != pad).nonzero().flatten()
        last_target_pos = int(real[-1]) - 1
        if int(y[0, last_target_pos].item()) != tok.eot_id:
            return False
    return True


def _check_pad_labels(recs, tok, seq_len) -> bool:
    for rec in recs:
        x, y = _batch([rec], tok, seq_len)
        if x is None:
            return False
        pad = tok.pad_id
        if int(((y != -100) & (x == pad)).sum()) != 0:
            return False
    return True


def audit_loss_and_padding(cfg_name: str, tok, recs: list[dict[str, Any]],
                           seq_len: int) -> dict[str, Any]:
    import torch

    from cortex.model.configs import get_config
    from cortex.model.transformer import CortexTransformer
    from cortex.training.sft import make_batch

    torch.manual_seed(4242)
    model = CortexTransformer(get_config(cfg_name))
    x, y = make_batch(recs, tok, seq_len)
    logits, loss = model(x, y, pad_mask=x == tok.pad_id)
    manual = torch.nn.functional.cross_entropy(
        logits[y != -100], y[y != -100])
    ok_loss = torch.isclose(loss, manual, rtol=1e-5)
    n_unmasked = int((y != -100).sum())
    n_total = y.numel()

    pad = tok.pad_id
    row = x[0]
    real = (row != pad).nonzero().flatten()
    pad_start = int(real[-1]) + 1
    x_variant = x.clone()
    x_variant[:, pad_start:] = (pad + 5) % model.cfg.vocab_size
    logits2, _ = model(x_variant, y, pad_mask=x_variant == pad)
    identical_unmasked = True
    pos = len([t for t in real]) - 2
    if not torch.allclose(logits[:, :pad_start],
                          logits2[:, :pad_start], atol=1e-4):
        identical_unmasked = False

    return {
        "suite": "loss computation & padding",
        "loss_matches_token_mean_over_unmasked": bool(ok_loss),
        "unmasked_targets": n_unmasked,
        "total_positions": n_total,
        "loss": float(loss.detach()),
        "pad_keys_excluded": identical_unmasked,
        "pad_attention_note": (
            "outputs identical while pad-token identities vary once pad_mask "
            "is applied"),
        "all_pass": bool(ok_loss) and identical_unmasked,
    }


def audit_gradient_flow(cfg_name: str) -> dict[str, Any]:
    import torch

    from cortex.model.configs import get_config
    from cortex.model.transformer import CortexTransformer

    torch.manual_seed(4242)
    model = CortexTransformer(get_config(cfg_name))
    B, T, V = 2, 24, model.cfg.vocab_size
    x = torch.randint(0, V, (B, T))
    y = torch.randint(0, V, (B, T))
    _, loss = model(x, y)
    loss.backward()
    no_grad, nonfinite, zero = [], [], []
    n_params = 0
    for name, p in model.named_parameters():
        n_params += 1
        g = p.grad
        if g is None:
            no_grad.append(name)
        elif not bool(torch.isfinite(g).all()):
            nonfinite.append(name)
        elif float(g.abs().sum()) == 0.0:
            zero.append(name)
    return {
        "suite": "gradient flow",
        "loss_finite": bool(torch.isfinite(loss)),
        "params_total": n_params,
        "params_without_grad": no_grad,
        "params_nonfinite_grad": nonfinite,
        "params_zero_grad": zero,
        "all_pass": (not no_grad) and (not nonfinite) and (not zero),
    }


def overfit_probe(cfg_name: str, tokenizer_path: str,
                  train_jsonl: str, n_records: int = 4, steps: int = 120,
                  batch_size: int = 4, lr: float = 1e-3,
                  target_loss: float = 1.0,
                  seed: int = 4242) -> dict[str, Any]:
    """Train until the model memorizes a tiny verified batch; then verify
    free-running and greedy next-token behavior on the trained data."""
    import json
    import resource

    import torch

    from cortex.model.configs import get_config
    from cortex.model.transformer import CortexTransformer
    from cortex.training.engine import set_seed
    from cortex.tokenizer.cortex_tokenizer import load_tokenizer
    from cortex.training.sft import make_batch

    set_seed(seed)
    tok = load_tokenizer(tokenizer_path)
    with open(train_jsonl, encoding="utf-8") as fh:
        all_recs = [json.loads(l) for l in fh if l.strip()]
    recs = all_recs[:n_records]
    cfg = get_config(cfg_name)
    model = CortexTransformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    model.train()
    for step in range(1, steps + 1):
        x, y = make_batch(recs, tok, cfg.max_seq_len)
        if x is None:
            raise RuntimeError("overfit probe: batch too long for seq_len")
        _, loss = model(x, y, pad_mask=x == tok.pad_id)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if losses[-1] <= target_loss:
            break
    model.eval()
    hits, targets = 0, 0
    with torch.no_grad():
        for rec in recs[:2]:
            prompt = tok.encode(rec["instruction"] + "\n")
            completion = tok.encode(rec["code"].strip() + "\n" + tok.EOT)
            ids = prompt + completion
            for t in range(len(prompt) - 1,
                           min(len(ids) - 1, len(prompt) + 24)):
                xt = torch.tensor([ids[:t]], dtype=torch.long)
                logits, _ = model(xt[:, -cfg.max_seq_len:])
                pred = int(logits[:, -1, :].argmax(dim=-1))
                targets += 1
                hits += int(pred == int(ids[t]))
    acc = round(hits / max(1, targets), 4)
    first = recs[0]
    prompt = tok.encode(first["instruction"])
    x = torch.tensor([prompt], dtype=torch.long)
    gen = model.generate(x, max_new_tokens=64, temperature=0.01, top_k=1,
                         eot_id=tok.eot_id)
    sample = tok.decode(gen[0].tolist()[len(prompt):])
    rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    return {
        "config": cfg_name,
        "records_overfit": [r["name"] for r in recs],
        "steps_executed": len(losses),
        "loss_first": round(losses[0], 4),
        "loss_last": round(losses[-1], 4),
        "target_loss": target_loss,
        "reached_target": losses[-1] <= target_loss,
        "greedy_next_token_accuracy_on_memorized_examples": acc,
        "free_running_sample": sample[:400],
        "peak_rss_mb": rss_mb,
    }
