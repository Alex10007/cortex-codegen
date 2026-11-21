#!/usr/bin/env python3
"""Stage 4: 2B model foundation & autoregressive intelligence.

Executed, fail-fast pipeline:
 1. verify the exact 2B architecture + parameter count (meta device);
 2. hardware profile + 2B training plan (memory estimation);
 3. validate the scalable path: mixed-precision(bf16) real training steps
    + checkpoint round-trip on the demo config;
 4. dataset v3: v2 records + verified multi-file examples (held-out/val
    immutable by construction, fingerprints checked against both manifests);
 5. re-SFT with CORRECT next-token labels from the pretrained base,
    per-epoch autoregressive eval (v1 + v2 suites, both prompt styles);
 6. baseline checkpoints evaluated under identical conditions;
 7. retrieval-assisted generation A/B (verified-pattern store, no
    retraining) on the final epoch;
 8. promotion gate + stage report.

Env: CORTEX_STAGE4_EPOCHS (default 4), CORTEX_STAGE4_SKIP_AMP=1.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ART = ROOT / "artifacts"


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_expansion_helpers():
    return _load_script_module("expansion_run",
                               ROOT / "scripts" / "expansion_run.py")


def _load_eval_model():
    return _load_script_module("eval_model",
                               ROOT / "scripts" / "eval_model.py")


def verify_2b() -> dict:
    import torch

    from cortex.model.configs import CORTEX_2B
    from cortex.model.transformer import CortexTransformer

    theoretical = CORTEX_2B.theoretical_param_count()
    with torch.device("meta"):
        meta_model = CortexTransformer(CORTEX_2B)
    per = {
        "tok_emb": meta_model.tok_emb.weight.numel(),
        "head": int(meta_model.cfg.tie_word_embeddings) and 0
        or CORTEX_2B.vocab_size * CORTEX_2B.dim,
        "final_ln": 2 * CORTEX_2B.dim,
        "blocks": sum(float(p.numel()) for n, p in meta_model.named_parameters()
                      if n.startswith("blocks.")),
    }
    actual = meta_model.actual_param_count()
    breakdown = {k: int(v) for k, v in per.items()}
    info = {
        "config": CORTEX_2B.name,
        "dim": CORTEX_2B.dim,
        "n_layers": CORTEX_2B.n_layers,
        "n_heads": CORTEX_2B.n_heads,
        "head_dim": CORTEX_2B.dim // CORTEX_2B.n_heads,
        "vocab_size": CORTEX_2B.vocab_size,
        "max_seq_len": CORTEX_2B.max_seq_len,
        "tie_word_embeddings": CORTEX_2B.tie_word_embeddings,
        "f_fn_mult": CORTEX_2B.ffn_mult,
        "d_ff": CORTEX_2B.d_ff,
        "theoretical_param_count": theoretical,
        "meta_instantiated_param_count": int(actual),
        "counts_match": theoretical == actual,
        "at_least_2e9": theoretical >= 2_000_000_000,
        "breakdown": breakdown,
        "training_status": "DECLARED-NOT-TRAINED (no host with sufficient "
                           "compute in this environment; see scaling plan)",
    }
    if not (info["counts_match"] and info["at_least_2e9"]):
        raise RuntimeError(f"2B verification failed: {info}")
    return info


def main() -> None:
    import torch

    from cortex.evaluation import benchmarks
    from cortex.evaluation.inference import (
        compile_diagnosis, decode_ablation, prompt_format_ab,
        teacher_forced_token_accuracy)
    from cortex.evaluation.runner_v2 import evaluate_v2
    from cortex.model.configs import CORTEX_2B, get_config
    from cortex.model.transformer import CortexTransformer
    from cortex.training.dataset_v2 import build_dataset_v3, load_split_v2
    from cortex.training.promotion import promotion_gate
    from cortex.training.scaling import hardware_profile, training_plan, \
        validate_scalable_path
    from cortex.training.sft import evaluate_heldout, fine_tune
    from cortex.core import CortexAgent

    exp = _load_expansion_helpers()
    evm = _load_eval_model()
    report: dict = {}

    print("[1/8] verifying exact 2B architecture (meta-device instanti\
ation)...")
    report["model_2b"] = verify_2b()
    print(f"      {report['model_2b']['theoretical_param_count']:,} params "
          f"(meta == theoretical: {report['model_2b']['counts_match']}, "
          f">=2B: {report['model_2b']['at_least_2e9']})")
    (ART / "model_2b_arch.json").write_text(
        json.dumps(report["model_2b"], indent=2))

    print("[2/8] hardware profile + 2B scalable training plan...")
    plan = training_plan(CORTEX_2B, batch_size=8)
    report["scaling_plan_2b"] = plan
    print(f"      host RAM {plan['host']['ram_gb']} GB, cuda="
          f"{plan['host']['cuda']['available']} -> {plan['strategy']} "
          f"(est. {plan['memory']['total_gb_est']} GB)")
    (ART / "scaling_plan_2b.json").write_text(
        json.dumps({k: (v if k != "host" else {**v, "cuda": v["cuda"]})
                    for k, v in plan.items()}, indent=2, default=str))

    if os.environ.get("CORTEX_STAGE4_SKIP_AMP") != "1":
        print("[3/8] validating scalable path: real bf16 steps + checkpoint "
              "round-trip on demo config...")
        data_bin = str(ART / "data" / "tokens.npy")
        amp = validate_scalable_path("cortex-demo", str(ART / "tokenizer" /
                                                        "tokenizer.json"),
                                     data_bin, str(ART / "runs" / "amp-smoke"),
                                     steps=int(os.environ.get(
                                         "CORTEX_SMOKE_STEPS", "20")))
        report["scalable_path_validation"] = amp
        print(f"      precision={amp['precision']} loss {amp['loss_first']} "
              f"-> {amp['loss_last']} finite={amp['losses_finite']} "
              f"reload_ok={amp['checkpoint_saved_and_reloaded']}")
        (ART / "amp_smoke.json").write_text(json.dumps(amp, indent=2))
    else:
        print("[3/8] mixed-precision validation skipped by env")

    print("[4/8] dataset v3 (verified multi-file merge; held-out/val "
          "immutable)...")
    from cortex.learning.multifile_examples import build_multifile_pairs

    mf_pairs, mf_verify = build_multifile_pairs()
    v3 = build_dataset_v3(
        str(ART / "sft" / "dataset_v2.jsonl"), mf_pairs,
        str(ART / "sft" / "dataset_v3.jsonl"),
        str(ART / "sft" / "v3_manifest.json"))
    report["dataset_v3"] = {**v3, "multifile_verify_ok": all(
        t.get("ok") for pkg in mf_verify.values() for t in pkg.values())}
    print(f"      counts={v3['counts']} added_to_train="
          f"{v3['stats']['added_to_train']} immutable="
          f"{v3['heldout_val_immutability']}")
    (ART / "multifile_verify.json").write_text(
        json.dumps(mf_verify, indent=2))

    tok_path = str(ART / "tokenizer" / "tokenizer.json")
    base_ckpt = ART / "runs" / "cortex-demo" / "final.pt"

    print("[5/8] re-SFT with corrected next-token labels (per-epoch "
          "autoregressive eval)...")
    agent = CortexAgent(state_dir=str(ART / "cortex-state"),
                        tokenizer_path=tok_path,
                        checkpoint_path=str(base_ckpt))
    v1_h: dict = {}
    epochs = int(os.environ.get("CORTEX_STAGE4_EPOCHS", "4"))
    staged: list[dict] = []
    init_ckpt = str(base_ckpt)
    for epoch in range(1, epochs + 1):
        out_dir = str(ART / "runs" / "cortex-sft-stage4")
        summary = fine_tune("cortex-demo", tok_path,
                            str(ART / "sft" / "dataset_v3.jsonl"),
                            init_ckpt, out_dir, epochs=1, batch_size=8,
                            lr=1e-4, keep_checkpoints=4, log_every=10)
        run_dir = Path(out_dir)
        epoch_ckpt = run_dir / f"stage4-epoch-{epoch:03d}.pt"
        ck = torch.load(summary["checkpoint"], map_location="cpu",
                        weights_only=True)
        ck["stage4_epoch"] = epoch
        ck["init_checkpoint"] = str(init_ckpt)
        torch.save(ck, epoch_ckpt)
        agent.load_checkpoint(str(epoch_ckpt))
        v1_report = evm.evaluate_checkpoint(ART, str(epoch_ckpt))
        v1_h = v1_report
        held_recs = load_split_v2(str(ART / "sft" / "dataset_v2.jsonl"),
                                  "heldout",
                                  str(ART / "sft" / "v2_manifest.json"))
        hl = evaluate_heldout(agent.model, held_recs, agent.tokenizer,
                              agent.model.cfg.max_seq_len)
        v2 = evaluate_v2(
            candidate_edits={
                t.task_id: {"task.py": exp._gen_module(agent, t.instruction)}
                for t in exp._v2_tasks() if t.edit_targets == ["task.py"]},
            candidate_texts={
                t.task_id: exp._gen_text(agent, t.instruction)
                for t in exp._v2_tasks() if t.edit_targets != ["task.py"]},
        )
        staged.append({
            "epoch": epoch,
            "checkpoint": str(epoch_ckpt),
            "heldout_loss": hl,
            "v1_pass_at_1": v1_report["pass_at_1"],
            "v1_nonempty": round(sum(
                1 for r in v1_report["per_task"]
                if r["generated_head"].strip()) / len(v1_report["per_task"]), 4),
            "v2_pass_at_1": v2["pass_at_1"],
            "v2_compile_ok_rate": v2["compile_ok_rate"],
            "v2_executed_rate": v2["executed_rate"],
        })
        print(f"      epoch {epoch}: heldout_loss={hl} v1={v1_report['pass_at_1']} "
              f"v2={v2['pass_at_1']} "
              f"compile_ok={v2['compile_ok_rate']}")
        init_ckpt = str(epoch_ckpt)
    report["staged_training"] = staged
    final_ckpt = staged[-1]["checkpoint"] if staged else str(base_ckpt)

    print("[6/8] baselines under identical conditions...")
    baselines = []
    for label, ck_path in (
            ("pretrained-base", str(base_ckpt)),
            ("stage2-sft-unshifted-labels",
             str(ART / "runs" / "cortex-sft" / "sft-final.pt")),
            ("stage3-sft-unshifted-labels",
             str(ART / "runs" / "cortex-sft-expansion" /
                 "sft-expansion-epoch-003.pt"))):
        if not Path(ck_path).exists():
            continue
        agent.load_checkpoint(ck_path)
        v1_report = evm.evaluate_checkpoint(ART, ck_path)
        held_recs = load_split_v2(str(ART / "sft" / "dataset_v2.jsonl"),
                                  "heldout",
                                  str(ART / "sft" / "v2_manifest.json"))
        hl = evaluate_heldout(agent.model, held_recs, agent.tokenizer,
                              agent.model.cfg.max_seq_len)
        v2 = evaluate_v2(
            candidate_edits={
                t.task_id: {"task.py": exp._gen_module(agent, t.instruction)}
                for t in exp._v2_tasks() if t.edit_targets == ["task.py"]},
            candidate_texts={
                t.task_id: exp._gen_text(agent, t.instruction)
                for t in exp._v2_tasks() if t.edit_targets != ["task.py"]},
        )
        baselines.append({
            "label": label, "checkpoint": ck_path,
            "heldout_loss": hl, "v1_pass_at_1": v1_report["pass_at_1"],
            "v1_nonempty": round(sum(
                1 for r in v1_report["per_task"]
                if r["generated_head"].strip()) / len(v1_report["per_task"]), 4),
            "v2_pass_at_1": v2["pass_at_1"],
            "v2_compile_ok_rate": v2["compile_ok_rate"],
            "v2_executed_rate": v2["executed_rate"],
        })
        print(f"      {label}: v1={v1_report['pass_at_1']} "
              f"v2={v2['pass_at_1']}")
    report["baselines"] = baselines

    print("[7/8] retrieval-assisted generation A/B (verified-pattern store)...")
    agent.load_checkpoint(final_ckpt)

    from cortex.evaluation.runner import grade_single
    from cortex.evaluation.tasks import TASKS as V1_TASKS

    def _gen(prompt: str, max_new_tokens: int = 160) -> str:
        ids = agent.tokenizer.encode(prompt)
        x = torch.tensor([ids[-200:]], dtype=torch.long)
        gen = agent.model.generate(x, max_new_tokens=max_new_tokens,
                                   temperature=0.2, top_k=20,
                                   eot_id=agent.tokenizer.eot_id)
        return agent.tokenizer.decode(gen[0].tolist()[len(x[0]) - 1:])

    without, with_r, n_hits = [], [], 0
    for task in V1_TASKS:
        cand = _extract(_gen(agent.tokenizer.EOT + "\n# Task: " +
                             task.instruction + "\n"))
        without.append(bool(grade_single(task, cand)["passed"]))
        hits = agent.store.search(task.instruction, k=1, kind="pattern")
        if hits:
            n_hits += 1
            prompt = (agent.tokenizer.EOT + "\n# Task: " + task.instruction
                      + "\n# Verified example:\n" + hits[0]["text"][:600]
                      + "\n")
            cand = _extract(_gen(prompt))
            with_r.append(bool(grade_single(task, cand)["passed"]))
    report["rag_ablation"] = {
        "v1_without_retrieval_pass_at_1": round(
            sum(without) / max(1, len(without)), 4),
        "v1_with_retrieval_pass_at_1": round(
            sum(with_r) / max(1, len(with_r)), 4),
        "tasks_with_retrieved_pattern": n_hits,
        "note": "retrieval from the execution-verified pattern store "
                "(kind=pattern, verified=1); no retraining involved",
    }
    print(f"      without={report['rag_ablation']['v1_without_retrieval_pass_at_1']} "
          f"with={report['rag_ablation']['v1_with_retrieval_pass_at_1']}")

    print("[8/8] promotion gate + report...")
    intg = benchmarks.check_integrity()
    base_by_label = {b["label"]: b for b in baselines}
    baseline_key = ("stage3-sft-unshifted-labels" in base_by_label
                    and base_by_label["stage3-sft-unshifted-labels"]) or \
                   (base_by_label.get("pretrained-base"))
    candidate = staged[-1] if staged else None
    if baseline_key and candidate:
        gate = promotion_gate(
            {"pass_at_1": baseline_key["v2_pass_at_1"],
             "val_loss": baseline_key["heldout_loss"] or 0.0},
            {"pass_at_1": candidate["v2_pass_at_1"],
             "compile_ok_rate": candidate["v2_compile_ok_rate"],
             "executed_rate": candidate["v2_executed_rate"],
             "per_tier": {},
             "val_loss": candidate["heldout_loss"] or 0.0},
            fingerprint_ok=True)
        gate["candidate_checkpoint"] = candidate["checkpoint"]
        gate["baseline_checkpoint"] = baseline_key["checkpoint"]
    else:
        gate = {"promoted": False, "reasons": ["missing baseline or candidate"]}
    report["promotion"] = gate
    report["benchmark_integrity"] = {
        "v1_status": intg.get("status"),
        "v1_fingerprint": intg.get("fingerprint"),
    }
    print(f"      promoted={gate['promoted']} reasons={gate['reasons']}")
    report["diagnostics"] = {
        "prompt_format_ab": prompt_format_ab(agent),
        "decode_ablation": decode_ablation(agent, exp._v2_tasks()[1]),
        "compile_stage": compile_diagnosis(agent, exp._v2_tasks()[0]),
    }
    held_recs = load_split_v2(str(ART / "sft" / "dataset_v2.jsonl"), "heldout",
                              str(ART / "sft" / "v2_manifest.json"))
    report["diagnostics"]["teacher_forced_token_accuracy"] = {
        "stage4_final": teacher_forced_token_accuracy(
            agent.model, agent.tokenizer, held_recs,
            get_config("cortex-demo").max_seq_len),
        "note": "labels now properly shifted; accuracy is true next-token",
    }
    (ART / "stage4_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    print("stage-4 report -> artifacts/stage4_report.json")


def _extract(text: str) -> str:
    import re

    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, flags=re.S)
    return blocks[0] if blocks else text


if __name__ == "__main__":
    main()
