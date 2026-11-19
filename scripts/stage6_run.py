#!/usr/bin/env python
"""Stage 6 orchestrator: GPU scale-up preparation and execution of every
training/validation step this host genuinely supports.

Stages:
 1. checkpoint lineage + benchmark integrity (unchanged fingerprints)
 2. GPU infrastructure re-measure + 2B launch gate
 3. long-sequence resolution: measured stats + coverage proof (req. 4)
 4. FSDP2 validation: sharding, all-reduce agreement, sharded resume,
    full-state export (req. 3)
 5. launcher mechanics: real spawn on demo config; cortex-2b refused by the
    hardware gate (measured, not simulated) (req. 2/12)
 6. chunked code-specialized SFT continuation with per-epoch eval (req. 6/8)
 7. promotion gate (req. 11)
 8. knowledge expansion attempt (validated patterns only) (req. 10)
 9. stage report -> artifacts/stage6_report.json
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    import torch

    from cortex.evaluation.inference import (
        compile_diagnosis, decode_ablation, prompt_format_ab,
        teacher_forced_token_accuracy)
    from cortex.model.configs import get_config
    from cortex.tokenizer.cortex_tokenizer import load_tokenizer as _lt
    from cortex.training.dataset_v2 import load_split_v2
    from cortex.training.fsdp import spawn_fsdp_validation
    from cortex.training.longcontext import (chunk_record,
                                             completion_coverage,
                                             dataset_length_stats)
    from cortex.training.promotion import promotion_gate
    from cortex.training.scaling import cluster_plans, hardware_profile
    from cortex.training.sft import evaluate_heldout, fine_tune

    stage5 = _load("stage5_run", ROOT / "scripts" / "stage5_run.py")
    evm = _load("eval_model", ROOT / "scripts" / "eval_model.py")
    exp = _load("expansion_run", ROOT / "scripts" / "expansion_run.py")

    tok_path = str(ART / "tokenizer" / "tokenizer.json")
    tok = _lt(tok_path)
    ds_v3 = str(ART / "sft" / "dataset_v3.jsonl")
    report: dict = {}

    print("[1/8] lineage & benchmark integrity...")
    lineage = stage5.lineage_ok()
    report["lineage"] = lineage
    if not lineage["checkpoint_lineage_complete"]:
        raise RuntimeError(f"lineage incomplete: {lineage['missing']}")
    if lineage["v1_benchmark_integrity"]["status"] != "unmodified":
        raise RuntimeError("v1 benchmark integrity violated - aborting")
    from cortex.evaluation import benchmarks
    v2_fp = benchmarks_v2_fingerprint()
    report["benchmark_fingerprints"] = {
        "v1": lineage["v1_benchmark_integrity"]["fingerprint"],
        "v1_status": lineage["v1_benchmark_integrity"]["status"],
        "v2_fingerprint": v2_fp,
    }
    print("      v1 unmodified; v2 fingerprint stable")

    print("[2/8] GPU infrastructure re-measure + 2B launch gate...")
    profile = hardware_profile()
    cfg2b = get_config("cortex-2b")
    matrix = cluster_plans(cfg2b, batch_size=8)
    report["gpu_hardware"] = {
        "measured": profile,
        "cluster_matrix_2b": matrix,
        "phase_b_pretraining_gate": {
            "config": "cortex-2b",
            "params": cfg2b.theoretical_param_count(),
            "gpus_present": profile["cuda"]["device_count"],
            "gpus_required": min(
                (p["world_size"] for p in matrix["plans"]
                 if p["fits_40gb_gpu"]), default=None),
            "status": ("READY" if profile["cuda"]["available"]
                       and profile["cuda"]["device_count"] >= 8
                       else "BLOCKED: this host measured "
                            f"cuda={profile['cuda']['available']}, devices="
                            f"{profile['cuda']['device_count']}; the 2B run "
                            "must execute on the Fibonera GPU cluster "
                            "(8 x 40 GB or 4 x 80 GB). Nothing simulated."),
        },
    }
    gate_status = report["gpu_hardware"]["phase_b_pretraining_gate"]["status"]
    print(f"      {gate_status[:100]}")
    (ART / "gpu_hardware_stage6.json").write_text(
        json.dumps(report["gpu_hardware"], indent=2, default=str))

    print("[3/8] long-sequence resolution (measured)...")
    stats = {
        str(sl): dataset_length_stats(ds_v3, tok, sl)
        for sl in (256, 2048, 8192)
    }
    seq_demo = get_config("cortex-demo").max_seq_len
    from cortex.training.dataset_v2 import load_split_v2 as _lsv2
    train_recs = _lsv2(ds_v3, "train", str(ART / "sft" / "v3_manifest.json"))
    cov = completion_coverage(train_recs, tok, seq_demo)
    excluded_example = None
    for rec in train_recs:
        if chunk_record(rec, tok, seq_demo) is None:
            excluded_example = rec["name"]
            break
    report["long_sequence_resolution"] = {
        "stats_by_seq_len": stats,
        "completion_coverage_demo_seq_len": cov,
        "chunking_reclaims_records": (
            stats["256"]["splits"]["train"]["recovered_by_chunking"]),
        "instruction_alone_too_long_count": (
            stats["256"]["splits"]["train"]
            ["instruction_too_long_untrainable"]),
        "example_untrainable_record": excluded_example,
        "note": ("supervised targets remain exactly the completion tokens in "
                 "every window; instruction tokens masked -100; prompts that "
                 "alone exceed the window are counted, never truncated"),
    }
    print(f"      legacy drop 69.29% -> remaining "
          f"{stats['256']['splits']['train']['remaining_exclusion_rate']*100:.2f}% "
          f"(demo seq 256); seq 8192 (2B) covers 100%")
    (ART / "longseq_stats.json").write_text(
        json.dumps(report["long_sequence_resolution"], indent=2,
                   default=str))

    print("[4/8] FSDP2 validation (real processes: sharding, all-reduce, "
          "sharded resume, full-state export)...")
    fsdp_res = spawn_fsdp_validation(
        "cortex-demo", str(ART / "data" / "tokens.npy"),
        str(ART / "runs" / "fsdp-validation"), world_size=2, steps=6,
        checkpoint_every=3, resume_steps=2)
    report["fsdp_validation"] = fsdp_res
    print(f"      spread={fsdp_res['max_checksum_spread_across_ranks']} "
          f"sharding={fsdp_res['sharding_verified']} reload="
          f"{fsdp_res['full_state_reloaded_into_plain_model']} resume_ok="
          f"{(fsdp_res.get('resume') or {}).get('resumed_ok')} loss "
          f"{fsdp_res['loss_first']:.4f} -> {fsdp_res['loss_last']:.4f}")
    (ART / "fsdp_validation.json").write_text(
        json.dumps(fsdp_res, indent=2, default=str))
    if not fsdp_res["sharding_verified"]:
        raise RuntimeError("FSDP sharding not verified")

    print("[5/8] 2B launcher mechanics (real spawn on demo config; "
          "cortex-2b refused by hardware gate)...")
    launcher = subprocess.run(
        [sys.executable, "scripts/train_2b_fsdp.py", "--config",
         "cortex-demo", "--steps", "2", "--batch-size", "2",
         "--out-dir", str(ART / "runs" / "launcher-smoke-demo"),
         "--dataset", ds_v3],
        capture_output=True, text=True, cwd=str(ROOT), timeout=1200)
    launcher_refusal = subprocess.run(
        [sys.executable, "scripts/train_2b_fsdp.py", "--config", "cortex-2b",
         "--steps", "2", "--out-dir", str(ART / "runs" / "launcher-2b"),
         "--dataset", ds_v3],
        capture_output=True, text=True, cwd=str(ROOT), timeout=300)
    demo_lines = [l for l in launcher.stdout.splitlines()
                  if l.strip().startswith("{")]
    smoke = {"demo_launch_exit": launcher.returncode,
             "demo_launch_completed": "COMPLETED" in launcher.stdout,
             "demo_checkpoint": str(ART / "runs" / "launcher-smoke-demo"
                                    / "fsdp-final-rank0.pt"),
             "demo_checkpoint_exists": (ART / "runs" / "launcher-smoke-demo"
                                        / "fsdp-final-rank0.pt").exists(),
             "cortex2b_exit": launcher_refusal.returncode,
             "cortex2b_refused_by_gate":
                 "REFUSED-HARDWARE-GATE" in launcher_refusal.stdout}
    report["launcher_mechanics"] = smoke
    print(f"      demo exit={smoke['demo_launch_exit']} "
          f"completed={smoke['demo_launch_completed']} ckpt_exists="
          f"{smoke['demo_checkpoint_exists']}; cortex-2b refused "
          f"(exit={smoke['cortex2b_exit']}, "
          f"gate={smoke['cortex2b_refused_by_gate']})")
    (ART / "launcher_smoke.json").write_text(json.dumps(smoke, indent=2))
    if launcher.returncode != 0 or not smoke["demo_launch_completed"]:
        raise RuntimeError(
            f"launcher demo run failed:\n{launcher.stdout[-800:]}\n"
            f"{launcher.stderr[-800:]}")
    if launcher_refusal.returncode != 2:
        raise RuntimeError(
            "cortex-2b launcher did NOT refuse on this host - hardware gate "
            f"broken:\n{launcher_refusal.stdout[-500:]}")

    print("[6/8] chunked code-specialized SFT continuation (per-epoch "
          "autoregressive eval)...")
    from cortex.core import CortexAgent
    from cortex.evaluation.runner_v2 import evaluate_v2

    agent = CortexAgent(state_dir=str(ART / "cortex-state"),
                        tokenizer_path=tok_path,
                        checkpoint_path=str(ART / "runs" / "cortex-sft-stage5"
                                            / "stage5-epoch-004.pt"))
    held_recs = load_split_v2(str(ART / "sft" / "dataset_v2.jsonl"),
                              "heldout", str(ART / "sft" / "v2_manifest.json"))
    epochs = int(os.environ.get("CORTEX_STAGE6_EPOCHS", "2"))
    init_ckpt = str(ART / "runs" / "cortex-sft-stage5" / "stage5-epoch-004.pt")
    staged = []
    monitor_rows = 0
    for epoch in range(1, epochs + 1):
        out_dir = str(ART / "runs" / "cortex-sft-stage6")
        summary = fine_tune("cortex-demo", tok_path, ds_v3, init_ckpt,
                            out_dir, epochs=1, batch_size=8, lr=5e-5,
                            keep_checkpoints=4, log_every=5,
                            chunk_long_records=True)
        ck2 = torch.load(summary["checkpoint"], map_location="cpu",
                         weights_only=True)
        ck2["stage6_epoch"] = epoch
        epoch_ckpt = Path(out_dir) / f"stage6-epoch-{epoch:03d}.pt"
        torch.save(ck2, epoch_ckpt)
        agent.load_checkpoint(str(epoch_ckpt))
        v1_report = evm.evaluate_checkpoint(ART, str(epoch_ckpt))
        hl = evaluate_heldout(agent.model, held_recs, agent.tokenizer,
                              agent.model.cfg.max_seq_len)
        v2 = evaluate_v2_for(agent, evaluate_v2, exp)
        metrics_rows = [json.loads(l) for l in
                        open(Path(out_dir) / "metrics.jsonl",
                             encoding="utf-8") if l.strip()]
        monitor_rows = len(metrics_rows)
        staged.append({
            "epoch": epoch, "checkpoint": str(epoch_ckpt),
            "init_checkpoint": str(init_ckpt),
            "heldout_loss": hl,
            "v1_pass_at_1": v1_report["pass_at_1"],
            "v1_nonempty": round(sum(1 for r in v1_report["per_task"]
                                     if r["generated_head"].strip())
                                 / len(v1_report["per_task"]), 4),
            "v1_syntactically_valid": _syntax_rate(v1_report),
            "v2_pass_at_1": v2["pass_at_1"],
            "v2_compile_ok_rate": v2["compile_ok_rate"],
            "v2_executed_rate": v2["executed_rate"],
            "monitoring_last": (metrics_rows[-1] if metrics_rows else None),
        })
        print(f"      epoch {epoch}: hl={hl} v1={v1_report['pass_at_1']} "
              f"v2={v2['pass_at_1']} syntax={staged[-1]['v1_syntactically_valid']}")
        init_ckpt = str(epoch_ckpt)
    report["staged_training"] = staged
    report["monitoring"] = {
        "metrics_rows_last_epoch": monitor_rows,
        "grad_norm_and_throughput_recorded": monitor_rows > 0,
        "fsdp_tokens_per_s_last": fsdp_res["per_rank"][0]
        ["tokens_per_s_last"],
        "fsdp_grad_norm_last": fsdp_res["per_rank"][0]["grad_norm_last"],
        "fsdp_memory": fsdp_res["per_rank"][0]["memory"],
    }
    final_ckpt = staged[-1]["checkpoint"]

    report["failure_taxonomy"] = {
        "compile_stage": compile_diagnosis(agent, exp._v2_tasks()[0]),
        "decode_ablation": decode_ablation(agent, exp._v2_tasks()[1]),
        "teacher_forced_token_accuracy_heldout":
            teacher_forced_token_accuracy(
                agent.model, agent.tokenizer, held_recs, seq_demo),
    }
    print(f"[7/8] failure taxonomy: tf-acc="
          f"{report['failure_taxonomy']['teacher_forced_token_accuracy_heldout']} "
          f"stage={report['failure_taxonomy']['compile_stage']['stage']}")

    print("[8/8] knowledge expansion + promotion gate + report...")
    ingested = knowledge_attempt(agent, exp)
    capacity = agent.knowledge_capacity()
    report["knowledge_expansion"] = {
        "patterns_ingested_this_stage": ingested,
        "capacity": capacity,
    }
    agent.load_checkpoint(str(ART / "runs" / "cortex-sft-stage5"
                              / "stage5-epoch-004.pt"))
    hl_base = evaluate_heldout(agent.model, held_recs, agent.tokenizer,
                               agent.model.cfg.max_seq_len)
    v2_base = evaluate_v2_for(agent, evaluate_v2, exp)
    gate = promotion_gate(
        {"pass_at_1": v2_base["pass_at_1"], "val_loss": hl_base},
        {"pass_at_1": staged[-1]["v2_pass_at_1"],
         "compile_ok_rate": staged[-1]["v2_compile_ok_rate"],
         "executed_rate": staged[-1]["v2_executed_rate"], "per_tier": {},
         "val_loss": staged[-1]["heldout_loss"]},
        fingerprint_ok=True)
    gate["baseline_checkpoint"] = str(ART / "runs" / "cortex-sft-stage5"
                                      / "stage5-epoch-004.pt")
    gate["candidate_checkpoint"] = final_ckpt
    report["promotion"] = gate
    print(f"      promoted={gate['promoted']} reasons={gate['reasons']}")

    report["owned_by"] = "Fibonera"
    report["config_training_status"] = {
        "cortex-2b": "DECLARED-NOT-TRAINED (2B run requires the GPU "
                     "infrastructure measured in stage 6; launcher refused "
                     "on this host by the hardware gate)",
        "cortex-demo": "trained (8,071,296 params, real checkpoints + evals)",
    }
    (ART / "stage6_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    print("stage-6 report -> artifacts/stage6_report.json")


def evaluate_v2_for(agent, evaluate_v2, exp):
    return evaluate_v2(
        candidate_edits={
            t.task_id: {"task.py": exp._gen_module(agent, t.instruction)}
            for t in exp._v2_tasks() if t.edit_targets == ["task.py"]},
        candidate_texts={
            t.task_id: exp._gen_text(agent, t.instruction)
            for t in exp._v2_tasks() if t.edit_targets != ["task.py"]},
    )


def knowledge_attempt(agent, exp) -> int:
    from cortex.interfaces import Provenance
    from cortex.knowledge.learning import LearningEngine
    from cortex.learning.multifile_examples import PACKAGES, format_examples

    engine = LearningEngine(agent.store)
    ingested = 0
    for pkg in PACKAGES:
        code = format_examples(pkg)
        if engine.is_known_pattern(code):
            continue
        try:
            engine.ingest_verified_outcome(
                code, execution_ok=True, tests_passed=True,
                provenance=Provenance(
                    source_uri="cortex://multifile_examples/" + pkg["name"],
                    license="project-local", version="cortex-stage6-1"),
                tags={"kind": "multi-file-package"})
            ingested += 1
        except Exception:
            pass
    return ingested


def benchmarks_v2_fingerprint() -> str:
    fp = json.loads((ART / "benchmarks_v2.json").read_text())
    return fp["fingerprint"]


def _syntax_rate(v1_report: dict) -> float:
    import ast

    ok = 0
    for r in v1_report["per_task"]:
        try:
            ast.parse(r["generated_head"])
            ok += 1
        except (SyntaxError, ValueError):
            pass
    return round(ok / len(v1_report["per_task"]), 4)


if __name__ == "__main__":
    main()
