#!/usr/bin/env python3
"""Build + verify everything that can be built on this host, for real.

Stages (each idempotent, each records artifacts):
 1. hardware profile              -> artifacts/system_profile.json
 2. tokenizer training            -> artifacts/tokenizer/
 3. corpus packing                -> artifacts/data/tokens.npy
 4. demo model training (real)    -> artifacts/runs/cortex-demo/
 5. knowledge ingestion           -> artifacts/cortex-state/knowledge.db(+graph)
 6. capacity measurement          -> artifacts/capacity.json
 7. evaluation (harness sanity)   -> artifacts/eval_harness.json
 8. master report                 -> artifacts/report.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ART = ROOT / "artifacts"
STDLIB = "/usr/local/lib/python3.12"


def main() -> None:
    ART.mkdir(exist_ok=True)
    report: dict = {}

    from cortex.hardware.profiler import can_train_locally, profile_hardware

    profile = profile_hardware()
    (ART / "system_profile.json").write_text(json.dumps(profile.__dict__, indent=2))
    report["hardware"] = profile.__dict__
    ok, reason = can_train_locally(profile, 2_056_012_800)
    report["hardware"]["can_train_2b_locally"] = ok
    report["hardware"]["reason"] = reason
    print("[1/8] hardware profiled:", profile.summary())
    print("      2B trainable locally:", ok, "-", reason)

    tok_path = ART / "tokenizer" / "tokenizer.json"
    if not tok_path.exists():
        from cortex.tokenizer.cortex_tokenizer import train_tokenizer

        train_tokenizer([STDLIB], str(ART / "tokenizer"), vocab_size=32768)
    print("[2/8] tokenizer ready:", tok_path)

    data_bin = ART / "data" / "tokens.npy"
    if not data_bin.exists():
        from cortex.training.engine import pack_corpus

        path, n_tokens, n_files = pack_corpus(str(tok_path), [STDLIB], str(data_bin))
        report["corpus"] = {"tokens": n_tokens, "files": n_files, "path": path}
    print("[3/8] corpus packed:", data_bin.exists())

    from cortex.model.configs import MODEL_STATUS, declared_configs

    report["configs"] = {k: {"params": v, "status": MODEL_STATUS[k]["status"]}
                         for k, v in declared_configs().items()}
    run_dir = ART / "runs" / "cortex-demo"
    ckpt = run_dir / "final.pt"
    if not ckpt.exists():
        from cortex.training.engine import train

        summary = train("cortex-demo", str(tok_path), str(data_bin), str(run_dir),
                        steps=int(os.environ.get("CORTEX_TRAIN_STEPS", "600")),
                        batch_size=int(os.environ.get("CORTEX_TRAIN_BATCH", "16")))
        report["training"] = summary
    print("[4/8] demo training done:", ckpt.exists(), MODEL_STATUS["cortex-demo"])

    from cortex.core import CortexAgent

    agent = CortexAgent(state_dir=str(ART / "cortex-state"),
                        tokenizer_path=str(tok_path), checkpoint_path=str(ckpt))
    counts = agent.store.counts()
    if counts["stored"] == 0:
        ingest = agent.ingest_corpus([STDLIB])
        report["knowledge_ingest"] = ingest
    print("[5/8] knowledge ingested:", agent.store.counts())

    capacity = agent.knowledge_capacity(probe_n=200)
    (ART / "capacity.json").write_text(json.dumps(capacity, indent=2))
    report["capacity"] = capacity
    print("[6/8] capacity measured:", capacity)

    harness = agent.eval_run(reference=True)
    (ART / "eval_harness.json").write_text(json.dumps(harness, indent=2))
    report["eval_harness_sanity"] = {"pass_at_1": harness["pass_at_1"],
                                     "n_tasks": harness["n_tasks"]}
    print("[7/8] harness sanity pass@1:", harness["pass_at_1"])

    reports = agent.service_reports()
    report["services"] = reports
    (ART / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print("[8/8] master report written to artifacts/report.json")


if __name__ == "__main__":
    main()
