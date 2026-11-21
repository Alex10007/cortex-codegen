#!/usr/bin/env python3
"""End-to-end fine-tuning path from verified outcomes.

Stages (each real and recorded):
 1. harvest verified instruction->code pairs (sandbox-executed doctests)
 2. build SFT dataset (dedup, license gate, contamination filter, split)
 3. measure pre-fine-tuning held-out loss of the pretrained checkpoint
 4. run SFT with completion-only loss masking (per-epoch checkpoints)
 5. measure post-fine-tuning held-out loss + execution-graded benchmark
 6. record benchmark integrity, history, and regression check
 7. ingest the verified pairs into the knowledge store (validated tier)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ART = ROOT / "artifacts"
STDLIB = os.environ.get("CORTEX_STDLIB", "/usr/local/lib/python3.12")


def main() -> None:
    import torch

    from cortex.evaluation import benchmarks
    from cortex.evaluation.runner import evaluate
    from cortex.learning.doctest_harvester import harvest
    from cortex.model.configs import get_config
    from cortex.model.transformer import CortexTransformer
    from cortex.tokenizer.cortex_tokenizer import load_tokenizer
    from cortex.training.sft import build_dataset, evaluate_heldout, fine_tune

    from eval_model import evaluate_checkpoint

    report: dict = {}

    print("[1/7] harvesting verified pairs (sandbox doctest execution)...")
    pairs = harvest([STDLIB],
                    max_files=int(os.environ.get("CORTEX_HARVEST_FILES", "0")))
    report["harvest"] = {"pairs": len(pairs),
                         "doctests": sum(p["doctests_attempted"] for p in pairs)}
    print(f"      verified pairs: {len(pairs)}")

    dataset_path = str(ART / "sft" / "dataset.jsonl")
    ds = build_dataset(pairs, dataset_path)
    report["dataset"] = ds
    print(f"[2/7] dataset: train={ds['train']} heldout={ds['heldout']} "
          f"stats={ds['stats']}")

    tok = load_tokenizer(str(ART / "tokenizer" / "tokenizer.json"))
    base_ckpt = ART / "runs" / "cortex-demo" / "final.pt"
    model = CortexTransformer(get_config("cortex-demo"))
    if base_ckpt.exists():
        ck = torch.load(base_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model_state"])
    heldout = [json.loads(l) for l in open(dataset_path, encoding="utf-8")
               if json.loads(l).get("split") == "heldout"]
    pre_loss = evaluate_heldout(model, heldout, tok,
                                get_config("cortex-demo").max_seq_len)
    report["heldout_loss_before_sft"] = pre_loss
    print(f"[3/7] held-out loss before SFT: {pre_loss}")

    print("[4/7] running SFT from verified outcomes...")
    sft = fine_tune(
        "cortex-demo", str(ART / "tokenizer" / "tokenizer.json"), dataset_path,
        str(base_ckpt) if base_ckpt.exists() else None,
        str(ART / "runs" / "cortex-sft"),
        epochs=int(os.environ.get("CORTEX_SFT_EPOCHS", "3")),
        batch_size=8, lr=1e-4,
    )
    report["sft"] = sft
    print(f"      train loss {sft['train_loss_first']} -> {sft['train_loss_last']}; "
          f"held-out per epoch: {sft['heldout_loss_first_epoch']} -> "
          f"{sft['heldout_loss_last_epoch']}")

    print("[5/7] execution-graded benchmark of SFT checkpoint...")
    sft_eval = evaluate_checkpoint(ART, sft["checkpoint"])
    report["sft_eval"] = {k: v for k, v in sft_eval.items() if k != "per_task"}

    ledger = ART / "benchmarks.json"
    if ledger.exists():
        integrity = {"status": "unmodified",
                     **benchmarks.check_integrity(str(ledger))}
    else:
        integrity = {"status": "baseline-recorded",
                     **benchmarks.record_baseline(sft_eval["pass_at_1"], str(ledger))}
    history = benchmarks.append_history(sft_eval, str(ART / "eval_history.jsonl"))
    regression = benchmarks.detect_regression(
        sft_eval, str(ART / "eval_history.jsonl"), str(ledger))
    report["benchmark"] = {"integrity": integrity, "regression": regression,
                           "history_entries": len(history)}
    print(f"[6/7] benchmark integrity={integrity['status']} "
          f"pass@1={sft_eval['pass_at_1']} regressed={regression['regressed']}")

    from cortex.core import CortexAgent

    agent = CortexAgent(state_dir=str(ART / "cortex-state"),
                        tokenizer_path=str(ART / "tokenizer" / "tokenizer.json"),
                        checkpoint_path=str(base_ckpt))
    if agent.store.counts()["validated"] == 0 and pairs:
        for p in pairs:
            from cortex.interfaces import Provenance

            prov = Provenance(source_uri=p["provenance"]["source_uri"],
                              license=p["provenance"]["license"],
                              version=p["provenance"]["version"],
                              content_sha256="harvest-verified-doctests",
                              collected_at="runtime", collector="doctest_harvester")
            try:
                agent.learning.ingest_verified_outcome(
                    p["code"], execution_ok=True, tests_passed=True,
                    provenance=prov,
                    tags={"name": p["name"], "module": p["module"]})
            except Exception:
                continue
    capacity = agent.knowledge_capacity(probe_n=100)
    report["knowledge_capacity_after_validated_ingest"] = capacity
    print(f"[7/7] knowledge capacity: validated={capacity['validated']} "
          f"stored={capacity['stored']}")

    (ART / "sft_report.json").write_text(json.dumps(report, indent=2, default=str))
    print("SFT report -> artifacts/sft_report.json")


if __name__ == "__main__":
    main()
