import json

import pytest

from cortex.evaluation import benchmarks
from cortex.evaluation.tasks_v2 import TASKS_V2
from cortex.training.dataset_v2 import (
    _canonical_fingerprint,
    build_dataset_v2,
    load_split_v2,
)
from cortex.training.promotion import apply_promotion, promotion_gate


def _pair(code, name="fn"):
    return {
        "name": name,
        "module": "m",
        "instruction": "Implement function " + name,
        "code": code,
        "provenance": {"license": "Python-2.0 / PSF",
                       "source_uri": "file:///x", "version": "3.12"},
    }


def test_dataset_v2_three_way_split_and_fingerprints(tmp_path):
    pairs = [_pair(f"def f{i}():\n    return {i}\n", name=f"f{i}")
             for i in range(12)]
    out = tmp_path / "ds.jsonl"
    manifest = build_dataset_v2(pairs, str(out),
                                manifest_path=str(tmp_path / "m.json"))
    counts = manifest["counts"]
    assert counts["train"] + counts["val"] + counts["heldout"] == 12
    assert counts["val"] >= 1 and counts["heldout"] >= 1
    roles = {json.loads(l)["split"] for l in out.read_text().splitlines()}
    assert roles == {"train", "val", "heldout"}
    assert len(manifest["fingerprints"]["heldout"]) == 64
    m2 = build_dataset_v2(pairs, str(tmp_path / "ds2.jsonl"),
                          manifest_path=str(tmp_path / "m2.json"))
    assert m2["fingerprints"] == manifest["fingerprints"]


def test_heldout_immutability_enforced_on_load(tmp_path):
    pairs = [_pair(f"def f{i}():\n    return {i}\n", name=f"f{i}")
             for i in range(10)]
    out = tmp_path / "ds.jsonl"
    build_dataset_v2(pairs, str(out), manifest_path=str(tmp_path / "m.json"))
    records = [json.loads(l) for l in out.read_text().splitlines()]
    for r in records:
        if r["split"] == "heldout":
            r["code"] = r["code"] + "\n# tampered\n"
            r["split"] = "train"
    with open(out, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    with pytest.raises(RuntimeError):
        load_split_v2(str(out), "heldout", str(tmp_path / "m.json"))


def test_v2_fingerprint_separate_and_stable():
    v1 = benchmarks.fingerprint_tasks("v1")
    v2 = benchmarks.fingerprint_tasks("v2")
    assert v1 != v2
    assert v2 == benchmarks.fingerprint_tasks("v2")


def test_v2_regression_detection_scoped_by_suite(tmp_path):
    benchmarks.record_baseline(1.0, str(tmp_path / "bl.json"))
    hist = tmp_path / "hist.jsonl"
    good = {"pass_at_1": 1.0, "n_tasks": len(TASKS_V2),
            "per_task": [{"subject": "model", "passed": True}]}
    benchmarks.append_history(good, str(hist), suite="v2")
    bad = {"pass_at_1": 0.4, "n_tasks": len(TASKS_V2),
           "per_task": [{"subject": "model", "passed": False}]}
    res = benchmarks.detect_regression(bad, str(hist),
                                       str(tmp_path / "bl.json"), suite="v2")
    assert res["regressed"] is True
    assert res["suite"] == "v2"


def test_promotion_gate():
    base = {"pass_at_1": 0.0, "val_loss": 3.0}
    no_improve = {"pass_at_1": 0.0, "val_loss": 2.6}
    gate = promotion_gate(base, no_improve)
    assert gate["promoted"] is False
    assert any("pass@1" in r for r in gate["reasons"])
    improved = {"pass_at_1": 0.5, "val_loss": 2.6}
    gate2 = promotion_gate(base, improved)
    assert gate2["promoted"] is True
    ledger: dict = {}
    assert apply_promotion(gate2, "/ck/sft.pt", ledger, "cortex-demo") is True
    assert ledger["cortex-demo"]["promoted_checkpoint"] == "/ck/sft.pt"
    assert apply_promotion(gate, "/ck/x.pt", ledger, "cortex-demo") is False
    assert "promoted_checkpoint" not in ledger
    gate3 = promotion_gate(base, improved, fingerprint_ok=False)
    assert gate3["promoted"] is False
