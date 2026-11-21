import sysconfig
from pathlib import Path

import pytest

from cortex.interfaces import KnowledgeItem, Provenance
from cortex.knowledge.capacity import measure_capacity
from cortex.knowledge.graph import KnowledgeGraph
from cortex.knowledge.indexer import index_code_root
from cortex.knowledge.learning import LearningEngine, UnverifiedOutcomeError
from cortex.knowledge.vector_store import EMBED_DIM, VectorStore


def _prov(uri="file:///tmp/fake.py", sha="deadbeef"):
    return Provenance(source_uri=uri, license="PSF", version="3.12.0",
                      content_sha256=sha, collected_at="now", collector="test")


def _mkitem(store, text, kind="function", iid=None):
    item = KnowledgeItem(
        item_id=iid or "k:" + text[:12], kind=kind, language="python", text=text,
        provenance=_prov(), tags={})
    store.add(item)
    return item


def test_vector_store_roundtrip_and_search(tmp_path):
    store = VectorStore(str(tmp_path / "kv.db"))
    items = [
        _mkitem(store, "serialize json encoder dumps loads object hooks"),
        _mkitem(store, "http server handles GET POST requests routing"),
        _mkitem(store, "sqlite connect cursor execute transactions commit"),
        _mkitem(store, "zip file archive compression deflate write read"),
    ]
    assert store.counts()["stored"] == 4
    assert store.counts()["indexed"] == 4
    hits = store.search("json encode decode objects", k=1)
    assert hits and hits[0]["id"] == items[0].item_id
    store.mark_verified(items[0].item_id, "unit-test")
    assert store.get(items[0].item_id)["verified"] == 1


def test_knowledge_graph_edges(tmp_path):
    g = KnowledgeGraph(str(tmp_path / "kg.db"))
    g.upsert_node("m1", "module", "json/encoder.py", _prov())
    g.upsert_node("m2", "module", "json/decoder.py", _prov())
    g.upsert_node("pymod:json", "external_module", "json", _prov())
    g.upsert_edge("m1", "pymod:json", "imports", "3.12")
    g.upsert_edge("m2", "pymod:json", "imports", "3.12")
    g.commit()
    assert set(n["node"] for n in g.neighbors("pymod:json", direction="in")) == {"m1", "m2"}
    assert g.stats()["nodes"] == 3
    assert g.stats()["edges"] == 2


def test_indexer_indexes_real_stdlib(tmp_path):
    store = VectorStore(str(tmp_path / "kv.db"))
    graph = KnowledgeGraph(str(tmp_path / "kg.db"))
    counts = index_code_root(str(Path(sysconfig.get_paths()["stdlib"]) / "json"), store, graph,
                             license_tag="PSF")
    assert counts["files"] >= 3
    assert counts["modules"] >= 3
    assert counts["functions"] > 3
    assert counts["edges"] > 3
    rec = store.get(counts and [r[0] for r in store.iter_texts()][0])
    assert rec["provenance"]["license"] == "PSF"
    assert rec["provenance"]["content_sha256"]


def test_capacity_measurement_reports_real_numbers(tmp_path):
    store = VectorStore(str(tmp_path / "kv.db"))
    texts = [
        "hashlib blake2 digest secure hash function",
        "urllib request urlopen fetch http response",
        "json dump loads serialize deserialize encoder",
        "logging warning error handlers propagate level",
        "multiprocessing pool map workers processes spawn",
        "sqlite cursor execute select insert transactions",
        "pickle dump loads protocol serialization objects",
        "queue queue put get task done worker",
        "threading lock acquire release condition event",
        "pathlib path glob iterdir read text write",
    ]
    for i, t in enumerate(texts):
        _mkitem(store, t, iid=f"probe-item-{i}")
    cap = measure_capacity(store, probe_n=10, k=3)
    assert cap["stored"] == len(texts)
    assert cap["indexed"] == len(texts)
    assert 0.0 <= cap["retrieval_recall_at_3"] <= 1.0
    assert cap["retrievable"] <= cap["indexed"]


def test_learning_rejects_unverified_code(tmp_path):
    store = VectorStore(str(tmp_path / "kv.db"))
    eng = LearningEngine(store)
    with pytest.raises(UnverifiedOutcomeError):
        eng.ingest_verified_outcome("x = 1", execution_ok=False, tests_passed=False,
                                    provenance=_prov(sha="aa"))
    with pytest.raises(UnverifiedOutcomeError):
        eng.ingest_verified_outcome("x = 1", execution_ok=True, tests_passed=False,
                                    provenance=_prov(sha="ab"))


def test_learning_accepts_verified_code_and_dedups(tmp_path):
    store = VectorStore(str(tmp_path / "kv.db"))
    eng = LearningEngine(store)
    code = "def add(a, b):\n    return a + b\n"
    item = eng.ingest_verified_outcome(code, execution_ok=True, tests_passed=True,
                                       provenance=_prov(sha="cc"))
    assert eng.is_known_pattern(code)
    assert store.get(item.item_id)["verified"] == 1
    again = eng.ingest_verified_outcome(code, execution_ok=True, tests_passed=True,
                                        provenance=_prov(sha="cc"))
    assert again.item_id == item.item_id
