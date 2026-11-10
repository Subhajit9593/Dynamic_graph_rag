"""Graph builders on a tiny synthetic corpus, with a fake embedder and a fake LLM judge (no GPU needed)."""

import hashlib
import json
import re

import numpy as np
import pytest

from dgrag.config import ResolutionConfig
from dgrag.resolve.builder import GlobalBuilder, IncrementalBuilder, functional_key
from dgrag.resolve.judge import Judge
from dgrag.resolve.names import name_compatible
from dgrag.store import ChunkInput, FaissManager, GraphView

DIM = 64


def fake_embed(texts):
    """Bag of hashed words: texts sharing words get high cosine similarity."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
    return out


class FakeAsk:
    def __init__(self, same=False, verdict="coexists"):
        self.same, self.verdict, self.calls = same, verdict, 0

    def __call__(self, messages, schema):
        self.calls += 1
        if "same" in schema["properties"]:
            return {"same": "yes" if self.same else "no"}
        return {"verdict": self.verdict}


def add_doc(store, url, date, chunks):
    """chunks: list of (entities, relations); entities = [(name, type, desc)], relations = [(src, pred, tgt)]."""
    doc, _ = store.add_document(f"body {url}", url=url, published_at=date)
    ids = store.add_chunks(doc, [ChunkInput(i, f"chunk {i} of {url}") for i in range(len(chunks))])
    for cid, (ents, rels) in zip(ids, chunks):
        ej = [{"name": n, "type": t, "desc": d} for n, t, d in ents]
        rj = [{"src": s, "pred": p, "tgt": t, "desc": ""} for s, p, t in rels]
        store.save_extraction(cid, prompt_version="v2", model="m", raw="{}", entities_json=json.dumps(ej),
                              relations_json=json.dumps(rj), n_entities=len(ej), n_relations=len(rj),
                              n_dropped_entities=0, n_dropped_relations=0, ok=True, finish_reason="stop",
                              prompt_tokens=0, completion_tokens=0, seconds=0)
    return doc


RES = ResolutionConfig(tau_high=0.9, tau_low=0.5, top_k=5)


@pytest.fixture
def fm(store, tmp_path):
    m = FaissManager(store, tmp_path / "faiss", DIM)
    m.open()
    return m


def _corpus(store):
    d1 = add_doc(store, "a", "2023-09-01", [(
        [("Sam Bankman-Fried", "PERSON", "founder of crypto exchange FTX"), ("FTX", "ORG", "crypto exchange")],
        [("Sam Bankman-Fried", "founded", "FTX")])])
    d2 = add_doc(store, "b", "2023-10-01", [(
        [("Bankman-Fried", "PERSON", "founder of crypto exchange FTX on trial"), ("FTX", "ORG", "bankrupt exchange"),
         ("Caroline Ellison", "PERSON", "former Alameda CEO")],
        [("Bankman-Fried", "founded", "FTX"), ("Caroline Ellison", "testified against", "Bankman-Fried")])])
    return [d1, d2]


def test_name_rules():
    assert name_compatible("bankman-fried", "sam bankman-fried")
    assert name_compatible("sbf", "sam bankman-fried")
    assert name_compatible("doj", "department of justice")
    assert name_compatible("jermey kelce", "jeremy kelce")
    assert not name_compatible("goodman", "dexus")
    assert not name_compatible("apple", "microsoft")
    assert functional_key("CEO of") == "ceo" and functional_key("defeated") is None


def test_incremental_merges_short_form_and_reinforces_edge(store, fm):
    b = IncrementalBuilder(store, fm, fake_embed, Judge(store, None), RES)
    b.add_documents(_corpus(store))
    names = sorted(r["name"] for r in store.iter_entities())
    assert names == ["Caroline Ellison", "FTX", "Sam Bankman-Fried"]   # "Bankman-Fried" merged
    sbf = store.find_entities_by_alias("Bankman-Fried")[0]
    assert sbf["name"] == "Sam Bankman-Fried" and sbf["mention_count"] == 2
    founded = store.find_relations(sbf["id"], predicate="founded")
    assert len(founded) == 1 and founded[0]["weight"] == 2.0           # same fact from 2 articles
    assert len(store.evidence_chunks(founded[0]["id"])) == 2           # pointers to both chunks
    assert b.stats["merge_alias"] == 1 and b.stats["merge_vector"] == 1
    for t in ("entities", "relations"):
        assert fm.is_consistent(t)
    assert GraphView.from_store(store).matches_store(store)


def test_judge_only_for_unrelated_names(store, fm):
    add_doc(store, "x", "2023-09-01", [([("Goodman", "ORG", "REIT property trust")], [])])
    add_doc(store, "y", "2023-09-02", [([("Dexus", "ORG", "REIT property trust")], [])])
    ask = FakeAsk(same=False)
    b = IncrementalBuilder(store, fm, fake_embed, Judge(store, ask), ResolutionConfig(0.3, 0.2, 5))
    b.add_documents(store.documents_with_status("loaded"))
    assert ask.calls == 1 and len(list(store.iter_entities())) == 2      # asked, and kept apart


def test_time_aware_supersede(store, fm):
    add_doc(store, "old", "2023-01-10", [([("Sam Altman", "PERSON", "OpenAI chief"), ("OpenAI", "ORG", "AI lab")],
                                          [("Sam Altman", "ceo of", "OpenAI")])])
    add_doc(store, "new", "2023-11-17", [([("Mira Murati", "PERSON", "interim chief"), ("OpenAI", "ORG", "AI lab")],
                                          [("Mira Murati", "ceo of", "OpenAI")])])
    ask = FakeAsk(verdict="supersedes")
    b = IncrementalBuilder(store, fm, fake_embed, Judge(store, ask), RES)
    b.add_documents(store.documents_with_status("loaded"))
    rows = {r["predicate"] + ":" + str(r["src_id"]): r for r in store.iter_relations(status=None)}
    statuses = sorted(r["status"] for r in rows.values())
    assert statuses == ["active", "superseded"] and b.stats["relations_superseded"] == 1
    old = next(r for r in rows.values() if r["status"] == "superseded")
    assert old["valid_to"] == "2023-11-17"
    assert len(GraphView.from_store(store).g.edges) == 1                 # only the current CEO edge is walked


def test_judge_cache_avoids_repeat_calls(store, fm):
    add_doc(store, "x", "2023-09-01", [([("Goodman", "ORG", "REIT property trust")], [])])
    add_doc(store, "y", "2023-09-02", [([("Dexus", "ORG", "REIT property trust")], [])])
    ask = FakeAsk(same=False)
    res = ResolutionConfig(0.3, 0.2, 5)
    IncrementalBuilder(store, fm, fake_embed, Judge(store, ask), res).add_documents(
        store.documents_with_status("loaded"))
    store.clear_graph()
    for t in ("entities", "relations"):
        fm[t].reset()
    IncrementalBuilder(store, fm, fake_embed, Judge(store, ask), res).add_documents(
        store.documents_with_status("extracted"))
    assert ask.calls == 1  # second build answered from judge_cache


def test_global_builder_matches_on_simple_corpus(store, fm):
    docs = _corpus(store)
    out = GlobalBuilder(store, fm, fake_embed, Judge(store, None), RES).build(docs)
    assert out["documents"] == 2
    assert sorted(r["name"] for r in store.iter_entities()) == ["Caroline Ellison", "FTX", "Sam Bankman-Fried"]
    sbf = store.find_entities_by_alias("bankman-fried")[0]
    assert store.find_relations(sbf["id"], predicate="founded")[0]["weight"] == 2.0
    for t in ("entities", "relations"):
        assert fm.is_consistent(t)
    assert all(store.get_document(d)["status"] == "resolved" for d in docs)
