"""SQLite storage layer: idempotency, atomicity, alias lookup, temporal relation primitives."""

import numpy as np
import pytest

from dgrag.store import ChunkInput, SQLiteStore
from dgrag.store.schema import SCHEMA_VERSION
from dgrag.text_utils import normalize_name, normalize_predicate

from .conftest import DIM, rand_vec


# ---------------------------------------------------------------- text utils
@pytest.mark.parametrize(
    "raw, norm",
    [
        ("Apple Inc.", "apple"),
        ("The OpenAI", "openai"),
        ("U.S.", "us"),
        ("Apple's", "apple"),
        ("  Microsoft   Corporation ", "microsoft"),
        ("Société Générale", "societe generale"),
        ("The", "the"),  # never strip to empty
    ],
)
def test_normalize_name(raw, norm):
    assert normalize_name(raw) == norm


def test_normalize_predicate():
    assert normalize_predicate("Is CEO of") == normalize_predicate("is_ceo_of") == "is_ceo_of"


# ---------------------------------------------------------------- schema / meta
def test_schema_version_persisted(tmp_path):
    p = tmp_path / "db.sqlite"
    with SQLiteStore(p) as s:
        assert s.get_meta("schema_version") == str(SCHEMA_VERSION)
    with SQLiteStore(p) as s:  # reopening must not re-run / fail
        assert s.get_meta("schema_version") == str(SCHEMA_VERSION)


def test_migration_from_v1(tmp_path):
    """A v1 database (no extractions table) is upgraded in place, keeping its data."""
    p = tmp_path / "old.sqlite"
    with SQLiteStore(p) as s:
        s.add_document("old doc", url="u")
        s.conn.execute("DROP TABLE extractions")
        s.conn.execute("DROP TABLE judge_cache")
        s.conn.execute("ALTER TABLE relations DROP COLUMN embedding")
        s.set_meta("schema_version", "1")
    with SQLiteStore(p) as s:
        assert s.get_meta("schema_version") == str(SCHEMA_VERSION)
        assert s.stats()["documents"] == 1
        assert s.conn.execute("SELECT COUNT(*) FROM extractions").fetchone()[0] == 0
        assert s.conn.execute("SELECT COUNT(*) FROM judge_cache").fetchone()[0] == 0
        assert "embedding" in [r[1] for r in s.conn.execute("PRAGMA table_info(relations)")]


def test_extraction_cache_and_resume(store):
    doc, _ = store.add_document("body", url="x")
    store.set_document_status(doc, "chunked")
    c1, c2 = store.add_chunks(doc, [ChunkInput(0, "a"), ChunkInput(1, "b")])
    kw = dict(prompt_version="v1", model="m", raw="{}", entities_json="[]", relations_json="[]", n_entities=0,
              n_relations=0, n_dropped_entities=0, n_dropped_relations=0, finish_reason="stop",
              prompt_tokens=1, completion_tokens=1, seconds=0.1)
    assert store.chunks_pending_extraction("v1") == [c1, c2]
    store.save_extraction(c1, ok=False, **kw)            # failed attempt -> still pending
    assert store.chunks_pending_extraction("v1") == [c1, c2]
    store.save_extraction(c1, ok=True, **kw)             # retry succeeds -> overwritten
    assert store.chunks_pending_extraction("v1") == [c2]
    assert store.chunks_pending_extraction("v2") == [c1, c2]  # new prompt version -> redo all
    assert store.mark_documents_extracted() == 0
    store.save_extraction(c2, ok=True, **kw)
    assert store.mark_documents_extracted() == 1 and store.get_document(doc)["status"] == "extracted"


def test_wal_and_foreign_keys(store):
    assert store.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


# ---------------------------------------------------------------- documents
def test_add_document_is_idempotent(store):
    d1, new1 = store.add_document("Body text.", title="A", url="http://a")
    d2, new2 = store.add_document("Body   TEXT.", title="A again")  # same content after normalisation
    d3, new3 = store.add_document("Different body", url="http://a")  # same url
    assert new1 and not new2 and not new3
    assert d1 == d2 == d3
    assert store.stats()["documents"] == 1


def test_transaction_rolls_back_on_error(store):
    with pytest.raises(RuntimeError):
        with store.transaction():
            store.add_document("doc one", url="u1")
            store.add_document("doc two", url="u2")
            raise RuntimeError("simulated crash mid-article")
    assert store.stats()["documents"] == 0


def test_nested_transaction_savepoint(store):
    with store.transaction():
        store.add_document("outer", url="o")
        with pytest.raises(ValueError):
            with store.transaction():
                store.add_document("inner", url="i")
                raise ValueError
        # inner rolled back, outer continues
    urls = {r["url"] for r in store.conn.execute("SELECT url FROM documents")}
    assert urls == {"o"}


def test_delete_document_cascades(store, rng):
    doc, _ = store.add_document("body", url="x")
    (cid,) = store.add_chunks(doc, [ChunkInput(0, "chunk", embedding=rand_vec(rng))])
    e = store.add_entity("OpenAI", "ORG")
    store.add_mention(e, cid)
    store.delete_document(doc)
    assert store.stats()["chunks"] == 0
    assert store.conn.execute("SELECT COUNT(*) FROM entity_mentions").fetchone()[0] == 0
    assert store.get_entity(e) is not None  # entities outlive the docs that mentioned them


# ---------------------------------------------------------------- chunks & embeddings
def test_chunk_embedding_roundtrip(store, rng):
    doc, _ = store.add_document("body", url="x")
    vecs = rand_vec(rng, 3)
    ids = store.add_chunks(doc, [ChunkInput(i, f"c{i}", embedding=vecs[i]) for i in range(3)])
    out_ids, out_vecs = store.load_embeddings("chunks", dim=DIM)
    assert out_ids.tolist() == ids
    np.testing.assert_array_equal(out_vecs, vecs)
    assert [r["text"] for r in store.get_chunks(reversed(ids))] == ["c2", "c1", "c0"]  # order preserved


def test_duplicate_chunk_idx_rejected_atomically(store):
    doc, _ = store.add_document("body", url="x")
    with pytest.raises(Exception):
        store.add_chunks(doc, [ChunkInput(0, "a"), ChunkInput(0, "dup")])
    assert store.stats()["chunks"] == 0


# ---------------------------------------------------------------- entities
def test_alias_lookup(store):
    e = store.add_entity("Apple Inc.", "ORG", "maker of iPhone")
    store.add_alias(e, "AAPL")
    assert [r["id"] for r in store.find_entities_by_alias("apple")] == [e]
    assert [r["id"] for r in store.find_entities_by_alias("Apple Inc")] == [e]
    assert [r["id"] for r in store.find_entities_by_alias("aapl")] == [e]
    assert store.find_entities_by_alias("apple", type_="PERSON") == []
    assert set(store.aliases_of(e)) == {"apple", "aapl"}


def test_mentions_count_once_per_chunk(store):
    doc, _ = store.add_document("body", url="x")
    c1, c2 = store.add_chunks(doc, [ChunkInput(0, "a"), ChunkInput(1, "b")])
    e = store.add_entity("OpenAI", "ORG")
    store.update_entity(e, dirty=False)
    assert store.add_mention(e, c1) and not store.add_mention(e, c1) and store.add_mention(e, c2)
    row = store.get_entity(e)
    assert row["mention_count"] == 2 and row["dirty"] == 1


# ---------------------------------------------------------------- relations
def test_relation_reinforce_and_supersede(store):
    altman = store.add_entity("Sam Altman", "PERSON")
    murati = store.add_entity("Mira Murati", "PERSON")
    openai = store.add_entity("OpenAI", "ORG")

    r1 = store.add_relation(altman, openai, "CEO of", observed_at="2023-09-01")
    store.reinforce_relation(r1, observed_at="2023-06-01")
    store.reinforce_relation(r1, observed_at="2023-10-01")
    row = store.get_relation(r1)
    assert row["weight"] == 3.0
    assert (row["first_seen"], row["last_seen"], row["valid_from"]) == ("2023-06-01", "2023-10-01", "2023-06-01")

    # lookup by normalised predicate
    assert [r["id"] for r in store.find_relations(altman, predicate="ceo_of")] == [r1]

    r2 = store.add_relation(murati, openai, "CEO of", observed_at="2023-11-17")
    store.supersede_relation(r1, r2, valid_to="2023-11-17")
    old = store.get_relation(r1)
    assert old["status"] == "superseded" and old["valid_to"] == "2023-11-17" and old["superseded_by"] == r2
    assert store.find_relations(altman) == []                       # no longer active
    assert len(store.find_relations(altman, status=None)) == 1      # but history kept
    s = store.stats()
    assert s["relations_active"] == 1 and s["relations_superseded"] == 1


def test_evidence_links(store):
    doc, _ = store.add_document("body", url="x")
    (c,) = store.add_chunks(doc, [ChunkInput(0, "a")])
    a, b = store.add_entity("A", "ORG"), store.add_entity("B", "ORG")
    r = store.add_relation(a, b, "partners with")
    assert store.add_evidence(r, c) and not store.add_evidence(r, c)
    assert store.evidence_chunks(r) == [c]


def test_invalid_status_rejected(store):
    a, b = store.add_entity("A", "ORG"), store.add_entity("B", "ORG")
    r = store.add_relation(a, b, "x")
    with pytest.raises(Exception):
        store.set_relation_status(r, "deleted")
