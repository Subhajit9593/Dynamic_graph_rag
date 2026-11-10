"""FAISS manager (incremental add/remove, persistence, reconcile with SQLite) and graph view sync."""

import numpy as np

from dgrag.store import ChunkInput, FaissManager, GraphView, VectorIndex

from .conftest import DIM, rand_vec


def _doc_with_chunks(store, rng, n, url="x"):
    doc, _ = store.add_document(f"body {url}", url=url)
    vecs = rand_vec(rng, n)
    ids = store.add_chunks(doc, [ChunkInput(i, f"c{i}", embedding=vecs[i]) for i in range(n)])
    return doc, ids, vecs


# ---------------------------------------------------------------- VectorIndex
def test_vector_index_add_search_remove(rng):
    idx = VectorIndex("t", DIM)
    vecs = rand_vec(rng, 5)
    idx.add([10, 20, 30, 40, 50], vecs)
    hits = idx.search(vecs[2], k=3)[0]
    assert hits[0][0] == 30 and abs(hits[0][1] - 1.0) < 1e-5  # exact self-match, cosine = 1
    assert idx.remove([30]) == 1
    assert 30 not in idx and idx.ntotal == 4
    assert all(i != 30 for i, _ in idx.search(vecs[2], k=4)[0])


def test_vector_index_upsert_keeps_one_vector_per_id(rng):
    idx = VectorIndex("t", DIM)
    v1, v2 = rand_vec(rng, 2)
    idx.add([7], v1[None])
    idx.add([7], v2[None])
    assert idx.ntotal == 1
    np.testing.assert_allclose(idx.reconstruct(7), v2, atol=1e-6)


def test_vector_index_normalises_input(rng):
    idx = VectorIndex("t", DIM)
    v = rand_vec(rng)
    idx.add([1], (v * 42.0)[None])
    assert abs(idx.search(v, 1)[0][0][1] - 1.0) < 1e-5


def test_vector_index_save_load(tmp_path, rng):
    p = tmp_path / "x.faiss"
    a = VectorIndex("t", DIM, p)
    a.add([3, 4], rand_vec(rng, 2))
    a.save()
    b = VectorIndex("t", DIM, p)
    assert b.load() and b.ids() == {3, 4}


def test_empty_index_search(rng):
    assert VectorIndex("t", DIM).search(rand_vec(rng, 2), k=5) == [[], []]


# ---------------------------------------------------------------- FaissManager <-> SQLite
def test_reconcile_from_scratch_equals_rebuild(store, faiss_mgr, rng):
    _, ids, vecs = _doc_with_chunks(store, rng, 6)
    rep = faiss_mgr.open()  # no files on disk -> everything comes from SQLite blobs
    assert rep["chunks"] == {"loaded_from_disk": 0, "added": 6, "removed": 0, "ntotal": 6}
    assert faiss_mgr.is_consistent("chunks")
    hits = faiss_mgr.chunks.search(vecs[4], k=1)[0]
    assert hits[0][0] == ids[4]


def test_reconcile_after_crash(store, tmp_path, rng):
    """Simulate: index saved, then SQLite changed (new doc committed, old doc deleted) before FAISS was saved."""
    doc_a, ids_a, _ = _doc_with_chunks(store, rng, 3, url="a")
    fm = FaissManager(store, tmp_path / "faiss", DIM)
    fm.open()
    fm.save()

    _, ids_b, _ = _doc_with_chunks(store, rng, 2, url="b")  # committed to SQLite, never reached FAISS
    store.delete_document(doc_a)                             # deleted in SQLite, still in FAISS file

    fm2 = FaissManager(store, tmp_path / "faiss", DIM)
    rep = fm2.open()
    assert rep["chunks"]["loaded_from_disk"] == 1
    assert rep["chunks"]["added"] == 2 and rep["chunks"]["removed"] == 3
    assert fm2.chunks.ids() == set(ids_b)
    assert fm2.is_consistent("chunks")


def test_corrupt_index_file_triggers_rebuild(store, tmp_path, rng):
    _doc_with_chunks(store, rng, 4)
    (tmp_path / "faiss").mkdir()
    (tmp_path / "faiss" / "chunks.faiss").write_bytes(b"not a faiss index")
    fm = FaissManager(store, tmp_path / "faiss", DIM)
    rep = fm.open()
    assert rep["chunks"]["loaded_from_disk"] == 0 and fm.is_consistent("chunks")


def test_entity_index_incremental(store, faiss_mgr, rng):
    faiss_mgr.open()
    v = rand_vec(rng, 2)
    e1 = store.add_entity("OpenAI", "ORG", embedding=v[0])
    faiss_mgr.entities.add([e1], v[0][None])
    e2 = store.add_entity("Microsoft", "ORG", embedding=v[1])
    faiss_mgr.entities.add([e2], v[1][None])
    assert faiss_mgr.is_consistent("entities")
    assert faiss_mgr.entities.search(v[1], 1)[0][0][0] == e2


# ---------------------------------------------------------------- GraphView
def test_graph_view_tracks_active_edges(store):
    a, b, c = (store.add_entity(n, "PERSON") for n in ("A", "B", "C"))
    org = store.add_entity("Org", "ORG")
    r1 = store.add_relation(a, org, "CEO of", observed_at="2023-01-01")
    gv = GraphView.from_store(store)
    assert gv.g.has_edge(a, org, key=r1) and gv.matches_store(store)

    r2 = store.add_relation(b, org, "CEO of", observed_at="2023-11-17")
    store.supersede_relation(r1, r2, "2023-11-17")
    gv.upsert_relation(store.get_relation(r2))
    gv.upsert_relation(store.get_relation(r1))  # now superseded -> removed from active view
    assert not gv.g.has_edge(a, org, key=r1) and gv.g.has_edge(b, org, key=r2)
    assert gv.matches_store(store)

    full = GraphView.from_store(store, active_only=False)
    assert full.g.number_of_edges() == 2
    assert gv.stats()["nodes"] == 4 and gv.stats()["isolated_nodes"] == 2  # a and c
