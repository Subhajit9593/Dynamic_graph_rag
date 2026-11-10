"""Add ONE new article to an existing graph, end to end (the "dynamic" path, PLAN.md §6).

    store document -> chunk -> embed (SQLite + FAISS) -> LLM extraction (parallel per chunk)
    -> IncrementalBuilder.add_document (resolve entities against the existing graph, upsert edges, supersede)

Nothing else in the graph is rebuilt. Returns per-stage timings and what changed.
"""

from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from dgrag.config import Config
from dgrag.extract.extractor import ChunkJob, _call, _save
from dgrag.extract.prompts import build_schema
from dgrag.ingest.chunker import chunk_text, hf_token_counter
from dgrag.resolve.builder import IncrementalBuilder
from dgrag.resolve.judge import Judge, llm_ask
from dgrag.store import ChunkInput, FaissManager, SQLiteStore


@dataclass
class NewArticle:
    title: str
    body: str
    source: str | None = None
    published_at: str | None = None
    url: str | None = None
    category: str | None = None


def add_article(cfg: Config, store: SQLiteStore, fm: FaissManager, embedder: Any, client: Any,
                art: NewArticle, token_counter: Any = None) -> dict[str, Any]:
    t: dict[str, float] = {}
    t0 = time.perf_counter()
    doc_id, inserted = store.add_document(art.body, title=art.title, source=art.source, url=art.url,
                                          category=art.category, published_at=art.published_at)
    if not inserted:
        return {"doc_id": doc_id, "inserted": False, "message": "article already in the graph (same text or url)"}

    c = cfg.chunking
    count = token_counter or hf_token_counter(cfg.embedder.model_name)
    chunks = chunk_text(art.body, count, c.chunk_tokens, c.overlap_tokens, c.min_chunk_tokens)
    vecs = embedder.encode([ch.text for ch in chunks])
    with store.transaction():
        ids = store.add_chunks(doc_id, [ChunkInput(ch.idx, ch.text, ch.n_tokens, ch.char_start, ch.char_end, v)
                                        for ch, v in zip(chunks, vecs)])
        store.set_document_status(doc_id, "chunked")
    fm.chunks.add(ids, vecs)
    t["chunk_embed"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    schema = build_schema(cfg.extraction.max_entities, cfg.extraction.max_relations)
    jobs = [ChunkJob(cid, ch.text, art.title, art.source, art.published_at) for cid, ch in zip(ids, chunks)]
    stats: Counter = Counter()
    drops: Counter = Counter()
    with ThreadPoolExecutor(max_workers=min(len(jobs), 8) or 1) as pool:
        results = list(pool.map(lambda j: _call(client, j, schema, cfg.llm.max_tokens, cfg.llm.temperature,
                                                cfg.llm.seed), jobs))
    model = cfg.llm.model_path.name.replace("-00001-of-00002", "")
    for job, (obj, res, _) in zip(jobs, results):
        _save(store, job, obj, res, cfg.extraction.prompt_version, model, stats, drops)
    store.give_up_failed_extractions()
    store.mark_documents_extracted()
    t["llm_extract"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    before = {r["id"] for r in store.conn.execute("SELECT id FROM entities")}
    judge = Judge(store, llm_ask(client))
    b = IncrementalBuilder(store, fm, embedder.encode, judge, cfg.resolution)
    b.add_document(doc_id)
    fm.save()
    t["graph_update"] = time.perf_counter() - t2

    mentioned = [int(r[0]) for r in store.conn.execute(
        "SELECT DISTINCT m.entity_id FROM entity_mentions m JOIN chunks c ON c.id = m.chunk_id WHERE c.doc_id = ?",
        (doc_id,))]
    ents = store.get_entities(mentioned)
    new_rel = store.conn.execute(
        "SELECT r.id, s.name, r.predicate, t.name, r.status FROM relations r JOIN entities s ON s.id = r.src_id "
        "JOIN entities t ON t.id = r.tgt_id WHERE r.id IN (SELECT relation_id FROM relation_evidence e "
        "JOIN chunks c ON c.id = e.chunk_id WHERE c.doc_id = ?)", (doc_id,)).fetchall()
    superseded = store.conn.execute(
        "SELECT s.name, r.predicate, t.name, r.valid_to FROM relations r JOIN entities s ON s.id = r.src_id "
        "JOIN entities t ON t.id = r.tgt_id WHERE r.status = 'superseded' AND r.superseded_by IN "
        f"({','.join('?' * len(new_rel)) or 'NULL'})", [r[0] for r in new_rel]).fetchall()
    return {
        "doc_id": doc_id, "inserted": True, "chunks": len(ids),
        "seconds": {k: round(v, 2) for k, v in t.items()} | {"total": round(time.perf_counter() - t0, 2)},
        "entities_linked_to_existing": [e["name"] for e in ents if e["id"] in before],
        "entities_new": [e["name"] for e in ents if e["id"] not in before],
        "relations": [{"src": r[1], "pred": r[2], "tgt": r[3], "status": r[4]} for r in new_rel],
        "superseded_old_facts": [{"src": r[0], "pred": r[1], "tgt": r[2], "valid_to": r[3]} for r in superseded],
        "build_stats": dict(b.stats), **judge.stats(),
    }
