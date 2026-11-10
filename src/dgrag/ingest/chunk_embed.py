"""Stage: documents(status=loaded) -> chunks with embeddings in SQLite + FAISS -> status=chunked.

Order of writes per batch (PLAN.md D5): SQLite commit first, then FAISS add, FAISS saved at the end.
A crash in between is healed by FaissManager.reconcile() on the next start.
"""

from __future__ import annotations

import logging
import time

from dgrag.config import Config
from dgrag.embedder import Embedder
from dgrag.ingest.chunker import chunk_text, hf_token_counter
from dgrag.store import ChunkInput, FaissManager, SQLiteStore

log = logging.getLogger(__name__)


def chunk_and_embed(cfg: Config, store: SQLiteStore, limit: int | None = None, docs_per_batch: int = 32) -> dict:
    doc_ids = store.documents_with_status("loaded")
    if limit:
        doc_ids = doc_ids[:limit]
    if not doc_ids:
        log.info("No documents with status 'loaded'")
        return {"documents": 0, "chunks": 0}

    c = cfg.chunking
    count = hf_token_counter(cfg.embedder.model_name)
    emb = Embedder(cfg.embedder.model_name, cfg.embedder.device, cfg.embedder.batch_size)
    if emb.dim != cfg.embedder.dim:
        raise ValueError(f"Embedder dim {emb.dim} != config embedder.dim {cfg.embedder.dim}")
    fm = FaissManager(store, cfg.resolve(cfg.paths.faiss_dir), cfg.embedder.dim)
    fm.open()
    store.set_meta("embedder_model", cfg.embedder.model_name)

    t0 = time.perf_counter()
    n_chunks = 0
    for b in range(0, len(doc_ids), docs_per_batch):
        batch = doc_ids[b : b + docs_per_batch]
        tb = time.perf_counter()
        per_doc = []
        for did in batch:
            doc = store.get_document(did)
            per_doc.append((did, chunk_text(doc["body"], count, c.chunk_tokens, c.overlap_tokens,
                                            c.min_chunk_tokens)))
        texts = [ch.text for _, chunks in per_doc for ch in chunks]
        vecs = emb.encode(texts)  # one GPU call per batch of documents

        new_ids, k = [], 0
        with store.transaction():
            for did, chunks in per_doc:
                inputs = []
                for ch in chunks:
                    inputs.append(ChunkInput(ch.idx, ch.text, ch.n_tokens, ch.char_start, ch.char_end, vecs[k]))
                    k += 1
                new_ids += store.add_chunks(did, inputs)
                store.set_document_status(did, "chunked")
            store.log_stage("chunk_embed", time.perf_counter() - tb, batch_id=f"chunk_embed_{b}")
        fm.chunks.add(new_ids, vecs)  # after commit
        n_chunks += len(new_ids)
        log.info("chunk+embed: %d/%d docs, %d chunks so far", min(b + docs_per_batch, len(doc_ids)), len(doc_ids),
                 n_chunks)
    fm.save()
    emb.close()
    dt = time.perf_counter() - t0
    return {"documents": len(doc_ids), "chunks": n_chunks, "seconds": round(dt, 1),
            "faiss_chunks": fm.chunks.ntotal, "faiss_consistent": fm.is_consistent("chunks")}
