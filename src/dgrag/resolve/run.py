"""Running graph builds and the streaming experiment (PLAN.md §9.3, D14).

Each graph *variant* (e.g. "dynamic", "static") lives in its own SQLite file + FAISS dir, created as a copy of the
main database (documents, chunks, embeddings, cached extractions). Builds never modify the main database.
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from dgrag.config import Config
from dgrag.ingest.loader import batched
from dgrag.resolve.builder import GlobalBuilder, IncrementalBuilder
from dgrag.resolve.judge import Judge, llm_ask
from dgrag.store import FaissManager, SQLiteStore

log = logging.getLogger(__name__)


def variant_paths(cfg: Config, variant: str) -> tuple[Path, Path]:
    return cfg.resolve(Path(f"data/db/{variant}.sqlite")), cfg.resolve(Path(f"data/faiss/{variant}"))


def make_variant(cfg: Config, variant: str, fresh: bool = True) -> tuple[Path, Path]:
    """Copy the main DB (consistent snapshot via SQLite backup API) and clear its graph tables."""
    db, fdir = variant_paths(cfg, variant)
    if fresh or not db.exists():
        for p in (db, db.with_suffix(".sqlite-wal"), db.with_suffix(".sqlite-shm")):
            p.unlink(missing_ok=True)
        shutil.rmtree(fdir, ignore_errors=True)
        db.parent.mkdir(parents=True, exist_ok=True)
        src = SQLiteStore(cfg.resolve(cfg.paths.db_path))  # also migrates the main DB if needed
        dst = sqlite3.connect(str(db))
        src.conn.backup(dst)
        dst.close()
        src.close()
        with SQLiteStore(db) as s:
            s.clear_graph()
            s.conn.execute("DELETE FROM judge_cache")
            s.conn.execute("DELETE FROM ingest_log WHERE stage LIKE 'graph_%'")
            s.conn.execute("UPDATE documents SET status = 'extracted' WHERE status IN ('resolved')")
        log.info("Created variant %s at %s", variant, db)
    return db, fdir


class _Resources:
    """Embedder + (optional) llama-server for the judge, shared across builds."""

    def __init__(self, cfg: Config, use_judge: bool):
        self.cfg, self.use_judge = cfg, use_judge
        self.stack = ExitStack()
        self.ask = None
        self.embedder = None

    def __enter__(self) -> _Resources:
        from dgrag.embedder import Embedder

        self.embedder = Embedder(self.cfg.embedder.model_name, self.cfg.embedder.device, self.cfg.embedder.batch_size)
        if self.use_judge:
            from dgrag.llm.server import LlamaServer

            srv = LlamaServer.from_config(self.cfg, parallel=2, ctx_per_slot=2048)
            self.stack.enter_context(srv)
            self.ask = llm_ask(srv.client())
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stack.close()
        if self.embedder:
            self.embedder.close()


def _open(cfg: Config, variant: str) -> tuple[SQLiteStore, FaissManager]:
    db, fdir = variant_paths(cfg, variant)
    store = SQLiteStore(db)
    fm = FaissManager(store, fdir, cfg.embedder.dim)
    fm.open()
    return store, fm


def _graph_size(store: SQLiteStore) -> dict[str, int]:
    q = lambda s: store.conn.execute(s).fetchone()[0]  # noqa: E731
    return {"entities": q("SELECT COUNT(*) FROM entities"),
            "relations": q("SELECT COUNT(*) FROM relations WHERE status != 'superseded'"),
            "superseded": q("SELECT COUNT(*) FROM relations WHERE status = 'superseded'"),
            "mentions": q("SELECT COUNT(*) FROM entity_mentions")}


def build_graph(cfg: Config, variant: str, mode: str, use_judge: bool = True, limit_docs: int | None = None,
                fresh: bool = True) -> dict[str, Any]:
    make_variant(cfg, variant, fresh=fresh)
    with _Resources(cfg, use_judge) as rs:
        store, fm = _open(cfg, variant)
        try:
            judge = Judge(store, rs.ask)
            docs = store.documents_with_status("extracted")[:limit_docs]
            if mode == "incremental":
                b = IncrementalBuilder(store, fm, rs.embedder.encode, judge, cfg.resolution)
                out = b.add_documents(docs, batch_id="full")
            else:
                b = GlobalBuilder(store, fm, rs.embedder.encode, judge, cfg.resolution)
                out = b.build(docs, batch_id="full")
            return {**out, **_graph_size(store), "build_stats": dict(b.stats), **judge.stats()}
        finally:
            fm.save()
            store.close()


def stream_experiment(cfg: Config, n_batches: int = 10, use_judge: bool = True) -> dict[str, Any]:
    """Feed documents in date order, batch by batch. dynamic = add only the new batch; static = rebuild all."""
    results: dict[str, Any] = {"n_batches": n_batches, "batches": []}
    make_variant(cfg, "dynamic")
    make_variant(cfg, "static")
    with _Resources(cfg, use_judge) as rs:
        dyn_store, dyn_fm = _open(cfg, "dynamic")
        st_store, st_fm = _open(cfg, "static")
        try:
            docs = dyn_store.documents_with_status("extracted")  # chronological
            dyn = IncrementalBuilder(dyn_store, dyn_fm, rs.embedder.encode, Judge(dyn_store, rs.ask), cfg.resolution)
            st_judge = Judge(st_store, rs.ask)
            seen: list[int] = []
            for k, batch in enumerate(batched(docs, n_batches), 1):
                seen += batch
                d = dyn.add_documents(batch, batch_id=f"stream_{k}")
                st = GlobalBuilder(st_store, st_fm, rs.embedder.encode, st_judge, cfg.resolution).build(
                    seen, batch_id=f"stream_{k}")
                row = {
                    "batch": k, "new_docs": len(batch), "total_docs": len(seen),
                    "until": dyn_store.get_document(batch[-1])["published_at"][:10],
                    "dynamic": {**d, **_graph_size(dyn_store)},
                    "static": {**st, **_graph_size(st_store)},
                }
                results["batches"].append(row)
                log.info("batch %d/%d (%d docs, until %s): dynamic %.1fs | static rebuild %.1fs", k, n_batches,
                         len(seen), row["until"], d["seconds"], st["seconds"])
            results["dynamic_total_seconds"] = round(sum(b["dynamic"]["seconds"] for b in results["batches"]), 1)
            results["static_total_seconds"] = round(sum(b["static"]["seconds"] for b in results["batches"]), 1)
            results["dynamic_stats"] = dict(dyn.stats)
        finally:
            for s, f in ((dyn_store, dyn_fm), (st_store, st_fm)):
                f.save()
                s.close()
    out = cfg.resolve(cfg.paths.results_dir) / "stream_experiment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    results["created"] = time.strftime("%Y-%m-%d %H:%M")
    out.write_text(json.dumps(results, indent=2))
    return results
