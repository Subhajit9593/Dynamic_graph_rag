"""Stage: chunks -> LLM entity/relation extraction -> `extractions` table (PLAN.md §6 step 4).

Concurrency design:
* llama-server runs `parallel` slots on the GPU. We keep 2x that many requests in flight so a slot is never
  idle waiting for Python.
* Worker threads ONLY do HTTP calls (I/O-bound, the GIL is released while waiting).
* All SQLite writes happen in the main thread (one connection, no locking issues), one small transaction per
  chunk -> progress is durable; Ctrl-C or a crash loses at most the in-flight requests.
* Resumable: pending = chunks without an OK extraction for the current prompt version.
* Retries: invalid JSON because the output hit max_tokens -> retry once with a larger budget;
  HTTP errors -> retry up to 2 times.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any

from dgrag.config import Config
from dgrag.extract.prompts import build_messages, build_schema
from dgrag.extract.validate import validate_extraction
from dgrag.llm.server import ChatResult, LlamaClient, LlamaServer
from dgrag.store import SQLiteStore

log = logging.getLogger(__name__)


@dataclass
class ChunkJob:
    chunk_id: int
    text: str
    title: str | None
    source: str | None
    published_at: str | None


def _load_jobs(store: SQLiteStore, chunk_ids: list[int]) -> list[ChunkJob]:
    jobs = []
    for i in range(0, len(chunk_ids), 900):
        part = chunk_ids[i : i + 900]
        rows = store.conn.execute(
            f"SELECT c.id, c.text, d.title, d.source, d.published_at FROM chunks c JOIN documents d ON d.id = c.doc_id "
            f"WHERE c.id IN ({','.join('?' * len(part))})", part).fetchall()
        by_id = {r[0]: ChunkJob(*r) for r in rows}
        jobs += [by_id[c] for c in part if c in by_id]
    return jobs


def _call(client: LlamaClient, job: ChunkJob, schema: dict, max_tokens: int, temperature: float, seed: int
          ) -> tuple[Any | None, ChatResult, int]:
    """Runs in a worker thread. Returns (parsed_json_or_None, last_result, attempts)."""
    msgs = build_messages(job.text, job.title, job.source, job.published_at)
    budget, attempts, http_errors = max_tokens, 0, 0
    while True:
        attempts += 1
        try:
            obj, res = client.chat_json(msgs, schema, max_tokens=budget, temperature=temperature, seed=seed)
        except Exception:
            http_errors += 1
            if http_errors > 2:
                raise
            time.sleep(2 * http_errors)
            continue
        if obj is not None or res.finish_reason != "length" or budget > max_tokens:
            return obj, res, attempts
        budget = int(max_tokens * 1.5)  # cut off mid-JSON: one retry with more room


def run_extraction(cfg: Config, store: SQLiteStore, limit: int | None = None, server: LlamaServer | None = None
                   ) -> dict[str, Any]:
    pv = cfg.extraction.prompt_version
    pending = store.chunks_pending_extraction(pv, limit)
    if not pending:
        log.info("Nothing to extract for prompt version %s", pv)
        return {"processed": 0}
    jobs = _load_jobs(store, pending)
    schema = build_schema(cfg.extraction.max_entities, cfg.extraction.max_relations)
    model_name = cfg.llm.model_path.name.replace("-00001-of-00002", "")
    store.set_meta("extraction_model", model_name)

    own_server = server is None
    srv = server or LlamaServer.from_config(cfg)
    if own_server:
        srv.start()
    client = srv.client()
    in_flight_max = cfg.llm.parallel * 2
    stats: Counter = Counter()
    drop_reasons: Counter = Counter()
    t0 = time.perf_counter()
    done = 0
    interrupted = False
    queue = iter(jobs)
    futures: dict[Future, ChunkJob] = {}
    # Not a `with` block on purpose: on Ctrl-C we must NOT wait for in-flight requests before stopping the
    # server (that deadlocked, PLAN.md Issue #8). Order on exit: cancel queue -> stop server -> threads error out.
    pool = ThreadPoolExecutor(max_workers=in_flight_max, thread_name_prefix="extract")

    def submit_next() -> bool:
        job = next(queue, None)
        if job is None:
            return False
        futures[pool.submit(_call, client, job, schema, cfg.llm.max_tokens, cfg.llm.temperature,
                            cfg.llm.seed)] = job
        return True

    try:
        for _ in range(in_flight_max):
            if not submit_next():
                break
        while futures:
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in finished:
                job = futures.pop(fut)
                submit_next()
                try:
                    obj, res, attempts = fut.result()
                except Exception as e:  # server unreachable after retries
                    log.error("chunk %d failed: %s", job.chunk_id, e)
                    stats["http_failed"] += 1
                    continue
                stats["retries"] += attempts - 1
                _save(store, job, obj, res, pv, model_name, stats, drop_reasons)
                done += 1
                if done % 100 == 0 or done == len(jobs):
                    el = time.perf_counter() - t0
                    log.info("extract: %d/%d chunks | %.2f s/chunk | ETA %.1f min", done, len(jobs),
                             el / done, (len(jobs) - done) * el / done / 60)
    except KeyboardInterrupt:
        interrupted = True
        log.warning("Interrupted: %d chunks saved; %d in-flight chunks stay pending and will be redone next run",
                    done, len(futures))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        if own_server:
            srv.stop()  # in-flight HTTP calls now fail fast; their chunks remain pending
        n_docs = store.mark_documents_extracted()
    elapsed = time.perf_counter() - t0
    store.log_stage("extract", elapsed, batch_id=f"extract_{pv}", llm_calls=done + stats["retries"])
    return {"processed": done, "interrupted": interrupted, "seconds": round(elapsed, 1), "sec_per_chunk": round(elapsed / max(done, 1), 2),
            "documents_completed": n_docs, **dict(stats), "drop_reasons": dict(drop_reasons)}


def _save(store: SQLiteStore, job: ChunkJob, obj: Any, res: ChatResult, pv: str, model: str, stats: Counter,
          drop_reasons: Counter) -> None:
    chunk_id = job.chunk_id
    ok = isinstance(obj, dict)
    if ok:
        v = validate_extraction(obj, text=job.text, title=job.title)
        ents, rels = v.entities, v.relations
        de, dr = sum(v.dropped_entities.values()), sum(v.dropped_relations.values())
        drop_reasons.update({f"entity:{k}": n for k, n in v.dropped_entities.items()})
        drop_reasons.update({f"relation:{k}": n for k, n in v.dropped_relations.items()})
        stats["repaired_relations"] += v.repaired_relations
        stats["auto_added_entities"] += v.auto_added_entities
        stats["ok"] += 1
    else:
        ents, rels, de, dr = [], [], 0, 0
        stats["invalid_json"] += 1
    with store.transaction():
        store.save_extraction(
            chunk_id, prompt_version=pv, model=model, raw=res.content,
            entities_json=json.dumps(ents, ensure_ascii=False), relations_json=json.dumps(rels, ensure_ascii=False),
            n_entities=len(ents), n_relations=len(rels), n_dropped_entities=de, n_dropped_relations=dr, ok=ok,
            finish_reason=res.finish_reason, prompt_tokens=res.prompt_tokens,
            completion_tokens=res.completion_tokens, seconds=round(res.seconds, 3))


def revalidate_all(store: SQLiteStore) -> dict[str, Any]:
    """Re-run validate_extraction on every stored raw LLM output (no LLM calls).

    Lets us improve the cleaning rules after the expensive extraction run.
    """
    stats: Counter = Counter()
    rows = store.conn.execute(
        "SELECT x.chunk_id, x.raw, c.text, d.title FROM extractions x JOIN chunks c ON c.id = x.chunk_id "
        "JOIN documents d ON d.id = c.doc_id WHERE x.ok = 1"
    ).fetchall()
    with store.transaction():
        for cid, raw, text, title in rows:
            v = validate_extraction(json.loads(raw), text=text, title=title)
            store.conn.execute(
                "UPDATE extractions SET entities_json = ?, relations_json = ?, n_entities = ?, n_relations = ?, "
                "n_dropped_entities = ?, n_dropped_relations = ? WHERE chunk_id = ?",
                (json.dumps(v.entities, ensure_ascii=False), json.dumps(v.relations, ensure_ascii=False),
                 len(v.entities), len(v.relations), sum(v.dropped_entities.values()),
                 sum(v.dropped_relations.values()), cid))
            stats["chunks"] += 1
            stats["entities"] += len(v.entities)
            stats["relations"] += len(v.relations)
            stats.update({f"drop_entity:{k}": n for k, n in v.dropped_entities.items()})
            stats["auto_added_entities"] += v.auto_added_entities
    return dict(stats)
