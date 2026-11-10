"""Question-answering evaluation on MultiHop-RAG (PLAN.md §9, D13).

Systems:
  none           LLM only, no retrieval                       (B0)
  vector         top-k chunks by cosine                       (B1)
  vector_rerank  top-30 by cosine -> cross-encoder -> top-k   (B1+)
  static         graph retriever on the GlobalBuilder graph   (B2)
  dynamic        graph retriever on the IncrementalBuilder graph (ours)

Two phases per system:
  1. retrieval for all questions (GPU embedder, CPU graph walk), ranked list of up to `k_rank` chunks;
     the first `k_prompt` chunks go into the prompt, the full ranking is used for retrieval metrics.
  2. generation for all questions in parallel on llama-server (16 slots, q8_0 KV cache).
Results: results/eval/<system>.jsonl (one line per question, resumable) and results/eval/<system>_metrics.json.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from dgrag.config import Config
from dgrag.generate.answer import ANSWER_SCHEMA, answer_correct, build_answer_messages, is_insufficient, parse_answer
from dgrag.ingest.loader import Question, load_questions
from dgrag.retrieve.retrievers import GraphRetriever, RerankRetriever, Retrieval, VectorRetriever
from dgrag.store import FaissManager, SQLiteStore

log = logging.getLogger(__name__)

SYSTEMS = ("none", "vector", "vector_rerank", "static", "dynamic")
RERANKER = "BAAI/bge-reranker-base"


def select_questions(cfg: Config, limit: int | None, seed: int = 0) -> list[Question]:
    qs = load_questions(cfg.resolve(cfg.paths.raw_dir) / cfg.dataset.questions_file)
    if not limit or limit >= len(qs):
        return qs
    # stratified by question type, deterministic
    by_type: dict[str, list[Question]] = defaultdict(list)
    for q in qs:
        by_type[q.question_type].append(q)
    rng = random.Random(seed)
    out = []
    for t, items in sorted(by_type.items()):
        n = max(1, round(limit * len(items) / len(qs)))
        out += rng.sample(items, min(n, len(items)))
    return sorted(out, key=lambda q: q.qid)


def _db_for(cfg: Config, system: str) -> tuple[Path, Path]:
    if system in ("static", "dynamic"):
        return cfg.resolve(Path(f"data/db/{system}.sqlite")), cfg.resolve(Path(f"data/faiss/{system}"))
    return cfg.resolve(cfg.paths.db_path), cfg.resolve(cfg.paths.faiss_dir)


def retrieve_all(cfg: Config, system: str, questions: list[Question], k_rank: int = 20,
                 retriever_kw: dict | None = None) -> dict[int, Retrieval | None]:
    if system == "none":
        return {q.qid: None for q in questions}
    from dgrag.embedder import Embedder

    db, fdir = _db_for(cfg, system)
    if not db.exists():
        raise FileNotFoundError(f"{db} missing - build the graph first (dgrag build-graph / dgrag stream)")
    store = SQLiteStore(db)
    fm = FaissManager(store, fdir, cfg.embedder.dim)
    fm.open()
    emb = Embedder(cfg.embedder.model_name, cfg.embedder.device, cfg.embedder.batch_size)
    reranker = None
    try:
        q_vecs = emb.encode([q.query for q in questions], is_query=True)
        kw = retriever_kw or {}
        if system == "vector":
            r = VectorRetriever(store, fm, k=k_rank, **kw)
        elif system == "vector_rerank":
            from sentence_transformers import CrossEncoder

            reranker = CrossEncoder(RERANKER, device=cfg.embedder.device)
            r = RerankRetriever(store, fm, reranker, k=k_rank, n_candidates=max(30, k_rank), **kw)
        else:
            r = GraphRetriever(store, fm, k=k_rank, **kw)
        out = {}
        t0 = time.perf_counter()
        for q, v in zip(questions, q_vecs):
            out[q.qid] = r.retrieve(q.query, v[None, :])
        log.info("%s retrieval: %d questions in %.1fs", system, len(questions), time.perf_counter() - t0)
        return out
    finally:
        emb.close()
        del reranker
        store.close()


def _metrics_retrieval(ranked: list[str], gold: set[str]) -> dict[str, float]:
    if not gold:
        return {}
    hits = [1 if u in gold else 0 for u in ranked]
    first = next((i for i, h in enumerate(hits[:10]) if h), None)
    prec_sum, n_rel = 0.0, 0
    for i, h in enumerate(hits[:10]):
        if h:
            n_rel += 1
            prec_sum += n_rel / (i + 1)
    return {
        "hits@4": float(any(hits[:4])), "hits@10": float(any(hits[:10])),
        "recall@4": sum(hits[:4]) / len(gold), "recall@10": sum(hits[:10]) / len(gold),
        "mrr@10": 1.0 / (first + 1) if first is not None else 0.0,
        "map@10": prec_sum / min(len(gold), 10),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_type[r["type"]].append(r)
    acc = {t: round(sum(r["correct"] for r in rs) / len(rs), 4) for t, rs in sorted(by_type.items())}
    acc["overall"] = round(sum(r["correct"] for r in records) / len(records), 4) if records else 0.0
    answerable = [r for r in records if r["type"] != "null_query"]
    acc["answerable_only"] = round(sum(r["correct"] for r in answerable) / max(len(answerable), 1), 4)
    ret = [r["retrieval"] for r in records if r.get("retrieval")]
    ret_avg = {k: round(float(np.mean([m[k] for m in ret])), 4) for k in ret[0]} if ret else {}
    refusal = [r for r in answerable if is_insufficient(r["pred"])]
    return {"n": len(records), "accuracy": acc, "retrieval": ret_avg,
            "false_refusal_rate": round(len(refusal) / max(len(answerable), 1), 4),
            "avg_prompt_tokens": round(float(np.mean([r["prompt_tokens"] for r in records])), 1) if records else 0,
            "by_type_counts": dict(Counter(r["type"] for r in records))}


def evaluate(cfg: Config, system: str, limit: int | None = None, k_prompt: int = 8, k_rank: int = 20,
             tag: str | None = None, retriever_kw: dict | None = None) -> dict[str, Any]:
    from dgrag.llm.server import LlamaServer

    name = tag or system
    out_dir = cfg.resolve(cfg.paths.results_dir) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.jsonl"
    questions = select_questions(cfg, limit)
    done: dict[int, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done[r["qid"]] = r
    todo = [q for q in questions if q.qid not in done]
    log.info("%s: %d questions (%d already done)", name, len(questions), len(questions) - len(todo))

    if todo:
        rets = retrieve_all(cfg, system, todo, k_rank=k_rank, retriever_kw=retriever_kw)
        srv = LlamaServer.from_config(cfg, parallel=16, ctx_per_slot=6144, kv_cache_type="q8_0")
        with srv, open(path, "a") as f, ThreadPoolExecutor(max_workers=32) as pool:
            client = srv.client()

            def run(q: Question) -> dict[str, Any]:
                ret = rets[q.qid]
                prompt_ret = replace(ret, chunks=ret.chunks[:k_prompt]) if ret is not None else None
                msgs = build_answer_messages(q.query, prompt_ret)
                obj, res = client.chat_json(msgs, ANSWER_SCHEMA, max_tokens=256, temperature=0.0)
                pred, reasoning = parse_answer(obj, res.content)
                rec = {"qid": q.qid, "type": q.question_type, "question": q.query, "gold": q.answer,
                       "pred": pred, "reasoning": reasoning,
                       "correct": answer_correct(pred, q.answer, q.question_type),
                       "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                       "seconds": round(res.seconds, 2)}
                if ret is not None:
                    ranked = ret.doc_ranking()
                    rec["retrieved_urls"] = ranked[:10]
                    rec["prompt_chunk_ids"] = [c.chunk_id for c in prompt_ret.chunks]
                    rec["n_facts"] = len(ret.facts)
                    if q.question_type != "null_query":
                        rec["retrieval"] = _metrics_retrieval(ranked, q.evidence_urls)
                return rec

            t0 = time.perf_counter()
            futs = [pool.submit(run, q) for q in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                done[rec["qid"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if i % 200 == 0 or i == len(futs):
                    el = time.perf_counter() - t0
                    log.info("%s: %d/%d answered | %.2f s/q | ETA %.1f min", name, i, len(futs), el / i,
                             (len(futs) - i) * el / i / 60)

    records = [done[q.qid] for q in questions if q.qid in done]
    metrics = {"system": system, "tag": name, "k_prompt": k_prompt, **summarize(records)}
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
