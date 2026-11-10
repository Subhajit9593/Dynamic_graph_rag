"""Retrieval-only ablations on all answerable questions (no LLM; minutes, not hours).

Which part of the graph retriever helps? Each row switches one component off.
Output: results/ablation_retrieval.json
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from dgrag.config import Config
from dgrag.embedder import Embedder
from dgrag.eval.run_eval import _metrics_retrieval, select_questions
from dgrag.retrieve.retrievers import GraphRetriever, VectorRetriever
from dgrag.store import FaissManager, SQLiteStore

log = logging.getLogger(__name__)

GRAPH_ABLATIONS: list[tuple[str, dict[str, Any]]] = [
    ("full (graph bonus + source-aware)", {}),
    ("- source-aware (graph bonus only)", {"w_source": 0.0, "per_source": 0}),
    ("- graph bonus (source-aware only)", {"w_graph": 0.0}),
    ("- both (= vector, 2 chunks per article)", {"w_graph": 0.0, "w_source": 0.0, "per_source": 0}),
    ("1-hop instead of 2-hop", {"hops": 1}),
    ("v1 scoring (graph replaces similarity)", {"chunk_scoring": "facts", "w_source": 0.0, "per_source": 0}),
]


def _avg(ms: list[dict[str, float]]) -> dict[str, float]:
    return {k: round(float(np.mean([m[k] for m in ms])), 4) for k in ms[0]}


def run_ablations(cfg: Config) -> dict[str, Any]:
    qs = [q for q in select_questions(cfg, None) if q.question_type != "null_query"]
    emb = Embedder(cfg.embedder.model_name, cfg.embedder.device, cfg.embedder.batch_size)
    qv = emb.encode([q.query for q in qs], is_query=True)
    emb.close()

    def score(retriever: Any) -> dict[str, Any]:
        t = time.perf_counter()
        ms = [_metrics_retrieval(retriever.retrieve(q.query, v[None, :]).doc_ranking(), q.evidence_urls)
              for q, v in zip(qs, qv)]
        return {**_avg(ms), "ms_per_query": round(1000 * (time.perf_counter() - t) / len(qs), 1)}

    out: dict[str, Any] = {"n_questions": len(qs), "rows": []}
    main = SQLiteStore(cfg.resolve(cfg.paths.db_path))
    mfm = FaissManager(main, cfg.resolve(cfg.paths.faiss_dir), cfg.embedder.dim)
    mfm.open()
    out["rows"].append({"graph": "-", "setting": "vector RAG (top-k)", **score(VectorRetriever(main, mfm, k=20))})
    main.close()
    for variant in ("dynamic", "static"):
        st = SQLiteStore(cfg.resolve(Path(f"data/db/{variant}.sqlite")))
        fm = FaissManager(st, cfg.resolve(Path(f"data/faiss/{variant}")), cfg.embedder.dim)
        fm.open()
        for name, kw in (GRAPH_ABLATIONS if variant == "dynamic" else GRAPH_ABLATIONS[:1]):
            row = {"graph": variant, "setting": name, **score(GraphRetriever(st, fm, k=20, **kw))}
            out["rows"].append(row)
            log.info("%s | %s | %s", variant, name, row)
        st.close()
    path = cfg.resolve(cfg.paths.results_dir) / "ablation_retrieval.json"
    path.write_text(json.dumps(out, indent=2))
    return out
