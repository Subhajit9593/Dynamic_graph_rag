"""Retrievers (PLAN.md §8). All return a ranked list of chunks (+ for the graph retriever, a list of facts).

* VectorRetriever  (baseline B1): top-k chunks by cosine similarity.
* RerankRetriever  (baseline B1+): top-n by vector, re-scored by a cross-encoder, keep top-k.
* GraphRetriever   (B2 static / ours dynamic, depending on which DB it is given):
    1. seeds: entities whose alias appears in the question (n-gram lookup, no LLM), entities / relations /
       chunks found by vector search
    2. expand: 2-hop neighbourhood of seed entities (hub-capped), personalised PageRank on it
    3. score edges: similarity(question, edge vector) + PageRank of its endpoints + log(evidence count)
    4. chunks: evidence chunks of the top edges + top vector chunks, scored, max `per_doc` chunks per article
       (multi-hop questions need several *different* articles)
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from dgrag.store import FaissManager, GraphView, SQLiteStore
from dgrag.store.sqlite_store import from_blob
from dgrag.text_utils import normalize_name

log = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: int
    doc_id: int
    score: float
    text: str
    title: str
    source: str
    published_at: str
    url: str | None


@dataclass
class Fact:
    relation_id: int
    text: str            # "Caroline Ellison testified against Sam Bankman-Fried (2023-10-10)"
    score: float
    chunk_ids: list[int] = field(default_factory=list)


@dataclass
class Retrieval:
    chunks: list[RetrievedChunk]
    facts: list[Fact] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)

    def doc_ranking(self) -> list[str]:
        """Article urls in rank order (first occurrence) - used for retrieval metrics."""
        seen, out = set(), []
        for c in self.chunks:
            if c.url and c.url not in seen:
                seen.add(c.url)
                out.append(c.url)
        return out


def _chunk_rows(store: SQLiteStore, scored: list[tuple[int, float]]) -> list[RetrievedChunk]:
    if not scored:
        return []
    ids = [c for c, _ in scored]
    rows = store.conn.execute(
        f"SELECT c.id, c.doc_id, c.text, d.title, d.source, d.published_at, d.url FROM chunks c "
        f"JOIN documents d ON d.id = c.doc_id WHERE c.id IN ({','.join('?' * len(ids))})", ids).fetchall()
    by = {r[0]: r for r in rows}
    return [RetrievedChunk(c, by[c][1], s, by[c][2], by[c][3] or "", by[c][4] or "", by[c][5] or "", by[c][6])
            for c, s in scored if c in by]


def _cap_per_doc(store: SQLiteStore, scored: list[tuple[int, float]], k: int, per_doc: int | None
                 ) -> list[tuple[int, float]]:
    if per_doc is None:
        return scored[:k]
    doc_of = dict(store.conn.execute(
        f"SELECT id, doc_id FROM chunks WHERE id IN ({','.join('?' * len(scored))})",
        [c for c, _ in scored]).fetchall()) if scored else {}
    count: dict[int, int] = defaultdict(int)
    out = []
    for c, s in scored:
        d = doc_of.get(c)
        if count[d] < per_doc:
            out.append((c, s))
            count[d] += 1
        if len(out) == k:
            break
    return out


class VectorRetriever:
    def __init__(self, store: SQLiteStore, fm: FaissManager, k: int = 8, per_doc: int | None = None):
        self.store, self.fm, self.k, self.per_doc = store, fm, k, per_doc

    def retrieve(self, question: str, q_vec: np.ndarray) -> Retrieval:
        hits = self.fm.chunks.search(q_vec, self.k * 4 if self.per_doc else self.k)[0]
        return Retrieval(_chunk_rows(self.store, _cap_per_doc(self.store, hits, self.k, self.per_doc)))


class RerankRetriever:
    """Cross-encoder reranking of the top-n vector hits (baseline B1+)."""

    def __init__(self, store: SQLiteStore, fm: FaissManager, reranker: Any, k: int = 8, n_candidates: int = 30):
        self.store, self.fm, self.reranker, self.k, self.n = store, fm, reranker, k, n_candidates

    def retrieve(self, question: str, q_vec: np.ndarray) -> Retrieval:
        cands = _chunk_rows(self.store, self.fm.chunks.search(q_vec, self.n)[0])
        if not cands:
            return Retrieval([])
        scores = self.reranker.predict([(question, c.text) for c in cands], batch_size=32, show_progress_bar=False)
        for c, s in zip(cands, scores):
            c.score = float(s)
        cands.sort(key=lambda c: -c.score)
        return Retrieval(cands[: self.k])


class GraphRetriever:
    def __init__(self, store: SQLiteStore, fm: FaissManager, k: int = 8, per_doc: int | None = 2,
                 n_seed_entities: int = 10, n_seed_relations: int = 20, n_seed_chunks: int = 10,
                 hops: int = 2, max_neighbors: int = 25, n_facts: int = 15, ppr_alpha: float = 0.85,
                 w_sim: float = 1.0, w_ppr: float = 0.5, w_weight: float = 0.05, w_vec_chunk: float = 1.0):
        self.store, self.fm = store, fm
        self.k, self.per_doc = k, per_doc
        self.n_seed_entities, self.n_seed_relations, self.n_seed_chunks = n_seed_entities, n_seed_relations, n_seed_chunks
        self.hops, self.max_neighbors, self.n_facts, self.ppr_alpha = hops, max_neighbors, n_facts, ppr_alpha
        self.w_sim, self.w_ppr, self.w_weight, self.w_vec_chunk = w_sim, w_ppr, w_weight, w_vec_chunk
        self._load()

    def _load(self) -> None:
        self.graph = GraphView.from_store(self.store).g.to_undirected(as_view=False)  # walk ignores direction
        self.names = {int(r["id"]): r["name"] for r in self.store.iter_entities()}
        self.rel = {}
        for r in self.store.conn.execute(
                "SELECT id, src_id, tgt_id, predicate, description, weight, valid_from, status, embedding "
                "FROM relations WHERE status != 'superseded'"):
            self.rel[int(r[0])] = {"src": r[1], "tgt": r[2], "pred": r[3], "desc": r[4], "weight": r[5],
                                   "date": (r[6] or "")[:10], "status": r[7],
                                   "vec": from_blob(r[8]) if r[8] is not None else None}
        self.max_alias_tokens = 6
        # n-gram alias index: alias_norm -> entity ids (only aliases of >= 3 chars to avoid "us", "ai" noise)
        self.alias = defaultdict(list)
        for a, e in self.store.conn.execute("SELECT alias_norm, entity_id FROM entity_aliases"):
            if len(a) >= 3:
                self.alias[a].append(int(e))
        mc = dict(self.store.conn.execute("SELECT id, mention_count FROM entities").fetchall())
        self.mention_count = {int(k): v for k, v in mc.items()}

    # ---------------------------------------------------------------- seeds
    def _alias_seeds(self, question: str) -> dict[int, float]:
        toks = re.findall(r"[\w'’\-&.]+", question)
        found: dict[int, float] = {}
        for n in range(self.max_alias_tokens, 0, -1):
            for i in range(len(toks) - n + 1):
                norm = normalize_name(" ".join(toks[i : i + n]))
                for e in self.alias.get(norm, []):
                    found[e] = max(found.get(e, 0.0), 1.0)
        return found

    def retrieve(self, question: str, q_vec: np.ndarray) -> Retrieval:
        seeds: dict[int, float] = dict(self._alias_seeds(question))
        n_alias = len(seeds)
        for eid, s in self.fm.entities.search(q_vec, self.n_seed_entities)[0]:
            seeds[eid] = max(seeds.get(eid, 0.0), s)
        rel_hits = self.fm.relations.search(q_vec, self.n_seed_relations)[0]
        for rid, s in rel_hits:
            r = self.rel.get(rid)
            if r:
                for e in (r["src"], r["tgt"]):
                    seeds[e] = max(seeds.get(e, 0.0), s * 0.9)
        chunk_hits = self.fm.chunks.search(q_vec, self.n_seed_chunks)[0]

        # ---------------- expand: hub-capped 2-hop neighbourhood
        g = self.graph
        nodes = {e for e in seeds if e in g}
        frontier = set(nodes)
        for _ in range(self.hops):
            nxt = set()
            for u in frontier:
                nbrs = sorted(g[u], key=lambda v: -max(d.get("weight", 1.0) for d in g[u][v].values()))
                nxt.update(nbrs[: self.max_neighbors])
            nxt -= nodes
            nodes |= nxt
            frontier = nxt
        sub = g.subgraph(nodes)

        ppr: dict[int, float] = {}
        if sub.number_of_edges():
            pers = {e: s for e, s in seeds.items() if e in sub}
            simple = nx.Graph()
            simple.add_nodes_from(sub.nodes)
            for u, v, d in sub.edges(data=True):
                w = simple[u][v]["weight"] + d.get("weight", 1.0) if simple.has_edge(u, v) else d.get("weight", 1.0)
                simple.add_edge(u, v, weight=w)
            try:
                ppr = nx.pagerank(simple, alpha=self.ppr_alpha, personalization=pers, weight="weight",
                                  max_iter=100, tol=1e-6)
            except nx.PowerIterationFailedConvergence:
                ppr = {}
        ppr_max = max(ppr.values()) if ppr else 1.0

        # ---------------- score edges in the neighbourhood (+ relation vector hits)
        q = q_vec.reshape(-1)
        cand = {k for _, _, k in sub.edges(keys=True)} | {rid for rid, _ in rel_hits}
        scored_facts = []
        for rid in cand:
            r = self.rel.get(rid)
            if r is None:
                continue
            sim = float(np.dot(r["vec"], q)) if r["vec"] is not None else 0.0
            pr = (ppr.get(r["src"], 0.0) + ppr.get(r["tgt"], 0.0)) / (2 * ppr_max) if ppr else 0.0
            score = self.w_sim * sim + self.w_ppr * pr + self.w_weight * math.log1p(r["weight"])
            scored_facts.append((score, rid))
        scored_facts.sort(reverse=True)
        top = scored_facts[: self.n_facts]

        facts: list[Fact] = []
        chunk_score: dict[int, float] = defaultdict(float)
        for score, rid in top:
            r = self.rel[rid]
            ev = [int(c[0]) for c in self.store.conn.execute(
                "SELECT chunk_id FROM relation_evidence WHERE relation_id = ?", (rid,))]
            txt = f"{self.names.get(r['src'], '?')} {r['pred']} {self.names.get(r['tgt'], '?')}"
            if r["desc"]:
                txt += f" — {r['desc']}"
            if r["date"]:
                txt += f" ({r['date']}{', disputed' if r['status'] == 'conflicted' else ''})"
            facts.append(Fact(rid, txt, score, ev))
            for c in ev:
                chunk_score[c] = max(chunk_score[c], score)  # a chunk is as good as its best supporting fact
        for c, s in chunk_hits:
            chunk_score[c] = max(chunk_score[c], self.w_vec_chunk * s + 0.05)  # direct hits as a safety net
        ranked = sorted(chunk_score.items(), key=lambda x: -x[1])
        chosen = _cap_per_doc(self.store, ranked, self.k, self.per_doc)
        return Retrieval(_chunk_rows(self.store, chosen), facts,
                         {"seeds": len(seeds), "alias_seeds": n_alias, "subgraph_nodes": len(nodes),
                          "candidate_edges": len(cand)})
