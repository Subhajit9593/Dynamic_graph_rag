"""Graph construction from cached extractions (PLAN.md §6, D13/D14).

Two builders share the relation logic:

* IncrementalBuilder (OURS, dynamic): processes one document at a time in date order. Each entity mention is
  resolved against the graph built so far (alias lookup -> vector search + name rules -> LLM judge -> new).
  Adding a batch only touches the new documents' mentions, so update cost ~ size of the batch.

* GlobalBuilder (static baseline): looks at ALL mentions at once and clusters them (union-find over
  nearest-neighbour pairs). Order-independent and sees everything, but any change means redoing the whole
  clustering, so update cost ~ size of the corpus.

Both write the same tables: entities, entity_aliases, entity_mentions, relations (+ embedding), relation_evidence.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from dgrag.config import ResolutionConfig
from dgrag.resolve.judge import Judge
from dgrag.resolve.names import (STRICT_SUBSET_TYPES, compat_kind, entity_text, name_compatible, relation_text,
                                 types_compatible, types_soft_compatible)
from dgrag.store import FaissManager, SQLiteStore, VectorIndex, l2_normalize
from dgrag.store.sqlite_store import from_blob
from dgrag.text_utils import normalize_name, normalize_predicate

log = logging.getLogger(__name__)

EmbedFn = Callable[[Sequence[str]], np.ndarray]

# Predicates where a new value can replace an old one (one CEO per company, one team per player at a time).
# Only these trigger the (LLM) conflict check; "defeated", "partnered with", ... can have many targets.
FUNCTIONAL_KEYWORDS = ("ceo", "chief executive", "president", "chair", "head coach", "coach", "manager",
                       "leader", "leads", "owner", "owns", "plays for", "signed with", "governor",
                       "prime minister", "director", "head of", "led by", "founder")
_FUNC_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in FUNCTIONAL_KEYWORDS) + r")\b")
SAME_RELATION_SIM = 0.90  # two relations between the same pair with this cosine are the same fact


def functional_key(predicate: str) -> str | None:
    m = _FUNC_RE.search(predicate.lower().replace("_", " "))
    return m.group(1) if m else None


@dataclass
class Mention:
    chunk_id: int
    date: str | None
    name: str
    type: str
    desc: str
    vec: np.ndarray | None = None


@dataclass
class RelMention:
    chunk_id: int
    date: str | None
    src: str
    pred: str
    tgt: str
    desc: str
    vec: np.ndarray | None = None


def load_document_mentions(store: SQLiteStore, doc_id: int) -> tuple[list[Mention], list[RelMention]]:
    rows = store.conn.execute(
        "SELECT c.id, d.published_at, x.entities_json, x.relations_json FROM chunks c "
        "JOIN documents d ON d.id = c.doc_id JOIN extractions x ON x.chunk_id = c.id AND x.ok = 1 "
        "WHERE c.doc_id = ? ORDER BY c.idx", (doc_id,)).fetchall()
    ents, rels = [], []
    for cid, date, ej, rj in rows:
        ents += [Mention(cid, date, e["name"], e["type"], e.get("desc", "")) for e in json.loads(ej)]
        rels += [RelMention(cid, date, r["src"], r["pred"], r["tgt"], r.get("desc", "")) for r in json.loads(rj)]
    return ents, rels


class _GraphWriter:
    """Shared relation upsert + time-aware conflict handling."""

    def __init__(self, store: SQLiteStore, fm: FaissManager, embed: EmbedFn, judge: Judge,
                 res: ResolutionConfig, conflict_detection: bool = True):
        self.store, self.fm, self.embed, self.judge, self.res = store, fm, embed, judge, res
        self.conflict_detection = conflict_detection
        self.stats: Counter = Counter()

    def _embed_mentions(self, ents: list[Mention], rels: list[RelMention]) -> None:
        texts = [entity_text(m.name, m.type, m.desc) for m in ents] + \
                [relation_text(r.src, r.pred, r.tgt, r.desc) for r in rels]
        if not texts:
            return
        vecs = self.embed(texts)
        for m, v in zip(ents, vecs[: len(ents)]):
            m.vec = v
        for r, v in zip(rels, vecs[len(ents):]):
            r.vec = v

    # ---------------------------------------------------------------- relations
    def _upsert_relation(self, src_id: int, tgt_id: int, r: RelMention) -> int | None:
        st = self.stats
        if src_id == tgt_id:
            st["relations_self_loop"] += 1
            return None
        pnorm = normalize_predicate(r.pred)
        for rel in self.store.relations_between(src_id, tgt_id, status=None):
            if rel["status"] == "superseded":
                continue
            same = rel["predicate_norm"] == pnorm
            if not same and rel["embedding"] is not None and r.vec is not None:
                same = float(np.dot(from_blob(rel["embedding"]), r.vec)) >= SAME_RELATION_SIM
            if same:
                self.store.reinforce_relation(rel["id"], r.date)
                self.store.add_evidence(rel["id"], r.chunk_id)
                st["relations_reinforced"] += 1
                return int(rel["id"])

        new_id = self.store.add_relation(src_id, tgt_id, r.pred, description=r.desc, observed_at=r.date,
                                         embedding=r.vec)
        self.store.add_evidence(new_id, r.chunk_id)
        if r.vec is not None:
            self.fm.relations.add([new_id], r.vec[None, :])
        st["relations_new"] += 1
        if self.conflict_detection:
            self._check_conflicts(new_id, src_id, tgt_id, r)
        return new_id

    def _check_conflicts(self, new_id: int, src_id: int, tgt_id: int, r: RelMention) -> None:
        """Time-aware edges (PLAN.md §6.2): a newer functional fact may supersede an older one."""
        fkey = functional_key(r.pred)
        if fkey is None:
            return
        cands = [x for x in self.store.find_relations(src_id, status="active") if x["tgt_id"] != tgt_id]
        cands += [x for x in self.store.find_relations_to(tgt_id, status="active") if x["src_id"] != src_id]
        cands = [x for x in cands if x["id"] != new_id and functional_key(x["predicate"]) == fkey]
        if not cands:
            return
        names = {e["id"]: e["name"] for e in self.store.get_entities(
            {src_id, tgt_id} | {x["src_id"] for x in cands} | {x["tgt_id"] for x in cands})}
        new_txt = f"{names[src_id]} {r.pred} {names[tgt_id]}" + (f" ({r.desc})" if r.desc else "")
        for old in cands:
            old_txt = f"{names[old['src_id']]} {old['predicate']} {names[old['tgt_id']]}" + \
                      (f" ({old['description']})" if old["description"] else "")
            old_date, new_date = old["last_seen"] or "", r.date or ""
            if old_date <= new_date:
                verdict = self.judge.relation_conflict({"text": old_txt, "date": old_date},
                                                       {"text": new_txt, "date": new_date})
                if verdict == "supersedes":
                    self.store.supersede_relation(old["id"], new_id, valid_to=new_date or None)
                    self.stats["relations_superseded"] += 1
            else:  # the "new" fact comes from an older article than the existing one
                verdict = self.judge.relation_conflict({"text": new_txt, "date": new_date},
                                                       {"text": old_txt, "date": old_date})
                if verdict == "supersedes":
                    self.store.supersede_relation(new_id, old["id"], valid_to=old_date or None)
                    self.stats["relations_superseded"] += 1
                    return
            if verdict == "contradicts":
                self.store.set_relation_status(old["id"], "conflicted")
                self.store.set_relation_status(new_id, "conflicted")
                self.stats["relations_conflicted"] += 1

    def _write_relations(self, rels: list[RelMention], resolve_local: Callable[[int, str], int | None]) -> None:
        for r in rels:
            s, t = resolve_local(r.chunk_id, r.src), resolve_local(r.chunk_id, r.tgt)
            if s is None or t is None:
                self.stats["relations_unresolved_endpoint"] += 1
                continue
            self._upsert_relation(s, t, r)


class IncrementalBuilder(_GraphWriter):
    """Dynamic graph: add documents one at a time; never revisits old documents."""

    def add_documents(self, doc_ids: Sequence[int], batch_id: str | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        calls0 = self.judge.calls
        for did in doc_ids:
            self.add_document(did)
        self.fm.save()
        out = {"documents": len(doc_ids), "seconds": round(time.perf_counter() - t0, 2),
               "judge_llm_calls": self.judge.calls - calls0}
        self.store.log_stage("graph_incremental", out["seconds"], batch_id=batch_id,
                             llm_calls=out["judge_llm_calls"])
        return out

    def add_document(self, doc_id: int) -> None:
        ents, rels = load_document_mentions(self.store, doc_id)
        self._embed_mentions(ents, rels)
        local: dict[tuple[int, str], int] = {}  # (chunk_id, norm name) -> entity id
        touched: dict[int, np.ndarray] = {}     # entity id -> updated running-mean vector (FAISS upsert at end)
        try:
            with self.store.transaction():
                for m in ents:
                    eid = self._resolve(m, touched)
                    local[(m.chunk_id, normalize_name(m.name))] = eid
                    self.store.add_mention(eid, m.chunk_id)
                    self.stats["mentions"] += 1
                self._write_relations(rels, lambda cid, name: local.get((cid, normalize_name(name))))
                self.store.set_document_status(doc_id, "resolved")
        except Exception:
            # FAISS may contain ids from the rolled-back transaction -> make it match SQLite again
            for t in ("entities", "relations"):
                self.fm.reconcile(t)
            raise
        if touched:  # one batched FAISS update per document (flat-index removal is O(n), avoid per mention)
            ids = list(touched)
            self.fm.entities.add(ids, np.vstack([touched[i] for i in ids]))

    def _resolve(self, m: Mention, touched: dict[int, np.ndarray]) -> int:
        st, res = self.stats, self.res
        norm = normalize_name(m.name)
        # 1) exact alias. A one-word alias that is only PART of a longer name ("wilson" of "Zach Wilson") is not
        #    trusted blindly - another Wilson may appear - so it goes through the similarity check in step 2.
        for row in self.store.find_entities_by_alias(m.name):
            if not types_compatible(row["type"], m.type):
                continue
            if " " not in norm and norm != row["name_norm"] and " " in row["name_norm"]:
                continue
            st["merge_alias"] += 1
            return self._merge(row, m, touched)
        # 2) nearest neighbours by meaning, gated by name rules / LLM judge
        if m.vec is not None and self.fm.entities.ntotal:
            for eid, score in self.fm.entities.search(m.vec, res.top_k)[0]:
                if score < res.tau_low:
                    break
                row = self.store.get_entity(eid)
                if row is None:
                    continue
                if not types_compatible(row["type"], m.type):
                    # soft pair (ORG/PRODUCT, ...): only the exact same name with similar meaning
                    if types_soft_compatible(row["type"], m.type) and norm == row["name_norm"]:
                        st["merge_soft_type"] += 1
                        return self._merge(row, m, touched)
                    continue
                # compare with the entity's MAIN name only (not every alias it collected): aliases like
                # "world cup" or "sportsbook" otherwise chain unrelated entities together (PLAN.md Issue #11)
                kind = compat_kind(norm, row["name_norm"])
                strict = kind == "subset" and (m.type in STRICT_SUBSET_TYPES or row["type"] in STRICT_SUBSET_TYPES)
                if kind is not None and (not strict or score >= res.tau_high):
                    st["merge_vector"] += 1
                    return self._merge(row, m, touched)
                if score >= res.tau_high and self.judge.same_entity(
                        {"name": m.name, "type": m.type, "desc": m.desc},
                        {"name": row["name"], "type": row["type"], "desc": row["description"] or ""}):
                    st["merge_judge"] += 1
                    return self._merge(row, m, touched)
        # 3) new entity
        eid = self.store.add_entity(m.name, m.type, m.desc or None, m.vec)
        if m.vec is not None:
            self.fm.entities.add([eid], m.vec[None, :])
        st["entities_new"] += 1
        return eid

    def _merge(self, row: Any, m: Mention, touched: dict[int, np.ndarray]) -> int:
        eid = int(row["id"])
        self.store.add_alias(eid, m.name)
        upd: dict[str, Any] = {}
        mnorm, rnorm = normalize_name(m.name), row["name_norm"]
        if set(rnorm.split()) < set(mnorm.split()) and len(m.name) <= 60:
            upd["name"] = m.name  # "Kelce" -> "Travis Kelce": keep the more complete name for display
        if row["type"] == "OTHER" and m.type != "OTHER":
            upd["type_"] = m.type
        if not row["description"] and m.desc:
            upd["description"] = m.desc
        if m.vec is not None:  # running mean of mention vectors -> entity vector drifts to its typical context
            old = touched.get(eid)
            if old is None and row["embedding"] is not None:
                old = from_blob(row["embedding"])
            n = max(int(row["mention_count"]), 1)
            new = l2_normalize(old * n + m.vec)[0] if old is not None else m.vec
            touched[eid] = new
            upd["embedding"] = new
        if upd:
            self.store.update_entity(eid, **upd)
        return eid


class GlobalBuilder(_GraphWriter):
    """Static baseline: cluster ALL mentions at once (union-find), then write relations. Full rebuild each time."""

    def build(self, doc_ids: Sequence[int], batch_id: str | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        calls0 = self.judge.calls
        self.store.clear_graph()
        for t in ("entities", "relations"):
            self.fm[t].reset()
        doc_ids = list(doc_ids)
        all_ents: list[Mention] = []
        all_rels: list[RelMention] = []
        for did in doc_ids:
            e, r = load_document_mentions(self.store, did)
            all_ents += e
            all_rels += r
        self._embed_mentions(all_ents, all_rels)
        res = self.res

        # groups = mentions with the same normalised name (OTHER folded into the most common typed group)
        by_norm: dict[str, list[int]] = defaultdict(list)
        for i, m in enumerate(all_ents):
            by_norm[normalize_name(m.name)].append(i)
        groups: list[list[int]] = []
        for norm, idxs in by_norm.items():
            typed: dict[str, list[int]] = defaultdict(list)
            for i in idxs:
                typed[all_ents[i].type].append(i)
            other = typed.pop("OTHER", [])
            if typed:
                main = max(typed, key=lambda k: len(typed[k]))
                typed[main] += other
            else:
                typed["OTHER"] = other
            groups += list(typed.values())
        g_norm = [normalize_name(all_ents[g[0]].name) for g in groups]
        g_type = [Counter(all_ents[i].type for i in g).most_common(1)[0][0] for g in groups]
        g_vec = l2_normalize(np.vstack([np.mean([all_ents[i].vec for i in g], axis=0) for g in groups]))

        # union-find over nearest-neighbour pairs
        parent = list(range(len(groups)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        tmp = VectorIndex("tmp", g_vec.shape[1])
        tmp.add(np.arange(len(groups)), g_vec, normalize=False)
        neigh = tmp.search(g_vec, res.top_k + 1, normalize=False)
        for a, hits in enumerate(neigh):
            for b, score in hits:
                if b <= a or score < res.tau_low or find(a) == find(b):
                    continue
                if not types_compatible(g_type[a], g_type[b]):
                    if types_soft_compatible(g_type[a], g_type[b]) and g_norm[a] == g_norm[b]:
                        parent[find(a)] = find(b)
                        self.stats["merge_soft_type"] += 1
                    continue
                kind = compat_kind(g_norm[a], g_norm[b])
                strict = kind == "subset" and (g_type[a] in STRICT_SUBSET_TYPES or g_type[b] in STRICT_SUBSET_TYPES)
                if kind is not None and (not strict or score >= res.tau_high):
                    parent[find(a)] = find(b)
                    self.stats["merge_vector"] += 1
                elif score >= res.tau_high:
                    ma, mb = all_ents[groups[a][0]], all_ents[groups[b][0]]
                    if self.judge.same_entity({"name": ma.name, "type": ma.type, "desc": ma.desc},
                                              {"name": mb.name, "type": mb.type, "desc": mb.desc}):
                        parent[find(a)] = find(b)
                        self.stats["merge_judge"] += 1

        clusters: dict[int, list[int]] = defaultdict(list)
        for gi in range(len(groups)):
            clusters[find(gi)].append(gi)

        local: dict[tuple[int, str], int] = {}
        with self.store.transaction():
            new_ids, new_vecs = [], []
            for gis in clusters.values():
                mids = [i for gi in gis for i in groups[gi]]
                name = Counter(all_ents[i].name for i in mids).most_common(1)[0][0]
                types = Counter(all_ents[i].type for i in mids if all_ents[i].type != "OTHER")
                etype = types.most_common(1)[0][0] if types else "OTHER"
                desc = next((all_ents[i].desc for i in mids if all_ents[i].desc), None)
                vec = l2_normalize(np.mean([all_ents[i].vec for i in mids], axis=0))[0]
                eid = self.store.add_entity(name, etype, desc, vec)
                new_ids.append(eid)
                new_vecs.append(vec)
                for i in mids:
                    m = all_ents[i]
                    self.store.add_alias(eid, m.name)
                    self.store.add_mention(eid, m.chunk_id)
                    local[(m.chunk_id, normalize_name(m.name))] = eid
                self.stats["entities_new"] += 1
            self.stats["mentions"] += len(all_ents)
            if new_ids:
                self.fm.entities.add(new_ids, np.vstack(new_vecs))
            all_rels.sort(key=lambda r: (r.date or "", r.chunk_id))
            self._write_relations(all_rels, lambda cid, nm: local.get((cid, normalize_name(nm))))
            for did in doc_ids:
                self.store.set_document_status(did, "resolved")
        self.fm.save()
        out = {"documents": len(doc_ids), "seconds": round(time.perf_counter() - t0, 2),
               "judge_llm_calls": self.judge.calls - calls0}
        self.store.log_stage("graph_global_rebuild", out["seconds"], batch_id=batch_id,
                             llm_calls=out["judge_llm_calls"])
        return out
