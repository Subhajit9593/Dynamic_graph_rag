"""FAISS vector indexes as a *derived*, rebuildable view of SQLite (PLAN.md §10).

* ``IndexIDMap2(IndexFlatIP)``: exact inner-product search on L2-normalised vectors (= cosine).
  Vector id == SQLite row id, so search results map straight back to rows.
  IDMap2 (vs IDMap) also supports ``reconstruct(id)`` - handy for debugging and tests.
* Incremental: ``add`` / ``remove`` never rebuild the index.
* Crash safety: the write order is "commit SQLite, then update FAISS, then save periodically".
  If the process dies between steps, ``reconcile`` on next start adds missing ids and drops
  extra ids by diffing against SQLite, instead of trusting the on-disk file.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import faiss
import numpy as np

from dgrag.store.sqlite_store import SQLiteStore

log = logging.getLogger(__name__)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class VectorIndex:
    """One FAISS index (e.g. 'chunks' or 'entities') with id-mapped add/remove/search and persistence."""

    def __init__(self, name: str, dim: int, path: Path | None = None):
        self.name = name
        self.dim = dim
        self.path = Path(path) if path else None
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self._ids: set[int] = set()  # cached id set: avoids O(n) id_map copies on every add()
        self._dirty = False

    # ---------------------------------------------------------------- basic ops
    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    def ids(self) -> set[int]:
        return set(self._ids)

    def __contains__(self, id_: int) -> bool:
        return int(id_) in self._ids

    def _ids_from_index(self) -> set[int]:
        if self.index.ntotal == 0:
            return set()
        return set(faiss.vector_to_array(self.index.id_map).astype(np.int64).tolist())

    def add(self, ids: np.ndarray | list[int], vectors: np.ndarray, normalize: bool = True) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        vecs = l2_normalize(vectors) if normalize else np.ascontiguousarray(vectors, dtype=np.float32)
        if vecs.shape[0] != ids.shape[0]:
            raise ValueError(f"{self.name}: {ids.shape[0]} ids but {vecs.shape[0]} vectors")
        if vecs.shape[1] != self.dim:
            raise ValueError(f"{self.name}: vector dim {vecs.shape[1]} != index dim {self.dim}")
        if len(ids) == 0:
            return
        if len(set(ids.tolist())) != len(ids):
            raise ValueError(f"{self.name}: duplicate ids in a single add() call")
        # IDMap does not reject duplicate ids itself; replacing keeps 1 vector per id (upsert semantics).
        id_list = ids.tolist()
        existing = self._ids.intersection(id_list)
        if existing:
            self.remove(list(existing))
        self.index.add_with_ids(vecs, ids)
        self._ids.update(id_list)
        self._dirty = True

    def remove(self, ids: list[int] | np.ndarray) -> int:
        ids = np.asarray(list(ids), dtype=np.int64)
        if len(ids) == 0:
            return 0
        n = int(self.index.remove_ids(faiss.IDSelectorBatch(ids)))
        self._ids.difference_update(ids.tolist())
        if n:
            self._dirty = True
        return n

    def search(self, queries: np.ndarray, k: int, normalize: bool = True) -> list[list[tuple[int, float]]]:
        """Return, per query, a list of (id, cosine_score) sorted by score desc. Missing slots are dropped."""
        if self.ntotal == 0:
            q = queries if queries.ndim == 2 else queries[None, :]
            return [[] for _ in range(q.shape[0])]
        q = l2_normalize(queries) if normalize else np.ascontiguousarray(queries, dtype=np.float32)
        scores, ids = self.index.search(q, min(k, self.ntotal))
        out: list[list[tuple[int, float]]] = []
        for row_ids, row_scores in zip(ids, scores):
            out.append([(int(i), float(s)) for i, s in zip(row_ids, row_scores) if i != -1])
        return out

    def reconstruct(self, id_: int) -> np.ndarray:
        return self.index.reconstruct(int(id_))

    def reset(self) -> None:
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))
        self._ids = set()
        self._dirty = True

    # ---------------------------------------------------------------- persistence
    def save(self, force: bool = False) -> None:
        if self.path is None or (not self._dirty and not force):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        faiss.write_index(self.index, str(tmp))
        os.replace(tmp, self.path)  # atomic rename: never leave a half-written index on disk
        meta = {"name": self.name, "dim": self.dim, "ntotal": self.ntotal}
        self.path.with_suffix(".json").write_text(json.dumps(meta))
        self._dirty = False
        log.debug("Saved %s index (%d vectors) -> %s", self.name, self.ntotal, self.path)

    def load(self) -> bool:
        """Load from disk. Returns False (and keeps an empty index) if missing or incompatible."""
        if self.path is None or not self.path.exists():
            return False
        try:
            idx = faiss.read_index(str(self.path))
        except Exception as e:  # corrupt file -> rebuild from SQLite
            log.warning("Could not read %s (%s); will rebuild from SQLite", self.path, e)
            return False
        if idx.d != self.dim:
            log.warning("%s index dim %d != configured %d; will rebuild", self.name, idx.d, self.dim)
            return False
        # NOTE: faiss.read_index already returns the concrete subclass and OWNS the C++ object.
        # Do NOT wrap it in faiss.downcast_index(): that returns a non-owning proxy, so the memory is freed
        # when `idx` is garbage-collected -> segfault on the next remove_ids/search (PLAN.md Issue #4).
        if not isinstance(idx, faiss.IndexIDMap2):
            log.warning("%s index has unexpected type %s; will rebuild", self.name, type(idx).__name__)
            self.reset()
            return False
        self.index = idx
        self._ids = self._ids_from_index()
        self._dirty = False
        return True


class FaissManager:
    """Owns the 'chunks' and 'entities' indexes and keeps them consistent with SQLite."""

    TABLES = ("chunks", "entities", "relations")

    def __init__(self, store: SQLiteStore, faiss_dir: str | Path, dim: int):
        self.store = store
        self.dir = Path(faiss_dir)
        self.dim = dim
        self.indexes = {t: VectorIndex(t, dim, self.dir / f"{t}.faiss") for t in self.TABLES}

    def __getitem__(self, table: str) -> VectorIndex:
        return self.indexes[table]

    @property
    def chunks(self) -> VectorIndex:
        return self.indexes["chunks"]

    @property
    def entities(self) -> VectorIndex:
        return self.indexes["entities"]

    @property
    def relations(self) -> VectorIndex:
        return self.indexes["relations"]

    def open(self) -> dict[str, dict[str, int]]:
        """Load each index from disk (if present) and reconcile it with SQLite."""
        report = {}
        for table, idx in self.indexes.items():
            loaded = idx.load()
            report[table] = {"loaded_from_disk": int(loaded), **self.reconcile(table)}
        return report

    def reconcile(self, table: str) -> dict[str, int]:
        """Make the FAISS id set equal to the set of SQLite rows that have embeddings.

        Only the *difference* is touched: missing ids are added from SQLite BLOBs, ids no longer in
        SQLite (e.g. deleted documents) are removed. Vectors that exist in both are trusted.
        """
        idx = self.indexes[table]
        truth = self.store.embedding_ids(table)
        have = idx.ids()
        missing = truth - have
        extra = have - truth
        if extra:
            idx.remove(list(extra))
        if missing:
            ids, vecs = self.store.load_embeddings(table, ids=sorted(missing), dim=self.dim)
            idx.add(ids, vecs)
        if missing or extra:
            log.info("Reconciled %s index: +%d added, -%d removed (now %d)", table, len(missing), len(extra),
                     idx.ntotal)
        return {"added": len(missing), "removed": len(extra), "ntotal": idx.ntotal}

    def rebuild(self, table: str) -> int:
        """Full rebuild from SQLite (used by tests / when the embedder changes)."""
        idx = self.indexes[table]
        idx.reset()
        ids, vecs = self.store.load_embeddings(table, dim=self.dim)
        if len(ids):
            idx.add(ids, vecs)
        return idx.ntotal

    def is_consistent(self, table: str) -> bool:
        return self.indexes[table].ids() == self.store.embedding_ids(table)

    def save(self, force: bool = False) -> None:
        for idx in self.indexes.values():
            idx.save(force=force)
