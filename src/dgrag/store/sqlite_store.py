"""SQLite repository: the single source of truth for documents, chunks, entities and relations.

Design notes (PLAN.md §10):
* Connection runs in autocommit mode (isolation_level=None); multi-statement units of work use
  ``with store.transaction():`` which issues BEGIN IMMEDIATE / COMMIT / ROLLBACK explicitly.
  Transactions nest: inner blocks become SAVEPOINTs, so a failing inner step can roll back on its own.
* WAL journal mode -> readers never block the writer; synchronous=NORMAL is crash-safe under WAL.
* Embeddings are float32 BLOBs; FAISS ids == SQLite row ids, so no id-mapping table is needed.
* This layer offers *primitives* only (insert / lookup / update). Policy (entity resolution, conflict
  handling, staleness) lives in resolve/ and community/.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dgrag.store.schema import DDL, MIGRATIONS, SCHEMA_VERSION
from dgrag.text_utils import content_hash, normalize_name, normalize_predicate

log = logging.getLogger(__name__)

EMB_DTYPE = np.float32


def to_blob(vec: np.ndarray | None) -> bytes | None:
    if vec is None:
        return None
    return np.ascontiguousarray(vec, dtype=EMB_DTYPE).tobytes()


def from_blob(blob: bytes | None) -> np.ndarray | None:
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=EMB_DTYPE)


@dataclass
class ChunkInput:
    idx: int
    text: str
    n_tokens: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    embedding: np.ndarray | None = None


class SQLiteStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self._tx_depth = 0
        self._configure()
        self._init_schema()

    # ------------------------------------------------------------------ lifecycle
    def _configure(self) -> None:
        c = self.conn
        c.execute("PRAGMA foreign_keys = ON")
        if str(self.path) != ":memory:":
            c.execute("PRAGMA journal_mode = WAL")
        c.execute("PRAGMA synchronous = NORMAL")
        c.execute("PRAGMA temp_store = MEMORY")
        c.execute("PRAGMA cache_size = -65536")  # 64 MB page cache

    def _init_schema(self) -> None:
        current = self._read_schema_version()
        if current is None:
            self.conn.executescript(DDL)
            self.set_meta("schema_version", str(SCHEMA_VERSION))
            log.debug("Initialised schema v%d at %s", SCHEMA_VERSION, self.path)
        elif current < SCHEMA_VERSION:
            self._migrate(current)
        elif current > SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema v{current} is newer than code (v{SCHEMA_VERSION}); update the code."
            )

    def _read_schema_version(self) -> int | None:
        row = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'"
        ).fetchone()
        if row is None:
            return None
        v = self.get_meta("schema_version")
        return int(v) if v is not None else None

    def _migrate(self, from_version: int) -> None:
        """Apply MIGRATIONS[v] for v in (from_version, SCHEMA_VERSION], each in its own transaction."""
        for v in range(from_version + 1, SCHEMA_VERSION + 1):
            if v not in MIGRATIONS:
                raise RuntimeError(f"No migration to schema v{v}")
            with self.transaction():
                for stmt in MIGRATIONS[v].split(";"):
                    if stmt.strip():
                        self.conn.execute(stmt)
                self.set_meta("schema_version", str(v))
            log.info("Migrated database schema v%d -> v%d", v - 1, v)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ transactions
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Atomic unit of work. Nested calls use SAVEPOINTs."""
        if self._tx_depth == 0:
            self.conn.execute("BEGIN IMMEDIATE")
            name = None
        else:
            name = f"sp_{self._tx_depth}"
            self.conn.execute(f"SAVEPOINT {name}")
        self._tx_depth += 1
        try:
            yield self.conn
        except BaseException:
            self._tx_depth -= 1
            if name is None:
                self.conn.execute("ROLLBACK")
            else:
                self.conn.execute(f"ROLLBACK TO {name}")
                self.conn.execute(f"RELEASE {name}")
            raise
        else:
            self._tx_depth -= 1
            if name is None:
                self.conn.execute("COMMIT")
            else:
                self.conn.execute(f"RELEASE {name}")

    @property
    def in_transaction(self) -> bool:
        return self._tx_depth > 0

    # ------------------------------------------------------------------ meta
    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    # ------------------------------------------------------------------ documents
    def add_document(
        self,
        body: str,
        *,
        title: str | None = None,
        author: str | None = None,
        source: str | None = None,
        url: str | None = None,
        category: str | None = None,
        published_at: str | None = None,
    ) -> tuple[int, bool]:
        """Insert a document if its content is new.

        Returns (doc_id, inserted). Re-adding identical content (or the same url) returns the existing id
        with inserted=False, which makes ingestion idempotent.
        """
        h = content_hash(body)
        row = self.conn.execute(
            "SELECT id FROM documents WHERE content_hash = ? OR (url IS NOT NULL AND url = ?)",
            (h, url),
        ).fetchone()
        if row:
            return int(row["id"]), False
        cur = self.conn.execute(
            "INSERT INTO documents(content_hash, title, author, source, url, category, published_at, body) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (h, title, author, source, url, category, published_at, body),
        )
        return int(cur.lastrowid), True

    def get_document(self, doc_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()

    def get_document_by_url(self, url: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM documents WHERE url = ?", (url,)).fetchone()

    def set_document_status(self, doc_id: int, status: str) -> None:
        self.conn.execute("UPDATE documents SET status = ? WHERE id = ?", (status, doc_id))

    def documents_with_status(self, status: str) -> list[int]:
        rows = self.conn.execute(
            "SELECT id FROM documents WHERE status = ? ORDER BY published_at, id", (status,)
        ).fetchall()
        return [int(r["id"]) for r in rows]

    def delete_document(self, doc_id: int) -> None:
        """Cascades to chunks, mentions and evidence. Caller must also remove chunk vectors from FAISS."""
        self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    # ------------------------------------------------------------------ chunks
    def add_chunks(self, doc_id: int, chunks: Sequence[ChunkInput]) -> list[int]:
        ids: list[int] = []
        with self.transaction():
            for ch in chunks:
                cur = self.conn.execute(
                    "INSERT INTO chunks(doc_id, idx, text, n_tokens, char_start, char_end, embedding) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (doc_id, ch.idx, ch.text, ch.n_tokens, ch.char_start, ch.char_end, to_blob(ch.embedding)),
                )
                ids.append(int(cur.lastrowid))
        return ids

    def get_chunks(self, chunk_ids: Iterable[int]) -> list[sqlite3.Row]:
        ids = list(chunk_ids)
        if not ids:
            return []
        q = f"SELECT * FROM chunks WHERE id IN ({','.join('?' * len(ids))})"
        by_id = {r["id"]: r for r in self.conn.execute(q, ids).fetchall()}
        return [by_id[i] for i in ids if i in by_id]  # preserve caller order

    def chunks_for_document(self, doc_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM chunks WHERE doc_id = ? ORDER BY idx", (doc_id,)
        ).fetchall()

    def set_chunk_embedding(self, chunk_id: int, vec: np.ndarray) -> None:
        self.conn.execute("UPDATE chunks SET embedding = ? WHERE id = ?", (to_blob(vec), chunk_id))

    def mark_chunk_extracted(self, chunk_id: int) -> None:
        self.conn.execute("UPDATE chunks SET extracted = 1 WHERE id = ?", (chunk_id,))

    # ------------------------------------------------------------------ entities
    def add_entity(
        self,
        name: str,
        type_: str,
        description: str | None = None,
        embedding: np.ndarray | None = None,
    ) -> int:
        """Create a new entity and register its own name as an alias."""
        norm = normalize_name(name)
        with self.transaction():
            cur = self.conn.execute(
                "INSERT INTO entities(name, name_norm, type, description, embedding) VALUES (?, ?, ?, ?, ?)",
                (name, norm, type_, description, to_blob(embedding)),
            )
            eid = int(cur.lastrowid)
            self.add_alias(eid, name)
        return eid

    def get_entity(self, entity_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()

    def get_entities(self, entity_ids: Iterable[int]) -> list[sqlite3.Row]:
        ids = list(entity_ids)
        if not ids:
            return []
        q = f"SELECT * FROM entities WHERE id IN ({','.join('?' * len(ids))})"
        by_id = {r["id"]: r for r in self.conn.execute(q, ids).fetchall()}
        return [by_id[i] for i in ids if i in by_id]

    def add_alias(self, entity_id: int, alias: str) -> None:
        norm = normalize_name(alias)
        if norm:
            self.conn.execute(
                "INSERT OR IGNORE INTO entity_aliases(alias_norm, entity_id) VALUES (?, ?)", (norm, entity_id)
            )

    def find_entities_by_alias(self, name: str, type_: str | None = None) -> list[sqlite3.Row]:
        """Exact match on normalised alias (step 2 of entity resolution, PLAN.md §6.1)."""
        norm = normalize_name(name)
        sql = (
            "SELECT e.* FROM entity_aliases a JOIN entities e ON e.id = a.entity_id "
            "WHERE a.alias_norm = ?"
        )
        args: list[Any] = [norm]
        if type_ is not None:
            sql += " AND e.type = ?"
            args.append(type_)
        sql += " ORDER BY e.mention_count DESC"
        return self.conn.execute(sql, args).fetchall()

    def aliases_of(self, entity_id: int) -> list[str]:
        rows = self.conn.execute(
            "SELECT alias_norm FROM entity_aliases WHERE entity_id = ? ORDER BY alias_norm", (entity_id,)
        ).fetchall()
        return [r["alias_norm"] for r in rows]

    def update_entity(
        self,
        entity_id: int,
        *,
        description: str | None = None,
        embedding: np.ndarray | None = None,
        dirty: bool | None = None,
        community_id: int | None = None,
        type_: str | None = None,
    ) -> None:
        sets, args = ["updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')"], []
        if type_ is not None:
            sets.append("type = ?")
            args.append(type_)
        if description is not None:
            sets.append("description = ?")
            args.append(description)
        if embedding is not None:
            sets.append("embedding = ?")
            args.append(to_blob(embedding))
        if dirty is not None:
            sets.append("dirty = ?")
            args.append(int(dirty))
        if community_id is not None:
            sets.append("community_id = ?")
            args.append(community_id)
        args.append(entity_id)
        self.conn.execute(f"UPDATE entities SET {', '.join(sets)} WHERE id = ?", args)

    def add_mention(self, entity_id: int, chunk_id: int) -> bool:
        """Link entity -> chunk. Returns True if this is a new mention (mention_count incremented)."""
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO entity_mentions(entity_id, chunk_id) VALUES (?, ?)", (entity_id, chunk_id)
        )
        if cur.rowcount:
            self.conn.execute(
                "UPDATE entities SET mention_count = mention_count + 1, dirty = 1, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id = ?",
                (entity_id,),
            )
            return True
        return False

    def dirty_entities(self) -> list[int]:
        return [int(r["id"]) for r in self.conn.execute("SELECT id FROM entities WHERE dirty = 1")]

    # ------------------------------------------------------------------ relations
    def add_relation(
        self,
        src_id: int,
        tgt_id: int,
        predicate: str,
        *,
        description: str | None = None,
        weight: float = 1.0,
        valid_from: str | None = None,
        observed_at: str | None = None,
        embedding: np.ndarray | None = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO relations(src_id, tgt_id, predicate, predicate_norm, description, weight, "
            "valid_from, first_seen, last_seen, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (src_id, tgt_id, predicate, normalize_predicate(predicate), description, weight,
             valid_from or observed_at, observed_at, observed_at, to_blob(embedding)),
        )
        return int(cur.lastrowid)

    def get_relation(self, relation_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM relations WHERE id = ?", (relation_id,)).fetchone()

    def find_relations(
        self,
        src_id: int,
        *,
        tgt_id: int | None = None,
        predicate: str | None = None,
        status: str | None = "active",
    ) -> list[sqlite3.Row]:
        sql, args = "SELECT * FROM relations WHERE src_id = ?", [src_id]
        if tgt_id is not None:
            sql += " AND tgt_id = ?"
            args.append(tgt_id)
        if predicate is not None:
            sql += " AND predicate_norm = ?"
            args.append(normalize_predicate(predicate))
        if status is not None:
            sql += " AND status = ?"
            args.append(status)
        return self.conn.execute(sql, args).fetchall()

    def reinforce_relation(self, relation_id: int, observed_at: str | None, delta: float = 1.0) -> None:
        """Existing edge seen again: bump weight and widen the observation window."""
        self.conn.execute(
            "UPDATE relations SET weight = weight + ?, "
            "first_seen = CASE WHEN first_seen IS NULL OR ? < first_seen THEN ? ELSE first_seen END, "
            "last_seen  = CASE WHEN last_seen  IS NULL OR ? > last_seen  THEN ? ELSE last_seen  END, "
            "valid_from = CASE WHEN valid_from IS NULL OR ? < valid_from THEN ? ELSE valid_from END "
            "WHERE id = ?",
            (delta, observed_at, observed_at, observed_at, observed_at, observed_at, observed_at, relation_id),
        )

    def supersede_relation(self, old_id: int, new_id: int, valid_to: str | None) -> None:
        """Close an edge's validity interval instead of deleting it (PLAN.md §6.2)."""
        self.conn.execute(
            "UPDATE relations SET status = 'superseded', valid_to = ?, superseded_by = ? WHERE id = ?",
            (valid_to, new_id, old_id),
        )

    def set_relation_status(self, relation_id: int, status: str) -> None:
        self.conn.execute("UPDATE relations SET status = ? WHERE id = ?", (status, relation_id))

    def add_evidence(self, relation_id: int, chunk_id: int) -> bool:
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO relation_evidence(relation_id, chunk_id) VALUES (?, ?)", (relation_id, chunk_id)
        )
        return bool(cur.rowcount)

    def evidence_chunks(self, relation_id: int) -> list[int]:
        rows = self.conn.execute(
            "SELECT chunk_id FROM relation_evidence WHERE relation_id = ? ORDER BY chunk_id", (relation_id,)
        ).fetchall()
        return [int(r["chunk_id"]) for r in rows]

    def iter_relations(self, status: str | None = "active") -> Iterator[sqlite3.Row]:
        if status is None:
            yield from self.conn.execute("SELECT * FROM relations")
        else:
            yield from self.conn.execute("SELECT * FROM relations WHERE status = ?", (status,))

    def iter_entities(self) -> Iterator[sqlite3.Row]:
        yield from self.conn.execute(
            "SELECT id, name, name_norm, type, description, mention_count, community_id FROM entities"
        )

    # ------------------------------------------------------------------ extractions (LLM output cache)
    def chunks_pending_extraction(self, prompt_version: str, limit: int | None = None) -> list[int]:
        """Chunk ids without an OK extraction for this prompt version, in chronological document order."""
        sql = (
            "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id "
            "LEFT JOIN extractions x ON x.chunk_id = c.id AND x.prompt_version = ? AND x.ok = 1 "
            "WHERE x.chunk_id IS NULL ORDER BY d.published_at, d.id, c.idx"
        )
        args: list[Any] = [prompt_version]
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        return [int(r[0]) for r in self.conn.execute(sql, args)]

    def save_extraction(self, chunk_id: int, *, prompt_version: str, model: str, raw: str | None,
                        entities_json: str | None, relations_json: str | None, n_entities: int, n_relations: int,
                        n_dropped_entities: int, n_dropped_relations: int, ok: bool, finish_reason: str | None,
                        prompt_tokens: int | None, completion_tokens: int | None, seconds: float | None) -> None:
        """Upsert (a failed attempt can later be overwritten by a successful retry)."""
        self.conn.execute(
            "INSERT INTO extractions(chunk_id, prompt_version, model, raw, entities_json, relations_json, "
            "n_entities, n_relations, n_dropped_entities, n_dropped_relations, ok, finish_reason, prompt_tokens, "
            "completion_tokens, seconds) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(chunk_id) DO UPDATE SET prompt_version=excluded.prompt_version, model=excluded.model, "
            "raw=excluded.raw, entities_json=excluded.entities_json, relations_json=excluded.relations_json, "
            "n_entities=excluded.n_entities, n_relations=excluded.n_relations, "
            "n_dropped_entities=excluded.n_dropped_entities, n_dropped_relations=excluded.n_dropped_relations, "
            "ok=excluded.ok, finish_reason=excluded.finish_reason, prompt_tokens=excluded.prompt_tokens, "
            "completion_tokens=excluded.completion_tokens, seconds=excluded.seconds, "
            "created_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')",
            (chunk_id, prompt_version, model, raw, entities_json, relations_json, n_entities, n_relations,
             n_dropped_entities, n_dropped_relations, int(ok), finish_reason, prompt_tokens, completion_tokens,
             seconds),
        )
        if ok:
            self.mark_chunk_extracted(chunk_id)

    def get_extraction(self, chunk_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM extractions WHERE chunk_id = ?", (chunk_id,)).fetchone()

    def give_up_failed_extractions(self) -> int:
        """Chunks whose extraction still fails after retries (e.g. the model loops on a long fixture list) are marked
        done with zero facts, so their documents can move on. The chunk text stays searchable by vector search."""
        cur = self.conn.execute(
            "UPDATE chunks SET extracted = 1 WHERE extracted = 0 AND id IN (SELECT chunk_id FROM extractions WHERE ok = 0)")
        return cur.rowcount

    def mark_documents_extracted(self) -> int:
        """Documents whose every chunk is extracted move chunked -> extracted. Returns rows changed."""
        cur = self.conn.execute(
            "UPDATE documents SET status = 'extracted' WHERE status = 'chunked' AND NOT EXISTS "
            "(SELECT 1 FROM chunks c WHERE c.doc_id = documents.id AND c.extracted = 0)"
        )
        return cur.rowcount

    def extraction_stats(self) -> dict[str, Any]:
        r = self.conn.execute(
            "SELECT COUNT(*), SUM(ok), SUM(n_entities), SUM(n_relations), SUM(n_dropped_entities), "
            "SUM(n_dropped_relations), SUM(finish_reason = 'length'), AVG(completion_tokens), AVG(prompt_tokens) "
            "FROM extractions"
        ).fetchone()
        n, ok, ne, nr, de, dr, trunc, avg_out, avg_in = (x or 0 for x in r)
        return {
            "chunks_total": self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            "extracted": n, "ok": ok, "failed": n - ok, "truncated": trunc,
            "entities": ne, "relations": nr,
            "dropped_entities": de, "dropped_relations": dr,
            "relation_drop_rate": round(dr / (nr + dr), 3) if (nr + dr) else 0.0,
            "avg_prompt_tokens": round(avg_in), "avg_completion_tokens": round(avg_out),
        }

    def find_relations_to(self, tgt_id: int, *, predicate: str | None = None, status: str | None = "active"
                          ) -> list[sqlite3.Row]:
        sql, args = "SELECT * FROM relations WHERE tgt_id = ?", [tgt_id]
        if predicate is not None:
            sql += " AND predicate_norm = ?"
            args.append(normalize_predicate(predicate))
        if status is not None:
            sql += " AND status = ?"
            args.append(status)
        return self.conn.execute(sql, args).fetchall()

    def relations_between(self, src_id: int, tgt_id: int, status: str | None = "active") -> list[sqlite3.Row]:
        return self.find_relations(src_id, tgt_id=tgt_id, status=status)

    # ------------------------------------------------------------------ judge cache
    def judge_get(self, key: str) -> str | None:
        r = self.conn.execute("SELECT decision FROM judge_cache WHERE key = ?", (key,)).fetchone()
        return r[0] if r else None

    def judge_put(self, key: str, kind: str, decision: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO judge_cache(key, kind, decision) VALUES (?, ?, ?)",
                          (key, kind, decision))

    def clear_graph(self) -> None:
        """Delete the graph (entities, relations, mentions, evidence, communities); keep docs/chunks/extractions."""
        with self.transaction():
            for t in ("relation_evidence", "entity_mentions", "entity_aliases", "relations", "entities",
                      "community_summaries", "communities"):
                self.conn.execute(f"DELETE FROM {t}")
            self.conn.execute("UPDATE documents SET status = 'extracted' WHERE status = 'resolved'")

    # ------------------------------------------------------------------ embeddings (for FAISS rebuild)
    _EMB_TABLES = {"chunks": "chunks", "entities": "entities", "relations": "relations"}

    def embedding_ids(self, table: str) -> set[int]:
        t = self._EMB_TABLES[table]
        return {int(r[0]) for r in self.conn.execute(f"SELECT id FROM {t} WHERE embedding IS NOT NULL")}

    def load_embeddings(
        self, table: str, ids: Iterable[int] | None = None, dim: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (ids int64[n], vectors float32[n, dim]) for rows with embeddings."""
        t = self._EMB_TABLES[table]
        if ids is None:
            rows = self.conn.execute(
                f"SELECT id, embedding FROM {t} WHERE embedding IS NOT NULL ORDER BY id"
            ).fetchall()
        else:
            id_list = list(ids)
            rows = []
            for i in range(0, len(id_list), 900):  # stay under SQLite's host-parameter limit
                part = id_list[i : i + 900]
                rows += self.conn.execute(
                    f"SELECT id, embedding FROM {t} WHERE embedding IS NOT NULL "
                    f"AND id IN ({','.join('?' * len(part))})",
                    part,
                ).fetchall()
        if not rows:
            return np.empty(0, dtype=np.int64), np.empty((0, dim or 0), dtype=EMB_DTYPE)
        out_ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
        vecs = np.vstack([from_blob(r[1]) for r in rows]).astype(EMB_DTYPE, copy=False)
        if dim is not None and vecs.shape[1] != dim:
            raise ValueError(f"{table}: stored embedding dim {vecs.shape[1]} != expected {dim}")
        return out_ids, vecs

    # ------------------------------------------------------------------ ingest log & stats
    def log_stage(
        self, stage: str, seconds: float, *, doc_id: int | None = None, batch_id: str | None = None,
        llm_calls: int = 0,
    ) -> None:
        self.conn.execute(
            "INSERT INTO ingest_log(batch_id, doc_id, stage, seconds, llm_calls) VALUES (?, ?, ?, ?, ?)",
            (batch_id, doc_id, stage, seconds, llm_calls),
        )

    def stats(self) -> dict[str, Any]:
        q = lambda sql: self.conn.execute(sql).fetchone()[0]  # noqa: E731
        return {
            "documents": q("SELECT COUNT(*) FROM documents"),
            "documents_by_status": dict(
                self.conn.execute("SELECT status, COUNT(*) FROM documents GROUP BY status").fetchall()
            ),
            "date_range": tuple(self.conn.execute(
                "SELECT MIN(published_at), MAX(published_at) FROM documents").fetchone()),
            "chunks": q("SELECT COUNT(*) FROM chunks"),
            "chunks_embedded": q("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"),
            "entities": q("SELECT COUNT(*) FROM entities"),
            "aliases": q("SELECT COUNT(*) FROM entity_aliases"),
            "relations_active": q("SELECT COUNT(*) FROM relations WHERE status = 'active'"),
            "relations_superseded": q("SELECT COUNT(*) FROM relations WHERE status = 'superseded'"),
            "relations_conflicted": q("SELECT COUNT(*) FROM relations WHERE status = 'conflicted'"),
            "communities": q("SELECT COUNT(*) FROM communities"),
            "extractions_ok": q("SELECT COUNT(*) FROM extractions WHERE ok = 1"),
            "schema_version": self.get_meta("schema_version"),
        }
