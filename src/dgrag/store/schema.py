"""SQLite schema: the single source of truth (PLAN.md §10).

FAISS indexes and the networkx graph are *derived* views that can always be rebuilt from these tables.
Embeddings are stored as float32 BLOBs so FAISS can be rebuilt without re-running the embedder.
Bump SCHEMA_VERSION whenever the DDL changes and add a migration step in SQLiteStore._migrate().
"""

SCHEMA_VERSION = 3

DDL = """
-- ---------------------------------------------------------------- documents & chunks
CREATE TABLE IF NOT EXISTS documents (
    id            INTEGER PRIMARY KEY,
    content_hash  TEXT    NOT NULL UNIQUE,          -- dedupe key (idempotent ingestion)
    title         TEXT,
    author        TEXT,
    source        TEXT,
    url           TEXT    UNIQUE,                   -- MultiHop-RAG evidence references articles by url
    category      TEXT,
    published_at  TEXT,                             -- ISO-8601; drives temporal edges
    body          TEXT    NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'loaded' -- loaded -> chunked -> extracted -> resolved
                  CHECK (status IN ('loaded','chunked','extracted','resolved')),
    ingested_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_documents_published ON documents(published_at);
CREATE INDEX IF NOT EXISTS idx_documents_status    ON documents(status);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,               -- also the FAISS id in the chunk index
    doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    idx         INTEGER NOT NULL,                  -- position within the document
    text        TEXT    NOT NULL,
    n_tokens    INTEGER,
    char_start  INTEGER,
    char_end    INTEGER,
    embedding   BLOB,                              -- float32[dim], L2-normalised
    extracted   INTEGER NOT NULL DEFAULT 0,        -- 1 once LLM extraction has run on this chunk
    UNIQUE (doc_id, idx)
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

-- ---------------------------------------------------------------- entities
CREATE TABLE IF NOT EXISTS entities (
    id             INTEGER PRIMARY KEY,            -- also the FAISS id in the entity index
    name           TEXT    NOT NULL,               -- canonical display name
    name_norm      TEXT    NOT NULL,
    type           TEXT    NOT NULL,
    description    TEXT,
    embedding      BLOB,                           -- embedding of "name (type): description"
    mention_count  INTEGER NOT NULL DEFAULT 0,
    community_id   INTEGER REFERENCES communities(id) ON DELETE SET NULL,
    dirty          INTEGER NOT NULL DEFAULT 1,     -- description / community needs refresh
    created_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_entities_name_norm ON entities(name_norm);
CREATE INDEX IF NOT EXISTS idx_entities_type      ON entities(type);
CREATE INDEX IF NOT EXISTS idx_entities_dirty     ON entities(dirty) WHERE dirty = 1;

CREATE TABLE IF NOT EXISTS entity_aliases (
    alias_norm  TEXT    NOT NULL,
    entity_id   INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (alias_norm, entity_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS entity_mentions (
    entity_id  INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    chunk_id   INTEGER NOT NULL REFERENCES chunks(id)   ON DELETE CASCADE,
    PRIMARY KEY (entity_id, chunk_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_mentions_chunk ON entity_mentions(chunk_id);

-- ---------------------------------------------------------------- relations (temporal edges)
CREATE TABLE IF NOT EXISTS relations (
    id              INTEGER PRIMARY KEY,
    src_id          INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    tgt_id          INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    predicate       TEXT    NOT NULL,
    predicate_norm  TEXT    NOT NULL,
    description     TEXT,
    weight          REAL    NOT NULL DEFAULT 1.0,  -- number of supporting mentions
    valid_from      TEXT,                          -- earliest supporting article date
    valid_to        TEXT,                          -- NULL = still valid
    status          TEXT    NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active','superseded','conflicted')),
    superseded_by   INTEGER REFERENCES relations(id) ON DELETE SET NULL,
    embedding       BLOB,                          -- v3: bge vector of "src pred tgt: desc" (PLAN.md D12)
    first_seen      TEXT,
    last_seen       TEXT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_relations_src      ON relations(src_id);
CREATE INDEX IF NOT EXISTS idx_relations_tgt      ON relations(tgt_id);
CREATE INDEX IF NOT EXISTS idx_relations_src_pred ON relations(src_id, predicate_norm);
CREATE INDEX IF NOT EXISTS idx_relations_status   ON relations(status);

CREATE TABLE IF NOT EXISTS relation_evidence (
    relation_id  INTEGER NOT NULL REFERENCES relations(id) ON DELETE CASCADE,
    chunk_id     INTEGER NOT NULL REFERENCES chunks(id)    ON DELETE CASCADE,
    PRIMARY KEY (relation_id, chunk_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_evidence_chunk ON relation_evidence(chunk_id);

-- ---------------------------------------------------------------- communities (Phase 5)
CREATE TABLE IF NOT EXISTS communities (
    id            INTEGER PRIMARY KEY,
    level         INTEGER NOT NULL,
    parent_id     INTEGER REFERENCES communities(id) ON DELETE SET NULL,
    members_hash  TEXT,
    size          INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS community_summaries (
    community_id           INTEGER PRIMARY KEY REFERENCES communities(id) ON DELETE CASCADE,
    text                   TEXT,
    embedding              BLOB,
    edge_count_at_summary  INTEGER NOT NULL DEFAULT 0,
    pending_changes        REAL    NOT NULL DEFAULT 0,   -- weighted count of edge changes since summary
    staleness              REAL    NOT NULL DEFAULT 0,
    updated_at             TEXT
);

-- ---------------------------------------------------------------- raw LLM extractions (schema v2)
EXTRACTIONS_DDL_PLACEHOLDER

-- ---------------------------------------------------------------- bookkeeping
CREATE TABLE IF NOT EXISTS ingest_log (
    id         INTEGER PRIMARY KEY,
    batch_id   TEXT,
    doc_id     INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    stage      TEXT    NOT NULL,
    seconds    REAL    NOT NULL,
    llm_calls  INTEGER NOT NULL DEFAULT 0,
    created_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS meta (
    key    TEXT PRIMARY KEY,
    value  TEXT
);
"""

# v2: cache of per-chunk LLM extraction output. Extraction is the expensive step (~1 GPU-hour for the corpus),
# and its output does not depend on the graph, so it is stored once and the graph can be (re)built from it
# many times (e.g. static vs dynamic, different resolution thresholds) without re-running the LLM.
EXTRACTIONS_DDL = """
CREATE TABLE IF NOT EXISTS extractions (
    chunk_id           INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    prompt_version     TEXT    NOT NULL,
    model              TEXT    NOT NULL,
    raw                TEXT,                    -- exact LLM output
    entities_json      TEXT,                    -- validated entities
    relations_json     TEXT,                    -- validated relations
    n_entities         INTEGER NOT NULL DEFAULT 0,
    n_relations        INTEGER NOT NULL DEFAULT 0,
    n_dropped_entities INTEGER NOT NULL DEFAULT 0,
    n_dropped_relations INTEGER NOT NULL DEFAULT 0,
    ok                 INTEGER NOT NULL,        -- 1 = valid JSON
    finish_reason      TEXT,
    prompt_tokens      INTEGER,
    completion_tokens  INTEGER,
    seconds            REAL,
    created_at         TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
"""

# v3: LLM judge cache. Same question (e.g. "is 'AAPL' the same entity as 'Apple'?") is asked at most once;
# also lets us count how many LLM calls a build *would* need vs actually made.
JUDGE_DDL = """
CREATE TABLE IF NOT EXISTS judge_cache (
    key        TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,                    -- same_entity | relation_conflict
    decision   TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
"""

DDL = DDL.replace("EXTRACTIONS_DDL_PLACEHOLDER", EXTRACTIONS_DDL + JUDGE_DDL)

# Migrations: version -> SQL that upgrades (version-1) -> version. Applied in order by SQLiteStore._migrate.
MIGRATIONS: dict[int, str] = {
    2: EXTRACTIONS_DDL,
    3: "ALTER TABLE relations ADD COLUMN embedding BLOB;" + JUDGE_DDL,
}
