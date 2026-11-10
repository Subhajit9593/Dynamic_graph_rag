"""In-memory graph view (networkx) derived from SQLite (PLAN.md §10).

* MultiDiGraph: two entities can be linked by several predicates; edge key = relations.id.
* By default superseded relations are left out (what query-time traversal should see); 'active' and
  'conflicted' edges are both kept (a conflict is shown to the LLM with both sources). Superseded edges stay in
  SQLite for temporal questions.
* Kept in sync incrementally by the ingest pipeline via add_/update_ methods, never rebuilt per article.
"""

from __future__ import annotations

import logging
import sqlite3

import networkx as nx

from dgrag.store.sqlite_store import SQLiteStore

log = logging.getLogger(__name__)


class GraphView:
    def __init__(self, active_only: bool = True):
        self.g = nx.MultiDiGraph()
        self.active_only = active_only

    @classmethod
    def from_store(cls, store: SQLiteStore, active_only: bool = True) -> GraphView:
        gv = cls(active_only=active_only)
        for row in store.iter_entities():
            gv.upsert_entity(row)
        for row in store.iter_relations(status=None):
            gv.upsert_relation(row)
        log.info("Loaded graph: %d nodes, %d edges", gv.g.number_of_nodes(), gv.g.number_of_edges())
        return gv

    # ---------------------------------------------------------------- incremental sync
    def upsert_entity(self, row: sqlite3.Row | dict) -> None:
        self.g.add_node(
            int(row["id"]),
            name=row["name"],
            type=row["type"],
            mention_count=row["mention_count"],
            community_id=row["community_id"],
        )

    def upsert_relation(self, row: sqlite3.Row | dict) -> None:
        if self.active_only and row["status"] == "superseded":
            self.remove_relation(row)
            return
        self.g.add_edge(
            int(row["src_id"]),
            int(row["tgt_id"]),
            key=int(row["id"]),
            predicate=row["predicate"],
            weight=float(row["weight"]),
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            status=row["status"],
        )

    def remove_relation(self, row: sqlite3.Row | dict) -> None:
        u, v, k = int(row["src_id"]), int(row["tgt_id"]), int(row["id"])
        if self.g.has_edge(u, v, key=k):
            self.g.remove_edge(u, v, key=k)

    def remove_entity(self, entity_id: int) -> None:
        if entity_id in self.g:
            self.g.remove_node(entity_id)

    # ---------------------------------------------------------------- consistency
    def matches_store(self, store: SQLiteStore) -> bool:
        """True if node/edge sets equal what SQLite would produce (used in tests and `dgrag check`)."""
        nodes = {int(r["id"]) for r in store.iter_entities()}
        edges = {(int(r["src_id"]), int(r["tgt_id"]), int(r["id"])) for r in store.iter_relations(status=None)
                 if not (self.active_only and r["status"] == "superseded")}
        return set(self.g.nodes) == nodes and set(self.g.edges(keys=True)) == edges

    def stats(self) -> dict[str, int | float]:
        g = self.g
        n = g.number_of_nodes()
        return {
            "nodes": n,
            "edges": g.number_of_edges(),
            "weakly_connected_components": nx.number_weakly_connected_components(g) if n else 0,
            "isolated_nodes": nx.number_of_isolates(g),
            "avg_degree": (2 * g.number_of_edges() / n) if n else 0.0,
        }
