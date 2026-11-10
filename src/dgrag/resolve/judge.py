"""LLM judge for the uncertain cases (PLAN.md §6.1 step 3, §6.2). Answers are cached in SQLite.

Two questions:
* same_entity(a, b)      -> "yes" / "no"
* relation_conflict(...) -> "supersedes" / "coexists" / "contradicts"

`Judge` takes any `ask(messages, schema) -> dict` callable, so tests can plug in a fake without a GPU.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from typing import Any

from dgrag.store import SQLiteStore

log = logging.getLogger(__name__)

AskFn = Callable[[list[dict[str, str]], dict], dict | None]

SAME_ENTITY_SCHEMA = {"type": "object", "properties": {"same": {"type": "string", "enum": ["yes", "no"]}},
                      "required": ["same"]}
CONFLICT_SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string",
                   "enum": ["supersedes", "coexists", "contradicts"]}}, "required": ["verdict"]}


def _key(kind: str, *parts: str) -> str:
    return kind + ":" + hashlib.sha1("\x1f".join(parts).encode()).hexdigest()


class Judge:
    def __init__(self, store: SQLiteStore, ask: AskFn | None):
        self.store = store
        self.ask = ask
        self.calls = 0        # actual LLM calls
        self.cache_hits = 0
        self.unavailable = 0  # needed a judgement but no LLM configured -> conservative default

    def same_entity(self, a: dict[str, str], b: dict[str, str]) -> bool:
        """a, b: {"name", "type", "desc"}. Conservative default (no LLM): not the same."""
        pa = f"{a['name']} | {a['type']} | {a.get('desc') or ''}"
        pb = f"{b['name']} | {b['type']} | {b.get('desc') or ''}"
        key = _key("same", *sorted([pa, pb]))
        cached = self.store.judge_get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached == "yes"
        if self.ask is None:
            self.unavailable += 1
            return False
        msgs = [
            {"role": "system", "content": "You decide whether two entity mentions from news articles refer to the "
                                          "same real-world entity. Answer yes only if you are confident."},
            {"role": "user", "content": f"Mention A: {pa}\nMention B: {pb}\nSame real-world entity?"},
        ]
        out = self.ask(msgs, SAME_ENTITY_SCHEMA)
        self.calls += 1
        decision = "yes" if out and out.get("same") == "yes" else "no"
        self.store.judge_put(key, "same_entity", decision)
        return decision == "yes"

    def relation_conflict(self, old: dict[str, Any], new: dict[str, Any]) -> str:
        """old/new: {"text", "date"}. Conservative default (no LLM): coexists (nothing is invalidated)."""
        key = _key("conflict", old["text"], str(old.get("date")), new["text"], str(new.get("date")))
        cached = self.store.judge_get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        if self.ask is None:
            self.unavailable += 1
            return "coexists"
        msgs = [
            {"role": "system", "content": (
                "Two facts come from news articles published on different dates. Decide:\n"
                "supersedes  = the newer fact replaces the older one (the older is no longer true, e.g. a new CEO)\n"
                "coexists    = both can be true at the same time\n"
                "contradicts = they disagree about the same moment")},
            {"role": "user", "content": f"Older fact ({old.get('date')}): {old['text']}\n"
                                        f"Newer fact ({new.get('date')}): {new['text']}"},
        ]
        out = self.ask(msgs, CONFLICT_SCHEMA)
        self.calls += 1
        verdict = out.get("verdict") if out else None
        verdict = verdict if verdict in ("supersedes", "coexists", "contradicts") else "coexists"
        self.store.judge_put(key, "relation_conflict", verdict)
        return verdict

    def stats(self) -> dict[str, int]:
        return {"judge_llm_calls": self.calls, "judge_cache_hits": self.cache_hits,
                "judge_unavailable": self.unavailable}


def llm_ask(client: Any, max_tokens: int = 32) -> AskFn:
    """Adapter: LlamaClient -> AskFn."""
    def ask(messages: list[dict[str, str]], schema: dict) -> dict | None:
        try:
            obj, _ = client.chat_json(messages, schema, max_tokens=max_tokens, temperature=0.0)
            return obj
        except Exception as e:  # never let a judge failure kill a build
            log.warning("judge call failed: %s", e)
            return None
    return ask


def dumps(o: Any) -> str:
    return json.dumps(o, ensure_ascii=False, sort_keys=True)
