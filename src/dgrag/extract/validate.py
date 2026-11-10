"""Clean and check one chunk's LLM output before it is used (PLAN.md Issue #6).

The JSON schema guarantees the *shape* of the output; this module checks the *content*:
* entities: drop empty / over-long / purely numeric / generic names; merge duplicates within the chunk
* relations: both ends must be extracted entities. Two repairs:
    1. a short form is mapped onto a unique full name ("Altman" -> "Sam Altman");
    2. an endpoint the model forgot to list as an entity is auto-added IF it looks like a proper name
       (capitalised words, not "X and Y") AND appears verbatim in the chunk text (grounding check -> no
       hallucinated names). Type = OTHER (unknown).
  Relations to concepts ("social decay", "health") are dropped on purpose. Self-loops and duplicates dropped.
Every drop is counted by reason, so prompt quality can be measured.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from dgrag.extract.prompts import ENTITY_TYPES
from dgrag.text_utils import normalize_name, normalize_predicate, normalize_whitespace

# amounts / quantities: "63", "$565 million", "2.6 per cent", "€1.2bn", "10%"
_NUMERIC = re.compile(
    r"^[\s$€£¥¢+\-~(]*\d[\d\s.,:/%¢)\-]*"
    r"(?:k|m|bn|million|billion|trillion|thousand|hundred|percent|per cent|pct|points?|yards?|cents?)?\.?\s*$",
    re.IGNORECASE,
)
GENERIC_NAMES = {
    "shares", "share", "stock", "stocks", "company", "companies", "team", "teams", "fans", "market", "markets",
    "government", "people", "players", "player", "investors", "customers", "users", "officials", "sector",
    "industry", "deal", "report", "season", "game", "match", "league", "court", "police", "ceo", "coach",
    "president", "he", "she", "they", "it", "we", "i", "you",
}
MAX_NAME_CHARS = 80
_NAME_CONNECTORS = {"of", "the", "de", "for", "&", "-", "von", "van", "da", "del", "la", "le", "on", "at"}


def looks_like_proper_name(s: str) -> bool:
    toks = s.split()
    if not 1 <= len(toks) <= 6 or " and " in f" {s.lower()} ":
        return False
    first = toks[0]
    if first[0].isdigit() and not any(ch.isalpha() for ch in first):  # "49ers", "3M" ok; "2,035 GWh" not
        return False
    if not (first[0].isupper() or first[0].isdigit()):
        return False
    return all(t[0].isupper() or t[0].isdigit() or t.lower() in _NAME_CONNECTORS for t in toks)
MAX_DESC_CHARS = 300


@dataclass
class ValidatedExtraction:
    entities: list[dict[str, str]] = field(default_factory=list)
    relations: list[dict[str, str]] = field(default_factory=list)
    dropped_entities: Counter = field(default_factory=Counter)   # reason -> count
    dropped_relations: Counter = field(default_factory=Counter)
    repaired_relations: int = 0
    auto_added_entities: int = 0


def _clean(s: Any, limit: int) -> str:
    return normalize_whitespace(str(s or ""))[:limit]


def validate_extraction(obj: dict[str, Any], text: str | None = None) -> ValidatedExtraction:
    """`text` = the chunk text; enables grounded auto-adding of forgotten relation endpoints."""
    out = ValidatedExtraction()

    # ---------------------------------------------------------------- entities
    by_norm: dict[str, dict[str, str]] = {}
    for e in obj.get("entities", []) or []:
        name = _clean(e.get("name"), 10_000)
        norm = normalize_name(name) if name else ""
        if not name or not norm:
            out.dropped_entities["empty"] += 1
            continue
        if len(name) > MAX_NAME_CHARS:
            out.dropped_entities["too_long"] += 1
            continue
        if _NUMERIC.match(name) and not re.fullmatch(r"\d+[kKmM]", name):  # keep names like "3M"
            out.dropped_entities["numeric"] += 1
            continue
        if norm in GENERIC_NAMES:
            out.dropped_entities["generic"] += 1
            continue
        if not any(ch.isupper() for ch in name) and (" " in name or any(ch.isdigit() for ch in name)):
            # lowercase multi-word phrase or "1930s"/"177 cities": a concept, not a named entity
            out.dropped_entities["concept_phrase"] += 1
            continue
        if norm in by_norm:
            out.dropped_entities["duplicate"] += 1
            continue
        etype = e.get("type") if e.get("type") in ENTITY_TYPES else "OTHER"
        ent = {"name": name, "type": etype, "desc": _clean(e.get("desc"), MAX_DESC_CHARS)}
        by_norm[norm] = ent
        out.entities.append(ent)

    text_lower = text.lower() if text is not None else None
    text_norm = f" {normalize_name(text)} " if text is not None else ""
    # token sets for the "short form -> full name" repair
    tokens = {norm: set(norm.split()) for norm in by_norm}

    def resolve(name: Any) -> tuple[str | None, bool]:
        norm = normalize_name(_clean(name, 10_000))
        if not norm:
            return None, False
        if norm in by_norm:
            return by_norm[norm]["name"], False
        toks = set(norm.split())
        cands = [n for n, t in tokens.items() if toks and toks < t]  # strict subset: "altman" < "sam altman"
        if len(cands) == 1:
            return by_norm[cands[0]]["name"], True
        raw = _clean(name, MAX_NAME_CHARS + 1)
        if (text_lower is not None and len(raw) <= MAX_NAME_CHARS and looks_like_proper_name(raw)
                and (raw.lower() in text_lower or f" {norm} " in text_norm)  # "The Chiefs" ~ "Chiefs"
                and norm not in GENERIC_NAMES and not _NUMERIC.match(raw)):
            ent = {"name": raw, "type": "OTHER", "desc": ""}
            by_norm[norm] = ent
            tokens[norm] = toks
            out.entities.append(ent)
            out.auto_added_entities += 1
            return raw, True
        return None, False

    # ---------------------------------------------------------------- relations
    seen: set[tuple[str, str, str]] = set()
    for r in obj.get("relations", []) or []:
        src, rep_s = resolve(r.get("src"))
        tgt, rep_t = resolve(r.get("tgt"))
        pred = _clean(r.get("pred"), 60).lower()
        if src is None or tgt is None:
            out.dropped_relations["endpoint_not_entity"] += 1
            continue
        if not normalize_predicate(pred):
            out.dropped_relations["empty_predicate"] += 1
            continue
        if src == tgt:
            out.dropped_relations["self_loop"] += 1
            continue
        key = (src, normalize_predicate(pred), tgt)
        if key in seen:
            out.dropped_relations["duplicate"] += 1
            continue
        seen.add(key)
        out.repaired_relations += int(rep_s or rep_t)
        out.relations.append({"src": src, "pred": pred, "tgt": tgt, "desc": _clean(r.get("desc"), MAX_DESC_CHARS)})
    return out
