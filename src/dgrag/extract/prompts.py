"""Extraction prompt + JSON schema (versioned: bump extraction.prompt_version in config when the prompt changes).

v1 -> v2 (64-chunk trial, PLAN.md Issue #7): 64% of relations were dropped because an endpoint was not an
entity. v2 asks for every relation endpoint to be listed, skips concept relations, shortens desc.

Fixes from the llama-server benchmark (PLAN.md Issues #3, #6):
* roles/titles are relations, not entities ("CEO" is not a node)
* relation endpoints must be names from the entity list (no "shares", "acquire" as targets)
* numbers / amounts / dates go into the relation description, not into entities
"""

from __future__ import annotations

ENTITY_TYPES = ["PERSON", "ORG", "LOCATION", "EVENT", "PRODUCT", "WORK", "OTHER"]

SYSTEM_PROMPT_V2 = """You extract a knowledge graph from a news article excerpt.

ENTITIES = specific, named things:
- PERSON (people), ORG (companies, teams, leagues, agencies, governments), LOCATION (countries, cities, venues),
  EVENT (named events: trials, games, tournaments, conferences), PRODUCT (named products, services, models),
  WORK (films, games, books, shows, songs), OTHER (any other named thing, e.g. a law or currency).
- Use the most complete name in the text ("Sam Bankman-Fried", not "he" or "Bankman-Fried").
- NOT entities: job titles or roles (CEO, coach, president), generic nouns (shares, company, fans, market),
  numbers, money amounts, percentages, dates.
- "desc": what the text says about the entity, at most 15 words.

RELATIONS = facts linking two named entities:
- "src" and "tgt" MUST be copied exactly from your entity names. Every person, organisation or other named
  thing you use in a relation must ALSO appear in the entities list - list it first.
- Skip facts whose other side is a topic, concept, trend or quantity (e.g. "faces social decay",
  "reduces mortality", "shares fell 2%"); if a number matters, put it in the desc of a relation between
  two named entities instead.
- If a short name is ambiguous (e.g. two people named Kelce), use the full name.
- "pred": short lowercase verb phrase ("ceo of", "acquired", "defeated", "testified against", "located in").
- "desc": the fact in at most 12 words, including any key number, amount or date.
- Only facts stated in the text. Prefer the most important facts.

Example
Text: "Apple CEO Tim Cook said on Tuesday that the company will spend $1 billion on a new campus in Austin, Texas."
Output: {"entities": [
  {"name": "Apple", "type": "ORG", "desc": "Technology company building a new campus"},
  {"name": "Tim Cook", "type": "PERSON", "desc": "CEO of Apple"},
  {"name": "Austin", "type": "LOCATION", "desc": "City in Texas, site of Apple's new campus"}],
 "relations": [
  {"src": "Tim Cook", "pred": "ceo of", "tgt": "Apple", "desc": "Tim Cook is the CEO of Apple."},
  {"src": "Apple", "pred": "building campus in", "tgt": "Austin", "desc": "Apple will spend $1 billion on a new campus in Austin."}]}"""

USER_TEMPLATE = "Article: {title}\nSource: {source}, published {date}\n\nText:\n{text}"


def build_schema(max_entities: int = 20, max_relations: int = 20) -> dict:
    return {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array", "maxItems": max_entities,
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "type": {"type": "string", "enum": ENTITY_TYPES},
                                   "desc": {"type": "string"}},
                    "required": ["name", "type", "desc"],
                },
            },
            "relations": {
                "type": "array", "maxItems": max_relations,
                "items": {
                    "type": "object",
                    "properties": {"src": {"type": "string"}, "pred": {"type": "string"},
                                   "tgt": {"type": "string"}, "desc": {"type": "string"}},
                    "required": ["src", "pred", "tgt", "desc"],
                },
            },
        },
        "required": ["entities", "relations"],
    }


def build_messages(text: str, title: str | None, source: str | None, published_at: str | None) -> list[dict]:
    user = USER_TEMPLATE.format(title=title or "", source=source or "unknown",
                                date=(published_at or "")[:10] or "unknown", text=text)
    return [{"role": "system", "content": SYSTEM_PROMPT_V2}, {"role": "user", "content": user}]
