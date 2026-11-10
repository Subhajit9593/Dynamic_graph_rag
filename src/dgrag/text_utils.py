"""Small text helpers shared across modules (hashing, name normalisation)."""

from __future__ import annotations

import hashlib
import re
import unicodedata

# Legal / corporate suffixes stripped for alias matching ("Apple Inc." == "Apple").
_ORG_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
    "llc", "plc", "gmbh", "ag", "sa", "nv", "bv", "lp", "llp", "group", "holdings",
}
_WS = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s&+\-]")


def normalize_whitespace(text: str) -> str:
    return _WS.sub(" ", text).strip()


def content_hash(text: str) -> str:
    """SHA-256 of whitespace/case-normalised text -> stable dedupe key for documents."""
    norm = normalize_whitespace(unicodedata.normalize("NFKC", text)).lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def normalize_name(name: str) -> str:
    """Canonical key for entity alias lookup.

    "Apple Inc."  -> "apple"
    "The  OpenAI" -> "openai"
    "U.S."        -> "us"
    """
    s = unicodedata.normalize("NFKD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))  # strip accents
    s = s.lower().replace("’", "'")
    s = re.sub(r"'s\b", "", s)                                   # possessive: "Apple's" -> "apple"
    s = s.replace(".", "")                                       # U.S. -> us
    s = _NON_WORD.sub(" ", s)
    tokens = normalize_whitespace(s).split(" ")
    if tokens and tokens[0] == "the" and len(tokens) > 1:
        tokens = tokens[1:]
    while len(tokens) > 1 and tokens[-1] in _ORG_SUFFIXES:
        tokens = tokens[:-1]
    return " ".join(tokens)


def normalize_predicate(predicate: str) -> str:
    """'Is CEO of' / 'is_ceo_of' / 'IS-CEO-OF' -> 'is_ceo_of'."""
    s = re.sub(r"[^a-z0-9]+", "_", predicate.lower())
    return s.strip("_")
