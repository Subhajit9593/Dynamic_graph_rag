"""Name-level rules for entity resolution (PLAN.md §6.1).

Embeddings of "name (type): description" alone are not enough to decide a merge: two *different* entities with
similar descriptions ("Goodman: REIT" vs "Dexus: REIT") get high cosine similarity. So a merge also needs the
names to be compatible, or an LLM judge to confirm.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

# types that can be merged with each other ("OTHER" = unknown, e.g. auto-added endpoints)
WILDCARD_TYPES = {"OTHER"}


def types_compatible(a: str, b: str) -> bool:
    return a == b or a in WILDCARD_TYPES or b in WILDCARD_TYPES


def _initials(norm: str) -> str:
    return "".join(t[0] for t in re.split(r"[\s\-]+", norm) if t)


# Extra words that mark a *different variant* of a thing: "MacBook Pro" vs "MacBook Pro 16-inch", "Switch" vs
# "Switch Lite", "World Cup" vs "Rugby World Cup 2023". A token-subset match is not allowed if these differ.
VARIANT_TOKENS = {
    "pro", "max", "mini", "lite", "oled", "lcd", "plus", "ultra", "air", "se", "series", "gen", "generation",
    "edition", "inch", "mm", "gb", "tb", "online", "kids", "remastered", "remake", "season", "part", "vol",
    "rugby", "cricket", "icc", "fifa", "women", "womens", "men", "mens", "junior", "u21", "u23", "jr", "ii", "iii",
    "iv", "classic", "deluxe", "standard", "sportsbook", "yield", "treasury", "fund", "foundation", "dst", "d st",
}
# for these types a short-name-inside-long-name match must also have very high similarity
STRICT_SUBSET_TYPES = {"PRODUCT", "EVENT", "WORK", "OTHER"}


def compat_kind(a_norm: str, b_norm: str) -> str | None:
    """'equal' | 'subset' | 'acronym' | 'fuzzy' | None (see name_compatible)."""
    if not a_norm or not b_norm:
        return None
    if a_norm == b_norm:
        return "equal"
    ta, tb = set(a_norm.split()), set(b_norm.split())
    if ta < tb or tb < ta:
        extra = (tb - ta) if ta < tb else (ta - tb)
        shorter = ta if ta < tb else tb
        if shorter <= VARIANT_TOKENS:
            return None  # the short name is only a generic word: "sportsbook" vs "draftkings sportsbook"
        if any(any(ch.isdigit() for ch in t) or t in VARIANT_TOKENS for t in extra):
            return None  # "13-inch macbook pro" vs "16-inch macbook pro": different variants
        return "subset"
    short, long_ = sorted((a_norm, b_norm), key=len)
    if " " not in short and len(short) >= 2:
        stop = {"of", "the", "and", "for", "de"}
        initials_all = _initials(long_)
        initials_content = "".join(t[0] for t in re.split(r"[\s\-]+", long_) if t and t not in stop)
        if short in (initials_all, initials_content):
            return "acronym"
    if min(len(a_norm), len(b_norm)) >= 5 and SequenceMatcher(None, a_norm, b_norm).ratio() >= 0.88:
        # but not if they differ in a number: "2-year treasury" vs "10-year treasury", "iphone 14" vs "iphone 15"
        if re.findall(r"\d+", a_norm) == re.findall(r"\d+", b_norm):
            return "fuzzy"
    return None


def name_compatible(a_norm: str, b_norm: str) -> bool:
    """Could two normalised names refer to the same entity, judging by the names alone?

    equal                      "apple" / "apple"
    token subset               "altman" / "sam altman", "chiefs" / "kansas city chiefs"
    acronym                    "sbf" / "sam bankman-fried", "doj" / "department of justice"
    near-identical spelling    "jermey kelce" / "jeremy kelce"
    """
    return compat_kind(a_norm, b_norm) is not None


def entity_text(name: str, type_: str, desc: str | None) -> str:
    """Text that is embedded for an entity mention (name first: it matters most)."""
    return f"{name} ({type_}): {desc}" if desc else f"{name} ({type_})"


def relation_text(src: str, pred: str, tgt: str, desc: str | None) -> str:
    return f"{src} {pred} {tgt}: {desc}" if desc else f"{src} {pred} {tgt}"
