"""Sentence-aware chunking with token-based sizes (PLAN.md §6 step 2).

How it works, simply:
1. Find sentence boundaries (regex; no NLTK download needed).
2. Count tokens per sentence with the *embedder's* tokenizer, so a chunk never exceeds the
   embedder's 512-token limit (anything longer would be silently truncated when embedding).
3. Greedily pack whole sentences until the next one would exceed `chunk_tokens`.
4. Start the next chunk with the last few sentences of the previous one (`overlap_tokens`), so a fact
   that spans a boundary is fully contained in at least one chunk.
5. A sentence longer than `chunk_tokens` on its own is split at word boundaries.

Chunks are stored as character spans (char_start, char_end) into the original body, so the chunk text is
always an exact substring of the article - useful for citations.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

# A sentence ends at . ! ? (optionally followed by closing quotes/brackets) and is followed by whitespace
# and an upper-case letter, digit, or opening quote. Also break on blank lines / newlines.
_SENT_END = re.compile(r"""(?<=[.!?])["'”’)\]]*\s+(?=["'“‘(\[]?[A-Z0-9])|\n+""")
# Common abbreviations that end with '.' but do not end a sentence.
_ABBREV = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|Inc|Ltd|Co|Corp|No|U\.S|U\.K|Jan|Feb|Mar|Apr|Jun|"
                     r"Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.$")

TokenCounter = Callable[[Sequence[str]], list[int]]


@dataclass(frozen=True)
class Chunk:
    idx: int
    text: str
    char_start: int
    char_end: int
    n_tokens: int


def whitespace_counter(texts: Sequence[str]) -> list[int]:
    """Cheap fallback counter (used in unit tests)."""
    return [len(t.split()) for t in texts]


def hf_token_counter(tokenizer_name: str) -> TokenCounter:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    def count(texts: Sequence[str]) -> list[int]:
        if not texts:
            return []
        enc = tok(list(texts), add_special_tokens=False, return_attention_mask=False,
                  return_token_type_ids=False, verbose=False)
        return [len(ids) for ids in enc["input_ids"]]

    return count


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) character spans of sentences, whitespace-trimmed."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENT_END.finditer(text):
        end = m.start()
        if _ABBREV.search(text[max(start, end - 6):end]) and "\n" not in m.group():
            continue  # "Mr. Smith" - not a sentence end
        spans.append((start, end))
        start = m.end()
    spans.append((start, len(text)))
    out = []
    for s, e in spans:  # trim whitespace, drop empties
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append((s, e))
    return out


def _split_long(text: str, span: tuple[int, int], n_tok: int, limit: int, count: TokenCounter
                ) -> list[tuple[int, int, int]]:
    """Split one over-long sentence into word windows of <= limit tokens."""
    s, e = span
    words = [(s + m.start(), s + m.end()) for m in re.finditer(r"\S+", text[s:e])]
    approx = max(1, int(len(words) * limit / max(n_tok, 1) * 0.9))  # words per piece, 10% safety margin
    pieces = []
    for i in range(0, len(words), approx):
        ws = words[i : i + approx]
        pieces.append((ws[0][0], ws[-1][1]))
    counts = count([text[a:b] for a, b in pieces])
    return [(a, b, c) for (a, b), c in zip(pieces, counts)]


def chunk_text(
    text: str,
    count: TokenCounter,
    chunk_tokens: int = 400,
    overlap_tokens: int = 50,
    min_chunk_tokens: int = 40,
) -> list[Chunk]:
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens")
    spans = sentence_spans(text)
    if not spans:
        return []
    counts = count([text[s:e] for s, e in spans])

    # units = (start, end, tokens); over-long sentences are pre-split
    units: list[tuple[int, int, int]] = []
    for (s, e), n in zip(spans, counts):
        units.extend(_split_long(text, (s, e), n, chunk_tokens, count) if n > chunk_tokens else [(s, e, n)])

    groups: list[list[tuple[int, int, int]]] = []
    cur: list[tuple[int, int, int]] = []
    cur_tok = 0
    for u in units:
        if cur and cur_tok + u[2] > chunk_tokens:
            groups.append(cur)
            # overlap: carry trailing units of the finished chunk (but never the whole chunk)
            carry, carry_tok = [], 0
            for prev in reversed(cur[1:]):
                if carry_tok + prev[2] > overlap_tokens:
                    break
                carry.insert(0, prev)
                carry_tok += prev[2]
            cur, cur_tok = carry, carry_tok
        cur.append(u)
        cur_tok += u[2]
    if cur:
        new_tok = sum(u[2] for u in cur if not groups or u not in groups[-1])
        if groups and new_tok < min_chunk_tokens:
            # tiny tail: merge its *new* units into the previous chunk instead of emitting a near-duplicate
            groups[-1].extend(u for u in cur if u not in groups[-1])
        else:
            groups.append(cur)

    chunks = []
    for i, g in enumerate(groups):
        a, b = g[0][0], g[-1][1]
        chunks.append(Chunk(idx=i, text=text[a:b], char_start=a, char_end=b, n_tokens=sum(u[2] for u in g)))
    return chunks
