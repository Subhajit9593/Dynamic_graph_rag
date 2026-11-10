"""Answer generation (PLAN.md §8 step 5) and answer checking.

The LLM gets: the question, (optionally) a list of graph facts, and the raw evidence chunks with source and date.
It returns JSON {"reasoning": <=60 words, "answer": short}. A short reasoning step helps a 7B model on comparison /
temporal questions; MultiHop-RAG gold answers are short ("Yes", "No", a name, "Insufficient information.").
"""

from __future__ import annotations

import re
from typing import Any

from dgrag.retrieve.retrievers import Retrieval

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"}, "answer": {"type": "string"}},
    "required": ["reasoning", "answer"],
}

SYSTEM = """You answer questions about news articles using ONLY the provided evidence.
Rules:
- "reasoning": at most 60 words; say which evidence items you used.
- "answer": as short as possible - a name, a short phrase, or "Yes"/"No" for yes/no questions.
- If the evidence does not contain the answer, the answer must be exactly: Insufficient information.
- Pay attention to sources and publication dates when the question mentions them."""

SYSTEM_NO_RETRIEVAL = """Answer the question about news from late 2023.
- "reasoning": at most 60 words.
- "answer": as short as possible - a name, a short phrase, or "Yes"/"No" for yes/no questions.
- If you do not know, the answer must be exactly: Insufficient information."""


def build_answer_messages(question: str, retrieval: Retrieval | None, max_facts: int = 15) -> list[dict[str, str]]:
    if retrieval is None:
        return [{"role": "system", "content": SYSTEM_NO_RETRIEVAL}, {"role": "user", "content": question}]
    parts = []
    if retrieval.facts:
        parts.append("Knowledge-graph facts:\n" + "\n".join(f"- {f.text}" for f in retrieval.facts[:max_facts]))
    ev = []
    for i, c in enumerate(retrieval.chunks, 1):
        ev.append(f"[{i}] {c.source}, {c.published_at[:10]} — \"{c.title}\"\n{c.text}")
    parts.append("Evidence:\n" + "\n\n".join(ev) if ev else "Evidence: (none)")
    parts.append(f"Question: {question}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


# ---------------------------------------------------------------- answer checking
def _norm(s: str) -> str:
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def is_insufficient(pred: str) -> bool:
    p = _norm(pred)
    return "insufficient" in p or "not enough information" in p or p in {"unknown", "i don t know"}


def answer_correct(pred: str, gold: str, question_type: str) -> bool:
    """MultiHop-RAG style: correct if the (normalised) gold answer appears in the prediction.

    Yes/No answers must match exactly on the first word (so "No, ..." does not count for gold "yes" just because
    the text contains "yes" later). Null questions are correct when the model says it lacks information.
    """
    if question_type == "null_query":
        return is_insufficient(pred)
    g, p = _norm(gold), _norm(pred)
    if not g:
        return False
    if g in ("yes", "no"):
        return p.split()[:1] == [g]
    return f" {g} " in f" {p} "


def parse_answer(obj: Any, raw: str) -> tuple[str, str]:
    if isinstance(obj, dict):
        return str(obj.get("answer", "")).strip(), str(obj.get("reasoning", "")).strip()
    return raw.strip(), ""
