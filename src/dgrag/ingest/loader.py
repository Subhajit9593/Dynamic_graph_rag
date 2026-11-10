"""MultiHop-RAG dataset: download, parse into dataclasses, order articles for streaming (PLAN.md §4)."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Article:
    title: str
    body: str
    source: str | None
    url: str | None
    category: str | None
    published_at: str | None  # ISO-8601 string, kept as text (SQLite stores it as TEXT)
    author: str | None = None

    @property
    def published_dt(self) -> datetime | None:
        return datetime.fromisoformat(self.published_at) if self.published_at else None


@dataclass(frozen=True)
class Evidence:
    title: str
    url: str | None
    source: str | None
    published_at: str | None
    fact: str


@dataclass(frozen=True)
class Question:
    qid: int
    query: str
    answer: str
    question_type: str  # inference_query | comparison_query | temporal_query | null_query
    evidence: tuple[Evidence, ...] = field(default_factory=tuple)

    @property
    def evidence_urls(self) -> set[str]:
        return {e.url for e in self.evidence if e.url}

    @property
    def latest_evidence_date(self) -> str | None:
        dates = [e.published_at for e in self.evidence if e.published_at]
        return max(dates) if dates else None


def download_multihoprag(raw_dir: str | Path, repo_id: str = "yixuantt/MultiHopRAG") -> Path:
    """Fetch corpus.json and MultiHopRAG.json from the Hugging Face Hub (no-op if already present)."""
    from huggingface_hub import hf_hub_download

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("corpus.json", "MultiHopRAG.json"):
        if not (raw_dir / fname).exists():
            log.info("Downloading %s/%s", repo_id, fname)
            hf_hub_download(repo_id=repo_id, filename=fname, repo_type="dataset", local_dir=raw_dir)
    return raw_dir


def load_corpus(path: str | Path, sort_by_date: bool = True) -> list[Article]:
    with open(path) as f:
        raw = json.load(f)
    articles = [
        Article(
            title=(r.get("title") or "").strip(),
            body=r["body"],
            source=r.get("source"),
            url=r.get("url"),
            category=r.get("category"),
            published_at=r.get("published_at"),
            author=r.get("author"),
        )
        for r in raw
        if r.get("body")
    ]
    if sort_by_date:
        # Chronological order = the streaming order for the dynamic-ingestion experiment (PLAN.md §9.3).
        articles.sort(key=lambda a: (a.published_at or "", a.url or ""))
    log.info("Loaded %d articles from %s", len(articles), path)
    return articles


def load_questions(path: str | Path) -> list[Question]:
    with open(path) as f:
        raw = json.load(f)
    questions = []
    for i, r in enumerate(raw):
        ev = tuple(
            Evidence(
                title=e.get("title", ""),
                url=e.get("url"),
                source=e.get("source"),
                published_at=e.get("published_at"),
                fact=e.get("fact", ""),
            )
            for e in r.get("evidence_list", [])
        )
        questions.append(Question(i, r["query"], r["answer"], r["question_type"], ev))
    log.info("Loaded %d questions from %s", len(questions), path)
    return questions


def batched(items: list, n_batches: int) -> Iterator[list]:
    """Split items into n roughly equal consecutive batches (keeps chronological order)."""
    if n_batches <= 0:
        raise ValueError("n_batches must be positive")
    size, rem = divmod(len(items), n_batches)
    start = 0
    for b in range(n_batches):
        end = start + size + (1 if b < rem else 0)
        if end > start:
            yield items[start:end]
        start = end
