"""Dataset loader. Tests on the real files are skipped if the dataset has not been downloaded."""

import json

import pytest

from dgrag.config import load_config
from dgrag.ingest.loader import batched, load_corpus, load_questions

CFG = load_config()
RAW = CFG.resolve(CFG.paths.raw_dir)
HAVE_DATA = (RAW / CFG.dataset.corpus_file).exists()


def test_load_corpus_sorted(tmp_path):
    data = [
        {"title": "b", "body": "x", "published_at": "2023-10-02T00:00:00+00:00", "url": "u2"},
        {"title": "a", "body": "y", "published_at": "2023-09-01T00:00:00+00:00", "url": "u1"},
        {"title": "empty", "body": "", "published_at": "2023-01-01T00:00:00+00:00", "url": "u0"},
    ]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(data))
    arts = load_corpus(p)
    assert [a.title for a in arts] == ["a", "b"]  # chronological, empty body dropped
    assert arts[0].published_dt.year == 2023


def test_batched_preserves_order_and_items():
    items = list(range(23))
    batches = list(batched(items, 5))
    assert len(batches) == 5 and sum(batches, []) == items
    assert max(map(len, batches)) - min(map(len, batches)) <= 1


def test_config_paths_resolve_to_project():
    assert CFG.resolve(CFG.paths.db_path).is_absolute()
    assert (CFG.root / "config.yaml").exists()


@pytest.mark.skipif(not HAVE_DATA, reason="dataset not downloaded (run `dgrag download`)")
def test_real_dataset_shapes():
    arts = load_corpus(RAW / CFG.dataset.corpus_file)
    qs = load_questions(RAW / CFG.dataset.questions_file)
    assert len(arts) == 609 and len(qs) == 2556
    assert arts == sorted(arts, key=lambda a: (a.published_at or "", a.url or ""))
    urls = {a.url for a in arts}
    answerable = [q for q in qs if q.question_type != "null_query"]
    # every evidence url of an answerable question must exist in the corpus (needed for retrieval metrics)
    missing = sum(1 for q in answerable for u in q.evidence_urls if u not in urls)
    assert missing == 0
