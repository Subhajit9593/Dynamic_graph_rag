from dgrag.ingest.chunker import chunk_text, sentence_spans, whitespace_counter


def _sentences(n, words=10):
    return " ".join(f"Sentence {i} " + " ".join(["word"] * (words - 3)) + " end." for i in range(n))


def test_sentence_spans_basic():
    t = "Mr. Smith went to Washington. He met Dr. Jones! Did it work? Yes."
    sents = [t[s:e] for s, e in sentence_spans(t)]
    assert sents == ["Mr. Smith went to Washington.", "He met Dr. Jones!", "Did it work?", "Yes."]


def test_chunks_respect_limit_and_are_exact_substrings():
    text = _sentences(100)  # 1000 words
    chunks = chunk_text(text, whitespace_counter, chunk_tokens=100, overlap_tokens=20, min_chunk_tokens=10)
    assert len(chunks) > 5
    for c in chunks:
        assert c.n_tokens <= 100
        assert text[c.char_start : c.char_end] == c.text
    assert [c.idx for c in chunks] == list(range(len(chunks)))


def test_overlap_and_full_coverage():
    text = _sentences(40)
    chunks = chunk_text(text, whitespace_counter, chunk_tokens=100, overlap_tokens=20, min_chunk_tokens=10)
    for a, b in zip(chunks, chunks[1:]):
        assert b.char_start < a.char_end  # consecutive chunks overlap
    covered = set()
    for c in chunks:
        covered.update(range(c.char_start, c.char_end))
    non_space = {i for i, ch in enumerate(text) if not ch.isspace()}
    assert non_space <= covered  # no text lost


def test_long_sentence_is_split():
    text = " ".join(["word"] * 500) + "."
    chunks = chunk_text(text, whitespace_counter, chunk_tokens=100, overlap_tokens=10, min_chunk_tokens=5)
    assert len(chunks) >= 5 and all(c.n_tokens <= 100 for c in chunks)


def test_tiny_tail_is_merged():
    text = _sentences(10) + " Tail."  # 100 words + 1-word sentence
    chunks = chunk_text(text, whitespace_counter, chunk_tokens=100, overlap_tokens=0, min_chunk_tokens=5)
    assert len(chunks) == 1 and chunks[0].text.endswith("Tail.")


def test_short_text_single_chunk_and_empty():
    assert len(chunk_text("Just one sentence.", whitespace_counter)) == 1
    assert chunk_text("   ", whitespace_counter) == []
