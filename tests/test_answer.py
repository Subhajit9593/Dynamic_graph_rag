from dgrag.generate.answer import answer_correct, build_answer_messages, is_insufficient
from dgrag.retrieve.retrievers import Fact, RetrievedChunk, Retrieval


def test_answer_matching():
    assert answer_correct("Sam Bankman-Fried", "Sam Bankman-Fried", "inference_query")
    assert answer_correct("It was Sam Bankman-Fried.", "Sam Bankman-Fried", "inference_query")
    assert not answer_correct("Bankman", "Sam Bankman-Fried", "inference_query")
    assert answer_correct("Yes", "Yes", "comparison_query")
    assert answer_correct("yes.", "Yes", "comparison_query")
    assert not answer_correct("No, not yes", "Yes", "comparison_query")
    assert answer_correct("Insufficient information.", "Insufficient information.", "null_query")
    assert not answer_correct("Apple", "Insufficient information.", "null_query")
    assert is_insufficient("There is not enough information")


def test_prompt_contains_facts_sources_and_dates():
    r = Retrieval([RetrievedChunk(1, 1, 0.9, "chunk text", "Title", "TechCrunch", "2023-10-01T00:00:00", "u")],
                  [Fact(5, "A founded B (2023-10-01)", 1.0, [1])])
    m = build_answer_messages("Who founded B?", r)
    u = m[1]["content"]
    assert "A founded B" in u and "[1] TechCrunch, 2023-10-01" in u and u.endswith("Question: Who founded B?")
    assert build_answer_messages("q", None)[1]["content"] == "q"
