from dgrag.extract.prompts import build_messages, build_schema
from dgrag.extract.validate import validate_extraction


def test_entities_filtered_and_deduped():
    v = validate_extraction({"entities": [
        {"name": "OpenAI", "type": "ORG", "desc": "AI lab"},
        {"name": "OpenAI Inc.", "type": "ORG", "desc": "dup"},
        {"name": "shares", "type": "OTHER", "desc": ""},
        {"name": "$565 million", "type": "OTHER", "desc": ""},
        {"name": "CEO", "type": "PERSON", "desc": ""},
        {"name": "", "type": "ORG", "desc": ""},
        {"name": "Weird", "type": "ALIEN", "desc": ""},
        {"name": "3M", "type": "ORG", "desc": "company, not a number"},
    ], "relations": []})
    assert [e["name"] for e in v.entities] == ["OpenAI", "Weird", "3M"]
    assert v.entities[1]["type"] == "OTHER"  # unknown type coerced
    assert v.dropped_entities == {"duplicate": 1, "generic": 2, "numeric": 1, "empty": 1}


def test_relations_endpoints_must_be_entities_with_repair():
    v = validate_extraction({
        "entities": [{"name": "Sam Altman", "type": "PERSON", "desc": ""},
                     {"name": "OpenAI", "type": "ORG", "desc": ""},
                     {"name": "Star Entertainment Group", "type": "ORG", "desc": ""}],
        "relations": [
            {"src": "Altman", "pred": "CEO of", "tgt": "OpenAI", "desc": "repaired"},       # short form
            {"src": "Sam Altman", "pred": "ceo of", "tgt": "OpenAI", "desc": "dup"},        # duplicate
            {"src": "Star Entertainment Group", "pred": "closed at", "tgt": "shares", "desc": "63c"},
            {"src": "OpenAI", "pred": "is", "tgt": "OpenAI", "desc": "loop"},
            {"src": "OpenAI", "pred": "  ", "tgt": "Sam Altman", "desc": ""},
        ]})
    assert v.relations == [{"src": "Sam Altman", "pred": "ceo of", "tgt": "OpenAI", "desc": "repaired"}]
    assert v.repaired_relations == 1
    assert v.dropped_relations == {"duplicate": 1, "endpoint_not_entity": 1, "self_loop": 1, "empty_predicate": 1}


def test_ambiguous_short_form_not_repaired():
    v = validate_extraction({
        "entities": [{"name": "John Smith", "type": "PERSON", "desc": ""},
                     {"name": "Jane Smith", "type": "PERSON", "desc": ""},
                     {"name": "Acme", "type": "ORG", "desc": ""}],
        "relations": [{"src": "Smith", "pred": "works at", "tgt": "Acme", "desc": ""}]})
    assert v.relations == [] and v.dropped_relations["endpoint_not_entity"] == 1


def test_schema_and_messages():
    s = build_schema(5, 7)
    assert s["properties"]["entities"]["maxItems"] == 5 and s["properties"]["relations"]["maxItems"] == 7
    m = build_messages("body", "Title", "Src", "2023-10-01T00:00:00+00:00")
    assert m[0]["role"] == "system" and "2023-10-01" in m[1]["content"] and "Title" in m[1]["content"]


def test_forgotten_endpoint_auto_added_only_if_grounded():
    text = "Travis Kelce grew a moustache for Andy Reid. America faces social decay."
    obj = {"entities": [{"name": "Travis Kelce", "type": "PERSON", "desc": ""},
                        {"name": "America", "type": "LOCATION", "desc": ""}],
           "relations": [
               {"src": "Travis Kelce", "pred": "grew moustache for", "tgt": "Andy Reid", "desc": ""},  # recovered
               {"src": "America", "pred": "faces", "tgt": "social decay", "desc": ""},             # concept
               {"src": "Travis Kelce", "pred": "met", "tgt": "Joe Biden", "desc": ""},              # not in text
               {"src": "Travis Kelce", "pred": "likes", "tgt": "Kim Kardashian and Kanye West", "desc": ""}]}
    v = validate_extraction(obj, text=text)
    assert [r["tgt"] for r in v.relations] == ["Andy Reid"]
    assert v.auto_added_entities == 1 and {"name": "Andy Reid", "type": "OTHER", "desc": ""} in v.entities
    assert v.dropped_relations["endpoint_not_entity"] == 3
    assert validate_extraction(obj).relations == []  # without text: no auto-add


def test_concept_phrases_dropped_but_lowercase_brand_kept():
    v = validate_extraction({"entities": [
        {"name": "heat inequality", "type": "OTHER", "desc": ""},
        {"name": "177 cities", "type": "LOCATION", "desc": ""},
        {"name": "1930s", "type": "OTHER", "desc": ""},
        {"name": "bitcoin", "type": "PRODUCT", "desc": ""},
        {"name": "iPhone 15", "type": "PRODUCT", "desc": ""}], "relations": []})
    assert [e["name"] for e in v.entities] == ["bitcoin", "iPhone 15"]
    assert v.dropped_entities["concept_phrase"] == 3


def test_dates_and_amount_phrases_dropped():
    v = validate_extraction({"entities": [
        {"name": "Jan. 10, 2024", "type": "EVENT", "desc": ""},
        {"name": "October 5", "type": "EVENT", "desc": ""},
        {"name": "$100 in Bonus Bets", "type": "PRODUCT", "desc": ""},
        {"name": "May Day", "type": "EVENT", "desc": ""},
        {"name": "Super Bowl LVIII", "type": "EVENT", "desc": ""}], "relations": []})
    assert [e["name"] for e in v.entities] == ["May Day", "Super Bowl LVIII"]
    assert v.dropped_entities["date_or_amount"] == 3


def test_grounding_drops_hallucinated_and_prompt_copied_names():
    text = "Sam Reich founded Dropout after leaving CollegeHumor. The Chiefs won."
    v = validate_extraction({"entities": [
        {"name": "Sam Reich", "type": "PERSON", "desc": ""},
        {"name": "Sam Bankman-Fried", "type": "PERSON", "desc": "Founder of Dropout"},   # hallucinated
        {"name": "TechCrunch", "type": "ORG", "desc": "publisher"},                       # copied from "Source:"
        {"name": "Kansas City Chiefs", "type": "ORG", "desc": ""},                        # head word grounded
        {"name": "Dropout", "type": "ORG", "desc": ""}], "relations": []}, text=text, title="Dropout drops CH")
    assert [e["name"] for e in v.entities] == ["Sam Reich", "Kansas City Chiefs", "Dropout"]
    assert v.dropped_entities["ungrounded"] == 2
