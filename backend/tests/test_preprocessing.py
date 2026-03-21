"""Tests for text preprocessing utilities."""

from app.inference.preprocessing import clean_text, extract_entities


def test_clean_text_removes_urls():
    text = "Check https://example.com for details"
    assert "https://" not in clean_text(text)


def test_clean_text_normalizes_whitespace():
    text = "Too   many    spaces"
    assert clean_text(text) == "Too many spaces"


def test_extract_tickers():
    entities = extract_entities("$AAPL and $TSLA are up today")
    assert "AAPL" in entities
    assert "TSLA" in entities


def test_extract_quarters():
    entities = extract_entities("Strong Q4 results announced")
    assert "Q4" in entities


def test_extract_known_entities():
    entities = extract_entities("The Fed raised rates amid OPEC cuts")
    assert "Fed" in entities
    assert "OPEC" in entities


def test_extract_entities_deduplicates():
    entities = extract_entities("$AAPL $AAPL $AAPL")
    assert entities.count("AAPL") == 1
