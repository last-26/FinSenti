"""Text preprocessing for financial sentiment analysis."""

from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """Clean and normalize financial text."""
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_entities(text: str) -> list[str]:
    """Extract financial entities from text using regex heuristics."""
    entities = []

    # Stock tickers: $AAPL, $TSLA
    tickers = re.findall(r"\$([A-Z]{1,5})\b", text)
    entities.extend(tickers)

    # Uppercase company-like words (2+ consecutive capitalized words)
    company_patterns = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text)
    entities.extend(company_patterns)

    # Known entities in common financial text
    known = [
        "Fed", "OPEC", "SEC", "ECB", "IMF", "GDP", "IPO", "CEO",
        "S&P", "Nasdaq", "NYSE", "Dow",
    ]
    for entity in known:
        if entity in text:
            entities.append(entity)

    # Quarter references: Q1, Q2, Q3, Q4
    quarters = re.findall(r"\b(Q[1-4])\b", text)
    entities.extend(quarters)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for e in entities:
        if e not in seen:
            seen.add(e)
            unique.append(e)

    return unique
