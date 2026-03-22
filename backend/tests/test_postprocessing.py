"""Tests for postprocessing utilities."""

from app.inference.postprocessing import LABEL_NAMES, MARKET_SIGNALS, format_prediction


def test_format_prediction_positive():
    result = format_prediction(
        text="Revenue surged 20%",
        probabilities=[0.01, 0.04, 0.95],
        model_name="finbert-lora",
        inference_time_ms=42.5,
        entities=["Q4"],
    )
    assert result["sentiment"] == "positive"
    assert result["confidence"] == 0.95
    assert result["market_signal"] == "bullish"
    assert result["model_used"] == "finbert-lora"
    assert result["entities"] == ["Q4"]
    assert result["inference_time_ms"] == 42.5


def test_format_prediction_negative():
    result = format_prediction(
        text="Shares dropped 15%",
        probabilities=[0.88, 0.07, 0.05],
        model_name="test-model",
        inference_time_ms=30.0,
        entities=[],
    )
    assert result["sentiment"] == "negative"
    assert result["confidence"] == 0.88
    assert result["market_signal"] == "bearish"


def test_format_prediction_neutral():
    result = format_prediction(
        text="The board will meet Tuesday",
        probabilities=[0.1, 0.8, 0.1],
        model_name="test-model",
        inference_time_ms=25.0,
        entities=[],
    )
    assert result["sentiment"] == "neutral"
    assert result["market_signal"] == "hold"


def test_format_prediction_rounds_values():
    result = format_prediction(
        text="test",
        probabilities=[0.123456789, 0.234567891, 0.641975320],
        model_name="m",
        inference_time_ms=42.456789,
        entities=[],
    )
    assert result["confidence"] == 0.642
    assert result["inference_time_ms"] == 42.46
    assert result["probabilities"]["negative"] == 0.1235
    assert result["probabilities"]["neutral"] == 0.2346


def test_label_names_order():
    assert LABEL_NAMES == ["negative", "neutral", "positive"]


def test_market_signals_mapping():
    assert MARKET_SIGNALS["positive"] == "bullish"
    assert MARKET_SIGNALS["negative"] == "bearish"
    assert MARKET_SIGNALS["neutral"] == "hold"
