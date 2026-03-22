"""Tests for predict endpoint."""


def test_predict_returns_sentiment(client):
    resp = client.post(
        "/api/v1/predict",
        json={"text": "Revenue surged 20% in Q4"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["sentiment"] == "positive"
    assert data["confidence"] > 0
    assert "probabilities" in data
    assert data["probabilities"]["positive"] > 0
    assert data["market_signal"] == "bullish"
    assert data["model_used"] == "test-model"
    assert data["inference_time_ms"] > 0
    assert "Q4" in data["entities"]


def test_predict_extracts_ticker_entities(client):
    resp = client.post(
        "/api/v1/predict",
        json={"text": "$AAPL and $TSLA reported strong earnings"},
    )
    assert resp.status_code == 200
    entities = resp.json()["entities"]
    assert "AAPL" in entities
    assert "TSLA" in entities


def test_predict_empty_text_rejected(client):
    resp = client.post(
        "/api/v1/predict",
        json={"text": ""},
    )
    assert resp.status_code == 422


def test_predict_missing_text_rejected(client):
    resp = client.post(
        "/api/v1/predict",
        json={},
    )
    assert resp.status_code == 422


def test_predict_text_too_long_rejected(client):
    resp = client.post(
        "/api/v1/predict",
        json={"text": "x" * 513},
    )
    assert resp.status_code == 422
