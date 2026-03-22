"""Tests for batch prediction endpoint."""


def test_batch_predict_returns_results(client):
    resp = client.post(
        "/api/v1/batch",
        json={
            "texts": [
                "Revenue surged 20%",
                "Shares dropped after weak guidance",
                "The board will meet on Tuesday",
            ]
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 3
    assert "summary" in data
    assert data["total_inference_time_ms"] >= 0


def test_batch_summary_counts(client):
    resp = client.post(
        "/api/v1/batch",
        json={"texts": ["Text A", "Text B"]},
    )
    data = resp.json()
    summary = data["summary"]
    total = summary["positive"] + summary["negative"] + summary["neutral"]
    assert total == 2
    assert summary["avg_confidence"] > 0


def test_batch_empty_list_rejected(client):
    resp = client.post(
        "/api/v1/batch",
        json={"texts": []},
    )
    assert resp.status_code == 422


def test_batch_too_many_texts_rejected(client):
    resp = client.post(
        "/api/v1/batch",
        json={"texts": ["text"] * 65},
    )
    assert resp.status_code == 422


def test_batch_single_text(client):
    resp = client.post(
        "/api/v1/batch",
        json={"texts": ["Just one text"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 1
