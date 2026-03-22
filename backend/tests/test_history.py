"""Tests for history endpoint."""


def test_history_empty_initially(client):
    resp = client.get("/api/v1/history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 0
    assert "entries" in data
    assert data["page"] == 1
    assert data["page_size"] == 20


def test_history_after_prediction(client):
    # Make a prediction first
    client.post("/api/v1/predict", json={"text": "Revenue grew 15%"})

    resp = client.get("/api/v1/history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1


def test_history_pagination_params(client):
    resp = client.get("/api/v1/history?page=1&page_size=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["page"] == 1
    assert data["page_size"] == 5


def test_history_invalid_page(client):
    resp = client.get("/api/v1/history?page=0")
    assert resp.status_code == 422


def test_history_page_size_too_large(client):
    resp = client.get("/api/v1/history?page_size=101")
    assert resp.status_code == 422
