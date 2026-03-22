"""Tests for root endpoint."""


def test_root_returns_app_info(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "FinSenti API"
    assert "version" in data
