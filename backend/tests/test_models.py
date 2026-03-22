"""Tests for models endpoint."""


def test_list_models_includes_active(client):
    resp = client.get("/api/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    active = [m for m in data if m["is_active"]]
    assert len(active) == 1
    assert active[0]["name"] == "test-model"


def test_active_model(client):
    resp = client.get("/api/v1/models/active")
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_active"] is True
    assert data["name"] == "test-model"
    assert data["status"] == "loaded"


def test_switch_model_not_found(client):
    resp = client.post("/api/v1/models/switch?model_name=nonexistent")
    assert resp.status_code == 404
