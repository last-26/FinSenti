"""Shared test fixtures."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.inference.postprocessing import format_prediction
from app.inference.preprocessing import extract_entities


def _make_mock_engine(model_name: str = "test-model") -> MagicMock:
    """Create a mock SentimentEngine that returns deterministic results."""
    engine = MagicMock()
    engine.is_loaded = True
    engine.model_name = model_name
    engine.device = "cpu"

    def mock_predict(text: str) -> dict:
        entities = extract_entities(text)
        # Return a fixed positive prediction
        probs = [0.02, 0.05, 0.93]
        return format_prediction(text, probs, model_name, 10.0, entities)

    def mock_predict_batch(texts: list[str], batch_size: int = 32) -> list[dict]:
        return [mock_predict(t) for t in texts]

    engine.predict.side_effect = mock_predict
    engine.predict_batch.side_effect = mock_predict_batch
    return engine


@pytest.fixture()
def mock_engine() -> MagicMock:
    return _make_mock_engine()


@pytest.fixture()
def client(mock_engine: MagicMock) -> TestClient:
    """TestClient with mocked engine — no real model loading."""
    from app.main import app

    # Override the lifespan by directly setting app state
    app.state.engine = mock_engine
    return TestClient(app, raise_server_exceptions=False)
