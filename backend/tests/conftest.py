"""Shared test fixtures."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

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
    """TestClient with mocked engine and in-memory DB."""
    from app.database import Base, get_session
    from app.main import app

    # Use in-memory SQLite for tests
    test_engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    test_session_factory = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )

    # Create tables
    asyncio.get_event_loop_policy().new_event_loop()
    loop = asyncio.new_event_loop()
    loop.run_until_complete(_create_tables(test_engine, Base))
    loop.close()

    # Override DB session dependency
    async def _override_get_session():
        async with test_session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = _override_get_session
    app.state.engine = mock_engine

    client = TestClient(app, raise_server_exceptions=False)
    yield client

    # Cleanup
    app.dependency_overrides.clear()


async def _create_tables(engine, base):
    async with engine.begin() as conn:
        await conn.run_sync(base.metadata.create_all)
