"""
FinSenti FastAPI application.

Loads the sentiment model on startup and serves prediction endpoints.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.inference.engine import SentimentEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    os.environ.setdefault("HF_HOME", settings.hf_home)

    # Initialize database
    await init_db()

    # Load model
    engine = SentimentEngine()

    # Try loading LoRA adapter first, fallback to pretrained
    model_dir = Path(settings.model_dir)
    adapter_path = model_dir / settings.default_model / "adapter"
    if adapter_path.exists():
        engine.load_from_adapter(str(adapter_path), model_name=settings.default_model)
    else:
        # Fallback: load base FinBERT for development
        print(f"No adapter found at {adapter_path}, loading base FinBERT...")
        engine.load_pretrained("ProsusAI/finbert")

    app.state.engine = engine

    yield

    # Cleanup
    del app.state.engine


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
from app.api.routes import batch, experiments, health, history, models, predict

app.include_router(health.router, prefix=settings.api_prefix, tags=["health"])
app.include_router(predict.router, prefix=settings.api_prefix, tags=["predict"])
app.include_router(batch.router, prefix=settings.api_prefix, tags=["batch"])
app.include_router(models.router, prefix=settings.api_prefix, tags=["models"])
app.include_router(experiments.router, prefix=settings.api_prefix, tags=["experiments"])
app.include_router(history.router, prefix=settings.api_prefix, tags=["history"])


@app.get("/")
async def root():
    return {"name": settings.app_name, "version": settings.app_version}
