"""Model management endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from app.config import settings
from app.schemas.experiment import ModelInfo

router = APIRouter()


@router.get("/models")
async def list_models(request: Request) -> list[ModelInfo]:
    """List available models (adapters found in model_dir)."""
    engine = request.app.state.engine
    model_dir = Path(settings.model_dir)
    models = []

    if model_dir.exists():
        for adapter_dir in model_dir.iterdir():
            if adapter_dir.is_dir() and (adapter_dir / "adapter_config.json").exists():
                models.append(ModelInfo(
                    name=adapter_dir.name,
                    base_model="unknown",
                    is_active=adapter_dir.name == engine.model_name,
                    status="loaded" if adapter_dir.name == engine.model_name else "available",
                ))

    # Include currently loaded model even if not in model_dir
    if engine.is_loaded and not any(m.name == engine.model_name for m in models):
        models.append(ModelInfo(
            name=engine.model_name,
            base_model=engine.model_name,
            is_active=True,
            status="loaded",
        ))

    return models


@router.get("/models/active")
async def active_model(request: Request) -> ModelInfo:
    engine = request.app.state.engine
    if not engine.is_loaded:
        raise HTTPException(status_code=404, detail="No model currently loaded")
    return ModelInfo(
        name=engine.model_name,
        base_model=engine.model_name,
        is_active=True,
        status="loaded",
    )


@router.post("/models/switch")
async def switch_model(request: Request, model_name: str) -> ModelInfo:
    """Switch the active model by loading a different adapter."""
    engine = request.app.state.engine
    model_dir = Path(settings.model_dir)
    adapter_path = model_dir / model_name / "adapter"

    if not adapter_path.exists():
        adapter_path = model_dir / model_name
        if not adapter_path.exists():
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    engine.load_from_adapter(str(adapter_path), model_name=model_name)
    return ModelInfo(
        name=engine.model_name,
        base_model=engine.model_name,
        is_active=True,
        status="loaded",
    )
