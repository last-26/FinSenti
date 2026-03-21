"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health_check(request: Request) -> dict:
    engine = request.app.state.engine
    return {
        "status": "healthy",
        "model_loaded": engine.is_loaded,
        "model_name": engine.model_name if engine.is_loaded else None,
        "device": engine.device,
    }
