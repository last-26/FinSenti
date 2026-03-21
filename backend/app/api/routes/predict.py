"""Single text prediction endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import PredictionLog, get_session
from app.schemas.predict import PredictRequest, SentimentResult

router = APIRouter()


@router.post("/predict", response_model=SentimentResult)
async def predict(
    request: Request,
    body: PredictRequest,
    session: AsyncSession = Depends(get_session),
) -> SentimentResult:
    engine = request.app.state.engine
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = engine.predict(body.text)

    # Log prediction
    log = PredictionLog(
        text=body.text,
        sentiment=result["sentiment"],
        confidence=result["confidence"],
        prob_positive=result["probabilities"]["positive"],
        prob_neutral=result["probabilities"]["neutral"],
        prob_negative=result["probabilities"]["negative"],
        model_used=result["model_used"],
        inference_time_ms=result["inference_time_ms"],
    )
    session.add(log)
    await session.commit()

    return SentimentResult(**result)
