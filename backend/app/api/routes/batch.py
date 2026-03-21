"""Batch prediction endpoint."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import PredictionLog, get_session
from app.schemas.predict import BatchRequest, BatchResponse, BatchSummary, SentimentResult

router = APIRouter()


@router.post("/batch", response_model=BatchResponse)
async def batch_predict(
    request: Request,
    body: BatchRequest,
    session: AsyncSession = Depends(get_session),
) -> BatchResponse:
    engine = request.app.state.engine
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    raw_results = engine.predict_batch(body.texts)
    total_ms = (time.perf_counter() - start) * 1000

    results = [SentimentResult(**r) for r in raw_results]

    # Log all predictions
    for r in raw_results:
        log = PredictionLog(
            text=r["text"],
            sentiment=r["sentiment"],
            confidence=r["confidence"],
            prob_positive=r["probabilities"]["positive"],
            prob_neutral=r["probabilities"]["neutral"],
            prob_negative=r["probabilities"]["negative"],
            model_used=r["model_used"],
            inference_time_ms=r["inference_time_ms"],
        )
        session.add(log)
    await session.commit()

    # Compute summary
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for r in results:
        sentiment_counts[r.sentiment] += 1

    avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0

    return BatchResponse(
        results=results,
        summary=BatchSummary(
            **sentiment_counts,
            avg_confidence=round(avg_confidence, 4),
        ),
        total_inference_time_ms=round(total_ms, 2),
    )
