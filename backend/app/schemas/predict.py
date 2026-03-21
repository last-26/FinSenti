"""Request/response schemas for prediction endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, description="Financial text to classify")
    model: str | None = Field(None, description="Model to use (default: active model)")


class SentimentResult(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: dict[str, float]
    entities: list[str]
    market_signal: str
    model_used: str
    inference_time_ms: float


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64, description="List of texts")
    model: str | None = Field(None, description="Model to use")


class BatchSummary(BaseModel):
    positive: int
    negative: int
    neutral: int
    avg_confidence: float


class BatchResponse(BaseModel):
    results: list[SentimentResult]
    summary: BatchSummary
    total_inference_time_ms: float
