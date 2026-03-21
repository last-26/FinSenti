"""Schemas for experiment and model endpoints."""

from __future__ import annotations

from pydantic import BaseModel


class ModelInfo(BaseModel):
    name: str
    base_model: str
    is_active: bool
    status: str


class ExperimentSummary(BaseModel):
    experiment_id: str
    experiment_name: str
    run_count: int


class RunSummary(BaseModel):
    run_id: str
    run_name: str
    status: str
    base_model: str | None
    f1_macro: float | None
    accuracy: float | None
    latency_p50_ms: float | None


class HistoryEntry(BaseModel):
    id: int
    text: str
    sentiment: str
    confidence: float
    model_used: str
    created_at: str


class HistoryResponse(BaseModel):
    entries: list[HistoryEntry]
    total: int
    page: int
    page_size: int
