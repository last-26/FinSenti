"""Prediction history endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import PredictionLog, get_session
from app.schemas.experiment import HistoryEntry, HistoryResponse

router = APIRouter()


@router.get("/history", response_model=HistoryResponse)
async def get_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
) -> HistoryResponse:
    """Get paginated prediction history."""
    # Count total
    count_stmt = select(func.count()).select_from(PredictionLog)
    total_result = await session.execute(count_stmt)
    total = total_result.scalar() or 0

    # Fetch page
    offset = (page - 1) * page_size
    stmt = (
        select(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    result = await session.execute(stmt)
    logs = result.scalars().all()

    entries = [
        HistoryEntry(
            id=log.id,
            text=log.text,
            sentiment=log.sentiment,
            confidence=log.confidence,
            model_used=log.model_used,
            created_at=str(log.created_at),
        )
        for log in logs
    ]

    return HistoryResponse(
        entries=entries,
        total=total,
        page=page,
        page_size=page_size,
    )
