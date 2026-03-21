"""SQLite database setup for prediction logging."""

from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

engine = create_async_engine(settings.database_url, echo=settings.debug)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=False)
    sentiment = Column(String(10), nullable=False)
    confidence = Column(Float, nullable=False)
    prob_positive = Column(Float)
    prob_neutral = Column(Float)
    prob_negative = Column(Float)
    model_used = Column(String(50), nullable=False)
    inference_time_ms = Column(Float)
    created_at = Column(DateTime, server_default=func.now())


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session
