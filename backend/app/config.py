"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "FinSenti API"
    app_version: str = "0.1.0"
    debug: bool = False

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"

    # Database
    database_url: str = "sqlite+aiosqlite:///./predictions.db"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlruns_dir: str = "../training/mlruns"

    # Model
    default_model: str = "finbert-lora"
    model_dir: str = "./models"
    hf_home: str = "./cache/huggingface"

    # Inference
    max_batch_size: int = 64
    max_text_length: int = 512

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
