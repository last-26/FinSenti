"""MLflow experiment browsing endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.schemas.experiment import ExperimentSummary, RunSummary

router = APIRouter()


def _get_mlflow_client():
    import mlflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return mlflow.tracking.MlflowClient()


@router.get("/experiments")
async def list_experiments() -> list[ExperimentSummary]:
    """List all MLflow experiments with run counts."""
    try:
        client = _get_mlflow_client()
        experiments = client.search_experiments()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"MLflow unavailable: {e}")

    results = []
    for exp in experiments:
        if exp.name == "Default":
            continue
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        results.append(ExperimentSummary(
            experiment_id=exp.experiment_id,
            experiment_name=exp.name,
            run_count=len(runs),
        ))

    return results


@router.get("/experiments/{experiment_id}/runs")
async def list_runs(experiment_id: str) -> list[RunSummary]:
    """List runs for a specific experiment."""
    try:
        client = _get_mlflow_client()
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["metrics.eval_f1_macro DESC"],
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"MLflow unavailable: {e}")

    return [
        RunSummary(
            run_id=run.info.run_id,
            run_name=run.data.tags.get("mlflow.runName", "N/A"),
            status=run.info.status,
            base_model=run.data.params.get("base_model"),
            f1_macro=run.data.metrics.get("eval_f1_macro"),
            accuracy=run.data.metrics.get("eval_accuracy"),
            latency_p50_ms=run.data.metrics.get("latency_p50_ms"),
        )
        for run in runs
    ]
