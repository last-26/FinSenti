"""
Register the best model in MLflow Model Registry.

Promotes a run's model to a registered model with a stage
(Staging or Production).

Usage:
    python register_model.py --model-name finsenti-finbert --stage Production
    python register_model.py --run-id <id> --model-name finsenti-best --stage Staging
"""

from __future__ import annotations

import argparse

import mlflow
from mlflow.tracking import MlflowClient

DEFAULT_EXPERIMENTS = ["finsenti-finbert", "finsenti-distilbert"]


def find_best_run(
    client: MlflowClient,
    experiment_names: list[str],
    metric: str = "eval_f1_macro",
) -> str | None:
    """Find the best run across experiments by metric."""
    best_run_id = None
    best_value = -1.0

    for exp_name in experiment_names:
        experiment = client.get_experiment_by_name(exp_name)
        if experiment is None:
            continue

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
        )

        if runs and runs[0].data.metrics.get(metric, -1) > best_value:
            best_value = runs[0].data.metrics[metric]
            best_run_id = runs[0].info.run_id

    return best_run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Register model in MLflow")
    parser.add_argument("--model-name", required=True, help="Name for the registered model")
    parser.add_argument("--run-id", help="Specific run ID (auto-detects best if omitted)")
    parser.add_argument("--stage", default="Production", choices=["Staging", "Production"])
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--metric", default="eval_f1_macro", help="Metric for auto-selection")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    # Determine run ID
    run_id = args.run_id
    if not run_id:
        print(f"Auto-selecting best run by {args.metric}...")
        run_id = find_best_run(client, DEFAULT_EXPERIMENTS, args.metric)
        if not run_id:
            print("No runs found. Train models first.")
            return

    run = client.get_run(run_id)
    print(f"Run ID: {run_id}")
    print(f"Base model: {run.data.params.get('base_model', 'unknown')}")
    print(f"F1 Macro: {run.data.metrics.get('eval_f1_macro', 'N/A')}")

    # Register model
    artifact_uri = f"runs:/{run_id}/adapter"
    result = mlflow.register_model(artifact_uri, args.model_name)
    print(f"\nRegistered model: {args.model_name} (version {result.version})")

    # Transition to stage
    client.transition_model_version_stage(
        name=args.model_name,
        version=result.version,
        stage=args.stage,
    )
    print(f"Transitioned to stage: {args.stage}")


if __name__ == "__main__":
    main()
