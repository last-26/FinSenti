"""
Compare MLflow experiment runs side-by-side.

Fetches all runs from specified experiments and generates a comparison
table with key metrics (F1, accuracy, latency, model size).

Usage:
    python compare_models.py
    python compare_models.py --experiments finsenti-finbert finsenti-distilbert
"""

from __future__ import annotations

import argparse

import mlflow
import pandas as pd

DEFAULT_EXPERIMENTS = ["finsenti-finbert", "finsenti-distilbert"]

COMPARISON_METRICS = [
    "eval_f1_macro",
    "eval_f1_micro",
    "eval_accuracy",
    "latency_mean_ms",
    "latency_p50_ms",
    "model_size_mb",
    "trainable_params",
    "train_loss",
    "edge_cases_passed",
]


def get_best_runs(
    experiment_names: list[str],
    tracking_uri: str,
    metric: str = "eval_f1_macro",
) -> pd.DataFrame:
    """Fetch the best run from each experiment based on the given metric."""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    rows = []
    for exp_name in experiment_names:
        experiment = client.get_experiment_by_name(exp_name)
        if experiment is None:
            print(f"  Experiment '{exp_name}' not found, skipping.")
            continue

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
        )

        if not runs:
            print(f"  No runs found in '{exp_name}', skipping.")
            continue

        best_run = runs[0]
        row = {
            "experiment": exp_name,
            "run_id": best_run.info.run_id,
            "run_name": best_run.data.tags.get("mlflow.runName", "N/A"),
            "base_model": best_run.data.params.get("base_model", "N/A"),
        }

        for m in COMPARISON_METRICS:
            row[m] = best_run.data.metrics.get(m)

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model experiments")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=DEFAULT_EXPERIMENTS,
        help="Experiment names to compare",
    )
    parser.add_argument(
        "--tracking-uri",
        default="http://localhost:5000",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--metric",
        default="eval_f1_macro",
        help="Metric to rank runs by",
    )
    args = parser.parse_args()

    print(f"Comparing experiments: {args.experiments}")
    print(f"Ranking by: {args.metric}\n")

    df = get_best_runs(args.experiments, args.tracking_uri, args.metric)

    if df.empty:
        print("No runs found. Make sure MLflow server is running and models are trained.")
        return

    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    for _, row in df.iterrows():
        print(f"\n{row['experiment']} ({row['base_model']})")
        print(f"  Run: {row['run_id'][:8]}...")
        print(f"  F1 Macro:    {row.get('eval_f1_macro', 'N/A')}")
        print(f"  F1 Micro:    {row.get('eval_f1_micro', 'N/A')}")
        print(f"  Accuracy:    {row.get('eval_accuracy', 'N/A')}")
        print(f"  Latency p50: {row.get('latency_p50_ms', 'N/A')} ms")
        print(f"  Model Size:  {row.get('model_size_mb', 'N/A')} MB")
        print(f"  Edge Cases:  {row.get('edge_cases_passed', 'N/A')}")

    # Determine winner
    if len(df) >= 2 and args.metric in df.columns:
        valid = df.dropna(subset=[args.metric])
        if not valid.empty:
            best_idx = valid[args.metric].idxmax()
            winner = valid.loc[best_idx]
            print(f"\n{'=' * 80}")
            print(f"WINNER: {winner['experiment']} ({winner['base_model']})")
            print(f"  {args.metric}: {winner[args.metric]}")
            print(f"  Run ID: {winner['run_id']}")


if __name__ == "__main__":
    main()
