"""
Evaluation suite for trained models.

Generates classification report, confusion matrix, inference latency,
and edge case test results. Logs all artifacts to MLflow.

Usage:
    python evaluate.py --run-id <mlflow_run_id>
    python evaluate.py --adapter-dir ./outputs/finbert-lora/adapter
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import torch

matplotlib.use("Agg")
from datasets import load_from_disk
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABEL_NAMES = ["negative", "neutral", "positive"]

EDGE_CASES = [
    ("The company maintained its dividend", "neutral"),
    ("Revenue increased but margins declined sharply", "negative"),
    ("EPS $2.45 vs $2.30 expected, revenue $12.1B vs $11.8B consensus", "positive"),
    ("$TSLA to the moon", "positive"),
    ("The board will meet on Tuesday to discuss Q2 results", "neutral"),
    ("Shares dropped 15% after disappointing guidance", "negative"),
]


def load_model_from_adapter(adapter_dir: str, device: str):
    """Load base model + LoRA adapter and merge."""
    adapter_path = Path(adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))

    # Load adapter config to get base model name
    from peft import PeftConfig
    peft_config = PeftConfig.from_pretrained(str(adapter_path))
    base_model_name = peft_config.base_model_name_or_path

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=3,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model = model.merge_and_unload()
    model.to(device)
    model.eval()
    return model, tokenizer


def load_model_from_mlflow(run_id: str, tracking_uri: str, device: str):
    """Load adapter from MLflow artifacts."""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    # Download adapter artifacts
    local_dir = client.download_artifacts(run_id, "adapter")
    return load_model_from_adapter(local_dir, device)


def predict_batch(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    max_length: int = 128,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on a batch of texts. Returns predictions and probabilities."""
    all_preds = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true: list, y_pred: list, output_path: str) -> None:
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {output_path}")


def measure_latency(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    n_runs: int = 100,
) -> dict:
    """Measure single-sample inference latency."""
    sample_text = texts[0] if texts else "The company reported strong earnings."
    inputs = tokenizer(
        sample_text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    ).to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(**inputs)

    # Measure
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            model(**inputs)
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    return {
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_p50_ms": float(np.median(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
    }


def evaluate_edge_cases(
    model,
    tokenizer,
    device: str,
) -> list[dict]:
    """Run edge case tests and return results."""
    results = []
    for text, expected in EDGE_CASES:
        preds, probs = predict_batch(model, tokenizer, [text], device)
        predicted_label = LABEL_NAMES[preds[0]]
        confidence = float(probs[0][preds[0]])
        passed = predicted_label == expected

        results.append({
            "text": text,
            "expected": expected,
            "predicted": predicted_label,
            "confidence": confidence,
            "passed": passed,
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--run-id", help="MLflow run ID to load model from")
    parser.add_argument("--adapter-dir", help="Direct path to adapter directory")
    parser.add_argument("--data-dir", default="./data/processed", help="Path to processed dataset")
    parser.add_argument(
        "--tracking-uri", default="http://localhost:5000", help="MLflow tracking URI"
    )
    parser.add_argument(
        "--output-dir", default="./outputs/eval", help="Directory for evaluation outputs"
    )
    args = parser.parse_args()

    if not args.run_id and not args.adapter_dir:
        parser.error("Either --run-id or --adapter-dir must be provided")

    os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "cache" / "huggingface"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    if args.adapter_dir:
        print(f"Loading model from adapter: {args.adapter_dir}")
        model, tokenizer = load_model_from_adapter(args.adapter_dir, device)
    else:
        print(f"Loading model from MLflow run: {args.run_id}")
        model, tokenizer = load_model_from_mlflow(args.run_id, args.tracking_uri, device)

    # Load test data
    data_dir = Path(args.data_dir)
    dataset = load_from_disk(str(data_dir))
    test_data = dataset["test"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Classification metrics on test set
    print("\n--- Classification Report ---")
    texts = test_data["text"]
    true_labels = test_data["label"]
    pred_labels, pred_probs = predict_batch(model, tokenizer, texts, device)

    report = classification_report(true_labels, pred_labels, target_names=LABEL_NAMES)
    print(report)

    report_path = output_dir / "classification_report.txt"
    report_path.write_text(report)

    f1_macro = f1_score(true_labels, pred_labels, average="macro")
    f1_micro = f1_score(true_labels, pred_labels, average="micro")
    accuracy = accuracy_score(true_labels, pred_labels)

    # 2. Confusion matrix
    print("\n--- Confusion Matrix ---")
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(true_labels, pred_labels, str(cm_path))

    # 3. Inference latency
    print("\n--- Inference Latency ---")
    latency = measure_latency(model, tokenizer, texts, device)
    for k, v in latency.items():
        print(f"  {k}: {v:.2f}")

    # 4. Edge case tests
    print("\n--- Edge Case Tests ---")
    edge_results = evaluate_edge_cases(model, tokenizer, device)
    passed = sum(1 for r in edge_results if r["passed"])
    print(f"  Passed: {passed}/{len(edge_results)}")
    for r in edge_results:
        status = "PASS" if r["passed"] else "FAIL"
        text_preview = r["text"][:50]
        pred, conf, exp = r["predicted"], r["confidence"], r["expected"]
        print(f"  [{status}] '{text_preview}...' ->{pred} ({conf:.2f}), expected: {exp}")

    # 5. Model size
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    # Log to MLflow if run_id was provided
    if args.run_id:
        mlflow.set_tracking_uri(args.tracking_uri)
        with mlflow.start_run(run_id=args.run_id):
            mlflow.log_metrics({
                "eval_f1_macro": f1_macro,
                "eval_f1_micro": f1_micro,
                "eval_accuracy": accuracy,
                "model_params_total": total_params,
                "model_size_mb": model_size_mb,
                "edge_cases_passed": passed,
                "edge_cases_total": len(edge_results),
                **latency,
            })
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(report_path))

    print("\n--- Summary ---")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Micro: {f1_micro:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model Size: {model_size_mb:.1f} MB ({total_params:,} params)")
    print(f"Edge Cases: {passed}/{len(edge_results)}")
    print(f"Latency (p50): {latency['latency_p50_ms']:.2f} ms")


if __name__ == "__main__":
    main()
