"""
LoRA fine-tuning script with MLflow experiment tracking.

Usage:
    python train.py --config configs/finbert_lora.yaml
    python train.py --config configs/distilbert_lora.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_metrics(eval_pred) -> dict:
    """Compute classification metrics for Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_micro": f1_score(labels, predictions, average="micro"),
        "precision_macro": precision_score(labels, predictions, average="macro"),
        "recall_macro": recall_score(labels, predictions, average="macro"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune model with LoRA")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--data-dir", default="./data/processed", help="Path to processed dataset")
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = Path(args.data_dir)

    os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "cache" / "huggingface"))

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = config["training"]["fp16"] and device == "cuda"
    print(f"Device: {device}, FP16: {use_fp16}")

    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    dataset = load_from_disk(str(data_dir))

    # Load tokenizer and model
    model_name = config["base_model"]
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=config["num_labels"],
    )

    # Apply LoRA
    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.SEQ_CLS,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Tokenize dataset
    max_length = config["training"]["max_length"]

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text", "source"])

    # Training arguments
    train_cfg = config["training"]
    eval_cfg = config["eval"]
    output_dir = train_cfg["output_dir"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"] * 2,
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        fp16=use_fp16,
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        eval_strategy=eval_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        load_best_model_at_end=eval_cfg["load_best_model_at_end"],
        metric_for_best_model=eval_cfg["metric_for_best_model"],
        greater_is_better=True,
        logging_steps=train_cfg["logging_steps"],
        report_to="none",  # We log to MLflow manually
        seed=train_cfg["seed"],
        remove_unused_columns=False,
    )

    # MLflow setup — use local file store if server unreachable
    mlflow_cfg = config["mlflow"]
    tracking_uri = mlflow_cfg["tracking_uri"]
    try:
        import urllib.request
        resp = urllib.request.urlopen(f"{tracking_uri}/api/2.0/mlflow/experiments/search", timeout=3)
        resp.read()
    except Exception:
        mlruns_path = Path(__file__).parent / "mlruns"
        tracking_uri = mlruns_path.as_uri()
        print(f"MLflow server not available, using local store: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run(run_name=mlflow_cfg["run_name"]) as run:
        # Log config
        mlflow.log_params({
            "base_model": model_name,
            "lora_r": lora_cfg["r"],
            "lora_alpha": lora_cfg["lora_alpha"],
            "lora_dropout": lora_cfg["lora_dropout"],
            "lora_target_modules": str(lora_cfg["target_modules"]),
            "epochs": train_cfg["epochs"],
            "batch_size": train_cfg["batch_size"],
            "learning_rate": train_cfg["learning_rate"],
            "max_length": max_length,
            "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
            "fp16": use_fp16,
            "train_samples": len(tokenized["train"]),
            "val_samples": len(tokenized["validation"]),
            "test_samples": len(tokenized["test"]),
        })

        # Train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        print("Starting training...")
        train_result = trainer.train()

        # Log training metrics
        mlflow.log_metrics({
            "train_loss": train_result.metrics["train_loss"],
            "train_runtime": train_result.metrics["train_runtime"],
            "train_samples_per_second": train_result.metrics["train_samples_per_second"],
        })

        # Evaluate on test set
        print("Evaluating on test set...")
        test_metrics = trainer.evaluate(tokenized["test"])
        mlflow.log_metrics({
            f"test_{k.replace('eval_', '')}": v
            for k, v in test_metrics.items()
            if isinstance(v, (int, float))
        })

        # Save LoRA adapter
        adapter_dir = Path(output_dir) / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        mlflow.log_artifacts(str(adapter_dir), artifact_path="adapter")

        # Log model info
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        mlflow.log_metrics({
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": trainable / total * 100,
        })

        print(f"\nRun ID: {run.info.run_id}")
        print(f"Experiment: {mlflow_cfg['experiment_name']}")
        print(f"Test F1 (macro): {test_metrics.get('eval_f1_macro', 'N/A')}")
        print(f"Adapter saved to: {adapter_dir}")


if __name__ == "__main__":
    main()
