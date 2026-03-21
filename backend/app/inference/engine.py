"""
Inference engine for financial sentiment prediction.

Loads a base model + LoRA adapter, runs inference, and returns results.
Designed to be loaded once on app startup and reused across requests.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.inference.postprocessing import format_prediction
from app.inference.preprocessing import clean_text, extract_entities


class SentimentEngine:
    """Manages model loading and inference for sentiment prediction."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_name: str = ""
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length: int = 128

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load_from_adapter(self, adapter_dir: str, model_name: str = "") -> None:
        """Load a base model + LoRA adapter and merge for inference."""
        adapter_path = Path(adapter_dir)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

        print(f"Loading adapter from {adapter_dir}...")
        peft_config = PeftConfig.from_pretrained(str(adapter_path))
        base_model_name = peft_config.base_model_name_or_path

        self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=3,
        )
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
        self.model = peft_model.merge_and_unload()
        self.model.to(self.device)
        self.model.eval()
        self.model_name = model_name or adapter_path.parent.name
        print(f"Model loaded: {self.model_name} on {self.device}")

    def load_pretrained(self, model_name: str = "ProsusAI/finbert") -> None:
        """Load a pretrained model directly (no LoRA adapter)."""
        print(f"Loading pretrained model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
        )
        self.model.to(self.device)
        self.model.eval()
        self.model_name = model_name
        print(f"Model loaded: {self.model_name} on {self.device}")

    def predict(self, text: str) -> dict:
        """Run sentiment prediction on a single text."""
        if not self.is_loaded:
            raise RuntimeError("No model loaded. Call load_from_adapter() first.")

        cleaned = clean_text(text)
        entities = extract_entities(text)

        inputs = self.tokenizer(
            cleaned,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        elapsed_ms = (time.perf_counter() - start) * 1000

        probabilities = probs[0].cpu().numpy().tolist()
        return format_prediction(text, probabilities, self.model_name, elapsed_ms, entities)

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        """Run sentiment prediction on a batch of texts."""
        if not self.is_loaded:
            raise RuntimeError("No model loaded. Call load_from_adapter() first.")

        results = []
        total_start = time.perf_counter()

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            cleaned = [clean_text(t) for t in batch_texts]

            inputs = self.tokenizer(
                cleaned,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            start = time.perf_counter()
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            elapsed_ms = (time.perf_counter() - start) * 1000
            per_sample_ms = elapsed_ms / len(batch_texts)

            for j, text in enumerate(batch_texts):
                entities = extract_entities(texts[i + j])
                probabilities = probs[j].cpu().numpy().tolist()
                results.append(
                    format_prediction(
                        texts[i + j], probabilities, self.model_name, per_sample_ms, entities
                    )
                )

        return results
