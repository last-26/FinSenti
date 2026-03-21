"""
Download, merge, and split Financial PhraseBank + FiQA datasets.

Produces train/val/test splits (80/10/10) with stratified sampling.
Saves processed datasets to ./data/processed/ as HuggingFace Dataset.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

# Label mapping: 0=negative, 1=neutral, 2=positive
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
SEED = 42
OUTPUT_DIR = Path(__file__).parent / "processed"


def load_financial_phrasebank() -> list[dict]:
    """Load Financial PhraseBank (sentences_allagree subset).

    Downloads the zip from HuggingFace Hub and parses the raw text file
    since the dataset's loading script is no longer supported.
    """
    print("Loading Financial PhraseBank...")
    zip_path = hf_hub_download(
        repo_id="financial_phrasebank",
        filename="data/FinancialPhraseBank-v1.0.zip",
        repo_type="dataset",
    )

    # Label mapping in the raw file: negative=0, neutral=1, positive=2
    fpb_label_map = {"negative": 0, "neutral": 1, "positive": 2}
    target_file = "Sentences_AllAgree.txt"

    records = []
    with zipfile.ZipFile(zip_path) as zf:
        # Find the target file in the zip
        matching = [n for n in zf.namelist() if target_file in n]
        if not matching:
            raise FileNotFoundError(f"{target_file} not found in zip: {zf.namelist()}")

        with zf.open(matching[0]) as f:
            for line in io.TextIOWrapper(f, encoding="utf-8", errors="replace"):
                line = line.strip()
                if not line:
                    continue
                # Format: "sentence text@label"
                if "@" not in line:
                    continue
                text, label_str = line.rsplit("@", 1)
                label_str = label_str.strip().lower()
                if label_str not in fpb_label_map:
                    continue
                records.append({
                    "text": text.strip(),
                    "label": fpb_label_map[label_str],
                    "source": "financial_phrasebank",
                })

    print(f"  Loaded {len(records)} samples from Financial PhraseBank")
    return records


def load_fiqa() -> list[dict]:
    """Load FiQA 2018 sentiment dataset and bin scores to 3 classes."""
    print("Loading FiQA Sentiment...")
    ds = load_dataset("pauri32/fiqa-2018")

    records = []
    for split_name in ds:
        for row in ds[split_name]:
            score = row.get("score") or row.get("sentiment_score")
            text = row.get("sentence") or row.get("text", "")
            if score is None or not text:
                continue

            score = float(score)
            text = text.strip()
            if not text:
                continue

            # Bin continuous score to 3 classes
            # FiQA scores: -1 to 1 range
            if score <= -0.2:
                label = LABEL_MAP["negative"]
            elif score >= 0.2:
                label = LABEL_MAP["positive"]
            else:
                label = LABEL_MAP["neutral"]

            records.append({
                "text": text,
                "label": label,
                "source": "fiqa",
            })

    # Deduplicate by text
    seen = set()
    unique_records = []
    for r in records:
        if r["text"] not in seen:
            seen.add(r["text"])
            unique_records.append(r)

    print(f"  Loaded {len(unique_records)} unique samples from FiQA")
    return unique_records


def merge_and_split(
    phrasebank: list[dict],
    fiqa: list[dict],
) -> DatasetDict:
    """Merge datasets and create stratified train/val/test splits."""
    all_records = phrasebank + fiqa
    np.random.seed(SEED)
    np.random.shuffle(all_records)

    texts = [r["text"] for r in all_records]
    labels = [r["label"] for r in all_records]
    sources = [r["source"] for r in all_records]

    print(f"\nTotal samples: {len(texts)}")
    for label_name, label_id in LABEL_MAP.items():
        count = labels.count(label_id)
        print(f"  {label_name}: {count} ({count / len(labels) * 100:.1f}%)")

    # Stratified split: 80% train, 10% val, 10% test
    train_texts, temp_texts, train_labels, temp_labels, train_sources, temp_sources = (
        train_test_split(
            texts, labels, sources,
            test_size=0.2,
            random_state=SEED,
            stratify=labels,
        )
    )
    val_texts, test_texts, val_labels, test_labels, val_sources, test_sources = (
        train_test_split(
            temp_texts, temp_labels, temp_sources,
            test_size=0.5,
            random_state=SEED,
            stratify=temp_labels,
        )
    )

    def make_dataset(texts: list, labels: list, sources: list) -> Dataset:
        return Dataset.from_dict({
            "text": texts,
            "label": labels,
            "source": sources,
        })

    dataset_dict = DatasetDict({
        "train": make_dataset(train_texts, train_labels, train_sources),
        "validation": make_dataset(val_texts, val_labels, val_sources),
        "test": make_dataset(test_texts, test_labels, test_sources),
    })

    print(f"\nSplit sizes:")
    for split_name, split_ds in dataset_dict.items():
        print(f"  {split_name}: {len(split_ds)}")

    return dataset_dict


def main() -> None:
    os.environ.setdefault("HF_HOME", str(Path(__file__).parent.parent / "cache" / "huggingface"))

    phrasebank = load_financial_phrasebank()
    fiqa = load_fiqa()
    dataset = merge_and_split(phrasebank, fiqa)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(OUTPUT_DIR))
    print(f"\nDataset saved to {OUTPUT_DIR}")

    # Save a few examples for quick inspection
    print("\nSample entries:")
    label_names = {v: k for k, v in LABEL_MAP.items()}
    for i in range(min(5, len(dataset["train"]))):
        row = dataset["train"][i]
        print(f"  [{label_names[row['label']]}] {row['text'][:80]}...")


if __name__ == "__main__":
    main()
