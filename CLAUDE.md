# FinSenti — Financial Sentiment Analysis MLOps Pipeline

## Project Overview

FinSenti is an **end-to-end MLOps pipeline** for financial sentiment analysis. It fine-tunes pre-trained language models on financial text data, tracks experiments with MLflow, evaluates model performance with comprehensive benchmarks, and serves the best model via a production-grade FastAPI backend with a Next.js frontend.

This project demonstrates the **full ML lifecycle**: data preparation → fine-tuning (LoRA) → experiment tracking → evaluation → model registry → serving → monitoring. Two base models are compared side-by-side — FinBERT (domain-specific) vs distilBERT (general-purpose) — to demonstrate experiment-driven model selection.

**Core Problem:** Given a financial text (news headline, tweet, analyst statement), classify the sentiment as **positive**, **negative**, or **neutral** with a confidence score, and extract key financial entities.

**Current Status (March 2026):** Phases 1–4 complete. Full pipeline working: training, evaluation, API serving, frontend UI. Phase 5 (polish & DevOps) next.

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Training | **Hugging Face Transformers + PEFT (LoRA)** | Industry standard, LoRA reduces memory/compute |
| Experiment Tracking | **MLflow** | Track hyperparameters, metrics, artifacts, compare runs |
| Data | **Financial PhraseBank + FiQA** (HF Datasets) | High-quality labeled financial sentiment data |
| Evaluation | **scikit-learn + custom benchmarks** | F1, precision, recall, confusion matrix |
| Model Serving | **FastAPI** + Transformers inference | Async, fast, consistent with NextHire |
| Frontend | **Next.js 14 (App Router)** + TypeScript + TailwindCSS + shadcn/ui | Modern React, consistent portfolio |
| Database | **SQLite** (prediction logs) | Lightweight, no separate DB needed |
| Containerization | **Docker + Docker Compose** | MLflow + API + Frontend in containers |
| CI/CD | **GitHub Actions** | Lint → Test → Docker Build |

---

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Next.js Frontend                         │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌──────────────┐  │
│  │Dashboard │ │  Predict  │ │ Experi-  │ │    Model     │  │
│  │(stats &  │ │ (single & │ │  ments   │ │  Comparison  │  │
│  │ history) │ │   batch)  │ │ (MLflow) │ │    View      │  │
│  └──────────┘ └─────┬─────┘ └──────────┘ └──────────────┘  │
└─────────────────────┼───────────────────────────────────────┘
                      │ REST API
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ POST /predict│  │ POST /batch  │  │ GET /models      │  │
│  │ (single text)│  │ (CSV/list)   │  │ GET /experiments │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
│         │                 │                    │             │
│  ┌──────▼─────────────────▼────────────────────▼─────────┐  │
│  │              Inference Engine                          │  │
│  │  ┌────────────────┐  ┌─────────────────────────────┐  │  │
│  │  │ Model Registry │  │ Preprocessing Pipeline      │  │  │
│  │  │ (best model    │  │ (tokenize, clean, normalize)│  │  │
│  │  │  from MLflow)  │  │                             │  │  │
│  │  └────────────────┘  └─────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│                         │                                    │
│  ┌──────────────────────▼────────────────────────────────┐  │
│  │         SQLite (prediction logs & history)             │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  Training Pipeline (offline)                  │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │  Data    │→ │  Fine-   │→ │  Eval    │→ │  Register  │  │
│  │  Prep    │  │  Tune    │  │  Suite   │  │  in MLflow │  │
│  │ (HF DS) │  │ (LoRA)   │  │ (F1,etc) │  │            │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────────┘  │
│       │              │             │              │          │
│       ▼              ▼             ▼              ▼          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              MLflow Tracking Server                   │   │
│  │  - Hyperparameters (lr, epochs, lora_r, batch_size)  │   │
│  │  - Metrics (F1, accuracy, loss curves)               │   │
│  │  - Artifacts (model checkpoints, confusion matrix)   │   │
│  │  - Model Registry (staging → production)             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Training Pipeline Flow

```
Financial PhraseBank (4,840 sentences)
         +
FiQA Sentiment (~1,100 sentences)
         │
         ▼
┌─────────────────────┐
│   Data Preparation   │
│  - Merge datasets    │
│  - Stratified split  │
│    (80/10/10)        │
│  - Label encoding    │
│  - Text cleaning     │
└─────────┬───────────┘
          │
    ┌─────┴──────┐
    ▼            ▼
┌────────┐  ┌────────────┐
│FinBERT │  │ distilBERT │
│ + LoRA │  │  + LoRA    │
└───┬────┘  └─────┬──────┘
    │             │
    ▼             ▼
┌────────────────────────┐
│   MLflow Experiment    │
│   - Compare F1 scores  │
│   - Compare loss curves│
│   - Compare latency    │
└─────────┬──────────────┘
          │
          ▼
┌────────────────────────┐
│   Evaluation Suite     │
│   - F1 (macro/micro)   │
│   - Confusion matrix   │
│   - Inference latency  │
│   - Edge cases test    │
└─────────┬──────────────┘
          │
          ▼
┌────────────────────────┐
│   Best Model → MLflow  │
│   Model Registry       │
│   (stage: Production)  │
└────────────────────────┘
```

---

## Datasets

### Financial PhraseBank
- **Source:** `financial_phrasebank` on Hugging Face
- **Size:** 4,840 sentences from financial news
- **Labels:** positive, negative, neutral
- **Quality:** Annotated by 5-8 finance professionals, `sentences_allagree` subset
- **Example:** "Operating profit rose to EUR 14.0 mn from EUR 8.3 mn" → positive

### FiQA Sentiment
- **Source:** `pauri32/fiqa-2018` on Hugging Face
- **Size:** ~1,100 financial tweets and headlines
- **Labels:** Continuous score (-1 to 1), binned to positive/negative/neutral

### Combined Dataset Split
```
Total: ~5,940 samples
├── Train: ~4,750 (80%)
├── Validation: ~595 (10%)
└── Test: ~595 (10%)
```

---

## Project Structure

```
FinSenti/
├── CLAUDE.md
├── README.md
├── docker-compose.yml
├── .env.example
├── .gitignore
│
├── training/                          # ★ ML Training Pipeline ★
│   ├── requirements.txt
│   ├── data/
│   │   ├── prepare_dataset.py         # Download, merge, split
│   │   └── README.md
│   ├── configs/
│   │   ├── finbert_lora.yaml
│   │   └── distilbert_lora.yaml
│   ├── train.py                       # ★ LoRA fine-tuning + MLflow
│   ├── evaluate.py                    # ★ Evaluation suite
│   ├── compare_models.py             # Compare runs
│   └── register_model.py             # Promote best to registry
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py                    # FastAPI (model on startup)
│   │   ├── config.py
│   │   ├── database.py                # SQLite
│   │   ├── api/routes/
│   │   │   ├── predict.py             # POST /api/v1/predict
│   │   │   ├── batch.py               # POST /api/v1/batch
│   │   │   ├── models.py
│   │   │   ├── experiments.py
│   │   │   ├── history.py
│   │   │   └── health.py
│   │   ├── inference/
│   │   │   ├── engine.py             # ★ Model loading & inference
│   │   │   ├── preprocessing.py
│   │   │   └── postprocessing.py
│   │   ├── models/
│   │   │   └── prediction_log.py
│   │   └── schemas/
│   │       ├── predict.py
│   │       └── experiment.py
│   └── tests/
│
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── src/app/
│   │   ├── layout.tsx
│   │   ├── page.tsx                   # Dashboard
│   │   ├── predict/page.tsx           # ★ Text → Sentiment
│   │   ├── batch/page.tsx
│   │   ├── experiments/page.tsx       # MLflow metrics
│   │   ├── history/page.tsx
│   │   └── models/page.tsx            # Model comparison
│   ├── src/components/
│   │   ├── predict/
│   │   │   ├── SentimentInput.tsx
│   │   │   ├── SentimentResult.tsx
│   │   │   └── SentimentGauge.tsx
│   │   ├── experiments/
│   │   │   ├── MetricsTable.tsx
│   │   │   ├── ConfusionMatrix.tsx
│   │   │   └── LossChart.tsx
│   │   └── models/
│   │       ├── ModelCard.tsx
│   │       └── ModelComparison.tsx
│   └── src/lib/
│       ├── api.ts
│       └── utils.ts
│
├── mlflow/                            # MLflow data (gitignored)
│
└── .github/workflows/ci.yml
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/predict` | Classify single financial text |
| POST | `/api/v1/batch` | Classify multiple texts |
| GET | `/api/v1/models` | List registered models |
| GET | `/api/v1/models/active` | Currently loaded model |
| POST | `/api/v1/models/switch` | Switch active model |
| GET | `/api/v1/experiments` | MLflow experiment summaries |
| GET | `/api/v1/experiments/{id}/runs` | Runs for experiment |
| GET | `/api/v1/history` | Prediction log (paginated) |
| GET | `/api/v1/health` | Health check |

### Request/Response Examples

**POST /api/v1/predict**
```json
// Request
{
  "text": "Tesla reported record Q4 deliveries beating analyst expectations by 12%",
  "model": "finbert-lora"
}

// Response
{
  "text": "Tesla reported record Q4 deliveries beating analyst expectations by 12%",
  "sentiment": "positive",
  "confidence": 0.94,
  "probabilities": {
    "positive": 0.94,
    "neutral": 0.04,
    "negative": 0.02
  },
  "entities": ["Tesla", "Q4"],
  "market_signal": "bullish",
  "model_used": "finbert-lora-v1",
  "inference_time_ms": 45
}
```

**POST /api/v1/batch**
```json
// Request
{
  "texts": [
    "Fed raised interest rates by 25 basis points",
    "Apple announced a new stock buyback program worth $90B",
    "Oil prices remained steady amid OPEC uncertainty"
  ]
}

// Response
{
  "results": [
    {"text": "...", "sentiment": "negative", "confidence": 0.89},
    {"text": "...", "sentiment": "positive", "confidence": 0.96},
    {"text": "...", "sentiment": "neutral", "confidence": 0.78}
  ],
  "summary": {
    "positive": 1, "negative": 1, "neutral": 1,
    "avg_confidence": 0.88
  },
  "total_inference_time_ms": 128
}
```

---

## Training Configuration

### FinBERT + LoRA (configs/finbert_lora.yaml)

```yaml
base_model: "ProsusAI/finbert"
num_labels: 3

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules: ["query", "value"]
  task_type: "SEQ_CLS"

training:
  epochs: 5
  batch_size: 16
  learning_rate: 2e-4
  weight_decay: 0.01
  warmup_ratio: 0.1
  max_length: 128
  fp16: true
  gradient_accumulation_steps: 2

eval:
  eval_strategy: "epoch"
  metric_for_best_model: "f1_macro"
  load_best_model_at_end: true

mlflow:
  experiment_name: "finsenti-finbert"
  tracking_uri: "http://localhost:5000"
```

### distilBERT + LoRA (configs/distilbert_lora.yaml)

```yaml
base_model: "distilbert-base-uncased"
num_labels: 3

lora:
  r: 32
  lora_alpha: 64
  lora_dropout: 0.1
  target_modules: ["q_lin", "v_lin"]
  task_type: "SEQ_CLS"

training:
  epochs: 8
  batch_size: 32
  learning_rate: 3e-4
  weight_decay: 0.01
  warmup_ratio: 0.1
  max_length: 128
  fp16: true
  gradient_accumulation_steps: 1

eval:
  eval_strategy: "epoch"
  metric_for_best_model: "f1_macro"
  load_best_model_at_end: true

mlflow:
  experiment_name: "finsenti-distilbert"
  tracking_uri: "http://localhost:5000"
```

---

## Evaluation Suite

### Metrics per model:
1. **Classification Report:** Per-class precision, recall, F1
2. **Confusion Matrix:** Heatmap (MLflow artifact)
3. **Macro/Micro F1:** Primary comparison metric
4. **Inference Latency:** Average ms per prediction (CPU)
5. **Edge Case Tests:** Predefined tricky sentences
6. **Model Size:** Base + adapter weights

### Edge Cases
```python
EDGE_CASES = [
    ("The company maintained its dividend", "neutral"),
    ("Revenue increased but margins declined sharply", "negative"),
    ("EPS $2.45 vs $2.30 expected, revenue $12.1B vs $11.8B consensus", "positive"),
    ("$TSLA to the moon", "positive"),
    ("The board will meet on Tuesday to discuss Q2 results", "neutral"),
    ("Shares dropped 15% after disappointing guidance", "negative"),
]
```

---

## Implementation Phases

### Phase 1: Data & Training Foundation ✅
- [x] Project scaffolding
- [x] `data/prepare_dataset.py` — downloads FPB zip + FiQA, merges, stratified split (3,375 samples)
- [x] Training config YAMLs
- [x] `train.py` — LoRA + MLflow logging (local file store fallback)
- [x] `evaluate.py` — metrics + confusion matrix + latency + edge cases
- [x] Docker Compose with MLflow
- [x] Train FinBERT + LoRA — **F1: 0.8976, Acc: 91.1%**

### Phase 2: Model Comparison & Registry ✅
- [x] Train distilBERT + LoRA — **F1: 0.8724, Acc: 88.8%**
- [x] `compare_models.py`
- [x] `register_model.py`
- [x] Edge case evaluation — **6/6 both models**
- [x] Document results in README

### Phase 3: Inference API ✅
- [x] FastAPI backend with lifespan model loading
- [x] Inference engine (base model + LoRA adapter merge)
- [x] Preprocessing (text cleaning, entity extraction) + postprocessing
- [x] `/predict` and `/batch` endpoints with SQLite logging
- [x] `/models`, `/experiments`, `/history`, `/health` endpoints
- [x] MLflow experiments with local mlruns fallback
- [x] Backend venv setup + server verified on trained FinBERT adapter
- [x] Integration testing — all endpoints verified with real model
- [x] Unit tests — **32 tests passing** (endpoints, preprocessing, postprocessing)
- [x] Ruff lint clean

### Phase 4: Frontend ✅
- [x] Next.js 14 (App Router) + TypeScript + TailwindCSS setup
- [x] Layout with navigation header + API client library
- [x] Dashboard page — API status, active model, quick actions, recent predictions
- [x] Predict page — text input with examples, sentiment gauge, entity tags, market signal
- [x] Batch page — multi-text input, summary cards, results table
- [x] Experiments page — MLflow experiment selector, runs metrics table
- [x] Models page — model cards with switch capability
- [x] History page — paginated prediction history table
- [x] Build verified — zero errors

### Phase 5: Polish & DevOps
- [ ] CI/CD pipeline
- [ ] Full Docker Compose
- [ ] README with results + screenshots
- [ ] Latency benchmarks

---

## Commands

### Training
```bash
cd training
pip install -r requirements.txt
python data/prepare_dataset.py
python train.py --config configs/finbert_lora.yaml
python train.py --config configs/distilbert_lora.yaml
python evaluate.py --run-id <mlflow_run_id>
python compare_models.py
python register_model.py --model-name finsenti-finbert --stage Production
```

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
pytest -v
ruff check .
```

### Frontend
```bash
cd frontend
npm install
npm run dev
npm run build
npm run lint
```

### Docker
```bash
docker compose up --build
docker compose up mlflow
```

---

## Key Design Decisions

1. **LoRA over full fine-tuning:** ~100x fewer trainable params, trains on laptop, adapter ~5MB vs ~400MB.
2. **Two models compared:** Demonstrates experiment-driven model selection — core MLOps.
3. **MLflow (self-hosted):** No external accounts, runs in Docker, industry standard.
4. **YAML configs:** Reproducible experiments, version-controllable.
5. **SQLite for prediction logs:** Lightweight, no separate DB.
6. **Same frontend stack as NextHire:** Portfolio consistency.
7. **Entity extraction via heuristic:** Regex-based, keeps focus on MLOps pipeline.

---

## Critical Notes for Claude Code

1. **Python 3.11** — System has Python 3.14 as default but PyTorch requires 3.11. Always use `py -3.11` to create venvs: `py -3.11 -m venv venv`. PyTorch 2.5.1+cu121 is installed in system Python 3.11.
2. **Training scripts are standalone** — no dependency on FastAPI backend.
3. **MLflow URI** — `http://mlflow:5000` in Docker, `http://localhost:5000` locally.
4. **Model loading in FastAPI** — load once on startup (`lifespan`), store in `app.state`.
5. **LoRA adapters** — save only adapter weights. Inference loads base + merges adapter.
6. **HF cache** — set `HF_HOME=./cache/huggingface`. Don't re-download per run.
7. **fp16** — auto-detect GPU support, fallback to fp32.
8. **PEFT library** — `peft>=0.7.0`. Pattern: load base → load adapter → merge for inference.
9. **HuggingFace Datasets** — `financial_phrasebank` and `pauri32/fiqa-2018` are public. No API key needed.

---

## Branch Strategy

- `main` — production-ready
- `dev` — active development
- Feature branches: `feat/training-pipeline`, `feat/inference-api`, etc.

---

## Coding Standards

- **Python:** PEP 8, type hints, ruff. Training scripts use `if __name__ == "__main__"`.
- **TypeScript:** Strict mode, explicit types, no `any`.
- **Commits:** Conventional commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`)
- **API:** RESTful, `{"detail": "message"}` error format.
- **Config:** All paths/URIs from `.env` or YAML. Never hardcode.
