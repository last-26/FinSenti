# FinSenti - Financial Sentiment Analysis MLOps Pipeline

An end-to-end MLOps pipeline for financial sentiment analysis. Fine-tunes FinBERT and distilBERT with LoRA adapters, tracks experiments with MLflow, and serves predictions via FastAPI + Next.js.
![Python](https://img.shields.io/badge/Python-3.11-blue)
![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178c6)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/last-26/FinSenti/actions/workflows/ci.yml/badge.svg)
## Features

- **LoRA Fine-tuning** of FinBERT and distilBERT on financial text data
- **MLflow Experiment Tracking** for hyperparameters, metrics, and model registry
- **Model Comparison** — side-by-side evaluation of domain-specific vs general-purpose models
- **FastAPI Backend** with single and batch prediction endpoints, SQLite logging
- **Next.js Frontend** with dashboard, sentiment visualization, experiment browser, and prediction history
- **Docker Compose** deployment with health checks and service orchestration
- **CI/CD Pipeline** with GitHub Actions (lint, test, build, Docker)
- **Latency Benchmarks** for API endpoint performance profiling

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  Next.js Frontend (:3000)                  │
│  Dashboard │ Predict │ Batch │ Experiments │ Models │ History │
└──────────────────────────┬───────────────────────────────┘
                           │ REST API
┌──────────────────────────▼───────────────────────────────┐
│                  FastAPI Backend (:8000)                   │
│  /predict │ /batch │ /models │ /experiments │ /history     │
│                                                           │
│  Inference Engine (LoRA adapter merge) + SQLite logging   │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│               MLflow Tracking Server (:5000)              │
│  Experiments │ Metrics │ Artifacts │ Model Registry        │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│               Training Pipeline (offline)                 │
│  Data Prep → LoRA Fine-tune → Evaluate → Register Model  │
└──────────────────────────────────────────────────────────┘
```

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
git clone https://github.com/last-26/FinSenti.git
cd FinSenti

# Start all services (MLflow + Backend + Frontend)
docker compose up --build

# Access:
#   Frontend:  http://localhost:3000
#   Backend:   http://localhost:8000
#   MLflow:    http://localhost:5000
```

> **Note:** Training outputs must exist before starting the backend. See the Training section below.

### Option 2: Local Development

```bash
# 1. Training pipeline
cd training
py -3.11 -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python data/prepare_dataset.py
python train.py --config configs/finbert_lora.yaml

# 2. Backend API
cd ../backend
py -3.11 -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# 3. Frontend
cd ../frontend
npm install
npm run dev
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Training | Hugging Face Transformers + PEFT (LoRA) |
| Experiment Tracking | MLflow |
| Data | Financial PhraseBank + FiQA |
| Evaluation | scikit-learn + custom benchmarks |
| Model Serving | FastAPI + Transformers inference |
| Frontend | Next.js 14 (App Router) + TypeScript + TailwindCSS |
| Database | SQLite (prediction logs) |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |

## Training Results

### Dataset
- **Financial PhraseBank** (2,264 samples) + **FiQA** (1,111 samples) = **3,375 total**
- Split: Train 2,700 / Val 337 / Test 338 (stratified 80/10/10)

### Model Comparison

| Metric | FinBERT + LoRA | distilBERT + LoRA |
|--------|:---:|:---:|
| **F1 Macro** | **0.8976** | 0.8724 |
| **Accuracy** | **91.1%** | 88.8% |
| Latency (p50) | 9.93 ms | **5.51 ms** |
| Model Size | 417.7 MB | **255.4 MB** |
| Trainable Params | 592K (0.54%) | 1.18M (1.74%) |
| Edge Cases | **6/6** | **6/6** |
| Training Time | 82s | **65s** |

**Winner: FinBERT + LoRA** — Domain-specific pre-training provides superior F1 despite fewer trainable parameters.

### Edge Case Results (Both models: 6/6)

| Text | Expected | Result |
|------|----------|--------|
| "The company maintained its dividend" | neutral | PASS |
| "Revenue increased but margins declined sharply" | negative | PASS |
| "EPS $2.45 vs $2.30 expected, revenue $12.1B vs $11.8B consensus" | positive | PASS |
| "$TSLA to the moon" | positive | PASS |
| "The board will meet on Tuesday to discuss Q2 results" | neutral | PASS |
| "Shares dropped 15% after disappointing guidance" | negative | PASS |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/predict` | Classify single financial text |
| `POST` | `/api/v1/batch` | Classify multiple texts (up to 64) |
| `GET` | `/api/v1/models` | List registered models |
| `GET` | `/api/v1/models/active` | Currently loaded model |
| `POST` | `/api/v1/models/switch` | Switch active model |
| `GET` | `/api/v1/experiments` | MLflow experiment summaries |
| `GET` | `/api/v1/experiments/{id}/runs` | Runs for experiment |
| `GET` | `/api/v1/history` | Prediction log (paginated) |
| `GET` | `/api/v1/health` | Health check |

### Example

```bash
# Single prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Tesla reported record Q4 deliveries beating expectations"}'

# Response
{
  "text": "Tesla reported record Q4 deliveries beating expectations",
  "sentiment": "positive",
  "confidence": 0.94,
  "probabilities": {"positive": 0.94, "neutral": 0.04, "negative": 0.02},
  "entities": ["Tesla", "Q4"],
  "market_signal": "bullish",
  "model_used": "finbert-lora",
  "inference_time_ms": 45.2
}
```

## Frontend Pages

| Page | Route | Description |
|------|-------|-------------|
| Dashboard | `/` | API status, active model, quick actions, recent predictions |
| Predict | `/predict` | Single text input with example texts, sentiment gauge, entity tags |
| Batch | `/batch` | Multi-text input, summary cards, results table |
| Experiments | `/experiments` | MLflow experiment selector, runs metrics table |
| Models | `/models` | Model cards with switch capability |
| History | `/history` | Paginated prediction history table |

## Benchmarks

Run the latency benchmark against a running API:

```bash
cd backend
python benchmark.py --url http://localhost:8000 --rounds 50
```

Options:
- `--rounds N` — Number of benchmark rounds per test (default: 50)
- `--warmup N` — Number of warmup requests (default: 5)
- `--batch-sizes 1 4 8 16` — Batch sizes to test

## Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `frontend` | 3000 | Next.js production build |
| `backend` | 8000 | FastAPI with model inference |
| `mlflow` | 5000 | Experiment tracking server |

All services include health checks, restart policies, and proper dependency ordering.

```bash
# Start all services
docker compose up --build

# Start only MLflow (for training)
docker compose up mlflow

# View logs
docker compose logs -f backend
```

## CI/CD

GitHub Actions pipeline runs on push to `master`/`dev` and PRs to `master`:

1. **Backend Lint** — ruff check
2. **Backend Tests** — pytest (32 tests)
3. **Training Lint** — ruff check
4. **Frontend Lint** — next lint
5. **Frontend Build** — next build
6. **Docker Build** — docker compose build (runs after all checks pass)

## Project Structure

```
FinSenti/
├── training/                  # ML training pipeline
│   ├── data/                  # Dataset preparation
│   ├── configs/               # LoRA training configs (YAML)
│   ├── train.py               # LoRA fine-tuning + MLflow
│   ├── evaluate.py            # Evaluation suite
│   ├── compare_models.py      # Model comparison
│   └── register_model.py      # MLflow model registry
├── backend/                   # FastAPI application
│   ├── app/
│   │   ├── api/routes/        # REST endpoints
│   │   ├── inference/         # Model loading & prediction
│   │   ├── schemas/           # Pydantic models
│   │   └── models/            # Database models
│   ├── tests/                 # Unit tests (32 tests)
│   └── benchmark.py           # API latency benchmark
├── frontend/                  # Next.js 14 application
│   └── src/
│       ├── app/               # Pages (App Router)
│       ├── components/        # React components
│       └── lib/               # API client & utilities
├── .github/workflows/ci.yml   # CI/CD pipeline
├── docker-compose.yml         # Multi-service orchestration
└── CLAUDE.md                  # Detailed architecture docs
```

## License

MIT
