# FinSenti - Financial Sentiment Analysis MLOps Pipeline

An end-to-end MLOps pipeline for financial sentiment analysis. Fine-tunes FinBERT and distilBERT with LoRA adapters, tracks experiments with MLflow, and serves predictions via FastAPI + Next.js.

## Features

- **LoRA Fine-tuning** of FinBERT and distilBERT on financial text data
- **MLflow Experiment Tracking** for hyperparameters, metrics, and model registry
- **FastAPI Backend** with single and batch prediction endpoints
- **Next.js Frontend** with sentiment visualization dashboard
- **Docker Compose** for reproducible deployment

## Quick Start

```bash
# Clone
git clone https://github.com/last-26/FinSenti.git
cd FinSenti

# Training pipeline
cd training
py -3.11 -m venv venv
source venv/Scripts/activate  # Windows
pip install -r requirements.txt
python data/prepare_dataset.py
python train.py --config configs/finbert_lora.yaml

# Backend
cd ../backend
py -3.11 -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend
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
| Model Serving | FastAPI |
| Frontend | Next.js 14 + TypeScript + TailwindCSS |
| Containerization | Docker + Docker Compose |

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

**Winner: FinBERT + LoRA** - Domain-specific pre-training provides superior F1 despite fewer trainable parameters.

### Edge Case Results (Both models: 6/6)
- "The company maintained its dividend" -> neutral
- "Revenue increased but margins declined sharply" -> negative
- "EPS $2.45 vs $2.30 expected, revenue $12.1B vs $11.8B consensus" -> positive
- "$TSLA to the moon" -> positive
- "The board will meet on Tuesday to discuss Q2 results" -> neutral
- "Shares dropped 15% after disappointing guidance" -> negative

## Project Status

**Phase 1 - Complete:** Data & Training Foundation
**Phase 2 - Complete:** Model Comparison & Evaluation

See [CLAUDE.md](CLAUDE.md) for detailed architecture and implementation plan.

## License

MIT
