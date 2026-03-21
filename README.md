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

## Project Status

**Phase 1 - In Progress:** Data & Training Foundation

See [CLAUDE.md](CLAUDE.md) for detailed architecture and implementation plan.

## License

MIT
