"""Post-processing utilities for inference results."""

from __future__ import annotations

LABEL_NAMES = ["negative", "neutral", "positive"]

# Mapping from sentiment to market signal
MARKET_SIGNALS = {
    "positive": "bullish",
    "negative": "bearish",
    "neutral": "hold",
}


def format_prediction(
    text: str,
    probabilities: list[float],
    model_name: str,
    inference_time_ms: float,
    entities: list[str],
) -> dict:
    """Format raw model output into API response."""
    pred_idx = max(range(len(probabilities)), key=lambda i: probabilities[i])
    sentiment = LABEL_NAMES[pred_idx]
    confidence = probabilities[pred_idx]

    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "probabilities": {
            name: round(prob, 4)
            for name, prob in zip(LABEL_NAMES, probabilities)
        },
        "entities": entities,
        "market_signal": MARKET_SIGNALS[sentiment],
        "model_used": model_name,
        "inference_time_ms": round(inference_time_ms, 2),
    }
