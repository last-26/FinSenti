"""
API Latency Benchmark for FinSenti.

Tests single and batch prediction endpoints under various loads
and reports latency metrics (p50, p95, p99, throughput).

Usage:
    python benchmark.py                          # default: http://localhost:8000
    python benchmark.py --url http://localhost:8000 --rounds 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request

# Sample financial texts for benchmarking
SAMPLE_TEXTS = [
    "Tesla reported record Q4 deliveries beating analyst expectations by 12%",
    "Fed raised interest rates by 25 basis points as expected",
    "Apple announced a new stock buyback program worth $90B",
    "Oil prices remained steady amid OPEC uncertainty",
    "Revenue increased but margins declined sharply in Q3",
    "The company maintained its quarterly dividend at $0.82 per share",
    "Shares dropped 15% after disappointing forward guidance",
    "EPS $2.45 vs $2.30 expected, revenue $12.1B vs $11.8B consensus",
    "The board will meet on Tuesday to discuss Q2 results",
    "Goldman Sachs upgraded the stock to buy with a price target of $450",
    "Unemployment claims fell to their lowest level since March",
    "Supply chain disruptions continue to impact semiconductor stocks",
    "The merger is expected to close in Q1 2027 pending regulatory approval",
    "Bitcoin surged past $100K amid institutional buying pressure",
    "Retail sales data came in flat, missing expectations of 0.3% growth",
    "The CEO announced plans to cut 10% of the workforce globally",
]


def make_request(url: str, data: dict) -> tuple[dict, float]:
    """Make a POST request and return (response_json, elapsed_ms)."""
    payload = json.dumps(data).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=30)
    elapsed_ms = (time.perf_counter() - start) * 1000
    body = json.loads(resp.read().decode())
    return body, elapsed_ms


def percentile(data: list[float], p: float) -> float:
    """Calculate percentile from sorted data."""
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def benchmark_single(base_url: str, rounds: int, warmup: int) -> dict:
    """Benchmark single prediction endpoint."""
    url = f"{base_url}/api/v1/predict"
    latencies: list[float] = []

    # Warmup
    print(f"  Warming up ({warmup} requests)...")
    for i in range(warmup):
        text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
        make_request(url, {"text": text})

    # Benchmark
    print(f"  Benchmarking ({rounds} requests)...")
    for i in range(rounds):
        text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
        _, elapsed = make_request(url, {"text": text})
        latencies.append(elapsed)

    return {
        "endpoint": "POST /api/v1/predict",
        "requests": rounds,
        "mean_ms": round(statistics.mean(latencies), 2),
        "median_ms": round(statistics.median(latencies), 2),
        "p50_ms": round(percentile(latencies, 50), 2),
        "p95_ms": round(percentile(latencies, 95), 2),
        "p99_ms": round(percentile(latencies, 99), 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "stdev_ms": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0,
        "throughput_rps": round(rounds / (sum(latencies) / 1000), 2),
    }


def benchmark_batch(base_url: str, batch_sizes: list[int], rounds: int, warmup: int) -> list[dict]:
    """Benchmark batch prediction endpoint with various batch sizes."""
    url = f"{base_url}/api/v1/batch"
    results = []

    for batch_size in batch_sizes:
        texts = [SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)] for i in range(batch_size)]
        latencies: list[float] = []

        # Warmup
        print(f"  Warming up batch_size={batch_size} ({warmup} requests)...")
        for _ in range(warmup):
            make_request(url, {"texts": texts})

        # Benchmark
        print(f"  Benchmarking batch_size={batch_size} ({rounds} requests)...")
        for _ in range(rounds):
            _, elapsed = make_request(url, {"texts": texts})
            latencies.append(elapsed)

        results.append({
            "endpoint": f"POST /api/v1/batch (size={batch_size})",
            "batch_size": batch_size,
            "requests": rounds,
            "mean_ms": round(statistics.mean(latencies), 2),
            "median_ms": round(statistics.median(latencies), 2),
            "p50_ms": round(percentile(latencies, 50), 2),
            "p95_ms": round(percentile(latencies, 95), 2),
            "p99_ms": round(percentile(latencies, 99), 2),
            "min_ms": round(min(latencies), 2),
            "max_ms": round(max(latencies), 2),
            "stdev_ms": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0,
            "throughput_rps": round(rounds / (sum(latencies) / 1000), 2),
            "per_text_ms": round(statistics.mean(latencies) / batch_size, 2),
        })

    return results


def print_results(title: str, results: list[dict]) -> None:
    """Print benchmark results in a formatted table."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    header = f"{'Endpoint':<35} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'RPS':>8}"
    print(header)
    print("-" * 80)

    for r in results:
        row = (
            f"{r['endpoint']:<35} "
            f"{r['mean_ms']:>7.1f}ms "
            f"{r['p50_ms']:>7.1f}ms "
            f"{r['p95_ms']:>7.1f}ms "
            f"{r['p99_ms']:>7.1f}ms "
            f"{r['throughput_rps']:>7.1f}"
        )
        print(row)

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="FinSenti API Latency Benchmark")
    parser.add_argument(
        "--url", default="http://localhost:8000", help="Base API URL"
    )
    parser.add_argument(
        "--rounds", type=int, default=50, help="Number of benchmark rounds per test"
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Number of warmup requests"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16],
        help="Batch sizes to test",
    )
    args = parser.parse_args()

    # Check API is available
    try:
        resp = urllib.request.urlopen(f"{args.url}/api/v1/health", timeout=5)
        health = json.loads(resp.read().decode())
        print(f"API Status: {health['status']}")
        print(f"Model: {health.get('model_name', 'unknown')}")
        print(f"Device: {health.get('device', 'unknown')}")
    except Exception as e:
        print(f"Error: Cannot reach API at {args.url} - {e}")
        print("Make sure the backend is running: uvicorn app.main:app --port 8000")
        return

    print(f"\nBenchmark config: {args.rounds} rounds, {args.warmup} warmup")
    print(f"Target: {args.url}")

    # Single prediction benchmark
    print("\n[1/2] Single Prediction Benchmark")
    single_result = benchmark_single(args.url, args.rounds, args.warmup)
    print_results("Single Prediction Latency", [single_result])

    # Batch prediction benchmark
    print("[2/2] Batch Prediction Benchmark")
    batch_results = benchmark_batch(args.url, args.batch_sizes, args.rounds, args.warmup)
    print_results("Batch Prediction Latency", batch_results)

    # Summary
    print("=" * 80)
    print("  Summary")
    print("=" * 80)
    print(f"  Single prediction: {single_result['p50_ms']:.1f}ms (p50), "
          f"{single_result['p95_ms']:.1f}ms (p95)")
    if batch_results:
        largest = batch_results[-1]
        print(f"  Batch (size={largest['batch_size']}): "
              f"{largest['p50_ms']:.1f}ms total, "
              f"{largest['per_text_ms']:.1f}ms/text (p50)")
    print(f"  Throughput: {single_result['throughput_rps']:.1f} req/s (single)")
    print()


if __name__ == "__main__":
    main()
