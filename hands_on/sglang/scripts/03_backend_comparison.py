#!/usr/bin/env python3
"""
SGLang Attention Backend Comparison
====================================

Compares FlashInfer vs Triton attention backends.

Usage:
    # Start server with FlashInfer backend
    python -m sglang.launch_server --attention-backend flashinfer --port 30000

    # Run comparison
    python 03_backend_comparison.py --url http://localhost:30000

Note: Requires restarting server with different backends to compare.
"""

import argparse
import json
import time
import requests
import subprocess
import signal
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import statistics


@dataclass
class BackendResult:
    """Results for a specific backend configuration."""
    backend: str
    batch_size: int
    seq_len: int
    ttft_ms: float
    itl_avg_ms: float
    throughput_tokens_per_sec: float
    total_time_ms: float


def generate_prompt(num_tokens: int) -> str:
    """Generate a prompt with approximately the target number of tokens."""
    base = "Please provide a detailed explanation of "
    topics = [
        "machine learning algorithms",
        "neural network architectures",
        "natural language processing",
        "computer vision techniques",
        "reinforcement learning",
    ]
    prompt = base + ", ".join(topics * (num_tokens // 20 + 1))
    return prompt[:num_tokens * 4]


def run_benchmark(
    url: str,
    batch_size: int,
    seq_len: int,
    output_len: int = 64
) -> Tuple[float, float, float, float]:
    """Run benchmark and return (ttft, itl_avg, throughput, total_time)."""

    prompt = generate_prompt(seq_len)
    all_ttfts = []
    all_itls = []
    total_tokens = 0

    start_time = time.perf_counter()

    for i in range(batch_size):
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": output_len,
            "stream": True,
            "temperature": 0.7,
        }

        req_start = time.perf_counter()
        first_token_time = None
        last_token_time = req_start
        token_count = 0

        try:
            response = requests.post(
                f"{url}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=120
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        if chunk.get('choices', [{}])[0].get('delta', {}).get('content'):
                            now = time.perf_counter()
                            if first_token_time is None:
                                first_token_time = now
                                all_ttfts.append((now - req_start) * 1000)
                            else:
                                all_itls.append((now - last_token_time) * 1000)
                            last_token_time = now
                            token_count += 1
                    except json.JSONDecodeError:
                        continue

            total_tokens += token_count

        except Exception as e:
            print(f"  Request {i+1} failed: {e}")

    total_time = time.perf_counter() - start_time

    ttft_avg = statistics.mean(all_ttfts) if all_ttfts else 0
    itl_avg = statistics.mean(all_itls) if all_itls else 0
    throughput = total_tokens / total_time if total_time > 0 else 0

    return ttft_avg, itl_avg, throughput, total_time * 1000


def run_backend_benchmark(url: str, backend_name: str) -> List[BackendResult]:
    """Run full benchmark suite for a backend."""

    results = []

    # Test configurations: (batch_size, seq_len)
    configs = [
        (1, 512),
        (1, 2048),
        (8, 512),
        (8, 2048),
        (32, 512),
    ]

    print(f"\n  Testing {backend_name} backend...")

    for batch_size, seq_len in configs:
        print(f"    Batch={batch_size}, SeqLen={seq_len}...", end=' ')

        ttft, itl, throughput, total = run_benchmark(url, batch_size, seq_len)

        results.append(BackendResult(
            backend=backend_name,
            batch_size=batch_size,
            seq_len=seq_len,
            ttft_ms=ttft,
            itl_avg_ms=itl,
            throughput_tokens_per_sec=throughput,
            total_time_ms=total
        ))

        print(f"TTFT={ttft:.1f}ms, ITL={itl:.1f}ms, {throughput:.0f} tok/s")

    return results


def print_comparison_report(results: List[BackendResult]):
    """Print comparison report."""

    print("\n" + "=" * 76)
    print("ATTENTION BACKEND COMPARISON")
    print("=" * 76)

    # Group by config
    configs = set((r.batch_size, r.seq_len) for r in results)

    for batch_size, seq_len in sorted(configs):
        config_results = [r for r in results if r.batch_size == batch_size and r.seq_len == seq_len]

        print(f"\nBatch={batch_size}, SeqLen={seq_len}")
        print("-" * 70)
        print(f"{'Backend':<15} {'TTFT (ms)':<12} {'ITL (ms)':<12} {'Throughput':<15} {'Total (ms)':<12}")
        print("-" * 70)

        for r in config_results:
            print(f"{r.backend:<15} {r.ttft_ms:<12.1f} {r.itl_avg_ms:<12.1f} {r.throughput_tokens_per_sec:<15.0f} {r.total_time_ms:<12.1f}")

        # Calculate winner
        if len(config_results) == 2:
            r1, r2 = config_results
            ttft_diff = (r2.ttft_ms - r1.ttft_ms) / r1.ttft_ms * 100 if r1.ttft_ms > 0 else 0
            winner = r1.backend if r1.ttft_ms < r2.ttft_ms else r2.backend
            print(f"  Winner: {winner} ({abs(ttft_diff):.1f}% faster TTFT)")

    print("\n" + "=" * 76)
    print("ANALYSIS")
    print("=" * 76)

    print("""
EXPECTED OBSERVATIONS:
──────────────────────

FlashInfer Backend:
├── Generally 10-20% faster than Triton
├── Better tensor core utilization
├── Optimized for paged KV cache
├── More complex implementation (harder to customize)
└── Supports more advanced features (MLA, etc.)

Triton Backend:
├── Pure Triton implementation
├── Easier to modify and experiment
├── Good baseline performance
├── Better for custom attention patterns
└── Deterministic mode support

HARDWARE DIFFERENCES:
─────────────────────

                        FlashInfer          Triton
────────────────────────────────────────────────────────────
TC Utilization         Higher (35-40%)     Lower (28-35%)
HBM Efficiency         Better tiling       Standard tiling
Warp Scheduling        Optimized           Standard
SMEM Usage             ~96KB              ~64-80KB
────────────────────────────────────────────────────────────

RECOMMENDATIONS:
────────────────
• Use FlashInfer for production (default)
• Use Triton for experimentation/debugging
• Use Triton for custom attention patterns
• FlashInfer advantage grows with batch size
""")


def main():
    parser = argparse.ArgumentParser(description='SGLang Backend Comparison')
    parser.add_argument('--url', type=str, default='http://localhost:30000',
                        help='SGLang server URL')
    parser.add_argument('--backend', type=str, default='auto',
                        choices=['auto', 'flashinfer', 'triton'],
                        help='Backend to test (auto detects current)')
    parser.add_argument('--json', action='store_true',
                        help='Output as JSON')
    args = parser.parse_args()

    # Check server
    try:
        resp = requests.get(f"{args.url}/health", timeout=5)
        if resp.status_code != 200:
            print(f"Server health check failed")
            return
    except Exception as e:
        print(f"Cannot connect to server: {e}")
        return

    print("\n" + "=" * 76)
    print("SGLANG ATTENTION BACKEND COMPARISON")
    print("=" * 76)

    # For a full comparison, user needs to run with different backends
    # This script tests the currently running server
    backend_name = args.backend if args.backend != 'auto' else 'current'

    print(f"\nTesting backend: {backend_name}")
    print("\nNote: For full comparison, restart server with different --attention-backend")
    print("      and run this script again. Results can be compared manually.")

    results = run_backend_benchmark(args.url, backend_name)

    if args.json:
        output = [
            {
                "backend": r.backend,
                "batch_size": r.batch_size,
                "seq_len": r.seq_len,
                "ttft_ms": r.ttft_ms,
                "itl_avg_ms": r.itl_avg_ms,
                "throughput": r.throughput_tokens_per_sec
            } for r in results
        ]
        print(json.dumps(output, indent=2))
    else:
        # Print single backend results with analysis template
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\n{'Config':<20} {'TTFT (ms)':<12} {'ITL (ms)':<12} {'Throughput':<15}")
        print("-" * 70)
        for r in results:
            config = f"B={r.batch_size}, S={r.seq_len}"
            print(f"{config:<20} {r.ttft_ms:<12.1f} {r.itl_avg_ms:<12.1f} {r.throughput_tokens_per_sec:<15.0f}")

        print_comparison_report(results)


if __name__ == "__main__":
    main()
