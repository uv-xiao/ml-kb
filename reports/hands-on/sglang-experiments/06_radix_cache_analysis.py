#!/usr/bin/env python3
"""
Experiment 6: RadixCache Prefix Sharing Analysis

This script analyzes RadixCache behavior by:
1. Measuring cache hit rates with shared prefixes
2. Testing eviction behavior under memory pressure
3. Profiling different scheduling policies (LPM vs FCFS)

Run: python 06_radix_cache_analysis.py
"""

import os
import json
import time
import requests
import statistics
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

SERVER_URL = "http://localhost:30000"
OUTPUT_DIR = "./profiling_artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def send_request(prompt: str, max_tokens: int = 64) -> Tuple[float, dict]:
    """Send request and measure latency."""
    start = time.perf_counter()
    response = requests.post(
        f"{SERVER_URL}/generate",
        json={
            "text": prompt,
            "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0.7}
        }
    )
    end = time.perf_counter()
    return (end - start) * 1000, response.json()

def warmup():
    """Warmup the server."""
    for _ in range(5):
        send_request("Hello", max_tokens=10)

def experiment_1_prefix_sharing():
    """
    Experiment 1: Measure cache hit benefit with shared prefix.

    Hypothesis: First request is slow (full prefill), subsequent requests
    with same prefix should be faster due to RadixCache hits.
    """
    print("\n" + "="*60)
    print("Experiment 1: Prefix Sharing Benefit")
    print("="*60)

    shared_prefix = """You are an expert AI assistant specialized in programming.
Your task is to help users understand and write code effectively.
Please provide clear, accurate, and helpful responses.

Question: """

    questions = [
        "What is a function in Python?",
        "How do I create a class in Python?",
        "What is inheritance in object-oriented programming?",
        "How do decorators work in Python?",
        "What is the difference between list and tuple?",
        "How do I handle exceptions in Python?",
        "What is a generator in Python?",
        "How do async/await work in Python?",
    ]

    # Clear cache by sending unrelated requests
    print("Clearing cache with unrelated requests...")
    for i in range(10):
        send_request(f"Random text {i} to clear cache", max_tokens=10)

    print("\nMeasuring latencies with shared prefix...")
    latencies = []
    for i, q in enumerate(questions):
        prompt = shared_prefix + q
        latency, _ = send_request(prompt, max_tokens=64)
        latencies.append(latency)
        cache_status = "COLD" if i == 0 else "WARM"
        print(f"  Request {i+1} [{cache_status}]: {latency:.2f}ms")

    results = {
        "first_request_cold": latencies[0],
        "subsequent_avg": statistics.mean(latencies[1:]),
        "subsequent_std": statistics.stdev(latencies[1:]) if len(latencies) > 2 else 0,
        "speedup_ratio": latencies[0] / statistics.mean(latencies[1:]),
        "all_latencies": latencies,
    }

    print(f"\nResults:")
    print(f"  First request (cold): {results['first_request_cold']:.2f}ms")
    print(f"  Subsequent avg (warm): {results['subsequent_avg']:.2f}ms")
    print(f"  Speedup from cache: {results['speedup_ratio']:.2f}x")

    return results

def experiment_2_prefix_length_impact():
    """
    Experiment 2: How does prefix length affect cache benefit?

    Test different prefix lengths to see the relationship between
    cached tokens and latency reduction.
    """
    print("\n" + "="*60)
    print("Experiment 2: Prefix Length Impact")
    print("="*60)

    base_text = "The quick brown fox jumps over the lazy dog. "
    suffix = "What is the meaning of this sentence?"

    results = []
    for prefix_repeat in [1, 4, 16, 64]:
        prefix = base_text * prefix_repeat
        prompt = prefix + suffix

        # Send first request (cold)
        latency_cold, _ = send_request(prompt, max_tokens=32)

        # Send second request (warm)
        latency_warm, _ = send_request(prompt, max_tokens=32)

        approx_tokens = len(prefix.split()) + len(suffix.split())
        speedup = latency_cold / latency_warm

        results.append({
            "prefix_repeats": prefix_repeat,
            "approx_tokens": approx_tokens,
            "cold_latency": latency_cold,
            "warm_latency": latency_warm,
            "speedup": speedup,
        })

        print(f"  Prefix {prefix_repeat}x (~{approx_tokens} tokens): "
              f"cold={latency_cold:.2f}ms, warm={latency_warm:.2f}ms, "
              f"speedup={speedup:.2f}x")

    return results

def experiment_3_concurrent_prefix_sharing():
    """
    Experiment 3: Concurrent requests with shared prefix.

    Test in-batch prefix caching: when multiple concurrent requests
    share a prefix, the scheduler should optimize.
    """
    print("\n" + "="*60)
    print("Experiment 3: Concurrent Prefix Sharing")
    print("="*60)

    shared_prefix = """System: You are a helpful AI assistant.
User: Please answer my question about programming.
Question: """

    questions = [f"What is concept number {i} in computer science?" for i in range(16)]

    # Sequential baseline
    print("Sequential baseline...")
    seq_latencies = []
    for q in questions[:8]:
        latency, _ = send_request(shared_prefix + q, max_tokens=32)
        seq_latencies.append(latency)

    seq_total = sum(seq_latencies)
    print(f"  Sequential total: {seq_total:.2f}ms")
    print(f"  Sequential avg: {statistics.mean(seq_latencies):.2f}ms")

    # Concurrent execution
    print("\nConcurrent execution...")
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(send_request, shared_prefix + q, 32)
            for q in questions[:8]
        ]
        conc_latencies = [f.result()[0] for f in as_completed(futures)]
    conc_total = (time.perf_counter() - start) * 1000

    print(f"  Concurrent total: {conc_total:.2f}ms")
    print(f"  Concurrent per-request: {statistics.mean(conc_latencies):.2f}ms")
    print(f"  Speedup vs sequential: {seq_total/conc_total:.2f}x")

    return {
        "sequential_total": seq_total,
        "sequential_avg": statistics.mean(seq_latencies),
        "concurrent_total": conc_total,
        "concurrent_avg": statistics.mean(conc_latencies),
        "batch_speedup": seq_total / conc_total,
    }

def experiment_4_cache_eviction():
    """
    Experiment 4: Cache eviction behavior.

    Send many unique requests to fill cache, then check if
    previously cached prefixes get evicted.
    """
    print("\n" + "="*60)
    print("Experiment 4: Cache Eviction Behavior")
    print("="*60)

    # Create a specific prefix we'll track
    tracked_prefix = "TRACKED_PREFIX_FOR_EVICTION_TEST: "
    tracked_suffix = "What is machine learning?"

    # First, cache the tracked prefix
    print("Caching tracked prefix...")
    latency_initial, _ = send_request(tracked_prefix + tracked_suffix, max_tokens=32)
    print(f"  Initial (cold): {latency_initial:.2f}ms")

    latency_cached, _ = send_request(tracked_prefix + tracked_suffix, max_tokens=32)
    print(f"  After cache (warm): {latency_cached:.2f}ms")

    # Send many unique requests to potentially evict the cache
    print("\nSending 50 unique requests to pressure cache...")
    for i in range(50):
        unique_prompt = f"Unique prompt number {i}: " + "x " * 100
        send_request(unique_prompt, max_tokens=16)
        if (i + 1) % 10 == 0:
            print(f"  Sent {i+1}/50 unique requests")

    # Check if tracked prefix is still cached
    print("\nChecking if tracked prefix is still cached...")
    latency_after, _ = send_request(tracked_prefix + tracked_suffix, max_tokens=32)
    print(f"  After pressure (should be warm if not evicted): {latency_after:.2f}ms")

    evicted = latency_after > latency_cached * 1.5
    print(f"\nConclusion: Cache {'was EVICTED' if evicted else 'survived (still warm)'}")

    return {
        "initial_cold": latency_initial,
        "initial_warm": latency_cached,
        "after_pressure": latency_after,
        "likely_evicted": evicted,
    }

def main():
    print("="*60)
    print("RadixCache Analysis Experiments")
    print("="*60)

    # Check server
    try:
        response = requests.get(f"{SERVER_URL}/health")
        if response.status_code != 200:
            print("Server not healthy. Please start the server first.")
            return
    except requests.exceptions.ConnectionError:
        print("Cannot connect to server. Please run 01_start_server.sh first.")
        return

    warmup()

    all_results = {}
    all_results["exp1_prefix_sharing"] = experiment_1_prefix_sharing()
    all_results["exp2_prefix_length"] = experiment_2_prefix_length_impact()
    all_results["exp3_concurrent"] = experiment_3_concurrent_prefix_sharing()
    all_results["exp4_eviction"] = experiment_4_cache_eviction()

    # Save results
    output_file = f"{OUTPUT_DIR}/radix_cache_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print(f"Results saved to {output_file}")
    print("="*60)

    # Summary
    print("\nSUMMARY")
    print("-"*40)
    print(f"Prefix sharing speedup: {all_results['exp1_prefix_sharing']['speedup_ratio']:.2f}x")
    print(f"Concurrent batch speedup: {all_results['exp3_concurrent']['batch_speedup']:.2f}x")
    print(f"Cache eviction under pressure: {'Yes' if all_results['exp4_eviction']['likely_evicted'] else 'No'}")

if __name__ == "__main__":
    main()
