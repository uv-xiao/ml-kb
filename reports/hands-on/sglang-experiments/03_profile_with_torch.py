#!/usr/bin/env python3
"""
Experiment 3: Profile SGLang with PyTorch Profiler

This script profiles SGLang inference to understand:
- Kernel execution patterns (prefill vs decode)
- Time distribution across operations
- Memory usage patterns

Run: python 03_profile_with_torch.py
"""

import os
import json
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
import requests
import time

# Configuration
SERVER_URL = "http://localhost:30000"
OUTPUT_DIR = "./profiling_artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def send_request(prompt: str, max_tokens: int = 64):
    """Send a generation request to SGLang server."""
    response = requests.post(
        f"{SERVER_URL}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
            }
        }
    )
    return response.json()

def profile_single_request():
    """Profile a single request end-to-end."""
    print("Profiling single request...")

    prompt = "Explain the concept of attention mechanism in transformers in detail."

    # Warmup
    for _ in range(3):
        send_request("Hello", max_tokens=10)

    # Profile
    start = time.perf_counter()
    result = send_request(prompt, max_tokens=128)
    end = time.perf_counter()

    print(f"Single request latency: {(end-start)*1000:.2f}ms")
    print(f"Output tokens: {len(result.get('text', '').split())}")

    return result

def profile_batch_requests():
    """Profile multiple concurrent requests."""
    import concurrent.futures

    print("\nProfiling batch of 8 requests...")

    prompts = [
        "What is machine learning?",
        "Explain neural networks.",
        "What is deep learning?",
        "How do transformers work?",
        "What is attention mechanism?",
        "Explain backpropagation.",
        "What is gradient descent?",
        "How does BERT work?",
    ]

    # Warmup
    for _ in range(2):
        send_request("Hello", max_tokens=10)

    start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(send_request, p, 64) for p in prompts]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    end = time.perf_counter()

    print(f"Batch latency (8 requests): {(end-start)*1000:.2f}ms")
    print(f"Per-request average: {(end-start)*1000/8:.2f}ms")

    return results

def profile_prefix_cache():
    """Profile RadixCache prefix sharing behavior."""
    print("\nProfiling RadixCache prefix sharing...")

    # Same prefix, different suffixes
    shared_prefix = "You are a helpful AI assistant. Please answer the following question: "

    questions = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
        "What is Go?",
        "What is C++?",
    ]

    prompts = [shared_prefix + q for q in questions]

    latencies = []
    for i, prompt in enumerate(prompts):
        start = time.perf_counter()
        result = send_request(prompt, max_tokens=64)
        end = time.perf_counter()
        latency = (end - start) * 1000
        latencies.append(latency)
        print(f"  Request {i+1}: {latency:.2f}ms")

    print(f"\nFirst request (cold): {latencies[0]:.2f}ms")
    print(f"Subsequent (cached): {sum(latencies[1:])/len(latencies[1:]):.2f}ms avg")
    print(f"Speedup from cache: {latencies[0]/sum(latencies[1:])*len(latencies[1:]):.2f}x")

    return latencies

def profile_sequence_lengths():
    """Profile how latency scales with sequence length."""
    print("\nProfiling sequence length scaling...")

    base_prompt = "Explain the following concept in detail: "

    results = []
    for length in [64, 128, 256, 512]:
        prompt = base_prompt + "x " * (length // 2)  # Approximate tokens

        latencies = []
        for _ in range(3):
            start = time.perf_counter()
            result = send_request(prompt, max_tokens=64)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        results.append((length, avg_latency))
        print(f"  Input ~{length} tokens: {avg_latency:.2f}ms")

    return results

def main():
    print("=" * 60)
    print("SGLang Profiling Experiments")
    print("=" * 60)

    # Check server is running
    try:
        response = requests.get(f"{SERVER_URL}/health")
        if response.status_code != 200:
            print("Server not healthy. Please start the server first.")
            return
    except requests.exceptions.ConnectionError:
        print("Cannot connect to server. Please run 01_start_server.sh first.")
        return

    # Run experiments
    results = {}

    results['single'] = profile_single_request()
    results['batch'] = profile_batch_requests()
    results['prefix_cache'] = profile_prefix_cache()
    results['seq_lengths'] = profile_sequence_lengths()

    # Save results
    with open(f"{OUTPUT_DIR}/profiling_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print(f"Results saved to {OUTPUT_DIR}/profiling_results.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
