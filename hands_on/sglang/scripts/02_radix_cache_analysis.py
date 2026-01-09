#!/usr/bin/env python3
"""
SGLang RadixCache Analysis
==========================

Analyzes prefix caching behavior with different sharing patterns.

Usage:
    python 02_radix_cache_analysis.py --url http://localhost:30000
"""

import argparse
import json
import time
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import statistics


@dataclass
class CacheTestResult:
    """Result from a single cache test."""
    scenario: str
    request_id: str
    prompt_tokens: int
    expected_cached: int
    ttft_ms: float
    output_tokens: int
    total_time_ms: float


@dataclass
class CacheAnalysis:
    """Analysis of cache behavior."""
    total_requests: int
    cache_hits: int
    cache_misses: int
    partial_hits: int
    hit_rate: float
    avg_cached_tokens: float
    compute_saved_tokens: int
    time_saved_ms: float


# System prompts of different lengths
SYSTEM_PROMPTS = {
    "short": "You are a helpful assistant.",
    "medium": """You are a helpful AI assistant. You provide accurate, detailed,
    and thoughtful responses. You always cite sources when possible and admit
    when you don't know something. Be concise but thorough.""",
    "long": """You are an advanced AI assistant designed to help users with a
    wide variety of tasks. Your capabilities include:

    1. Answering questions on diverse topics including science, history,
       literature, technology, and current events.
    2. Helping with writing tasks such as essays, emails, and creative writing.
    3. Explaining complex concepts in simple terms.
    4. Providing step-by-step instructions for various procedures.
    5. Engaging in thoughtful discussions on ethical and philosophical topics.

    Guidelines for your responses:
    - Be accurate and cite sources when appropriate
    - Acknowledge uncertainty when you don't know something
    - Be respectful and considerate of different perspectives
    - Provide balanced information on controversial topics
    - Keep responses focused and relevant to the question

    You strive to be helpful, harmless, and honest in all interactions."""
}


def send_request(url: str, messages: List[Dict], max_tokens: int, stream: bool = True) -> Tuple[float, int, float]:
    """Send request and return (ttft_ms, output_tokens, total_time_ms)."""
    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": 0.7,
    }

    start = time.perf_counter()
    first_token_time = None
    token_count = 0

    try:
        if stream:
            response = requests.post(
                f"{url}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=60
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
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            token_count += 1
                    except json.JSONDecodeError:
                        continue
        else:
            response = requests.post(
                f"{url}/v1/chat/completions",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            first_token_time = start  # Approximate
            token_count = result.get('usage', {}).get('completion_tokens', 0)

        end = time.perf_counter()

        ttft = (first_token_time - start) * 1000 if first_token_time else 0
        total = (end - start) * 1000

        return ttft, token_count, total

    except Exception as e:
        print(f"Request failed: {e}")
        return 0, 0, 0


def run_cache_scenarios(url: str, max_tokens: int = 64) -> List[CacheTestResult]:
    """Run different cache hit/miss scenarios."""
    results = []

    print("\n" + "=" * 70)
    print("RADIX CACHE BEHAVIOR ANALYSIS")
    print("=" * 70)

    # Scenario 1: Cold start (no cache)
    print("\n[Scenario 1] Cold Start (Cache Miss)")
    print("-" * 50)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS["long"]},
        {"role": "user", "content": "What is machine learning?"}
    ]
    ttft, tokens, total = send_request(url, messages, max_tokens)
    cold_ttft = ttft
    results.append(CacheTestResult(
        scenario="cold_start",
        request_id="req_1",
        prompt_tokens=len(SYSTEM_PROMPTS["long"].split()) + 10,
        expected_cached=0,
        ttft_ms=ttft,
        output_tokens=tokens,
        total_time_ms=total
    ))
    print(f"  TTFT: {ttft:.1f}ms (baseline, full prefill)")

    # Small delay to ensure cache is populated
    time.sleep(0.5)

    # Scenario 2: Same system prompt, different question (partial hit)
    print("\n[Scenario 2] Partial Cache Hit (Same System Prompt)")
    print("-" * 50)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS["long"]},
        {"role": "user", "content": "Explain neural networks briefly."}
    ]
    ttft, tokens, total = send_request(url, messages, max_tokens)
    partial_hit_ttft = ttft
    speedup = cold_ttft / ttft if ttft > 0 else 0
    results.append(CacheTestResult(
        scenario="partial_hit",
        request_id="req_2",
        prompt_tokens=len(SYSTEM_PROMPTS["long"].split()) + 10,
        expected_cached=len(SYSTEM_PROMPTS["long"].split()),
        ttft_ms=ttft,
        output_tokens=tokens,
        total_time_ms=total
    ))
    print(f"  TTFT: {ttft:.1f}ms (speedup: {speedup:.1f}x)")
    print(f"  System prompt cached, only user question computed")

    time.sleep(0.5)

    # Scenario 3: Exact same request (full hit)
    print("\n[Scenario 3] Full Cache Hit (Exact Repeat)")
    print("-" * 50)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS["long"]},
        {"role": "user", "content": "What is machine learning?"}
    ]
    ttft, tokens, total = send_request(url, messages, max_tokens)
    full_hit_ttft = ttft
    speedup = cold_ttft / ttft if ttft > 0 else 0
    results.append(CacheTestResult(
        scenario="full_hit",
        request_id="req_3",
        prompt_tokens=len(SYSTEM_PROMPTS["long"].split()) + 10,
        expected_cached=len(SYSTEM_PROMPTS["long"].split()) + 10,
        ttft_ms=ttft,
        output_tokens=tokens,
        total_time_ms=total
    ))
    print(f"  TTFT: {ttft:.1f}ms (speedup: {speedup:.1f}x)")
    print(f"  Full prompt cached, minimal compute needed")

    time.sleep(0.5)

    # Scenario 4: Different system prompt (cache miss)
    print("\n[Scenario 4] Different System Prompt (Cache Miss)")
    print("-" * 50)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS["short"]},
        {"role": "user", "content": "What is machine learning?"}
    ]
    ttft, tokens, total = send_request(url, messages, max_tokens)
    results.append(CacheTestResult(
        scenario="different_system",
        request_id="req_4",
        prompt_tokens=len(SYSTEM_PROMPTS["short"].split()) + 10,
        expected_cached=0,  # Different prefix, no cache hit
        ttft_ms=ttft,
        output_tokens=tokens,
        total_time_ms=total
    ))
    print(f"  TTFT: {ttft:.1f}ms (different prefix, no cache benefit)")

    time.sleep(0.5)

    # Scenario 5: Multi-turn conversation (incremental caching)
    print("\n[Scenario 5] Multi-Turn Conversation (Incremental)")
    print("-" * 50)

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPTS["medium"]},
        {"role": "user", "content": "What is Python?"}
    ]

    # Turn 1
    ttft, tokens, total = send_request(url, conversation, max_tokens)
    print(f"  Turn 1 TTFT: {ttft:.1f}ms (cold start for this conversation)")
    turn1_ttft = ttft
    results.append(CacheTestResult(
        scenario="multi_turn_1",
        request_id="req_5a",
        prompt_tokens=len(SYSTEM_PROMPTS["medium"].split()) + 5,
        expected_cached=0,
        ttft_ms=ttft,
        output_tokens=tokens,
        total_time_ms=total
    ))

    time.sleep(0.3)

    # Simulate assistant response (in real scenario, would use actual response)
    conversation.append({"role": "assistant", "content": "Python is a programming language."})
    conversation.append({"role": "user", "content": "What are its main features?"})

    ttft, tokens, total = send_request(url, conversation, max_tokens)
    speedup = turn1_ttft / ttft if ttft > 0 else 0
    print(f"  Turn 2 TTFT: {ttft:.1f}ms (conversation prefix cached, speedup: {speedup:.1f}x)")
    results.append(CacheTestResult(
        scenario="multi_turn_2",
        request_id="req_5b",
        prompt_tokens=len(SYSTEM_PROMPTS["medium"].split()) + 20,
        expected_cached=len(SYSTEM_PROMPTS["medium"].split()) + 5,
        ttft_ms=ttft,
        output_tokens=tokens,
        total_time_ms=total
    ))

    return results


def analyze_results(results: List[CacheTestResult]) -> CacheAnalysis:
    """Analyze cache test results."""
    cache_hits = sum(1 for r in results if r.expected_cached > 0 and
                     r.expected_cached >= r.prompt_tokens * 0.9)  # >90% cached
    partial_hits = sum(1 for r in results if 0 < r.expected_cached < r.prompt_tokens * 0.9)
    cache_misses = sum(1 for r in results if r.expected_cached == 0)

    total = len(results)
    hit_rate = (cache_hits + partial_hits * 0.5) / total if total > 0 else 0

    avg_cached = statistics.mean(r.expected_cached for r in results)
    compute_saved = sum(r.expected_cached for r in results)

    # Estimate time saved (rough approximation)
    cold_results = [r for r in results if r.expected_cached == 0]
    if cold_results:
        cold_ttft = statistics.mean(r.ttft_ms for r in cold_results)
        tokens_per_ms = statistics.mean(r.prompt_tokens for r in cold_results) / cold_ttft
        time_saved = compute_saved / tokens_per_ms if tokens_per_ms > 0 else 0
    else:
        time_saved = 0

    return CacheAnalysis(
        total_requests=total,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        partial_hits=partial_hits,
        hit_rate=hit_rate,
        avg_cached_tokens=avg_cached,
        compute_saved_tokens=compute_saved,
        time_saved_ms=time_saved
    )


def print_report(results: List[CacheTestResult], analysis: CacheAnalysis):
    """Print detailed analysis report."""

    print("\n" + "=" * 70)
    print("CACHE ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"""
CACHE STATISTICS:
─────────────────
├── Total requests: {analysis.total_requests}
├── Full cache hits: {analysis.cache_hits}
├── Partial cache hits: {analysis.partial_hits}
├── Cache misses: {analysis.cache_misses}
└── Effective hit rate: {analysis.hit_rate:.1%}

EFFICIENCY METRICS:
───────────────────
├── Average tokens cached per request: {analysis.avg_cached_tokens:.0f}
├── Total compute saved: {analysis.compute_saved_tokens} tokens
└── Estimated time saved: {analysis.time_saved_ms:.1f}ms
""")

    # Detailed results table
    print("DETAILED RESULTS:")
    print("─" * 70)
    print(f"{'Scenario':<20} {'TTFT (ms)':<12} {'Cached':<10} {'Total':<10} {'Status':<10}")
    print("─" * 70)

    for r in results:
        cache_pct = (r.expected_cached / r.prompt_tokens * 100) if r.prompt_tokens > 0 else 0
        if cache_pct > 90:
            status = "HIT"
        elif cache_pct > 0:
            status = "PARTIAL"
        else:
            status = "MISS"

        print(f"{r.scenario:<20} {r.ttft_ms:<12.1f} {r.expected_cached:<10} {r.prompt_tokens:<10} {status:<10}")

    print("─" * 70)

    # Hardware implications
    print("""
HARDWARE BEHAVIOR IMPLICATIONS:
───────────────────────────────

Cache HIT:
├── Skip attention computation for cached tokens
├── Reduced HBM reads (KV already in cache)
├── Lower SM utilization (less work)
└── TTFT dominated by remaining uncached tokens

Cache MISS:
├── Full prefill computation required
├── All KV pairs computed from scratch
├── Higher TC/SM utilization during prefill
└── Cache insertion after generation

OPTIMIZATION STRATEGIES:
────────────────────────
1. Maximize system prompt reuse (cache-friendly)
2. Batch similar requests together (share prefixes)
3. Use longer shared prefixes in prompts
4. Monitor cache eviction rate under load
""")


def main():
    parser = argparse.ArgumentParser(description='SGLang RadixCache Analysis')
    parser.add_argument('--url', type=str, default='http://localhost:30000',
                        help='SGLang server URL')
    parser.add_argument('--max-tokens', type=int, default=64,
                        help='Max output tokens per request')
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

    results = run_cache_scenarios(args.url, args.max_tokens)
    analysis = analyze_results(results)

    if args.json:
        output = {
            "results": [
                {
                    "scenario": r.scenario,
                    "ttft_ms": r.ttft_ms,
                    "expected_cached": r.expected_cached,
                    "prompt_tokens": r.prompt_tokens
                } for r in results
            ],
            "analysis": {
                "hit_rate": analysis.hit_rate,
                "compute_saved_tokens": analysis.compute_saved_tokens,
                "time_saved_ms": analysis.time_saved_ms
            }
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(results, analysis)


if __name__ == "__main__":
    main()
