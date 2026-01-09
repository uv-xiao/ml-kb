#!/usr/bin/env python3
"""Simple kernel profiling for FlashInfer attention."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

import torch
import json
from flashinfer import single_prefill_with_kv_cache, single_decode_with_kv_cache

OUTPUT_DIR = "./profiling_artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("FlashInfer Kernel Profiling")
print("=" * 60)
print(f"GPU: {torch.cuda.get_device_name(0)}")

results = {}

# Test 1: Prefill kernel timing at different sequence lengths
print("\n1. Prefill Kernel Scaling")
print("-" * 40)

num_heads = 16
head_dim = 64
prefill_results = []

for seq_len in [64, 128, 256, 512, 1024]:
    q = torch.randn(1, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(1, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(1, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)

    # Warmup
    for _ in range(5):
        _ = single_prefill_with_kv_cache(q, k, v)

    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()
    for _ in range(50):
        _ = single_prefill_with_kv_cache(q, k, v)
    end.record()
    torch.cuda.synchronize()

    avg_time = start.elapsed_time(end) / 50
    per_token = avg_time * 1000 / seq_len  # us per token

    prefill_results.append({
        "seq_len": seq_len,
        "time_ms": avg_time,
        "us_per_token": per_token
    })
    print(f"  seq_len={seq_len:4d}: {avg_time:.3f} ms ({per_token:.1f} us/token)")

results["prefill_scaling"] = prefill_results

# Test 2: Decode kernel at different KV lengths
print("\n2. Decode Kernel Scaling")
print("-" * 40)

decode_results = []

for kv_len in [64, 128, 256, 512, 1024, 2048]:
    # For single_decode_with_kv_cache: q is 2D [num_heads, head_dim], k/v are 3D [kv_len, num_heads, head_dim]
    q = torch.randn(num_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(kv_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(kv_len, num_heads, head_dim, device="cuda", dtype=torch.float16)

    # Warmup
    for _ in range(5):
        _ = single_decode_with_kv_cache(q, k, v)

    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()
    for _ in range(100):
        _ = single_decode_with_kv_cache(q, k, v)
    end.record()
    torch.cuda.synchronize()

    avg_time = start.elapsed_time(end) / 100

    decode_results.append({
        "kv_len": kv_len,
        "time_ms": avg_time
    })
    print(f"  kv_len={kv_len:4d}: {avg_time:.4f} ms")

results["decode_scaling"] = decode_results

# Test 3: Prefill vs Decode comparison
print("\n3. Prefill vs Decode Comparison")
print("-" * 40)

# Same scenario: 256 query tokens for prefill, 1 for decode, KV=512
kv_len = 512

q_prefill = torch.randn(1, 256, num_heads, head_dim, device="cuda", dtype=torch.float16)
k_prefill = torch.randn(1, kv_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
v_prefill = torch.randn(1, kv_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
# Decode uses different tensor layout
q_decode = torch.randn(num_heads, head_dim, device="cuda", dtype=torch.float16)
k_decode = torch.randn(kv_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
v_decode = torch.randn(kv_len, num_heads, head_dim, device="cuda", dtype=torch.float16)

# Prefill timing
for _ in range(5):
    _ = single_prefill_with_kv_cache(q_prefill, k_prefill, v_prefill)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
start.record()
for _ in range(50):
    _ = single_prefill_with_kv_cache(q_prefill, k_prefill, v_prefill)
end.record()
torch.cuda.synchronize()
prefill_time = start.elapsed_time(end) / 50

# Decode timing
for _ in range(5):
    _ = single_decode_with_kv_cache(q_decode, k_decode, v_decode)

torch.cuda.synchronize()
start.record()
for _ in range(100):
    _ = single_decode_with_kv_cache(q_decode, k_decode, v_decode)
end.record()
torch.cuda.synchronize()
decode_time = start.elapsed_time(end) / 100

print(f"  Prefill (256 Q, 512 KV): {prefill_time:.3f} ms")
print(f"  Decode  (1 Q, 512 KV):   {decode_time:.4f} ms")
print(f"  Ratio: {prefill_time/decode_time:.1f}x")
print(f"  Per Q-token prefill: {prefill_time*1000/256:.1f} us")
print(f"  Per Q-token decode:  {decode_time*1000:.1f} us")

results["prefill_vs_decode"] = {
    "prefill_256q_512kv_ms": prefill_time,
    "decode_1q_512kv_ms": decode_time,
    "ratio": prefill_time / decode_time
}

# Test 4: Batch size effect (using prefill as proxy since decode doesn't batch easily)
print("\n4. Batch Size Effect on Prefill")
print("-" * 40)

batch_results = []
kv_len = 512
q_len = 1  # Single token per request to simulate decode-like workload

for batch_size in [1, 2, 4, 8, 16, 32]:
    # Use prefill with 1 Q token per batch item to simulate batched decode
    q = torch.randn(batch_size, q_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch_size, kv_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch_size, kv_len, num_heads, head_dim, device="cuda", dtype=torch.float16)

    # Warmup
    for _ in range(5):
        _ = single_prefill_with_kv_cache(q, k, v)

    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()
    for _ in range(50):
        _ = single_prefill_with_kv_cache(q, k, v)
    end.record()
    torch.cuda.synchronize()

    avg_time = start.elapsed_time(end) / 50
    per_request = avg_time / batch_size

    batch_results.append({
        "batch_size": batch_size,
        "time_ms": avg_time,
        "per_request_ms": per_request
    })
    print(f"  batch={batch_size:2d}: {avg_time:.3f} ms total, {per_request:.4f} ms/req")

results["batch_scaling"] = batch_results

# Save results
with open(f"{OUTPUT_DIR}/kernel_analysis.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print(f"Results saved to {OUTPUT_DIR}/kernel_analysis.json")
print("=" * 60)

# Generate summary for report
print("\n5. Summary for Report")
print("-" * 40)
print("""
┌──────────────────────────────────────────────────────────────────┐
│                KERNEL PERFORMANCE SUMMARY                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PREFILL KERNEL (SinglePrefillWithKVCacheKernel):               │
│  • Compute-bound: O(n² × d) attention computation                │
│  • Scaling: Subquadratic due to FlashAttention tiling           │
│  • Typical: {:.1f} us/token at seq_len=256                       │
│                                                                  │
│  DECODE KERNEL (SingleDecodeWithKVCacheKernel):                 │
│  • Memory-bound: O(n × d) KV cache read                          │
│  • Scaling: Linear with KV length                                │
│  • Typical: {:.1f} ms at kv_len=512                              │
│                                                                  │
│  BATCHING EFFICIENCY:                                            │
│  • Single decode: {:.4f} ms                                      │
│  • Batch=32 decode: {:.4f} ms/request ({:.1f}x speedup)          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
""".format(
    prefill_results[2]["us_per_token"],  # seq_len=256
    decode_results[3]["time_ms"],  # kv_len=512
    batch_results[0]["per_request_ms"],  # batch=1
    batch_results[5]["per_request_ms"],  # batch=32
    batch_results[0]["per_request_ms"] / batch_results[5]["per_request_ms"]
))
