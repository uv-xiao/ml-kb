#!/usr/bin/env python3
"""
Experiment 7: Nsys Kernel Profiling for SGLang

This script runs SGLang inference with CUDA profiling enabled to capture
kernel-level metrics including:
- Kernel execution times
- GPU utilization
- Memory transfers
- CPU-GPU timeline

Run with nsys:
    nsys profile -o sglang_kernels --trace=cuda,nvtx \
        python 07_nsys_kernel_profile.py

Or run standalone for quick kernel timing:
    python 07_nsys_kernel_profile.py
"""

import os
import sys
import time
import json

# Set environment before imports
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

import torch
import torch.cuda

# Check for NVTX
try:
    import nvtx
    HAS_NVTX = True
except ImportError:
    HAS_NVTX = False
    print("Warning: nvtx not available, markers will be skipped")

OUTPUT_DIR = "./profiling_artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def nvtx_range(name, color="blue"):
    """Context manager for NVTX markers."""
    if HAS_NVTX:
        return nvtx.annotate(name, color=color)
    else:
        from contextlib import nullcontext
        return nullcontext()


def profile_model_forward():
    """Profile a single forward pass with kernel timing."""
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs
    from transformers import AutoConfig

    print("=" * 60)
    print("SGLang Kernel Profiling")
    print("=" * 60)

    # Load model configuration
    model_path = "Qwen/Qwen3-0.6B"
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    print(f"\nModel: {model_path}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Num layers: {config.num_hidden_layers}")
    print(f"Num heads: {config.num_attention_heads}")
    print(f"Head dim: {config.hidden_size // config.num_attention_heads}")

    # For kernel timing, we'll use direct torch profiler
    print("\n" + "=" * 60)
    print("Running PyTorch Profiler for Kernel Analysis")
    print("=" * 60)

    # Simple attention benchmark to capture FlashInfer kernels
    from flashinfer import single_prefill_with_kv_cache

    # Setup tensors
    batch_size = 1
    seq_len = 256
    num_heads = 16
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim,
                    device="cuda", dtype=torch.float16)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim,
                    device="cuda", dtype=torch.float16)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim,
                    device="cuda", dtype=torch.float16)

    print(f"\nTensor shapes:")
    print(f"  Q/K/V: [{batch_size}, {seq_len}, {num_heads}, {head_dim}]")

    # Warmup
    print("\nWarmup (3 iterations)...")
    for _ in range(3):
        _ = single_prefill_with_kv_cache(q, k, v)
    torch.cuda.synchronize()

    # Profile with PyTorch profiler
    print("\nProfiling forward passes...")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with nvtx_range("attention_forward", "green"):
            for i in range(10):
                with nvtx_range(f"iter_{i}", "blue"):
                    output = single_prefill_with_kv_cache(q, k, v)
        torch.cuda.synchronize()

    # Print kernel summary
    print("\n" + "=" * 60)
    print("Kernel Summary (sorted by CUDA time)")
    print("=" * 60)

    # Get kernel statistics
    key_averages = prof.key_averages()

    # Print top kernels
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=15,
        top_level_events_only=False
    ))

    # Extract kernel data for report
    kernel_data = []
    for event in key_averages:
        cuda_time = getattr(event, 'self_cuda_time_total', 0) or getattr(event, 'cuda_time_total', 0) or 0
        if cuda_time > 0:
            kernel_data.append({
                "name": event.key,
                "cuda_time_total_us": cuda_time,
                "cuda_time_avg_us": cuda_time / max(event.count, 1),
                "count": event.count,
                "cpu_time_total_us": event.cpu_time_total,
            })

    # Sort by CUDA time
    kernel_data.sort(key=lambda x: x["cuda_time_total_us"], reverse=True)

    # Save kernel data
    with open(f"{OUTPUT_DIR}/kernel_profile.json", "w") as f:
        json.dump(kernel_data[:20], f, indent=2)

    # Export chrome trace
    prof.export_chrome_trace(f"{OUTPUT_DIR}/sglang_trace.json")
    print(f"\nChrome trace saved to {OUTPUT_DIR}/sglang_trace.json")
    print("  → Open in chrome://tracing or https://ui.perfetto.dev")

    # Detailed timing with CUDA events
    print("\n" + "=" * 60)
    print("Detailed Kernel Timing (CUDA Events)")
    print("=" * 60)

    # Measure different sequence lengths
    seq_lengths = [64, 128, 256, 512]
    results = []

    for seq_len in seq_lengths:
        q = torch.randn(1, seq_len, num_heads, head_dim,
                        device="cuda", dtype=torch.float16)
        k = torch.randn(1, seq_len, num_heads, head_dim,
                        device="cuda", dtype=torch.float16)
        v = torch.randn(1, seq_len, num_heads, head_dim,
                        device="cuda", dtype=torch.float16)

        # Warmup
        for _ in range(5):
            _ = single_prefill_with_kv_cache(q, k, v)

        # Measure
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start.record()
        for _ in range(100):
            _ = single_prefill_with_kv_cache(q, k, v)
        end.record()
        torch.cuda.synchronize()

        avg_time = start.elapsed_time(end) / 100

        # Calculate arithmetic intensity
        # FlashAttention: O(n * d) memory, O(n^2 * d) compute
        flops = 4 * seq_len * seq_len * head_dim * num_heads  # Approximate
        bytes_accessed = 3 * seq_len * head_dim * num_heads * 2  # Q, K, V in FP16
        arithmetic_intensity = flops / bytes_accessed

        results.append({
            "seq_len": seq_len,
            "avg_time_ms": avg_time,
            "flops": flops,
            "bytes": bytes_accessed,
            "arithmetic_intensity": arithmetic_intensity,
        })

        print(f"  seq_len={seq_len:4d}: {avg_time:.3f} ms, AI={arithmetic_intensity:.1f} FLOPS/byte")

    # Save results
    with open(f"{OUTPUT_DIR}/kernel_scaling.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Profiling Complete")
    print("=" * 60)
    print(f"\nArtifacts saved to {OUTPUT_DIR}/:")
    print(f"  - kernel_profile.json (top kernels)")
    print(f"  - kernel_scaling.json (sequence length scaling)")
    print(f"  - sglang_trace.json (Chrome trace for visualization)")

    return kernel_data, results


def profile_decode_vs_prefill():
    """Compare decode and prefill kernel characteristics."""
    from flashinfer import (
        single_prefill_with_kv_cache,
        single_decode_with_kv_cache,
    )

    print("\n" + "=" * 60)
    print("Prefill vs Decode Kernel Comparison")
    print("=" * 60)

    num_heads = 16
    head_dim = 64
    kv_len = 512  # KV cache length

    # Prefill: multiple query tokens
    q_prefill = torch.randn(1, 256, num_heads, head_dim,
                            device="cuda", dtype=torch.float16)
    k = torch.randn(1, kv_len, num_heads, head_dim,
                    device="cuda", dtype=torch.float16)
    v = torch.randn(1, kv_len, num_heads, head_dim,
                    device="cuda", dtype=torch.float16)

    # Decode: single query token
    q_decode = torch.randn(1, 1, num_heads, head_dim,
                           device="cuda", dtype=torch.float16)

    # Warmup
    for _ in range(5):
        _ = single_prefill_with_kv_cache(q_prefill, k, v)
        _ = single_decode_with_kv_cache(q_decode, k, v)

    # Measure prefill
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()
    for _ in range(100):
        _ = single_prefill_with_kv_cache(q_prefill, k, v)
    end.record()
    torch.cuda.synchronize()
    prefill_time = start.elapsed_time(end) / 100

    # Measure decode
    torch.cuda.synchronize()
    start.record()
    for _ in range(100):
        _ = single_decode_with_kv_cache(q_decode, k, v)
    end.record()
    torch.cuda.synchronize()
    decode_time = start.elapsed_time(end) / 100

    print(f"\nPrefill (256 tokens → 512 KV): {prefill_time:.3f} ms")
    print(f"Decode  (1 token → 512 KV):   {decode_time:.3f} ms")
    print(f"\nPrefill/Decode ratio: {prefill_time/decode_time:.1f}x")
    print(f"Per-token prefill: {prefill_time/256*1000:.1f} us/token")
    print(f"Per-token decode:  {decode_time*1000:.1f} us/token")

    # Characterization
    print("\nKernel Characterization:")
    print("  Prefill: Compute-bound (many Q tokens, batch GEMM-like)")
    print("  Decode:  Memory-bound (single Q token, GEMV-like)")

    return {
        "prefill_time_ms": prefill_time,
        "decode_time_ms": decode_time,
        "prefill_tokens": 256,
        "kv_length": kv_len,
    }


if __name__ == "__main__":
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available!")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    # Run profiling
    kernel_data, scaling_results = profile_model_forward()
    prefill_decode = profile_decode_vs_prefill()

    # Save combined results
    all_results = {
        "top_kernels": kernel_data[:10],
        "seq_len_scaling": scaling_results,
        "prefill_vs_decode": prefill_decode,
    }

    with open(f"{OUTPUT_DIR}/full_kernel_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nFull analysis saved to {OUTPUT_DIR}/full_kernel_analysis.json")
