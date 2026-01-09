#!/usr/bin/env python3
"""
FlashInfer RoPE Profiling Script

This script profiles the Rotary Position Embedding kernels to understand:
- In-place vs out-of-place variants
- Position encoding computation patterns
- Memory access efficiency
- Interleaved vs non-interleaved layouts

Output follows the process+hardware joint analysis format from PLAN.md.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Disable version check
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

import torch
import flashinfer


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default test configurations
# (batch_size, seq_len, num_heads, head_dim)
DEFAULT_CONFIGS = [
    (1, 512, 32, 128),       # Single batch, medium sequence
    (1, 2048, 32, 128),      # Single batch, long sequence
    (8, 512, 32, 128),       # Small batch
    (32, 512, 32, 128),      # Medium batch
    (128, 512, 32, 128),     # Large batch
    (32, 512, 8, 128),       # Fewer heads (GQA-style K heads)
    (32, 512, 32, 64),       # Smaller head dim
]


# ============================================================================
# PROFILING UTILITIES
# ============================================================================

def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}\n")


def benchmark_kernel(
    fn,
    warmup: int = 10,
    repeat: int = 50,
    sync: bool = True
) -> Tuple[float, float]:
    """Benchmark a kernel and return (median_ms, std_ms)."""
    # Warmup
    for _ in range(warmup):
        fn()
        if sync:
            torch.cuda.synchronize()

    # Timing
    times = []
    for _ in range(repeat):
        if sync:
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    import statistics
    median = statistics.median(times)
    std = statistics.stdev(times) if len(times) > 1 else 0
    return median, std


# ============================================================================
# ROPE ANALYSIS
# ============================================================================

class RoPEAnalyzer:
    """Analyzer for RoPE kernels."""

    def __init__(self, device: int = 0):
        self.device = torch.device(f"cuda:{device}")
        torch.cuda.set_device(self.device)

        # Get device properties
        props = torch.cuda.get_device_properties(self.device)
        self.sm_count = props.multi_processor_count
        self.compute_capability = f"{props.major}.{props.minor}"

    def analyze_rope_inplace(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
    ) -> Dict:
        """Analyze in-place RoPE kernel."""
        # Calculate total tokens
        nnz = batch_size * seq_len

        # Setup inputs (ragged tensor format)
        q = torch.randn(nnz, num_heads, head_dim, dtype=dtype, device=self.device)
        k = torch.randn(nnz, num_heads, head_dim, dtype=dtype, device=self.device)

        # Indptr for batch
        indptr = torch.tensor(
            [i * seq_len for i in range(batch_size + 1)],
            dtype=torch.int32, device=self.device
        )

        # Position offsets
        offsets = torch.zeros(batch_size, dtype=torch.int32, device=self.device)

        # Warmup
        q_copy = q.clone()
        k_copy = k.clone()
        flashinfer.apply_rope_inplace(q_copy, k_copy, indptr, offsets)
        torch.cuda.synchronize()

        # Benchmark
        def run_rope():
            q_copy = q.clone()
            k_copy = k.clone()
            flashinfer.apply_rope_inplace(q_copy, k_copy, indptr, offsets)

        median_time, std_time = benchmark_kernel(run_rope, repeat=30)

        # Memory analysis
        bytes_per_elem = 2 if dtype in [torch.float16, torch.bfloat16] else 4

        # In-place: read Q, K; write Q, K (same locations)
        q_bytes = nnz * num_heads * head_dim * bytes_per_elem
        k_bytes = nnz * num_heads * head_dim * bytes_per_elem
        total_bytes = (q_bytes + k_bytes) * 2  # Read + write

        # FLOPs: For each element pair, compute sin/cos and apply rotation
        # Per position: ~8 FLOPs (2 muls for sin/cos lookup, 4 for rotation)
        flops_per_pos = head_dim * 8
        total_flops = nnz * num_heads * flops_per_pos

        # Throughput
        achieved_bw_gbps = (total_bytes / 1e9) / (median_time / 1000)
        peak_bw = 2039

        arithmetic_intensity = total_flops / total_bytes

        results = {
            "config": {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "total_tokens": nnz,
                "dtype": str(dtype),
            },
            "timing": {
                "run_median_ms": median_time,
                "run_std_ms": std_time,
                "per_token_us": (median_time * 1000) / nnz,
            },
            "algorithm": {
                "variant": "in-place",
                "position_encoding": "computed on-the-fly",
                "layout": "non-interleaved (NeoX style)",
            },
            "memory": {
                "q_bytes": q_bytes,
                "k_bytes": k_bytes,
                "total_bytes": total_bytes,
            },
            "compute": {
                "total_flops": total_flops,
                "arithmetic_intensity": arithmetic_intensity,
            },
            "throughput": {
                "achieved_bw_gbps": achieved_bw_gbps,
                "memory_utilization_pct": (achieved_bw_gbps / peak_bw) * 100,
            },
        }

        return results

    def analyze_rope_with_cache(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int = 8192,
        dtype: torch.dtype = torch.float16,
    ) -> Dict:
        """Analyze RoPE with precomputed cos/sin cache."""
        # Calculate total tokens
        nnz = batch_size * seq_len

        # Setup inputs
        q = torch.randn(nnz, num_heads * head_dim, dtype=dtype, device=self.device)
        k = torch.randn(nnz, num_heads * head_dim, dtype=dtype, device=self.device)

        # Position IDs
        positions = torch.arange(nnz, dtype=torch.int32, device=self.device)

        # Precomputed cos/sin cache (FP32 required)
        rotary_dim = head_dim
        cos_sin_cache = torch.randn(
            max_seq_len, rotary_dim,
            dtype=torch.float32, device=self.device
        )

        # Warmup
        _, _ = flashinfer.apply_rope_with_cos_sin_cache(
            positions, q.clone(), k.clone(), head_dim, cos_sin_cache
        )
        torch.cuda.synchronize()

        # Benchmark
        def run_rope_cached():
            flashinfer.apply_rope_with_cos_sin_cache(
                positions, q.clone(), k.clone(), head_dim, cos_sin_cache
            )

        median_time, std_time = benchmark_kernel(run_rope_cached, repeat=30)

        # Memory analysis
        bytes_per_elem = 2 if dtype in [torch.float16, torch.bfloat16] else 4

        # Read Q, K, cos_sin_cache; Write Q_out, K_out
        qk_bytes = nnz * num_heads * head_dim * bytes_per_elem * 2  # Q and K
        cache_bytes = nnz * rotary_dim * 4  # FP32 cache lookups
        output_bytes = qk_bytes
        total_bytes = qk_bytes + cache_bytes + output_bytes

        # Throughput
        achieved_bw_gbps = (total_bytes / 1e9) / (median_time / 1000)
        peak_bw = 2039

        results = {
            "config": {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "total_tokens": nnz,
                "max_seq_len": max_seq_len,
                "dtype": str(dtype),
            },
            "timing": {
                "run_median_ms": median_time,
                "run_std_ms": std_time,
                "per_token_us": (median_time * 1000) / nnz,
            },
            "algorithm": {
                "variant": "with cos_sin_cache",
                "position_encoding": "precomputed lookup",
                "layout": "NeoX style (non-interleaved)",
            },
            "memory": {
                "qk_bytes": qk_bytes,
                "cache_bytes": cache_bytes,
                "output_bytes": output_bytes,
                "total_bytes": total_bytes,
            },
            "throughput": {
                "achieved_bw_gbps": achieved_bw_gbps,
                "memory_utilization_pct": (achieved_bw_gbps / peak_bw) * 100,
            },
        }

        return results

    def print_analysis(self, results: Dict, kernel_name: str = "RoPE"):
        """Print analysis in the expected format."""
        config = results["config"]
        timing = results["timing"]
        algo = results["algorithm"]
        mem = results["memory"]
        throughput = results["throughput"]

        print(f"""
{kernel_name.upper()} ANALYSIS:
+-- Variant: {algo['variant']}
+-- Execution:
|   +-- Position encoding: {algo['position_encoding']}
|   +-- Layout: {algo['layout']}
|   +-- Total tokens: {config['total_tokens']}
+-- Hardware:
|   +-- Memory BW: {throughput['achieved_bw_gbps']:.1f} GB/s ({throughput['memory_utilization_pct']:.1f}% of peak)
+-- Timing:
|   +-- Run phase: {timing['run_median_ms']:.4f} +/- {timing.get('run_std_ms', 0):.4f} ms
|   +-- Per-token: {timing['per_token_us']:.2f} us
+-- Memory:
|   +-- Total traffic: {mem['total_bytes'] / 1e6:.2f} MB
+-- Classification: Memory-bound elementwise kernel
""")


def run_profiling(configs: List[Tuple], output_dir: Path, device: int = 0):
    """Run profiling for all configurations."""
    analyzer = RoPEAnalyzer(device=device)

    print_section("ROPE KERNEL PROFILING")
    print(f"Device: GPU {device} ({analyzer.compute_capability})")
    print(f"SM Count: {analyzer.sm_count}")

    all_results = {"rope_inplace": [], "rope_with_cache": []}

    # Profile in-place RoPE
    print_section("RoPE In-Place (apply_rope_inplace)")
    for config in configs:
        batch_size, seq_len, num_heads, head_dim = config
        print(f"\n--- Config: batch={batch_size}, seq={seq_len}, heads={num_heads}, dim={head_dim} ---")

        try:
            results = analyzer.analyze_rope_inplace(batch_size, seq_len, num_heads, head_dim)
            analyzer.print_analysis(results, "RoPE In-Place")
            all_results["rope_inplace"].append(results)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Profile RoPE with cos/sin cache
    print_section("RoPE with Cos/Sin Cache (apply_rope_with_cos_sin_cache)")
    for config in configs[:5]:  # Subset
        batch_size, seq_len, num_heads, head_dim = config
        print(f"\n--- Config: batch={batch_size}, seq={seq_len}, heads={num_heads} ---")

        try:
            results = analyzer.analyze_rope_with_cache(batch_size, seq_len, num_heads, head_dim)
            analyzer.print_analysis(results, "RoPE with Cache")
            all_results["rope_with_cache"].append(results)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_file = output_dir / "rope_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    # Print kernel implementation summary
    print_section("ROPE KERNEL IMPLEMENTATION SUMMARY")
    print("""
RoPE (Rotary Position Embedding) Implementation:
=================================================

+-- Algorithm:
|   For each position p and dimension pair (i, i+d/2):
|   1. Compute rotation angle: theta_i = p * base^(-2i/d)
|   2. Apply rotation:
|      q'[i]     = q[i] * cos(theta) - q[i+d/2] * sin(theta)
|      q'[i+d/2] = q[i] * sin(theta) + q[i+d/2] * cos(theta)

+-- Layout Options:
|   +-- Non-interleaved (NeoX): Rotate [0:d/2] with [d/2:d]
|   +-- Interleaved: Rotate [0::2] with [1::2]

+-- Variants Available:
|   +-- apply_rope_inplace: Modify Q, K in place
|   +-- apply_rope: Return new Q, K tensors
|   +-- apply_rope_with_cos_sin_cache: Use precomputed cos/sin
|   +-- apply_rope_pos_ids: Custom position IDs per token
|   +-- apply_llama31_rope: Llama 3.1 style with scaling

+-- Memory Access Pattern:
|   +-- Coalesced reads/writes on the head dimension
|   +-- Each thread handles multiple consecutive elements
|   +-- Vectorized loads (128-bit)

+-- Expected Hardware Behavior:
|   +-- Occupancy: High (low register/SMEM usage)
|   +-- Warp stalls: long_scoreboard (HBM reads)
|   +-- Compute: Minimal (trig lookup + 4 multiplies per pair)
|   +-- Memory BW: 70-85% of peak

+-- When to Use Each Variant:
|   +-- apply_rope_inplace: Standard prefill/decode, saves memory
|   +-- apply_rope_with_cos_sin_cache: When cos/sin precomputed (SGLang/vLLM style)
|   +-- apply_llama31_rope: For Llama 3.1+ models with extended context
""")

    return all_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FlashInfer RoPE Profiler")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*70)
    print(" FLASHINFER ROPE PROFILER")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    run_profiling(DEFAULT_CONFIGS, output_dir, device=args.device)

    print_section("PROFILING COMPLETE")


if __name__ == "__main__":
    main()
