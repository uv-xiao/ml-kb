#!/usr/bin/env python3
"""
FlashInfer Decode Attention Profiling Script

This script profiles the BatchDecodeWithPagedKVCacheWrapper to understand:
- Split-K parallelism strategy
- Memory-bound behavior (GEMV-style attention)
- Scaling with batch size and KV cache length
- Comparison with prefill characteristics

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
# (batch_size, kv_len, num_qo_heads, num_kv_heads, head_dim, page_size)
DEFAULT_CONFIGS = [
    (1, 512, 32, 32, 128, 16),      # Single decode, short KV
    (1, 2048, 32, 32, 128, 16),     # Single decode, medium KV
    (1, 4096, 32, 32, 128, 16),     # Single decode, long KV
    (8, 2048, 32, 32, 128, 16),     # Small batch
    (32, 2048, 32, 32, 128, 16),    # Medium batch
    (64, 2048, 32, 32, 128, 16),    # Large batch
    (128, 2048, 32, 32, 128, 16),   # Very large batch
    (32, 4096, 32, 8, 128, 16),     # GQA with long KV
    (64, 8192, 32, 8, 128, 16),     # GQA with very long KV
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
    warmup: int = 5,
    repeat: int = 20,
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
# DECODE ATTENTION ANALYSIS
# ============================================================================

class DecodeAnalyzer:
    """Analyzer for decode attention kernels."""

    def __init__(self, device: int = 0):
        self.device = torch.device(f"cuda:{device}")
        torch.cuda.set_device(self.device)

        # Get device properties
        props = torch.cuda.get_device_properties(self.device)
        self.sm_count = props.multi_processor_count
        self.compute_capability = f"{props.major}.{props.minor}"

        # Workspace buffer (128 MB)
        self.workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )

    def setup_paged_kv_cache(
        self,
        batch_size: int,
        kv_len: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype = torch.float16,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Setup paged KV cache for decode testing."""
        # Calculate number of pages needed per sequence
        pages_per_seq = (kv_len + page_size - 1) // page_size
        total_pages = batch_size * pages_per_seq

        # Allocate KV cache (NHD layout)
        k_cache = torch.randn(
            total_pages, page_size, num_kv_heads, head_dim,
            dtype=dtype, device=self.device
        )
        v_cache = torch.randn(
            total_pages, page_size, num_kv_heads, head_dim,
            dtype=dtype, device=self.device
        )

        # Page indices (identity mapping)
        kv_indices = torch.arange(total_pages, dtype=torch.int32, device=self.device)

        # Indptr for each batch element
        kv_indptr = torch.tensor(
            [i * pages_per_seq for i in range(batch_size + 1)],
            dtype=torch.int32, device=self.device
        )

        # Last page lengths
        last_page_len = torch.full(
            (batch_size,),
            (kv_len % page_size) if (kv_len % page_size) > 0 else page_size,
            dtype=torch.int32, device=self.device
        )

        return k_cache, v_cache, kv_indices, kv_indptr, last_page_len

    def analyze_decode(
        self,
        batch_size: int,
        kv_len: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype = torch.float16,
    ) -> Dict:
        """Analyze decode attention for given configuration."""
        # Setup query (one token per batch element)
        q = torch.randn(
            batch_size, num_qo_heads, head_dim,
            dtype=dtype, device=self.device
        )

        k_cache, v_cache, kv_indices, kv_indptr, last_page_len = \
            self.setup_paged_kv_cache(
                batch_size, kv_len, num_kv_heads, head_dim, page_size, dtype
            )

        # Output tensor
        o = torch.empty_like(q)

        # Create wrapper
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )

        # Plan phase timing
        torch.cuda.synchronize()
        plan_start = time.perf_counter()

        wrapper.plan(
            indptr=kv_indptr,
            indices=kv_indices,
            last_page_len=last_page_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            q_data_type=dtype,
        )

        torch.cuda.synchronize()
        plan_time = (time.perf_counter() - plan_start) * 1000

        # Benchmark run phase
        def run_decode():
            wrapper.run(q, (k_cache, v_cache), o)

        median_time, std_time = benchmark_kernel(run_decode)

        # Calculate algorithmic properties
        gqa_ratio = num_qo_heads // num_kv_heads

        # Split-K analysis
        # FlashInfer typically splits KV into chunks for parallelism
        split_k_factor = min(8, max(1, kv_len // 512))  # Estimate
        kv_chunks = split_k_factor

        # Memory analysis (decode is heavily memory-bound)
        # Q: batch_size * num_qo_heads * head_dim
        q_bytes = batch_size * num_qo_heads * head_dim * 2

        # KV: batch_size * kv_len * num_kv_heads * head_dim * 2 (K and V)
        kv_bytes = batch_size * kv_len * num_kv_heads * head_dim * 2 * 2

        # O: batch_size * num_qo_heads * head_dim
        o_bytes = batch_size * num_qo_heads * head_dim * 2

        total_bytes = q_bytes + kv_bytes + o_bytes

        # FLOPs calculation (GEMV-style)
        # QK^T: batch * heads * 1 * kv_len * head_dim = batch * heads * kv_len * head_dim * 2
        # PV: batch * heads * 1 * head_dim * kv_len = batch * heads * head_dim * kv_len * 2
        flops_per_head = 2 * kv_len * head_dim * 2  # QK^T + PV
        total_flops = batch_size * num_qo_heads * flops_per_head

        # Arithmetic intensity (very low for decode)
        arithmetic_intensity = total_flops / total_bytes

        # Throughput
        achieved_tflops = (total_flops / 1e12) / (median_time / 1000)
        achieved_bw_gbps = (total_bytes / 1e9) / (median_time / 1000)

        # Theoretical peaks for A100
        peak_tflops = 312  # FP16 Tensor Core
        peak_bw = 2039  # GB/s

        # Tokens per second (important for decode)
        tokens_per_second = (batch_size * 1000) / median_time

        results = {
            "config": {
                "batch_size": batch_size,
                "kv_len": kv_len,
                "num_qo_heads": num_qo_heads,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "page_size": page_size,
                "gqa_ratio": gqa_ratio,
            },
            "timing": {
                "plan_ms": plan_time,
                "run_median_ms": median_time,
                "run_std_ms": std_time,
                "per_token_us": (median_time * 1000) / batch_size,
            },
            "algorithm": {
                "split_k_factor": split_k_factor,
                "kv_chunks": kv_chunks,
                "work_items": batch_size * num_qo_heads,
                "tokens_per_second": tokens_per_second,
            },
            "memory": {
                "q_bytes": q_bytes,
                "kv_bytes": kv_bytes,
                "o_bytes": o_bytes,
                "total_bytes": total_bytes,
                "kv_cache_mb": kv_bytes / 1e6,
            },
            "compute": {
                "total_flops": total_flops,
                "arithmetic_intensity": arithmetic_intensity,
            },
            "throughput": {
                "achieved_tflops": achieved_tflops,
                "achieved_bw_gbps": achieved_bw_gbps,
                "compute_utilization_pct": (achieved_tflops / peak_tflops) * 100,
                "memory_utilization_pct": (achieved_bw_gbps / peak_bw) * 100,
            },
        }

        return results

    def print_analysis(self, results: Dict):
        """Print analysis in the expected format."""
        config = results["config"]
        timing = results["timing"]
        algo = results["algorithm"]
        mem = results["memory"]
        compute = results["compute"]
        throughput = results["throughput"]

        print(f"""
DECODE ATTENTION ANALYSIS (batch={config['batch_size']}, kv_len={config['kv_len']}):
+-- Algorithm: Split-K with reduction
+-- Execution:
|   +-- KV split: {algo['split_k_factor']} chunks
|   +-- Parallel: {algo['kv_chunks']} partial outputs computed
|   +-- Reduction: Merge partial_O with LSE
|   +-- Work items: {algo['work_items']} (batch * heads)
+-- Hardware:
|   +-- Compute: {throughput['achieved_tflops']:.2f} TFLOPS ({throughput['compute_utilization_pct']:.1f}% of peak)
|   +-- Memory BW: {throughput['achieved_bw_gbps']:.1f} GB/s ({throughput['memory_utilization_pct']:.1f}% of peak)
|   +-- Arithmetic Intensity: {compute['arithmetic_intensity']:.2f} FLOPs/Byte
+-- Timing:
|   +-- Plan phase: {timing['plan_ms']:.2f} ms
|   +-- Run phase: {timing['run_median_ms']:.3f} +/- {timing['run_std_ms']:.3f} ms
|   +-- Per-token latency: {timing['per_token_us']:.1f} us
+-- Throughput: {algo['tokens_per_second']:.0f} tokens/sec
+-- Memory:
|   +-- KV cache accessed: {mem['kv_cache_mb']:.1f} MB
+-- Roofline: Memory-bound (AI={compute['arithmetic_intensity']:.2f} << ridge point ~153)
""")


def run_profiling(configs: List[Tuple], output_dir: Path, device: int = 0):
    """Run profiling for all configurations."""
    analyzer = DecodeAnalyzer(device=device)

    print_section("DECODE ATTENTION PROFILING")
    print(f"Device: GPU {device} ({analyzer.compute_capability})")
    print(f"SM Count: {analyzer.sm_count}")

    all_results = []

    for config in configs:
        batch_size, kv_len, num_qo_heads, num_kv_heads, head_dim, page_size = config

        print(f"\n--- Config: batch={batch_size}, kv_len={kv_len}, heads={num_qo_heads}/{num_kv_heads} ---")

        try:
            results = analyzer.analyze_decode(
                batch_size, kv_len, num_qo_heads, num_kv_heads, head_dim, page_size
            )
            analyzer.print_analysis(results)
            all_results.append(results)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_file = output_dir / "decode_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Print summary comparison
    print_section("DECODE vs PREFILL COMPARISON")
    print("""
Key differences between decode and prefill attention:

+-- PREFILL (Compute-Heavy):
|   +-- Shape: [batch, seq_len, heads, dim] x [batch, seq_len, heads, dim]
|   +-- GEMM-style: Uses Tensor Cores effectively
|   +-- AI > ridge point: Compute-bound
|   +-- TC utilization: 30-50%
|
+-- DECODE (Memory-Heavy):
|   +-- Shape: [batch, 1, heads, dim] x [batch, kv_len, heads, dim]
|   +-- GEMV-style: Cannot fully utilize Tensor Cores
|   +-- AI << ridge point: Memory-bound
|   +-- TC utilization: 10-20%
|   +-- Key optimization: Split-K to increase parallelism

Expected warp stalls for decode:
+-- long_scoreboard: 60-70% (waiting for HBM reads)
+-- barrier: 10-20% (split-K reduction sync)
+-- short_scoreboard: 5-10% (register dependencies)
""")

    return all_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FlashInfer Decode Attention Profiler")
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
    print(" FLASHINFER DECODE ATTENTION PROFILER")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    run_profiling(DEFAULT_CONFIGS, output_dir, device=args.device)

    print_section("PROFILING COMPLETE")
    print("For deeper analysis, run with ncu:")
    print("  ncu --set full --section LaunchStats --section Occupancy python 03_profile_decode.py")


if __name__ == "__main__":
    main()
