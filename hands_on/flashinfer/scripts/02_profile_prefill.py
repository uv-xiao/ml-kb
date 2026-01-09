#!/usr/bin/env python3
"""
FlashInfer Prefill Attention Profiling Script

This script profiles the BatchPrefillWithPagedKVCacheWrapper to understand:
- FlashAttention-2 algorithm behavior
- Tile scheduling and work distribution
- Hardware utilization patterns
- Scaling with sequence length and batch size

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
DEFAULT_CONFIGS = [
    # (batch_size, seq_len, num_qo_heads, num_kv_heads, head_dim, page_size)
    (1, 512, 32, 32, 128, 16),     # Single request, medium sequence
    (1, 2048, 32, 32, 128, 16),    # Single request, long sequence
    (1, 4096, 32, 32, 128, 16),    # Single request, very long sequence
    (8, 512, 32, 32, 128, 16),     # Small batch, medium sequence
    (32, 512, 32, 32, 128, 16),    # Large batch, medium sequence
    (1, 512, 32, 8, 128, 16),      # GQA configuration
    (8, 512, 32, 8, 128, 16),      # GQA with batch
]


# ============================================================================
# PROFILING UTILITIES
# ============================================================================

def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}\n")


def cuda_sync_time() -> float:
    """Get synchronized time."""
    torch.cuda.synchronize()
    return time.perf_counter()


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
# PREFILL ATTENTION ANALYSIS
# ============================================================================

class PrefillAnalyzer:
    """Analyzer for prefill attention kernels."""

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
        seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype = torch.float16,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Setup paged KV cache for testing."""
        # Calculate number of pages needed
        pages_per_seq = (seq_len + page_size - 1) // page_size
        total_pages = batch_size * pages_per_seq

        # Allocate KV cache
        k_cache = torch.randn(
            total_pages, page_size, num_kv_heads, head_dim,
            dtype=dtype, device=self.device
        )
        v_cache = torch.randn(
            total_pages, page_size, num_kv_heads, head_dim,
            dtype=dtype, device=self.device
        )

        # Page indices (identity mapping for simplicity)
        kv_indices = torch.arange(total_pages, dtype=torch.int32, device=self.device)

        # Indptr for each batch element
        kv_indptr = torch.tensor(
            [i * pages_per_seq for i in range(batch_size + 1)],
            dtype=torch.int32, device=self.device
        )

        # Last page lengths
        last_page_len = torch.full(
            (batch_size,),
            (seq_len % page_size) if (seq_len % page_size) > 0 else page_size,
            dtype=torch.int32, device=self.device
        )

        return k_cache, v_cache, kv_indices, kv_indptr, last_page_len

    def analyze_prefill(
        self,
        batch_size: int,
        seq_len: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype = torch.float16,
    ) -> Dict:
        """Analyze prefill attention for given configuration."""
        # Setup inputs
        total_tokens = batch_size * seq_len
        q = torch.randn(
            total_tokens, num_qo_heads, head_dim,
            dtype=dtype, device=self.device
        )

        k_cache, v_cache, kv_indices, kv_indptr, last_page_len = \
            self.setup_paged_kv_cache(
                batch_size, seq_len, num_kv_heads, head_dim, page_size, dtype
            )

        # Query indptr (for batch prefill)
        qo_indptr = torch.tensor(
            [i * seq_len for i in range(batch_size + 1)],
            dtype=torch.int32, device=self.device
        )

        # Output tensor
        o = torch.empty_like(q)

        # Create wrapper
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )

        # Plan phase timing
        torch.cuda.synchronize()
        plan_start = time.perf_counter()

        wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=last_page_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            page_size=page_size,
            q_data_type=dtype,
            causal=True,
        )

        torch.cuda.synchronize()
        plan_time = (time.perf_counter() - plan_start) * 1000

        # Benchmark run phase
        def run_prefill():
            wrapper.run(q, (k_cache, v_cache), o)

        median_time, std_time = benchmark_kernel(run_prefill)

        # Calculate algorithmic properties
        gqa_ratio = num_qo_heads // num_kv_heads
        tile_size_q = 128  # FlashInfer default Q tile size
        tile_size_kv = 128  # FlashInfer default KV tile size

        num_q_tiles = (seq_len + tile_size_q - 1) // tile_size_q
        num_kv_tiles = (seq_len + tile_size_kv - 1) // tile_size_kv

        # Work distribution estimate
        total_work_items = batch_size * num_qo_heads * num_q_tiles
        work_per_sm = total_work_items / self.sm_count

        # Memory analysis
        q_bytes = total_tokens * num_qo_heads * head_dim * 2  # FP16
        kv_bytes = total_tokens * num_kv_heads * head_dim * 2 * 2  # K and V
        o_bytes = total_tokens * num_qo_heads * head_dim * 2
        total_bytes = q_bytes + kv_bytes + o_bytes

        # FLOPs calculation (2 * N * N * D for each head, for attention)
        # QK^T: 2 * seq_len * seq_len * head_dim
        # softmax: ~5 * seq_len * seq_len
        # PV: 2 * seq_len * seq_len * head_dim
        flops_per_head = 2 * seq_len * seq_len * head_dim * 2  # QK^T + PV
        total_flops = batch_size * num_qo_heads * flops_per_head

        # Arithmetic intensity
        arithmetic_intensity = total_flops / total_bytes

        # Throughput
        achieved_tflops = (total_flops / 1e12) / (median_time / 1000)
        achieved_bw_gbps = (total_bytes / 1e9) / (median_time / 1000)

        # Theoretical peaks for A100
        peak_tflops = 312  # FP16 Tensor Core
        peak_bw = 2039  # GB/s

        results = {
            "config": {
                "batch_size": batch_size,
                "seq_len": seq_len,
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
            },
            "algorithm": {
                "tile_size_q": tile_size_q,
                "tile_size_kv": tile_size_kv,
                "num_q_tiles": num_q_tiles,
                "num_kv_tiles": num_kv_tiles,
                "total_work_items": total_work_items,
                "work_per_sm": work_per_sm,
            },
            "memory": {
                "q_bytes": q_bytes,
                "kv_bytes": kv_bytes,
                "o_bytes": o_bytes,
                "total_bytes": total_bytes,
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
PREFILL ATTENTION ANALYSIS (seq_len={config['seq_len']}, heads={config['num_qo_heads']}):
+-- Algorithm: FlashAttention-2 with online softmax
+-- Execution:
|   +-- Q tiles: {algo['num_q_tiles']} (each {algo['tile_size_q']} tokens)
|   +-- K tiles: {algo['num_kv_tiles']} (each {algo['tile_size_kv']} tokens)
|   +-- Work distribution: {algo['total_work_items']} items across {self.sm_count} SMs
|   +-- Work per SM: {algo['work_per_sm']:.1f} items
+-- Hardware:
|   +-- Compute: {throughput['achieved_tflops']:.1f} TFLOPS ({throughput['compute_utilization_pct']:.1f}% of peak)
|   +-- Memory BW: {throughput['achieved_bw_gbps']:.1f} GB/s ({throughput['memory_utilization_pct']:.1f}% of peak)
|   +-- Arithmetic Intensity: {compute['arithmetic_intensity']:.1f} FLOPs/Byte
+-- Timing:
|   +-- Plan phase: {timing['plan_ms']:.2f} ms
|   +-- Run phase: {timing['run_median_ms']:.3f} +/- {timing['run_std_ms']:.3f} ms
+-- Roofline: {'Compute-bound' if compute['arithmetic_intensity'] > 153 else 'Memory-bound'} (ridge ~153 FLOPs/Byte)
""")


def run_profiling(configs: List[Tuple], output_dir: Path, device: int = 0):
    """Run profiling for all configurations."""
    analyzer = PrefillAnalyzer(device=device)

    print_section("PREFILL ATTENTION PROFILING")
    print(f"Device: GPU {device} ({analyzer.compute_capability})")
    print(f"SM Count: {analyzer.sm_count}")

    all_results = []

    for config in configs:
        batch_size, seq_len, num_qo_heads, num_kv_heads, head_dim, page_size = config

        print(f"\n--- Config: batch={batch_size}, seq={seq_len}, heads={num_qo_heads}/{num_kv_heads} ---")

        try:
            results = analyzer.analyze_prefill(
                batch_size, seq_len, num_qo_heads, num_kv_heads, head_dim, page_size
            )
            analyzer.print_analysis(results)
            all_results.append(results)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_file = output_dir / "prefill_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return all_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FlashInfer Prefill Attention Profiler")
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
    print(" FLASHINFER PREFILL ATTENTION PROFILER")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    run_profiling(DEFAULT_CONFIGS, output_dir, device=args.device)

    print_section("PROFILING COMPLETE")
    print("Next steps:")
    print("  1. Run with nsys for detailed timeline: nsys profile -o prefill python 02_profile_prefill.py")
    print("  2. Run with ncu for kernel analysis: ncu --set full -o prefill_ncu python 02_profile_prefill.py")


if __name__ == "__main__":
    main()
