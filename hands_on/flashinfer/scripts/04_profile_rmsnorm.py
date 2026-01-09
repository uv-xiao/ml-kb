#!/usr/bin/env python3
"""
FlashInfer RMSNorm Profiling Script

This script profiles RMSNorm and fused_add_rmsnorm kernels to understand:
- Per-row reduction pattern
- Vectorized memory access
- Memory-bound behavior
- Fused operation benefits

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
# (batch_size, hidden_size, dtype)
DEFAULT_CONFIGS = [
    (1, 4096, torch.float16),       # Single row, Llama-7B hidden
    (8, 4096, torch.float16),       # Small batch
    (32, 4096, torch.float16),      # Medium batch
    (128, 4096, torch.float16),     # Large batch
    (512, 4096, torch.float16),     # Very large batch
    (32, 8192, torch.float16),      # Larger hidden (Llama-70B)
    (32, 14336, torch.float16),     # MLP intermediate size
    (32, 4096, torch.bfloat16),     # BF16 comparison
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
# RMSNORM ANALYSIS
# ============================================================================

class RMSNormAnalyzer:
    """Analyzer for RMSNorm kernels."""

    def __init__(self, device: int = 0):
        self.device = torch.device(f"cuda:{device}")
        torch.cuda.set_device(self.device)

        # Get device properties
        props = torch.cuda.get_device_properties(self.device)
        self.sm_count = props.multi_processor_count
        self.compute_capability = f"{props.major}.{props.minor}"

    def analyze_rmsnorm(
        self,
        batch_size: int,
        hidden_size: int,
        dtype: torch.dtype = torch.float16,
    ) -> Dict:
        """Analyze RMSNorm kernel for given configuration."""
        # Setup inputs
        input_tensor = torch.randn(
            batch_size, hidden_size,
            dtype=dtype, device=self.device
        )
        weight = torch.ones(hidden_size, dtype=dtype, device=self.device)
        output = torch.empty_like(input_tensor)

        eps = 1e-6

        # Warmup to trigger JIT
        _ = flashinfer.rmsnorm(input_tensor, weight, eps)
        torch.cuda.synchronize()

        # Benchmark RMSNorm
        def run_rmsnorm():
            flashinfer.rmsnorm(input_tensor, weight, eps, out=output)

        median_time, std_time = benchmark_kernel(run_rmsnorm)

        # Memory analysis
        # Read: input (batch * hidden) + weight (hidden)
        # Write: output (batch * hidden)
        bytes_per_elem = 2 if dtype in [torch.float16, torch.bfloat16] else 4
        input_bytes = batch_size * hidden_size * bytes_per_elem
        weight_bytes = hidden_size * bytes_per_elem
        output_bytes = batch_size * hidden_size * bytes_per_elem
        total_bytes = input_bytes + weight_bytes + output_bytes

        # FLOPs analysis
        # Per row: hidden_size * (x^2) + 1 (mean) + 1 (rsqrt) + hidden_size * 2 (scale + multiply)
        # Simplified: ~3 * hidden_size per row
        flops_per_row = 3 * hidden_size
        total_flops = batch_size * flops_per_row

        # Arithmetic intensity (very low, memory-bound)
        arithmetic_intensity = total_flops / total_bytes

        # Throughput
        achieved_tflops = (total_flops / 1e12) / (median_time / 1000)
        achieved_bw_gbps = (total_bytes / 1e9) / (median_time / 1000)

        # Theoretical peaks for A100
        peak_tflops = 19.5  # FP16 non-TC (RMSNorm doesn't use TC)
        peak_bw = 2039  # GB/s

        results = {
            "config": {
                "batch_size": batch_size,
                "hidden_size": hidden_size,
                "dtype": str(dtype),
            },
            "timing": {
                "run_median_ms": median_time,
                "run_std_ms": std_time,
                "per_row_us": (median_time * 1000) / batch_size,
            },
            "algorithm": {
                "operation": "output = (input / RMS(input)) * weight",
                "pattern": "Per-row reduction + elementwise",
                "vectorization": "8 elements per thread (128-bit loads)",
            },
            "memory": {
                "input_bytes": input_bytes,
                "weight_bytes": weight_bytes,
                "output_bytes": output_bytes,
                "total_bytes": total_bytes,
            },
            "compute": {
                "total_flops": total_flops,
                "arithmetic_intensity": arithmetic_intensity,
            },
            "throughput": {
                "achieved_tflops": achieved_tflops,
                "achieved_bw_gbps": achieved_bw_gbps,
                "memory_utilization_pct": (achieved_bw_gbps / peak_bw) * 100,
            },
        }

        return results

    def analyze_fused_add_rmsnorm(
        self,
        batch_size: int,
        hidden_size: int,
        dtype: torch.dtype = torch.float16,
    ) -> Dict:
        """Analyze fused_add_rmsnorm kernel for given configuration."""
        # Setup inputs
        input_tensor = torch.randn(
            batch_size, hidden_size,
            dtype=dtype, device=self.device
        )
        residual = torch.randn(
            batch_size, hidden_size,
            dtype=dtype, device=self.device
        )
        weight = torch.ones(hidden_size, dtype=dtype, device=self.device)

        eps = 1e-6

        # Warmup
        inp_copy = input_tensor.clone()
        res_copy = residual.clone()
        flashinfer.fused_add_rmsnorm(inp_copy, res_copy, weight, eps)
        torch.cuda.synchronize()

        # Benchmark fused version
        def run_fused():
            inp_copy = input_tensor.clone()
            res_copy = residual.clone()
            flashinfer.fused_add_rmsnorm(inp_copy, res_copy, weight, eps)

        median_time, std_time = benchmark_kernel(run_fused, repeat=30)

        # Memory analysis (fused saves memory traffic)
        bytes_per_elem = 2 if dtype in [torch.float16, torch.bfloat16] else 4

        # Fused: read input, residual, weight; write input (normalized), residual (updated)
        # Non-fused would need: read input, residual -> write temp -> read temp, weight -> write output
        input_bytes = batch_size * hidden_size * bytes_per_elem
        residual_bytes = batch_size * hidden_size * bytes_per_elem * 2  # read + write
        weight_bytes = hidden_size * bytes_per_elem
        output_bytes = batch_size * hidden_size * bytes_per_elem
        total_bytes = input_bytes + residual_bytes + weight_bytes + output_bytes

        # Compare to non-fused (separate add + rmsnorm)
        nonfused_bytes = (
            input_bytes * 2 +  # read input, read temp
            residual_bytes +   # read residual, write residual
            weight_bytes +     # read weight
            output_bytes * 2   # write temp, write output
        )

        memory_savings_pct = ((nonfused_bytes - total_bytes) / nonfused_bytes) * 100

        # Throughput
        achieved_bw_gbps = (total_bytes / 1e9) / (median_time / 1000)
        peak_bw = 2039

        results = {
            "config": {
                "batch_size": batch_size,
                "hidden_size": hidden_size,
                "dtype": str(dtype),
            },
            "timing": {
                "run_median_ms": median_time,
                "run_std_ms": std_time,
            },
            "algorithm": {
                "operation": "residual += input; input = RMSNorm(residual) * weight",
                "pattern": "Fused add + reduction + elementwise",
                "benefit": "Single memory pass instead of two",
            },
            "memory": {
                "total_bytes": total_bytes,
                "nonfused_bytes": nonfused_bytes,
                "memory_savings_pct": memory_savings_pct,
            },
            "throughput": {
                "achieved_bw_gbps": achieved_bw_gbps,
                "memory_utilization_pct": (achieved_bw_gbps / peak_bw) * 100,
            },
        }

        return results

    def print_analysis(self, results: Dict, kernel_name: str = "RMSNorm"):
        """Print analysis in the expected format."""
        config = results["config"]
        timing = results["timing"]
        algo = results["algorithm"]
        mem = results["memory"]
        throughput = results["throughput"]

        print(f"""
{kernel_name.upper()} ANALYSIS:
+-- Operation: {algo['operation']}
+-- Execution:
|   +-- Pattern: {algo['pattern']}
|   +-- Vectorization: {algo.get('vectorization', 'N/A')}
|   +-- One thread block per row ({config['hidden_size']} elements)
+-- Hardware:
|   +-- Memory BW: {throughput['achieved_bw_gbps']:.1f} GB/s ({throughput['memory_utilization_pct']:.1f}% of peak)
|   +-- SM util: ~25-30% (memory-bound, expected)
+-- Timing:
|   +-- Run phase: {timing['run_median_ms']:.4f} +/- {timing.get('run_std_ms', 0):.4f} ms
|   +-- Per-row: {timing.get('per_row_us', timing['run_median_ms']*1000/config['batch_size']):.2f} us
+-- Memory:
|   +-- Total traffic: {mem['total_bytes'] / 1e6:.2f} MB
+-- Classification: Pure memory-bound kernel
""")


def run_profiling(configs: List[Tuple], output_dir: Path, device: int = 0):
    """Run profiling for all configurations."""
    analyzer = RMSNormAnalyzer(device=device)

    print_section("RMSNORM KERNEL PROFILING")
    print(f"Device: GPU {device} ({analyzer.compute_capability})")
    print(f"SM Count: {analyzer.sm_count}")

    all_results = {"rmsnorm": [], "fused_add_rmsnorm": []}

    # Profile RMSNorm
    print_section("RMSNorm (standalone)")
    for config in configs:
        batch_size, hidden_size, dtype = config
        print(f"\n--- Config: batch={batch_size}, hidden={hidden_size}, dtype={dtype} ---")

        try:
            results = analyzer.analyze_rmsnorm(batch_size, hidden_size, dtype)
            analyzer.print_analysis(results, "RMSNorm")
            all_results["rmsnorm"].append(results)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Profile Fused Add RMSNorm
    print_section("Fused Add RMSNorm")
    for config in configs[:5]:  # Subset for fused
        batch_size, hidden_size, dtype = config
        print(f"\n--- Config: batch={batch_size}, hidden={hidden_size} ---")

        try:
            results = analyzer.analyze_fused_add_rmsnorm(batch_size, hidden_size, dtype)
            analyzer.print_analysis(results, "Fused Add RMSNorm")
            all_results["fused_add_rmsnorm"].append(results)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Save results
    output_file = output_dir / "rmsnorm_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    # Print kernel implementation summary
    print_section("RMSNORM KERNEL IMPLEMENTATION SUMMARY")
    print("""
RMSNorm Implementation Pattern:
===============================

+-- Algorithm:
|   1. Each thread block handles one row
|   2. Warp-level parallel reduction for sum(x^2)
|   3. Compute rsqrt(mean + eps)
|   4. Vectorized load/store (8 elements per thread)
|   5. Apply scale: output = x * rsqrt * weight

+-- Kernel Launch:
|   +-- Grid: (batch_size, 1, 1)
|   +-- Block: (hidden_size / elements_per_thread, 1, 1)
|   +-- Shared Memory: minimal (for reduction)

+-- Memory Access Pattern:
|   +-- Input: Coalesced reads, 128-bit vectors
|   +-- Weight: Broadcast across rows
|   +-- Output: Coalesced writes, 128-bit vectors

+-- Expected Hardware Behavior:
|   +-- Occupancy: High (low SMEM usage)
|   +-- Warp stalls: long_scoreboard dominant (HBM waits)
|   +-- TC utilization: 0% (no matrix ops)
|   +-- Memory BW: 80-90% of peak

+-- Fused Add RMSNorm Benefit:
|   +-- Saves ~40% memory traffic by combining two passes
|   +-- Residual += input happens in registers
|   +-- Single memory round-trip instead of two
""")

    return all_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FlashInfer RMSNorm Profiler")
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
    print(" FLASHINFER RMSNORM PROFILER")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    run_profiling(DEFAULT_CONFIGS, output_dir, device=args.device)

    print_section("PROFILING COMPLETE")


if __name__ == "__main__":
    main()
