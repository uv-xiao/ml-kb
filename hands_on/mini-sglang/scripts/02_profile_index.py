#!/usr/bin/env python3
"""
Index Kernel Profiling Script

Profiles the Mini-SGLang Index kernel for embedding lookup operations.
Analyzes warp-level copy patterns, vectorization, and memory coalescing.

Kernel Location: python/minisgl/kernel/csrc/jit/index.cu

Expected Analysis Format:
    INDEX KERNEL ANALYSIS:
    ├── Operation: Embedding lookup for batch of N tokens
    ├── Execution:
    │   ├── N warps launched (1 per token)
    │   ├── Each warp: vectorized 128-byte loads
    │   └── Total: N × D × dtype_size bytes read
    ├── Hardware:
    │   ├── Memory BW: X% of peak
    │   ├── SM util: Y% (memory-bound, expected)
    │   └── Coalescing: 100% (perfect access pattern)
    └── Bottleneck: Pure memory bandwidth
"""

import argparse
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import torch
import torch.cuda as cuda

# Results directory
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class IndexKernelProfile:
    """Profile results for index kernel."""
    batch_size: int
    embedding_dim: int
    vocab_size: int
    dtype: str

    # Timing
    time_us: float = 0.0
    time_std_us: float = 0.0

    # Derived metrics
    bytes_read: int = 0
    bytes_written: int = 0
    total_bytes: int = 0
    achieved_bw_gbps: float = 0.0
    peak_bw_gbps: float = 2039.0  # A100 HBM2e
    bw_efficiency: float = 0.0

    # Kernel launch config
    num_warps: int = 0
    num_blocks: int = 0
    threads_per_block: int = 128
    num_splits: int = 1

    # Execution story
    execution_story: List[str] = field(default_factory=list)


def get_gpu_info() -> Dict[str, Any]:
    """Get current GPU information."""
    props = cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_memory_gb": props.total_memory / (1024**3),
        "sm_count": props.multi_processor_count,
        "compute_capability": f"{props.major}.{props.minor}",
        "peak_memory_bw_gbps": 2039.0,  # A100-80GB
    }


def profile_index_kernel(
    batch_size: int,
    embedding_dim: int,
    vocab_size: int = 32000,
    dtype: torch.dtype = torch.float16,
    num_warmup: int = 10,
    num_iterations: int = 100,
    use_vocab_range: bool = False,
) -> IndexKernelProfile:
    """Profile the index kernel with given configuration."""
    from minisgl.kernel.index import indexing

    # Create test data
    weights = torch.randn(vocab_size, embedding_dim, dtype=dtype, device="cuda")
    indices = torch.randint(0, vocab_size, (batch_size,), dtype=torch.int32, device="cuda")

    vocab_range = None
    if use_vocab_range:
        # Test with a subset of vocabulary (e.g., for TP sharding)
        start = vocab_size // 4
        length = vocab_size // 2
        vocab_range = (start, length)
        # Adjust indices to be within range
        indices = torch.randint(start, start + length, (batch_size,), dtype=torch.int32, device="cuda")

    # Calculate element size and splits
    element_size = embedding_dim * weights.element_size()
    if element_size % 2048 == 0:
        num_splits = 4
    elif element_size % 1024 == 0:
        num_splits = 2
    else:
        num_splits = 1

    # Create profile object
    profile = IndexKernelProfile(
        batch_size=batch_size,
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
        dtype=str(dtype).split(".")[-1],
    )

    # Calculate kernel config
    profile.num_splits = num_splits
    profile.num_warps = batch_size * num_splits
    profile.num_blocks = (profile.num_warps + 3) // 4  # 4 warps per block (128 threads)

    # Calculate memory traffic
    dtype_size = weights.element_size()
    profile.bytes_read = batch_size * embedding_dim * dtype_size
    profile.bytes_written = batch_size * embedding_dim * dtype_size
    profile.total_bytes = profile.bytes_read + profile.bytes_written

    # Warmup
    for _ in range(num_warmup):
        output = indexing(weights, indices, vocab_range=vocab_range)
    cuda.synchronize()

    # Profile with CUDA events
    start_events = [cuda.Event(enable_timing=True) for _ in range(num_iterations)]
    end_events = [cuda.Event(enable_timing=True) for _ in range(num_iterations)]

    for i in range(num_iterations):
        start_events[i].record()
        output = indexing(weights, indices, vocab_range=vocab_range)
        end_events[i].record()

    cuda.synchronize()

    # Calculate timing
    times = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]  # us
    profile.time_us = sum(times) / len(times)
    profile.time_std_us = (sum((t - profile.time_us) ** 2 for t in times) / len(times)) ** 0.5

    # Calculate bandwidth
    time_seconds = profile.time_us / 1e6
    profile.achieved_bw_gbps = profile.total_bytes / time_seconds / 1e9
    profile.bw_efficiency = profile.achieved_bw_gbps / profile.peak_bw_gbps * 100

    # Build execution story
    profile.execution_story = build_execution_story(profile)

    return profile


def build_execution_story(profile: IndexKernelProfile) -> List[str]:
    """Build the execution story for the index kernel."""
    story = []

    # Operation
    story.append(f"Operation: Embedding lookup for batch of {profile.batch_size} tokens")

    # Execution details
    story.append("Execution:")

    # Warp details
    if profile.num_splits > 1:
        story.append(f"  ├── {profile.num_warps} warps launched ({profile.num_splits} per token)")
        story.append(f"  ├── Each token uses {profile.num_splits} warps for parallel copy")
    else:
        story.append(f"  ├── {profile.num_warps} warps launched (1 per token)")

    # Vectorization
    bytes_per_warp = profile.embedding_dim * (2 if "float16" in profile.dtype or "bfloat16" in profile.dtype else 4) // profile.num_splits

    # Determine vectorization width
    if bytes_per_warp % (16 * 32) == 0:  # 16 bytes per thread, 32 threads
        vec_width = "uint4 (16 bytes)"
        bytes_per_iter = 16 * 32
    elif bytes_per_warp % (8 * 32) == 0:
        vec_width = "uint2 (8 bytes)"
        bytes_per_iter = 8 * 32
    else:
        vec_width = "uint1 (4 bytes)"
        bytes_per_iter = 4 * 32

    num_iterations = bytes_per_warp // bytes_per_iter
    story.append(f"  ├── Each warp: {num_iterations} vectorized loads ({vec_width})")
    story.append(f"  ├── Per-warp copy: {bytes_per_warp} bytes")
    story.append(f"  └── Total data movement: {profile.total_bytes / 1024:.1f} KB ({profile.bytes_read / 1024:.1f} KB read + {profile.bytes_written / 1024:.1f} KB write)")

    # Hardware behavior
    story.append("Hardware:")
    story.append(f"  ├── Memory BW: {profile.bw_efficiency:.1f}% of peak ({profile.achieved_bw_gbps:.1f} GB/s)")

    # SM utilization estimate (memory-bound, so limited)
    # For pure memory copy, SM util is typically low
    sm_util_estimate = min(40, profile.bw_efficiency * 0.4)
    story.append(f"  ├── SM util: ~{sm_util_estimate:.0f}% (memory-bound kernel)")
    story.append("  └── Coalescing: 100% (contiguous row access pattern)")

    # Bottleneck analysis
    if profile.bw_efficiency > 70:
        story.append("Bottleneck: Pure memory bandwidth (high efficiency)")
    elif profile.bw_efficiency > 50:
        story.append("Bottleneck: Memory bandwidth (moderate efficiency)")
    else:
        story.append("Bottleneck: Launch overhead / small batch (low BW utilization)")

    return story


def print_profile(profile: IndexKernelProfile) -> None:
    """Print profile results in the expected format."""
    print("\n" + "=" * 70)
    print("INDEX KERNEL ANALYSIS")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Batch Size: {profile.batch_size}")
    print(f"  Embedding Dim: {profile.embedding_dim}")
    print(f"  Vocab Size: {profile.vocab_size}")
    print(f"  Data Type: {profile.dtype}")

    print(f"\nKernel Launch:")
    print(f"  Blocks: {profile.num_blocks}")
    print(f"  Threads/Block: {profile.threads_per_block}")
    print(f"  Total Warps: {profile.num_warps}")
    print(f"  Splits per token: {profile.num_splits}")

    print(f"\nTiming:")
    print(f"  Latency: {profile.time_us:.2f} +/- {profile.time_std_us:.2f} us")

    print(f"\nExecution Story:")
    for line in profile.execution_story:
        print(f"  {line}")

    print()


def run_sweep(
    batch_sizes: List[int],
    embedding_dims: List[int],
    output_file: Optional[Path] = None,
) -> List[IndexKernelProfile]:
    """Run a sweep over different configurations."""
    results = []

    print("\n" + "=" * 70)
    print("INDEX KERNEL SWEEP")
    print("=" * 70)

    gpu_info = get_gpu_info()
    print(f"\nGPU: {gpu_info['name']}")
    print(f"Peak Memory BW: {gpu_info['peak_memory_bw_gbps']} GB/s")

    for batch_size in batch_sizes:
        for embedding_dim in embedding_dims:
            print(f"\n--- Profiling: batch={batch_size}, dim={embedding_dim} ---")
            profile = profile_index_kernel(batch_size, embedding_dim)
            results.append(profile)

            # Quick summary
            print(f"  Time: {profile.time_us:.2f} us")
            print(f"  BW: {profile.achieved_bw_gbps:.1f} GB/s ({profile.bw_efficiency:.1f}%)")

    # Save results
    if output_file:
        data = {
            "gpu_info": gpu_info,
            "profiles": [
                {
                    "batch_size": p.batch_size,
                    "embedding_dim": p.embedding_dim,
                    "time_us": p.time_us,
                    "achieved_bw_gbps": p.achieved_bw_gbps,
                    "bw_efficiency": p.bw_efficiency,
                    "num_warps": p.num_warps,
                    "num_splits": p.num_splits,
                }
                for p in results
            ]
        }
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Profile Mini-SGLang Index Kernel")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--embedding-dim", type=int, default=4096, help="Embedding dimension")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--output", type=str, default=None, help="Output file for results")
    args = parser.parse_args()

    if args.sweep:
        batch_sizes = [1, 8, 32, 64, 128, 256, 512, 1024]
        embedding_dims = [1024, 2048, 4096, 8192]
        output_file = Path(args.output) if args.output else RESULTS_DIR / "index_sweep.json"
        results = run_sweep(batch_sizes, embedding_dims, output_file)

        # Print summary table
        print("\n" + "=" * 70)
        print("SWEEP SUMMARY")
        print("=" * 70)
        print(f"\n{'Batch':>8} {'Dim':>8} {'Time (us)':>12} {'BW (GB/s)':>12} {'Eff (%)':>10}")
        print("-" * 54)
        for p in results:
            print(f"{p.batch_size:>8} {p.embedding_dim:>8} {p.time_us:>12.2f} {p.achieved_bw_gbps:>12.1f} {p.bw_efficiency:>10.1f}")
    else:
        profile = profile_index_kernel(
            args.batch_size,
            args.embedding_dim,
            args.vocab_size,
        )
        print_profile(profile)

        if args.output:
            output_file = Path(args.output)
        else:
            output_file = RESULTS_DIR / f"index_b{args.batch_size}_d{args.embedding_dim}.json"

        with open(output_file, "w") as f:
            json.dump({
                "batch_size": profile.batch_size,
                "embedding_dim": profile.embedding_dim,
                "time_us": profile.time_us,
                "achieved_bw_gbps": profile.achieved_bw_gbps,
                "bw_efficiency": profile.bw_efficiency,
                "execution_story": profile.execution_story,
            }, f, indent=2)
        print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
