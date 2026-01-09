#!/usr/bin/env python3
"""
Store Kernel Profiling Script

Profiles the Mini-SGLang Store kernel for KV cache scatter operations.
Analyzes K,V copy patterns, scatter behavior, and async copy opportunities.

Kernel Location: python/minisgl/kernel/csrc/jit/store.cu

Expected Analysis Format:
    STORE KERNEL ANALYSIS:
    ├── Operation: Scatter K,V to paged cache
    ├── Execution:
    │   ├── K copy: warps read contiguous, write scattered
    │   ├── V copy: same pattern, same kernel launch
    │   └── Total: 1 kernel launch for both K,V
    ├── Hardware:
    │   ├── Memory BW: X% per copy
    │   └── Observation: K,V handled in single pass
    ├── Optimization opportunity:
    │   ├── Current: contiguous read, scattered write
    │   └── Potential: async copy if K,V separated
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
class StoreKernelProfile:
    """Profile results for store kernel."""
    num_tokens: int
    head_dim: int  # per-head dimension (e.g., 128)
    num_kv_heads: int  # number of KV heads
    cache_size: int  # total cache slots
    dtype: str

    # Timing
    time_us: float = 0.0
    time_std_us: float = 0.0

    # Derived metrics
    k_bytes: int = 0
    v_bytes: int = 0
    total_bytes: int = 0
    achieved_bw_gbps: float = 0.0
    peak_bw_gbps: float = 2039.0  # A100 HBM2e
    bw_efficiency: float = 0.0

    # Kernel launch config
    num_warps: int = 0
    num_blocks: int = 0
    threads_per_block: int = 128

    # Access pattern analysis
    read_pattern: str = "contiguous"
    write_pattern: str = "scattered"

    # Execution story
    execution_story: List[str] = field(default_factory=list)


def profile_store_kernel(
    num_tokens: int,
    head_dim: int = 128,
    num_kv_heads: int = 8,
    cache_size: int = 4096,
    dtype: torch.dtype = torch.float16,
    num_warmup: int = 10,
    num_iterations: int = 100,
    scatter_pattern: str = "random",  # "random" or "sequential"
) -> StoreKernelProfile:
    """Profile the store kernel with given configuration."""
    from minisgl.kernel.store import store_cache

    # Flatten dimensions for the kernel: (num_slots, kv_heads * head_dim)
    kv_dim = num_kv_heads * head_dim

    # Create cache tensors
    k_cache = torch.zeros(cache_size, kv_dim, dtype=dtype, device="cuda")
    v_cache = torch.zeros(cache_size, kv_dim, dtype=dtype, device="cuda")

    # Create input tensors
    k = torch.randn(num_tokens, kv_dim, dtype=dtype, device="cuda")
    v = torch.randn(num_tokens, kv_dim, dtype=dtype, device="cuda")

    # Create indices (scatter pattern)
    if scatter_pattern == "random":
        indices = torch.randperm(cache_size, dtype=torch.int32, device="cuda")[:num_tokens]
    else:
        indices = torch.arange(num_tokens, dtype=torch.int32, device="cuda")

    # Calculate element size
    element_size = kv_dim * k.element_size()

    # Create profile object
    profile = StoreKernelProfile(
        num_tokens=num_tokens,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        cache_size=cache_size,
        dtype=str(dtype).split(".")[-1],
    )

    # Kernel config (from store.cu)
    profile.num_warps = num_tokens  # 1 warp per token
    profile.num_blocks = (profile.num_warps + 3) // 4  # 4 warps per block

    # Memory traffic: read K,V (contiguous) + write K,V (scattered)
    dtype_size = k.element_size()
    profile.k_bytes = num_tokens * kv_dim * dtype_size * 2  # read + write
    profile.v_bytes = num_tokens * kv_dim * dtype_size * 2  # read + write
    profile.total_bytes = profile.k_bytes + profile.v_bytes

    profile.write_pattern = scatter_pattern

    # Warmup
    for _ in range(num_warmup):
        store_cache(k_cache, v_cache, indices, k, v)
    cuda.synchronize()

    # Profile with CUDA events
    start_events = [cuda.Event(enable_timing=True) for _ in range(num_iterations)]
    end_events = [cuda.Event(enable_timing=True) for _ in range(num_iterations)]

    for i in range(num_iterations):
        start_events[i].record()
        store_cache(k_cache, v_cache, indices, k, v)
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


def build_execution_story(profile: StoreKernelProfile) -> List[str]:
    """Build the execution story for the store kernel."""
    story = []

    # Operation
    story.append(f"Operation: Scatter K,V for {profile.num_tokens} tokens to paged cache")

    # Execution details
    story.append("Execution:")
    story.append(f"  ├── {profile.num_warps} warps launched (1 per token)")
    story.append(f"  ├── Each warp handles both K and V for one token")

    # Per-warp copy details
    kv_dim = profile.num_kv_heads * profile.head_dim
    dtype_size = 2 if "float16" in profile.dtype or "bfloat16" in profile.dtype else 4
    bytes_per_kv = kv_dim * dtype_size

    # Vectorization analysis (from warp.cuh)
    if bytes_per_kv % (16 * 32) == 0:
        vec_width = "uint4 (16 bytes)"
        bytes_per_iter = 16 * 32
    elif bytes_per_kv % (8 * 32) == 0:
        vec_width = "uint2 (8 bytes)"
        bytes_per_iter = 8 * 32
    else:
        vec_width = "uint1 (4 bytes)"
        bytes_per_iter = 4 * 32

    num_iterations = bytes_per_kv // bytes_per_iter if bytes_per_iter > 0 else 1

    story.append(f"  ├── K copy: {num_iterations} vectorized loads/stores ({vec_width})")
    story.append(f"  ├── V copy: {num_iterations} vectorized loads/stores ({vec_width})")
    story.append(f"  ├── Per-token data: {bytes_per_kv * 2} bytes (K: {bytes_per_kv}, V: {bytes_per_kv})")
    story.append(f"  └── Total data movement: {profile.total_bytes / 1024:.1f} KB")

    # Access pattern analysis
    story.append("Access Pattern:")
    story.append(f"  ├── Read: Contiguous (K and V are packed)")
    story.append(f"  ├── Write: {profile.write_pattern.capitalize()} (indices determine positions)")
    if profile.write_pattern == "random":
        story.append("  └── Note: Random writes may have reduced coalescing")
    else:
        story.append("  └── Note: Sequential writes have perfect coalescing")

    # Hardware behavior
    story.append("Hardware:")
    story.append(f"  ├── Memory BW: {profile.bw_efficiency:.1f}% of peak ({profile.achieved_bw_gbps:.1f} GB/s)")

    # Efficiency analysis
    if profile.write_pattern == "random":
        story.append("  ├── Write coalescing: Degraded (scattered indices)")
        story.append("  └── Observation: BW affected by random write pattern")
    else:
        story.append("  ├── Write coalescing: Perfect (sequential indices)")
        story.append("  └── Observation: Near-optimal memory access")

    # Bottleneck and optimization
    story.append("Optimization Opportunities:")
    story.append("  ├── Current: Single kernel handles both K and V")
    story.append("  ├── Positive: No separate kernel launch overhead")
    if profile.bw_efficiency < 60:
        story.append("  └── Consider: Larger batch sizes for better BW utilization")
    else:
        story.append("  └── Status: Good efficiency, limited optimization needed")

    return story


def print_profile(profile: StoreKernelProfile) -> None:
    """Print profile results in the expected format."""
    print("\n" + "=" * 70)
    print("STORE KERNEL ANALYSIS")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Num Tokens: {profile.num_tokens}")
    print(f"  Head Dim: {profile.head_dim}")
    print(f"  Num KV Heads: {profile.num_kv_heads}")
    print(f"  Cache Size: {profile.cache_size}")
    print(f"  Data Type: {profile.dtype}")

    print(f"\nKernel Launch:")
    print(f"  Blocks: {profile.num_blocks}")
    print(f"  Threads/Block: {profile.threads_per_block}")
    print(f"  Total Warps: {profile.num_warps}")

    print(f"\nTiming:")
    print(f"  Latency: {profile.time_us:.2f} +/- {profile.time_std_us:.2f} us")

    print(f"\nExecution Story:")
    for line in profile.execution_story:
        print(f"  {line}")

    print()


def compare_scatter_patterns(
    num_tokens: int,
    head_dim: int = 128,
    num_kv_heads: int = 8,
) -> Dict[str, StoreKernelProfile]:
    """Compare random vs sequential scatter patterns."""
    print("\n" + "=" * 70)
    print("SCATTER PATTERN COMPARISON")
    print("=" * 70)

    results = {}

    for pattern in ["sequential", "random"]:
        print(f"\n--- Profiling: {pattern} scatter ---")
        profile = profile_store_kernel(
            num_tokens=num_tokens,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            scatter_pattern=pattern,
        )
        results[pattern] = profile

        print(f"  Time: {profile.time_us:.2f} us")
        print(f"  BW: {profile.achieved_bw_gbps:.1f} GB/s ({profile.bw_efficiency:.1f}%)")

    # Compare
    seq_bw = results["sequential"].achieved_bw_gbps
    rand_bw = results["random"].achieved_bw_gbps
    overhead = (results["random"].time_us - results["sequential"].time_us) / results["sequential"].time_us * 100

    print(f"\nComparison:")
    print(f"  Sequential BW: {seq_bw:.1f} GB/s")
    print(f"  Random BW: {rand_bw:.1f} GB/s")
    print(f"  Random overhead: {overhead:.1f}%")
    print(f"  Coalescing impact: {(seq_bw - rand_bw) / seq_bw * 100:.1f}% BW reduction")

    return results


def run_sweep(
    token_counts: List[int],
    head_dims: List[int],
    output_file: Optional[Path] = None,
) -> List[StoreKernelProfile]:
    """Run a sweep over different configurations."""
    results = []

    print("\n" + "=" * 70)
    print("STORE KERNEL SWEEP")
    print("=" * 70)

    for num_tokens in token_counts:
        for head_dim in head_dims:
            print(f"\n--- Profiling: tokens={num_tokens}, head_dim={head_dim} ---")
            profile = profile_store_kernel(num_tokens, head_dim)
            results.append(profile)

            print(f"  Time: {profile.time_us:.2f} us")
            print(f"  BW: {profile.achieved_bw_gbps:.1f} GB/s ({profile.bw_efficiency:.1f}%)")

    # Save results
    if output_file:
        data = {
            "profiles": [
                {
                    "num_tokens": p.num_tokens,
                    "head_dim": p.head_dim,
                    "num_kv_heads": p.num_kv_heads,
                    "time_us": p.time_us,
                    "achieved_bw_gbps": p.achieved_bw_gbps,
                    "bw_efficiency": p.bw_efficiency,
                }
                for p in results
            ]
        }
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Profile Mini-SGLang Store Kernel")
    parser.add_argument("--num-tokens", type=int, default=64, help="Number of tokens")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--num-kv-heads", type=int, default=8, help="Number of KV heads")
    parser.add_argument("--cache-size", type=int, default=4096, help="Cache size in tokens")
    parser.add_argument("--compare-patterns", action="store_true", help="Compare scatter patterns")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--output", type=str, default=None, help="Output file for results")
    args = parser.parse_args()

    if args.compare_patterns:
        results = compare_scatter_patterns(args.num_tokens, args.head_dim, args.num_kv_heads)
        if args.output:
            output_file = Path(args.output)
            with open(output_file, "w") as f:
                json.dump({
                    pattern: {
                        "time_us": p.time_us,
                        "achieved_bw_gbps": p.achieved_bw_gbps,
                        "bw_efficiency": p.bw_efficiency,
                    }
                    for pattern, p in results.items()
                }, f, indent=2)
    elif args.sweep:
        token_counts = [1, 8, 32, 64, 128, 256, 512]
        head_dims = [64, 128, 256]
        output_file = Path(args.output) if args.output else RESULTS_DIR / "store_sweep.json"
        run_sweep(token_counts, head_dims, output_file)
    else:
        profile = profile_store_kernel(
            args.num_tokens,
            args.head_dim,
            args.num_kv_heads,
            args.cache_size,
        )
        print_profile(profile)

        if args.output:
            output_file = Path(args.output)
        else:
            output_file = RESULTS_DIR / f"store_t{args.num_tokens}_h{args.head_dim}.json"

        with open(output_file, "w") as f:
            json.dump({
                "num_tokens": profile.num_tokens,
                "head_dim": profile.head_dim,
                "num_kv_heads": profile.num_kv_heads,
                "time_us": profile.time_us,
                "achieved_bw_gbps": profile.achieved_bw_gbps,
                "bw_efficiency": profile.bw_efficiency,
                "execution_story": profile.execution_story,
            }, f, indent=2)
        print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
