#!/usr/bin/env python3
"""
Attention Backend Profiling Script

Profiles FlashInfer/FlashAttention integration in Mini-SGLang.
Analyzes prefill vs decode kernel selection and performance characteristics.

Kernel Integration: python/minisgl/attention/

Expected Analysis Format:
    ATTENTION ANALYSIS:
    ├── Phase: Prefill/Decode
    ├── Backend: FlashInfer/FlashAttention
    ├── Execution:
    │   ├── Kernel: flash_attention_fwd / flash_decode_fwd
    │   ├── Work distribution: heads x tiles
    │   └── Warp specialization: TMA producer + TC consumers
    ├── Hardware:
    │   ├── Occupancy: X% (SMEM limited)
    │   ├── TC utilization: Y%
    │   ├── HBM BW: Z%
    │   └── Warp stalls: breakdown
    └── Roofline position: compute/memory bound
"""

import argparse
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.cuda as cuda

# Results directory
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class AttentionProfile:
    """Profile results for attention operations."""
    phase: str  # "prefill" or "decode"
    backend: str  # "flashinfer" or "flashattention"

    batch_size: int
    seq_len: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: str

    # Timing
    time_us: float = 0.0
    time_std_us: float = 0.0

    # Derived metrics
    flops: int = 0
    bytes_accessed: int = 0
    achieved_tflops: float = 0.0
    achieved_bw_gbps: float = 0.0
    arithmetic_intensity: float = 0.0

    # Hardware constants
    peak_tflops: float = 312.0  # A100 FP16 Tensor Core
    peak_bw_gbps: float = 2039.0  # A100 HBM2e

    # Roofline analysis
    roofline_bound: str = "unknown"
    efficiency: float = 0.0

    # Execution story
    execution_story: List[str] = field(default_factory=list)


def estimate_attention_flops(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    is_prefill: bool,
) -> int:
    """Estimate FLOPs for attention computation."""
    if is_prefill:
        # Q @ K^T: [B, H, S, D] x [B, H, D, S] = [B, H, S, S]
        # 2 * B * H * S * D * S (matmul FLOPs)
        qk_flops = 2 * batch_size * num_heads * seq_len * head_dim * seq_len
        # Softmax: approximately 5 * B * H * S * S
        softmax_flops = 5 * batch_size * num_heads * seq_len * seq_len
        # Attn @ V: [B, H, S, S] x [B, H, S, D] = [B, H, S, D]
        # 2 * B * H * S * S * D
        av_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
        return qk_flops + softmax_flops + av_flops
    else:
        # Decode: single query token
        # Q @ K^T: [B, H, 1, D] x [B, H, D, S] = [B, H, 1, S]
        qk_flops = 2 * batch_size * num_heads * head_dim * seq_len
        # Softmax: 5 * B * H * S
        softmax_flops = 5 * batch_size * num_heads * seq_len
        # Attn @ V: [B, H, 1, S] x [B, H, S, D] = [B, H, 1, D]
        av_flops = 2 * batch_size * num_heads * seq_len * head_dim
        return qk_flops + softmax_flops + av_flops


def estimate_attention_bytes(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_size: int,
    is_prefill: bool,
) -> int:
    """Estimate memory bytes accessed for attention."""
    if is_prefill:
        # Read Q, K, V
        q_bytes = batch_size * num_heads * seq_len * head_dim * dtype_size
        k_bytes = batch_size * num_kv_heads * seq_len * head_dim * dtype_size
        v_bytes = batch_size * num_kv_heads * seq_len * head_dim * dtype_size
        # Write O
        o_bytes = batch_size * num_heads * seq_len * head_dim * dtype_size
        return q_bytes + k_bytes + v_bytes + o_bytes
    else:
        # Decode: read all K,V cache, single Q
        q_bytes = batch_size * num_heads * 1 * head_dim * dtype_size
        k_bytes = batch_size * num_kv_heads * seq_len * head_dim * dtype_size
        v_bytes = batch_size * num_kv_heads * seq_len * head_dim * dtype_size
        o_bytes = batch_size * num_heads * 1 * head_dim * dtype_size
        return q_bytes + k_bytes + v_bytes + o_bytes


def profile_attention(
    phase: str,
    batch_size: int,
    seq_len: int,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    dtype: torch.dtype = torch.float16,
    backend: str = "flashinfer",
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> AttentionProfile:
    """Profile attention with given configuration."""

    is_prefill = (phase == "prefill")

    profile = AttentionProfile(
        phase=phase,
        backend=backend,
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=str(dtype).split(".")[-1],
    )

    # Create tensors based on phase
    if is_prefill:
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    else:
        # Decode: single query token, full KV cache
        q = torch.randn(batch_size, 1, num_heads, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")

    # Calculate theoretical metrics
    dtype_size = q.element_size()
    profile.flops = estimate_attention_flops(batch_size, seq_len, num_heads, head_dim, is_prefill)
    profile.bytes_accessed = estimate_attention_bytes(
        batch_size, seq_len, num_heads, num_kv_heads, head_dim, dtype_size, is_prefill
    )
    profile.arithmetic_intensity = profile.flops / profile.bytes_accessed

    # Attention function based on backend
    if backend == "flashinfer":
        try:
            import flashinfer
            if is_prefill:
                # Use single_prefill_with_kv_cache for simplicity
                def attn_fn():
                    # Reshape for FlashInfer: (B*S, H, D)
                    q_fi = q.view(-1, num_heads, head_dim)
                    k_fi = k.view(-1, num_kv_heads, head_dim)
                    v_fi = v.view(-1, num_kv_heads, head_dim)
                    return flashinfer.single_prefill_with_kv_cache(q_fi, k_fi, v_fi)
            else:
                def attn_fn():
                    q_fi = q.view(-1, num_heads, head_dim)
                    k_fi = k.view(-1, num_kv_heads, head_dim)
                    v_fi = v.view(-1, num_kv_heads, head_dim)
                    return flashinfer.single_decode_with_kv_cache(q_fi, k_fi, v_fi)
        except ImportError:
            print("FlashInfer not available, using PyTorch SDPA")
            def attn_fn():
                return torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                )
    else:  # flashattention
        try:
            from flash_attn import flash_attn_func
            def attn_fn():
                return flash_attn_func(q, k, v)
        except ImportError:
            print("FlashAttention not available, using PyTorch SDPA")
            def attn_fn():
                return torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                )

    # Warmup
    for _ in range(num_warmup):
        output = attn_fn()
    cuda.synchronize()

    # Profile with CUDA events
    start_events = [cuda.Event(enable_timing=True) for _ in range(num_iterations)]
    end_events = [cuda.Event(enable_timing=True) for _ in range(num_iterations)]

    for i in range(num_iterations):
        start_events[i].record()
        output = attn_fn()
        end_events[i].record()

    cuda.synchronize()

    # Calculate timing
    times = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]  # us
    profile.time_us = sum(times) / len(times)
    profile.time_std_us = (sum((t - profile.time_us) ** 2 for t in times) / len(times)) ** 0.5

    # Calculate achieved performance
    time_seconds = profile.time_us / 1e6
    profile.achieved_tflops = profile.flops / time_seconds / 1e12
    profile.achieved_bw_gbps = profile.bytes_accessed / time_seconds / 1e9

    # Roofline analysis
    ridge_point = profile.peak_tflops * 1e12 / (profile.peak_bw_gbps * 1e9)  # FLOPS/Byte
    if profile.arithmetic_intensity < ridge_point:
        profile.roofline_bound = "memory"
        profile.efficiency = profile.achieved_bw_gbps / profile.peak_bw_gbps * 100
    else:
        profile.roofline_bound = "compute"
        profile.efficiency = profile.achieved_tflops / profile.peak_tflops * 100

    # Build execution story
    profile.execution_story = build_execution_story(profile, is_prefill)

    return profile


def build_execution_story(profile: AttentionProfile, is_prefill: bool) -> List[str]:
    """Build the execution story for attention."""
    story = []

    # Phase and backend
    story.append(f"Phase: {profile.phase.capitalize()}")
    story.append(f"Backend: {profile.backend.capitalize()}")

    # Configuration
    story.append("Configuration:")
    story.append(f"  ├── Batch: {profile.batch_size}, Seq: {profile.seq_len}")
    story.append(f"  ├── Heads: {profile.num_heads} (Q) / {profile.num_kv_heads} (KV)")
    story.append(f"  └── Head dim: {profile.head_dim}")

    # Execution
    story.append("Execution:")
    if is_prefill:
        # Tile calculation (approximate FlashAttention-2 style)
        tile_q = min(128, profile.seq_len)
        tile_kv = min(128, profile.seq_len)
        num_q_tiles = (profile.seq_len + tile_q - 1) // tile_q
        num_kv_tiles = (profile.seq_len + tile_kv - 1) // tile_kv
        total_tiles = num_q_tiles * profile.num_heads * profile.batch_size

        story.append(f"  ├── Kernel: flash_attention_prefill")
        story.append(f"  ├── Q tiles: {num_q_tiles} (size {tile_q})")
        story.append(f"  ├── KV tiles: {num_kv_tiles} (size {tile_kv})")
        story.append(f"  ├── Total work items: {total_tiles}")
        story.append(f"  └── Algorithm: FlashAttention-2 with online softmax")
    else:
        # Decode uses split-K
        split_k = min(8, (profile.seq_len + 255) // 256)
        story.append(f"  ├── Kernel: flash_attention_decode")
        story.append(f"  ├── Split-K: {split_k} parallel chunks")
        story.append(f"  ├── Per-chunk KV: {profile.seq_len // split_k} tokens")
        story.append(f"  └── Algorithm: Split-K with LSE reduction")

    # Theoretical metrics
    story.append("Theoretical:")
    story.append(f"  ├── FLOPs: {profile.flops / 1e9:.2f} GFLOPs")
    story.append(f"  ├── Data: {profile.bytes_accessed / 1e6:.2f} MB")
    story.append(f"  └── Arithmetic Intensity: {profile.arithmetic_intensity:.1f} FLOP/Byte")

    # Hardware behavior
    story.append("Hardware:")
    story.append(f"  ├── Achieved: {profile.achieved_tflops:.2f} TFLOPs, {profile.achieved_bw_gbps:.1f} GB/s")
    story.append(f"  ├── Roofline: {profile.roofline_bound}-bound")
    story.append(f"  ├── Efficiency: {profile.efficiency:.1f}% of peak")

    # Expected hardware utilization
    if is_prefill:
        story.append("  ├── Expected SM util: 60-80% (high parallelism)")
        story.append("  ├── Expected TC util: 30-50% (mixed with softmax)")
        story.append("  └── Warp stalls: barrier (warp spec), long_scoreboard (HBM)")
    else:
        story.append("  ├── Expected SM util: 40-60% (less parallelism)")
        story.append("  ├── Expected TC util: 10-20% (GEMV-style)")
        story.append("  └── Warp stalls: long_scoreboard dominant (memory-bound)")

    return story


def print_profile(profile: AttentionProfile) -> None:
    """Print profile results in the expected format."""
    print("\n" + "=" * 70)
    print("ATTENTION ANALYSIS")
    print("=" * 70)

    print(f"\nExecution Story:")
    for line in profile.execution_story:
        print(f"  {line}")

    print(f"\nTiming: {profile.time_us:.2f} +/- {profile.time_std_us:.2f} us")
    print()


def compare_phases(
    seq_len: int,
    batch_size: int = 1,
    num_heads: int = 32,
    backend: str = "flashinfer",
) -> Dict[str, AttentionProfile]:
    """Compare prefill vs decode phases."""
    print("\n" + "=" * 70)
    print("PREFILL vs DECODE COMPARISON")
    print("=" * 70)

    results = {}

    for phase in ["prefill", "decode"]:
        print(f"\n--- Profiling: {phase} ---")
        profile = profile_attention(
            phase=phase,
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            backend=backend,
        )
        results[phase] = profile

        print(f"  Time: {profile.time_us:.2f} us")
        print(f"  TFLOPs: {profile.achieved_tflops:.2f}")
        print(f"  BW: {profile.achieved_bw_gbps:.1f} GB/s")
        print(f"  Bound: {profile.roofline_bound}")

    # Transition analysis
    print(f"\nPhase Transition Analysis:")
    print(f"  Prefill → Decode kernel switch")
    print(f"  Prefill: compute-oriented ({results['prefill'].roofline_bound}-bound)")
    print(f"  Decode: memory-oriented ({results['decode'].roofline_bound}-bound)")

    return results


def run_sweep(
    phase: str,
    seq_lens: List[int],
    batch_sizes: List[int],
    output_file: Optional[Path] = None,
) -> List[AttentionProfile]:
    """Run a sweep over different configurations."""
    results = []

    print("\n" + "=" * 70)
    print(f"ATTENTION {phase.upper()} SWEEP")
    print("=" * 70)

    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            print(f"\n--- Profiling: batch={batch_size}, seq={seq_len} ---")
            profile = profile_attention(phase=phase, batch_size=batch_size, seq_len=seq_len)
            results.append(profile)

            print(f"  Time: {profile.time_us:.2f} us")
            print(f"  TFLOPs: {profile.achieved_tflops:.2f}")
            print(f"  Bound: {profile.roofline_bound}")

    # Save results
    if output_file:
        data = {
            "phase": phase,
            "profiles": [
                {
                    "batch_size": p.batch_size,
                    "seq_len": p.seq_len,
                    "time_us": p.time_us,
                    "achieved_tflops": p.achieved_tflops,
                    "achieved_bw_gbps": p.achieved_bw_gbps,
                    "roofline_bound": p.roofline_bound,
                    "efficiency": p.efficiency,
                }
                for p in results
            ]
        }
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Profile Attention in Mini-SGLang")
    parser.add_argument("--phase", type=str, default="prefill", choices=["prefill", "decode"])
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--num-heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--num-kv-heads", type=int, default=8, help="Number of KV heads (GQA)")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--backend", type=str, default="flashinfer", choices=["flashinfer", "flashattention"])
    parser.add_argument("--compare", action="store_true", help="Compare prefill vs decode")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--output", type=str, default=None, help="Output file for results")
    args = parser.parse_args()

    if args.compare:
        results = compare_phases(args.seq_len, args.batch_size, args.num_heads, args.backend)
        if args.output:
            output_file = Path(args.output)
            with open(output_file, "w") as f:
                json.dump({
                    phase: {
                        "time_us": p.time_us,
                        "achieved_tflops": p.achieved_tflops,
                        "achieved_bw_gbps": p.achieved_bw_gbps,
                        "roofline_bound": p.roofline_bound,
                    }
                    for phase, p in results.items()
                }, f, indent=2)
    elif args.sweep:
        seq_lens = [128, 256, 512, 1024, 2048, 4096]
        batch_sizes = [1, 4, 8, 16]
        output_file = Path(args.output) if args.output else RESULTS_DIR / f"attention_{args.phase}_sweep.json"
        run_sweep(args.phase, seq_lens, batch_sizes, output_file)
    else:
        profile = profile_attention(
            phase=args.phase,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            backend=args.backend,
        )
        print_profile(profile)

        if args.output:
            output_file = Path(args.output)
        else:
            output_file = RESULTS_DIR / f"attention_{args.phase}_b{args.batch_size}_s{args.seq_len}.json"

        with open(output_file, "w") as f:
            json.dump({
                "phase": profile.phase,
                "backend": profile.backend,
                "batch_size": profile.batch_size,
                "seq_len": profile.seq_len,
                "time_us": profile.time_us,
                "achieved_tflops": profile.achieved_tflops,
                "achieved_bw_gbps": profile.achieved_bw_gbps,
                "roofline_bound": profile.roofline_bound,
                "execution_story": profile.execution_story,
            }, f, indent=2)
        print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
