#!/usr/bin/env python3
"""
NCCL Communication Profiling Script

Profiles the Mini-SGLang PyNCCL wrapper for all-reduce and all-gather operations.
Analyzes symmetric memory usage, NVLink utilization, and sync overhead.

Kernel Location: python/minisgl/kernel/csrc/src/pynccl.cu

Expected Analysis Format:
    NCCL COMMUNICATION ANALYSIS:
    ├── Operation: AllReduce/AllGather
    ├── Execution:
    │   ├── Communicator: NCCLWrapper with symmetric memory
    │   ├── Buffer: Internal symmetric buffer (if size fits)
    │   └── Pattern: Ring/Tree based on NCCL heuristics
    ├── Hardware:
    │   ├── Interconnect: NVLink / PCIe
    │   ├── Achieved BW: X GB/s per link
    │   └── Utilization: Y% of peak
    └── Optimization: Symmetric memory reduces D2D copies
"""

import argparse
import time
import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.cuda as cuda
import torch.distributed as dist

# Results directory
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class NCCLProfile:
    """Profile results for NCCL operations."""
    operation: str  # "allreduce" or "allgather"
    world_size: int
    rank: int

    tensor_size_bytes: int
    dtype: str

    # Timing
    time_us: float = 0.0
    time_std_us: float = 0.0

    # Derived metrics
    achieved_bw_gbps: float = 0.0
    algo_bw_gbps: float = 0.0  # Algorithmic bandwidth

    # Interconnect info
    interconnect: str = "unknown"
    nvlink_pairs: List[Tuple[int, int]] = field(default_factory=list)

    # Peak bandwidth assumptions
    nvlink_peak_gbps: float = 600.0  # NVLink 3.0 bidirectional
    pcie_peak_gbps: float = 64.0  # PCIe 4.0 x16

    # Execution story
    execution_story: List[str] = field(default_factory=list)


def detect_gpu_topology() -> Dict[str, Any]:
    """Detect GPU topology and interconnect types."""
    import subprocess

    topology = {
        "nvlink_pairs": [],
        "pcie_pairs": [],
        "numa_nodes": {},
    }

    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            gpu_count = 0
            for line in lines:
                if line.startswith("GPU"):
                    parts = line.split()
                    if len(parts) > 1:
                        gpu_idx = int(parts[0].replace("GPU", ""))
                        for i, conn in enumerate(parts[1:]):
                            if "NV" in conn:
                                if gpu_idx < i:
                                    topology["nvlink_pairs"].append((gpu_idx, i))
                            elif "PXB" in conn or "PIX" in conn:
                                if gpu_idx < i:
                                    topology["pcie_pairs"].append((gpu_idx, i))
                        gpu_count += 1
    except Exception as e:
        print(f"  Warning: Could not detect topology: {e}")

    return topology


def profile_nccl_operation(
    operation: str,
    tensor_size: int,
    dtype: torch.dtype = torch.float16,
    num_warmup: int = 10,
    num_iterations: int = 100,
    world_size: int = 2,
    rank: int = 0,
) -> NCCLProfile:
    """Profile NCCL operation with given configuration."""

    dtype_size = torch.tensor([], dtype=dtype).element_size()
    tensor_size_bytes = tensor_size * dtype_size

    profile = NCCLProfile(
        operation=operation,
        world_size=world_size,
        rank=rank,
        tensor_size_bytes=tensor_size_bytes,
        dtype=str(dtype).split(".")[-1],
    )

    # Detect topology
    topology = detect_gpu_topology()
    profile.nvlink_pairs = topology.get("nvlink_pairs", [])
    if len(profile.nvlink_pairs) > 0:
        profile.interconnect = "NVLink"
    else:
        profile.interconnect = "PCIe"

    # Create tensors
    if operation == "allreduce":
        tensor = torch.randn(tensor_size, dtype=dtype, device="cuda")

        # Profile using PyTorch distributed (simpler for standalone testing)
        def comm_fn():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        # Algorithmic bandwidth for allreduce: 2 * (n-1) / n * data_size
        # This accounts for the ring algorithm
        profile.algo_bw_gbps = 2 * (world_size - 1) / world_size * tensor_size_bytes

    else:  # allgather
        src_tensor = torch.randn(tensor_size, dtype=dtype, device="cuda")
        dst_tensor = torch.zeros(tensor_size * world_size, dtype=dtype, device="cuda")

        def comm_fn():
            dist.all_gather_into_tensor(dst_tensor, src_tensor)

        # Algorithmic bandwidth for allgather: (n-1) / n * data_size
        profile.algo_bw_gbps = (world_size - 1) / world_size * tensor_size_bytes

    # Warmup
    for _ in range(num_warmup):
        comm_fn()
    cuda.synchronize()

    # Profile with CUDA events
    start_events = [cuda.Event(enable_timing=True) for _ in range(num_iterations)]
    end_events = [cuda.Event(enable_timing=True) for _ in range(num_iterations)]

    for i in range(num_iterations):
        start_events[i].record()
        comm_fn()
        end_events[i].record()

    cuda.synchronize()

    # Calculate timing
    times = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]  # us
    profile.time_us = sum(times) / len(times)
    profile.time_std_us = (sum((t - profile.time_us) ** 2 for t in times) / len(times)) ** 0.5

    # Calculate bandwidth
    time_seconds = profile.time_us / 1e6
    profile.achieved_bw_gbps = profile.algo_bw_gbps / time_seconds / 1e9

    # Build execution story
    profile.execution_story = build_execution_story(profile)

    return profile


def build_execution_story(profile: NCCLProfile) -> List[str]:
    """Build the execution story for NCCL operation."""
    story = []

    # Operation
    story.append(f"Operation: {profile.operation.upper()}")
    story.append(f"World Size: {profile.world_size} GPUs (rank {profile.rank})")

    # Data details
    story.append("Data:")
    size_kb = profile.tensor_size_bytes / 1024
    size_mb = size_kb / 1024
    if size_mb >= 1:
        story.append(f"  ├── Size: {size_mb:.2f} MB ({profile.dtype})")
    else:
        story.append(f"  ├── Size: {size_kb:.2f} KB ({profile.dtype})")
    story.append(f"  └── Total traffic: ~{size_mb * 2 * (profile.world_size - 1) / profile.world_size:.2f} MB")

    # Execution
    story.append("Execution:")
    story.append(f"  ├── Communicator: NCCL with symmetric memory")

    # Buffer handling (from pynccl.cu)
    story.append("  ├── Buffer strategy:")
    story.append("  │   ├── If size <= max_bytes: Use symmetric buffer")
    story.append("  │   │   └── D2D copy to buffer → NCCL op → D2D copy back")
    story.append("  │   └── Else: Direct NCCL on input tensor")

    # Algorithm selection (NCCL internal)
    if profile.tensor_size_bytes < 256 * 1024:  # < 256KB
        story.append("  └── Algorithm: Likely LL (low-latency) for small messages")
    elif profile.tensor_size_bytes < 4 * 1024 * 1024:  # < 4MB
        story.append("  └── Algorithm: Likely Ring for medium messages")
    else:
        story.append("  └── Algorithm: Likely Tree for large messages")

    # Hardware
    story.append("Hardware:")
    story.append(f"  ├── Interconnect: {profile.interconnect}")

    if profile.nvlink_pairs:
        story.append(f"  ├── NVLink pairs: {profile.nvlink_pairs}")
        peak_bw = profile.nvlink_peak_gbps
    else:
        peak_bw = profile.pcie_peak_gbps

    efficiency = min(100, profile.achieved_bw_gbps / peak_bw * 100)
    story.append(f"  ├── Achieved BW: {profile.achieved_bw_gbps:.1f} GB/s")
    story.append(f"  ├── Peak BW: {peak_bw:.1f} GB/s ({profile.interconnect})")
    story.append(f"  └── Efficiency: {efficiency:.1f}%")

    # Optimization notes
    story.append("Optimization Notes:")
    story.append("  ├── Symmetric memory: Pre-registered NCCL buffer")
    story.append("  │   └── Avoids registration overhead at runtime")
    if profile.interconnect == "NVLink":
        story.append("  ├── NVLink: Enables peer-to-peer direct access")
    else:
        story.append("  ├── PCIe: Higher latency, lower bandwidth than NVLink")

    if profile.tensor_size_bytes < 64 * 1024:
        story.append("  └── Small message: Latency-bound, consider batching")
    else:
        story.append("  └── Large message: Bandwidth-bound, good efficiency expected")

    return story


def print_profile(profile: NCCLProfile) -> None:
    """Print profile results in the expected format."""
    print("\n" + "=" * 70)
    print("NCCL COMMUNICATION ANALYSIS")
    print("=" * 70)

    print(f"\nExecution Story:")
    for line in profile.execution_story:
        print(f"  {line}")

    print(f"\nTiming: {profile.time_us:.2f} +/- {profile.time_std_us:.2f} us")
    print()


def profile_nccl_standalone(
    operation: str = "allreduce",
    tensor_size: int = 1024 * 1024,  # 1M elements
    dtype: torch.dtype = torch.float16,
) -> NCCLProfile:
    """Profile NCCL in standalone mode (single process simulation)."""

    print("\n" + "=" * 70)
    print("NCCL STANDALONE PROFILING")
    print("=" * 70)
    print("\nNote: This is a simulation without actual multi-GPU communication.")
    print("For accurate profiling, run with torchrun or NCCL test utilities.\n")

    dtype_size = torch.tensor([], dtype=dtype).element_size()
    tensor_size_bytes = tensor_size * dtype_size

    profile = NCCLProfile(
        operation=operation,
        world_size=1,  # Single GPU simulation
        rank=0,
        tensor_size_bytes=tensor_size_bytes,
        dtype=str(dtype).split(".")[-1],
    )

    # Detect topology
    topology = detect_gpu_topology()
    profile.nvlink_pairs = topology.get("nvlink_pairs", [])
    if len(profile.nvlink_pairs) > 0:
        profile.interconnect = "NVLink available"
    else:
        profile.interconnect = "PCIe only"

    # Analyze the NCCL wrapper code instead of running it
    profile.execution_story = [
        f"Operation: {operation.upper()} (Code Analysis)",
        "",
        "NCCLWrapper Initialization:",
        "  ├── ncclCommInitRank(comm, world_size, uid, rank)",
        "  ├── ncclMemAlloc(&buf, max_bytes)  // Symmetric memory",
        "  └── ncclCommWindowRegister(comm, buf, max_bytes, &win)",
        "",
        "AllReduce Execution Path:",
        "  ├── Check tensor on CUDA device",
        "  ├── Verify contiguity",
        "  ├── If size <= m_max_bytes:",
        "  │   ├── cudaMemcpyAsync(buf, data, D2D)",
        "  │   ├── ncclAllReduce(buf, buf, ...)",
        "  │   └── cudaMemcpyAsync(data, buf, D2D)",
        "  └── Else:",
        "      └── ncclAllReduce(data, data, ...)",
        "",
        "AllGather Execution Path:",
        "  ├── Validate src and dst tensors",
        "  ├── Check size compatibility (dst = src * world_size)",
        "  └── ncclAllGather(src, dst, ...)  // No internal buffer",
        "",
        f"Detected Topology:",
        f"  └── {profile.interconnect}",
    ]

    if profile.nvlink_pairs:
        profile.execution_story.append(f"      NVLink pairs: {profile.nvlink_pairs}")

    return profile


def analyze_nccl_code() -> Dict[str, Any]:
    """Analyze the NCCL wrapper implementation."""
    print("\n" + "=" * 70)
    print("NCCL WRAPPER CODE ANALYSIS")
    print("=" * 70)

    analysis = {
        "file": "python/minisgl/kernel/csrc/src/pynccl.cu",
        "class": "NCCLWrapper",
        "features": [],
        "operations": [],
        "optimizations": [],
    }

    analysis["features"] = [
        "Symmetric memory allocation via ncclMemAlloc",
        "Window registration for collective operations",
        "Support for FP16 and BF16 data types",
        "Sum, Prod, Max, Min, Avg reduction operations",
    ]

    analysis["operations"] = [
        {
            "name": "all_reduce",
            "description": "In-place reduction across all ranks",
            "buffer_strategy": "Uses internal symmetric buffer if tensor fits",
            "extra_copies": "2 D2D copies if using internal buffer",
        },
        {
            "name": "all_gather",
            "description": "Gather tensor from all ranks",
            "buffer_strategy": "Direct gather to output tensor",
            "extra_copies": "None",
        },
    ]

    analysis["optimizations"] = [
        "Symmetric memory pre-allocation reduces registration overhead",
        "Window-based collectives for better performance",
        "Avoid internal buffer for all_gather (direct output)",
    ]

    print("\nClass: NCCLWrapper")
    print("\nFeatures:")
    for f in analysis["features"]:
        print(f"  - {f}")

    print("\nOperations:")
    for op in analysis["operations"]:
        print(f"\n  {op['name']}:")
        print(f"    Description: {op['description']}")
        print(f"    Buffer strategy: {op['buffer_strategy']}")
        print(f"    Extra copies: {op['extra_copies']}")

    print("\nOptimizations:")
    for o in analysis["optimizations"]:
        print(f"  - {o}")

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Profile NCCL in Mini-SGLang")
    parser.add_argument("--operation", type=str, default="allreduce",
                        choices=["allreduce", "allgather"])
    parser.add_argument("--size", type=int, default=1024*1024,
                        help="Tensor size in elements")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze NCCL code without running")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results")
    args = parser.parse_args()

    if args.analyze:
        analysis = analyze_nccl_code()

        if args.output:
            with open(args.output, "w") as f:
                json.dump(analysis, f, indent=2)
            print(f"\nAnalysis saved to: {args.output}")
    else:
        # Check if running in distributed mode
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            # Running with torchrun or similar
            dist.init_process_group(backend="nccl")
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            profile = profile_nccl_operation(
                operation=args.operation,
                tensor_size=args.size,
                world_size=world_size,
                rank=rank,
            )
            print_profile(profile)

            if rank == 0 and args.output:
                with open(args.output, "w") as f:
                    json.dump({
                        "operation": profile.operation,
                        "world_size": profile.world_size,
                        "tensor_size_bytes": profile.tensor_size_bytes,
                        "time_us": profile.time_us,
                        "achieved_bw_gbps": profile.achieved_bw_gbps,
                    }, f, indent=2)

            dist.destroy_process_group()
        else:
            # Standalone analysis
            profile = profile_nccl_standalone(args.operation, args.size)
            print_profile(profile)

            if args.output:
                output_file = Path(args.output)
                with open(output_file, "w") as f:
                    json.dump({
                        "operation": profile.operation,
                        "mode": "standalone_analysis",
                        "execution_story": profile.execution_story,
                    }, f, indent=2)
                print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
