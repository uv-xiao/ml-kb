#!/usr/bin/env python3
"""
SGLang Environment Detection Script
===================================

Detects GPU configuration, topology, and validates SGLang installation.
Outputs environment information for profiling and analysis.

Usage:
    python 00_check_env.py [--json] [--verbose]
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import platform


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    index: int
    name: str
    memory_total_mb: int
    compute_capability: str
    uuid: str
    pcie_bus_id: str
    numa_node: int


@dataclass
class TopologyInfo:
    """GPU topology information."""
    nvlink_pairs: List[Tuple[int, int]]
    pcie_pairs: List[Tuple[int, int]]
    sys_pairs: List[Tuple[int, int]]
    numa_mapping: Dict[int, List[int]]  # numa_node -> [gpu_indices]


@dataclass
class EnvironmentInfo:
    """Complete environment information."""
    hostname: str
    platform: str
    cuda_version: str
    driver_version: str
    python_version: str
    torch_version: str
    flashinfer_version: Optional[str]
    sglang_version: Optional[str]
    gpus: List[GPUInfo]
    topology: TopologyInfo
    recommended_tp_configs: Dict[str, List[int]]


def run_command(cmd: str) -> Tuple[int, str, str]:
    """Run a shell command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def get_gpu_info() -> List[GPUInfo]:
    """Get information about all GPUs."""
    gpus = []

    # Query GPU properties
    query = "index,name,memory.total,compute_cap,uuid,pci.bus_id"
    code, stdout, _ = run_command(f"nvidia-smi --query-gpu={query} --format=csv,noheader,nounits")

    if code != 0:
        return gpus

    for line in stdout.strip().split('\n'):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 6:
            gpu = GPUInfo(
                index=int(parts[0]),
                name=parts[1],
                memory_total_mb=int(float(parts[2])),
                compute_capability=parts[3],
                uuid=parts[4],
                pcie_bus_id=parts[5],
                numa_node=-1  # Will be filled later
            )
            gpus.append(gpu)

    # Get NUMA mapping
    for gpu in gpus:
        numa_path = f"/sys/bus/pci/devices/{gpu.pcie_bus_id.lower()}/numa_node"
        try:
            with open(numa_path, 'r') as f:
                gpu.numa_node = int(f.read().strip())
        except (FileNotFoundError, ValueError):
            gpu.numa_node = 0

    return gpus


def get_topology() -> Tuple[TopologyInfo, str]:
    """Get GPU topology information."""
    nvlink_pairs = []
    pcie_pairs = []
    sys_pairs = []
    numa_mapping: Dict[int, List[int]] = {}

    code, stdout, _ = run_command("nvidia-smi topo -m")
    raw_topology = stdout if code == 0 else ""

    if code == 0:
        lines = stdout.strip().split('\n')
        gpu_lines = []

        for line in lines:
            if line.startswith('GPU'):
                gpu_lines.append(line)

        # Parse topology matrix
        for line in gpu_lines:
            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                src_gpu = int(parts[0].replace('GPU', ''))
            except ValueError:
                continue

            for i, p in enumerate(parts[1:]):
                if p.startswith('NV'):
                    nvlink_pairs.append((src_gpu, i))
                elif p == 'PXB' or p == 'PIX':
                    pcie_pairs.append((src_gpu, i))
                elif p == 'SYS':
                    sys_pairs.append((src_gpu, i))

    # Remove duplicates and self-connections
    nvlink_pairs = list(set((min(a, b), max(a, b)) for a, b in nvlink_pairs if a != b))
    pcie_pairs = list(set((min(a, b), max(a, b)) for a, b in pcie_pairs if a != b))
    sys_pairs = list(set((min(a, b), max(a, b)) for a, b in sys_pairs if a != b))

    return TopologyInfo(
        nvlink_pairs=sorted(nvlink_pairs),
        pcie_pairs=sorted(pcie_pairs),
        sys_pairs=sorted(sys_pairs),
        numa_mapping=numa_mapping
    ), raw_topology


def get_software_versions() -> Dict[str, Optional[str]]:
    """Get versions of relevant software."""
    versions = {}

    # CUDA version
    code, stdout, _ = run_command("nvcc --version")
    if code == 0:
        for line in stdout.split('\n'):
            if 'release' in line.lower():
                parts = line.split('release')
                if len(parts) > 1:
                    versions['cuda'] = parts[1].split(',')[0].strip()
                    break
    else:
        # Try nvidia-smi
        code, stdout, _ = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
        versions['cuda'] = "Unknown"

    # Driver version
    code, stdout, _ = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
    versions['driver'] = stdout.strip().split('\n')[0] if code == 0 else "Unknown"

    # Python packages
    try:
        import torch
        versions['torch'] = torch.__version__
    except ImportError:
        versions['torch'] = None

    try:
        import flashinfer
        versions['flashinfer'] = flashinfer.__version__
    except ImportError:
        versions['flashinfer'] = None

    try:
        import sglang
        versions['sglang'] = sglang.__version__
    except ImportError:
        versions['sglang'] = None

    return versions


def recommend_tp_configs(gpus: List[GPUInfo], topology: TopologyInfo) -> Dict[str, List[int]]:
    """Recommend tensor parallelism configurations based on topology."""
    configs = {}
    num_gpus = len(gpus)

    # Build NUMA groups
    numa_groups: Dict[int, List[int]] = {}
    for gpu in gpus:
        if gpu.numa_node not in numa_groups:
            numa_groups[gpu.numa_node] = []
        numa_groups[gpu.numa_node].append(gpu.index)

    # NVLink pairs for TP=2
    if topology.nvlink_pairs:
        configs['tp2_nvlink'] = []
        for pair in topology.nvlink_pairs:
            configs['tp2_nvlink'].append(list(pair))

    # Same NUMA for latency-sensitive
    for numa_node, gpu_indices in numa_groups.items():
        if len(gpu_indices) >= 2:
            key = f'tp{len(gpu_indices)}_numa{numa_node}'
            configs[key] = sorted(gpu_indices)

    # Full TP configurations
    if num_gpus >= 4:
        # Prefer same NUMA node
        for numa_node, gpu_indices in numa_groups.items():
            if len(gpu_indices) >= 4:
                configs['tp4_recommended'] = sorted(gpu_indices[:4])
                break
        else:
            # Fall back to first 4 GPUs
            configs['tp4_fallback'] = list(range(4))

    return configs


def check_sglang_installation() -> Dict[str, bool]:
    """Check SGLang installation and dependencies."""
    checks = {}

    try:
        import torch
        checks['torch'] = True
        checks['cuda_available'] = torch.cuda.is_available()
        if checks['cuda_available']:
            checks['cuda_device_count'] = torch.cuda.device_count()
    except ImportError:
        checks['torch'] = False
        checks['cuda_available'] = False

    try:
        import sglang
        checks['sglang'] = True
    except ImportError:
        checks['sglang'] = False

    try:
        import flashinfer
        checks['flashinfer'] = True
    except ImportError:
        checks['flashinfer'] = False

    try:
        import triton
        checks['triton'] = True
    except ImportError:
        checks['triton'] = False

    return checks


def print_environment_report(env: EnvironmentInfo, raw_topology: str, verbose: bool = False):
    """Print a formatted environment report."""

    print("=" * 70)
    print("SGLANG ENVIRONMENT DETECTION REPORT")
    print("=" * 70)

    # System Info
    print("\n## System Information")
    print(f"  Hostname: {env.hostname}")
    print(f"  Platform: {env.platform}")
    print(f"  Python: {env.python_version}")

    # Software Versions
    print("\n## Software Versions")
    print(f"  CUDA: {env.cuda_version}")
    print(f"  Driver: {env.driver_version}")
    print(f"  PyTorch: {env.torch_version or 'Not installed'}")
    print(f"  FlashInfer: {env.flashinfer_version or 'Not installed'}")
    print(f"  SGLang: {env.sglang_version or 'Not installed'}")

    # GPU Configuration
    print("\n## GPU Configuration")
    print(f"  Total GPUs: {len(env.gpus)}")
    print()

    for gpu in env.gpus:
        pcie_type = "SXM4" if "SXM" in gpu.name.upper() or gpu.memory_total_mb > 80000 else "PCIe"
        print(f"  GPU {gpu.index}: {gpu.name}")
        print(f"    Memory: {gpu.memory_total_mb / 1024:.0f} GB")
        print(f"    Compute: SM {gpu.compute_capability}")
        print(f"    NUMA Node: {gpu.numa_node}")
        print()

    # Topology
    print("\n## GPU Topology")
    if verbose and raw_topology:
        print(raw_topology)

    print(f"\n  NVLink Pairs: {env.topology.nvlink_pairs}")
    print(f"  PCIe Pairs: {len(env.topology.pcie_pairs)} connections")
    print(f"  Cross-NUMA (SYS): {len(env.topology.sys_pairs)} connections")

    # Recommended Configurations
    print("\n## Recommended TP Configurations")
    for config_name, gpus in env.recommended_tp_configs.items():
        print(f"  {config_name}: GPUs {gpus}")

    # ASCII Topology Diagram
    print("\n## Topology Diagram")
    print_topology_diagram(env.gpus, env.topology)

    print("\n" + "=" * 70)


def print_topology_diagram(gpus: List[GPUInfo], topology: TopologyInfo):
    """Print an ASCII diagram of GPU topology."""
    num_gpus = len(gpus)

    # Build adjacency info
    nvlink_set = set(topology.nvlink_pairs)

    # Group by NUMA
    numa_groups: Dict[int, List[int]] = {}
    for gpu in gpus:
        if gpu.numa_node not in numa_groups:
            numa_groups[gpu.numa_node] = []
        numa_groups[gpu.numa_node].append(gpu.index)

    print()
    print("  GPU Interconnect Topology:")
    print("  " + "-" * 50)

    for numa_node, gpu_indices in sorted(numa_groups.items()):
        print(f"  NUMA Node {numa_node}:")

        # Print GPUs in this NUMA node
        gpu_strs = []
        for idx in gpu_indices:
            gpu = gpus[idx]
            mem_gb = gpu.memory_total_mb / 1024
            gpu_strs.append(f"[GPU{idx}:{mem_gb:.0f}GB]")

        print("    " + "  ".join(gpu_strs))

        # Print NVLink connections within this group
        for i, idx1 in enumerate(gpu_indices):
            for idx2 in gpu_indices[i+1:]:
                pair = (min(idx1, idx2), max(idx1, idx2))
                if pair in nvlink_set:
                    print(f"      GPU{idx1} <--NVLink--> GPU{idx2}")

        print()

    # Cross-NUMA connections
    if len(numa_groups) > 1:
        print("  Cross-NUMA Connections (SYS/PCIe):")
        for pair in topology.sys_pairs[:5]:  # Limit output
            print(f"    GPU{pair[0]} <--SYS--> GPU{pair[1]}")


def main():
    parser = argparse.ArgumentParser(description='SGLang Environment Detection')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    # Collect information
    gpus = get_gpu_info()
    topology, raw_topology = get_topology()
    versions = get_software_versions()

    # Build NUMA mapping
    for gpu in gpus:
        if gpu.numa_node not in topology.numa_mapping:
            topology.numa_mapping[gpu.numa_node] = []
        topology.numa_mapping[gpu.numa_node].append(gpu.index)

    # Recommend configurations
    recommended = recommend_tp_configs(gpus, topology)

    env = EnvironmentInfo(
        hostname=platform.node(),
        platform=platform.platform(),
        cuda_version=versions.get('cuda', 'Unknown'),
        driver_version=versions.get('driver', 'Unknown'),
        python_version=platform.python_version(),
        torch_version=versions.get('torch'),
        flashinfer_version=versions.get('flashinfer'),
        sglang_version=versions.get('sglang'),
        gpus=gpus,
        topology=topology,
        recommended_tp_configs=recommended
    )

    if args.json:
        # Convert to JSON-serializable format
        output = asdict(env)
        print(json.dumps(output, indent=2))
    else:
        print_environment_report(env, raw_topology, args.verbose)

        # Installation checks
        print("\n## Installation Checks")
        checks = check_sglang_installation()
        for check, result in checks.items():
            status = "OK" if result else "MISSING"
            print(f"  {check}: {status}")


if __name__ == "__main__":
    main()
