#!/usr/bin/env python3
"""
Environment and Dependency Check for Mini-SGLang Hands-On Learning

This script verifies the environment is correctly configured for profiling
and analyzing Mini-SGLang kernels. It checks:
1. GPU configuration and topology
2. CUDA toolkit availability
3. Mini-SGLang dependencies (tvm-ffi, flashinfer, etc.)
4. JIT compilation readiness
5. Profiling tools (nsys, ncu)

Usage:
    python 00_check_env.py
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

# Output directory for results
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_command(cmd: str, timeout: int = 30) -> Tuple[bool, str]:
    """Run a shell command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def check_gpu_configuration() -> Dict[str, Any]:
    """Check GPU configuration and topology."""
    print("\n" + "=" * 60)
    print("GPU CONFIGURATION")
    print("=" * 60)

    info = {
        "gpus": [],
        "topology": None,
        "nvlink_pairs": [],
        "cuda_version": None,
        "driver_version": None,
    }

    # Check nvidia-smi
    success, output = run_command("nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader")
    if success:
        for line in output.strip().split("\n"):
            if line.strip():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpu = {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory": parts[2],
                        "compute_cap": parts[3],
                    }
                    info["gpus"].append(gpu)
                    print(f"  GPU {gpu['index']}: {gpu['name']} ({gpu['memory']}, CC {gpu['compute_cap']})")
    else:
        print("  ERROR: nvidia-smi not available")
        return info

    # Get driver and CUDA version
    success, output = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
    if success:
        info["driver_version"] = output.strip().split("\n")[0]
        print(f"\n  Driver Version: {info['driver_version']}")

    success, output = run_command("nvcc --version | grep 'release' | awk '{print $5}' | tr -d ','")
    if success:
        info["cuda_version"] = output.strip()
        print(f"  CUDA Version: {info['cuda_version']}")

    # Get GPU topology
    print("\nGPU Topology Matrix:")
    success, output = run_command("nvidia-smi topo -m")
    if success:
        info["topology"] = output
        # Parse NVLink pairs
        lines = output.strip().split("\n")
        for line in lines:
            if "NV" in line:
                parts = line.split()
                if parts:
                    gpu_idx = parts[0].replace("GPU", "")
                    for i, p in enumerate(parts[1:]):
                        if "NV" in p:
                            info["nvlink_pairs"].append((gpu_idx, str(i)))
        print(output)

    return info


def check_cuda_toolkit() -> Dict[str, Any]:
    """Check CUDA toolkit installation."""
    print("\n" + "=" * 60)
    print("CUDA TOOLKIT")
    print("=" * 60)

    info = {
        "nvcc": False,
        "nvcc_version": None,
        "cuda_home": None,
    }

    # Check nvcc
    success, output = run_command("nvcc --version")
    if success:
        info["nvcc"] = True
        for line in output.split("\n"):
            if "release" in line:
                info["nvcc_version"] = line.strip()
        print(f"  nvcc: OK ({info['nvcc_version']})")
    else:
        print("  nvcc: NOT FOUND")

    # Check CUDA_HOME
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        info["cuda_home"] = cuda_home
        print(f"  CUDA_HOME: {cuda_home}")
    else:
        # Try to find it
        for path in ["/usr/local/cuda", "/opt/cuda"]:
            if Path(path).exists():
                info["cuda_home"] = path
                print(f"  CUDA_HOME (auto-detected): {path}")
                break
        else:
            print("  CUDA_HOME: NOT SET (may cause JIT compilation issues)")

    return info


def check_python_dependencies() -> Dict[str, Any]:
    """Check Python dependencies."""
    print("\n" + "=" * 60)
    print("PYTHON DEPENDENCIES")
    print("=" * 60)

    info = {"packages": {}}

    packages = [
        ("torch", "PyTorch"),
        ("tvm_ffi", "TVM-FFI (kernel JIT)"),
        ("flashinfer", "FlashInfer"),
        ("flash_attn", "FlashAttention"),
        ("minisgl", "Mini-SGLang"),
    ]

    for pkg, desc in packages:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
            info["packages"][pkg] = {"available": True, "version": version}
            print(f"  {desc}: OK (v{version})")
        except ImportError as e:
            info["packages"][pkg] = {"available": False, "error": str(e)}
            print(f"  {desc}: NOT FOUND - {e}")

    return info


def check_jit_status() -> Dict[str, Any]:
    """Check JIT compilation status for Mini-SGLang kernels."""
    print("\n" + "=" * 60)
    print("JIT COMPILATION STATUS")
    print("=" * 60)

    info = {
        "jit_cache_dir": None,
        "kernels_cached": [],
        "jit_ready": False,
    }

    # Check tvm-ffi cache directory
    try:
        import tvm_ffi
        cache_dir = Path.home() / ".cache" / "tvm_ffi"
        if cache_dir.exists():
            info["jit_cache_dir"] = str(cache_dir)
            print(f"  JIT Cache Directory: {cache_dir}")

            # List cached kernels
            for item in cache_dir.iterdir():
                if item.is_dir() and "minisgl" in item.name:
                    info["kernels_cached"].append(item.name)

            if info["kernels_cached"]:
                print(f"  Cached Kernels: {len(info['kernels_cached'])}")
                for k in info["kernels_cached"][:5]:
                    print(f"    - {k}")
                if len(info["kernels_cached"]) > 5:
                    print(f"    ... and {len(info['kernels_cached']) - 5} more")
        else:
            print(f"  JIT Cache Directory: {cache_dir} (not yet created)")

        info["jit_ready"] = True
        print("  JIT Ready: YES")

    except ImportError:
        print("  JIT Ready: NO (tvm_ffi not installed)")

    return info


def check_profiling_tools() -> Dict[str, Any]:
    """Check availability of profiling tools."""
    print("\n" + "=" * 60)
    print("PROFILING TOOLS")
    print("=" * 60)

    info = {
        "nsys": False,
        "ncu": False,
        "nsys_version": None,
        "ncu_version": None,
    }

    # Check nsys
    success, output = run_command("nsys --version")
    if success:
        info["nsys"] = True
        info["nsys_version"] = output.strip().split("\n")[0]
        print(f"  nsys (Nsight Systems): OK ({info['nsys_version']})")
    else:
        print("  nsys (Nsight Systems): NOT FOUND")

    # Check ncu
    success, output = run_command("ncu --version")
    if success:
        info["ncu"] = True
        info["ncu_version"] = output.strip().split("\n")[0]
        print(f"  ncu (Nsight Compute): OK ({info['ncu_version']})")
    else:
        print("  ncu (Nsight Compute): NOT FOUND")

    return info


def test_kernel_compilation() -> Dict[str, Any]:
    """Test that kernels can be compiled."""
    print("\n" + "=" * 60)
    print("KERNEL COMPILATION TEST")
    print("=" * 60)

    info = {
        "index_kernel": False,
        "store_kernel": False,
        "nccl_kernel": False,
        "radix_kernel": False,
    }

    try:
        import torch

        # Test index kernel
        try:
            from minisgl.kernel.index import indexing
            # Create test tensors
            weights = torch.randn(1000, 128, dtype=torch.float16, device="cuda")
            indices = torch.randint(0, 1000, (10,), dtype=torch.int32, device="cuda")
            output = indexing(weights, indices)
            info["index_kernel"] = True
            print("  Index Kernel: OK (JIT compiled and tested)")
        except Exception as e:
            print(f"  Index Kernel: FAILED - {e}")

        # Test store kernel
        try:
            from minisgl.kernel.store import store_cache
            k_cache = torch.zeros(100, 128, dtype=torch.float16, device="cuda")
            v_cache = torch.zeros(100, 128, dtype=torch.float16, device="cuda")
            indices = torch.arange(10, dtype=torch.int32, device="cuda")
            k = torch.randn(10, 128, dtype=torch.float16, device="cuda")
            v = torch.randn(10, 128, dtype=torch.float16, device="cuda")
            store_cache(k_cache, v_cache, indices, k, v)
            info["store_kernel"] = True
            print("  Store Kernel: OK (JIT compiled and tested)")
        except Exception as e:
            print(f"  Store Kernel: FAILED - {e}")

        # Test radix kernel (CPU)
        try:
            from minisgl.kernel.radix import fast_compare_key
            x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
            y = torch.tensor([1, 2, 3, 6, 7], dtype=torch.int32)
            result = fast_compare_key(x, y)
            assert result == 3, f"Expected 3, got {result}"
            info["radix_kernel"] = True
            print("  Radix Kernel: OK (AOT compiled and tested)")
        except Exception as e:
            print(f"  Radix Kernel: FAILED - {e}")

        # NCCL requires multi-GPU setup, just check if module loads
        try:
            from minisgl.kernel.pynccl import _load_nccl_module
            _load_nccl_module()
            info["nccl_kernel"] = True
            print("  NCCL Kernel: OK (AOT compiled, multi-GPU init not tested)")
        except Exception as e:
            print(f"  NCCL Kernel: FAILED - {e}")

    except ImportError as e:
        print(f"  Cannot test kernels: {e}")

    return info


def generate_summary(all_info: Dict[str, Any]) -> None:
    """Generate a summary report."""
    print("\n" + "=" * 60)
    print("ENVIRONMENT SUMMARY")
    print("=" * 60)

    gpu_info = all_info.get("gpu", {})
    cuda_info = all_info.get("cuda", {})
    deps_info = all_info.get("deps", {})
    jit_info = all_info.get("jit", {})
    tools_info = all_info.get("tools", {})
    kernel_info = all_info.get("kernels", {})

    # GPU Summary
    gpus = gpu_info.get("gpus", [])
    if gpus:
        print(f"\n  GPUs: {len(gpus)}x {gpus[0]['name']}")
        print(f"  Total Memory: {sum(int(g['memory'].replace(' MiB', '')) for g in gpus) / 1024:.1f} GB")
        print(f"  Compute Capability: {gpus[0]['compute_cap']}")

    # Readiness check
    print("\n  Readiness Status:")

    ready = True

    # Check essential components
    if cuda_info.get("nvcc"):
        print("    [OK] CUDA Toolkit")
    else:
        print("    [FAIL] CUDA Toolkit - Required for JIT compilation")
        ready = False

    pkg_info = deps_info.get("packages", {})
    if pkg_info.get("minisgl", {}).get("available"):
        print("    [OK] Mini-SGLang")
    else:
        print("    [FAIL] Mini-SGLang - Required")
        ready = False

    if pkg_info.get("tvm_ffi", {}).get("available"):
        print("    [OK] TVM-FFI (JIT)")
    else:
        print("    [FAIL] TVM-FFI - Required for kernel JIT")
        ready = False

    if tools_info.get("nsys"):
        print("    [OK] Nsight Systems")
    else:
        print("    [WARN] Nsight Systems - Recommended for profiling")

    if tools_info.get("ncu"):
        print("    [OK] Nsight Compute")
    else:
        print("    [WARN] Nsight Compute - Recommended for kernel analysis")

    # Kernel status
    kernels_ok = sum(1 for v in kernel_info.values() if v)
    print(f"    [{'OK' if kernels_ok == 4 else 'WARN'}] Kernels: {kernels_ok}/4 tested successfully")

    print("\n" + "-" * 60)
    if ready:
        print("  Environment is READY for Mini-SGLang hands-on learning!")
    else:
        print("  Environment has ISSUES. Please resolve them before proceeding.")
    print("-" * 60)


def main():
    """Main entry point."""
    print("=" * 60)
    print("Mini-SGLang Hands-On Learning - Environment Check")
    print("=" * 60)

    all_info = {}

    # Run all checks
    all_info["gpu"] = check_gpu_configuration()
    all_info["cuda"] = check_cuda_toolkit()
    all_info["deps"] = check_python_dependencies()
    all_info["jit"] = check_jit_status()
    all_info["tools"] = check_profiling_tools()
    all_info["kernels"] = test_kernel_compilation()

    # Generate summary
    generate_summary(all_info)

    # Save results to file
    import json
    output_file = RESULTS_DIR / "environment_check.json"
    with open(output_file, "w") as f:
        # Convert non-serializable items
        serializable = {}
        for k, v in all_info.items():
            if isinstance(v, dict):
                serializable[k] = {
                    kk: str(vv) if not isinstance(vv, (dict, list, str, int, float, bool, type(None))) else vv
                    for kk, vv in v.items()
                }
            else:
                serializable[k] = str(v)
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
