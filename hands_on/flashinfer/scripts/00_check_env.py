#!/usr/bin/env python3
"""
FlashInfer Environment Check Script

This script verifies the FlashInfer installation, JIT compilation system,
and GPU environment for hands-on learning.

Outputs:
- GPU topology and capabilities
- FlashInfer version and installation status
- JIT cache state and compilation readiness
- Basic kernel smoke tests
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================

def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}\n")


def check_cuda_environment():
    """Check CUDA installation and environment variables."""
    print_section("CUDA ENVIRONMENT")

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    print(f"CUDA_HOME: {cuda_home or 'Not set'}")

    # Check nvcc version
    import subprocess
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        nvcc_version = result.stdout.strip().split("\n")[-1]
        print(f"NVCC Version: {nvcc_version}")
    except FileNotFoundError:
        print("NVCC: Not found in PATH")

    # Check nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        driver_version = result.stdout.strip().split("\n")[0]
        print(f"NVIDIA Driver: {driver_version}")
    except FileNotFoundError:
        print("nvidia-smi: Not found")


def check_gpu_topology():
    """Check GPU topology and configuration."""
    print_section("GPU TOPOLOGY")

    import subprocess

    # Get GPU info
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True
        )
        gpus = result.stdout.strip().split("\n")

        print("Available GPUs:")
        for gpu in gpus:
            if gpu.strip():
                parts = gpu.split(",")
                idx, name, mem, cc = [p.strip() for p in parts]
                print(f"  GPU {idx}: {name}, {mem}, SM {cc}")

        print(f"\nTotal GPUs: {len(gpus)}")
    except Exception as e:
        print(f"Error querying GPUs: {e}")

    # Get topology matrix
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True, text=True
        )
        print("\nGPU Topology Matrix:")
        print(result.stdout)
    except Exception as e:
        print(f"Error querying topology: {e}")


def check_pytorch():
    """Check PyTorch installation and CUDA support."""
    print_section("PYTORCH ENVIRONMENT")

    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA Version (PyTorch): {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")

            # Check current device
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            print(f"\nCurrent Device: GPU {device}")
            print(f"  Name: {props.name}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  SM Count: {props.multi_processor_count}")

            # Memory info
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  Memory Allocated: {allocated:.2f} GB")
            print(f"  Memory Reserved: {reserved:.2f} GB")
    except ImportError:
        print("PyTorch: Not installed")
    except Exception as e:
        print(f"Error checking PyTorch: {e}")


def check_flashinfer():
    """Check FlashInfer installation and JIT system."""
    print_section("FLASHINFER INSTALLATION")

    try:
        import flashinfer
        print(f"FlashInfer Version: {flashinfer.__version__}")

        # Check JIT environment variables
        print("\nJIT Environment Variables:")
        jit_vars = [
            "FLASHINFER_WORKSPACE_BASE",
            "FLASHINFER_JIT_VERBOSE",
            "FLASHINFER_JIT_DEBUG",
            "FLASHINFER_LOGLEVEL",
            "FLASHINFER_CUDA_ARCH_LIST",
            "FLASHINFER_NVCC_THREADS",
        ]
        for var in jit_vars:
            value = os.environ.get(var, "(not set)")
            print(f"  {var}: {value}")

        # Check JIT directories
        print("\nJIT Directories:")
        from flashinfer.jit import env as jit_env

        print(f"  Workspace Dir: {jit_env.FLASHINFER_WORKSPACE_DIR}")
        print(f"  JIT Dir: {jit_env.FLASHINFER_JIT_DIR}")
        print(f"  Gen Source Dir: {jit_env.FLASHINFER_GEN_SRC_DIR}")
        print(f"  CSRC Dir: {jit_env.FLASHINFER_CSRC_DIR}")

        # Check if directories exist
        for name, path in [
            ("Workspace", jit_env.FLASHINFER_WORKSPACE_DIR),
            ("JIT", jit_env.FLASHINFER_JIT_DIR),
        ]:
            exists = path.exists() if hasattr(path, 'exists') else os.path.exists(path)
            print(f"  {name} exists: {exists}")

        return True
    except ImportError as e:
        print(f"FlashInfer: Not installed - {e}")
        return False
    except Exception as e:
        print(f"Error checking FlashInfer: {e}")
        return False


def check_jit_cache():
    """Check the state of the JIT cache."""
    print_section("JIT CACHE STATE")

    try:
        from flashinfer.jit import env as jit_env

        cache_dir = jit_env.FLASHINFER_JIT_DIR

        if not os.path.exists(cache_dir):
            print(f"JIT Cache Directory: {cache_dir}")
            print("Status: Directory does not exist (cold cache)")
            return

        print(f"JIT Cache Directory: {cache_dir}")

        # Count cached modules
        cached_ops_dir = cache_dir / "cached_ops" if hasattr(cache_dir, '__truediv__') else Path(cache_dir) / "cached_ops"

        if cached_ops_dir.exists():
            cached_modules = list(cached_ops_dir.glob("*/*.so"))
            print(f"Cached Modules: {len(cached_modules)}")

            # Show last few cached modules
            if cached_modules:
                print("\nRecently cached modules:")
                sorted_modules = sorted(cached_modules, key=lambda x: x.stat().st_mtime, reverse=True)
                for mod in sorted_modules[:5]:
                    mtime = datetime.fromtimestamp(mod.stat().st_mtime)
                    size_kb = mod.stat().st_size / 1024
                    print(f"  {mod.parent.name}: {size_kb:.1f} KB ({mtime.strftime('%Y-%m-%d %H:%M')})")
        else:
            print("Cached Modules: 0 (no cached_ops directory)")

        # Check generated sources
        gen_src_dir = jit_env.FLASHINFER_GEN_SRC_DIR
        if gen_src_dir.exists():
            gen_dirs = list(gen_src_dir.iterdir())
            print(f"\nGenerated Source Directories: {len(gen_dirs)}")

    except Exception as e:
        print(f"Error checking JIT cache: {e}")
        import traceback
        traceback.print_exc()


def test_kernel_compilation():
    """Test that kernels can be JIT compiled."""
    print_section("KERNEL COMPILATION TEST")

    try:
        import torch

        if not torch.cuda.is_available():
            print("Skipping: CUDA not available")
            return

        # Test RMSNorm (simple kernel, good first test)
        print("Testing RMSNorm kernel compilation...")
        start = time.time()

        import flashinfer

        batch_size = 4
        hidden_size = 4096

        input_tensor = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.float16)
        weight = torch.ones(hidden_size, device="cuda", dtype=torch.float16)

        # First call triggers JIT compilation
        output = flashinfer.rmsnorm(input_tensor, weight)
        torch.cuda.synchronize()

        compile_time = time.time() - start
        print(f"  RMSNorm first call (includes JIT): {compile_time*1000:.1f} ms")

        # Second call uses cached module
        start = time.time()
        for _ in range(10):
            output = flashinfer.rmsnorm(input_tensor, weight)
        torch.cuda.synchronize()

        cached_time = (time.time() - start) / 10
        print(f"  RMSNorm cached call: {cached_time*1000:.3f} ms")

        # Test RoPE kernel
        print("\nTesting RoPE kernel compilation...")
        start = time.time()

        nnz = 512
        num_heads = 32
        head_dim = 128

        q = torch.randn(nnz, num_heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(nnz, num_heads, head_dim, device="cuda", dtype=torch.float16)
        indptr = torch.tensor([0, nnz], device="cuda", dtype=torch.int32)
        offsets = torch.zeros(1, device="cuda", dtype=torch.int32)

        q_rope, k_rope = flashinfer.apply_rope(q, k, indptr, offsets)
        torch.cuda.synchronize()

        compile_time = time.time() - start
        print(f"  RoPE first call (includes JIT): {compile_time*1000:.1f} ms")

        print("\nKernel compilation tests: PASSED")

    except Exception as e:
        print(f"Kernel compilation test failed: {e}")
        import traceback
        traceback.print_exc()


def test_attention_wrappers():
    """Test attention wrapper initialization."""
    print_section("ATTENTION WRAPPER TEST")

    try:
        import torch
        import flashinfer

        if not torch.cuda.is_available():
            print("Skipping: CUDA not available")
            return

        # Test batch prefill wrapper
        print("Testing BatchPrefillWithPagedKVCacheWrapper...")
        start = time.time()

        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, "NHD"
        )

        init_time = time.time() - start
        print(f"  Wrapper initialization: {init_time*1000:.1f} ms")

        # Test batch decode wrapper
        print("\nTesting BatchDecodeWithPagedKVCacheWrapper...")
        start = time.time()

        decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer, "NHD"
        )

        init_time = time.time() - start
        print(f"  Wrapper initialization: {init_time*1000:.1f} ms")

        print("\nAttention wrapper tests: PASSED")

    except Exception as e:
        print(f"Attention wrapper test failed: {e}")
        import traceback
        traceback.print_exc()


def save_environment_info():
    """Save environment info to a JSON file for later reference."""
    print_section("SAVING ENVIRONMENT INFO")

    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
    }

    # CUDA info
    try:
        import torch
        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpus"] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["gpus"].append({
                    "index": i,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory_gb": props.total_memory / 1024**3,
                    "sm_count": props.multi_processor_count,
                })
    except:
        pass

    # FlashInfer info
    try:
        import flashinfer
        info["flashinfer_version"] = flashinfer.__version__
    except:
        pass

    # Save to file
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "environment_info.json"
    with open(output_file, "w") as f:
        json.dump(info, f, indent=2)

    print(f"Environment info saved to: {output_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    print("\n" + "="*70)
    print(" FLASHINFER ENVIRONMENT CHECK")
    print(" " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*70)

    check_cuda_environment()
    check_gpu_topology()
    check_pytorch()

    flashinfer_ok = check_flashinfer()

    if flashinfer_ok:
        check_jit_cache()
        test_kernel_compilation()
        test_attention_wrappers()

    save_environment_info()

    print_section("SUMMARY")
    print("Environment check complete.")
    print("\nNext steps:")
    print("  1. Review environment.md report")
    print("  2. Run profiling scripts (02_profile_prefill.py, etc.)")
    print("  3. Check JIT cache behavior with 06_jit_analysis.py")


if __name__ == "__main__":
    main()
