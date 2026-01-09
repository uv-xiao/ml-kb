#!/usr/bin/env python3
"""
FlashInfer JIT Compilation Analysis Script

This script analyzes the JIT compilation system to understand:
- Compilation triggers and timing
- Cache behavior (hit/miss patterns)
- Parameter space coverage
- Code generation patterns

Output provides insights into FlashInfer's JIT architecture.
"""

import os
import sys
import time
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Disable version check
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

import torch


# ============================================================================
# PROFILING UTILITIES
# ============================================================================

def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}\n")


# ============================================================================
# JIT SYSTEM ANALYSIS
# ============================================================================

class JITAnalyzer:
    """Analyzer for FlashInfer JIT compilation system."""

    def __init__(self, device: int = 0):
        self.device = torch.device(f"cuda:{device}")
        torch.cuda.set_device(self.device)

        # Get device compute capability
        props = torch.cuda.get_device_properties(self.device)
        self.compute_capability = f"{props.major}{props.minor}"
        self.sm_major = props.major
        self.sm_minor = props.minor

        # Import FlashInfer JIT internals
        import flashinfer
        self.flashinfer = flashinfer
        self.version = flashinfer.__version__

        from flashinfer.jit import env as jit_env
        self.jit_env = jit_env

        self.workspace_dir = jit_env.FLASHINFER_WORKSPACE_DIR
        self.jit_dir = jit_env.FLASHINFER_JIT_DIR
        self.gen_src_dir = jit_env.FLASHINFER_GEN_SRC_DIR

    def analyze_cache_state(self) -> Dict:
        """Analyze current state of JIT cache."""
        print_section("JIT CACHE STATE ANALYSIS")

        results = {
            "version": self.version,
            "compute_capability": self.compute_capability,
            "workspace_dir": str(self.workspace_dir),
            "jit_dir": str(self.jit_dir),
            "gen_src_dir": str(self.gen_src_dir),
        }

        # Check directory existence
        results["workspace_exists"] = self.workspace_dir.exists()
        results["jit_exists"] = self.jit_dir.exists()
        results["gen_src_exists"] = self.gen_src_dir.exists()

        print(f"FlashInfer Version: {self.version}")
        print(f"Compute Capability: SM {self.compute_capability}")
        print(f"Workspace Dir: {self.workspace_dir}")
        print(f"JIT Cache Dir: {self.jit_dir}")

        if not self.jit_dir.exists():
            print("\nJIT cache directory does not exist (cold cache)")
            results["cached_modules"] = 0
            results["generated_sources"] = 0
            return results

        # Count cached modules
        cached_modules = list(self.jit_dir.glob("*/*.so"))
        results["cached_modules"] = len(cached_modules)
        print(f"\nCached Modules: {len(cached_modules)}")

        if cached_modules:
            # Analyze cached modules
            module_info = []
            total_size = 0
            for so_file in cached_modules:
                size = so_file.stat().st_size
                mtime = datetime.fromtimestamp(so_file.stat().st_mtime)
                module_info.append({
                    "name": so_file.parent.name,
                    "size_kb": size / 1024,
                    "modified": mtime.isoformat(),
                })
                total_size += size

            results["total_cache_size_mb"] = total_size / 1e6
            results["modules"] = module_info[:20]  # First 20

            print(f"Total Cache Size: {total_size / 1e6:.1f} MB")
            print("\nRecent modules:")
            sorted_modules = sorted(module_info, key=lambda x: x["modified"], reverse=True)
            for mod in sorted_modules[:10]:
                print(f"  {mod['name'][:50]}: {mod['size_kb']:.1f} KB")

        # Count generated sources
        if self.gen_src_dir.exists():
            gen_dirs = list(self.gen_src_dir.iterdir())
            results["generated_sources"] = len(gen_dirs)
            print(f"\nGenerated Source Directories: {len(gen_dirs)}")

        return results

    def analyze_compilation_trigger(self, kernel_type: str = "rmsnorm") -> Dict:
        """Analyze what triggers JIT compilation for a kernel."""
        print_section(f"COMPILATION TRIGGER ANALYSIS ({kernel_type.upper()})")

        results = {"kernel_type": kernel_type, "triggers": []}

        if kernel_type == "rmsnorm":
            # RMSNorm has minimal parameters
            test_configs = [
                {"dtype": torch.float16, "desc": "FP16"},
                {"dtype": torch.bfloat16, "desc": "BF16"},
            ]

            for config in test_configs:
                dtype = config["dtype"]
                desc = config["desc"]

                # Create test tensors
                input_tensor = torch.randn(4, 4096, dtype=dtype, device=self.device)
                weight = torch.ones(4096, dtype=dtype, device=self.device)

                # Time first call (may include JIT)
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = self.flashinfer.rmsnorm(input_tensor, weight)
                torch.cuda.synchronize()
                first_call_ms = (time.perf_counter() - start) * 1000

                # Time second call (cached)
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = self.flashinfer.rmsnorm(input_tensor, weight)
                torch.cuda.synchronize()
                cached_call_ms = (time.perf_counter() - start) * 1000

                trigger_info = {
                    "config": desc,
                    "first_call_ms": first_call_ms,
                    "cached_call_ms": cached_call_ms,
                    "jit_overhead_ms": first_call_ms - cached_call_ms,
                    "likely_compiled": first_call_ms > 10,  # >10ms suggests JIT
                }
                results["triggers"].append(trigger_info)

                status = "JIT COMPILED" if trigger_info["likely_compiled"] else "CACHE HIT"
                print(f"{desc}: First call {first_call_ms:.1f}ms, Cached {cached_call_ms:.3f}ms [{status}]")

        elif kernel_type == "attention":
            # Attention has many parameters
            test_configs = [
                {"dtype_q": torch.float16, "dtype_kv": torch.float16, "head_dim": 128, "desc": "FP16, dim=128"},
                {"dtype_q": torch.float16, "dtype_kv": torch.float16, "head_dim": 64, "desc": "FP16, dim=64"},
                {"dtype_q": torch.bfloat16, "dtype_kv": torch.bfloat16, "head_dim": 128, "desc": "BF16, dim=128"},
            ]

            workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)

            for config in test_configs:
                dtype = config["dtype_q"]
                head_dim = config["head_dim"]
                desc = config["desc"]

                # Time wrapper creation and plan
                torch.cuda.synchronize()
                start = time.perf_counter()

                wrapper = self.flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")

                # Create minimal test case
                qo_indptr = torch.tensor([0, 64], dtype=torch.int32, device=self.device)
                paged_kv_indptr = torch.tensor([0, 4], dtype=torch.int32, device=self.device)
                paged_kv_indices = torch.arange(4, dtype=torch.int32, device=self.device)
                paged_kv_last_page_len = torch.tensor([16], dtype=torch.int32, device=self.device)

                wrapper.plan(
                    qo_indptr=qo_indptr,
                    paged_kv_indptr=paged_kv_indptr,
                    paged_kv_indices=paged_kv_indices,
                    paged_kv_last_page_len=paged_kv_last_page_len,
                    num_qo_heads=32,
                    num_kv_heads=32,
                    head_dim_qk=head_dim,
                    head_dim_vo=head_dim,
                    page_size=16,
                    q_data_type=dtype,
                    causal=True,
                )

                torch.cuda.synchronize()
                first_call_ms = (time.perf_counter() - start) * 1000

                trigger_info = {
                    "config": desc,
                    "init_and_plan_ms": first_call_ms,
                }
                results["triggers"].append(trigger_info)

                print(f"{desc}: Init+Plan {first_call_ms:.1f}ms")

        return results

    def analyze_parameter_space(self) -> Dict:
        """Analyze which parameter combinations trigger new compilations."""
        print_section("PARAMETER SPACE ANALYSIS")

        results = {"parameters": {}}

        print("""
FlashInfer JIT Compilation Parameters:
======================================

Each unique combination of these parameters creates a new compiled module:

ATTENTION KERNELS:
+-- dtype_q: Query data type (float16, bfloat16, float8)
+-- dtype_kv: KV cache data type (float16, bfloat16, float8)
+-- dtype_o: Output data type
+-- head_dim_qk: Query/Key head dimension (64, 128, 256)
+-- head_dim_vo: Value/Output head dimension
+-- pos_encoding_mode: Position encoding (None=0, RoPE=1, ALiBi=2)
+-- use_sliding_window: Sliding window attention flag
+-- use_logits_soft_cap: Logits capping flag
+-- mask_mode: Mask mode for causal/custom masks

NORMALIZATION KERNELS:
+-- dtype: Data type (float16, bfloat16)
+-- hidden_size: Hidden dimension (affects thread config)

ROPE KERNELS:
+-- dtype: Data type
+-- head_dim: Head dimension
+-- rotary_dim: Rotary dimension
+-- interleave: Layout mode

ESTIMATED COMPILATION MATRIX:
+-- Attention: ~100-500 variants (dtype x head_dim x features)
+-- RMSNorm: ~2-4 variants (dtype)
+-- RoPE: ~4-8 variants (dtype x head_dim)
""")

        # Count current variants
        if self.jit_dir.exists():
            modules = list(self.jit_dir.glob("*/*.so"))

            # Categorize by kernel type
            categories = {
                "prefill": 0,
                "decode": 0,
                "norm": 0,
                "rope": 0,
                "other": 0,
            }

            for mod in modules:
                name = mod.parent.name.lower()
                if "prefill" in name:
                    categories["prefill"] += 1
                elif "decode" in name:
                    categories["decode"] += 1
                elif "norm" in name:
                    categories["norm"] += 1
                elif "rope" in name:
                    categories["rope"] += 1
                else:
                    categories["other"] += 1

            results["variants_by_type"] = categories
            print("\nCurrent cached variants by type:")
            for cat, count in categories.items():
                print(f"  {cat}: {count}")

        return results

    def analyze_code_generation(self) -> Dict:
        """Analyze generated code patterns."""
        print_section("CODE GENERATION ANALYSIS")

        results = {"generated_files": []}

        if not self.gen_src_dir.exists():
            print("No generated sources found (cache not populated)")
            return results

        # Find and analyze generated files
        gen_dirs = list(self.gen_src_dir.iterdir())[:5]  # First 5

        print(f"Analyzing {len(gen_dirs)} generated source directories...\n")

        for gen_dir in gen_dirs:
            if not gen_dir.is_dir():
                continue

            dir_info = {
                "name": gen_dir.name[:50],
                "files": [],
            }

            for f in gen_dir.iterdir():
                if f.is_file():
                    dir_info["files"].append(f.name)

            if dir_info["files"]:
                results["generated_files"].append(dir_info)
                print(f"Directory: {dir_info['name'][:40]}...")
                for fname in dir_info["files"][:5]:
                    print(f"  - {fname}")

        # Explain code generation flow
        print("""
Code Generation Flow:
=====================

1. API CALL (e.g., flashinfer.rmsnorm())
   |
   v
2. COMPUTE URI (hash of parameters)
   +-- dtype, head_dim, features, etc.
   +-- SHA256 hash -> unique identifier
   |
   v
3. CHECK CACHE (~/.cache/flashinfer/<version>/<sm>/cached_ops/<uri>/)
   +-- If .so exists and source hash matches: LOAD
   +-- If missing or outdated: COMPILE
   |
   v
4. GENERATE SOURCE (if needed)
   +-- Copy template files from package data
   +-- Render Jinja templates with parameters
   +-- Write to generated/<uri>/
   |
   v
5. COMPILE WITH NINJA
   +-- Generate build.ninja
   +-- nvcc -gencode arch=compute_<SM>,code=sm_<SM>
   +-- Link to .so
   |
   v
6. LOAD VIA TVM-FFI
   +-- Dynamic loading of .so
   +-- Register functions
   +-- Cache in Python functools.cache
""")

        return results


def run_analysis(output_dir: Path, device: int = 0):
    """Run complete JIT analysis."""
    analyzer = JITAnalyzer(device=device)

    all_results = {}

    # Cache state analysis
    all_results["cache_state"] = analyzer.analyze_cache_state()

    # Compilation triggers
    all_results["rmsnorm_triggers"] = analyzer.analyze_compilation_trigger("rmsnorm")
    all_results["attention_triggers"] = analyzer.analyze_compilation_trigger("attention")

    # Parameter space
    all_results["parameter_space"] = analyzer.analyze_parameter_space()

    # Code generation
    all_results["code_generation"] = analyzer.analyze_code_generation()

    # Save results
    output_file = output_dir / "jit_analysis.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    # Print summary
    print_section("JIT SYSTEM SUMMARY")
    print(f"""
FlashInfer JIT Compilation System:
==================================

+-- Architecture: Two-level caching
|   +-- Python functools.cache: In-memory module references
|   +-- Disk cache: Compiled .so files in ~/.cache/flashinfer/

+-- Cache Location: {analyzer.workspace_dir}

+-- Cache Invalidation:
|   +-- Source hash change (modified .cuh files)
|   +-- Compilation flags change
|   +-- FlashInfer version change
|   +-- CUDA architecture change

+-- Development Tips:
|   +-- Clear cache: rm -rf ~/.cache/flashinfer/
|   +-- Verbose logging: export FLASHINFER_JIT_VERBOSE=1
|   +-- Debug builds: export FLASHINFER_JIT_DEBUG=1
|   +-- Custom arch: export FLASHINFER_CUDA_ARCH_LIST="8.0"

+-- Performance Characteristics:
|   +-- First call: 10-500ms (includes JIT compilation)
|   +-- Cached calls: <1ms
|   +-- Disk cache hit: <10ms (load .so)
|   +-- Memory cache hit: <0.1ms (Python cache)
""")

    return all_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FlashInfer JIT Analysis")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--clear-cache", action="store_true", help="Clear JIT cache before analysis")
    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*70)
    print(" FLASHINFER JIT COMPILATION ANALYSIS")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    if args.clear_cache:
        print("\n*** Clearing JIT cache for fresh analysis ***")
        import flashinfer
        from flashinfer.jit import env as jit_env
        if jit_env.FLASHINFER_WORKSPACE_DIR.exists():
            shutil.rmtree(jit_env.FLASHINFER_WORKSPACE_DIR)
            print(f"Cleared: {jit_env.FLASHINFER_WORKSPACE_DIR}")

    run_analysis(output_dir, device=args.device)

    print_section("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
