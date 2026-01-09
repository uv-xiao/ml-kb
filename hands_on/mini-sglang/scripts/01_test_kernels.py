#!/usr/bin/env python3
"""
Kernel Correctness Testing Script

Tests all Mini-SGLang custom kernels for correctness:
1. Index Kernel - Embedding lookup
2. Store Kernel - KV cache scatter
3. Radix Kernel - Prefix matching (CPU)
4. NCCL Kernel - Communication (module load only)

This establishes a baseline before profiling.

Usage:
    python 01_test_kernels.py
"""

import sys
from pathlib import Path
from typing import Tuple, List, Optional
from dataclasses import dataclass

import torch
import torch.cuda as cuda

# Results directory
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class TestResult:
    """Result of a kernel test."""
    name: str
    passed: bool
    message: str
    details: Optional[str] = None


def test_index_kernel() -> TestResult:
    """Test the index kernel for embedding lookup."""
    try:
        from minisgl.kernel.index import indexing

        # Test parameters
        vocab_size = 1000
        embedding_dim = 128
        batch_size = 64

        # Create test data
        weights = torch.randn(vocab_size, embedding_dim, dtype=torch.float16, device="cuda")
        indices = torch.randint(0, vocab_size, (batch_size,), dtype=torch.int32, device="cuda")

        # Run kernel
        output = indexing(weights, indices)

        # Verify against PyTorch reference
        expected = weights[indices.long()]

        # Check correctness
        if torch.allclose(output, expected, rtol=1e-3, atol=1e-3):
            return TestResult(
                name="Index Kernel",
                passed=True,
                message="Embedding lookup matches PyTorch reference",
                details=f"batch={batch_size}, dim={embedding_dim}, vocab={vocab_size}"
            )
        else:
            max_diff = (output - expected).abs().max().item()
            return TestResult(
                name="Index Kernel",
                passed=False,
                message=f"Output mismatch (max diff: {max_diff:.6f})",
                details=f"batch={batch_size}, dim={embedding_dim}"
            )

    except Exception as e:
        return TestResult(
            name="Index Kernel",
            passed=False,
            message=f"Exception: {str(e)}",
        )


def test_index_kernel_with_mask() -> TestResult:
    """Test the masked index kernel for vocab range masking."""
    try:
        from minisgl.kernel.index import indexing

        # Test parameters
        vocab_size = 1000
        embedding_dim = 128
        batch_size = 32

        # Vocab range (simulating TP sharding)
        start = 200
        length = 500

        # Create test data
        weights = torch.randn(length, embedding_dim, dtype=torch.float16, device="cuda")
        indices = torch.randint(start, start + length, (batch_size,), dtype=torch.int32, device="cuda")

        # Run kernel with vocab_range
        output = indexing(weights, indices, vocab_range=(start, length))

        # Verify against PyTorch reference
        # indices in [start, start+length) should map to [0, length)
        adjusted_indices = indices.long() - start
        expected = weights[adjusted_indices]

        if torch.allclose(output, expected, rtol=1e-3, atol=1e-3):
            return TestResult(
                name="Index Kernel (Masked)",
                passed=True,
                message="Masked embedding lookup matches reference",
                details=f"range=[{start}, {start+length}), batch={batch_size}"
            )
        else:
            max_diff = (output - expected).abs().max().item()
            return TestResult(
                name="Index Kernel (Masked)",
                passed=False,
                message=f"Output mismatch (max diff: {max_diff:.6f})",
            )

    except Exception as e:
        return TestResult(
            name="Index Kernel (Masked)",
            passed=False,
            message=f"Exception: {str(e)}",
        )


def test_store_kernel() -> TestResult:
    """Test the store kernel for KV cache scatter."""
    try:
        from minisgl.kernel.store import store_cache

        # Test parameters
        cache_size = 1024
        kv_dim = 256  # num_kv_heads * head_dim
        num_tokens = 64

        # Create cache tensors
        k_cache = torch.zeros(cache_size, kv_dim, dtype=torch.float16, device="cuda")
        v_cache = torch.zeros(cache_size, kv_dim, dtype=torch.float16, device="cuda")

        # Create input tensors
        k = torch.randn(num_tokens, kv_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(num_tokens, kv_dim, dtype=torch.float16, device="cuda")

        # Random scatter indices
        indices = torch.randperm(cache_size, dtype=torch.int32, device="cuda")[:num_tokens]

        # Run kernel
        store_cache(k_cache, v_cache, indices, k, v)

        # Verify: check that scattered values match input
        k_expected = k_cache.clone()
        v_expected = v_cache.clone()
        k_expected[indices.long()] = k
        v_expected[indices.long()] = v

        k_match = torch.allclose(k_cache, k_expected, rtol=1e-3, atol=1e-3)
        v_match = torch.allclose(v_cache, v_expected, rtol=1e-3, atol=1e-3)

        # Actually, we need to check in a different way since we modified the cache
        # Let's just verify the scattered positions
        k_scattered = k_cache[indices.long()]
        v_scattered = v_cache[indices.long()]

        k_match = torch.allclose(k_scattered, k, rtol=1e-3, atol=1e-3)
        v_match = torch.allclose(v_scattered, v, rtol=1e-3, atol=1e-3)

        if k_match and v_match:
            return TestResult(
                name="Store Kernel",
                passed=True,
                message="KV cache scatter matches reference",
                details=f"tokens={num_tokens}, cache={cache_size}, dim={kv_dim}"
            )
        else:
            return TestResult(
                name="Store Kernel",
                passed=False,
                message=f"Mismatch: K={'OK' if k_match else 'FAIL'}, V={'OK' if v_match else 'FAIL'}",
            )

    except Exception as e:
        return TestResult(
            name="Store Kernel",
            passed=False,
            message=f"Exception: {str(e)}",
        )


def test_radix_kernel() -> TestResult:
    """Test the radix kernel for prefix matching."""
    try:
        from minisgl.kernel.radix import fast_compare_key

        # Test case 1: Partial match
        x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
        y = torch.tensor([1, 2, 3, 6, 7], dtype=torch.int32)
        result = fast_compare_key(x, y)

        if result != 3:
            return TestResult(
                name="Radix Kernel",
                passed=False,
                message=f"Partial match failed: expected 3, got {result}",
            )

        # Test case 2: Full match
        x = torch.tensor([1, 2, 3], dtype=torch.int32)
        y = torch.tensor([1, 2, 3], dtype=torch.int32)
        result = fast_compare_key(x, y)

        if result != 3:
            return TestResult(
                name="Radix Kernel",
                passed=False,
                message=f"Full match failed: expected 3, got {result}",
            )

        # Test case 3: No match
        x = torch.tensor([1, 2, 3], dtype=torch.int32)
        y = torch.tensor([4, 5, 6], dtype=torch.int32)
        result = fast_compare_key(x, y)

        if result != 0:
            return TestResult(
                name="Radix Kernel",
                passed=False,
                message=f"No match failed: expected 0, got {result}",
            )

        # Test case 4: int64
        x = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        y = torch.tensor([1, 2, 9, 10], dtype=torch.int64)
        result = fast_compare_key(x, y)

        if result != 2:
            return TestResult(
                name="Radix Kernel",
                passed=False,
                message=f"int64 match failed: expected 2, got {result}",
            )

        return TestResult(
            name="Radix Kernel",
            passed=True,
            message="All prefix matching cases passed",
            details="Tested: partial, full, no match, int64"
        )

    except Exception as e:
        return TestResult(
            name="Radix Kernel",
            passed=False,
            message=f"Exception: {str(e)}",
        )


def test_nccl_module() -> TestResult:
    """Test that NCCL module loads correctly."""
    try:
        from minisgl.kernel.pynccl import _load_nccl_module

        # Just try to load the module
        module = _load_nccl_module()

        # Check that create_nccl_uid function exists
        if hasattr(module, "create_nccl_uid"):
            return TestResult(
                name="NCCL Module",
                passed=True,
                message="Module loads successfully",
                details="create_nccl_uid function available"
            )
        else:
            return TestResult(
                name="NCCL Module",
                passed=False,
                message="create_nccl_uid function not found",
            )

    except Exception as e:
        return TestResult(
            name="NCCL Module",
            passed=False,
            message=f"Exception: {str(e)}",
        )


def run_all_tests() -> List[TestResult]:
    """Run all kernel tests."""
    print("=" * 60)
    print("MINI-SGLANG KERNEL CORRECTNESS TESTS")
    print("=" * 60)

    tests = [
        ("Index Kernel", test_index_kernel),
        ("Index Kernel (Masked)", test_index_kernel_with_mask),
        ("Store Kernel", test_store_kernel),
        ("Radix Kernel", test_radix_kernel),
        ("NCCL Module", test_nccl_module),
    ]

    results = []

    for name, test_fn in tests:
        print(f"\nTesting: {name}...")
        result = test_fn()
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.message}")
        if result.details:
            print(f"         {result.details}")

    return results


def print_summary(results: List[TestResult]) -> None:
    """Print test summary."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed tests:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.message}")

    print("\n" + "-" * 60)
    if failed == 0:
        print("All tests PASSED! Ready for profiling.")
    else:
        print(f"{failed} tests FAILED. Please fix before profiling.")
    print("-" * 60)


def main():
    """Main entry point."""
    results = run_all_tests()
    print_summary(results)

    # Save results
    import json
    output_file = RESULTS_DIR / "kernel_tests.json"
    with open(output_file, "w") as f:
        json.dump([
            {
                "name": r.name,
                "passed": r.passed,
                "message": r.message,
                "details": r.details,
            }
            for r in results
        ], f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Exit with appropriate code
    failed = sum(1 for r in results if not r.passed)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
