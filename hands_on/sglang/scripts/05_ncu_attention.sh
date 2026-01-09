#!/bin/bash
#
# SGLang Nsight Compute Attention Kernel Profiling
# =================================================
#
# Deep-dives into attention kernel performance.
#
# Usage:
#   ./05_ncu_attention.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/../results/ncu"

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "SGLANG NSIGHT COMPUTE PROFILING"
echo "========================================"
echo ""

# Check ncu availability
if ! command -v ncu &> /dev/null; then
    echo "ERROR: ncu (Nsight Compute) not found in PATH"
    echo "Please install NVIDIA Nsight Compute:"
    echo "  https://developer.nvidia.com/nsight-compute"
    exit 1
fi

# Create microbenchmark script
cat > "$OUTPUT_DIR/attention_microbench.py" << 'BENCHMARK_SCRIPT'
#!/usr/bin/env python3
"""
Attention Kernel Microbenchmark
"""

import torch
import time
import os

# Try to import flashinfer
try:
    import flashinfer
    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False
    print("WARNING: FlashInfer not available")

# Try to import triton attention
try:
    from sglang.srt.layers.attention.triton_ops.decode_attention import decode_attention_fwd
    from sglang.srt.layers.attention.triton_ops.extend_attention import extend_attention_fwd
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("WARNING: SGLang Triton attention not available")


def benchmark_flashinfer(batch_size=32, seq_len=2048, num_heads=32, head_dim=128, dtype=torch.float16):
    """Benchmark FlashInfer attention kernels."""
    if not HAS_FLASHINFER:
        return

    device = torch.device("cuda")

    # Prefill benchmark
    print(f"\n[FlashInfer] Prefill: batch={batch_size}, seq_len={seq_len}")

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)

    # Warmup
    for _ in range(3):
        out = flashinfer.single_prefill_with_kv_cache(q, k, v)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(10):
        out = flashinfer.single_prefill_with_kv_cache(q, k, v)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 10

    print(f"  Time: {elapsed*1000:.3f} ms")
    print(f"  Throughput: {batch_size * seq_len * num_heads * head_dim * 4 / elapsed / 1e12:.2f} TB/s (theoretical)")


def benchmark_decode(batch_size=32, kv_len=2048, num_heads=32, head_dim=128, dtype=torch.float16):
    """Benchmark decode attention (single query token)."""
    if not HAS_FLASHINFER:
        return

    device = torch.device("cuda")

    print(f"\n[FlashInfer] Decode: batch={batch_size}, kv_len={kv_len}")

    # Query is single token
    q = torch.randn(batch_size, 1, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, kv_len, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, kv_len, num_heads, head_dim, dtype=dtype, device=device)

    # Warmup
    for _ in range(3):
        out = flashinfer.single_decode_with_kv_cache(q.squeeze(1), k, v)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(10):
        out = flashinfer.single_decode_with_kv_cache(q.squeeze(1), k, v)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 10

    print(f"  Time: {elapsed*1000:.3f} ms")
    # Decode is memory-bound: read K, V
    bytes_read = batch_size * kv_len * num_heads * head_dim * 2 * 2  # K and V, FP16
    print(f"  Memory BW: {bytes_read / elapsed / 1e12:.2f} TB/s")


def main():
    print("=" * 60)
    print("ATTENTION KERNEL MICROBENCHMARK")
    print("=" * 60)

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    # CUDA profiler markers for ncu
    torch.cuda.cudart().cudaProfilerStart()

    configs = [
        # (batch_size, seq_len/kv_len)
        (1, 512),
        (1, 2048),
        (8, 512),
        (8, 2048),
        (32, 512),
        (32, 2048),
    ]

    for batch_size, seq_len in configs:
        print(f"\n{'='*60}")
        print(f"Config: batch={batch_size}, seq_len={seq_len}")
        print('='*60)

        # Prefill
        benchmark_flashinfer(
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=32,
            head_dim=128
        )

        # Decode
        benchmark_decode(
            batch_size=batch_size,
            kv_len=seq_len,
            num_heads=32,
            head_dim=128
        )

    torch.cuda.cudart().cudaProfilerStop()

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
BENCHMARK_SCRIPT

chmod +x "$OUTPUT_DIR/attention_microbench.py"

# Profile attention kernels
profile_attention() {
    echo "Profiling attention kernels..."
    echo ""

    ncu \
        --set full \
        --import-source yes \
        --kernel-regex ".*flash.*|.*attention.*|.*decode.*" \
        --launch-count 30 \
        --target-processes all \
        --force-overwrite \
        --export "$OUTPUT_DIR/attention_profile" \
        python3 "$OUTPUT_DIR/attention_microbench.py"

    echo ""
    echo "Profile saved: $OUTPUT_DIR/attention_profile.ncu-rep"
}

# Generate text report
generate_report() {
    echo ""
    echo "Generating analysis report..."

    # If profile exists, generate summary
    if [[ -f "$OUTPUT_DIR/attention_profile.ncu-rep" ]]; then
        ncu --import "$OUTPUT_DIR/attention_profile.ncu-rep" \
            --page details \
            > "$OUTPUT_DIR/attention_analysis.txt" 2>&1 || true

        echo "Report saved: $OUTPUT_DIR/attention_analysis.txt"
    fi
}

# Quick metrics only (no full set)
profile_quick() {
    echo "Quick attention kernel metrics..."
    echo ""

    ncu \
        --metrics gpu__time_duration.avg,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__sass_thread_inst_executed_op_ffma_pred_on.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sector_hit_rate.pct \
        --kernel-regex ".*flash.*|.*attention.*" \
        --launch-count 10 \
        python3 "$OUTPUT_DIR/attention_microbench.py"
}

# Main
echo "Select profiling mode:"
echo "  1. Full profile (comprehensive, slow)"
echo "  2. Quick metrics (key metrics only)"
echo ""
read -p "Choice [1/2]: " choice

case $choice in
    1)
        profile_attention
        generate_report
        ;;
    2)
        profile_quick
        ;;
    *)
        echo "Running quick profile by default..."
        profile_quick
        ;;
esac

echo ""
echo "========================================"
echo "ANALYSIS GUIDE"
echo "========================================"
echo ""
echo "Open the profile in Nsight Compute:"
echo "  ncu-ui $OUTPUT_DIR/attention_profile.ncu-rep"
echo ""
echo "Key Metrics to Examine:"
echo ""
echo "1. OCCUPANCY"
echo "   - Theoretical vs Achieved occupancy"
echo "   - Limiter (registers, shared memory, blocks)"
echo ""
echo "2. COMPUTE THROUGHPUT"
echo "   - SM utilization percentage"
echo "   - Tensor Core utilization"
echo "   - FP16/FP32 instruction mix"
echo ""
echo "3. MEMORY THROUGHPUT"
echo "   - DRAM (HBM) bandwidth utilization"
echo "   - L2 cache hit rate"
echo "   - Shared memory throughput"
echo ""
echo "4. WARP STALL REASONS"
echo "   - long_scoreboard: Waiting for memory"
echo "   - barrier: Synchronization"
echo "   - short_scoreboard: SMEM conflicts"
echo "   - not_selected: Low occupancy"
echo ""
echo "5. ROOFLINE ANALYSIS"
echo "   - Arithmetic Intensity"
echo "   - Distance from roofline ceiling"
echo "   - Compute-bound vs Memory-bound"
echo ""
echo "Expected Patterns:"
echo "  - Prefill: Near ridge point (mixed)"
echo "  - Decode: Memory-bound (below ridge)"
echo "========================================"
