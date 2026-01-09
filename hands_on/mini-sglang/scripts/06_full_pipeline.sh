#!/bin/bash
#
# Full Pipeline Profiling Script for Mini-SGLang
#
# This script orchestrates full pipeline profiling including:
# 1. Environment check
# 2. Individual kernel profiling
# 3. End-to-end inference profiling with Nsight Systems
# 4. Report generation
#
# Usage:
#   ./06_full_pipeline.sh [--quick] [--nsys] [--all]
#
# Options:
#   --quick   Run quick profiling (fewer iterations)
#   --nsys    Include Nsight Systems profiling
#   --all     Run all profiling steps
#

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../results"
REPORTS_DIR="${SCRIPT_DIR}/../reports"

# Create directories
mkdir -p "${RESULTS_DIR}"
mkdir -p "${REPORTS_DIR}"

# Parse arguments
QUICK=false
NSYS=false
ALL=false

for arg in "$@"; do
    case $arg in
        --quick)
            QUICK=true
            shift
            ;;
        --nsys)
            NSYS=true
            shift
            ;;
        --all)
            ALL=true
            shift
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

if [ "$ALL" = true ]; then
    NSYS=true
fi

echo "============================================================"
echo "MINI-SGLANG FULL PIPELINE PROFILING"
echo "============================================================"
echo ""
echo "Results directory: ${RESULTS_DIR}"
echo "Reports directory: ${REPORTS_DIR}"
echo "Quick mode: ${QUICK}"
echo "Nsight Systems: ${NSYS}"
echo ""

# Step 1: Environment Check
echo "============================================================"
echo "STEP 1: Environment Check"
echo "============================================================"
python3 "${SCRIPT_DIR}/00_check_env.py"

# Step 2: Index Kernel Profiling
echo ""
echo "============================================================"
echo "STEP 2: Index Kernel Profiling"
echo "============================================================"

if [ "$QUICK" = true ]; then
    python3 "${SCRIPT_DIR}/02_profile_index.py" \
        --batch-size 64 \
        --embedding-dim 4096 \
        --output "${RESULTS_DIR}/index_quick.json"
else
    python3 "${SCRIPT_DIR}/02_profile_index.py" \
        --sweep \
        --output "${RESULTS_DIR}/index_sweep.json"
fi

# Step 3: Store Kernel Profiling
echo ""
echo "============================================================"
echo "STEP 3: Store Kernel Profiling"
echo "============================================================"

if [ "$QUICK" = true ]; then
    python3 "${SCRIPT_DIR}/03_profile_store.py" \
        --num-tokens 64 \
        --head-dim 128 \
        --output "${RESULTS_DIR}/store_quick.json"
else
    python3 "${SCRIPT_DIR}/03_profile_store.py" \
        --compare-patterns \
        --output "${RESULTS_DIR}/store_patterns.json"

    python3 "${SCRIPT_DIR}/03_profile_store.py" \
        --sweep \
        --output "${RESULTS_DIR}/store_sweep.json"
fi

# Step 4: Attention Profiling
echo ""
echo "============================================================"
echo "STEP 4: Attention Profiling"
echo "============================================================"

if [ "$QUICK" = true ]; then
    python3 "${SCRIPT_DIR}/04_profile_attention.py" \
        --phase prefill \
        --seq-len 512 \
        --output "${RESULTS_DIR}/attention_prefill_quick.json"

    python3 "${SCRIPT_DIR}/04_profile_attention.py" \
        --phase decode \
        --seq-len 512 \
        --output "${RESULTS_DIR}/attention_decode_quick.json"
else
    python3 "${SCRIPT_DIR}/04_profile_attention.py" \
        --compare \
        --seq-len 1024 \
        --output "${RESULTS_DIR}/attention_compare.json"

    python3 "${SCRIPT_DIR}/04_profile_attention.py" \
        --phase prefill \
        --sweep \
        --output "${RESULTS_DIR}/attention_prefill_sweep.json"
fi

# Step 5: NCCL Analysis
echo ""
echo "============================================================"
echo "STEP 5: NCCL Communication Analysis"
echo "============================================================"

python3 "${SCRIPT_DIR}/05_profile_comm.py" \
    --analyze \
    --output "${RESULTS_DIR}/nccl_analysis.json"

# Step 6: Nsight Systems Profiling (optional)
if [ "$NSYS" = true ]; then
    echo ""
    echo "============================================================"
    echo "STEP 6: Nsight Systems Profiling"
    echo "============================================================"

    # Check if nsys is available
    if command -v nsys &> /dev/null; then
        # Create a simple inference script for profiling
        PROFILE_SCRIPT="${RESULTS_DIR}/profile_inference.py"
        cat > "${PROFILE_SCRIPT}" << 'EOFPY'
#!/usr/bin/env python3
"""Minimal inference for Nsight Systems profiling."""
import torch
import torch.cuda as cuda

# Simulate a simplified forward pass
def simulate_forward():
    try:
        from minisgl.kernel.index import indexing
        from minisgl.kernel.store import store_cache

        # Index kernel
        weights = torch.randn(32000, 4096, dtype=torch.float16, device="cuda")
        indices = torch.randint(0, 32000, (64,), dtype=torch.int32, device="cuda")
        embeddings = indexing(weights, indices)

        # Store kernel
        k_cache = torch.zeros(4096, 1024, dtype=torch.float16, device="cuda")
        v_cache = torch.zeros(4096, 1024, dtype=torch.float16, device="cuda")
        cache_indices = torch.arange(64, dtype=torch.int32, device="cuda")
        k = torch.randn(64, 1024, dtype=torch.float16, device="cuda")
        v = torch.randn(64, 1024, dtype=torch.float16, device="cuda")
        store_cache(k_cache, v_cache, cache_indices, k, v)

        cuda.synchronize()
        print("Forward simulation completed successfully")
    except Exception as e:
        print(f"Simulation failed: {e}")
        # Fallback to basic CUDA operations
        x = torch.randn(64, 4096, dtype=torch.float16, device="cuda")
        y = torch.matmul(x, x.T)
        cuda.synchronize()
        print("Fallback simulation completed")

if __name__ == "__main__":
    # Warmup
    for _ in range(3):
        simulate_forward()

    # Profiled run
    cuda.cudart().cudaProfilerStart()
    for _ in range(10):
        simulate_forward()
    cuda.cudart().cudaProfilerStop()
EOFPY

        echo "Running Nsight Systems profile..."
        nsys profile \
            --output="${RESULTS_DIR}/nsys_inference" \
            --force-overwrite=true \
            --trace=cuda,nvtx \
            --sample=none \
            python3 "${PROFILE_SCRIPT}"

        echo "Nsight Systems profile saved to: ${RESULTS_DIR}/nsys_inference.nsys-rep"

        # Generate stats
        if [ -f "${RESULTS_DIR}/nsys_inference.nsys-rep" ]; then
            nsys stats "${RESULTS_DIR}/nsys_inference.nsys-rep" \
                --report gputrace \
                --format csv \
                --output "${RESULTS_DIR}/nsys_kernel_stats"
            echo "Kernel stats saved to: ${RESULTS_DIR}/nsys_kernel_stats.csv"
        fi
    else
        echo "WARNING: nsys not found. Skipping Nsight Systems profiling."
    fi
fi

# Step 7: Generate Summary
echo ""
echo "============================================================"
echo "STEP 7: Results Summary"
echo "============================================================"

SUMMARY_FILE="${RESULTS_DIR}/pipeline_summary.txt"

cat > "${SUMMARY_FILE}" << EOF
MINI-SGLANG FULL PIPELINE PROFILING SUMMARY
============================================
Date: $(date)
Host: $(hostname)

PROFILING RESULTS
-----------------
EOF

# Add results from each step
for result_file in "${RESULTS_DIR}"/*.json; do
    if [ -f "$result_file" ]; then
        echo "" >> "${SUMMARY_FILE}"
        echo "File: $(basename "$result_file")" >> "${SUMMARY_FILE}"
        echo "Size: $(wc -c < "$result_file") bytes" >> "${SUMMARY_FILE}"
    fi
done

echo "" >> "${SUMMARY_FILE}"
echo "PROFILING COMPLETE" >> "${SUMMARY_FILE}"

cat "${SUMMARY_FILE}"

echo ""
echo "============================================================"
echo "PROFILING COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to: ${RESULTS_DIR}/"
echo "Summary: ${SUMMARY_FILE}"
echo ""
echo "Next steps:"
echo "  1. Review individual JSON results"
echo "  2. Open nsys_inference.nsys-rep in Nsight Systems GUI"
echo "  3. Generate kernel-dev-guide.md from analysis"
echo ""
