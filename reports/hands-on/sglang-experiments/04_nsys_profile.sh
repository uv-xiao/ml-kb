#!/bin/bash
# Experiment 4: Nsight Systems Profiling
# Captures GPU kernel traces for detailed analysis

source /home/uvxiao/mlkb/code-repos/sglang/.venv/bin/activate

OUTPUT_DIR="./profiling_artifacts"
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "NSight Systems Profiling"
echo "=========================================="

# Profile server startup + first few requests
# Note: This profiles the server process itself

export CUDA_VISIBLE_DEVICES=1

echo "Step 1: Profile with PyTorch profiler (server-side)"
echo "Enable profiler by setting env vars before starting server:"
echo ""
echo "  export SGLANG_TORCH_PROFILER_DIR=$OUTPUT_DIR/torch_traces"
echo "  # Then start server and send requests"
echo ""

echo "Step 2: Profile benchmark with nsys"
nsys profile -o $OUTPUT_DIR/sglang_benchmark \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --force-overwrite true \
    python -m sglang.bench_serving \
        --backend sglang \
        --base-url http://localhost:30000 \
        --num-prompts 20 \
        --random-input 256 \
        --random-output 64 \
        --request-rate inf

echo ""
echo "Step 3: Generate statistics"
nsys stats $OUTPUT_DIR/sglang_benchmark.nsys-rep \
    --report cuda_gpu_kern_sum \
    > $OUTPUT_DIR/kernel_summary.txt

nsys stats $OUTPUT_DIR/sglang_benchmark.nsys-rep \
    --report cuda_gpu_mem_size_sum \
    > $OUTPUT_DIR/memory_summary.txt

echo ""
echo "=========================================="
echo "Profiling complete. Output files:"
echo "  $OUTPUT_DIR/sglang_benchmark.nsys-rep  (Nsight trace)"
echo "  $OUTPUT_DIR/kernel_summary.txt         (Kernel stats)"
echo "  $OUTPUT_DIR/memory_summary.txt         (Memory stats)"
echo ""
echo "Open trace in Nsight Systems GUI:"
echo "  nsys-ui $OUTPUT_DIR/sglang_benchmark.nsys-rep"
echo "=========================================="
