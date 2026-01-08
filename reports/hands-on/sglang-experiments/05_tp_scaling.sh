#!/bin/bash
# Experiment 5: Tensor Parallelism Scaling
# Tests TP on NVLink pairs: (0,1), (2,3), (5,6)

source /home/uvxiao/mlkb/code-repos/sglang/.venv/bin/activate

OUTPUT_DIR="./profiling_artifacts"
mkdir -p $OUTPUT_DIR

# Use a larger model that benefits from TP
MODEL="meta-llama/Llama-3.1-8B-Instruct"
# Or use smaller model for testing:
# MODEL="Qwen/Qwen3-0.6B"

run_benchmark() {
    local name=$1
    local gpus=$2
    local tp=$3

    echo ""
    echo "=========================================="
    echo "Configuration: $name (TP=$tp, GPUs=$gpus)"
    echo "=========================================="

    # Start server
    CUDA_VISIBLE_DEVICES=$gpus python -m sglang.launch_server \
        --model-path $MODEL \
        --tp $tp \
        --port 30000 \
        --mem-fraction-static 0.85 \
        --log-level warning \
        2>&1 &
    SERVER_PID=$!

    # Wait for server to be ready
    echo "Waiting for server to start..."
    for i in {1..60}; do
        if curl -s http://localhost:30000/health > /dev/null 2>&1; then
            echo "Server ready!"
            break
        fi
        sleep 2
    done

    # Run benchmark
    python -m sglang.bench_serving \
        --backend sglang \
        --base-url http://localhost:30000 \
        --num-prompts 100 \
        --random-input 512 \
        --random-output 128 \
        --request-rate inf \
        2>&1 | tee $OUTPUT_DIR/tp_${name}.log

    # Stop server
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    sleep 5
}

echo "=========================================="
echo "Tensor Parallelism Scaling Experiment"
echo "=========================================="
echo ""
echo "GPU Topology:"
nvidia-smi topo -m
echo ""

# Single GPU baseline
run_benchmark "single_gpu1" "1" "1"

# TP=2 on NVLink pair (0,1)
run_benchmark "tp2_nvlink_01" "0,1" "2"

# TP=2 on NVLink pair (2,3)
run_benchmark "tp2_nvlink_23" "2,3" "2"

# TP=2 on NVLink pair (5,6)
run_benchmark "tp2_nvlink_56" "5,6" "2"

# TP=2 without NVLink (cross-NUMA: 1,5)
run_benchmark "tp2_no_nvlink_15" "1,5" "2"

# TP=4 on same NUMA (0,1,2,3) - mix of NVLink and PXB
run_benchmark "tp4_numa0" "0,1,2,3" "4"

echo ""
echo "=========================================="
echo "Results Summary"
echo "=========================================="
echo ""

for f in $OUTPUT_DIR/tp_*.log; do
    name=$(basename $f .log)
    throughput=$(grep -E "Throughput.*tok/s" $f | tail -1)
    echo "$name: $throughput"
done
