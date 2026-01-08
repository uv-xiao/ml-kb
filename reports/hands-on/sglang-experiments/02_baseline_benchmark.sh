#!/bin/bash
# Experiment 2: Baseline Benchmark
# Run after server is started (01_start_server.sh)

source /home/uvxiao/mlkb/code-repos/sglang/.venv/bin/activate

echo "=========================================="
echo "Baseline Benchmark: 100 requests"
echo "=========================================="

python -m sglang.bench_serving \
    --backend sglang \
    --base-url http://localhost:30000 \
    --num-prompts 100 \
    --random-input 256 \
    --random-output 64 \
    --request-rate inf \
    2>&1 | tee baseline_benchmark.log

echo ""
echo "=========================================="
echo "Throughput Benchmark: 500 requests"
echo "=========================================="

python -m sglang.bench_serving \
    --backend sglang \
    --base-url http://localhost:30000 \
    --num-prompts 500 \
    --random-input 512 \
    --random-output 128 \
    --request-rate 50 \
    2>&1 | tee throughput_benchmark.log
