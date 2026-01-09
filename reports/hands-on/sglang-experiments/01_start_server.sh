#!/bin/bash
# Experiment 1: Start SGLang Server
# Run on GPU 2 (full memory available)
#
# Expected output:
# - Server running on http://localhost:30001
# - Model loaded: Qwen/Qwen3-0.6B
# - ~70GB GPU memory usage (1.3GB model + 65GB KV cache)
# - Startup time: ~15 seconds

export CUDA_VISIBLE_DEVICES=2
export FLASHINFER_DISABLE_VERSION_CHECK=1  # Bypass 0.5.3 vs 0.6.0 cache mismatch
export TMPDIR=/home/uvxiao/tmp

# Activate venv
source /home/uvxiao/mlkb/code-repos/sglang/.venv/bin/activate

echo "Starting SGLang server..."
echo "Model: Qwen/Qwen3-0.6B"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Port: 30001"

# Start server with Qwen3-0.6B (small model for quick testing)
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --port 30001 \
    --mem-fraction-static 0.85 \
    --log-level info \
    2>&1 | tee profiling_artifacts/server.log

# For larger model (Llama-3.1-8B), use:
# python -m sglang.launch_server \
#     --model-path meta-llama/Llama-3.1-8B-Instruct \
#     --port 30001 \
#     --mem-fraction-static 0.85
