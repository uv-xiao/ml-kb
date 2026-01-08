#!/bin/bash
# Experiment 1: Start SGLang Server
# Run on GPU 1 (full memory available)

export CUDA_VISIBLE_DEVICES=1
export SGLANG_LOG_LEVEL=warning

# Activate venv
source /home/uvxiao/mlkb/code-repos/sglang/.venv/bin/activate

# Start server with Qwen3-0.6B (small model for quick testing)
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --port 30000 \
    --mem-fraction-static 0.85 \
    --enable-torch-compile \
    2>&1 | tee server.log

# For larger model (Llama-3.1-8B), use:
# python -m sglang.launch_server \
#     --model-path meta-llama/Llama-3.1-8B-Instruct \
#     --port 30000 \
#     --mem-fraction-static 0.85
