#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export FLASHINFER_DISABLE_VERSION_CHECK=1
source /home/uvxiao/mlkb/code-repos/sglang/.venv/bin/activate
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --port 30001 \
    --mem-fraction-static 0.85 \
    --log-level warning
