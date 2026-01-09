#!/bin/bash
#
# SGLang Nsight Systems Profiling Script
# =======================================
#
# Captures system-level timeline of SGLang inference.
#
# Usage:
#   ./04_nsys_profile.sh [model_path] [port]
#

set -e

# Configuration
MODEL_PATH="${1:-meta-llama/Llama-2-7b-chat-hf}"
PORT="${2:-30000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/../results/nsys"
WARMUP_REQUESTS=5
PROFILE_REQUESTS=10

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "SGLANG NSIGHT SYSTEMS PROFILING"
echo "========================================"
echo ""
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "Output: $OUTPUT_DIR"
echo ""

# Check nsys availability
if ! command -v nsys &> /dev/null; then
    echo "ERROR: nsys (Nsight Systems) not found in PATH"
    echo "Please install NVIDIA Nsight Systems:"
    echo "  https://developer.nvidia.com/nsight-systems"
    exit 1
fi

# Create a benchmark script to run during profiling
cat > "$OUTPUT_DIR/profile_benchmark.py" << 'BENCHMARK_SCRIPT'
#!/usr/bin/env python3
import sys
import time
import requests
import json

def send_request(url, prompt, max_tokens=64):
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    try:
        response = requests.post(
            f"{url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=60
        )
        tokens = 0
        for line in response.iter_lines():
            if line and line.decode().startswith('data: '):
                data = line.decode()[6:]
                if data != '[DONE]':
                    try:
                        if json.loads(data).get('choices', [{}])[0].get('delta', {}).get('content'):
                            tokens += 1
                    except:
                        pass
        return tokens
    except Exception as e:
        print(f"Request failed: {e}")
        return 0

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:30000"
    num_requests = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    prompt = "Explain the concept of machine learning in detail, covering supervised learning, unsupervised learning, and reinforcement learning. " * 3

    print(f"Sending {num_requests} requests to {url}...")

    # Trigger profiling start (if using cudaProfilerApi)
    try:
        import torch
        torch.cuda.cudart().cudaProfilerStart()
    except:
        pass

    start = time.time()
    total_tokens = 0
    for i in range(num_requests):
        tokens = send_request(url, prompt)
        total_tokens += tokens
        print(f"  Request {i+1}/{num_requests}: {tokens} tokens")

    # Stop profiling
    try:
        torch.cuda.cudart().cudaProfilerStop()
    except:
        pass

    elapsed = time.time() - start
    print(f"\nCompleted: {total_tokens} tokens in {elapsed:.2f}s ({total_tokens/elapsed:.0f} tok/s)")
BENCHMARK_SCRIPT

chmod +x "$OUTPUT_DIR/profile_benchmark.py"

# Method 1: Profile existing server
profile_existing_server() {
    echo "[Method 1] Profiling existing server at port $PORT..."

    # Check if server is running
    if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "ERROR: No server running at port $PORT"
        echo "Please start the server first with:"
        echo "  python -m sglang.launch_server --model-path $MODEL_PATH --port $PORT"
        return 1
    fi

    echo "Sending warmup requests..."
    python3 "$OUTPUT_DIR/profile_benchmark.py" "http://localhost:$PORT" $WARMUP_REQUESTS

    echo ""
    echo "Starting Nsight Systems capture..."

    nsys profile \
        --trace=cuda,nvtx,osrt,cudnn,cublas \
        --cuda-memory-usage=true \
        --gpuctxsw=true \
        --force-overwrite=true \
        --output="$OUTPUT_DIR/sglang_client_trace" \
        python3 "$OUTPUT_DIR/profile_benchmark.py" "http://localhost:$PORT" $PROFILE_REQUESTS

    echo ""
    echo "Profile captured: $OUTPUT_DIR/sglang_client_trace.nsys-rep"
}

# Method 2: Profile server startup and inference
profile_server_full() {
    echo "[Method 2] Full server profiling (startup + inference)..."
    echo "WARNING: This profiles the entire server process."
    echo ""

    # Create server launch script
    cat > "$OUTPUT_DIR/launch_and_benchmark.py" << LAUNCH_SCRIPT
#!/usr/bin/env python3
import subprocess
import time
import sys
import os

# Start server in background
server_cmd = [
    sys.executable, "-m", "sglang.launch_server",
    "--model-path", "$MODEL_PATH",
    "--port", "$PORT",
    "--log-level", "warning"
]

print("Starting SGLang server...")
server_proc = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Wait for server to be ready
import requests
max_wait = 120
for i in range(max_wait):
    try:
        r = requests.get("http://localhost:$PORT/health", timeout=2)
        if r.status_code == 200:
            print(f"Server ready after {i}s")
            break
    except:
        pass
    time.sleep(1)
else:
    print("Server failed to start")
    server_proc.terminate()
    sys.exit(1)

# Run benchmark
print("Running benchmark...")
os.system(f"python3 {os.path.dirname(__file__)}/profile_benchmark.py http://localhost:$PORT $PROFILE_REQUESTS")

# Shutdown
print("Stopping server...")
server_proc.terminate()
server_proc.wait(timeout=10)
LAUNCH_SCRIPT

    chmod +x "$OUTPUT_DIR/launch_and_benchmark.py"

    nsys profile \
        --trace=cuda,nvtx,osrt,cudnn,cublas \
        --cuda-memory-usage=true \
        --gpuctxsw=true \
        --force-overwrite=true \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --output="$OUTPUT_DIR/sglang_full_trace" \
        python3 "$OUTPUT_DIR/launch_and_benchmark.py"

    echo ""
    echo "Profile captured: $OUTPUT_DIR/sglang_full_trace.nsys-rep"
}

# Show usage
show_help() {
    echo "Usage: $0 [model_path] [port]"
    echo ""
    echo "Options:"
    echo "  model_path  Path or HuggingFace model name (default: meta-llama/Llama-2-7b-chat-hf)"
    echo "  port        Server port (default: 30000)"
    echo ""
    echo "Prerequisites:"
    echo "  1. Start SGLang server:"
    echo "     python -m sglang.launch_server --model-path <model> --port 30000"
    echo ""
    echo "  2. Run this script:"
    echo "     ./04_nsys_profile.sh"
    echo ""
    echo "  3. View results:"
    echo "     nsys-ui results/nsys/sglang_client_trace.nsys-rep"
}

# Main
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

profile_existing_server

echo ""
echo "========================================"
echo "ANALYSIS GUIDE"
echo "========================================"
echo ""
echo "Open the profile in Nsight Systems:"
echo "  nsys-ui $OUTPUT_DIR/sglang_client_trace.nsys-rep"
echo ""
echo "Key things to look for:"
echo ""
echo "1. TIMELINE VIEW"
echo "   - GPU kernel execution (green)"
echo "   - CPU activity (blue)"
echo "   - Memory transfers (orange)"
echo "   - NVTX ranges (if instrumented)"
echo ""
echo "2. KERNEL SUMMARY"
echo "   - Top kernels by total time"
echo "   - Attention kernels (flash_attention_*)"
echo "   - GEMM kernels (cublas_*)"
echo ""
echo "3. MEMORY ANALYSIS"
echo "   - GPU memory allocation patterns"
echo "   - Peak memory usage"
echo "   - Memory fragmentation"
echo ""
echo "4. CPU-GPU OVERLAP"
echo "   - Gaps in GPU execution (bubbles)"
echo "   - CPU scheduling overhead"
echo ""
echo "========================================"
