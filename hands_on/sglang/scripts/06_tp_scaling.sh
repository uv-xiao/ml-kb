#!/bin/bash
#
# SGLang Tensor Parallelism Scaling Analysis
# ==========================================
#
# Measures TP scaling efficiency across different GPU configurations.
#
# Usage:
#   ./06_tp_scaling.sh [model_path]
#

set -e

MODEL_PATH="${1:-meta-llama/Llama-2-70b-chat-hf}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/../results/tp_scaling"
RESULTS_FILE="${OUTPUT_DIR}/scaling_results.json"

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "TENSOR PARALLELISM SCALING ANALYSIS"
echo "========================================"
echo ""
echo "Model: $MODEL_PATH"
echo ""

# GPU topology reference (from environment.md)
cat << 'TOPOLOGY'
GPU Topology (Reference):
       GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6
GPU0    X    NV12  PXB   PXB   SYS   SYS   SYS
GPU1   NV12   X    PXB   PXB   SYS   SYS   SYS
GPU2   PXB   PXB    X    NV12  SYS   SYS   SYS
GPU3   PXB   PXB   NV12   X    SYS   SYS   SYS
GPU4   SYS   SYS   SYS   SYS    X    PXB   PXB
GPU5   SYS   SYS   SYS   SYS   PXB    X    NV12
GPU6   SYS   SYS   SYS   SYS   PXB   NV12   X

NVLink pairs: (0,1), (2,3), (5,6)
NUMA: GPUs 0-3 on node 0, GPUs 4-6 on node 1

TOPOLOGY

# Create benchmark script
cat > "$OUTPUT_DIR/benchmark_tp.py" << 'BENCHMARK_SCRIPT'
#!/usr/bin/env python3
"""
TP Scaling Benchmark
"""

import argparse
import json
import time
import requests
import statistics
import os


def generate_prompt(length=512):
    base = "Please provide a comprehensive analysis of "
    topics = [
        "the impact of artificial intelligence on healthcare",
        "climate change mitigation strategies",
        "the evolution of programming languages",
        "quantum computing applications",
    ]
    prompt = base + ", ".join(topics * (length // 50 + 1))
    return prompt[:length * 4]


def benchmark_server(url, num_requests=20, prompt_len=512, output_len=128):
    """Benchmark a running server and return metrics."""
    prompt = generate_prompt(prompt_len)

    ttfts = []
    itls = []
    total_tokens = 0
    errors = 0

    start_time = time.perf_counter()

    for i in range(num_requests):
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": output_len,
            "stream": True,
        }

        try:
            req_start = time.perf_counter()
            first_token_time = None
            last_token_time = req_start
            token_count = 0

            response = requests.post(
                f"{url}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=120
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        if chunk.get('choices', [{}])[0].get('delta', {}).get('content'):
                            now = time.perf_counter()
                            if first_token_time is None:
                                first_token_time = now
                                ttfts.append((now - req_start) * 1000)
                            else:
                                itls.append((now - last_token_time) * 1000)
                            last_token_time = now
                            token_count += 1
                    except json.JSONDecodeError:
                        pass

            total_tokens += token_count

        except Exception as e:
            print(f"Request {i+1} failed: {e}")
            errors += 1

        print(f"  Request {i+1}/{num_requests}", end='\r')

    total_time = time.perf_counter() - start_time
    print()

    return {
        "ttft_avg_ms": statistics.mean(ttfts) if ttfts else 0,
        "ttft_p50_ms": statistics.median(ttfts) if ttfts else 0,
        "ttft_p99_ms": sorted(ttfts)[int(len(ttfts)*0.99)] if len(ttfts) > 1 else 0,
        "itl_avg_ms": statistics.mean(itls) if itls else 0,
        "throughput_tokens_per_sec": total_tokens / total_time if total_time > 0 else 0,
        "total_tokens": total_tokens,
        "total_time_sec": total_time,
        "errors": errors
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default='http://localhost:30000')
    parser.add_argument('--config', type=str, required=True,
                        help='Configuration name (e.g., tp1, tp2_nvlink)')
    parser.add_argument('--num-requests', type=int, default=20)
    parser.add_argument('--prompt-len', type=int, default=512)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--output-file', type=str, default='results.json')
    args = parser.parse_args()

    print(f"Benchmarking {args.config}...")
    print(f"URL: {args.url}")

    # Wait for server
    for _ in range(30):
        try:
            r = requests.get(f"{args.url}/health", timeout=2)
            if r.status_code == 200:
                break
        except:
            pass
        time.sleep(1)
    else:
        print("Server not ready")
        return

    results = benchmark_server(
        args.url,
        args.num_requests,
        args.prompt_len,
        args.output_len
    )

    results['config'] = args.config

    # Append to results file
    all_results = []
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r') as f:
            all_results = json.load(f)

    all_results.append(results)

    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults for {args.config}:")
    print(f"  TTFT avg: {results['ttft_avg_ms']:.1f}ms")
    print(f"  ITL avg: {results['itl_avg_ms']:.1f}ms")
    print(f"  Throughput: {results['throughput_tokens_per_sec']:.0f} tokens/s")


if __name__ == "__main__":
    main()
BENCHMARK_SCRIPT

chmod +x "$OUTPUT_DIR/benchmark_tp.py"

# Define TP configurations
declare -A CONFIGS
CONFIGS["tp1"]="0"
CONFIGS["tp2_nvlink_01"]="0,1"
CONFIGS["tp2_nvlink_23"]="2,3"
CONFIGS["tp2_nvlink_56"]="5,6"
CONFIGS["tp4_numa0"]="0,1,2,3"
CONFIGS["tp4_cross_numa"]="0,1,4,5"

# Function to run benchmark for a config
run_benchmark() {
    local config_name=$1
    local gpus=$2
    local port=$3

    echo ""
    echo "========================================"
    echo "Configuration: $config_name"
    echo "GPUs: $gpus"
    echo "========================================"

    # Count GPUs
    local tp_size=$(echo "$gpus" | tr ',' '\n' | wc -l)

    # Start server
    echo "Starting server with TP=$tp_size on GPUs $gpus..."

    CUDA_VISIBLE_DEVICES=$gpus python -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --tp $tp_size \
        --port $port \
        --log-level warning \
        &
    local server_pid=$!

    # Wait for server
    echo "Waiting for server to start..."
    for i in $(seq 1 120); do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "Server ready after ${i}s"
            break
        fi
        sleep 1
    done

    # Check if server started
    if ! curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
        echo "ERROR: Server failed to start"
        kill $server_pid 2>/dev/null || true
        return 1
    fi

    # Run benchmark
    python3 "$OUTPUT_DIR/benchmark_tp.py" \
        --url "http://localhost:$port" \
        --config "$config_name" \
        --num-requests 20 \
        --prompt-len 512 \
        --output-len 128 \
        --output-file "$RESULTS_FILE"

    # Stop server
    echo "Stopping server..."
    kill $server_pid 2>/dev/null || true
    wait $server_pid 2>/dev/null || true

    sleep 5  # Allow cleanup
}

# Generate analysis report
generate_report() {
    echo ""
    echo "========================================"
    echo "GENERATING SCALING ANALYSIS REPORT"
    echo "========================================"

    if [[ ! -f "$RESULTS_FILE" ]]; then
        echo "No results file found"
        return
    fi

    python3 << ANALYSIS_SCRIPT
import json

with open("$RESULTS_FILE", 'r') as f:
    results = json.load(f)

# Sort by config name
results.sort(key=lambda x: x['config'])

# Find baseline (TP=1)
baseline = next((r for r in results if 'tp1' in r['config'].lower()), None)

print("\n" + "=" * 70)
print("TP SCALING ANALYSIS RESULTS")
print("=" * 70)

print("\nLATENCY METRICS:")
print("-" * 70)
print(f"{'Config':<20} {'TTFT (ms)':<12} {'ITL (ms)':<12} {'Throughput':<15} {'Speedup':<10}")
print("-" * 70)

for r in results:
    config = r['config']
    ttft = r['ttft_avg_ms']
    itl = r['itl_avg_ms']
    throughput = r['throughput_tokens_per_sec']

    if baseline and baseline['ttft_avg_ms'] > 0:
        speedup = baseline['ttft_avg_ms'] / ttft if ttft > 0 else 0
        speedup_str = f"{speedup:.2f}x"
    else:
        speedup_str = "-"

    print(f"{config:<20} {ttft:<12.1f} {itl:<12.1f} {throughput:<15.0f} {speedup_str:<10}")

print("-" * 70)

# Scaling efficiency analysis
print("\nSCALING EFFICIENCY:")
print("-" * 70)

for r in results:
    config = r['config']

    # Determine TP size from config name
    if 'tp1' in config:
        tp = 1
    elif 'tp2' in config:
        tp = 2
    elif 'tp4' in config:
        tp = 4
    else:
        tp = 1

    if baseline and tp > 1:
        ideal_speedup = tp
        actual_speedup = baseline['ttft_avg_ms'] / r['ttft_avg_ms'] if r['ttft_avg_ms'] > 0 else 0
        efficiency = actual_speedup / ideal_speedup * 100

        print(f"{config:<20}: {efficiency:.0f}% efficiency (actual {actual_speedup:.2f}x vs ideal {ideal_speedup}x)")

print("-" * 70)

# Recommendations
print("\nRECOMMENDATIONS:")
print("-" * 70)

# Find best configs
nvlink_configs = [r for r in results if 'nvlink' in r['config']]
if nvlink_configs:
    best_tp2 = min(nvlink_configs, key=lambda x: x['ttft_avg_ms'])
    print(f"  Best TP=2: {best_tp2['config']} (TTFT: {best_tp2['ttft_avg_ms']:.1f}ms)")

tp4_configs = [r for r in results if 'tp4' in r['config']]
if tp4_configs:
    best_tp4 = min(tp4_configs, key=lambda x: x['ttft_avg_ms'])
    print(f"  Best TP=4: {best_tp4['config']} (TTFT: {best_tp4['ttft_avg_ms']:.1f}ms)")

print("""
KEY INSIGHTS:
  - NVLink pairs provide best TP=2 performance
  - NUMA-local TP=4 better than cross-NUMA
  - Efficiency drops with more GPUs (communication overhead)
  - For latency: Use smallest TP that fits model
  - For throughput: Use larger TP with batching
""")

print("=" * 70)
ANALYSIS_SCRIPT
}

# Main menu
show_menu() {
    echo ""
    echo "Select option:"
    echo "  1. Run all configurations"
    echo "  2. Run specific configuration"
    echo "  3. Generate report from existing results"
    echo "  4. Show help"
    echo ""
    read -p "Choice [1-4]: " choice

    case $choice in
        1)
            # Initialize results file
            echo "[]" > "$RESULTS_FILE"

            port=30000
            for config in "${!CONFIGS[@]}"; do
                run_benchmark "$config" "${CONFIGS[$config]}" $port
            done

            generate_report
            ;;
        2)
            echo ""
            echo "Available configurations:"
            for config in "${!CONFIGS[@]}"; do
                echo "  $config: GPUs ${CONFIGS[$config]}"
            done
            echo ""
            read -p "Enter configuration name: " selected_config

            if [[ -n "${CONFIGS[$selected_config]}" ]]; then
                run_benchmark "$selected_config" "${CONFIGS[$selected_config]}" 30000
            else
                echo "Invalid configuration"
            fi
            ;;
        3)
            generate_report
            ;;
        4)
            echo ""
            echo "This script benchmarks SGLang with different TP configurations"
            echo "to measure scaling efficiency across GPU topologies."
            echo ""
            echo "Configurations tested:"
            echo "  tp1           - Single GPU baseline"
            echo "  tp2_nvlink_*  - TP=2 on NVLink pairs (best latency)"
            echo "  tp4_numa0     - TP=4 on same NUMA node"
            echo "  tp4_cross_numa- TP=4 crossing NUMA boundary"
            echo ""
            echo "Prerequisites:"
            echo "  - Model must fit in target GPU configuration"
            echo "  - All GPUs should be available"
            echo ""
            ;;
        *)
            echo "Invalid choice"
            ;;
    esac
}

# Run
show_menu
