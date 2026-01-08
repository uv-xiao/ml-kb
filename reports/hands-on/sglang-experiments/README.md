# SGLang Hands-On Learning Experiments

This directory contains executable experiments for deep-diving into SGLang's serving performance.

## Environment

- **SGLang**: 0.5.6.post3.dev (from source at `code-repos/sglang`)
- **Python venv**: `code-repos/sglang/.venv`
- **GPUs**: 7 × A100 80GB (NVLink pairs: 0-1, 2-3, 5-6)
- **Available GPUs**: 1, 2, 5, 6 have full memory (~80GB free)

## Quick Start

```bash
cd /home/uvxiao/mlkb/reports/hands-on/sglang-experiments

# Terminal 1: Start server
bash 01_start_server.sh

# Terminal 2: Run experiments (wait for server to be ready)
bash 02_baseline_benchmark.sh
```

## Experiments

### Experiment 1: Start Server (`01_start_server.sh`)

Launches SGLang server on GPU 1 with Qwen3-0.6B model.

**Expected output**:
```
Server running on http://0.0.0.0:30000
Loaded model: Qwen/Qwen3-0.6B
Memory usage: ~1.5GB
```

### Experiment 2: Baseline Benchmark (`02_baseline_benchmark.sh`)

Runs throughput and latency benchmarks.

**Key metrics to observe**:
- **TTFT** (Time To First Token): Prefill latency
- **ITL** (Inter-Token Latency): Decode latency per token
- **Throughput**: Tokens/second

**Expected output** (Qwen3-0.6B on A100):
```
Total requests: 100
Throughput: ~2000-3000 tok/s
Median TTFT: ~15-30ms
Median ITL: ~5-10ms
```

### Experiment 3: PyTorch Profiling (`03_profile_with_torch.py`)

Profiles end-to-end request flow.

```bash
source ../../../code-repos/sglang/.venv/bin/activate
python 03_profile_with_torch.py
```

**What to observe**:
- Single request latency breakdown
- Batch concurrency benefits
- RadixCache prefix hit patterns
- Sequence length scaling

### Experiment 4: Nsight Systems Profiling (`04_nsys_profile.sh`)

Captures GPU kernel traces.

```bash
bash 04_nsys_profile.sh
```

**Output files**:
- `profiling_artifacts/sglang_benchmark.nsys-rep` - GPU trace
- `profiling_artifacts/kernel_summary.txt` - Kernel statistics

**How to analyze**:
```bash
# Open in Nsight Systems GUI
nsys-ui profiling_artifacts/sglang_benchmark.nsys-rep

# Or view text summary
cat profiling_artifacts/kernel_summary.txt
```

**Key observations**:
- Attention kernel dominates GPU time
- Prefill vs decode kernel selection
- CPU-GPU overlap (scheduling hidden behind GPU work)

### Experiment 5: Tensor Parallelism Scaling (`05_tp_scaling.sh`)

Tests TP scaling on different GPU configurations.

```bash
bash 05_tp_scaling.sh
```

**Configurations tested**:
| Config | GPUs | Interconnect |
|--------|------|--------------|
| Single | 1 | - |
| TP=2 NVLink | 0,1 | NV12 |
| TP=2 NVLink | 2,3 | NV12 |
| TP=2 NVLink | 5,6 | NV12 |
| TP=2 No NVLink | 1,5 | SYS (cross-NUMA) |
| TP=4 | 0,1,2,3 | NVLink + PXB |

**Expected observations**:
- NVLink TP=2: Near-linear scaling
- Cross-NUMA TP=2: Higher communication overhead
- TP=4: Some overhead from PXB connections

### Experiment 6: RadixCache Analysis (`06_radix_cache_analysis.py`)

Deep-dive into prefix caching behavior.

```bash
source ../../../code-repos/sglang/.venv/bin/activate
python 06_radix_cache_analysis.py
```

**Sub-experiments**:
1. **Prefix sharing benefit**: First request slow, subsequent fast
2. **Prefix length impact**: Longer prefixes = more benefit
3. **Concurrent prefix sharing**: In-batch optimization
4. **Cache eviction**: Behavior under memory pressure

**Expected results**:
- Cache speedup: 1.5-3x for shared prefixes
- Concurrent batch speedup: 3-5x over sequential
- Eviction: Depends on cache size and workload

## Results Directory

All experiment outputs are saved to `profiling_artifacts/`:

```
profiling_artifacts/
├── baseline_benchmark.log
├── throughput_benchmark.log
├── profiling_results.json
├── radix_cache_results.json
├── sglang_benchmark.nsys-rep
├── kernel_summary.txt
├── memory_summary.txt
└── tp_*.log (one per configuration)
```

## Key Findings Template

After running experiments, fill in:

### Performance Baseline

| Metric | Value | Notes |
|--------|-------|-------|
| TTFT (median) | ___ ms | Prefill latency |
| ITL (median) | ___ ms | Decode latency |
| Throughput | ___ tok/s | End-to-end |

### RadixCache Effectiveness

| Metric | Value |
|--------|-------|
| Cache hit speedup | ___x |
| First vs subsequent | ___ ms vs ___ ms |
| Concurrent benefit | ___x |

### TP Scaling

| Config | Throughput | vs Single GPU |
|--------|------------|---------------|
| Single GPU | ___ tok/s | 1.0x |
| TP=2 NVLink | ___ tok/s | ___x |
| TP=2 No NVLink | ___ tok/s | ___x |
| TP=4 | ___ tok/s | ___x |

### Kernel Analysis

| Kernel | % GPU Time | Observation |
|--------|------------|-------------|
| Attention | ___% | ___ |
| MLP/FFN | ___% | ___ |
| LayerNorm | ___% | ___ |
| Other | ___% | ___ |

## Docker Alternative

If venv has issues, use Docker:

```bash
docker run --gpus '"device=1"' \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
        --model-path Qwen/Qwen3-0.6B \
        --host 0.0.0.0 \
        --port 30000
```

## References

- [SGLang Documentation](https://docs.sglang.io/)
- [RadixAttention Paper](https://arxiv.org/abs/2312.07104)
- [SGLang v0.4 Blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)
