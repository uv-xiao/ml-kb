# SGLang Experiment Results

**Date:** 2026-01-10
**Model:** Qwen/Qwen3-0.6B
**Hardware:** NVIDIA A100 80GB PCIe
**SGLang Version:** 0.5.6.post3.dev1000

---

## 1. Server Configuration

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0 \
  python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --port 30000 \
    --log-level warning
```

**Server Startup:**
- Model load time: ~1s (cached locally)
- CUDA graph capture: 36 batch sizes in ~4s
- KV cache allocation: ~7.4 GB (for 69K tokens)
- Available memory after init: ~11 GB

---

## 2. Serving Benchmark

**Command:**
```bash
python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port 30000 \
  --dataset-name random --random-input 128 --random-output 64 \
  --num-prompts 50 --request-rate 5
```

### Results

| Metric | Value |
|--------|-------|
| **Successful requests** | 50 |
| **Benchmark duration** | 9.68 s |
| **Request throughput** | 5.17 req/s |
| **Input token throughput** | 311.88 tok/s |
| **Output token throughput** | 151.04 tok/s |
| **Peak output throughput** | 337.00 tok/s |
| **Peak concurrent requests** | 10 |
| **Total token throughput** | 462.92 tok/s |

### Latency Distribution

| Metric | Value |
|--------|-------|
| **Mean E2E Latency** | 101.60 ms |
| **Median E2E Latency** | 89.84 ms |
| **P90 E2E Latency** | 157.32 ms |
| **P99 E2E Latency** | 384.29 ms |
| **Mean TTFT** | 30.40 ms |
| **Median TTFT** | 21.85 ms |
| **P99 TTFT** | 217.70 ms |
| **Mean TPOT** | 2.46 ms |
| **Median TPOT** | 2.37 ms |
| **P99 TPOT** | 4.29 ms |
| **Mean ITL** | 2.52 ms |
| **Median ITL** | 2.26 ms |
| **P99 ITL** | 13.85 ms |

---

## 3. One-Batch Benchmark (Decode Performance)

**Command:**
```bash
CUDA_VISIBLE_DEVICES=1 python -m sglang.bench_one_batch \
  --model-path Qwen/Qwen3-0.6B \
  --batch-size 1 4 16 64 \
  --input-len 128 --output-len 64
```

### Prefill Performance

| Batch Size | Latency (s) | Throughput (tok/s) |
|------------|-------------|-------------------|
| 1 | 0.0167 | 7,656 |
| 4 | 0.0484 | 10,588 |
| 16 | 0.0435 | 47,082 |
| 64 | 0.0543 | 150,867 |

### Decode Performance (Median)

| Batch Size | Latency (ms) | Throughput (tok/s) | Scaling |
|------------|--------------|-------------------|---------|
| 1 | 2.95 | 339 | 1.0x |
| 4 | 3.39 | 1,181 | 3.5x |
| 16 | 3.55 | 4,507 | 13.3x |
| 64 | 4.06 | 15,756 | 46.5x |

### Total Throughput (Prefill + Decode)

| Batch Size | Total Latency (s) | Throughput (tok/s) |
|------------|-------------------|-------------------|
| 1 | 0.202 | 952 |
| 4 | 0.373 | 2,060 |
| 16 | 0.268 | 11,462 |
| 64 | 0.311 | 39,495 |

---

## 4. Analysis

### Decode Scaling Efficiency

The decode throughput scales well with batch size:
- **BS 1→4**: 3.5x (ideal: 4x) - 87% efficiency
- **BS 4→16**: 3.8x (ideal: 4x) - 95% efficiency
- **BS 16→64**: 3.5x (ideal: 4x) - 87% efficiency

This near-linear scaling indicates the decode kernel is memory-bound (as expected), with good batching efficiency.

### Prefill vs Decode Trade-off

For the 0.6B model:
- **Prefill**: Compute-bound, 150K tok/s at BS=64
- **Decode**: Memory-bound, 15.7K tok/s at BS=64

The prefill is ~10x faster per token, demonstrating the classic LLM serving characteristic where decode is the bottleneck.

### Latency Observations

- **TTFT** (Time to First Token): 21-30ms median is good for a small model
- **TPOT** (Time per Output Token): 2.4ms means ~420 tok/s per request
- **ITL** (Inter-Token Latency): P99 of 13.85ms shows occasional scheduling delays

---

## 5. Key Findings

1. **Batch Size Matters**: Going from BS=1 to BS=64 gives 46x throughput improvement
2. **Memory-Bound Decode**: Near-linear scaling confirms memory bandwidth is the bottleneck
3. **FlashInfer Backend**: Working correctly (default backend, no issues)
4. **CUDA Graph Benefits**: Captured 36 batch sizes, reducing kernel launch overhead

---

## 6. Environment Details

```
Python: 3.12.12
SGLang: 0.5.6.post3.dev1000
FlashInfer: 0.5.3 (with version check disabled)
CUDA: 12.5
GPU: NVIDIA A100 80GB PCIe
Nsight Systems: 2024.2.3.38
```

---

## 7. Log Files

All benchmark outputs saved to:
- `results/logs/server.log` - Server startup log
- `results/logs/bench_serving.log` - Serving benchmark results
- `results/logs/bench_one_batch.log` - One-batch benchmark results
