# Hands-On Learning Plan: SGLang

## Goal Interpretation

**Original Goal**: Explore SGLang using the hands-on-learning skill

**Interpreted As**:
1. How does SGLang's RadixAttention/prefix caching work in practice?
2. What is the execution pattern of continuous batching?
3. How do different attention backends (FlashInfer, Triton) compare?
4. What hardware utilization does SGLang achieve on A100?
5. How does tensor parallelism scale with NVLink?

**Informed by Codebase Analysis**:
- SGLang uses RadixCache (radix tree) for prefix caching → track cache hit rates, eviction patterns
- FlashInfer backend with BatchDecodeWithPagedKVCacheWrapper → track kernel selection, plan-run overhead
- Zero-overhead scheduler with continuous batching → track batch formation, scheduling latency
- Two-level memory pool (ReqToToken, TokenToKV) → track memory allocation patterns
- Multiple scheduling policies (LPM, DFS-weight, FCFS) → compare cache-aware vs agnostic

## Environment

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         ENVIRONMENT SUMMARY                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  GPU Configuration:                                                          │
│  ═══════════════════                                                         │
│  • GPU Model:        7 × NVIDIA A100 80GB (mix PCIe + SXM4)                 │
│  • Compute Cap:      8.0 (Ampere)                                            │
│  • Total GPU Memory: 560 GB                                                  │
│                                                                              │
│  GPU Topology:                                                               │
│  ═════════════                                                               │
│         GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6                            │
│  GPU0    X    NV12  PXB   PXB   SYS   SYS   SYS                             │
│  GPU1   NV12   X    PXB   PXB   SYS   SYS   SYS                             │
│  GPU2   PXB   PXB    X    NV12  SYS   SYS   SYS                             │
│  GPU3   PXB   PXB   NV12   X    SYS   SYS   SYS                             │
│  GPU4   SYS   SYS   SYS   SYS    X    PXB   PXB                             │
│  GPU5   SYS   SYS   SYS   SYS   PXB    X    NV12                            │
│  GPU6   SYS   SYS   SYS   SYS   PXB   NV12   X                              │
│                                                                              │
│  • NVLink pairs: (0,1), (2,3), (5,6)                                        │
│  • NUMA nodes: GPUs 0-3 on node 0, GPUs 4-6 on node 1                       │
│                                                                              │
│  Software Stack:                                                             │
│  ═══════════════                                                             │
│  • CUDA Version:     12.5                                                    │
│  • Python:           3.x (will use 3.11)                                     │
│                                                                              │
│  Communication Implications:                                                 │
│  ═══════════════════════════                                                 │
│  • TP=2 optimal: Use NVLink pairs (GPU 0-1 or 2-3 or 5-6)                   │
│  • TP=4 on NUMA 0: GPUs 0-3 (NV12 + PXB, same NUMA)                         │
│  • Cross-NUMA: Higher latency, avoid for latency-sensitive                  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Tracking Requirements

### Hardware-Level Tracking

| Metric | Why | How to Measure |
|--------|-----|----------------|
| SM utilization | Scheduler efficiency | nsys GPU metrics |
| Tensor core util | Attention kernel efficiency | ncu TC metrics |
| HBM bandwidth | Memory-bound analysis | ncu DRAM throughput |
| NVLink utilization | TP communication cost | nsys with nvlink |

### Process-Level Tracking

| Event | Why | How to Capture |
|-------|-----|----------------|
| Batch formation | Scheduler behavior | NVTX markers in scheduler |
| Kernel launches | Prefill vs decode pattern | nsys timeline |
| RadixCache operations | Cache hit/miss patterns | Custom logging |
| Memory allocation | KV cache dynamics | torch.cuda.memory_stats |

### Application-Level Tracking

| Metric | Why | How to Measure |
|--------|-----|----------------|
| TTFT | Prefill latency | bench_serving output |
| ITL | Decode latency | bench_serving output |
| Throughput | Overall efficiency | bench_serving output |
| Cache hit rate | RadixCache effectiveness | Custom instrumentation |

## Experiments

### Experiment 1: Baseline Single-GPU Performance

**Purpose**: Establish baseline performance and understand basic execution pattern

**Configuration**:
```bash
# Launch server with single GPU
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 \
    --mem-fraction-static 0.85
```

**Profiling**:
```bash
# System trace
nsys profile -o sglang_baseline \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    python -m sglang.bench_serving \
    --backend sglang \
    --num-prompts 100 \
    --random-input 512 \
    --random-output 128

# Or use built-in profiler
python -m sglang.profiler --url http://localhost:30000 --num-steps 10
```

**Expected Observations**:
- FlashInfer kernels dominate GPU time
- Clear prefill vs decode kernel selection
- Continuous batching visible in timeline

### Experiment 2: RadixCache Behavior

**Purpose**: Understand prefix caching effectiveness

**Configuration**:
```bash
# Server with radix cache enabled (default)
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000
```

**Profiling**:
```python
# Custom script to test cache hits
import sglang as sgl

@sgl.function
def chat(s, question):
    s += sgl.system("You are a helpful assistant.")  # Shared prefix
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=100))

# Multiple requests with shared prefix
questions = [
    "What is Python?",
    "What is JavaScript?",
    "What is Rust?",
]

# First request: cold start
# Subsequent: should hit radix cache for system prompt
```

**Expected Observations**:
- First request: full prefill
- Subsequent requests: shorter prefill (cache hit on system prompt)
- Cache eviction pattern under memory pressure

### Experiment 3: Attention Backend Comparison

**Purpose**: Compare FlashInfer vs Triton backends

**Configuration**:
```bash
# FlashInfer backend (default)
SGLANG_ATTENTION_BACKEND=flashinfer python -m sglang.launch_server ...

# Triton backend
SGLANG_ATTENTION_BACKEND=triton python -m sglang.launch_server ...
```

**Profiling**:
```bash
# Profile each backend
ncu --set full \
    --kernel-name ".*attention.*" \
    --launch-skip 10 --launch-count 5 \
    -o sglang_flashinfer_ncu \
    python -m sglang.bench_serving --backend sglang --num-prompts 50
```

**Expected Observations**:
- FlashInfer: Lower kernel launch overhead, optimized for A100
- Triton: Higher flexibility, potentially more overhead
- Different SM utilization patterns

### Experiment 4: Tensor Parallelism Scaling

**Purpose**: Measure TP scaling on NVLink pairs

**Configuration**:
```bash
# TP=2 on NVLink pair (GPUs 0-1)
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp 2 \
    --port 30000

# TP=4 on same NUMA (GPUs 0-3)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp 4 \
    --port 30000
```

**Profiling**:
```bash
nsys profile -o sglang_tp2 \
    --trace=cuda,nvtx,nccl \
    --gpu-metrics-device=all \
    python -m sglang.bench_serving --backend sglang --num-prompts 100
```

**Expected Observations**:
- TP=2: AllReduce via NVLink (fast)
- TP=4: Mix of NVLink and PXB (some overhead)
- Communication time vs compute time ratio

### Experiment 5: Continuous Batching Dynamics

**Purpose**: Understand batch formation and scheduling

**Configuration**:
```bash
# High concurrency test
python -m sglang.bench_serving \
    --backend sglang \
    --num-prompts 500 \
    --request-rate 50 \
    --random-input 256 \
    --random-output 128
```

**Profiling**:
```bash
# Enable detailed scheduling logs
SGLANG_LOG_LEVEL=DEBUG python -m sglang.launch_server ...
```

**Expected Observations**:
- Dynamic batch size changes
- Prefill-decode interleaving
- Request preemption under pressure

## Analysis Plan

After collecting data:

1. **Timeline Analysis** (nsys)
   - Extract kernel sequence and timing
   - Identify prefill vs decode transitions
   - Measure CPU-GPU gaps (scheduling overhead)

2. **Kernel Analysis** (ncu)
   - Compare SM utilization across backends
   - Analyze memory bandwidth usage
   - Check tensor core utilization

3. **Cache Analysis**
   - Calculate cache hit rate from logs
   - Measure memory savings from prefix sharing
   - Analyze eviction patterns

4. **Scaling Analysis**
   - Plot throughput vs TP size
   - Measure communication overhead
   - Identify scaling bottlenecks

## Success Criteria

- [ ] Understood: RadixCache hit/miss behavior
- [ ] Measured: Prefill vs decode kernel characteristics
- [ ] Compared: FlashInfer vs Triton backend performance
- [ ] Analyzed: TP scaling with NVLink topology
- [ ] Identified: Key bottlenecks in SGLang serving

## Model Selection

For experiments, use:
- **Llama-3.1-8B-Instruct**: Single GPU baseline, detailed profiling
- **Llama-3.1-70B-Instruct**: TP scaling tests (requires TP≥2)

If models unavailable, alternatives:
- **Qwen2.5-7B-Instruct**: Similar size to Llama-8B
- **Mistral-7B-Instruct-v0.3**: Good single-GPU model
