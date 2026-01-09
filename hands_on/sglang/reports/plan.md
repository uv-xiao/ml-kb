# SGLang Experiment Plan

**Generated:** 2026-01-09
**Environment:** 7x A100 80GB (NVLink pairs: 0-1, 2-3, 5-6)

---

## Objectives

This plan defines experiments that capture **process details with hardware behavior** for:
1. RadixCache hit/miss patterns and memory efficiency
2. Attention backend comparison (FlashInfer vs Triton)
3. Continuous batching dynamics
4. Tensor parallelism scaling with NVLink topology

---

## Experiment 1: Baseline Performance with Execution Story

### Goal
Capture end-to-end inference timeline showing phase transitions and hardware state at each stage.

### Script
`scripts/01_baseline_perf.py`

### Execution
```bash
# Start SGLang server (single GPU baseline)
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b-chat-hf \
    --tp 1 \
    --port 30000

# Run baseline benchmark
python scripts/01_baseline_perf.py \
    --url http://localhost:30000 \
    --num-requests 100 \
    --prompt-len 512 \
    --output-len 128
```

### Expected Output Format
```
EXECUTION STORY: Single Request (Llama-2-7B, 512→128 tokens)
════════════════════════════════════════════════════════════════════════════

PHASE 1: Request Processing (CPU)
─────────────────────────────────
Time 0.0ms: Request received
├── Tokenization: 2.1ms
│   └── HuggingFace tokenizer, 512 tokens produced
├── Scheduling: 0.3ms
│   ├── Policy: LPM (Longest Prefix Match)
│   ├── Cache lookup: 0.1ms
│   └── Memory allocation: 0.2ms
└── Total CPU prep: 2.4ms

PHASE 2: Prefill (GPU)
──────────────────────
Time 2.4ms: GPU execution begins
├── Total prefill time: 38.4ms
│
├── Embedding lookup: 0.08ms
│   └── Hardware: SM 32%, HBM 78% (memory-bound)
│
├── Layer 0 breakdown (example):
│   ├── rms_norm: 0.02ms
│   │   └── SM 25%, HBM 85% (pure memory-bound)
│   ├── qkv_gemm: 0.15ms
│   │   └── SM 82%, TC 48% (compute-bound)
│   ├── flash_attention_prefill: 0.45ms  ← HOTSPOT
│   │   ├── SM 68%, TC 35%, HBM 72%
│   │   ├── Warp stalls: long_scoreboard 42%, barrier 28%
│   │   └── Interpretation: Mixed compute/memory
│   ├── o_proj: 0.08ms
│   │   └── SM 75%, TC 40%
│   ├── rms_norm: 0.02ms
│   ├── gate_up_gemm: 0.18ms
│   │   └── SM 80%, TC 45%
│   └── down_proj: 0.12ms
│       └── SM 78%, TC 42%
│
└── Prefill summary:
    ├── Per-layer average: 1.2ms
    ├── Attention fraction: 37%
    └── GEMM fraction: 45%

PHASE 3: Decode (GPU, 128 tokens)
─────────────────────────────────
Time 40.8ms: Decode begins
├── Total decode time: 1109ms (128 tokens)
├── Average ITL: 8.66ms
│
├── ITL trend (sample):
│   ├── Token 1: 8.2ms (KV len=513)
│   ├── Token 32: 8.8ms (KV len=544)
│   ├── Token 64: 9.3ms (KV len=576)
│   ├── Token 128: 10.3ms (KV len=640)
│   └── Observation: Linear increase with KV length
│
├── Decode kernel breakdown:
│   ├── flash_attention_decode: 4.2ms (48% of ITL)
│   │   ├── SM 45%, TC 15%, HBM 82%
│   │   ├── Warp stalls: long_scoreboard 65%
│   │   └── Classification: Memory-bound
│   └── Other ops: 4.5ms (52%)
│
└── Hardware transition from prefill:
    ├── TC utilization: 35% → 15% (significant drop)
    ├── HBM bandwidth: 72% → 82% (increased pressure)
    └── Explanation: GEMV replaces GEMM in decode

PHASE 4: Result Processing (CPU)
────────────────────────────────
Time 1150ms: Generation complete
├── Detokenization: 0.5ms
├── Response streaming: 0.1ms
└── Cache insertion: 0.2ms

SUMMARY
═══════
Total latency: 1152ms
├── TTFT (Time to First Token): 41ms
├── Decode time: 1109ms
├── ITL range: 8.2ms → 10.3ms
└── Hardware utilization shift: Prefill (TC-heavy) → Decode (HBM-heavy)
```

### Metrics to Collect
| Metric | Collection Method | Purpose |
|--------|-------------------|---------|
| TTFT | Python timing | First token latency |
| ITL per token | Python timing | Decode trend |
| Per-layer time | NVTX + nsys | Hotspot identification |
| SM/TC/HBM % | ncu sampling | Hardware state |
| Warp stalls | ncu | Bottleneck analysis |

---

## Experiment 2: RadixCache Analysis

### Goal
Understand cache hit/miss behavior and its impact on prefill computation.

### Script
`scripts/02_radix_cache_analysis.py`

### Scenarios

```
SCENARIO A: Cold Start (No Cache)
─────────────────────────────────
Request 1: "System prompt... User question 1"
├── Prefix match: 0 tokens
├── Full prefill: 38ms (512 tokens)
└── Cache insert: 512 tokens added

SCENARIO B: Warm Start (Shared Prefix)
──────────────────────────────────────
Request 2: "System prompt... User question 2"
├── Prefix match: 256 tokens (system prompt)
├── Partial prefill: 15ms (256 new tokens only)
├── Speedup: 2.5x vs cold start
└── Hardware: Skip 50% of attention compute

SCENARIO C: Full Cache Hit
─────────────────────────────────────
Request 3: "System prompt... User question 1" (exact repeat)
├── Prefix match: 512 tokens (full match)
├── Prefill time: ~0ms (just KV index copy)
└── Speedup: Near-instant TTFT

SCENARIO D: Cache Pressure (Eviction)
─────────────────────────────────────
After many requests, cache fills:
├── Available KV slots: 0
├── Eviction triggered: LRU evicts oldest
├── Eviction cost: ~0.1ms (pointer operations only)
└── New allocation: Fresh slots available
```

### Expected Output Format
```
RADIX CACHE BEHAVIOR ANALYSIS
════════════════════════════════════════════════════════════════════════════

CACHE CONFIGURATION:
├── Total KV slots: 150,000 tokens
├── Page size: 1
├── Eviction policy: LRU
└── Initial state: Empty

REQUEST SEQUENCE:
─────────────────

Request 1: Cold start
├── Input: "System prompt..." (256 tokens)
├── Cache lookup: MISS
├── Prefill: 256 tokens computed
│   ├── Duration: 18.5ms
│   └── Hardware: SM 65%, TC 32%, HBM 68%
├── Cache insert: Node created at root
└── Cache state: 256/150000 tokens (0.2%)

Request 2: Partial hit
├── Input: "System prompt... User message" (384 tokens)
├── Cache lookup: HIT (256 tokens matched)
├── Prefill: 128 new tokens computed
│   ├── Duration: 8.2ms (vs 22ms cold)
│   ├── Speedup: 2.7x
│   └── Hardware: SM 58%, TC 28%, HBM 72%
├── Cache state: Tree extended with new branch
└── Cache state: 384/150000 tokens (0.3%)

...

Request 50: Eviction scenario
├── Input: New unrelated prompt
├── Cache lookup: MISS
├── KV pool status: 149,000/150000 used
├── Eviction triggered: Yes
│   ├── Tokens evicted: 512 (oldest LRU node)
│   ├── Eviction time: 0.08ms
│   └── Freed indices returned to pool
└── Prefill proceeds with fresh allocation

CACHE EFFICIENCY SUMMARY:
─────────────────────────
Total requests: 100
├── Full hits: 12 (12%)
├── Partial hits: 45 (45%)
├── Misses: 43 (43%)
│
├── Avg prefix match length: 187 tokens
├── Total compute saved: 18,700 tokens
└── Time saved: ~850ms total
```

---

## Experiment 3: Backend Comparison

### Goal
Compare FlashInfer vs Triton attention backends with identical workloads.

### Script
`scripts/03_backend_comparison.py`

### Configuration
```python
CONFIGS = [
    {"backend": "flashinfer", "batch_sizes": [1, 8, 32], "seq_lens": [512, 2048, 8192]},
    {"backend": "triton", "batch_sizes": [1, 8, 32], "seq_lens": [512, 2048, 8192]},
]
```

### Expected Output Format
```
ATTENTION BACKEND COMPARISON
════════════════════════════════════════════════════════════════════════════

TEST MATRIX: Llama-2-7B, A100 80GB

PREFILL PERFORMANCE:
────────────────────

Batch=1, SeqLen=512:
┌──────────────┬───────────┬───────────┬───────────┬───────────┬───────────┐
│ Backend      │ Time (ms) │ SM Util % │ TC Util % │ HBM BW %  │ Warp Stall│
├──────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ FlashInfer   │ 0.42      │ 68        │ 35        │ 72        │ long_sb 42│
│ Triton       │ 0.48      │ 62        │ 30        │ 68        │ long_sb 48│
└──────────────┴───────────┴───────────┴───────────┴───────────┴───────────┘
Winner: FlashInfer (+14%)

Batch=32, SeqLen=2048:
┌──────────────┬───────────┬───────────┬───────────┬───────────┬───────────┐
│ Backend      │ Time (ms) │ SM Util % │ TC Util % │ HBM BW %  │ Warp Stall│
├──────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ FlashInfer   │ 12.8      │ 75        │ 42        │ 65        │ barrier 32│
│ Triton       │ 14.2      │ 70        │ 38        │ 62        │ barrier 35│
└──────────────┴───────────┴───────────┴───────────┴───────────┴───────────┘
Winner: FlashInfer (+11%)

DECODE PERFORMANCE:
───────────────────

Batch=32, KVLen=2048:
┌──────────────┬───────────┬───────────┬───────────┬───────────┬───────────┐
│ Backend      │ Time (ms) │ SM Util % │ TC Util % │ HBM BW %  │ Warp Stall│
├──────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ FlashInfer   │ 2.8       │ 52        │ 18        │ 80        │ long_sb 62│
│ Triton       │ 3.1       │ 48        │ 15        │ 75        │ long_sb 68│
└──────────────┴───────────┴───────────┴───────────┴───────────┴───────────┘
Winner: FlashInfer (+11%)

ANALYSIS:
─────────
1. FlashInfer consistently faster (10-15%)
2. Both backends memory-bound in decode
3. FlashInfer achieves higher TC utilization in prefill
4. Triton easier to customize for special attention patterns
5. Gap narrows with larger batch sizes (better GPU utilization)
```

---

## Experiment 4: Nsight Systems Profile

### Goal
Capture full system timeline with GPU/CPU overlapping.

### Script
`scripts/04_nsys_profile.sh`

### Commands
```bash
#!/bin/bash
# 04_nsys_profile.sh

MODEL="meta-llama/Llama-2-7b-chat-hf"
OUTPUT_DIR="../results/nsys"
mkdir -p $OUTPUT_DIR

# Profile server startup + warmup + requests
nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --gpuctxsw=true \
    --force-overwrite=true \
    -o $OUTPUT_DIR/sglang_trace \
    python -m sglang.launch_server \
        --model-path $MODEL \
        --tp 1 \
        --port 30000 \
        --log-level info \
        -- &

# Wait for server to start
sleep 30

# Send test requests
python scripts/send_test_requests.py --num-requests 10

# Stop server
kill %1
```

### Expected Timeline Analysis
```
NSIGHT SYSTEMS TIMELINE ANALYSIS
════════════════════════════════════════════════════════════════════════════

TIMELINE OVERVIEW (10 requests):
────────────────────────────────

Time (ms)  CPU Thread           GPU Stream 0         GPU Stream 1
────────────────────────────────────────────────────────────────────────────
0-5        recv_requests()      (idle)               (idle)
5-10       schedule_batch()     (idle)               (idle)
10-50      process_last_batch() ▓▓▓ PREFILL REQ1 ▓▓▓ (idle)
50-55      schedule_next()      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   (idle)
55-65      process_result()     ▓ DECODE REQ1 ▓▓▓▓▓  (idle)
65-105     schedule_batch()     ▓▓▓ PREFILL REQ2 ▓▓▓ ▓ DECODE REQ1 ▓
...

KERNEL BREAKDOWN:
─────────────────
Top 5 kernels by total time:
1. flash_attention_fwd_sm80      │ 45% │ 1,234 calls │ avg 0.42ms
2. cublas_gemm_fp16             │ 32% │ 9,600 calls │ avg 0.08ms
3. flash_attention_decode_sm80   │ 12% │   890 calls │ avg 0.32ms
4. rms_norm_kernel              │  5% │ 1,920 calls │ avg 0.02ms
5. embedding_kernel             │  2% │   100 calls │ avg 0.08ms

CPU-GPU OVERLAP EFFICIENCY:
───────────────────────────
├── CPU idle during GPU: 85% (good overlap)
├── GPU idle during CPU: 5% (minimal bubbles)
└── Scheduling overhead: ~2ms per batch

MEMORY TIMELINE:
────────────────
├── Peak GPU memory: 24.5 GB
├── KV cache allocation events: 100
├── KV cache eviction events: 0
└── Memory fragmentation: Low
```

---

## Experiment 5: NCU Attention Kernel Deep-Dive

### Goal
Detailed kernel-level analysis of attention kernels.

### Script
`scripts/05_ncu_attention.sh`

### Commands
```bash
#!/bin/bash
# 05_ncu_attention.sh

OUTPUT_DIR="../results/ncu"
mkdir -p $OUTPUT_DIR

# Profile attention kernels only
ncu \
    --set full \
    --import-source yes \
    --kernel-regex ".*flash.*attention.*|.*decode.*attention.*" \
    --launch-count 20 \
    --target-processes all \
    -o $OUTPUT_DIR/attention_profile \
    python scripts/attention_microbench.py
```

### Expected Kernel Analysis
```
NCU ATTENTION KERNEL ANALYSIS
════════════════════════════════════════════════════════════════════════════

KERNEL: flash_attention_fwd_sm80 (Prefill)
──────────────────────────────────────────

Launch Configuration:
├── Grid: (num_heads × num_q_tiles, batch_size, 1)
├── Block: (128, 1, 1) = 128 threads
└── Shared Memory: 96 KB per block

Occupancy Analysis:
├── Theoretical Occupancy: 50%
├── Achieved Occupancy: 47%
├── Limiter: Shared Memory (96KB > 48KB per SM limit for >50%)
└── Waves: 2.3 (good coverage)

Compute Analysis:
├── SM Utilization: 68%
├── Tensor Core Utilization: 35%
│   ├── FP16 MMA issued: 1.2M
│   └── FP16 MMA efficiency: 78%
├── FP32 utilization: 12% (softmax, scaling)
└── Roofline position: Near ridge point

Memory Analysis:
├── HBM Bandwidth: 1.44 TB/s (72% of peak)
├── L2 Hit Rate: 45%
├── Shared Memory Bandwidth: 12.8 TB/s
│   └── Bank Conflicts: 8% (acceptable)
├── Load Efficiency:
│   ├── Global load: 92% (good coalescing)
│   └── Shared load: 88%
└── Store Efficiency: 95%

Warp Stall Analysis:
├── long_scoreboard: 42% ← Waiting for HBM loads
│   └── Interpretation: Memory latency hiding imperfect
├── barrier: 28% ← Warp specialization sync
│   └── Interpretation: Producer-consumer overhead
├── short_scoreboard: 18% ← SMEM bank conflicts
│   └── Could improve with different SMEM layout
└── not_selected: 12% ← Low occupancy effect

───────────────────────────────────────────────────────────────────────────

KERNEL: flash_attention_decode_sm80 (Decode)
────────────────────────────────────────────

Launch Configuration:
├── Grid: (batch_size × num_heads, num_kv_splits, 1)
├── Block: (128, 1, 1)
└── Shared Memory: 48 KB per block

Occupancy Analysis:
├── Theoretical Occupancy: 75%
├── Achieved Occupancy: 72%
├── Limiter: Registers (64 registers per thread)
└── Waves: 4.5 (good coverage)

Compute Analysis:
├── SM Utilization: 52%
├── Tensor Core Utilization: 18%
│   └── Limited by GEMV-style operation (not true GEMM)
├── Roofline position: Memory-bound region
└── Arithmetic Intensity: 0.8 FLOP/byte (below ridge point)

Memory Analysis:
├── HBM Bandwidth: 1.64 TB/s (82% of peak)
├── L2 Hit Rate: 22% (KV cache too large for L2)
├── Memory-bound confirmation: 82% > 50% threshold
└── Bottleneck: HBM read bandwidth for KV cache

Warp Stall Analysis:
├── long_scoreboard: 65% ← Dominant (waiting for KV)
├── barrier: 15% ← Split-K reduction sync
├── short_scoreboard: 10%
└── Interpretation: Pure memory-bound, optimize HBM access

───────────────────────────────────────────────────────────────────────────

COMPARISON SUMMARY:
───────────────────
                    Prefill                 Decode
──────────────────────────────────────────────────────────
TC Utilization      35%                     18%
HBM Bandwidth       72%                     82%
Roofline            Ridge point             Memory-bound
Primary Stall       long_scoreboard (42%)   long_scoreboard (65%)
Optimization Focus  SMEM banking            HBM prefetch
```

---

## Experiment 6: Tensor Parallelism Scaling

### Goal
Measure TP scaling efficiency on NVLink topology.

### Script
`scripts/06_tp_scaling.sh`

### Configurations
```bash
# TP=2 on NVLink pair (optimal)
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server --tp 2 ...

# TP=4 on NUMA 0 (NVLink + PCIe)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server --tp 4 ...

# TP=4 cross-NUMA (worst case)
CUDA_VISIBLE_DEVICES=0,1,4,5 python -m sglang.launch_server --tp 4 ...
```

### Expected Output Format
```
TENSOR PARALLELISM SCALING ANALYSIS
════════════════════════════════════════════════════════════════════════════

MODEL: Llama-2-70B
BATCH: 32 requests, 512 input tokens each

CONFIGURATION COMPARISON:
─────────────────────────

TP=1 (Baseline, Single GPU):
├── TTFT: 185ms
├── Decode ITL: 42ms
├── Total time: 5.6s
└── Throughput: 720 tokens/s

TP=2 (NVLink pair 0-1):
├── TTFT: 98ms (1.9x speedup)
├── Decode ITL: 24ms (1.75x speedup)
├── Total time: 3.1s
├── Throughput: 1,290 tokens/s
├── Scaling efficiency: 90%
└── AllReduce overhead: 2.1ms per layer

TP=4 (NUMA 0, GPUs 0-3):
├── TTFT: 58ms (3.2x speedup)
├── Decode ITL: 15ms (2.8x speedup)
├── Total time: 2.0s
├── Throughput: 2,000 tokens/s
├── Scaling efficiency: 69%
├── AllReduce overhead: 4.8ms per layer
│   ├── NVLink pairs: 0.8ms
│   └── PCIe bridge: 4.0ms
└── Bottleneck: PCIe between pairs

TP=4 (Cross-NUMA, GPUs 0,1,4,5):
├── TTFT: 72ms (2.6x speedup)
├── Decode ITL: 22ms (1.9x speedup)
├── Total time: 2.9s
├── Throughput: 1,380 tokens/s
├── Scaling efficiency: 48%
├── AllReduce overhead: 8.5ms per layer
│   └── QPI/UPI crossing adds 4ms
└── Recommendation: AVOID for latency-sensitive

COMMUNICATION ANALYSIS:
───────────────────────

AllReduce per layer (Llama-70B, hidden=8192, FP16):
├── Data volume: 8192 × 2 = 16 KB per tensor
├── Tensors per layer: 2 (attention output, FFN output)
│
├── TP=2 (NVLink):
│   ├── Theoretical: 16KB ÷ 300GB/s = 0.05ms
│   ├── Actual: 0.8ms (startup overhead dominates)
│   └── Efficiency: 6% of peak (small tensor penalty)
│
├── TP=4 (NUMA 0, mixed):
│   ├── Ring AllReduce: 3 hops
│   ├── Slowest link: PCIe bridge (25 GB/s)
│   └── Actual: 4.8ms (PCIe limited)
│
└── TP=4 (Cross-NUMA):
    ├── Ring AllReduce: 3 hops
    ├── Slowest link: QPI (10 GB/s)
    └── Actual: 8.5ms (QPI bottleneck)

SCALING VISUALIZATION:
──────────────────────

Throughput (tokens/s)
     │
2500 ┤
2000 ┤                    ●──────● TP=4 NUMA0
1500 ┤          ●────────●
1290 ┤          ▲─────────────────────────● TP=4 Cross-NUMA
1000 ┤          │ TP=2 NVLink
 720 ┤●─────────│
     │ TP=1    │
     └──────────────────────────────────────
         1       2               4       → TP Size

RECOMMENDATIONS:
────────────────
1. TP=2 on NVLink: Best latency/efficiency for 70B models
2. TP=4 NUMA-local: Acceptable for throughput, 30% efficiency loss
3. Avoid cross-NUMA TP unless model requires it
4. For 70B models on 80GB GPUs: TP=4 minimum for memory fit
```

---

## Profiling Commands Reference

### Quick Profiling
```bash
# Basic nsys trace
nsys profile -o trace python script.py

# Basic ncu report
ncu --set full -o report python script.py
```

### Detailed Profiling
```bash
# Full nsys with all traces
nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas,cusparse \
    --cuda-memory-usage=true \
    --gpuctxsw=true \
    --capture-range=cudaProfilerApi \
    -o detailed_trace \
    python script.py

# Full ncu with source correlation
ncu \
    --set full \
    --import-source yes \
    --source-folder /path/to/source \
    --kernel-regex ".*" \
    --launch-count 50 \
    -o full_report \
    python script.py
```

### Specific Kernel Profiling
```bash
# Attention only
ncu --kernel-regex ".*attention.*" -o attention python script.py

# GEMM only
ncu --kernel-regex ".*gemm.*|.*cublas.*" -o gemm python script.py
```

---

## Success Criteria

| Experiment | Success Metric |
|------------|----------------|
| 1. Baseline | Captured per-phase hardware metrics |
| 2. RadixCache | Measured hit rate and compute savings |
| 3. Backend Comparison | Quantified FlashInfer vs Triton gap |
| 4. Nsys Profile | Identified CPU-GPU overlap efficiency |
| 5. NCU Attention | Explained warp stall patterns |
| 6. TP Scaling | Measured AllReduce overhead per config |

---

## Next Steps

1. Run scripts in order (00 → 06)
2. Collect results in `results/` directory
3. Document findings in `reports/experiments.md`
4. Synthesize in `reports/final-report.md`
