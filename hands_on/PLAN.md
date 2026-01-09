# Hands-On Learning Sessions Plan

**Status:** Awaiting Approval
**Created:** 2026-01-09
**Environment:** 7x NVIDIA A100 80GB, CUDA 12.5

---

## Plan Summary

This plan covers three hands-on learning sessions for exploring LLM serving infrastructure through profiling and experimentation. Each session follows a standardized directory structure under `hands_on/<project>/` with:

- `scripts/` - Executable profiling and experiment scripts
- `results/` - Profiling outputs (gitignored)
- `reports/` - Generated analysis reports

---

## Core Analysis Principles

**CRITICAL**: These principles must guide ALL analysis and reporting. Final metrics (throughput, latency) are NOT sufficient.

### Principle 1: Process Over Metrics

**Bad Analysis**: "Throughput is 1000 tokens/sec, TTFT is 50ms"

**Good Analysis**: "Here's what happens when a request goes through the system..."
- Timeline of events, not just averages
- Execution story: step-by-step what happens during inference
- Transitions between phases (prefill→decode, cache hit→miss)
- Intermediate states and their implications

**What to capture**:
```
Time 0.0ms: Request arrives
├── CPU: Tokenization (2.1ms) → 512 tokens
├── CPU: Plan phase (0.3ms) → FlashInfer computes work distribution
Time 2.4ms: GPU execution begins
├── Kernel: embedding_lookup (0.08ms) ← SM: 32%, Mem: 78%
├── LAYER 0 (1.2ms):
│   ├── rms_norm (0.02ms) ← SM: 25%, Mem: 85% (memory-bound)
│   ├── qkv_gemm (0.15ms) ← SM: 82%, TC: 48%
│   ├── flash_attention (0.45ms) ← HOTSPOT
│   │   └── Warp specialization: TMA producer + TC consumers
...
```

### Principle 2: Hardware Is Truth

Everything runs on physical hardware. Always track:

| Hardware Aspect | What to Observe | Why It Matters |
|-----------------|-----------------|----------------|
| **SM Utilization** | When active? How many? | Kernel efficiency |
| **Register Pressure** | Spilling to local memory? | Occupancy limiter |
| **Shared Memory** | Bank conflicts? Usage % | Occupancy & latency |
| **Tensor Cores** | Active %? Data feeding? | Compute efficiency |
| **Memory Bandwidth** | HBM %, L2 hit rate | Memory-bound detection |
| **Warp Stalls** | long_scoreboard, barrier, etc. | Bottleneck identification |

**Example kernel analysis (not just metrics)**:
```
Kernel: flash_attention_fwd
├── Occupancy: 47% (limited by SMEM at 96KB/block)
├── Warp stalls:
│   ├── long_scoreboard: 42% ← waiting for HBM loads
│   ├── barrier: 28% ← warp specialization sync
│   └── short_scoreboard: 18% ← some SMEM bank conflicts
├── Interpretation:
│   └── Memory-bound as expected, barrier overhead acceptable
```

### Principle 3: Joint Process+Hardware Analysis

Combine process flow with hardware state at each step:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EXECUTION STORY WITH HARDWARE CONTEXT                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PHASE 1: Prefill (512 tokens)                                          │
│  ══════════════════════════════                                         │
│                                                                         │
│  Event: QKV Projection                                                  │
│  ├── Duration: 0.15ms                                                   │
│  ├── Kernel: cublas_gemm_fp16                                           │
│  ├── Shape: [1, 512, 4096] × [4096, 12288]                             │
│  ├── Hardware:                                                          │
│  │   ├── SM util: 82% (good compute utilization)                       │
│  │   ├── TC active: 48% (data feeding could be better)                 │
│  │   └── HBM BW: 45% (not memory-bound)                                │
│  └── Observation: Compute-bound, TC underutilized due to small batch   │
│                                                                         │
│  Event: Flash Attention (prefill)                                       │
│  ├── Duration: 0.45ms ← HOTSPOT (37% of layer time)                    │
│  ├── Kernel: flash_attention_fwd_sm80                                   │
│  ├── Hardware:                                                          │
│  │   ├── SM util: 68%                                                  │
│  │   ├── TC active: 35%                                                │
│  │   ├── HBM BW: 72% (near memory-bound)                               │
│  │   └── Warp stalls: long_scoreboard 42%, barrier 28%                 │
│  ├── Observation: Mixed compute/memory, warp specialization active     │
│  └── Interpretation: Barrier stalls expected for producer-consumer     │
│                                                                         │
│  TRANSITION: Prefill → Decode                                           │
│  ═══════════════════════════════                                        │
│                                                                         │
│  Event: Kernel switch                                                   │
│  ├── Old kernel: flash_attention_fwd (compute-bound)                   │
│  ├── New kernel: flash_attention_decode (memory-bound)                 │
│  ├── Reason: Single query token vs full sequence                       │
│  └── Hardware change: TC drops to 15%, HBM rises to 82%                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### What Reports MUST Include

Every experiment report MUST contain:

1. **Execution Timeline** (not just summary stats)
   - Event sequence with timestamps
   - Hardware state at each event
   - Transitions between phases

2. **Hardware Behavior Analysis**
   - Per-kernel: occupancy, compute, memory, stalls
   - Roofline position (compute-bound vs memory-bound)
   - Warp stall breakdown with interpretations

3. **Process Visualizations**
   - ASCII timeline/flame graph
   - Memory usage over time
   - Multi-GPU communication patterns (if applicable)

4. **Observations with Hardware Context**
   - "Decode attention is memory-bound (HBM 82%, TC 15%)"
   - NOT just "Decode is slower than prefill"

## Directory Structure

```
hands_on/
├── PLAN.md                     # This file
├── sglang/
│   ├── scripts/
│   │   ├── 00_check_env.py
│   │   ├── 01_baseline_perf.py
│   │   ├── 02_radix_cache_analysis.py
│   │   ├── 03_backend_comparison.py
│   │   ├── 04_nsys_profile.sh
│   │   ├── 05_ncu_attention.sh
│   │   └── 06_tp_scaling.sh
│   ├── results/                # gitignored
│   └── reports/
│       ├── INDEX.md
│       ├── environment.md
│       ├── analysis.md
│       ├── plan.md
│       ├── experiments.md
│       ├── kernel-dev-guide.md
│       └── final-report.md
├── mini-sglang/
│   ├── scripts/
│   │   ├── 00_check_env.py
│   │   ├── 01_test_kernels.py
│   │   ├── 02_profile_index.py
│   │   ├── 03_profile_store.py
│   │   ├── 04_profile_attention.py
│   │   ├── 05_profile_comm.py
│   │   └── 06_full_pipeline.sh
│   ├── results/                # gitignored
│   └── reports/
│       ├── INDEX.md
│       ├── environment.md
│       ├── analysis.md
│       ├── plan.md
│       ├── experiments.md
│       ├── kernel-dev-guide.md
│       └── final-report.md
└── flashinfer/
    ├── scripts/
    │   ├── 00_check_env.py
    │   ├── 01_test_kernels.py
    │   ├── 02_profile_prefill.py
    │   ├── 03_profile_decode.py
    │   ├── 04_profile_rmsnorm.py
    │   ├── 05_profile_rope.py
    │   └── 06_jit_analysis.py
    ├── results/                # gitignored
    └── reports/
        ├── INDEX.md
        ├── environment.md
        ├── analysis.md
        ├── plan.md
        ├── experiments.md
        ├── kernel-dev-guide.md
        └── final-report.md
```

---

## Project 1: SGLang

### Overview

SGLang is a production LLM serving system with advanced features including RadixAttention prefix caching, continuous batching, and multiple attention backends. This session focuses on understanding system-level behavior through end-to-end profiling.

**Code Location:** `/home/uvxiao/mlkb/code-repos/sglang/`

### Learning Goals

1. Understand RadixAttention/prefix caching hit/miss patterns
2. Analyze continuous batching execution dynamics
3. Compare FlashInfer vs Triton attention backends
4. Measure tensor parallelism scaling on NVLink topology
5. Identify system bottlenecks through profiling

### Experiments with Process+Hardware Focus

| Script | Purpose | Process Details to Capture | Hardware to Analyze |
|--------|---------|---------------------------|---------------------|
| `00_check_env.py` | Environment detection | GPU topology, NUMA layout | Memory hierarchy, interconnect |
| `01_baseline_perf.py` | Baseline with execution story | Step-by-step inference flow, phase transitions | Per-phase SM/TC/HBM utilization |
| `02_radix_cache_analysis.py` | Cache behavior | Cache hit→miss transitions, eviction events | Memory access patterns during hits vs misses |
| `03_backend_comparison.py` | Backend differences | Kernel selection logic, plan overhead | Warp stall differences, occupancy |
| `04_nsys_profile.sh` | System timeline | Full execution story with gaps identified | CPU-GPU overlap, kernel sequencing |
| `05_ncu_attention.sh` | Attention kernel deep-dive | Tile scheduling, warp specialization | Occupancy limiters, stall reasons, roofline |
| `06_tp_scaling.sh` | TP communication | AllReduce timing in execution flow | NVLink vs PXB bandwidth, sync overhead |

### Expected Process+Hardware Observations

**01_baseline_perf.py** should produce:
```
EXECUTION STORY: Single Request
├── Tokenization: 2.1ms (CPU)
├── Prefill: 38.4ms
│   ├── Per-layer breakdown with SM/TC/HBM for each kernel
│   └── Hotspot: flash_attention (45% of time, 68% SM, 72% HBM)
├── Decode (×128 tokens): 1109ms
│   └── ITL trend: 8.2ms→10.3ms (linear with KV length)
└── Hardware transition: Prefill TC:35% → Decode TC:15%
```

**02_radix_cache_analysis.py** should produce:
```
CACHE BEHAVIOR STORY:
├── Request 1 (cold): Full prefill, 38ms
│   └── All layers execute attention from scratch
├── Request 2 (warm, shared prefix): Partial prefill, 12ms
│   ├── KV cache hit for first 256 tokens
│   └── Only 256 new tokens computed
└── Hardware difference: Warm request skips 65% of attention compute
```

---

## Project 2: Mini-SGLang

### Overview

Mini-SGLang is a minimal LLM inference framework (~5,000 LOC) designed for understanding core kernel implementations. This session focuses on per-kernel profiling and optimization for kernel developers.

**Code Location:** `/home/uvxiao/mlkb/code-repos/mini-sglang/`

### Learning Goals

1. Understand custom kernel implementations (Index, Store, NCCL, Radix)
2. Profile FlashInfer/FlashAttention integration
3. Analyze per-kernel memory bandwidth and compute efficiency
4. Create tutorial materials for kernel development beginners

### Kernel Catalog

| Kernel | File | Type | Purpose |
|--------|------|------|---------|
| Index | `kernel/csrc/jit/index.cu` | CUDA JIT | Embedding lookup |
| Store | `kernel/csrc/jit/store.cu` | CUDA JIT | KV cache scatter |
| PyNCCL | `kernel/csrc/src/pynccl.cu` | CUDA AOT | All-reduce/gather |
| Radix | `kernel/csrc/src/radix.cpp` | CPU | Prefix matching |

### Experiments with Process+Hardware Focus

| Script | Purpose | Process Details to Capture | Hardware to Analyze |
|--------|---------|---------------------------|---------------------|
| `00_check_env.py` | Environment detection | JIT compilation readiness | TVM-FFI, FlashInfer JIT cache |
| `01_test_kernels.py` | Correctness baseline | Which kernels are called, in what order | - |
| `02_profile_index.py` | Index kernel deep-dive | Warp-level copy pattern, vectorization | Memory coalescing, BW utilization |
| `03_profile_store.py` | Store kernel deep-dive | K,V copy sequence, scatter pattern | Async copy opportunity, BW% |
| `04_profile_attention.py` | FlashInfer integration | Prefill vs decode kernel selection | Occupancy, TC%, warp stalls |
| `05_profile_comm.py` | NCCL patterns | AllReduce timing in layer forward | NVLink utilization, sync stalls |
| `06_full_pipeline.sh` | Full inference story | Complete forward pass timeline | Per-kernel hardware state |

### Expected Process+Hardware Observations

**02_profile_index.py** should produce:
```
INDEX KERNEL ANALYSIS:
├── Operation: Embedding lookup for batch of 64 tokens
├── Execution:
│   ├── 64 warps launched (1 per token)
│   ├── Each warp: vectorized 128-byte loads
│   └── Total: 64 × 4096 × 2 bytes = 512KB read
├── Hardware:
│   ├── Memory BW: 78% of peak (1.56 TB/s)
│   ├── SM util: 32% (memory-bound, expected)
│   └── Coalescing: 100% (perfect access pattern)
└── Bottleneck: Pure memory bandwidth
```

**03_profile_store.py** should produce:
```
STORE KERNEL ANALYSIS:
├── Operation: Scatter K,V to paged cache
├── Execution:
│   ├── K copy: warps read contiguous, write scattered
│   ├── V copy: same pattern, separate kernel launch
│   └── Total: 2 kernel launches, 2 sync points
├── Hardware:
│   ├── Memory BW: 72% per copy
│   └── Observation: Two copies could be fused
├── Optimization opportunity:
│   ├── Current: 2 launches × 72% = 144% BW-time product
│   └── Fused: 1 launch × 85% = 85% (17% savings)
```

### Optimization Opportunities with Hardware Context

| Kernel | Current Behavior | Hardware Observation | Optimization | Expected Impact |
|--------|------------------|---------------------|--------------|-----------------|
| Store | Separate K,V copies | Two kernel launches, sync between | Fuse with async copy | 10-20% BW improvement |
| Index | Fixed 256 threads | Low occupancy at small batch | Dynamic thread count | Better SM utilization |
| NCCL | Symmetric buffer | Extra D2D copy overhead | Direct peer access | Lower latency |

---

## Project 3: FlashInfer

### Overview

FlashInfer is a GPU kernel library for LLM serving using JIT compilation. This session focuses on understanding the attention, normalization, and RoPE kernel implementations.

**Code Location:** `/home/uvxiao/mlkb/code-repos/flashinfer/`

### Learning Goals

1. Understand attention kernel internals (prefill/decode modes)
2. Analyze RMSNorm and fused operations
3. Study RoPE implementation
4. Explore paged KV cache operations
5. Learn the JIT compilation system

### Kernel Catalog

| Kernel | Python API | Purpose |
|--------|------------|---------|
| Prefill Attention | `BatchPrefillWithPagedKVCacheWrapper` | FlashAttention-2 with paged KV |
| Decode Attention | `BatchDecodeWithPagedKVCacheWrapper` | Split-K decode with tensor cores |
| RMSNorm | `flashinfer.rmsnorm()` | Layer normalization |
| Fused Add RMSNorm | `flashinfer.fused_add_rmsnorm()` | Fused residual + norm |
| RoPE | `flashinfer.apply_rope_with_cos_sin_cache_inplace()` | Position embedding |

### Experiments with Process+Hardware Focus

| Script | Purpose | Process Details to Capture | Hardware to Analyze |
|--------|---------|---------------------------|---------------------|
| `00_check_env.py` | Environment + JIT | JIT cache state, compilation triggers | - |
| `01_test_kernels.py` | Correctness baseline | Kernel variant selection | - |
| `02_profile_prefill.py` | Prefill attention deep-dive | Tile scheduling, work distribution | Occupancy (SMEM limit), TC%, warp stalls |
| `03_profile_decode.py` | Decode attention deep-dive | Split-K parallelism, reduction | Memory-bound behavior, long_scoreboard |
| `04_profile_rmsnorm.py` | Normalization deep-dive | Per-row reduction, vectorization | BW utilization, warp reduction |
| `05_profile_rope.py` | RoPE deep-dive | In-place modification pattern | Memory access pattern, BW% |
| `06_jit_analysis.py` | JIT system | Compilation triggers, caching | Compile time overhead |

### Expected Process+Hardware Observations

**02_profile_prefill.py** should produce:
```
PREFILL ATTENTION ANALYSIS (seq_len=512, heads=32):
├── Algorithm: FlashAttention-2 with online softmax
├── Execution:
│   ├── Q tiles: 4 (each 128 tokens)
│   ├── K tiles: 4 (each 128 tokens)
│   ├── Work distribution: 4×32 = 128 items across 108 SMs
│   └── Warp specialization: TMA producer (12%) + TC consumers (88%)
├── Hardware:
│   ├── Occupancy: 47% (limited by SMEM at 96KB/block)
│   ├── TC utilization: 35%
│   ├── HBM BW: 72%
│   └── Warp stalls:
│       ├── long_scoreboard: 42% (waiting for HBM)
│       ├── barrier: 28% (producer-consumer sync)
│       └── short_scoreboard: 18% (SMEM access)
└── Roofline: Near ridge point (mixed compute/memory)
```

**03_profile_decode.py** should produce:
```
DECODE ATTENTION ANALYSIS (batch=32, kv_len=4096):
├── Algorithm: Split-K with reduction
├── Execution:
│   ├── KV split: 8 chunks of 512 tokens
│   ├── Parallel: 8 partial outputs computed
│   └── Reduction: Merge partial_O with LSE
├── Hardware:
│   ├── Occupancy: 75% (not SMEM limited)
│   ├── TC utilization: 15% (GEMV-style, limited)
│   ├── HBM BW: 82% (memory-bound as expected)
│   └── Warp stalls:
│       └── long_scoreboard: 65% (waiting for KV reads)
└── Roofline: Memory-bound region (AI < ridge point)
```

**04_profile_rmsnorm.py** should produce:
```
RMSNORM ANALYSIS:
├── Operation: output = x * rsqrt(mean(x^2) + eps) * weight
├── Execution:
│   ├── One thread block per row (hidden_dim elements)
│   ├── Warp-level reduction for sum(x^2)
│   └── Vectorized load/store (8 elements per thread)
├── Hardware:
│   ├── Memory BW: 85% of peak
│   ├── SM util: 28% (memory-bound)
│   └── Fused variant saves 40% memory traffic
└── Classification: Pure memory-bound kernel
```

### Performance Targets with Hardware Context (A100)

| Kernel | Memory BW Target | Roofline Position | Expected Bottleneck |
|--------|------------------|-------------------|---------------------|
| Prefill Attention | >1.5 TB/s | Ridge point | Mixed (TC + HBM) |
| Decode Attention | >1.6 TB/s | Memory-bound | HBM reads (long_scoreboard) |
| RMSNorm | >1.8 TB/s | Memory-bound | HBM (no compute) |
| RoPE | >1.5 TB/s | Memory-bound | HBM access pattern |

---

## Execution Order

### Phase 1: Environment Setup (All Projects)
1. Create directory structure for all three projects
2. Run environment detection scripts
3. Verify all dependencies

### Phase 2: SGLang (System-Level)
1. Baseline performance benchmarks
2. RadixCache analysis
3. Backend comparison
4. NSys timeline capture
5. TP scaling tests
6. Generate reports

### Phase 3: Mini-SGLang (Kernel-Level)
1. Kernel correctness tests
2. Index kernel profiling
3. Store kernel profiling
4. Attention backend profiling
5. Communication profiling
6. Full pipeline analysis
7. Generate reports

### Phase 4: FlashInfer (Low-Level)
1. Kernel test suite
2. Prefill attention analysis
3. Decode attention analysis
4. Normalization analysis
5. RoPE analysis
6. JIT system analysis
7. Generate reports

---

## Hardware Configuration

```
GPU Configuration:
• 7 × NVIDIA A100 80GB (mix PCIe + SXM4)
• Compute Capability: 8.0 (Ampere)
• Total GPU Memory: 560 GB
• Peak Memory BW: ~2 TB/s (HBM2e)

GPU Topology:
       GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6
GPU0    X    NV12  PXB   PXB   SYS   SYS   SYS
GPU1   NV12   X    PXB   PXB   SYS   SYS   SYS
GPU2   PXB   PXB    X    NV12  SYS   SYS   SYS
GPU3   PXB   PXB   NV12   X    SYS   SYS   SYS
GPU4   SYS   SYS   SYS   SYS    X    PXB   PXB
GPU5   SYS   SYS   SYS   SYS   PXB    X    NV12
GPU6   SYS   SYS   SYS   SYS   PXB   NV12   X

NVLink pairs: (0,1), (2,3), (5,6)
NUMA nodes: GPUs 0-3 on node 0, GPUs 4-6 on node 1

Recommended TP configurations:
• TP=2: Use NVLink pairs (GPU 0-1 or 2-3 or 5-6)
• TP=4: GPUs 0-3 (same NUMA, NV12 + PXB)
• Avoid: Cross-NUMA for latency-sensitive workloads
```

---

## Success Criteria

### SGLang
- [ ] Understood RadixCache hit/miss behavior and cache eviction patterns
- [ ] Measured prefill vs decode kernel characteristics
- [ ] Compared FlashInfer vs Triton backend performance
- [ ] Analyzed TP scaling with NVLink topology
- [ ] Identified key bottlenecks in SGLang serving

### Mini-SGLang
- [ ] Profiled all custom kernels (Index, Store, NCCL, Radix)
- [ ] Analyzed FlashInfer integration patterns
- [ ] Created per-kernel optimization guides
- [ ] Generated tutorial materials for beginners

### FlashInfer
- [ ] Understood attention kernel internals (prefill/decode)
- [ ] Analyzed normalization kernel efficiency
- [ ] Studied RoPE implementation and variants
- [ ] Learned JIT compilation workflow
- [ ] Created kernel development guide

---

## References

- [Existing SGLang Plan](/home/uvxiao/mlkb/reports/hands-on/sglang-20250109-plan.md)
- [Existing Mini-SGLang Report](/home/uvxiao/mlkb/reports/implementations/mini-sglang-hands-on-learning.md)
- [Existing FlashInfer Guide](/home/uvxiao/mlkb/reports/implementations/flashinfer-kernel-dev-guide.md)
