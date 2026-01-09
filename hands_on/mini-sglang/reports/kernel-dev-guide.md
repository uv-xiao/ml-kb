# Mini-SGLang Kernel Development Guide

A comprehensive guide for understanding and developing CUDA kernels in Mini-SGLang.

---

## Table of Contents

1. [Overview](#overview)
2. [Index Kernel](#index-kernel)
3. [Store Kernel](#store-kernel)
4. [PyNCCL Wrapper](#pynccl-wrapper)
5. [Radix Kernel](#radix-kernel)
6. [Profiling Commands](#profiling-commands)
7. [Optimization Opportunities](#optimization-opportunities)

---

## Overview

Mini-SGLang implements four custom kernels that handle critical data movement operations in LLM inference:

| Kernel | Type | Purpose | Hardware Target |
|--------|------|---------|-----------------|
| Index | CUDA JIT | Embedding lookup | Memory-bound |
| Store | CUDA JIT | KV cache scatter | Memory-bound |
| PyNCCL | CUDA AOT | All-reduce/gather | Network-bound |
| Radix | CPU AOT | Prefix matching | CPU |

### Kernel Compilation System

Mini-SGLang uses TVM-FFI for kernel compilation:

```
┌─────────────────────────────────────────────────────────────┐
│                    COMPILATION PIPELINE                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Python API  ─────►  TVM-FFI  ─────►  NVCC  ─────►  .so     │
│  (index.py)         (load_jit)       (JIT)       (cache)    │
│                                                              │
│  Template Parameters:                                        │
│  ├── element_size: Embedding dimension × dtype size         │
│  ├── num_splits: Parallelism per token                      │
│  ├── num_threads: Threads per block (default: 128)          │
│  └── use_pdl: Programmatic Dependent Launch                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Index Kernel

### Purpose

The Index kernel performs embedding lookup - gathering rows from a weight matrix based on token indices.

### Code Location

```
python/minisgl/kernel/csrc/jit/index.cu:34-59
```

### Operation Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    INDEX KERNEL OPERATION                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  INPUT:                                                      │
│  ┌────────────────────────────┐    indices = [3, 7, 1, 5]   │
│  │ Embedding Weight Matrix    │                              │
│  │ [vocab_size × embed_dim]   │                              │
│  │                            │                              │
│  │ Row 0: [0.1, 0.2, ...]    │                              │
│  │ Row 1: [0.3, 0.4, ...]    │ ◄── Warp 2                   │
│  │ Row 2: [0.5, 0.6, ...]    │                              │
│  │ Row 3: [0.7, 0.8, ...]    │ ◄── Warp 0                   │
│  │ Row 4: [0.9, 1.0, ...]    │                              │
│  │ Row 5: [1.1, 1.2, ...]    │ ◄── Warp 3                   │
│  │ Row 6: [1.3, 1.4, ...]    │                              │
│  │ Row 7: [1.5, 1.6, ...]    │ ◄── Warp 1                   │
│  └────────────────────────────┘                              │
│                                                              │
│  OUTPUT:                                                     │
│  ┌────────────────────────────┐                              │
│  │ Output [batch × embed_dim] │                              │
│  │                            │                              │
│  │ [0.7, 0.8, ...]  ◄── Row 3 │                              │
│  │ [1.5, 1.6, ...]  ◄── Row 7 │                              │
│  │ [0.3, 0.4, ...]  ◄── Row 1 │                              │
│  │ [1.1, 1.2, ...]  ◄── Row 5 │                              │
│  └────────────────────────────┘                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Warp-Level Copy Pattern

```cpp
// From index.cu:50-56
if (warp_id < num_warps) {
    const auto pos = indices[warp_id / kNumSplits];
    const auto dst = pointer::offset(output, warp_id * kSizePerWarp);
    const auto src = pointer::offset(weight, pos * kSize,
                                     (warp_id % kNumSplits) * kSizePerWarp);
    warp::copy<kSizePerWarp>(dst, src);
}
```

### Vectorization Details

```
┌─────────────────────────────────────────────────────────────┐
│                   WARP COPY VECTORIZATION                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Warp: 32 threads working together                           │
│                                                              │
│  Memory Package Selection (from warp.cuh):                   │
│  ├── kBytes % (16 × 32) == 0  →  uint4 (16 bytes/thread)    │
│  ├── kBytes % (8 × 32) == 0   →  uint2 (8 bytes/thread)     │
│  └── kBytes % (4 × 32) == 0   →  uint1 (4 bytes/thread)     │
│                                                              │
│  Example: embed_dim=4096, dtype=fp16                         │
│  ├── Element size: 4096 × 2 = 8192 bytes                    │
│  ├── Package: uint4 (16 bytes)                               │
│  ├── Bytes per iteration: 16 × 32 = 512 bytes               │
│  └── Iterations: 8192 / 512 = 16 loads per warp             │
│                                                              │
│  Thread Layout:                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ T0  T1  T2  ...  T31 │ T0  T1  T2  ...  T31 │ ...  │    │
│  │ 16B 16B 16B      16B │ 16B 16B 16B      16B │      │    │
│  │     Iteration 0      │     Iteration 1      │      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Expected Hardware Behavior

```
INDEX KERNEL ANALYSIS (batch=64, embed_dim=4096, fp16):
├── Operation: Embedding lookup for 64 tokens
├── Execution:
│   ├── 256 warps launched (4 per token, num_splits=4)
│   ├── Each warp: 2 vectorized uint4 loads
│   └── Total: 64 × 4096 × 2 bytes = 512 KB read
├── Hardware:
│   ├── Memory BW: 70-80% of peak (1.4-1.6 TB/s)
│   ├── SM util: 30-40% (memory-bound, expected)
│   ├── Coalescing: 100% (contiguous row access)
│   └── Warp stalls: long_scoreboard (HBM latency)
└── Bottleneck: Pure memory bandwidth
```

### Profiling Commands

```bash
# Basic profiling
python scripts/02_profile_index.py --batch-size 64 --embedding-dim 4096

# Parameter sweep
python scripts/02_profile_index.py --sweep

# Nsight Compute (detailed)
ncu --set full \
    --target-processes all \
    -o index_profile \
    python -c "
from minisgl.kernel.index import indexing
import torch
w = torch.randn(32000, 4096, dtype=torch.float16, device='cuda')
i = torch.randint(0, 32000, (64,), dtype=torch.int32, device='cuda')
for _ in range(10): o = indexing(w, i)
torch.cuda.synchronize()
"
```

---

## Store Kernel

### Purpose

The Store kernel scatters K,V tensors to non-contiguous positions in the paged KV cache.

### Code Location

```
python/minisgl/kernel/csrc/jit/store.cu:27-53
```

### Operation Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    STORE KERNEL OPERATION                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  INPUT K,V (contiguous):         Cache (scattered write):   │
│  ┌───────────────────┐           ┌───────────────────┐      │
│  │ K[0]: [k0, k1...] │──┐        │ Slot 0: [...]     │      │
│  │ K[1]: [k0, k1...] │──┼──┐     │ Slot 1: [...]     │      │
│  │ K[2]: [k0, k1...] │──┼──┼──┐  │ Slot 2: [K[1]]    │ ◄────│
│  │ K[3]: [k0, k1...] │──┼──┼──┼─►│ Slot 3: [...]     │      │
│  └───────────────────┘  │  │  │  │ Slot 4: [K[3]]    │ ◄────│
│                         │  │  │  │ Slot 5: [...]     │      │
│  indices = [7, 2, 9, 4] │  │  │  │ Slot 6: [...]     │      │
│                         │  │  └─►│ Slot 7: [K[0]]    │ ◄────│
│                         │  │     │ Slot 8: [...]     │      │
│                         │  └────►│ Slot 9: [K[2]]    │ ◄────│
│                         │        └───────────────────┘      │
│                         │                                    │
│  Same pattern for V     │                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Kernel Implementation

```cpp
// From store.cu:42-50
if (warp_id < length) {
    const auto pos = static_cast<const T *>(indices)[warp_id];
    // K copy
    const auto dst_k = pointer::offset(k_cache, pos * kv_cache_stride);
    const auto src_k = pointer::offset(k, warp_id * kv_input_stride);
    warp::copy<kElementSize>(dst_k, src_k);
    // V copy
    const auto dst_v = pointer::offset(v_cache, pos * kv_cache_stride);
    const auto src_v = pointer::offset(v, warp_id * kv_input_stride);
    warp::copy<kElementSize>(dst_v, src_v);
}
```

### Memory Access Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                STORE KERNEL MEMORY ACCESS                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  READ Pattern (Input K,V - CONTIGUOUS):                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Warp 0 reads K[0]                                   │    │
│  │ Warp 1 reads K[1]  ← Adjacent in memory             │    │
│  │ Warp 2 reads K[2]    (perfect coalescing)           │    │
│  │ Warp 3 reads K[3]                                   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  WRITE Pattern (Cache - SCATTERED):                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Warp 0 writes to slot indices[0] (e.g., 7)          │    │
│  │ Warp 1 writes to slot indices[1] (e.g., 2)          │    │
│  │ Warp 2 writes to slot indices[2] (e.g., 9)  ← Non-  │    │
│  │ Warp 3 writes to slot indices[3] (e.g., 4)    contiguous│ │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Coalescing Analysis:                                        │
│  ├── Sequential indices: Perfect coalescing                 │
│  ├── Random indices: Degraded coalescing                    │
│  └── Impact: 10-20% BW reduction with random scatter        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Expected Hardware Behavior

```
STORE KERNEL ANALYSIS (tokens=64, kv_dim=1024, fp16):
├── Operation: Scatter K,V for 64 tokens to paged cache
├── Execution:
│   ├── 64 warps launched (1 per token)
│   ├── Each warp handles both K and V
│   ├── K copy: contiguous read, scattered write
│   └── V copy: same pattern, same kernel
├── Hardware:
│   ├── Memory BW: 60-75% of peak
│   ├── Read coalescing: 100% (contiguous input)
│   ├── Write coalescing: Variable (depends on indices)
│   └── Warp stalls: long_scoreboard (HBM writes)
├── Sequential vs Random:
│   ├── Sequential indices: ~75% BW efficiency
│   └── Random indices: ~60% BW efficiency
└── Bottleneck: Memory bandwidth + write coalescing
```

### Profiling Commands

```bash
# Basic profiling
python scripts/03_profile_store.py --num-tokens 64 --head-dim 128

# Compare scatter patterns
python scripts/03_profile_store.py --compare-patterns

# Parameter sweep
python scripts/03_profile_store.py --sweep
```

---

## PyNCCL Wrapper

### Purpose

The PyNCCL wrapper provides efficient all-reduce and all-gather operations for tensor parallelism.

### Code Location

```
python/minisgl/kernel/csrc/src/pynccl.cu:72-175
```

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   NCCL WRAPPER ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Initialization:                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ NCCLWrapper(rank, world_size, max_bytes, uid)        │    │
│  │ ├── ncclCommInitRank(comm, world_size, uid, rank)    │    │
│  │ ├── ncclMemAlloc(&buf, max_bytes)  ← Symmetric mem   │    │
│  │ └── ncclCommWindowRegister(comm, buf, max_bytes)     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  AllReduce Flow:                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ if (size <= m_max_bytes):                            │    │
│  │   ├── cudaMemcpyAsync(buf, data, D2D)  ← Copy in    │    │
│  │   ├── ncclAllReduce(buf, buf, ...)     ← In-place   │    │
│  │   └── cudaMemcpyAsync(data, buf, D2D)  ← Copy out   │    │
│  │ else:                                                │    │
│  │   └── ncclAllReduce(data, data, ...)   ← Direct     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Memory Layout:                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ GPU 0                    GPU 1                       │    │
│  │ ┌──────────────────┐    ┌──────────────────┐        │    │
│  │ │ Input Tensor     │    │ Input Tensor     │        │    │
│  │ └────────┬─────────┘    └────────┬─────────┘        │    │
│  │          ▼                       ▼                   │    │
│  │ ┌──────────────────┐    ┌──────────────────┐        │    │
│  │ │ Symmetric Buffer │◄──►│ Symmetric Buffer │ NVLink │    │
│  │ └──────────────────┘    └──────────────────┘        │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Symmetric Memory Benefits

```
┌─────────────────────────────────────────────────────────────┐
│               SYMMETRIC MEMORY OPTIMIZATION                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  WITHOUT Symmetric Memory:                                   │
│  ├── Each AllReduce requires buffer registration            │
│  ├── Registration overhead: ~10-50us per call               │
│  └── Cannot use optimized NCCL algorithms                   │
│                                                              │
│  WITH Symmetric Memory (ncclMemAlloc):                       │
│  ├── Pre-registered at initialization                       │
│  ├── Zero registration overhead per call                    │
│  ├── Enables NCCL window-based collectives                  │
│  └── Trade-off: Extra D2D copy for small tensors            │
│                                                              │
│  When to use internal buffer:                                │
│  ├── size <= max_bytes: Use buffer (2 D2D + NCCL)           │
│  └── size > max_bytes: Direct (no D2D, may need reg)        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Expected Hardware Behavior

```
NCCL ALLREDUCE ANALYSIS (tensor=8MB, TP=2, NVLink):
├── Operation: AllReduce with SUM
├── Execution:
│   ├── Buffer strategy: Symmetric memory (size fits)
│   ├── D2D copy in: ~4us (8MB @ 2TB/s)
│   ├── NCCL AllReduce: ~27us (Ring algorithm)
│   └── D2D copy out: ~4us
├── Hardware:
│   ├── Interconnect: NVLink 3.0 (600 GB/s bidirectional)
│   ├── Achieved BW: 85-95% of peak
│   └── Algorithm: Ring for medium messages
└── Total latency: ~35us
```

### Profiling Commands

```bash
# Code analysis (standalone)
python scripts/05_profile_comm.py --analyze

# Multi-GPU profiling (requires torchrun)
torchrun --nproc_per_node=2 scripts/05_profile_comm.py --operation allreduce --size 1048576
```

---

## Radix Kernel

### Purpose

The Radix kernel performs fast CPU-based prefix matching for the RadixCache trie structure.

### Code Location

```
python/minisgl/kernel/csrc/src/radix.cpp:19-39
```

### Algorithm

```cpp
auto fast_compare_key(const tvm::ffi::TensorView a,
                      const tvm::ffi::TensorView b) -> size_t {
    // Uses std::mismatch for SIMD-optimized comparison
    const auto diff_pos = std::mismatch(a_ptr, a_ptr + common_len, b_ptr);
    return static_cast<size_t>(diff_pos.first - a_ptr);
}
```

### Operation Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   RADIX PREFIX MATCHING                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input:                                                      │
│  ├── a = [1, 2, 3, 4, 5]  (new request tokens)              │
│  └── b = [1, 2, 3, 6, 7]  (cached prefix)                   │
│                                                              │
│  Comparison:                                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Index:  0   1   2   3   4                           │    │
│  │ a:     [1] [2] [3] [4] [5]                          │    │
│  │ b:     [1] [2] [3] [6] [7]                          │    │
│  │         ✓   ✓   ✓   ✗                               │    │
│  │                     ▲                                │    │
│  │                     └── First mismatch at index 3   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Output: 3 (length of matching prefix)                       │
│                                                              │
│  RadixCache Usage:                                           │
│  ├── Traverse trie to find longest matching prefix          │
│  ├── Reuse KV cache for matching tokens                     │
│  └── Only compute attention for new tokens                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Performance Characteristics

```
RADIX KERNEL ANALYSIS:
├── Operation: CPU prefix matching
├── Algorithm: std::mismatch (SIMD optimized)
├── Complexity: O(min(|a|, |b|))
├── Performance:
│   ├── Throughput: ~10 GB/s comparison rate
│   ├── Latency: <1us for typical prefix lengths
│   └── CPU-bound (no GPU involvement)
└── Integration:
    ├── Called during scheduler planning
    └── Determines KV cache reuse extent
```

---

## Profiling Commands

### Quick Reference

```bash
# Environment check
python scripts/00_check_env.py

# Kernel correctness tests
python scripts/01_test_kernels.py

# Index kernel profiling
python scripts/02_profile_index.py --sweep

# Store kernel profiling
python scripts/03_profile_store.py --compare-patterns

# Attention profiling
python scripts/04_profile_attention.py --compare

# NCCL analysis
python scripts/05_profile_comm.py --analyze

# Full pipeline
./scripts/06_full_pipeline.sh --all
```

### Nsight Compute Commands

```bash
# Detailed kernel analysis
ncu --set full --target-processes all \
    -o kernel_profile \
    python your_script.py

# Memory access analysis
ncu --section MemoryWorkloadAnalysis \
    -o memory_profile \
    python your_script.py

# Occupancy analysis
ncu --section Occupancy \
    -o occupancy_profile \
    python your_script.py
```

### Nsight Systems Commands

```bash
# Full system timeline
nsys profile --trace=cuda,nvtx,osrt \
    --sample=none \
    -o timeline \
    python your_script.py

# GPU-only trace
nsys profile --trace=cuda \
    -o gpu_trace \
    python your_script.py
```

---

## Optimization Opportunities

### Index Kernel

| Current | Observation | Optimization | Expected Impact |
|---------|-------------|--------------|-----------------|
| Fixed 128 threads | Good for large batches | Dynamic thread count for small batches | +10% for batch<32 |
| 4 splits max | Limited parallelism | More splits for very large embeddings | +5% for dim>8192 |
| No prefetch | Sequential loads | Add software prefetching | +5-10% latency |

### Store Kernel

| Current | Observation | Optimization | Expected Impact |
|---------|-------------|--------------|-----------------|
| Synchronous K,V copy | Sequential pattern | Async copy overlap | +15% throughput |
| Random scatter | Degraded coalescing | Sort indices first | +10% for random patterns |
| Same kernel for K,V | Good | Already fused | N/A |

### NCCL Wrapper

| Current | Observation | Optimization | Expected Impact |
|---------|-------------|--------------|-----------------|
| Extra D2D copy | Overhead for small tensors | Tune threshold | -5% latency |
| Fixed buffer size | May be suboptimal | Dynamic sizing | -10% memory |

### General Guidelines

1. **Memory-bound kernels**: Focus on coalescing and vectorization
2. **Compute-bound kernels**: Focus on occupancy and tensor core utilization
3. **Communication**: Overlap with computation when possible
4. **CPU kernels**: Use SIMD-optimized libraries (std::mismatch already does this)

---

## Appendix: Code References

### Key Files

| File | Purpose |
|------|---------|
| `python/minisgl/kernel/csrc/jit/index.cu` | Index kernel implementation |
| `python/minisgl/kernel/csrc/jit/store.cu` | Store kernel implementation |
| `python/minisgl/kernel/csrc/src/pynccl.cu` | NCCL wrapper implementation |
| `python/minisgl/kernel/csrc/src/radix.cpp` | Radix prefix matching |
| `python/minisgl/kernel/csrc/include/minisgl/warp.cuh` | Warp-level copy utilities |
| `python/minisgl/kernel/csrc/include/minisgl/utils.cuh` | CUDA utilities and PDL |

### Template Parameters

```cpp
// Index Kernel
template <std::size_t element_size,   // bytes per embedding row
          std::size_t num_splits = 1, // warps per token
          std::size_t num_threads = 128,
          std::size_t max_concurrency = 1,
          bool use_pdl = false>

// Store Kernel
template <std::size_t element_size,   // bytes per K/V row
          std::size_t num_threads = 128,
          std::size_t max_concurrency = 1,
          bool use_pdl = false>
```

---

*Generated for Mini-SGLang Hands-On Learning*
