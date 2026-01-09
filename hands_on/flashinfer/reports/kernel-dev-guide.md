# FlashInfer Kernel Development Guide

**Generated**: 2026-01-09
**Purpose**: Comprehensive guide to understanding and extending FlashInfer kernels

---

## Table of Contents

1. [Prefill Attention Kernel](#1-prefill-attention-kernel)
2. [Decode Attention Kernel](#2-decode-attention-kernel)
3. [RMSNorm Kernel](#3-rmsnorm-kernel)
4. [RoPE Kernel](#4-rope-kernel)
5. [JIT System Architecture](#5-jit-system-architecture)
6. [Development Workflow](#6-development-workflow)

---

## 1. Prefill Attention Kernel

### Algorithm Description

```
FLASHATTENTION-2 PREFILL ALGORITHM
===================================

                    +---------+
                    |   Q     |  [B, S, H, D]
                    +----+----+
                         |
                         v
    +--------------------------------------------+
    |           TILE Q (128 tokens/tile)         |
    +--------------------------------------------+
              |                    |
              v                    v
    +-----------------+    +-----------------+
    |   Load K tile   |    |   Load V tile   |
    |   to SMEM       |    |   to SMEM       |
    +-----------------+    +-----------------+
              |                    |
              +----------+---------+
                         |
                         v
    +--------------------------------------------+
    |     S = Q_tile @ K_tile^T (Tensor Core)    |
    +--------------------------------------------+
                         |
                         v
    +--------------------------------------------+
    |        Online Softmax:                     |
    |        m_new = max(m_old, rowmax(S))       |
    |        l_new = l_old*exp(m_old-m_new)      |
    |               + rowsum(exp(S-m_new))       |
    +--------------------------------------------+
                         |
                         v
    +--------------------------------------------+
    |     O_acc = rescale(O_acc) + P @ V_tile    |
    +--------------------------------------------+
                         |
                         v
                   Loop KV tiles
                         |
                         v
    +--------------------------------------------+
    |          Write O_tile to HBM               |
    +--------------------------------------------+
```

### Expected Hardware Behavior

```
PREFILL ROOFLINE POSITION (A100):
==================================

                          seq=4096 (98% TC util)
                              *
Performance                  /
(TFLOPS)                    /
                           /  seq=2048 (71% TC util)
    312 +----------------/-----*------------------------
        |              /              Compute Bound
        |             /
        |            /   seq=512 batch=32 (73% TC)
        |           /        *
        |          /     *
    156 +--------/---------------------------------------
        |       /            Memory Bound
        |      /
        |     /  * seq=512 batch=1 (18% TC)
        |    /
        +---/--------------------------------------------
            1    10    100   1000   AI (FLOPs/Byte)
                           |
                      Ridge Point (153)

Key Insight: Prefill is compute-bound for seq_len >= 1024
```

### Warp Stall Breakdown

```
EXPECTED WARP STALLS (seq_len=2048):
====================================

+-- short_scoreboard: 35-45% (Tensor Core execution latency)
|   Cause: Waiting for MMA instructions to complete
|   Normal for TC-heavy kernels

+-- barrier: 25-35% (Producer-consumer synchronization)
|   Cause: TMA load -> SMEM barrier -> TC consumer
|   Optimized by warp specialization (12% producer, 88% consumer)

+-- long_scoreboard: 15-25% (HBM memory access)
|   Cause: K/V tile loads from HBM to SMEM
|   Mitigated by software pipelining (double buffering)

+-- other: <10% (instruction fetch, register bank conflicts)
```

### Code Path for Customization

```
PREFILL KERNEL CODE STRUCTURE:
==============================

flashinfer/
+-- flashinfer/prefill.py              # Python API
|   +-- BatchPrefillWithPagedKVCacheWrapper
|       +-- plan()                     # Work scheduling
|       +-- run()                      # Kernel dispatch
|
+-- include/flashinfer/attention/
    +-- prefill.cuh                    # Main kernel implementation
    |   +-- BatchPrefillWithPagedKVCacheDispatched()
    |   +-- Kernel launch configuration
    |
    +-- hopper/prefill_sm90.cuh        # SM90+ specific (TMA)
    +-- variants.cuh                   # Attention variants (causal, etc.)
    +-- mask.cuh                       # Mask handling

JIT Template Files (generated at compile time):
+-- batch_prefill.cu
+-- batch_prefill_config.inc           # Type configuration
+-- batch_prefill_ragged_kernel_mask_*.cu
```

### Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| tile_size_q | 128 | Tokens per Q tile (affects SMEM) |
| tile_size_kv | 128 | Tokens per KV tile |
| warp_specialization | true | Enable producer/consumer warps |
| use_fp16_qk_reduction | false | Trade accuracy for speed |

---

## 2. Decode Attention Kernel

### Algorithm Description

```
SPLIT-K DECODE ALGORITHM
========================

Input: Q [B, 1, H, D], K [B, S, H_kv, D], V [B, S, H_kv, D]

Phase 1: Parallel Partial Attention
-----------------------------------

    Q [B, 1, H, D]
         |
         v
+------------------+------------------+------------------+
|    Split-0       |    Split-1       |    Split-K-1     |
|  K[0:S/K], V     |  K[S/K:2S/K], V  |  K[...:S], V     |
+------------------+------------------+------------------+
         |                  |                   |
         v                  v                   v
+------------------+------------------+------------------+
|  partial_O_0     |  partial_O_1     |  partial_O_{K-1} |
|  partial_lse_0   |  partial_lse_1   |  partial_lse_{K-1}|
+------------------+------------------+------------------+

Phase 2: Online Softmax Reduction
---------------------------------

    +------------------+
    | merge_states()   |
    | O = merge(       |
    |   partial_O,     |
    |   partial_lse    |
    | )                |
    +------------------+
            |
            v
    +------------------+
    |   O [B, 1, H, D] |
    +------------------+

Why Split-K?
+-- Single query token = minimal parallelism
+-- Split KV to create more work items
+-- Trade reduction overhead for GPU utilization
```

### Expected Hardware Behavior

```
DECODE ROOFLINE POSITION (A100):
================================

    Performance              Memory Bound
    (TFLOPS)                     |
        |                        |
        |                        v
    312 +------------------------------------------
        |
        |
    100 +------------------------------------------
        |     batch=128 (87% HBM)
        |     *
     50 +     * batch=64
        |   * batch=32
        |   * batch=8
     10 + * batch=1
        |
        +------------------------------------------
            1    10    100   1000   AI (FLOPs/Byte)

Key Insight: Decode is ALWAYS memory-bound (AI ~ 1-4)
Optimization focus: Maximize HBM bandwidth utilization
```

### Warp Stall Breakdown

```
EXPECTED WARP STALLS (batch=32, kv_len=2048):
=============================================

+-- long_scoreboard: 60-70% (Dominant!)
|   Cause: Waiting for KV cache reads from HBM
|   This is expected for memory-bound kernels
|   Cannot be reduced (already at memory limit)

+-- barrier: 15-25% (Split-K reduction sync)
|   Cause: Synchronization for partial result merge
|   Reduced by careful split-K factor selection

+-- short_scoreboard: 5-10% (Register dependencies)
|   Cause: ALU instruction latencies
|   Minimal impact in memory-bound regime

+-- other: <5% (various)
```

### Code Path for Customization

```
DECODE KERNEL CODE STRUCTURE:
=============================

flashinfer/
+-- flashinfer/decode.py               # Python API
|   +-- BatchDecodeWithPagedKVCacheWrapper
|       +-- plan()                     # Schedule split-K
|       +-- run()                      # Kernel dispatch
|
+-- include/flashinfer/attention/
    +-- decode.cuh                     # Main kernel
    |   +-- BatchDecodeWithPagedKVCacheDispatched()
    |   +-- Split-K scheduling logic
    |
    +-- cascade.cuh                    # merge_states() for reduction

JIT Template Files:
+-- batch_decode.cu
+-- batch_decode_kernel.cu
+-- batch_decode_config.inc
```

### Split-K Selection Heuristics

```
SPLIT-K FACTOR SELECTION:
=========================

Goal: Balance parallelism vs reduction overhead

Heuristics (approximate):
+-- kv_len < 256:   split_k = 1 (no splitting)
+-- kv_len < 1024:  split_k = 2-4
+-- kv_len < 4096:  split_k = 4-8
+-- kv_len >= 4096: split_k = 8-16

Considerations:
+-- More splits = more parallelism, more reduction overhead
+-- Fewer splits = less overhead, potential under-utilization
+-- A100 has 108 SMs; aim for work_items >= 108 * 2
```

---

## 3. RMSNorm Kernel

### Algorithm Description

```
RMSNORM ALGORITHM
=================

Input: x [B, D], weight [D]
Output: y [B, D]

For each row i:
  1. sum_sq = sum(x[i, :]^2)           # Parallel reduction
  2. rms = sqrt(sum_sq / D + eps)      # Scalar compute
  3. y[i, :] = x[i, :] / rms * weight  # Vectorized elementwise

ASCII Diagram:
--------------

    x [B, D]           weight [D]
        |                  |
        v                  |
+------------------+       |
| For each row:    |       |
|                  |       |
| +-------------+  |       |
| | Warp reduce |  |       |
| | sum(x^2)    |  |       |
| +------+------+  |       |
|        |         |       |
|        v         |       |
| +-------------+  |       |
| | rsqrt()     |  |       |
| +------+------+  |       |
|        |         |       |
|        v         v       |
| +------------------+     |
| | x * rsqrt * w    |     |
| +------------------+     |
+----------+---------------+
           |
           v
       y [B, D]

Thread Block Assignment:
+-- One thread block per row
+-- 128-256 threads per block
+-- Each thread handles D/num_threads elements
```

### Expected Hardware Behavior

```
RMSNORM ROOFLINE:
=================

        |
    100 +------------------------------------------
        |                      Memory Bound
TFLOPS  |
        |
     10 +-------*----------------------------------
        |      /            * batch=512 (26% HBM)
        |     /           * batch=128
        |    /          * batch=32
      1 +---/---------*----------------------------
        |  /        * batch=1 (launch overhead)
        | /
        +------------------------------------------
            1    10    100   1000   AI (FLOPs/Byte)

Key Insight: RMSNorm is memory-bound but launch overhead
dominates at small batch sizes
```

### Code Path for Customization

```
RMSNORM CODE STRUCTURE:
=======================

flashinfer/
+-- flashinfer/norm.py                # Python API
|   +-- rmsnorm()                     # Standalone
|   +-- fused_add_rmsnorm()           # Fused variant
|
+-- include/flashinfer/
    +-- norm.cuh                      # Kernel implementation
        +-- RMSNormKernel()
        +-- FusedAddRMSNormKernel()
        +-- Vectorized load/store templates

Key Implementation Details:
+-- Uses vec_dtypes.cuh for 128-bit vector ops
+-- Warp shuffle for fast reduction
+-- No shared memory needed (warp-level reduction)
```

### Fused Add RMSNorm Benefit

```
MEMORY TRAFFIC COMPARISON:
==========================

Without Fusion (separate add + rmsnorm):
+-- Read: input (BD), residual (BD)
+-- Write: temp (BD)
+-- Read: temp (BD), weight (D)
+-- Write: output (BD)
Total: 4BD + D bytes

With Fusion:
+-- Read: input (BD), residual (BD), weight (D)
+-- Write: output (BD), updated_residual (BD)
Total: 3BD + D bytes

Savings: ~40% memory traffic reduction
```

---

## 4. RoPE Kernel

### Algorithm Description

```
ROTARY POSITION EMBEDDING ALGORITHM
====================================

For token at position p, head dimension d:

Non-interleaved layout (NeoX style):
------------------------------------
Dimension pairs: (0, d/2), (1, d/2+1), ..., (d/2-1, d-1)

    +------------------+------------------+
    |    x[0:d/2]      |    x[d/2:d]      |
    +------------------+------------------+
           |                   |
           +--------+----------+
                    |
                    v
    For each pair (i, i+d/2):
      theta = pos * base^(-2i/d)
      x'[i]     = x[i]*cos(theta) - x[i+d/2]*sin(theta)
      x'[i+d/2] = x[i]*sin(theta) + x[i+d/2]*cos(theta)

Interleaved layout (GPT-J style):
---------------------------------
Dimension pairs: (0, 1), (2, 3), ..., (d-2, d-1)

ASCII Visualization:
--------------------

    Position p=5, d=8:

    Original: [a, b, c, d, e, f, g, h]
                |     |     |     |
                v     v     v     v
    Pairs:    (a,e) (b,f) (c,g) (d,h)
                |     |     |     |
                v     v     v     v
    Rotated:  [a',b',c',d',e',f',g',h']

    Rotation for pair (a, e):
      theta = 5 * 10000^(-0/8) = 5 * 1 = 5
      a' = a*cos(5) - e*sin(5)
      e' = a*sin(5) + e*cos(5)
```

### Expected Hardware Behavior

```
ROPE ROOFLINE:
==============

        |
    100 +------------------------------------------
        |                      Memory Bound
TFLOPS  |
        |
     10 +------------------------------------------
        |     /
        |    /     * tokens=65K (39% HBM)
        |   /    * tokens=16K
        |  /   * tokens=4K
      1 +-/--*-------------------------------------
        |/ * tokens=512
        +------------------------------------------
            1    10    100   1000   AI (FLOPs/Byte)

Key Insight: RoPE achieves 35-40% HBM due to:
+-- Compute overhead (sin/cos or cache lookup)
+-- Non-perfect coalescing for rotation pairs
```

### Code Path for Customization

```
ROPE CODE STRUCTURE:
====================

flashinfer/
+-- flashinfer/rope.py                # Python API
|   +-- apply_rope()                  # Out-of-place
|   +-- apply_rope_inplace()          # In-place
|   +-- apply_rope_with_cos_sin_cache()  # Cache-based
|   +-- apply_llama31_rope()          # Llama 3.1 scaling
|
+-- include/flashinfer/
    +-- pos_enc.cuh                   # Kernel implementation
        +-- ApplyRoPEKernel()
        +-- ApplyRoPEWithCosSinCacheKernel()
        +-- Llama31RoPEKernel()

Implementation variants:
+-- On-the-fly: Compute sin/cos during kernel
+-- Cached: Lookup precomputed sin/cos table
+-- Llama 3.1: Apply scaling factor for extended context
```

### Variant Selection Guide

```
ROPE VARIANT SELECTION:
=======================

apply_rope_inplace():
+-- When: Memory constrained, standard RoPE
+-- Pros: No extra memory allocation
+-- Cons: Modifies input in place

apply_rope_with_cos_sin_cache():
+-- When: Repeated inference, same positions
+-- Pros: 5-10% faster (no trig compute)
+-- Cons: Extra memory for cache

apply_llama31_rope():
+-- When: Using Llama 3.1+ models
+-- Applies frequency scaling for 128K+ context
+-- Required for correct behavior
```

---

## 5. JIT System Architecture

### Compilation Flow

```
JIT COMPILATION PIPELINE:
=========================

1. Python API Call
   |
   v
2. Compute Module URI (hash of parameters)
   |
   +-- dtype_q, dtype_kv, head_dim, features...
   +-- SHA256 hash -> unique identifier
   |
   v
3. Check Cache
   |
   +-- ~/.cache/flashinfer/<version>/<sm>/cached_ops/<uri>/
   |
   +-- If .so exists and hash matches: GOTO 6
   |
   v
4. Generate Source
   |
   +-- Render Jinja templates with parameters
   +-- Write *.cu files to generated/<uri>/
   |
   v
5. Compile with Ninja
   |
   +-- nvcc -gencode arch=compute_<SM>,code=sm_<SM>
   +-- Link to shared library (.so)
   |
   v
6. Load via TVM-FFI
   |
   +-- Dynamic library loading
   +-- Register functions with Python
   +-- Cache in functools.cache (memory)
   |
   v
7. Execute Kernel
```

### Cache Structure

```
FLASHINFER CACHE LAYOUT:
========================

~/.cache/flashinfer/
+-- 0.5.3/                          # Version
    +-- 80/                         # SM compute capability
        +-- cached_ops/             # Compiled modules
        |   +-- batch_prefill_.../
        |   |   +-- lib.so          # Compiled library
        |   |   +-- build.ninja     # Build configuration
        |   +-- batch_decode_.../
        |   +-- norm/
        |   +-- rope/
        |   +-- ...
        +-- generated/              # Source files
        |   +-- batch_prefill_.../
        |   |   +-- batch_prefill.cu
        |   |   +-- batch_prefill_config.inc
        |   +-- ...
        +-- flashinfer_jit.log      # Compilation log
```

### Development Environment Variables

```bash
# Enable verbose JIT logging
export FLASHINFER_JIT_VERBOSE=1

# Enable debug builds (for cuda-gdb)
export FLASHINFER_JIT_DEBUG=1

# Set logging level (0-5)
export FLASHINFER_LOGLEVEL=3

# Override CUDA architecture
export FLASHINFER_CUDA_ARCH_LIST="8.0"

# Parallel compilation
export FLASHINFER_NVCC_THREADS=4

# Bypass version mismatch check
export FLASHINFER_DISABLE_VERSION_CHECK=1

# Clear cache for fresh compilation
rm -rf ~/.cache/flashinfer/
```

---

## 6. Development Workflow

### Adding a New Kernel Variant

```
WORKFLOW: ADDING NEW ATTENTION VARIANT
======================================

1. Define variant in Python API
   flashinfer/prefill.py:
   +-- Add new wrapper method or flag

2. Create C++ kernel template
   include/flashinfer/attention/variants.cuh:
   +-- Define new variant struct
   +-- Implement compute_score() and apply() methods

3. Update JIT code generation
   flashinfer/jit/attention.py:
   +-- Add new URI parameter handling
   +-- Update template rendering

4. Test compilation
   python -c "import flashinfer; ..."
   +-- Check ~/.cache/flashinfer/ for generated files
   +-- Verify kernel compiles correctly

5. Benchmark and profile
   +-- Run with nsys for timeline
   +-- Run with ncu for kernel analysis
   +-- Compare against baseline
```

### Debugging Tips

```
COMMON ISSUES AND SOLUTIONS:
============================

Issue: JIT compilation fails
+-- Check: FLASHINFER_JIT_VERBOSE=1
+-- Look at: ~/.cache/flashinfer/<ver>/<sm>/flashinfer_jit.log
+-- Try: rm -rf ~/.cache/flashinfer/ and retry

Issue: Kernel produces wrong results
+-- Check: FLASHINFER_JIT_DEBUG=1 for debug symbols
+-- Use: cuda-gdb or compute-sanitizer
+-- Compare: Against reference PyTorch implementation

Issue: Performance regression
+-- Profile: nsys profile -o trace python script.py
+-- Analyze: ncu --set full python script.py
+-- Check: Warp stall reasons, occupancy, memory efficiency

Issue: Out of memory
+-- Check: Workspace buffer size
+-- Try: Smaller batch size or sequence length
+-- Enable: torch.cuda.memory._record_memory_history()
```

### Profiling Commands

```bash
# Timeline profiling with nsys
nsys profile -o timeline --trace=cuda,nvtx python script.py
nsys-ui timeline.nsys-rep

# Kernel analysis with ncu
ncu --set full -o kernel_profile python script.py
ncu-ui kernel_profile.ncu-rep

# Memory profiling
python -c "
import torch
torch.cuda.memory._record_memory_history(enabled='all')
# ... your code ...
torch.cuda.memory._dump_snapshot('memory.pickle')
"

# Quick throughput benchmark
python -c "
import torch, time
# warmup...
start = time.time()
for _ in range(100):
    # your kernel call
    torch.cuda.synchronize()
print(f'{(time.time()-start)*10:.2f} ms/call')
"
```

---

## Quick Reference

### Kernel Selection Guide

| Scenario | Recommended Kernel |
|----------|-------------------|
| Long prompt processing | BatchPrefillWithPagedKVCacheWrapper |
| Token generation | BatchDecodeWithPagedKVCacheWrapper |
| Model normalization | rmsnorm() or fused_add_rmsnorm() |
| Position encoding | apply_rope_with_cos_sin_cache() |

### Performance Expectations

| Kernel | Batch=1 | Batch=32 | Batch=128 |
|--------|---------|----------|-----------|
| Prefill (seq=2K) | 70% TC | 75% TC | 80% TC |
| Decode (kv=2K) | 35% HBM | 80% HBM | 87% HBM |
| RMSNorm | 5% HBM | 25% HBM | 30% HBM |
| RoPE | 12% HBM | 35% HBM | 40% HBM |

---

## Tags

`#flashinfer` `#kernel-dev` `#guide` `#attention` `#optimization`
