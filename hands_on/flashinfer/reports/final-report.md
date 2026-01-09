# FlashInfer Hands-On Learning: Final Report

**Project**: FlashInfer Kernel Analysis and Development Guide
**Hardware**: 7x NVIDIA A100 80GB (SM 8.0)
**FlashInfer Version**: 0.5.3
**Date**: 2026-01-09

---

## Executive Summary

This hands-on learning session analyzed FlashInfer's core kernels through systematic profiling
and hardware-level analysis. We examined five kernel families:

1. **Prefill Attention** (FlashAttention-2) - Achieves 98.5% Tensor Core utilization at long sequences
2. **Decode Attention** (Split-K) - Reaches 87% HBM bandwidth at large batch sizes
3. **RMSNorm** - Pure memory-bound, limited by launch overhead at small batches
4. **RoPE** - Memory-bound with 35-40% HBM efficiency
5. **JIT System** - Two-level caching with 10-500ms first-call overhead

The analysis follows the principle of **joint process+hardware observation**: every finding
combines execution behavior with underlying hardware state.

---

## Key Findings

### 1. Prefill Attention: Compute-Bound Excellence

```
PREFILL ATTENTION ANALYSIS (seq_len=4096, heads=32):
+-- Algorithm: FlashAttention-2 with online softmax
+-- Execution:
|   +-- Q tiles: 32 (each 128 tokens)
|   +-- K tiles: 32 (each 128 tokens)
|   +-- Work distribution: 1024 items across 108 SMs
|   +-- Work per SM: 9.5 items
+-- Hardware:
|   +-- Compute: 307.3 TFLOPS (98.5% of peak)
|   +-- Memory BW: 150 GB/s (7% of peak)
|   +-- Arithmetic Intensity: 2048 FLOPs/Byte
+-- Roofline: Compute-bound (ridge ~153 FLOPs/Byte)
```

**Insight**: FlashInfer's prefill attention is highly optimized for long sequences.
At seq_len=4096, it utilizes nearly 100% of the A100's Tensor Core capacity.
Shorter sequences (seq_len=512) show lower utilization due to insufficient work
to saturate all 108 SMs.

### 2. Decode Attention: Memory-Bound Reality

```
DECODE ATTENTION ANALYSIS (batch=128, kv_len=2048):
+-- Algorithm: Split-K with reduction
+-- Execution:
|   +-- KV split: 4 chunks per sequence
|   +-- Work items: 4096 (batch * heads)
|   +-- Reduction: Merge partial_O with LSE
+-- Hardware:
|   +-- Compute: 1.77 TFLOPS (0.6% of peak)
|   +-- Memory BW: 1771 GB/s (87% of peak)
|   +-- Arithmetic Intensity: 1.0 FLOPs/Byte
+-- Roofline: Memory-bound (153x below ridge point)
```

**Insight**: Decode attention is fundamentally GEMV-like and cannot utilize Tensor
Cores effectively. FlashInfer's Split-K strategy maximizes HBM bandwidth utilization
by creating artificial parallelism. At batch=128, we achieve 87% of the A100's
2039 GB/s theoretical bandwidth.

### 3. The Prefill vs Decode Gap

```
              PREFILL              DECODE
              =======              ======
Shape:        [B,S,H,D] x [B,S,H,D]   [B,1,H,D] x [B,K,H,D]
Operation:    GEMM                 GEMV
AI:           256-2048            1-4
TC Util:      50-98%              <1%
HBM Util:     10-40%              70-90%
Bottleneck:   Compute             Memory
```

**Implication**: Serving systems must carefully balance prefill and decode scheduling.
Prefill needs large batches of long sequences for efficiency; decode benefits from
maximum batching to amortize per-token memory access.

### 4. Auxiliary Kernels: Memory-Bound but Efficient

| Kernel | Peak HBM | Notes |
|--------|----------|-------|
| RMSNorm | 26% | Launch overhead dominates at small batch |
| Fused Add RMSNorm | 24% | 40% less traffic than separate ops |
| RoPE In-Place | 39% | Compute overhead (sin/cos) |
| RoPE with Cache | 40% | Slightly faster with precomputed |

**Insight**: Auxiliary kernels achieve 25-40% of peak HBM bandwidth. The gap from
100% is due to:
- Launch overhead (especially at small batch sizes)
- Compute overhead (trig functions in RoPE)
- Imperfect coalescing

Fusing operations (like fused_add_rmsnorm) provides significant benefits by reducing
memory traffic.

### 5. JIT Compilation: Startup Cost

```
JIT COMPILATION LATENCY:
========================

Scenario               | Latency
-----------------------|----------
First call (cold JIT)  | 10-500 ms
Disk cache hit         | <10 ms
Memory cache hit       | <0.1 ms

Cache Location: ~/.cache/flashinfer/0.5.3/80/
Current Size: 16.8 MB (6 modules)
```

**Recommendation**: Pre-warm the JIT cache at server startup by running inference
with representative configurations.

---

## Roofline Summary

```
A100 Roofline Analysis
======================

      |                         x Prefill seq=4096 (98% TC)
  300 |                       x   Prefill seq=2048 (71% TC)
      |                     x
TFLOPS|                   x
      |                 x     Prefill seq=512 (73% TC at batch=32)
  200 +--------------x-----------------------------------------
      |            /                Compute-Bound Region
      |          /
      |        /
  100 +------/---------------------------------------------
      |     /                      Memory-Bound Region
      |    /     x Decode batch=128 (87% HBM)
      |   /    x Decode batch=64
   50 +  /   x Decode batch=32
      | /  x RoPE
      |/ x RMSNorm
    0 +-+--x--+---------+---------+---------+------------
          1   10       100      1000     10000
                   Arithmetic Intensity (FLOPs/Byte)

Ridge Point: ~153 FLOPs/Byte (312 TFLOPS / 2039 GB/s)
```

---

## Recommendations

### For Serving System Developers

1. **Continuous Batching with Mixed Prefill/Decode**
   - Interleave prefill and decode requests to maximize GPU utilization
   - Prefill benefits from long sequences; decode benefits from large batches

2. **Use GQA Models When Possible**
   - 32/8 GQA ratio: 82K tokens/sec vs 53K for MHA (1.5x improvement)
   - Reduces KV cache memory and decode attention traffic

3. **Pre-warm JIT Cache**
   - Run representative configurations at startup
   - Avoids 10-500ms latency spike on first user request

4. **Fuse Operations**
   - Use `fused_add_rmsnorm()` instead of separate `add` + `rmsnorm`
   - Saves 40% memory traffic in residual connections

### For Kernel Developers

1. **Profile Before Optimizing**
   - Use nsys for timeline analysis
   - Use ncu for kernel-level metrics
   - Focus on the actual bottleneck (TC vs HBM)

2. **Understand the Roofline Position**
   - Compute-bound: Optimize for TC utilization, warp scheduling
   - Memory-bound: Optimize for coalescing, vectorization, bandwidth

3. **Consider Warp Stall Analysis**
   - `long_scoreboard` dominant -> Memory-bound
   - `short_scoreboard` dominant -> Compute-bound
   - `barrier` dominant -> Synchronization overhead

---

## Files Generated

### Scripts

| File | Purpose |
|------|---------|
| `scripts/00_check_env.py` | Environment detection and JIT verification |
| `scripts/02_profile_prefill.py` | Prefill attention profiling |
| `scripts/03_profile_decode.py` | Decode attention profiling |
| `scripts/04_profile_rmsnorm.py` | RMSNorm profiling |
| `scripts/05_profile_rope.py` | RoPE profiling |
| `scripts/06_jit_analysis.py` | JIT system analysis |

### Reports

| File | Purpose |
|------|---------|
| `reports/INDEX.md` | Navigation and overview |
| `reports/environment.md` | Hardware and software environment |
| `reports/analysis.md` | Detailed profiling results |
| `reports/kernel-dev-guide.md` | Development and customization guide |
| `reports/final-report.md` | This executive summary |

### Results (JSON)

| File | Contents |
|------|----------|
| `results/prefill_results.json` | Prefill profiling data |
| `results/decode_results.json` | Decode profiling data |
| `results/rmsnorm_results.json` | RMSNorm profiling data |
| `results/rope_results.json` | RoPE profiling data |
| `results/jit_analysis.json` | JIT system analysis |
| `results/environment_info.json` | System configuration |

---

## Conclusion

FlashInfer provides highly optimized kernels for LLM inference:

- **Prefill attention** achieves near-peak Tensor Core utilization (98.5%) through
  FlashAttention-2's tiled algorithm with online softmax
- **Decode attention** reaches 87% HBM bandwidth through Split-K parallelization
- **Auxiliary kernels** (RMSNorm, RoPE) are memory-bound but efficiently implemented

The JIT compilation system adds flexibility at the cost of first-call latency,
which can be mitigated through cache pre-warming.

For maximum throughput:
1. Batch decode requests (53K tokens/sec at batch=128 vs 28K at batch=1)
2. Use GQA models (1.5x improvement)
3. Fuse operations where possible

This analysis provides the foundation for understanding FlashInfer's performance
characteristics and developing optimizations for specific use cases.

---

## Tags

`#flashinfer` `#final-report` `#hands-on` `#profiling` `#attention` `#llm-inference`
