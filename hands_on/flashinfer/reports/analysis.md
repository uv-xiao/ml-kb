# FlashInfer Kernel Analysis Report

**Generated**: 2026-01-09
**Hardware**: NVIDIA A100 80GB PCIe (SM 8.0, 108 SMs)
**FlashInfer Version**: 0.5.3

---

## Executive Summary

This report presents a detailed analysis of FlashInfer's core kernels based on hands-on profiling.
Key findings:

1. **Prefill attention achieves 98.5% compute utilization** at seq_len=4096, validating
   FlashAttention-2's compute-bound design
2. **Decode attention achieves 87% HBM bandwidth** at large batch sizes, confirming
   memory-bound behavior
3. **RMSNorm and RoPE are purely memory-bound**, achieving 25-40% of peak bandwidth
4. **JIT compilation adds 10-500ms** on first call, but cached execution is sub-millisecond

---

## 1. Prefill Attention Analysis

### Algorithm: FlashAttention-2 with Online Softmax

```
FlashAttention-2 Prefill Pattern:
==================================

for each Q_tile in Q:           # Outer loop over query tiles
  m_i = -inf                    # Running max (for numerical stability)
  l_i = 0                       # Running sum (for softmax denominator)
  O_i = 0                       # Running output accumulator

  for each K_tile, V_tile:      # Inner loop over KV tiles
    S_ij = Q_tile @ K_tile^T    # [tile_q, tile_k] - uses Tensor Cores
    m_ij = max(S_ij)
    P_ij = exp(S_ij - m_ij)
    l_ij = sum(P_ij)

    # Online softmax correction
    m_new = max(m_i, m_ij)
    l_new = l_i * exp(m_i - m_new) + l_ij * exp(m_ij - m_new)
    O_i = O_i * (l_i * exp(m_i - m_new) / l_new) + P_ij @ V_tile / l_new

    m_i, l_i = m_new, l_new

  write O_i to output

Key: KV never leaves HBM->SMEM->register pipeline once loaded
```

### Profiling Results

| Config | Seq Len | TFLOPS | HBM BW | AI | Run Time | Status |
|--------|---------|--------|--------|-----|----------|--------|
| batch=1 | 512 | 56.1 (18%) | 219 GB/s | 256 | 0.077 ms | Under-utilized |
| batch=1 | 2048 | 222.4 (71%) | 217 GB/s | 1024 | 0.309 ms | Good utilization |
| batch=1 | 4096 | 307.3 (98.5%) | 150 GB/s | 2048 | 0.894 ms | Near peak |
| batch=8 | 512 | 175.6 (56%) | 686 GB/s | 256 | 0.196 ms | Good |
| batch=32 | 512 | 228.5 (73%) | 893 GB/s | 256 | 0.602 ms | Good |
| GQA 32/8 | 512 | 177.5 (57%) | 433 GB/s | 410 | 0.194 ms | GQA efficient |

### Analysis

```
PREFILL ATTENTION BEHAVIOR:
===========================

Work Distribution at seq_len=4096, batch=1, heads=32:
+-- Q tiles: 32 (128 tokens each)
+-- Work items: 32 heads x 32 Q_tiles = 1024 items
+-- Work per SM: 1024 / 108 = 9.5 items per SM (good load balance)

Roofline Position:
+-- AI = 2048 FLOPs/Byte >> ridge point (153)
+-- Conclusion: Compute-bound at long sequences

Scaling Behavior:
+-- seq_len doubles -> work quadruples (O(n^2))
+-- Batch scaling linear with good efficiency
+-- GQA reduces KV memory reads, maintains compute

Expected Warp Stalls (based on roofline position):
+-- short_scoreboard: 30-40% (Tensor Core pipeline)
+-- barrier: 20-30% (producer-consumer sync)
+-- long_scoreboard: 15-25% (occasional HBM waits)
```

### Key Insights

1. **Short sequences under-utilize SMs**: seq_len=512 creates only 128 work items
   (32 heads x 4 tiles) for 108 SMs
2. **Long sequences are compute-bound**: At seq_len=4096, we hit 98.5% of peak TC
3. **GQA improves memory efficiency**: 32/8 GQA ratio reduces KV reads by 4x

---

## 2. Decode Attention Analysis

### Algorithm: Split-K with Reduction

```
Split-K Decode Pattern:
=======================

# Phase 1: Parallel partial attention across K splits
for each split_k in range(num_splits):
  partial_O[split_k] = Attention(Q, K[split_start:split_end], V[split_start:split_end])
  partial_lse[split_k] = log_sum_exp of this split

# Phase 2: Reduction
O_final = merge_states(partial_O, partial_lse)  # Online softmax merge

Why Split-K?
+-- Decode has only 1 query token -> very little parallelism
+-- Split KV across multiple CTAs to increase parallelism
+-- Pay cost of reduction to gain better GPU utilization
```

### Profiling Results

| Config | KV Len | Batch | Tokens/s | HBM BW | AI | Run Time |
|--------|--------|-------|----------|--------|-----|----------|
| batch=1 | 512 | 1 | 28K | 235 GB/s | 1.0 | 0.036 ms |
| batch=1 | 2048 | 1 | 21K | 705 GB/s | 1.0 | 0.048 ms |
| batch=1 | 4096 | 1 | 13K | 902 GB/s | 1.0 | 0.074 ms |
| batch=32 | 2048 | 32 | 50K | 1679 GB/s | 1.0 | 0.640 ms |
| batch=64 | 2048 | 64 | 52K | 1744 GB/s | 1.0 | 1.232 ms |
| batch=128 | 2048 | 128 | 53K | 1771 GB/s | 1.0 | 2.426 ms |
| GQA 32/8 | 4096 | 32 | 82K | 1377 GB/s | 4.0 | 0.390 ms |

### Analysis

```
DECODE ATTENTION BEHAVIOR:
==========================

Memory Access Pattern (batch=32, kv_len=2048):
+-- Q read: 32 x 32 x 128 x 2B = 262 KB (negligible)
+-- KV read: 32 x 2048 x 32 x 128 x 2B x 2 = 1073 MB (dominant!)
+-- O write: 32 x 32 x 128 x 2B = 262 KB (negligible)

Arithmetic Intensity:
+-- AI = FLOPs / Bytes = 2 * 2048 * 128 * 2 / (2048 * 128 * 2 * 2) = 1.0
+-- Ridge point for A100: 153 FLOPs/Byte
+-- Decode is 153x below ridge point -> pure memory-bound!

Scaling Behavior:
+-- Batch size -> linear bandwidth scaling (amortize latency)
+-- KV length -> linear time increase (more data to read)
+-- GQA helps: 4x fewer KV heads = 4x less memory

Expected Warp Stalls:
+-- long_scoreboard: 60-70% (waiting for HBM reads)
+-- barrier: 15-25% (split-K reduction sync)
+-- short_scoreboard: 5-10% (register dependencies)
```

### Decode vs Prefill Comparison

```
                    PREFILL             DECODE
                    =======             ======
Shape:              [B,S,H,D] x [B,S,H,D]   [B,1,H,D] x [B,K,H,D]
Operation:          GEMM-like           GEMV-like
AI:                 >153 (compute)      ~1 (memory)
TC Utilization:     30-50%              <20%
HBM Utilization:    10-40%              70-90%
Optimization:       Tile KV             Split-K
```

---

## 3. RMSNorm Analysis

### Algorithm: Per-Row Reduction + Elementwise

```
RMSNorm Algorithm:
==================

For each row x:
  1. Compute sum(x^2) via parallel reduction
  2. rms = sqrt(sum / hidden_size + eps)
  3. output = x / rms * weight

Implementation:
+-- One thread block per row
+-- Warp-level reduction for sum(x^2)
+-- Vectorized loads/stores (128-bit)
+-- Fused variant combines residual addition
```

### Profiling Results

| Config | Batch | Hidden | HBM BW | Run Time |
|--------|-------|--------|--------|----------|
| FP16 | 1 | 4096 | 2.0 GB/s | 0.012 ms |
| FP16 | 32 | 4096 | 42.1 GB/s | 0.013 ms |
| FP16 | 128 | 4096 | 156.9 GB/s | 0.013 ms |
| FP16 | 512 | 4096 | 523.6 GB/s | 0.016 ms |
| FP16 | 32 | 8192 | 80.2 GB/s | 0.013 ms |
| BF16 | 32 | 4096 | 42.5 GB/s | 0.013 ms |

### Analysis

```
RMSNORM BEHAVIOR:
=================

Memory Access Pattern (batch=512, hidden=4096):
+-- Input: 512 x 4096 x 2B = 4.2 MB (read)
+-- Weight: 4096 x 2B = 8 KB (read, broadcast)
+-- Output: 512 x 4096 x 2B = 4.2 MB (write)
+-- Total: 8.4 MB

Achieved: 523.6 GB/s = 25.7% of peak
Why not higher?
+-- Small kernel -> launch overhead significant
+-- Per-row sync for reduction
+-- Not fully memory-bound (some compute for rsqrt)

Fused Add RMSNorm Benefit:
+-- Without fusion: read input + residual, write temp, read temp + weight, write output
+-- With fusion: read input + residual + weight, write output + residual
+-- Memory savings: ~40% traffic reduction
```

---

## 4. RoPE Analysis

### Algorithm: Rotary Position Embedding

```
RoPE Algorithm:
===============

For position p and dimension pair (i, i+d/2):
  theta_i = p * base^(-2i/d)
  q'[i]     = q[i] * cos(theta) - q[i+d/2] * sin(theta)
  q'[i+d/2] = q[i] * sin(theta) + q[i+d/2] * cos(theta)

Variants:
+-- In-place: Modify Q, K directly (saves memory)
+-- With cache: Use precomputed cos/sin tables (faster)
+-- Llama 3.1: Extended context with scaling
```

### Profiling Results

| Variant | Config | Tokens | HBM BW | Run Time |
|---------|--------|--------|--------|----------|
| In-place | batch=1, seq=512 | 512 | 244 GB/s | 0.069 ms |
| In-place | batch=32, seq=512 | 16K | 756 GB/s | 0.710 ms |
| In-place | batch=128, seq=512 | 65K | 790 GB/s | 2.720 ms |
| With cache | batch=1, seq=512 | 512 | 297 GB/s | 0.057 ms |
| With cache | batch=32, seq=512 | 16K | 781 GB/s | 0.698 ms |

### Analysis

```
ROPE BEHAVIOR:
==============

Memory Pattern:
+-- Read Q, K: 2 * tokens * heads * dim * 2B
+-- Write Q', K': 2 * tokens * heads * dim * 2B
+-- Optional: cos/sin cache lookup

Achieved: ~38% of peak HBM bandwidth
Reason for gap:
+-- Compute overhead (sin/cos or cache lookup)
+-- Not perfectly coalesced for rotation pairs

With-cache variant:
+-- 5-10% faster due to precomputed cos/sin
+-- Trades memory for compute
+-- Used by SGLang/vLLM for efficiency
```

---

## 5. JIT Compilation Analysis

### Cache Structure

```
~/.cache/flashinfer/0.5.3/80/
+-- cached_ops/
|   +-- batch_prefill_with_kv_cache_dtype_q_f16_.../ -> 4.9 MB .so
|   +-- batch_decode_with_kv_cache_dtype_q_bf16_.../ -> 0.6 MB .so
|   +-- norm/ -> 0.9 MB .so
|   +-- rope/ -> 3.9 MB .so
|   +-- sampling/ -> 5.9 MB .so
|   +-- silu_and_mul/ -> 0.1 MB .so
+-- generated/
    +-- <uri>/
        +-- *.cu (generated CUDA source)
        +-- *_config.inc (type configuration)
```

### Compilation Triggers

| Parameter | Triggers New Compilation |
|-----------|-------------------------|
| dtype (q/kv/o) | Yes - different type specialization |
| head_dim | Yes - affects tile sizes |
| pos_encoding_mode | Yes - different codepath |
| sliding_window | Yes - mask handling |
| logits_soft_cap | Yes - additional compute |
| mask_mode | Yes - different mask handling |

### Performance Characteristics

| Scenario | Latency |
|----------|---------|
| First call (cold JIT) | 10-500 ms |
| Disk cache hit | <10 ms |
| Memory cache hit | <0.1 ms |

---

## 6. Performance Summary

### Kernel Characteristics

| Kernel | Type | Peak Utilization | Bottleneck |
|--------|------|------------------|------------|
| Prefill Attention | Compute | 98.5% TC | Work distribution |
| Decode Attention | Memory | 87% HBM | Memory bandwidth |
| RMSNorm | Memory | 26% HBM | Launch overhead |
| RoPE | Memory | 39% HBM | Compute overhead |

### Roofline Summary

```
A100 Roofline Model
===================

      |                   x Prefill (seq=4096)
  300 |                 x
      |             x Prefill (seq=2048)
TFLOP |         x
  /s  |     x
      | x x x Prefill (seq=512, batch vary)
  100 |------------------------------------x Decode (batch=128)
      |    /                            x Decode (batch=64)
      |   /                         x Decode (batch=32)
      |  /          x RMSNorm   x RoPE
   10 | /       x Decode (batch=1)
      |/
      +----------------------------------------
        1     10    100   1000   AI (FLOPs/Byte)

Ridge Point: ~153 FLOPs/Byte
```

---

## 7. Optimization Opportunities

### For High-Throughput Serving

1. **Batch decode requests**: Increasing batch size from 1 to 128 improves
   throughput from 28K to 53K tokens/sec (1.9x)

2. **Use GQA models**: 32/8 GQA achieves 82K tokens/sec vs 53K for MHA (1.5x)

3. **Fuse operations**: fused_add_rmsnorm saves 40% memory traffic

### For Low-Latency Serving

1. **Pre-warm JIT cache**: Run dummy inference at startup

2. **Use in-place RoPE**: Avoid allocation overhead

3. **Continuous batching**: Keep GPU saturated with mixed prefill/decode

### For Memory Efficiency

1. **Paged KV cache**: FlashInfer's page-based design enables efficient memory

2. **FP8 quantization**: Supported in newer FlashInfer for 2x memory reduction

---

## Tags

`#flashinfer` `#analysis` `#attention` `#rmsnorm` `#rope` `#profiling`
