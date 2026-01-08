# FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving

**Paper**: [arXiv:2501.01005](https://arxiv.org/abs/2501.01005)
**Authors**: Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, Luis Ceze
**Date**: January 2025 (MLSys 2025)
**Code**: [github.com/flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)

## Core Contribution

FlashInfer is an **attention engine** for LLM serving that addresses three key challenges:

1. **KV-Cache Storage Heterogeneity**: Different serving scenarios require different KV-cache layouts (dense, paged, ragged)
2. **Attention Variant Explosion**: Many attention variants (GQA, MLA, RoPE, ALiBi, etc.) require separate kernel implementations
3. **Dynamic Request Patterns**: Variable-length sequences cause load imbalance across GPU SMs

## Key Technical Innovations

### 1. Block-Sparse Format with Composable Formats

FlashInfer unifies different KV-cache layouts under a **block-sparse abstraction**:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    COMPOSABLE KV-CACHE FORMATS                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Dense KV-Cache:       [batch, seq_len, num_heads, head_dim]                 │
│                        └─▶ All sequences same length, contiguous             │
│                                                                              │
│  Ragged KV-Cache:      [total_tokens, num_heads, head_dim] + indptr         │
│                        └─▶ Variable-length, packed contiguously              │
│                                                                              │
│  Paged KV-Cache:       [max_pages, page_size, num_heads, head_dim]          │
│                        + page_indices + indptr                               │
│                        └─▶ Memory-efficient, non-contiguous pages            │
│                                                                              │
│  UNIFIED ABSTRACTION:                                                        │
│  ════════════════════                                                        │
│                                                                              │
│    paged_kv_t {                                                              │
│      k_data, v_data,     // Underlying storage                               │
│      indices,            // Page index array (identity for dense/ragged)     │
│      indptr,             // CSR-style row pointers                           │
│      last_page_len,      // Tokens in last page per sequence                 │
│      page_size,          // Tokens per page (seq_len for dense)              │
│      stride_*            // Layout strides (HND or NHD)                      │
│    }                                                                         │
│                                                                              │
│  All formats map to same kernel code with different index computations!      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2. JIT Compilation for Attention Variants

FlashInfer uses **Just-In-Time compilation** to generate specialized kernels:

```python
# ══════════════════════════════════════════════════════════════════════════════
# JIT COMPILATION FLOW
# ══════════════════════════════════════════════════════════════════════════════

def first_api_call(dtype, head_dim, pos_encoding, ...):
    """
    On first call, FlashInfer:
    1. Computes unique URI from parameters
    2. Checks disk cache (~/.cache/flashinfer/)
    3. If miss: generates CUDA code via Jinja templates
    4. Compiles with ninja → .so file
    5. Loads via TVM-FFI
    6. Caches in-memory for subsequent calls
    """
    uri = hash(dtype, head_dim, pos_encoding, ...)

    if uri not in memory_cache:
        if uri not in disk_cache:
            # Generate specialized kernel code
            cuda_code = render_template(
                dtype_in=dtype,
                HEAD_DIM=head_dim,
                POS_ENCODING_MODE=pos_encoding,
                ...
            )
            # Compile with ninja
            compiled_module = ninja_build(cuda_code)
            disk_cache[uri] = compiled_module

        memory_cache[uri] = load_module(disk_cache[uri])

    return memory_cache[uri]
```

**Benefit**: No need to pre-compile all combinations. ~1000+ variants handled dynamically.

### 3. Load-Balanced Scheduling with Plan-Run Pattern

FlashInfer decouples **planning** (CPU) from **execution** (GPU) for:
- Load balancing across SMs
- CUDAGraph compatibility (static kernel launch config)

```python
# ══════════════════════════════════════════════════════════════════════════════
# PLAN-RUN PATTERN
# ══════════════════════════════════════════════════════════════════════════════

class BatchPrefillWithPagedKVCacheWrapper:
    def plan(self, qo_indptr, kv_indptr, num_qo_heads, num_kv_heads, ...):
        """
        CPU-side scheduling (runs once per batch shape change):
        1. Analyze workload distribution
        2. Compute work_indptr for load-balanced SM assignment
        3. Decide whether to split KV (for long sequences)
        4. Store plan in workspace buffers
        """
        # Partition work evenly across SMs
        total_tiles = sum(ceil_div(qo_len, TILE_SIZE) * num_heads for ...)
        tiles_per_sm = ceil_div(total_tiles, num_sms)

        # Create work_indptr: SM i handles work_indptr[i:i+1]
        work_indptr = compute_balanced_partition(tile_assignments, num_sms)

        # Store in workspace for run()
        self.plan_info = (work_indptr, split_kv, ...)

    def run(self, q, k, v, o):
        """
        GPU-side execution (can be captured by CUDAGraph):
        - Uses pre-computed plan_info
        - Kernel launch config is STATIC
        - Each SM reads its work from work_indptr
        """
        kernel<<<num_sms, threads>>>(q, k, v, o, self.plan_info)
```

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    LOAD-BALANCED TILE SCHEDULING                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  WITHOUT LOAD BALANCING:                                                     │
│  ═══════════════════════                                                     │
│                                                                              │
│  Request 0 (2048 tokens): ████████████████████████████████ (many tiles)     │
│  Request 1 (128 tokens):  ██ (few tiles)                                     │
│  Request 2 (64 tokens):   █                                                  │
│                                                                              │
│  SM 0: ████████████████████████████████  (overloaded)                        │
│  SM 1: ██                                 (idle)                             │
│  SM 2: █                                  (idle)                             │
│  ...                                                                         │
│                                                                              │
│  WITH FLASHINFER LOAD BALANCING:                                             │
│  ═══════════════════════════════                                             │
│                                                                              │
│  work_indptr partitions tiles evenly:                                        │
│                                                                              │
│  SM 0: ████████  (tiles 0-7 from req 0)                                      │
│  SM 1: ████████  (tiles 8-15 from req 0)                                     │
│  SM 2: ████████  (tiles 16-23 from req 0)                                    │
│  SM 3: ████████  (tiles 24-31 from req 0)                                    │
│  SM 4: ████████  (remaining req 0 + req 1 + req 2)                           │
│                                                                              │
│  All SMs finish at ~same time!                                               │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 4. Hopper Support (FA3-style)

FlashInfer provides FA3-style kernels for Hopper GPUs with:
- TMA (Tensor Memory Accelerator) for async data movement
- Warp specialization (producer-consumer pattern)
- Named barriers for fine-grained synchronization

## Performance Results

| Benchmark | Improvement |
|-----------|-------------|
| Inter-token latency | 29-69% reduction vs compiler backends |
| Long-context inference | 28-30% latency reduction |
| Parallel generation (speculative) | 13-17% speedup |

## Integration

FlashInfer is integrated into:
- **SGLang**: Default attention backend
- **vLLM**: Supported backend
- **MLC-LLM**: Primary attention engine
- **TensorRT-LLM**: Compatible kernels

## Key Architectural Insights

1. **Abstraction at the right level**: Block-sparse format abstracts storage while preserving efficiency
2. **JIT vs AOT tradeoff**: JIT handles variant explosion; optional AOT packages for deployment
3. **Plan-Run enables CUDAGraph**: Static launch config even with dynamic workloads
4. **Framework-agnostic kernels**: TVM-FFI allows multi-framework support

## Tags

`#attention` `#kv-cache` `#jit-compilation` `#load-balancing` `#cuda-graph` `#llm-serving` `#paged-attention` `#hopper`
