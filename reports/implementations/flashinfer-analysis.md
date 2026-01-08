# FlashInfer Codebase Analysis

**Repository**: [github.com/flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)
**Version**: Latest (as of analysis)
**Language**: CUDA C++, Python

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       FLASHINFER ARCHITECTURE                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         PYTHON API LAYER                                │ │
│  │  flashinfer/                                                            │ │
│  │  ├── decode.py      # BatchDecodeWithPagedKVCacheWrapper                │ │
│  │  ├── prefill.py     # BatchPrefillWithPagedKVCacheWrapper               │ │
│  │  ├── cascade.py     # MultiLevelCascadeAttentionWrapper                 │ │
│  │  ├── sparse.py      # BlockSparseAttentionWrapper                       │ │
│  │  ├── mla.py         # Multi-Latent Attention (DeepSeek)                 │ │
│  │  └── sampling.py    # Top-k/p sampling                                  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         JIT COMPILATION LAYER                           │ │
│  │  flashinfer/jit/                                                        │ │
│  │  ├── core.py        # JitSpec, compilation infrastructure               │ │
│  │  ├── cpp_ext.py     # Ninja build generation                            │ │
│  │  ├── env.py         # Workspace paths, cache management                 │ │
│  │  └── attention/     # gen_*_module() functions                          │ │
│  │                                                                         │ │
│  │  Flow: Parameters → URI hash → Cache check → Generate → Compile → Load  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         BINDING LAYER (TVM-FFI)                         │ │
│  │  csrc/                                                                  │ │
│  │  ├── *_jit_binding.cu    # TVM-FFI exports                              │ │
│  │  ├── *.cu                # Kernel launchers (tensor handling)           │ │
│  │  └── *.jinja             # Type specialization templates                │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         KERNEL LAYER (Framework-Agnostic)               │ │
│  │  include/flashinfer/                                                    │ │
│  │  ├── attention/          # Attention kernels                            │ │
│  │  │   ├── decode.cuh      # Decode attention                             │ │
│  │  │   ├── prefill.cuh     # Prefill attention                            │ │
│  │  │   ├── scheduler.cuh   # Load-balanced scheduling                     │ │
│  │  │   ├── hopper/         # FA3-style Hopper kernels                     │ │
│  │  │   └── blackwell/      # Blackwell support                            │ │
│  │  ├── page.cuh            # Paged KV-cache data structures               │ │
│  │  ├── gemm/               # GEMM kernels                                 │ │
│  │  └── sampling.cuh        # Sampling kernels                             │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Core Data Structures

### 1. Paged KV-Cache (`page.cuh`)

```cpp
template <typename DType, typename IdType>
struct paged_kv_t {
    // Storage
    DType* k_data;           // [max_pages, num_heads, page_size, head_dim] or NHD
    DType* v_data;           // Same layout as k_data

    // Indexing (CSR-style)
    IdType* indices;         // Page indices for each slot
    IdType* indptr;          // [batch_size + 1] cumulative page counts
    IdType* last_page_len;   // Tokens in last page per sequence

    // Dimensions
    uint_fastdiv page_size;  // Fast division for page computations
    uint32_t num_heads, head_dim, batch_size;
    uint32_t stride_page, stride_n, stride_h;  // Layout strides

    // Access methods
    __device__ DType* get_k_ptr(page_iter, head_idx, entry_idx, feat_idx);
    __device__ uint32_t get_length(batch_idx);  // Total tokens for request
};
```

**Key insight**: The `indices` + `indptr` pattern allows:
- Dense: indices = [0,1,2,...], indptr = [0,1,2,...]
- Paged: indices = scattered page IDs, indptr = cumulative counts
- Same kernel code handles both!

### 2. MLA KV-Cache (`page.cuh`)

For DeepSeek's Multi-Latent Attention:

```cpp
template <typename DType, typename IdType>
struct paged_kv_mla_t {
    // Compressed KV (512-dim) + K positional embedding (64-dim)
    DType* ckv_data;    // [max_pages, page_size, head_dim_ckv]
    DType* kpe_data;    // [max_pages, page_size, head_dim_kpe]

    // Same indexing as paged_kv_t
    IdType* indices, *indptr, *last_page_len;
};
```

### 3. Tile Schedulers (`attention/hopper/tile_scheduler.cuh`)

```cpp
// Simple scheduler: one tile per CTA
struct SingleTileScheduler {
    static dim3 get_grid_dim(args, num_sm) {
        return {num_qo_tiles, num_qo_heads};  // Direct mapping
    }

    WorkTileInfo get_initial_work(params) {
        return {blockIdx.x, blockIdx.y, ...};  // Static assignment
    }
};

// Persistent scheduler: load-balanced across SMs
template <typename IdType>
struct BatchPrefillPersistentTileScheduler {
    // Pre-computed work distribution
    IdType *work_indptr;      // [num_sm + 1] work boundaries per SM
    IdType *qo_tile_indices;  // Which tile to process
    IdType *batch_indices;    // Which batch element

    static dim3 get_grid_dim(args, num_sm) {
        return {num_sm};  // One CTA per SM
    }

    WorkTileInfo get_initial_work(params) {
        // SM reads its assigned work range
        int ptr_begin = work_indptr[blockIdx.x];
        int ptr_end = work_indptr[blockIdx.x + 1];
        return fetch_work(ptr_begin);
    }

    WorkTileInfo get_next_work(params, current_work) {
        // Iterate through assigned work
        return fetch_work(current_work.counter + 1);
    }
};
```

## Plan-Run Pattern Implementation

### Python Wrapper (`prefill.py`)

```python
class BatchPrefillWithPagedKVCacheWrapper:
    def __init__(self, workspace_buffer, kv_layout="NHD"):
        self._workspace_buffer = workspace_buffer
        self._kv_layout = kv_layout
        self._plan_info = None

    def plan(self, qo_indptr, kv_indptr, num_qo_heads, num_kv_heads,
             head_dim, page_size, ...):
        """
        CPU scheduling phase:
        1. Analyze workload
        2. Choose kernel variant (FA2 vs FA3, split-kv vs not)
        3. Compute work distribution
        4. Store plan in workspace
        """
        # Binary search for optimal KV chunk size
        split_kv, kv_chunk_size = binary_search_kv_chunk_size(
            max_batch_size_if_split, qo_lens, kv_lens, qo_chunk_size
        )

        # Compute load-balanced work assignment
        work_indptr = compute_work_indptr(qo_lens, kv_lens, num_sms)

        # Store in workspace (GPU memory)
        self._plan_info = pack_plan_info(work_indptr, split_kv, ...)

    def run(self, q, paged_kv_cache, o, ...):
        """
        GPU execution phase:
        - Static kernel launch (CUDAGraph compatible)
        - Uses pre-computed plan_info
        """
        module.run(
            self._workspace_buffer,
            self._plan_info,
            q, k_cache, v_cache,
            o, ...
        )
```

### C++ Scheduling (`scheduler.cuh`)

```cpp
// Binary search to find optimal pages-per-batch for load balancing
template <typename IdType>
auto PartitionPagedKVCacheBinarySearchMinNumPagePerBatch(
    uint32_t max_grid_size, uint32_t gdy,
    const std::vector<IdType>& num_pages,
    uint32_t min_num_pages_per_batch = 1
) {
    uint32_t low = min_num_pages_per_batch;
    uint32_t high = max_element(num_pages);

    while (low < high) {
        uint32_t mid = (low + high) / 2;
        uint32_t new_batch_size = 0;

        // Count tiles if we use mid pages per batch
        for (auto& pages : num_pages) {
            new_batch_size += ceil_div(pages, mid);
        }

        if (new_batch_size * gdy > max_grid_size) {
            low = mid + 1;  // Need larger chunks
        } else {
            high = mid;  // Can use smaller chunks
        }
    }

    return {low, compute_new_batch_size(num_pages, low)};
}
```

## JIT Compilation System

### Three Layers

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 1: JitSpec (flashinfer/jit/core.py)                             │
│  ═══════════════════════════════════════════                           │
│                                                                         │
│  JitSpec:                                                               │
│    name: str           # Unique URI (hash of parameters)                │
│    sources: List[Path] # .cu files to compile                           │
│    extra_cuda_cflags   # Compiler flags                                 │
│                                                                         │
│  Methods:                                                               │
│    write_ninja()       # Generate build.ninja                           │
│    build()             # Run ninja                                      │
│    build_and_load()    # Compile + load via TVM-FFI                     │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 2: Code Generation (gen_*_module functions)                      │
│  ═══════════════════════════════════════════════════                    │
│                                                                         │
│  def gen_batch_decode_module(dtype_q, dtype_kv, head_dim, ...):         │
│      uri = get_batch_decode_uri(dtype_q, dtype_kv, ...)                 │
│      gen_dir = FLASHINFER_GEN_SRC_DIR / uri                             │
│                                                                         │
│      # Optional: Render Jinja template for type config                  │
│      config = template.render(dtype_q=dtype_q, HEAD_DIM=head_dim, ...)  │
│      write_if_different(gen_dir / "config.inc", config)                 │
│                                                                         │
│      # Copy source files                                                │
│      sources = copy_sources(["decode.cu", "decode_jit_binding.cu"])     │
│                                                                         │
│      return gen_jit_spec(uri, sources, extra_flags)                     │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 3: Module Caching                                                │
│  ═══════════════════════                                                │
│                                                                         │
│  Python-level: @functools.cache on get_*_module() functions             │
│  File-level:   ~/.cache/flashinfer/<version>/<uri>/                     │
│                                                                         │
│  Cache invalidation (automatic):                                        │
│    - Source file SHA256 changes                                         │
│    - Compilation flags change                                           │
│    - CUDA architecture changes                                          │
│    - FlashInfer version changes                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Hopper Kernels (FA3-style)

### Kernel Structure (`attention/hopper/prefill_sm90.cuh`)

```cpp
template <typename KernelTraits, typename TileScheduler>
__global__ void PrefillKernel(Params params) {
    // TMA descriptors for async loads
    TmaDescriptor tma_q, tma_k, tma_v;

    // Shared memory for double-buffering
    __shared__ alignas(128) char smem[KernelTraits::SMEM_SIZE];

    // Get work assignment from scheduler
    auto scheduler = TileScheduler();
    auto work_tile = scheduler.get_initial_work(params);

    // Warp role assignment
    int warp_id = threadIdx.x / 32;
    bool is_producer = (warp_id == 0);

    while (work_tile.is_valid(params)) {
        auto [q_tile, qo_head, kv_head, ...] = work_tile.get_block_coord(params);

        if (is_producer) {
            // PRODUCER WARP: TMA loads
            producer_loop(tma_q, tma_k, tma_v, smem, params);
        } else {
            // CONSUMER WARPS: Tensor Core MMA
            consumer_loop(smem, params, output);
        }

        // Get next work item
        work_tile = scheduler.get_next_work(params, work_tile);
    }
}
```

### Warp Specialization in FlashInfer

```cpp
// Named barriers for producer-consumer sync (attention/hopper/named_barrier.cuh)
enum class NamedBarriers {
    ProducerWG = 0,     // Producer warp group
    ConsumerWG0 = 1,    // Consumer warp group 0
    ConsumerWG1 = 2,    // Consumer warp group 1
    // ...
};

// Producer signals data ready
__device__ void producer_arrive(NamedBarriers barrier) {
    asm volatile("bar.arrive %0, %1;" : : "r"(barrier), "r"(producer_count));
}

// Consumer waits for data
__device__ void consumer_wait(NamedBarriers barrier) {
    asm volatile("bar.sync %0, %1;" : : "r"(barrier), "r"(total_count));
}
```

## Key Design Patterns

### 1. Dispatch Macros

```cpp
// Handle combinatorial parameter spaces
DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    DISPATCH_GROUP_SIZE(group_size, GROUP_SIZE, {
        DISPATCH_POS_ENCODING_MODE(pos_mode, POS_MODE, {
            launch_kernel<HEAD_DIM, GROUP_SIZE, POS_MODE>(...);
        });
    });
});
```

### 2. Framework Separation

```
include/flashinfer/     # Framework-agnostic (raw pointers)
  └── attention/decode.cuh: void decode_kernel(DType* q, DType* k, ...)

csrc/                   # Framework bindings (TVM-FFI)
  └── decode.cu: void decode(tvm::runtime::NDArray q, ...)
```

### 3. Attention Variants via Templates

```cpp
// Variants defined as template parameters
template <typename AttentionVariant, typename Params>
__global__ void AttentionKernel(Params params) {
    // AttentionVariant customizes:
    // - Softmax computation
    // - Position encoding
    // - Masking logic

    float score = AttentionVariant::compute_score(q, k, params);
    float prob = AttentionVariant::apply_softmax(score, params);
    // ...
}

// Example variant
struct StandardAttention {
    static float compute_score(q, k, params) {
        return dot(q, k) * params.sm_scale;
    }
};

struct SoftCapAttention {
    static float compute_score(q, k, params) {
        float raw = dot(q, k) * params.sm_scale;
        return params.logits_soft_cap * tanh(raw / params.logits_soft_cap);
    }
};
```

## File Summary

| Directory | Purpose |
|-----------|---------|
| `include/flashinfer/` | Header-only CUDA kernels (framework-agnostic) |
| `include/flashinfer/attention/` | Attention kernels (decode, prefill, cascade) |
| `include/flashinfer/attention/hopper/` | FA3-style kernels for Hopper |
| `include/flashinfer/gemm/` | GEMM kernels for MoE, LoRA |
| `csrc/` | TVM-FFI bindings and launchers |
| `flashinfer/` | Python API |
| `flashinfer/jit/` | JIT compilation infrastructure |
| `tests/` | Test suite |
| `benchmarks/` | Performance benchmarks |

## Performance Optimizations

1. **FastDiv**: Uses `uint_fastdiv` for page index computations (avoids expensive integer division)
2. **Vectorized loads**: `vec_t<DType, vec_size>::memcpy` for coalesced memory access
3. **Double-buffering**: Multi-stage SMEM buffers for latency hiding
4. **Persistent kernels**: One CTA per SM for better occupancy
5. **TMA**: Hardware-accelerated async loads on Hopper

## Tags

`#flashinfer` `#attention` `#kv-cache` `#jit` `#cuda` `#hopper` `#tma` `#warp-specialization` `#load-balancing`
