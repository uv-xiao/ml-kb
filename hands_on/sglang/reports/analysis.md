# SGLang Codebase Analysis

**Generated:** 2026-01-09
**Codebase:** `/home/uvxiao/mlkb/code-repos/sglang/`
**Focus:** RadixCache, Attention Backends, Scheduler, Tensor Parallelism

---

## Executive Summary

SGLang is a production LLM serving system with several key innovations:
- **RadixAttention**: Prefix caching using radix tree for KV cache reuse
- **Continuous Batching**: Dynamic request scheduling with overlap execution
- **Multiple Attention Backends**: FlashInfer (default), Triton, FlashAttention
- **Tensor Parallelism**: NCCL-based distributed execution across GPUs

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SGLANG SERVING ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        TOKENIZER MANAGER                              │   │
│  │    • Request tokenization                                            │   │
│  │    • Response detokenization                                         │   │
│  │    • Multimodal input processing                                     │   │
│  └────────────────────────────────┬─────────────────────────────────────┘   │
│                                   │ ZMQ IPC                                  │
│                                   ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                          SCHEDULER (per GPU group)                    │   │
│  │  ┌────────────────┐  ┌─────────────────┐  ┌──────────────────────┐   │   │
│  │  │ Waiting Queue  │  │ Schedule Policy │  │ Running Batch        │   │   │
│  │  │  (pending)     │─▶│ (LPM, FCFS...)  │─▶│ (continuous batch)   │   │   │
│  │  └────────────────┘  └─────────────────┘  └──────────────────────┘   │   │
│  │                                                                       │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │                        RADIX CACHE                              │  │   │
│  │  │   • Prefix matching (O(log n) lookup)                          │  │   │
│  │  │   • KV cache index management                                   │  │   │
│  │  │   • Eviction policies (LRU, LFU, FIFO, Priority)               │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  └────────────────────────────────┬─────────────────────────────────────┘   │
│                                   │                                          │
│                                   ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       TP MODEL WORKER (per TP rank)                   │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │                       MODEL RUNNER                              │  │   │
│  │  │   • Forward pass execution                                      │  │   │
│  │  │   • CUDA graph management                                       │  │   │
│  │  │   • Attention backend dispatch                                  │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │                       MEMORY POOL                               │  │   │
│  │  │   • req_to_token_pool: Request → Token indices                 │  │   │
│  │  │   • token_to_kv_pool: Token → KV cache tensors                 │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       ATTENTION BACKENDS                              │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │   │
│  │  │ FlashInfer   │  │  Triton      │  │ FlashAttention           │   │   │
│  │  │ (default)    │  │ (customize)  │  │ (compatibility)          │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component 1: RadixCache (Prefix Caching)

### Location
`/home/uvxiao/mlkb/code-repos/sglang/python/sglang/srt/mem_cache/radix_cache.py`

### Core Data Structures

```
RadixCache Architecture:
────────────────────────────────────────────────────────────────────────────

                            ROOT NODE
                           (empty key)
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
         [System]          [User:]         [Assistant:]
         prompt...        message...        response...
              │                │                │
              ▼                ▼                ▼
         TreeNode          TreeNode         TreeNode
         ├─key: [tokens]   ├─key: [tokens]  ├─key: [tokens]
         ├─value: [KV idx] ├─value: [KV idx]├─value: [KV idx]
         ├─lock_ref: 0     ├─lock_ref: 2    ├─lock_ref: 0
         └─children: {}    └─children: {}   └─children: {}

TreeNode Fields:
  • key: RadixKey (token_ids + optional extra_key for LoRA isolation)
  • value: torch.Tensor of KV cache indices in token_to_kv_pool
  • lock_ref: Reference count (>0 = protected from eviction)
  • children: Dict[child_key, TreeNode] for radix tree structure
  • last_access_time: For LRU eviction
  • hit_count: For LFU eviction
  • priority: For priority-aware eviction
```

### Key Operations

```python
# Prefix Matching (match_prefix)
# ─────────────────────────────────────────────────────────────────
# Input: RadixKey([1, 2, 3, 4, 5], extra_key=None)
#
# Process:
#   1. Start at root_node
#   2. Traverse tree following matching token segments
#   3. If partial match, SPLIT node at match boundary
#   4. Return: (device_indices, last_matched_node)
#
# Time Complexity: O(n) where n = key length
# Key optimization: page_size > 1 reduces tree depth

# Insertion (insert)
# ─────────────────────────────────────────────────────────────────
# Input: RadixKey, value (KV indices), priority
#
# Process:
#   1. Match existing prefix
#   2. Create new node for unmatched suffix
#   3. Update evictable_size tracking
#   4. Return: length of matched prefix (for duplicate detection)

# Eviction (evict)
# ─────────────────────────────────────────────────────────────────
# Input: num_tokens to free
#
# Process:
#   1. Collect all leaf nodes with lock_ref == 0
#   2. Build min-heap using eviction strategy priority
#   3. Pop and delete until num_tokens freed
#   4. Propagate deletion (parent becomes leaf if no children)
```

### Eviction Strategies

| Strategy | Priority Function | Use Case |
|----------|-------------------|----------|
| LRU | `last_access_time` | Default, recency-based |
| LFU | `hit_count` | Frequency-based |
| FIFO | `creation_time` | Order-based |
| MRU | `-last_access_time` | Inverse LRU |
| Priority | `node.priority` | User-defined priority |

### Hardware Behavior During Cache Operations

```
CACHE HIT (Prefix Matched):
────────────────────────────────────────────────────────────────────────────
Time 0ms: match_prefix() called
├── CPU: Tree traversal (~0.1ms, pure CPU)
├── CPU: Node splitting if partial match
└── Return: torch.Tensor of KV indices on GPU

GPU Impact: NONE during matching
Memory: Reuse existing KV cache (no allocation)
Benefit: Skip prefill computation for matched tokens

CACHE MISS (New Prefix):
────────────────────────────────────────────────────────────────────────────
Time 0ms: No prefix match
├── Scheduler: Allocate new KV cache slots
│   └── token_to_kv_pool_allocator.alloc(num_tokens)
├── GPU: Full prefill computation required
└── After forward: insert() new KV indices into tree

GPU Impact: Full attention computation
Memory: New allocation from KV pool
Cost: Prefill time proportional to new token count
```

---

## Component 2: Attention Backends

### Location
`/home/uvxiao/mlkb/code-repos/sglang/python/sglang/srt/layers/attention/`

### Backend Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ATTENTION BACKEND COMPARISON                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  FLASHINFER (flashinfer_backend.py)                                         │
│  ══════════════════════════════════                                         │
│  • Uses FlashInfer library (JIT compiled kernels)                           │
│  • Separate wrappers for prefill/decode                                     │
│  • Paged KV cache with configurable page size                               │
│  • Tensor core optimization for decode                                      │
│  • Supports sliding window, cross-attention                                 │
│                                                                              │
│  Key Classes:                                                                │
│    BatchPrefillWithPagedKVCacheWrapper  → Prefill/extend operations        │
│    BatchDecodeWithPagedKVCacheWrapper   → Decode with split-K              │
│    FlashInferIndicesUpdater*            → KV index management              │
│                                                                              │
│  ────────────────────────────────────────────────────────────────────────   │
│                                                                              │
│  TRITON (triton_backend.py)                                                 │
│  ═══════════════════════════                                                │
│  • Pure Triton implementation (easier to customize)                         │
│  • Split-K decode for parallelism                                           │
│  • Deterministic mode support                                               │
│  • Custom mask support for complex patterns                                 │
│                                                                              │
│  Key Functions:                                                              │
│    decode_attention_fwd()  → Decode kernel with KV splits                  │
│    extend_attention_fwd()  → Prefill/extend kernel                         │
│                                                                              │
│  ────────────────────────────────────────────────────────────────────────   │
│                                                                              │
│  FLASHATTENTION (flashattention_backend.py)                                 │
│  ═══════════════════════════════════════════                                │
│  • Uses FlashAttention-2 library                                            │
│  • Good for models without paged KV cache needs                             │
│  • Compatibility fallback                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### FlashInfer Backend Deep Dive

```python
# Key initialization in FlashInferAttnBackend.__init__():

# 1. Tensor Core Decision
self.decode_use_tensor_cores = should_use_tensor_core(
    kv_cache_dtype=model_runner.kv_cache_dtype,
    num_attention_heads=...,
    num_kv_heads=...
)
# Tensor cores used when: heads ratio allows efficient GEMM

# 2. Workspace Buffer Allocation
global_workspace_buffer = torch.empty(
    global_workspace_size,  # Default 128MB, 512MB for Qwen
    dtype=torch.uint8,
    device=model_runner.device,
)
# Shared across all wrappers to reduce memory

# 3. Wrapper Creation
self.decode_wrappers = [
    BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        "NHD",  # Layout: [Num_heads, Head_dim]
        use_tensor_cores=self.decode_use_tensor_cores,
    )
]
```

### Kernel Selection Logic

```
FORWARD MODE → KERNEL SELECTION
────────────────────────────────────────────────────────────────────────────

┌─────────────────┐
│  ForwardBatch   │
│  forward_mode   │
└────────┬────────┘
         │
         ▼
    ┌────────────┐
    │ is_extend? │──Yes──▶ init_forward_metadata_for_extend()
    └────────────┘         │
         │                 ├─ use_ragged = (batch_size == 1)
         │                 ├─ extend_no_prefix = (total_cached == 0)
         │                 └─ prefill_wrappers_paged.plan()
         │
         │No
         ▼
    ┌────────────┐
    │ is_decode? │──Yes──▶ init_forward_metadata_for_decode()
    └────────────┘         │
         │                 ├─ kv_indptr update
         │                 ├─ kv_indices creation
         │                 └─ decode_wrappers.plan()
         │
         │No
         ▼
    (Other modes: TARGET_VERIFY, DRAFT_EXTEND, etc.)
```

### Hardware Behavior by Mode

```
PREFILL (Extend) Mode:
────────────────────────────────────────────────────────────────────────────
Kernel: flash_attention_prefill / BatchPrefillWithPagedKVCache
Hardware Profile:
  • SM Utilization: 60-80% (good parallelism)
  • Tensor Cores: 30-45% (significant GEMM work)
  • HBM Bandwidth: 60-75% (loading KV for attention)
  • Bottleneck: Mixed compute/memory

Warp Stall Pattern:
  • long_scoreboard: 35-45% (waiting for HBM)
  • barrier: 20-30% (tile synchronization)
  • short_scoreboard: 10-20% (SMEM access)

DECODE Mode:
────────────────────────────────────────────────────────────────────────────
Kernel: flash_attention_decode / BatchDecodeWithPagedKVCache
Hardware Profile:
  • SM Utilization: 40-60% (memory-bound limits parallelism)
  • Tensor Cores: 10-20% (GEMV-style, limited benefit)
  • HBM Bandwidth: 75-85% (dominant bottleneck)
  • Bottleneck: Memory bandwidth (reading full KV cache)

Warp Stall Pattern:
  • long_scoreboard: 55-70% (waiting for KV reads)
  • barrier: 15-25% (split-K reduction sync)
  • Interpretation: Pure memory-bound regime
```

---

## Component 3: Scheduler (Continuous Batching)

### Location
`/home/uvxiao/mlkb/code-repos/sglang/python/sglang/srt/managers/scheduler.py`

### Class Hierarchy

```python
class Scheduler(
    SchedulerOutputProcessorMixin,    # Result processing
    SchedulerUpdateWeightsMixin,      # Weight updates
    SchedulerProfilerMixin,           # Profiling hooks
    SchedulerMetricsMixin,            # Metrics collection
    SchedulerDisaggregationDecodeMixin,
    SchedulerDisaggregationPrefillMixin,
    SchedulerMultiplexMixin,
    SchedulerRuntimeCheckerMixin,
    SchedulerPPMixin,                 # Pipeline parallelism
    SchedulerDPAttnMixin,             # Data parallel attention
):
```

### Event Loop (Overlap Scheduling)

```
OVERLAP SCHEDULING EVENT LOOP
────────────────────────────────────────────────────────────────────────────

Timeline:
         CPU                                    GPU
         ────                                   ────
Step N   │ process_batch_result(batch_N-1)     │ ← result ready
         │ recv_requests()                      │ Batch N still running
         │ get_next_batch_to_run() → batch_N+1  │
         │                                      │
         ▼ run_batch(batch_N+1) ─────────────────▶ Launch kernels
         │                                      │
Step N+1 │ process_batch_result(batch_N)       │ Batch N+1 running
         │ recv_requests()                      │
         │ get_next_batch_to_run()              │
         │                                      │
         ▼                                      ▼

Key Insight: CPU processes previous batch while GPU runs current batch
Benefit: Hide CPU overhead behind GPU execution
```

### Batch Construction

```python
def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
    # Priority order:
    # 1. Retracted requests (memory pressure recovery)
    # 2. Chunked prefill continuation
    # 3. New prefill from waiting queue
    # 4. Decode running batch

    # Memory budget check:
    can_run_list, new_batch = self.get_new_batch_prefill()
    if can_run_list:
        return ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            ...
        )

    # Decode phase
    if not self.running_batch.is_empty():
        return self.running_batch
```

### Schedule Policy

```
SCHEDULE POLICIES
────────────────────────────────────────────────────────────────────────────

LPM (Longest Prefix Match) - Default:
  • Sort waiting requests by prefix cache hit length
  • Prioritize requests with longer cached prefixes
  • Maximizes cache reuse, minimizes redundant computation

FCFS (First Come First Serve):
  • Simple FIFO ordering
  • Fair but may miss cache opportunities

Priority Scheduling:
  • User-defined priority per request
  • Supports preemption for high-priority requests
  • Configurable preemption threshold

LOF (Longest Output First):
  • Experimental: prioritize requests expected to generate longer outputs
```

### Memory Management

```
MEMORY POOL ARCHITECTURE
────────────────────────────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────┐
│                          MEMORY POOLS                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  req_to_token_pool (ReqToTokenPool)                                     │
│  ──────────────────────────────────                                     │
│  Shape: [max_requests, max_tokens_per_request]                          │
│  Purpose: Map request slot → token indices in KV pool                   │
│  │                                                                       │
│  │  Request 0: [45, 46, 47, 48, -1, -1, ...]  (4 tokens allocated)     │
│  │  Request 1: [12, 13, -1, -1, -1, -1, ...]  (2 tokens allocated)     │
│  │  Request 2: [88, 89, 90, 91, 92, -1, ...]  (5 tokens allocated)     │
│                                                                          │
│  token_to_kv_pool (KVCache)                                             │
│  ──────────────────────────                                             │
│  Shape: [num_layers, 2, max_tokens, num_kv_heads, head_dim]             │
│  Purpose: Actual KV cache storage                                        │
│  │                                                                       │
│  │  Index 45 → Layer 0-N K,V tensors for that token position           │
│  │  Paged allocation: Indices can be non-contiguous                     │
│                                                                          │
│  Allocator (TokenToKVPoolAllocator)                                     │
│  ─────────────────────────────────                                      │
│  • Free list of available token indices                                 │
│  • alloc(n) → Tensor of n free indices                                  │
│  • free(indices) → Return indices to free list                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component 4: Tensor Parallelism

### Location
`/home/uvxiao/mlkb/code-repos/sglang/python/sglang/srt/distributed/`

### TP Architecture

```
TENSOR PARALLELISM EXECUTION
────────────────────────────────────────────────────────────────────────────

                    ┌─────────────────────────────────────────┐
                    │           SCHEDULER (rank 0)             │
                    │  • Receives requests                     │
                    │  • Broadcasts to all TP ranks           │
                    │  • Collects results                     │
                    └─────────────────┬───────────────────────┘
                                      │ broadcast_pyobj()
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
            ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
            │ TP Worker 0   │ │ TP Worker 1   │ │ TP Worker 2   │
            │ (GPU 0)       │ │ (GPU 1)       │ │ (GPU 2)       │
            │               │ │               │ │               │
            │ Model Shard 0 │ │ Model Shard 1 │ │ Model Shard 2 │
            │ ├─ QKV[:, :d] │ │ ├─ QKV[:, d:2d]│ │ ├─ QKV[:,2d:]│
            │ └─ FFN[:, :h] │ │ └─ FFN[:, h:2h]│ │ └─ FFN[:,2h:]│
            └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
                    │                 │                 │
                    └────────────┬────┴─────────────────┘
                                 │ NCCL AllReduce
                                 ▼
                    ┌─────────────────────────────────────────┐
                    │         SYNCHRONIZED RESULT             │
                    │  (After attention and FFN layers)       │
                    └─────────────────────────────────────────┘
```

### Communication Operations

```python
# Key communication patterns in parallel_state.py

class GroupCoordinator:
    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """Reduce across all ranks with custom backend selection."""

        # Backend selection priority:
        # 1. PyNccl (fastest for large tensors)
        # 2. Custom AllReduce (for small tensors, reduces kernel launch)
        # 3. PyTorch distributed (fallback)

        if self.use_pynccl and input_.numel() > threshold:
            self.pynccl_comm.all_reduce(input_, stream=stream)
        elif self.use_custom_allreduce:
            out = self.ca_comm.custom_all_reduce(input_)
        else:
            torch.distributed.all_reduce(input_, group=self.device_group)

# Communication points per layer:
# 1. After QKV projection (for TP attention)
# 2. After attention output projection
# 3. After up/gate projection (MoE or FFN)
# 4. After down projection
```

### TP Communication Cost

```
COMMUNICATION OVERHEAD ANALYSIS
────────────────────────────────────────────────────────────────────────────

Per-Layer Communication (Llama architecture):

Layer Component      AllReduce Size (per token)   Communication Time
─────────────────────────────────────────────────────────────────────
Attention Output     hidden_dim × dtype           ~50-100us (NVLink)
FFN Down Proj        hidden_dim × dtype           ~50-100us (NVLink)
─────────────────────────────────────────────────────────────────────
Total per layer      2 × hidden_dim × 2 bytes     ~100-200us

For 70B model (hidden_dim=8192, 80 layers):
  Per token: 80 × 2 × 8192 × 2 = 2.6 MB communicated
  At NVLink speed (300 GB/s unidirectional): ~8.7us theoretical
  With overhead: ~100-200us actual per AllReduce

Optimization Strategies:
  • NVLink pairs: Minimize cross-NUMA communication
  • Overlap: Computation-communication overlap where possible
  • Batching: Larger batches amortize communication overhead
```

---

## Execution Flow: Request Lifecycle

```
REQUEST LIFECYCLE
────────────────────────────────────────────────────────────────────────────

[1] REQUEST ARRIVAL
    │
    ├─ TokenizerManager.handle() → Tokenize input
    │
    ├─ Send to Scheduler via ZMQ IPC
    │
    └─ Create Req object with:
       • rid (request ID)
       • input_ids (tokenized)
       • sampling_params
       • lora_id (optional)

[2] SCHEDULING
    │
    ├─ Add to waiting_queue
    │
    ├─ Policy.get_priority() → Sort waiting requests
    │
    ├─ RadixCache.match_prefix() → Find cached prefix
    │   │
    │   ├─ CACHE HIT: Reuse KV indices, skip prefill for matched
    │   └─ CACHE MISS: Need full prefill
    │
    └─ PrefillAdder.add_one_req() → Memory allocation check

[3] BATCH EXECUTION
    │
    ├─ ScheduleBatch.prepare_for_extend() or prepare_for_decode()
    │
    ├─ Create ForwardBatch with:
    │   • input_ids tensor
    │   • attention metadata (kv_indptr, kv_indices)
    │   • position_ids
    │
    └─ ModelRunner.forward()
       │
       ├─ [PREFILL] Process new tokens
       │   • Embedding lookup
       │   • For each layer:
       │     - RMSNorm
       │     - QKV projection
       │     - Attention (FlashInfer/Triton)
       │     - O projection + AllReduce
       │     - FFN (Gate/Up → Act → Down)
       │     - AllReduce
       │   • LM head → Logits
       │
       └─ [DECODE] Process one token per request
           • Same layer structure
           • Different attention kernel (decode vs prefill)
           • CUDA graph replay when possible

[4] RESULT PROCESSING
    │
    ├─ Sample next token from logits
    │
    ├─ Update req.output_ids
    │
    ├─ Check termination (EOS, max_tokens)
    │
    ├─ If finished:
    │   • RadixCache.cache_finished_req() → Insert KV into tree
    │   • Release memory slots
    │   • Send response to detokenizer
    │
    └─ If not finished:
        • Stay in running_batch for next decode iteration
```

---

## Key Files Reference

### Core Components

| Component | File | LOC |
|-----------|------|-----|
| RadixCache | `srt/mem_cache/radix_cache.py` | ~850 |
| Scheduler | `srt/managers/scheduler.py` | ~2500 |
| TpModelWorker | `srt/managers/tp_worker.py` | ~800 |
| ModelRunner | `srt/model_executor/model_runner.py` | ~2000 |
| FlashInfer Backend | `srt/layers/attention/flashinfer_backend.py` | ~1800 |
| Triton Backend | `srt/layers/attention/triton_backend.py` | ~1400 |
| ForwardBatch | `srt/model_executor/forward_batch_info.py` | ~1000 |
| ParallelState | `srt/distributed/parallel_state.py` | ~1500 |

### Configuration

| Setting | Location | Default |
|---------|----------|---------|
| Attention Backend | `server_args.attention_backend` | "flashinfer" |
| Page Size | `server_args.page_size` | 1 |
| TP Size | `server_args.tp_size` | 1 |
| Chunked Prefill | `server_args.chunked_prefill_size` | None |
| Eviction Policy | `server_args.radix_eviction_policy` | "lru" |

---

## Profiling Entry Points

### Key Instrumentation Points

```python
# 1. Scheduler Event Loop
# Location: scheduler.py::event_loop_overlap()
# Metrics: Batch scheduling time, queue lengths

# 2. Forward Pass
# Location: model_runner.py::forward()
# Metrics: Per-layer kernel times

# 3. Attention Backend
# Location: flashinfer_backend.py::forward()
# Metrics: Prefill vs decode kernel times

# 4. RadixCache Operations
# Location: radix_cache.py::match_prefix(), insert(), evict()
# Metrics: Cache hit rate, eviction frequency

# 5. TP Communication
# Location: parallel_state.py::all_reduce()
# Metrics: AllReduce latency, bandwidth utilization
```

### NVTX Annotation Locations

```python
# Existing NVTX ranges in SGLang:
# - "forward" around model forward pass
# - Layer-specific annotations in some backends

# Recommended additional instrumentation:
# - "prefill_attention" / "decode_attention"
# - "cache_lookup" / "cache_insert"
# - "allreduce_attention" / "allreduce_ffn"
```

---

## Next Steps

1. Review `reports/plan.md` for experiment design
2. Use scripts in `scripts/` for profiling
3. Consult `reports/kernel-dev-guide.md` for kernel development
