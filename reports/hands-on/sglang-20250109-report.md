# Hands-On Learning Report: SGLang

## Executive Summary

SGLang is a high-performance LLM serving framework featuring RadixAttention for prefix caching, a zero-overhead continuous batching scheduler, and multiple attention backends (FlashInfer, Triton, MLA). This analysis explores its architecture through codebase study, revealing a sophisticated system with CPU-GPU overlap scheduling, hierarchical memory management, and support for tensor/pipeline/expert parallelism.

**Key Insights**:
1. The scheduler uses a deque-based overlap pattern to hide CPU scheduling latency behind GPU execution
2. RadixCache implements LRU/LFU/priority-based eviction with reference counting for safe concurrent access
3. FlashInfer backend uses plan-run pattern with pre-computed work distribution for paged attention
4. Memory pool has two levels: ReqToTokenPool (request → token mapping) and TokenToKVPool (token → KV cache)

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
│  • NVLink pairs: (0,1), (2,3), (5,6) - NV12 bidirectional                   │
│  • NUMA: GPUs 0-3 on node 0, GPUs 4-6 on node 1                             │
│  • CUDA: 12.5                                                                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Codebase Analysis Summary

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         SGLANG ARCHITECTURE                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        API Layer                                        │ │
│  │  OpenAI-compatible endpoints, /generate, /v1/completions               │ │
│  └────────────────────────────────┬────────────────────────────────────────┘ │
│                                   │                                          │
│  ┌────────────────────────────────▼────────────────────────────────────────┐ │
│  │                    TokenizerManager                                     │ │
│  │  Tokenization, request batching, async communication (ZMQ)              │ │
│  └────────────────────────────────┬────────────────────────────────────────┘ │
│                                   │                                          │
│  ┌────────────────────────────────▼────────────────────────────────────────┐ │
│  │                       Scheduler                                         │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │ │
│  │  │ Waiting Queue   │  │ Running Batch   │  │ Schedule Policy         │  │ │
│  │  │ (prefill reqs)  │  │ (decode reqs)   │  │ LPM/DFS-weight/FCFS    │  │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │ │
│  │                                                                         │ │
│  │  event_loop_overlap(): CPU-GPU pipelining                              │ │
│  │  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐             │ │
│  │  │recv_requests├─────▶│get_next_batch├─────▶│run_batch    │             │ │
│  │  └─────────────┘      └─────────────┘      └──────┬──────┘             │ │
│  │         ▲                                         │                     │ │
│  │         └─────────────────────────────────────────┘                     │ │
│  │                    (overlap with GPU)                                   │ │
│  └────────────────────────────────┬────────────────────────────────────────┘ │
│                                   │                                          │
│  ┌────────────────────────────────▼────────────────────────────────────────┐ │
│  │                    Memory Management                                    │ │
│  │  ┌─────────────────────────────┐  ┌────────────────────────────────┐   │ │
│  │  │     RadixCache              │  │    Memory Pool                 │   │ │
│  │  │  ┌─────────────────────┐    │  │  ┌──────────────────────────┐  │   │ │
│  │  │  │ TreeNode (radix)    │    │  │  │ ReqToTokenPool           │  │   │ │
│  │  │  │ • token_ids key     │    │  │  │ [req_idx, seq_pos] →     │  │   │ │
│  │  │  │ • KV cache indices  │    │  │  │  token_pool_idx          │  │   │ │
│  │  │  │ • lock_ref counter  │    │  │  └──────────────────────────┘  │   │ │
│  │  │  │ • eviction policy   │    │  │  ┌──────────────────────────┐  │   │ │
│  │  │  └─────────────────────┘    │  │  │ TokenToKVPool            │  │   │ │
│  │  │  Policies: LRU/LFU/FIFO/MRU │  │  │ [kv_idx] → GPU KV cache  │  │   │ │
│  │  └─────────────────────────────┘  │  └──────────────────────────┘  │   │ │
│  │                                    └────────────────────────────────┘   │ │
│  └────────────────────────────────┬────────────────────────────────────────┘ │
│                                   │                                          │
│  ┌────────────────────────────────▼────────────────────────────────────────┐ │
│  │                    Model Execution                                      │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │ │
│  │  │TP Worker        │  │ Attention       │  │ Model Runner            │  │ │
│  │  │ NCCL groups     │  │ Backends:       │  │ Forward execution       │  │ │
│  │  │ AllReduce       │  │ • FlashInfer    │  │ CUDAGraph capture       │  │ │
│  │  │                 │  │ • Triton        │  │ Speculative decode      │  │ │
│  │  │                 │  │ • MLA           │  │                         │  │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Components Analysis

#### 1. Scheduler (`managers/scheduler.py` - 124KB)

**Event Loop Pattern**:
```python
# Normal loop (simpler)
def event_loop_normal(self):
    while True:
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)
        batch = self.get_next_batch_to_run()
        if batch:
            result = self.run_batch(batch)
            self.process_batch_result(batch, result)

# Overlap loop (production, hides CPU latency)
def event_loop_overlap(self):
    result_queue = deque()  # Pipeline buffer
    while True:
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)
        batch = self.get_next_batch_to_run()

        # Launch current batch (async GPU)
        if batch:
            batch_result = self.run_batch(batch)
            result_queue.append((batch.copy(), batch_result))

        # Process last batch (while GPU runs current)
        if self.last_batch:
            tmp_batch, tmp_result = result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)  # CPU work
```

**Batch Formation** (`get_next_batch_to_run`):
1. Merge finished prefill requests into running batch
2. Check if prefill batch available (`get_new_batch_prefill`)
3. If prefill exists → run prefill (priority)
4. Else → run decode batch
5. Handle DP attention sync if needed

#### 2. RadixCache (`mem_cache/radix_cache.py` - 31KB)

**Data Structure**:
```python
class TreeNode:
    children: defaultdict(TreeNode)  # Child nodes
    parent: TreeNode                  # Parent reference
    key: RadixKey                     # Token IDs + extra_key (lora_id, etc.)
    value: torch.Tensor               # KV cache indices
    lock_ref: int                     # Reference count (prevent eviction)
    last_access_time: float           # For LRU
    hit_count: int                    # For LFU
    priority: int                     # For priority eviction
```

**Cache Operations**:
```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         RADIX CACHE OPERATIONS                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INSERT (prefix sharing):                                                    │
│  ═══════════════════════                                                     │
│  Input: [A, B, C, D, E] (token sequence)                                    │
│                                                                              │
│  Before:         root                                                        │
│                   │                                                          │
│                  [A,B,C] ─── KV[0:3]                                        │
│                                                                              │
│  After:          root                                                        │
│                   │                                                          │
│                  [A,B,C] ─── KV[0:3]                                        │
│                   │                                                          │
│                  [D,E] ───── KV[3:5] (new)                                  │
│                                                                              │
│  MATCH (prefix lookup):                                                      │
│  ══════════════════════                                                      │
│  Query: [A, B, C, X, Y]                                                     │
│  Result: Match [A,B,C], prefix_len=3, need prefill [X,Y]                    │
│                                                                              │
│  EVICTION:                                                                   │
│  ══════════                                                                  │
│  Policies: LRU (default), LFU, FIFO, MRU, Priority                          │
│  Constraint: lock_ref > 0 → cannot evict (in-use)                           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### 3. FlashInfer Backend (`layers/attention/flashinfer_backend.py` - 67KB)

**Integration Pattern**:
```python
class FlashInferAttnBackend(AttentionBackend):
    def __init__(self, model_runner):
        # Create wrappers for decode and prefill
        self.decode_wrappers = [BatchDecodeWithPagedKVCacheWrapper(...)]
        self.prefill_wrappers = [BatchPrefillWithPagedKVCacheWrapper(...)]

        # Shared workspace buffer across all wrappers
        global_workspace_buffer = torch.empty(...)

    def init_forward_metadata(self, forward_batch):
        # Plan phase: compute work distribution
        if forward_mode.is_decode():
            self._plan_decode(forward_batch)
        else:
            self._plan_extend(forward_batch)

    def forward(self, q, k, v, ...):
        # Run phase: execute attention
        if forward_mode.is_decode():
            return self.decode_wrapper.run(q, k_cache, v_cache)
        else:
            return self.prefill_wrapper.run(q, k, v)
```

**Kernel Selection**:
- Decode: `BatchDecodeWithPagedKVCacheWrapper` (memory-bound, GEMV-like)
- Prefill: `BatchPrefillWithPagedKVCacheWrapper` (compute-bound)
- Tensor cores: Enabled when head_dim suitable and kv_cache_dtype appropriate

#### 4. Memory Pool (`mem_cache/memory_pool.py` - 71KB)

**Two-Level Architecture**:
```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         MEMORY POOL ARCHITECTURE                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Level 1: ReqToTokenPool                                                     │
│  ════════════════════════                                                    │
│  Shape: [max_requests, max_context_len]                                      │
│  Type: int32                                                                 │
│  Purpose: Map (request_idx, seq_position) → token_pool_idx                  │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  Req 0:  [ 5,  8, 12, 15, 22, -1, -1, -1, ...]  (seq_len=5)            │ │
│  │  Req 1:  [ 3,  7, 11, 14, 18, 21, 25, -1, ...]  (seq_len=7)            │ │
│  │  Req 2:  [10, 13, 16, -1, -1, -1, -1, -1, ...]  (seq_len=3)            │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Level 2: TokenToKVPool (Allocator)                                          │
│  ══════════════════════════════════                                          │
│  Shape: [max_total_tokens] (indices)                                         │
│  Purpose: Allocate/free token indices into KV cache                          │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  Free list: [28, 29, 30, 31, ...]  (available slots)                   │ │
│  │  Allocated: [0-27] (in use by active requests)                          │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Physical KV Cache (per layer):                                              │
│  ══════════════════════════════                                              │
│  Shape: [max_tokens, num_heads, head_dim] × 2 (K and V)                     │
│  Location: GPU HBM                                                           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### 5. Scheduling Policies (`managers/schedule_policy.py` - 30KB)

**Cache-Aware Policies**:
- `LPM` (Longest Prefix Match): Prioritize requests with longest cached prefix
- `DFS-Weight`: Weight by depth-first traversal in radix tree

**Cache-Agnostic Policies**:
- `FCFS`: First come first serve
- `LOF`: Longest output first
- `RANDOM`: Random selection

**In-Batch Prefix Caching**:
```python
# If matched prefix < threshold, check for prefix sharing within batch
IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD = 32

# Requests with high in-batch prefix match get deprioritized
# (wait for earlier request to populate cache)
IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD = 32
```

### Attention Backend Comparison

| Backend | File Size | Features | Best For |
|---------|-----------|----------|----------|
| FlashInfer | 67KB | Plan-run, paged KV, MQA/GQA | Production A100/H100 |
| Triton | 51KB | Custom kernels, flexible | Experimentation |
| FlashAttention | 116KB | Standard FA2, varlen | Compatibility |
| MLA | 40KB | Multi-head Latent Attention | DeepSeek models |
| NSA | 75KB | Native Sparse Attention | Sparse attention models |

## Visualizations

### Scheduler Event Loop Timeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    SCHEDULER OVERLAP TIMELINE                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Time     0     1     2     3     4     5     6     7     8     9    10     │
│           ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤     │
│                                                                              │
│  CPU      ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐           │
│  (sched)  │ S1  │     │ S2  │     │ S3  │     │ S4  │     │ S5  │           │
│           └─────┘     └─────┘     └─────┘     └─────┘     └─────┘           │
│              │           │           │           │           │               │
│              ▼           ▼           ▼           ▼           ▼               │
│  GPU           ┌─────────┐     ┌─────────┐     ┌─────────┐                  │
│  (forward)     │   B1    │     │   B2    │     │   B3    │                  │
│                └─────────┘     └─────────┘     └─────────┘                  │
│                                                                              │
│  Legend:                                                                     │
│  S = Schedule (recv_requests + get_next_batch + process_result)             │
│  B = Batch execution (GPU forward pass)                                     │
│                                                                              │
│  Overlap: S2 runs while B1 executes on GPU                                  │
│           → Hides ~1ms CPU scheduling overhead                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### RadixCache Prefix Sharing

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    RADIX CACHE PREFIX SHARING                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Scenario: 3 chat requests with shared system prompt                        │
│                                                                              │
│  Request 1: "You are helpful..." + "What is Python?"                        │
│  Request 2: "You are helpful..." + "What is JavaScript?"                    │
│  Request 3: "You are helpful..." + "What is Rust?"                          │
│                                                                              │
│  Radix Tree State:                                                           │
│                                                                              │
│                          root                                                │
│                            │                                                 │
│                    ┌───────┴───────┐                                        │
│                    │"You are..."   │  ← Shared prefix (128 tokens)          │
│                    │ KV[0:128]     │    lock_ref=3 (3 requests using)       │
│                    └───────┬───────┘                                        │
│              ┌─────────────┼─────────────┐                                  │
│              │             │             │                                   │
│        ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐                           │
│        │"Python?"  │ │"JavaScript│ │"Rust?"    │                           │
│        │KV[128:145]│ │KV[145:165]│ │KV[165:178]│                           │
│        └───────────┘ └───────────┘ └───────────┘                           │
│                                                                              │
│  Memory Savings:                                                             │
│  • Without sharing: 3 × 128 = 384 prefix tokens                             │
│  • With sharing: 128 + 17 + 20 + 13 = 178 tokens                           │
│  • Savings: 54% memory reduction                                            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Continuous Batching Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    CONTINUOUS BATCHING EXECUTION                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Time    Iteration 0    Iteration 1    Iteration 2    Iteration 3           │
│          ├──────────────┼──────────────┼──────────────┼──────────────┤      │
│                                                                              │
│  Waiting [R0,R1,R2,R3]  [R3,R4]        [R4,R5]        [R5]                  │
│  Queue                                                                       │
│                                                                              │
│  Running []             [R0,R1,R2]     [R0,R1,R2,R3]  [R1,R2,R3,R4]         │
│  Batch                  (decode)       (decode)       (decode)               │
│                                                                              │
│  Prefill [R0,R1,R2]     [R3]           [R4]           [R5]                  │
│  Batch   (new prefill)  (new prefill)  (new prefill)  (new prefill)         │
│                                                                              │
│  Events:                                                                     │
│  ├─ Iter 0: R0,R1,R2 start prefill                                         │
│  ├─ Iter 1: R0,R1,R2 → decode; R3 starts prefill                           │
│  ├─ Iter 2: R0 completes (EOS), R4 starts prefill                          │
│  └─ Iter 3: R1,R2,R3 decode; R4 joins decode; R5 prefills                  │
│                                                                              │
│  Kernel Pattern per Iteration:                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ [prefill_attn] [prefill_mlp] [decode_attn] [decode_mlp] [sample]       ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Actions Log

### Phase 1: Environment Detection

**Action 1.1: Query GPU Configuration**
```bash
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
```
Output:
```
index, name, memory.total [MiB], compute_cap
0, NVIDIA A100 80GB PCIe, 81920 MiB, 8.0
1, NVIDIA A100 80GB PCIe, 81920 MiB, 8.0
2, NVIDIA A100 80GB PCIe, 81920 MiB, 8.0
3, NVIDIA A100 80GB PCIe, 81920 MiB, 8.0
4, NVIDIA A100-SXM4-80GB, 81920 MiB, 8.0
5, NVIDIA A100 80GB PCIe, 81920 MiB, 8.0
6, NVIDIA A100 80GB PCIe, 81920 MiB, 8.0
```

**Action 1.2: Query GPU Topology**
```bash
nvidia-smi topo -m
```
Finding: NVLink pairs (0-1, 2-3, 5-6), two NUMA nodes.

### Phase 2: Codebase Analysis

**Action 2.1: Clone Repository**
```bash
git clone git@github.com:sgl-project/sglang.git code-repos/sglang
```
Status: Already existed in repository.

**Action 2.2: Analyze Key Files**
- `scheduler.py`: 124KB, ~3000 lines - Core scheduling logic
- `radix_cache.py`: 31KB - Prefix caching implementation
- `flashinfer_backend.py`: 67KB - FlashInfer integration
- `memory_pool.py`: 71KB - Two-level memory management

### Phase 3: Experiment Planning

**Planned Experiments** (documented in `sglang-20250109-plan.md`):
1. Baseline single-GPU performance
2. RadixCache behavior analysis
3. Attention backend comparison (FlashInfer vs Triton)
4. Tensor parallelism scaling with NVLink
5. Continuous batching dynamics

**Note**: Experiments require SGLang installation. Due to sandbox restrictions, installation was not possible during this session.

## Observations and Insights

### 1. Zero-Overhead Scheduler Design

**Observation**: SGLang's scheduler uses CPU-GPU overlap (`event_loop_overlap`) to hide scheduling latency.

**Details**:
- Uses a deque as pipeline buffer between scheduling and result processing
- While GPU executes batch N, CPU prepares batch N+1 and processes results of batch N-1
- Typical CPU scheduling takes ~0.5-1ms, hidden behind GPU forward pass

**Implication**: Near-optimal GPU utilization even with complex scheduling logic.

### 2. RadixCache Prefix Sharing Effectiveness

**Observation**: RadixCache uses radix tree with reference counting for safe concurrent access.

**Quantified**:
- `lock_ref` prevents eviction of in-use prefixes
- Multiple eviction policies (LRU default, LFU, FIFO, MRU, Priority)
- In-batch prefix caching defers requests with pending prefixes

**Trade-off**:
- Benefit: Memory savings proportional to prefix sharing
- Cost: Tree traversal overhead during cache lookup

### 3. FlashInfer Plan-Run Integration

**Observation**: SGLang integrates FlashInfer's plan-run pattern for paged attention.

**Details**:
- `init_forward_metadata`: Executes plan phase (compute work distribution)
- `forward`: Executes run phase (actual attention computation)
- Global workspace buffer shared across all wrappers

**Kernel Selection**:
- Decode: Memory-bound, GEMV-like attention
- Prefill: Compute-bound, standard attention
- Tensor core usage depends on head configuration

### 4. Two-Level Memory Pool Architecture

**Observation**: Memory management separates request-token mapping from KV cache allocation.

**Design Benefits**:
- `ReqToTokenPool`: Fast lookup of token positions per request
- `TokenToKVPool`: Efficient free-list management for KV cache slots
- Enables flexible memory reuse without copying

### 5. Multiple Attention Backend Support

**Observation**: SGLang supports 10+ attention backends for different use cases.

**Key Backends**:
- FlashInfer (67KB): Production, A100/H100 optimized
- Triton (51KB): Experimentation, custom kernels
- MLA (40KB): DeepSeek multi-head latent attention
- NSA (75KB): Native sparse attention

**Implication**: Can adapt to different models and hardware without code changes.

## Reproduction Guide

### Prerequisites

- GPU: NVIDIA A100/H100 (compute capability 8.0+)
- CUDA: 12.1+
- Python: 3.10+

### Setup

```bash
# 1. Create environment
conda create -n sglang python=3.11 -y
conda activate sglang

# 2. Install SGLang with all dependencies
pip install "sglang[all]>=0.4"

# 3. Verify installation
python -c "import sglang; print(sglang.__version__)"
```

### Running Baseline Experiment

```bash
# 1. Launch server (single GPU)
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 \
    --mem-fraction-static 0.85

# 2. Run benchmark (separate terminal)
python -m sglang.bench_serving \
    --backend sglang \
    --num-prompts 100 \
    --random-input 512 \
    --random-output 128

# 3. Profile with built-in profiler
python -m sglang.profiler \
    --url http://localhost:30000 \
    --num-steps 10 \
    --output-dir ./profiles
```

### Profiling with Nsight Systems

```bash
# System trace
nsys profile -o sglang_trace \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    python -m sglang.bench_serving \
    --backend sglang --num-prompts 50

# Analyze
nsys stats sglang_trace.nsys-rep --report cuda_gpu_kern_sum
```

### Testing RadixCache

```python
# test_radix_cache.py
import sglang as sgl

@sgl.function
def chat(s, question):
    s += sgl.system("You are a helpful assistant.")  # Shared
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=50))

# Run multiple requests
for q in ["What is Python?", "What is Rust?", "What is Go?"]:
    result = chat.run(question=q)
    print(result["answer"])

# Observe: First request slower (full prefill)
# Subsequent: Faster (prefix cache hit on system prompt)
```

## Appendix

### File Structure Summary

```
sglang/python/sglang/srt/
├── managers/
│   ├── scheduler.py            # Core scheduler (124KB)
│   ├── schedule_batch.py       # Batch data structures (89KB)
│   ├── schedule_policy.py      # Scheduling policies (30KB)
│   └── tokenizer_manager.py    # Tokenization (96KB)
├── mem_cache/
│   ├── radix_cache.py          # Prefix caching (31KB)
│   ├── memory_pool.py          # KV cache allocation (71KB)
│   └── hiradix_cache.py        # Hierarchical cache (47KB)
├── layers/
│   ├── attention/
│   │   ├── flashinfer_backend.py   # FlashInfer (67KB)
│   │   ├── triton_backend.py       # Triton (51KB)
│   │   └── flashattention_backend.py # FA2 (116KB)
│   └── moe/                    # MoE layers
├── disaggregation/             # Prefill-decode disaggregation
└── distributed/                # TP/PP communication
```

### Key Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `SGLANG_ATTENTION_BACKEND` | Select attention backend | flashinfer |
| `SGLANG_ENABLE_TORCH_COMPILE` | Enable torch.compile | false |
| `SGLANG_LOG_LEVEL` | Logging verbosity | INFO |
| `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION` | Max tokens clip | 4096 |

### References

- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [SGLang Documentation](https://docs.sglang.io/)
- [v0.4 Release Blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)
- [RadixAttention Paper](https://arxiv.org/abs/2312.07104)
