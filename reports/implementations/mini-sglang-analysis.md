# Analysis: Mini-SGLang

**Repository:** https://github.com/sgl-project/mini-sglang
**Framework:** Python + PyTorch + Flash Attention
**Analysis Date:** 2026-01-06

---

## 1. Executive Summary

Mini-SGLang is a lightweight (~5k lines) yet production-grade LLM inference engine derived from SGLang. It demonstrates core serving system concepts including **RadixAttention** for KV cache reuse, **chunked prefill** for memory control, **overlap scheduling** to hide CPU overhead, and **tensor parallelism** for multi-GPU serving.

**Key Features:**
- OpenAI-compatible API server (FastAPI)
- RadixAttention with prefix caching and LRU eviction
- Continuous batching with chunked prefill
- Overlap scheduling (CPU-GPU pipelining)
- CUDA graph support for decode optimization
- Tensor Parallelism via PyNccl

---

## 2. Architecture Overview

### 2.1 System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MINI-SGLANG ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐   │
│  │   API Server    │────▶│  Tokenizer       │────▶│    Scheduler       │   │
│  │   (FastAPI)     │     │  Server          │     │    (per GPU)       │   │
│  │                 │◀────│                  │◀────│                    │   │
│  └─────────────────┘     └──────────────────┘     └─────────┬──────────┘   │
│         ZMQ                     ZMQ                         │              │
│                                                             │              │
│                         ┌───────────────────────────────────┴──────────┐   │
│                         │              SCHEDULER LOOP                   │   │
│                         │  ┌─────────────────────────────────────────┐ │   │
│                         │  │         Overlap Scheduling              │ │   │
│                         │  │  ┌───────────┐    ┌──────────────────┐  │ │   │
│                         │  │  │ CPU Work  │    │    GPU Work      │  │ │   │
│                         │  │  │ (Stream 1)│    │   (Stream 2)     │  │ │   │
│                         │  │  │           │    │                  │  │ │   │
│                         │  │  │ • Recv msg│    │ • Forward batch  │  │ │   │
│                         │  │  │ • Schedule│    │ • Sample tokens  │  │ │   │
│                         │  │  │ • Prepare │    │ • CUDA graph     │  │ │   │
│                         │  │  └───────────┘    └──────────────────┘  │ │   │
│                         │  └─────────────────────────────────────────┘ │   │
│                         └──────────────────────────────────────────────┘   │
│                                           │                                 │
│                         ┌─────────────────▼──────────────────────┐         │
│                         │              ENGINE                     │         │
│                         │  ┌──────────┐  ┌──────────┐  ┌───────┐ │         │
│                         │  │  Model   │  │ KV Cache │  │ CUDA  │ │         │
│                         │  │ (Llama/  │  │ (Paged)  │  │ Graph │ │         │
│                         │  │  Qwen)   │  │          │  │Runner │ │         │
│                         │  └──────────┘  └──────────┘  └───────┘ │         │
│                         └────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Module Hierarchy

```
minisgl/
├── server/
│   ├── api_server.py      # FastAPI endpoints, FrontendManager
│   ├── launch.py          # Process orchestration
│   └── args.py            # ServerArgs configuration
├── tokenizer/
│   ├── server.py          # Tokenizer process
│   ├── tokenize.py        # Input tokenization
│   └── detokenize.py      # Output detokenization
├── scheduler/
│   ├── scheduler.py       # Main scheduler loop
│   ├── prefill.py         # PrefillManager, ChunkedReq
│   ├── decode.py          # DecodeManager
│   ├── cache.py           # CacheManager wrapper
│   └── io.py              # ZMQ I/O mixin
├── engine/
│   ├── engine.py          # Engine: model + KV cache + sampler
│   ├── graph.py           # GraphRunner: CUDA graph capture/replay
│   └── sample.py          # Sampler: greedy/top-k/top-p
├── kvcache/
│   ├── radix_manager.py   # RadixCacheManager (RadixAttention)
│   ├── naive_manager.py   # Simple block allocator
│   └── mha_pool.py        # KV cache tensor pool
├── attention/
│   ├── fa.py              # FlashAttentionBackend
│   └── fi.py              # FlashInferBackend
├── models/
│   ├── llama.py           # LlamaModel
│   ├── qwen3.py           # Qwen3Model
│   └── weight.py          # Weight loading
├── layers/
│   ├── attention.py       # Attention layer
│   ├── linear.py          # TP-aware linear layers
│   └── rotary.py          # RoPE implementation
└── core.py                # Req, Batch, Context, SamplingParams
```

### 2.3 Execution Model

**Three-Process Architecture:**
1. **API Server** (frontend): Handles HTTP requests, manages async responses
2. **Tokenizer Server**: Tokenizes inputs, detokenizes outputs
3. **Scheduler** (backend): GPU execution, batching, KV cache management

Communication via ZeroMQ (ZMQ) push/pull queues.

---

## 3. Configuration & Parameters

### 3.1 Server Configuration (ServerArgs)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | required | HuggingFace model path |
| `tp_size` | 1 | Tensor parallelism degree |
| `max_running_req` | 256 | Max concurrent requests |
| `max_extend_tokens` | 8192 | Prefill budget per batch |
| `max_seq_len` | 8192 | Max sequence length |
| `memory_ratio` | 0.9 | GPU memory for KV cache |
| `cache_type` | "radix" | KV cache manager type |
| `attention_backend` | "fa" | "fa" or "fi" |

### 3.2 Model Configuration

| Parameter | Llama-style | Description |
|-----------|-------------|-------------|
| `hidden_size` | 4096 | Hidden dimension |
| `num_layers` | 32 | Decoder layers |
| `num_heads` | 32 | Attention heads |
| `num_kv_heads` | 8 | KV heads (GQA) |
| `head_dim` | 128 | Per-head dimension |
| `intermediate_size` | 14336 | MLP hidden size |

### 3.3 Constraints

- Page size: 1 token (fine-grained allocation)
- Page table alignment: 32 entries (128 bytes)
- CUDA graph batch sizes: Configurable list

---

## 4. Component Analysis

### 4.1 Request Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      REQUEST LIFECYCLE                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Client                                                                  │
│    │                                                                     │
│    │ POST /v1/completions                                               │
│    ▼                                                                     │
│  ┌─────────────┐                                                        │
│  │ API Server  │  ────▶ TokenizeMsg ────▶  Tokenizer  ────▶ UserMsg     │
│  └─────────────┘                                                        │
│                                                                          │
│  UserMsg ────────────────────────────────────────────▶ Scheduler        │
│                                                           │              │
│                                                           ▼              │
│                                              ┌────────────────────────┐ │
│                                              │   PREFILL MANAGER      │ │
│                                              │   • Match prefix cache │ │
│                                              │   • Allocate table_idx │ │
│                                              │   • Chunk if needed    │ │
│                                              └───────────┬────────────┘ │
│                                                          │              │
│                                                          ▼              │
│                                              ┌────────────────────────┐ │
│                                              │   BATCH EXECUTION      │ │
│                                              │   • Prefill tokens     │ │
│                                              │   • Store KV cache     │ │
│                                              │   • Sample next token  │ │
│                                              └───────────┬────────────┘ │
│                                                          │              │
│                                                          ▼              │
│                                              ┌────────────────────────┐ │
│                                              │   DECODE MANAGER       │ │
│                                              │   • Add to running     │ │
│                                              │   • Schedule batches   │ │
│                                              └───────────┬────────────┘ │
│                                                          │              │
│                                                          │ (loop until  │
│                                                          │  EOS/max_len)│
│                                                          ▼              │
│  ◀──────────────────── DetokenizeMsg ◀─────────── Scheduler            │
│                                                                          │
│  Client ◀──────── StreamingResponse ◀─────────── API Server            │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Core Data Structures

**Req (core.py:28)**
```python
@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor    # CPU tensor of token IDs
    table_idx: int             # Index into page_table
    cached_len: int            # Tokens with cached KV
    output_len: int            # Max output tokens
    uid: int                   # Unique request ID
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle  # RadixTree node reference

    # Derived:
    device_len: int            # Current total length
    max_device_len: int        # input_len + output_len

    @property
    def extend_len(self):
        return self.device_len - self.cached_len  # Tokens to compute
```

**Batch (core.py:70)**
```python
@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]

    # Set by scheduler:
    input_ids: torch.Tensor    # [total_tokens]
    out_loc: torch.Tensor      # Page indices for new KV
    padded_reqs: List[Req]     # May include dummy reqs

    # Set by attention backend:
    attn_metadata: BaseAttnMetadata
```

### 4.3 RadixAttention (KV Cache Manager)

**Purpose:** Prefix caching via radix tree for KV reuse

**Data Structure (kvcache/radix_manager.py:13):**
```python
class RadixTreeNode:
    children: Dict[int, RadixTreeNode]  # token_id → child
    _parent: RadixTreeNode | None
    ref_count: int                       # 0 = evictable
    timestamp: int                       # For LRU eviction
    _key: torch.Tensor                   # Token IDs for this edge
    _value: torch.Tensor                 # Page indices (KV locations)
    _length: int                         # Length of key/value
```

**Pseudocode:**
```python
def match_prefix(input_ids):
    """
    Walk radix tree to find longest cached prefix

    Returns: (handle, matched_page_indices)
    """
    node = root
    prefix_len = 0

    while prefix_len < len(input_ids):
        token = input_ids[prefix_len]
        if token not in node.children:
            break  # No more matches

        child = node.children[token]
        match_len = compare(child.key, input_ids[prefix_len:])
        prefix_len += match_len

        if match_len < child.length:
            # Partial match: split node
            child = child.split_at(match_len)
            break

        node = child
        node.timestamp = now()  # Update for LRU

    # Collect page indices from root to matched node
    return collect_values(root, node)

def evict(needed_size):
    """
    Evict leaf nodes with ref_count=0, LRU order
    """
    heap = min_heap([leaf for leaf in leaves if leaf.ref_count == 0])

    evicted = []
    while sum(evicted.length) < needed_size:
        node = heap.pop()  # Oldest leaf
        evicted.append(node.value)  # Page indices to free
        parent = node.parent
        del parent.children[node.key[0]]

        # Parent may become evictable leaf
        if parent.is_leaf() and parent.ref_count == 0:
            heap.push(parent)

    return concat(evicted)
```

**Visualization:**
```
                    RADIX TREE EXAMPLE
                    ==================

Requests:
  A: "The quick brown fox"     → tokens [101, 202, 303, 404]
  B: "The quick red dog"       → tokens [101, 202, 505, 606]
  C: "The slow turtle"         → tokens [101, 707, 808]

Tree Structure:
                    [root]
                      │
                 [101] "The"
                 pages=[0]
                      │
            ┌─────────┴─────────┐
            │                   │
       [202] "quick"       [707] "slow"
       pages=[1]           pages=[4]
            │                   │
      ┌─────┴─────┐        [808] "turtle"
      │           │        pages=[5]
 [303,404]    [505,606]
 "brown fox"  "red dog"
 pages=[2,3]  pages=[6,7]

Prefix Reuse:
  - Request A computed: all nodes
  - Request B: reuses [101,202], computes [505,606]
  - Request C: reuses [101], computes [707,808]
```

### 4.4 Overlap Scheduling

**Purpose:** Hide CPU overhead by pipelining batch preparation with GPU execution

**Implementation (scheduler/scheduler.py:231):**
```python
def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
    """
    Timeline:
    ─────────────────────────────────────────────────────▶ time

    Iteration N:
    CPU: [recv_msg][schedule][prepare_batch]────────────────
    GPU: ────────────────────[forward N][sample]────────────

    Iteration N+1:
    CPU: [process_last]──────[recv_msg][schedule][prepare]──
    GPU: ──────────────────────────────[forward N+1][sample]

    Overlap: CPU prepares N+1 while GPU executes N
    """
    # 1. Receive new messages (non-blocking if work pending)
    blocking = not (last_data or self.prefill_manager.runnable
                    or self.decode_manager.runnable)
    for msg in self.receive_msg(blocking=blocking):
        self._process_one_msg(msg)

    # 2. Schedule next batch (on CPU stream)
    forward_input = self._schedule_next_batch()

    # 3. Execute on GPU (on engine stream)
    ongoing_data = None
    if forward_input is not None:
        with self.engine_stream_ctx:
            self.engine.stream.wait_stream(self.stream)  # Sync point
            ongoing_data = (forward_input, self._forward(forward_input))

    # 4. Process last batch results (while GPU runs)
    self._process_last_data(last_data, ongoing_data)

    return ongoing_data
```

**Stream Diagram:**
```
CPU Stream (self.stream):
┌─────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐
│ Recv    │ │  Schedule   │ │   Prepare   │ │ Process │
│ Messages│ │  Next Batch │ │  Metadata   │ │  Last   │
└─────────┘ └─────────────┘ └─────────────┘ └─────────┘
────────────────────────────────────────────────────────▶ time

GPU Stream (self.engine.stream):
            ┌─────────────────────────────────────────┐
            │  wait_stream   │   Forward   │  Sample  │
            │                │   (Model)   │          │
            └─────────────────────────────────────────┘
────────────────────────────────────────────────────────▶ time
```

### 4.5 Chunked Prefill

**Purpose:** Process long prompts in chunks to bound memory and allow decode interleaving

**Implementation (scheduler/prefill.py:63):**
```python
def _add_one_req(self, pending_req, cache_handle, table_idx, cached_len):
    remain_len = pending_req.input_len - cached_len
    chunk_size = min(self.token_budget, remain_len)
    is_chunked = chunk_size < remain_len

    # ChunkedReq: partial prefill, will continue later
    # Req: complete prefill, moves to decode
    CLS = ChunkedReq if is_chunked else Req
    self.token_budget -= chunk_size

    return CLS(
        input_ids=pending_req.input_ids[:cached_len + chunk_size],
        table_idx=table_idx,
        cached_len=cached_len,
        ...
    )
```

**Pseudocode:**
```python
def schedule_next_batch(prefill_budget):
    """
    Schedule prefill requests within token budget

    Budget: max_extend_tokens (e.g., 8192)
    Each request consumes: input_len - cached_len tokens
    """
    reqs = []
    remaining_budget = prefill_budget

    for pending in pending_list:
        extend_len = pending.input_len - pending.cached_len

        if extend_len <= remaining_budget:
            # Full prefill
            reqs.append(create_req(pending, extend_len))
            remaining_budget -= extend_len
        elif remaining_budget > 0:
            # Partial prefill (chunked)
            reqs.append(create_chunked_req(pending, remaining_budget))
            remaining_budget = 0
            # Re-queue for continuation
            pending_list.prepend(pending)
        else:
            break  # Budget exhausted

    return Batch(reqs, phase="prefill") if reqs else None
```

### 4.6 Engine & CUDA Graphs

**Purpose:** Model forward pass with optional CUDA graph capture for decode

**Engine Initialization (engine/engine.py:36):**
```python
class Engine:
    def __init__(self, config):
        # 1. Initialize model on meta device, then load weights
        with torch.device("meta"):
            self.model = create_model(config.model_path)
        self.model.load_state_dict(load_weights())

        # 2. Determine KV cache size from free memory
        self.num_pages = (free_memory * memory_ratio) // cache_per_page
        self.kv_cache = create_kvcache(num_pages=self.num_pages)

        # 3. Initialize CUDA graph runner
        self.graph_runner = GraphRunner(
            cuda_graph_bs=[1, 2, 4, 8, 16, ...],  # Captured batch sizes
            max_seq_len=self.max_seq_len,
        )
```

**Forward with CUDA Graph (engine/engine.py:188):**
```python
def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
    with self.ctx.forward_batch(batch):
        if self.graph_runner.can_use_cuda_graph(batch):
            # Replay captured graph (decode only)
            logits = self.graph_runner.replay(batch)
        else:
            # Dynamic execution (prefill or unsupported BS)
            logits = self.model.forward()

    # Sample next tokens
    next_tokens_gpu = self.sampler.sample(logits[:batch.size], args)
    next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)

    return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)
```

---

## 5. Task/Operator Catalog

| Component | Module | Purpose | Key Methods |
|-----------|--------|---------|-------------|
| API Server | `server/api_server.py` | HTTP endpoints | `generate`, `chat_completions` |
| Tokenizer | `tokenizer/server.py` | Text ↔ tokens | `tokenize`, `detokenize` |
| Scheduler | `scheduler/scheduler.py` | Batch orchestration | `overlap_loop`, `run_forever` |
| PrefillManager | `scheduler/prefill.py` | Prompt processing | `schedule_next_batch`, `add_one_req` |
| DecodeManager | `scheduler/decode.py` | Token generation | `schedule_next_batch`, `add_req` |
| CacheManager | `scheduler/cache.py` | KV cache allocation | `allocate`, `free`, `match_req` |
| RadixManager | `kvcache/radix_manager.py` | Prefix caching | `match_prefix`, `insert_prefix`, `evict` |
| Engine | `engine/engine.py` | Model execution | `forward_batch` |
| GraphRunner | `engine/graph.py` | CUDA graph | `capture`, `replay` |
| FlashAttnBackend | `attention/fa.py` | Attention compute | `forward`, `prepare_metadata` |

---

## 6. Data Flow & Memory

### 6.1 KV Cache Layout

```
                        KV CACHE STRUCTURE
                        ==================

┌─────────────────────────────────────────────────────────────────────┐
│  KV Cache Pool: [num_pages, page_size=1, num_kv_heads, head_dim]   │
│                                                                     │
│  k_cache[layer_id]:                                                 │
│  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐         │
│  │Page 0│Page 1│Page 2│Page 3│Page 4│Page 5│Page 6│ ...  │         │
│  └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘         │
│                                                                     │
│  Page Table: [max_running_req, max_seq_len]                        │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │ Req 0: [0, 1, 2, 3, _, _, _, ...]     ← 4 tokens        │       │
│  │ Req 1: [4, 5, _, _, _, _, _, ...]     ← 2 tokens        │       │
│  │ Req 2: [0, 1, 6, 7, 8, _, _, ...]     ← prefix reuse!   │       │
│  │ ...                                                      │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                                                     │
│  Token Pool: [max_running_req, max_seq_len]                        │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │ Req 0: [101, 202, 303, 404, _, _, ...]   (CPU→GPU)      │       │
│  │ Req 1: [505, 606, _, _, _, _, _, ...]                   │       │
│  │ ...                                                      │       │
│  └─────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Batch Execution Flow

```
                    BATCH FORWARD PASS
                    ===================

┌──────────────────────────────────────────────────────────────────┐
│  1. PREPARE METADATA (scheduler)                                  │
│     • Compute cu_seqlens_q, cu_seqlens_k                         │
│     • Build page_table for batch                                  │
│     • Allocate out_loc (new page indices)                        │
└───────────────────────────┬──────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  2. LOAD TOKEN IDS (scheduler)                                    │
│     batch.input_ids = token_pool.view(-1)[load_indices]          │
└───────────────────────────┬──────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  3. MODEL FORWARD (engine)                                        │
│                                                                   │
│  input_ids ──▶ Embedding ──▶ [hidden_states]                     │
│                                                                   │
│  For each layer:                                                  │
│    hidden ──▶ RMSNorm ──▶ Attention ──▶ RMSNorm ──▶ MLP ──▶     │
│                              │                                    │
│                    ┌─────────┴─────────┐                         │
│                    │  ATTENTION FLOW   │                         │
│                    │                   │                         │
│                    │  Q,K,V = proj(h)  │                         │
│                    │  K,V → store KV   │ ◀── out_loc             │
│                    │  attn = FA(Q,K,V) │ ◀── page_table          │
│                    │  out = proj(attn) │                         │
│                    └───────────────────┘                         │
│                                                                   │
│  hidden ──▶ RMSNorm ──▶ LMHead ──▶ [logits]                     │
└───────────────────────────┬──────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  4. SAMPLE (engine)                                               │
│     next_tokens = sample(logits, temperature, top_k, top_p)      │
└───────────────────────────┬──────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  5. WRITE TOKEN IDS (scheduler)                                   │
│     token_pool.view(-1)[write_indices] = next_tokens             │
└──────────────────────────────────────────────────────────────────┘
```

---

## 7. Parallelism & Distribution

### 7.1 Tensor Parallelism

**Strategy:** Column-parallel for Q/K/V/Gate/Up, Row-parallel for O/Down

**Implementation (layers/linear.py):**
```python
class ColumnParallelLinear:
    """Split output dimension across TP ranks"""
    def __init__(self, in_features, out_features, tp_size):
        self.out_features_per_rank = out_features // tp_size
        self.weight = ...  # [out_features_per_rank, in_features]

    def forward(self, x):
        return F.linear(x, self.weight)  # Each rank has partial output


class RowParallelLinear:
    """Split input dimension across TP ranks, all-reduce output"""
    def __init__(self, in_features, out_features, tp_size):
        self.in_features_per_rank = in_features // tp_size
        self.weight = ...  # [out_features, in_features_per_rank]

    def forward(self, x):
        partial = F.linear(x, self.weight)
        return all_reduce(partial)  # Sum across ranks
```

### 7.2 Communication

| Operation | Location | Primitive |
|-----------|----------|-----------|
| After O projection | Attention | AllReduce |
| After Down projection | MLP | AllReduce |

**PyNccl Integration (kernel/pynccl.py):**
- Uses PyNccl for efficient NCCL operations
- Gloo group for CPU synchronization

---

## 8. Technical FAQ

### Q: How does prefix caching work?
**A:** RadixAttention stores KV cache keyed by token sequences in a radix tree. When a new request arrives, `match_prefix()` walks the tree to find the longest matching prefix. Matched KV blocks are reused, only non-cached suffix is computed.
*Reference: kvcache/radix_manager.py:116-128*

### Q: How are variable sequence lengths handled?
**A:** Flash Attention with variable-length support via `cu_seqlens_q` and `cu_seqlens_k` cumulative sequence length tensors. No padding within attention computation.
*Reference: attention/fa.py:67-105*

### Q: What is the scheduling policy?
**A:** Prefill-first: `schedule_next_batch` tries prefill queue before decode queue. Within prefill, FCFS with chunking for memory control.
*Reference: scheduler/scheduler.py:172-178*

### Q: How does overlap scheduling reduce latency?
**A:** Two CUDA streams: CPU work (message processing, scheduling, metadata prep) runs on `self.stream`, GPU work (forward, sample) runs on `self.engine.stream`. CPU prepares batch N+1 while GPU executes batch N.
*Reference: scheduler/scheduler.py:231-254*

### Q: When are CUDA graphs used?
**A:** Decode-only, when batch size matches a pre-captured size. Prefill and variable-length operations use dynamic execution.
*Reference: engine/graph.py*

### Q: How is memory managed?
**A:** Fine-grained page allocation (page_size=1 token). Free pages tracked in allocator. RadixAttention evicts LRU leaves with ref_count=0 when memory pressure.
*Reference: kvcache/radix_manager.py:166-193*

---

## 9. Key Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `scheduler/scheduler.py` | 284 | Main scheduler loop, overlap scheduling |
| `kvcache/radix_manager.py` | 221 | RadixAttention implementation |
| `engine/engine.py` | 208 | Model loading, forward execution |
| `attention/fa.py` | 186 | Flash Attention backend |
| `scheduler/prefill.py` | 153 | Chunked prefill manager |
| `engine/graph.py` | 144 | CUDA graph capture/replay |
| `core.py` | 129 | Req, Batch, Context data classes |
| `server/api_server.py` | 437 | FastAPI server, OpenAI API |

---

## 10. Appendix

### A. Glossary

| Term | Definition |
|------|------------|
| RadixAttention | Prefix tree-based KV cache sharing across requests |
| Chunked Prefill | Processing long prompts in budget-limited chunks |
| Overlap Scheduling | CPU-GPU pipelining to hide scheduling overhead |
| Page Table | Mapping from (request, position) to physical KV page |
| cu_seqlens | Cumulative sequence lengths for variable-length batching |

### B. References

- [Mini-SGLang Blog](https://lmsys.org/blog/2025-12-17-minisgl/)
- [SGLang: RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [vLLM PagedAttention](https://arxiv.org/abs/2309.06180)

---

*Generated by LLM Code Analysis Skill*
