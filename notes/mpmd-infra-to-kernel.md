# Making It Clear about MPMD from Serving to Kernel

**References**:
1. `notes/nanoflow-paper.md`, `reports/implementations/nanoflow-analysis.md`
2. `reports/implementations/triton-distributed-megakernel-analysis.md`
3. `reports/implementations/mini-sglang-analysis.md`
4. `notes/twill-paper.md` (software pipelining + warp specialization optimization)

## Purpose of the Note

This note provides a complete view of how LLM serving runs on GPU resources, from high-level infrastructure to low-level kernel execution. It addresses:

1. Multi-level task decomposition along the timeline
2. SPMD programming model implementing MPMD-like runtime behavior
3. Design space analysis for throughput and latency optimization

---

## 1. Multi-Level Task Hierarchy

### 1.1 The Three Levels

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       LLM SERVING TASK HIERARCHY                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LEVEL 1: INFRASTRUCTURE (Serving)                                           │
│  ══════════════════════════════════════════════════════════════════════════  │
│  Granularity: Requests, Batches                                              │
│  Timeline:    Seconds to minutes                                             │
│  Resources:   CPU scheduler, GPU memory, Network                             │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  Request Queue     Batch Scheduler     GPU Execution      Response     │  │
│  │  ┌───┐ ┌───┐      ┌─────────────┐     ┌───────────┐     ┌─────────┐    │  │
│  │  │R1 │ │R2 │ ──▶  │ Continuous  │ ──▶ │  Prefill   │ ──▶ │ Stream  │    │  │
│  │  │R3 │ │R4 │      │  Batching   │     │  + Decode │     │ Tokens  │    │  │
│  │  └───┘ └───┘      └─────────────┘     └───────────┘     └─────────┘    │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  LEVEL 2: MODEL FORWARDING                                                   │
│  ══════════════════════════════════════════════════════════════════════════  │
│  Granularity: Layers, Operations                                             │
│  Timeline:    Milliseconds                                                   │
│  Resources:   GPU compute, KV cache                                          │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  Per-Batch Forward Pass (for batch_size=B, seq_len=S, hidden=H)        │  │
│  │                                                                        │  │
│  │  Embed ──▶ [Layer 0] ──▶ [Layer 1] ──▶ ... ──▶ [Layer N-1] ──▶ LMHead  │  │
│  │              │                                                         │  │
│  │              └─▶ Attn ─▶ FFN ─▶ Add (residual)                         │  │
│  │                    │       │                                           │  │
│  │                    │       └─▶ FC1 ─▶ SiLU ─▶ FC2 ─▶ AllReduce         │  │
│  │                    └─▶ QKV ─▶ RoPE ─▶ Flash ─▶ O ─▶ AllReduce          │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  LEVEL 3: KERNEL EXECUTION                                                   │
│  ══════════════════════════════════════════════════════════════════════════  │
│  Granularity: Tiles, Warps, Threads                                          │
│  Timeline:    Microseconds                                                   │
│  Resources:   SMs, Registers, Shared Memory, Tensor Cores                    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  Kernel Launch ──▶ SM Assignment ──▶ Warp Execution ──▶ Synchronization│  │
│  │                                                                        │  │
│  │  ┌───────────────────────────────────────────────────────────────────┐ │  │
│  │  │  SM 0        SM 1        SM 2        ...        SM 107            │ │  │
│  │  │  ┌─────┐     ┌─────┐     ┌─────┐                ┌─────┐           │ │  │
│  │  │  │Tile0│     │Tile1│     │Tile2│     ...        │TileN│           │ │  │
│  │  │  │Warp0│     │Warp0│     │Warp0│                │Warp0│           │ │  │
│  │  │  │Warp1│     │Warp1│     │Warp1│                │Warp1│           │ │  │
│  │  │  │ ... │     │ ... │     │ ... │                │ ... │           │ │  │
│  │  │  └─────┘     └─────┘     └─────┘                └─────┘           │ │  │
│  │  └───────────────────────────────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 The Nested Loop Structure

The key insight is that LLM serving consists of **nested loops** at different time scales:

```python
# ══════════════════════════════════════════════════════════════════════════════
# COMPLETE LLM SERVING: Three Nested Loops
# ══════════════════════════════════════════════════════════════════════════════

def llm_serving_main():
    """
    LEVEL 1: Serving Loop (runs indefinitely, seconds per iteration)
    - Receives requests from clients
    - Forms batches using continuous batching
    - Dispatches to GPU for execution
    """
    while server_running:
        # ──────────────────────────────────────────────────────────────────────
        # SCHEDULING PHASE (CPU): Form batch from pending requests
        # ──────────────────────────────────────────────────────────────────────
        new_requests = receive_requests(non_blocking=True)
        pending_queue.extend(new_requests)

        # Continuous batching: mix prefill and decode requests
        batch = scheduler.form_batch(
            pending_queue,
            max_batch_tokens=8192,      # Budget for this iteration
            max_batch_size=256          # Max concurrent requests
        )

        if batch.is_empty():
            continue

        # ──────────────────────────────────────────────────────────────────────
        # EXECUTION PHASE (GPU): Run one forward pass
        # ──────────────────────────────────────────────────────────────────────
        forward_one_step(batch)  # See LEVEL 2

        # ──────────────────────────────────────────────────────────────────────
        # POST-PROCESSING: Handle completed requests
        # ──────────────────────────────────────────────────────────────────────
        for req in batch.requests:
            if req.is_finished():  # Hit EOS or max_length
                send_response(req)
                free_kv_cache(req)
            else:
                pending_queue.append(req)  # Continue in next iteration


def forward_one_step(batch):
    """
    LEVEL 2: Model Forward Loop (one iteration, ~10-100ms)
    - Processes all tokens in batch through all layers
    - Generates one new token per request
    """
    # Separate prefill (variable length) and decode (single token) requests
    prefill_reqs = [r for r in batch if r.is_prefill]
    decode_reqs = [r for r in batch if r.is_decode]

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER LOOP: Process through all transformer layers
    # ──────────────────────────────────────────────────────────────────────────
    hidden = embedding(batch.token_ids)  # [total_tokens, hidden_dim]

    for layer_idx in range(num_layers):  # 32-80 iterations for typical models
        # Each layer: ~0.3-1ms for decode, ~10-50ms for prefill
        hidden = transformer_layer(hidden, layer_idx, batch)

    # ──────────────────────────────────────────────────────────────────────────
    # SAMPLING: Generate next token for each request
    # ──────────────────────────────────────────────────────────────────────────
    logits = lm_head(hidden)  # [total_tokens, vocab_size]

    # Only sample from last token of each sequence
    next_tokens = sample(logits[batch.last_token_indices])
    batch.append_tokens(next_tokens)


def transformer_layer(hidden, layer_idx, batch):
    """
    LEVEL 3: Layer Execution (one layer, ~0.3-50ms depending on phase)
    - Attention: memory-bound for decode, compute-bound for prefill
    - MLP: compute-bound for both phases
    """
    residual = hidden

    # ──────────────────────────────────────────────────────────────────────────
    # ATTENTION BLOCK
    # ──────────────────────────────────────────────────────────────────────────
    hidden = rms_norm(hidden)                              # ~20µs

    # QKV projection: [tokens, hidden] @ [hidden, 3*hidden] -> [tokens, 3*hidden]
    q, k, v = qkv_proj(hidden).split(3)                    # GEMM: ~50-200µs

    # Apply rotary position embedding
    q, k = apply_rope(q, k, batch.positions)               # ~10µs

    # Store K,V to cache (for decode reuse)
    kv_cache[layer_idx].append(k, v, batch.cache_indices)  # ~20µs

    # Attention computation
    # - Prefill: Q @ K^T @ V, compute-bound, O(seq_len²)
    # - Decode: Q @ K_cache^T @ V_cache, memory-bound, O(context_len)
    attn_out = flash_attention(                            # ~100-500µs
        q, kv_cache[layer_idx],
        batch.seq_lens, batch.cache_lens
    )

    # Output projection
    hidden = o_proj(attn_out) + residual                   # GEMM: ~50-200µs

    # ──────────────────────────────────────────────────────────────────────────
    # MLP BLOCK (compute-bound for both phases)
    # ──────────────────────────────────────────────────────────────────────────
    residual = hidden
    hidden = rms_norm(hidden)                              # ~20µs

    # Gate and Up projections: [tokens, hidden] -> [tokens, intermediate]
    gate = gate_proj(hidden)                               # GEMM: ~100-400µs
    up = up_proj(hidden)                                   # GEMM: ~100-400µs

    # Activation and Down projection
    hidden = silu(gate) * up                               # ~20µs
    hidden = down_proj(hidden) + residual                  # GEMM: ~100-400µs

    return hidden
```

### 1.3 Key Parameters

| Level | Parameter | Typical Value | Controls |
|-------|-----------|---------------|----------|
| Serving | `max_batch_size` | 256 | Concurrent requests |
| Serving | `prefill_budget` | 8192 | Tokens per scheduling round |
| Model | `num_layers` | 32-80 | Layer loop iterations |
| Model | `hidden_size` | 4096-8192 | GEMM dimensions |
| Model | `num_kv_heads` | 8 | KV cache size (GQA) |
| Kernel | `block_size` | 128 | Tile dimensions |
| Kernel | `num_warps` | 4-8 | Parallelism per SM |
| Kernel | `num_stages` | 2-4 | Software pipelining depth |

---

## 2. Timeline Decomposition

### 2.1 Complete Request Timeline

```
Time ────────────────────────────────────────────────────────────────────────────▶
      │◀───────────────────── Request Lifetime (seconds) ─────────────────────▶│

LEVEL 1: Infrastructure
┌─────┬──────────┬──────────────────────────────────────────────────────┬──────┐
│Queue│ Schedule │              GPU Execution                           │Stream│
│ Wait│          │                                                      │Output│
└─────┴──────────┴──────────────────────────────────────────────────────┴──────┘
                 │◀────────── Prefill ──────────▶│◀──── Decode Loop ───▶│

LEVEL 2: Model Forward (per iteration)
                 ┌────────────────────────────────────────────────────────┐
                 │ Iter 0 (Prefill)                                        │
                 │ ┌─────┐┌─────┐┌─────┐     ┌─────┐┌─────┐┌──────┐       │
                 │ │Emb  ││L0   ││L1   │ ... │LN-1 ││Norm ││LMHead│       │
                 │ └─────┘└─────┘└─────┘     └─────┘└─────┘└──────┘       │
                 └────────────────────────────────────────────────────────┘
                                              ┌────────────────────────────┐
                                              │ Iter 1+ (Decode)           │
                                              │ ┌───┐┌───┐   ┌───┐┌─────┐  │
                                              │ │Emb││L0 │...│LN ││Head │  │
                                              │ └───┘└───┘   └───┘└─────┘  │ 
                                              └────────────────────────────┘
                                                    (repeated until EOS)

LEVEL 3: Kernel Execution (per layer)
                       ┌────────────────────────────────────────┐
                       │ Layer N forward                        │
                       │                                        │
Kernel                 │  ┌──────┐  ┌──────┐  ┌──────┐          │
Launches:              │  │RMSNrm│  │ QKV  │  │ Attn │  ...     │
                       │  │ 20µs │  │200µs │  │500µs │          │
                       │  └──────┘  └──────┘  └──────┘          │
                       │          ▲         ▲         ▲         │
                       │          │         │         │         │
Sync Points:           │       barrier   barrier   barrier      │
                       └────────────────────────────────────────┘
```

### 2.2 Computational Complexity by Level

```
Per-Iteration Breakdown (batch_size=B, seq_len=S, hidden=H, layers=L):

┌────────────────────────────────────────────────────────────────────────────┐
│  OPERATION           │  PREFILL FLOPs         │  DECODE FLOPs              │
├────────────────────────────────────────────────────────────────────────────┤
│  Embedding           │  O(B × S × H)          │  O(B × H)                  │
│  Per-Layer:          │                        │                            │
│    RMSNorm           │  O(B × S × H)          │  O(B × H)                  │
│    QKV Projection    │  O(B × S × H × 3H)     │  O(B × H × 3H)             │
│    Attention         │  O(B × S² × H)  ◀─────── Memory-bound               │
│    O Projection      │  O(B × S × H²)         │  O(B × H²)                 │
│    MLP (FC1)         │  O(B × S × H × 4H)     │  O(B × H × 4H)             │
│    MLP (FC2)         │  O(B × S × 4H × H)     │  O(B × 4H × H)             │
│  LM Head             │  O(B × S × H × V)      │  O(B × H × V)              │
├────────────────────────────────────────────────────────────────────────────┤
│  TOTAL (L layers)    │  O(L × B × S × H²)     │  O(L × B × H²)             │
│                      │  Compute-bound         │  Memory-bound              │
└────────────────────────────────────────────────────────────────────────────┘

Where: H=4096, L=32, V=128000 (vocab)
```

---

## 3. SPMD Programming Model with MPMD Runtime Behavior

### 3.1 Understanding SPMD vs MPMD

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   PROGRAMMING MODEL COMPARISON                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  SPMD (Single Program, Multiple Data)                                    │
│  ════════════════════════════════════                                    │
│                                                                          │
│  • ALL threads execute the SAME program                                  │
│  • Differentiation via thread/block IDs                                  │
│  • NVIDIA CUDA's native programming model                                │
│                                                                          │
│  __global__ void kernel(float* data) {                                    │
│      int tid = blockIdx.x * blockDim.x + threadIdx.x;                    │
│      data[tid] = data[tid] * 2.0f;  // Same instruction, different data  │
│  }                                                                       │
│                                                                          │
│  Thread 0: data[0] = data[0] * 2.0f                                      │
│  Thread 1: data[1] = data[1] * 2.0f     All run SAME code                │
│  Thread 2: data[2] = data[2] * 2.0f                                      │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  MPMD (Multiple Program, Multiple Data)                                  │
│  ═════════════════════════════════════                                   │
│                                                                          │
│  • Different threads execute DIFFERENT programs                          │
│  • Natural for heterogeneous workloads                                   │
│  • CPUs naturally support this (different cores, different code)         │
│                                                                          │
│  // Conceptual (not native CUDA):                                        │
│  Processor 0: run(program_A, data_A)                                     │
│  Processor 1: run(program_B, data_B)    Different code paths             │
│  Processor 2: run(program_C, data_C)                                     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.2 CUDA: SPMD that Mimics MPMD

NVIDIA GPUs use SPMD but achieve MPMD-like behavior through **conditional branching** and **warp specialization**. The key is that the **same kernel code** runs on all SMs, but each SM/warp executes **different loop iterations** or **different branches**.

```python
# ══════════════════════════════════════════════════════════════════════════════
# WARP SPECIALIZATION: Producer-Consumer Pattern (Hopper TMA + Tensor Core)
# ══════════════════════════════════════════════════════════════════════════════

def warp_specialized_gemm_kernel():
    """
    Single kernel with specialized warps running concurrent loops.
    All warps start with same code but branch to different execution paths.
    """
    warp_id = threadIdx.x // 32
    NUM_STAGES = 3  # Pipeline depth

    if warp_id == 0:
        # ══════════════════════════════════════════════════════════════════════
        # PRODUCER WARP: TMA data loading loop
        # ══════════════════════════════════════════════════════════════════════
        for tile_idx in range(num_tiles):
            # Wait for buffer slot to be free
            buffer_slot = tile_idx % NUM_STAGES
            wait_for_buffer_free(buffer_slot)

            # Issue async TMA copy: Global Memory -> Shared Memory
            tma_load_async(
                dst=shared_mem[buffer_slot],
                src=global_A[tile_idx],
                size=TILE_SIZE
            )

            # Signal that data is ready
            signal_tile_loaded(buffer_slot)

    else:
        # ══════════════════════════════════════════════════════════════════════
        # CONSUMER WARPS: Tensor Core compute loop
        # ══════════════════════════════════════════════════════════════════════
        accumulator = zeros(TILE_M, TILE_N)

        for tile_idx in range(num_tiles):
            buffer_slot = tile_idx % NUM_STAGES

            # Wait for producer to load this tile
            wait_for_tile_loaded(buffer_slot)

            # Execute WMMA (Tensor Core) on shared memory data
            a_frag = load_matrix_sync(shared_mem[buffer_slot].A)
            b_frag = load_matrix_sync(shared_mem[buffer_slot].B)
            accumulator = mma_sync(a_frag, b_frag, accumulator)

            # Signal buffer is free for reuse
            signal_buffer_free(buffer_slot)

        # Store result
        store_matrix_sync(global_C, accumulator)


# ══════════════════════════════════════════════════════════════════════════════
# MEGAKERNEL: Task-Based Scheduling Loop (triton-distributed style)
# ══════════════════════════════════════════════════════════════════════════════

def megakernel_main(task_queues, scoreboard, num_layers):
    """
    Persistent kernel: launched once, runs entire inference.
    Each SM runs the same loop but pulls different tasks from its queue.
    """
    sm_id = get_sm_id()
    my_queue = task_queues[sm_id]

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN TASK LOOP: Runs until all layers complete
    # ══════════════════════════════════════════════════════════════════════════
    while True:
        # Dequeue next task for this SM
        task = my_queue.pop()  # Blocking if queue empty

        if task.type == DONE:
            break

        # ──────────────────────────────────────────────────────────────────────
        # DEPENDENCY WAIT LOOP: Spin until all inputs ready
        # ──────────────────────────────────────────────────────────────────────
        for dep in task.dependencies:
            # dep = (layer_idx, task_type, tile_idx)
            while scoreboard[dep.layer][dep.task][dep.tile] == 0:
                # Busy wait - other SMs are producing these outputs
                __nanosleep(100)

        # ──────────────────────────────────────────────────────────────────────
        # TASK EXECUTION: Different task types = MPMD behavior
        # ──────────────────────────────────────────────────────────────────────
        if task.type == RMSNORM:
            rms_norm_tile(task.input_ptr, task.output_ptr, task.tile_bounds)

        elif task.type == QKV_GEMM:
            gemm_tile(task.A_ptr, task.B_ptr, task.C_ptr, task.tile_bounds)

        elif task.type == ATTENTION:
            flash_attention_tile(task.Q, task.K, task.V, task.O, task.tile_bounds)

        elif task.type == MLP_GEMM:
            gemm_tile(task.A_ptr, task.B_ptr, task.C_ptr, task.tile_bounds)

        elif task.type == ALLREDUCE:
            # Inter-GPU communication
            nccl_allreduce_tile(task.buffer, task.tile_bounds)

        # ──────────────────────────────────────────────────────────────────────
        # SIGNAL COMPLETION: Update scoreboard for dependent tasks
        # ──────────────────────────────────────────────────────────────────────
        atomicExch(scoreboard[task.layer][task.type][task.tile], 1)
```

### 3.3 Warp Specialization Patterns

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   WARP SPECIALIZATION TECHNIQUES                         │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Producer-Consumer Pattern (Hopper TMA + Tensor Core)                 │
│  ═══════════════════════════════════════════════════════                 │
│                                                                          │
│  ┌─────────────────┐    Shared Memory    ┌──────────────────┐            │
│  │   TMA Warp      │  ────────────────▶  │  Tensor Core     │            │
│  │   (Producer)    │    Buffer[PIPE]     │  Warps           │            │
│  │                 │  ◀────────────────  │  (Consumer)      │            │
│  └─────────────────┘   Release signal    └──────────────────┘            │
│                                                                          │
│  Synchronization via named barriers:                                     │
│    - signal_tile_loaded()  / wait_for_tile_loaded()                      │
│    - signal_tile_released() / wait_for_tile_released()                   │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  2. Worker-Scheduler Pattern (Megakernel)                                │
│  ════════════════════════════════════════                                │
│                                                                          │
│  SM 0:                                                                   │
│  ┌────────────────────────────────────────────────────────────┐          │
│  │  ┌───────────────┐     ┌───────────────┐                   │          │
│  │  │ Scheduler     │────▶│ Worker Warps  │                   │          │
│  │  │ Warps (×4)    │     │ (×remaining)  │                   │          │
│  │  │               │◀────│               │                   │          │
│  │  │ • Task queue  │     │ • Execute     │                   │          │
│  │  │ • Dependency  │     │   tasks       │                   │          │
│  │  │   tracking    │     │ • Signal done │                   │          │
│  │  └───────────────┘     └───────────────┘                   │          │
│  └────────────────────────────────────────────────────────────┘          │
│                                                                          │
│  Tasks flow: Global Queue ──▶ SM Queue ──▶ Worker ──▶ Scoreboard         │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  3. Resource Partitioning Pattern (NanoFlow)                             │
│  ═══════════════════════════════════════════                             │
│                                                                          │
│  108 SMs partitioned by category:                                        │
│                                                                          │
│  ┌──────────────────┬──────────────────┬──────────────────┐              │
│  │   COMPUTE SMs    │   MEMORY SMs     │   NETWORK SMs    │              │
│  │   (72 SMs)       │   (36 SMs)       │   (35 SMs)       │              │
│  │                  │                  │                  │              │
│  │   Dense GEMMs    │   Decode Attn    │   AllReduce      │              │
│  │   - KQV          │   (GEMV)         │   AllGather      │              │
│  │   - O Proj       │                  │                  │              │
│  │   - FC1, FC2     │                  │                  │              │
│  └──────────────────┴──────────────────┴──────────────────┘              │
│                                                                          │
│  SM allocation via: cublas.set_sm_count_target(sm_count)                 │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.4 How the Megakernel Achieves MPMD

```
┌──────────────────────────────────────────────────────────────────────────┐
│             MEGAKERNEL: SPMD CODE IMPLEMENTING MPMD RUNTIME              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  SPMD Program (same code loaded on all SMs):                             │
│  ════════════════════════════════════════════                            │
│                                                                          │
│  __global__ void mega_triton_kernel(WorkQueue* queues, Scoreboard* sb):  │
│      sm_id = get_sm_id()                                                 │
│      my_queue = queues[sm_id]                                            │
│                                                                          │
│      while not done:                                                     │
│          # 1. Dequeue task (different task per SM!)                      │
│          task = my_queue.pop()                                           │
│                                                                          │
│          # 2. Wait for dependencies via scoreboard                       │
│          for dep in task.dependencies:                                   │
│              while scoreboard[dep.layer, dep.task, dep.tile] == 0:       │
│                  spin()  # Busy wait                                     │
│                                                                          │
│          # 3. Execute task (MPMD: different task types!)                 │
│          switch task.type:                                               │
│              case RMSNORM:                                               │
│                  rms_norm_kernel(task.tile, task.io)                     │
│              case QKV_PROJ:                                              │
│                  gemm_kernel(task.tile, task.io)                         │
│              case ATTENTION:                                             │
│                  flash_attn_kernel(task.tile, task.io)                   │
│              case MLP_FC1:                                               │
│                  gemm_kernel(task.tile, task.io)                         │
│              ...                                                         │
│                                                                          │
│          # 4. Signal completion                                          │
│          scoreboard[task.layer, task.id, task.tile] = 1                  │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  RUNTIME BEHAVIOR (MPMD-like):                                           │
│  ═════════════════════════════                                           │
│                                                                          │
│  Time ───────────────────────────────────────────────────────────────▶   │
│                                                                          │
│  SM 0: │RMSNorm│ QKV_t0 │ Attn_t0 │  O_t0  │ MLP_t0 │ ...              │ │
│  SM 1: │RMSNorm│ QKV_t1 │ Attn_t1 │  O_t1  │ MLP_t1 │ ...              │ │
│  SM 2: │RMSNorm│ QKV_t2 │  wait.. │ Attn_t2   │ O_t2  │ ...            │ │
│  SM 3: │RMSNorm│ QKV_t3 │ Attn_t3 │  wait..   │ O_t3  │ ...            │ │
│                                                                          │
│  Same kernel code, but each SM executes different task sequence!         │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Intra-Kernel Design Space: Software Pipelining and Warp Specialization

At the kernel level, **software pipelining (SWP)** and **warp specialization (WS)** are the two key techniques for maximizing hardware utilization. These techniques determine how work is distributed across warps and how memory/compute operations overlap within a single kernel.

### 4.1 Software Pipelining: Hiding Latency

Software pipelining overlaps multiple loop iterations to hide memory latency. The key insight is that while one iteration waits for data, another can be computing.

```python
# ══════════════════════════════════════════════════════════════════════════════
# SOFTWARE PIPELINING: Multi-Stage Buffering
# ══════════════════════════════════════════════════════════════════════════════

def gemm_software_pipelined(A, B, C, num_stages=3):
    """
    Software pipelining with circular buffer.
    Each stage is in a different phase: loading, computing, or storing.
    """
    # Allocate circular buffer in shared memory
    smem_A = allocate_shared(num_stages, TILE_M, TILE_K)
    smem_B = allocate_shared(num_stages, TILE_K, TILE_N)
    accumulator = zeros(TILE_M, TILE_N)

    # ──────────────────────────────────────────────────────────────────────────
    # PROLOGUE: Fill the pipeline (load first num_stages-1 tiles)
    # ──────────────────────────────────────────────────────────────────────────
    for s in range(num_stages - 1):
        async_load(smem_A[s], A[s * TILE_K])
        async_load(smem_B[s], B[s * TILE_K])
        commit_group()  # Mark this load as a group

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN LOOP: Steady-state pipeline (overlapped load + compute)
    # ──────────────────────────────────────────────────────────────────────────
    for tile_idx in range(num_tiles):
        # Current buffer slot (circular)
        curr_slot = tile_idx % num_stages
        next_slot = (tile_idx + num_stages - 1) % num_stages

        # WAIT: Ensure current tile is loaded
        wait_group(num_stages - 1)  # Wait for oldest in-flight load

        # COMPUTE: Matrix multiply on current tile (while next loads)
        accumulator = mma_sync(smem_A[curr_slot], smem_B[curr_slot], accumulator)

        # LOAD: Start loading future tile (if any remain)
        if tile_idx + num_stages - 1 < num_tiles:
            async_load(smem_A[next_slot], A[(tile_idx + num_stages - 1) * TILE_K])
            async_load(smem_B[next_slot], B[(tile_idx + num_stages - 1) * TILE_K])
            commit_group()

    # ──────────────────────────────────────────────────────────────────────────
    # EPILOGUE: Drain remaining compute
    # ──────────────────────────────────────────────────────────────────────────
    store(C, accumulator)
```

### 4.2 Warp Specialization: Role-Based Execution

Warp specialization assigns different roles to different warps within the same thread block, enabling true concurrent execution of heterogeneous tasks.

```python
# ══════════════════════════════════════════════════════════════════════════════
# WARP SPECIALIZATION: Producer-Consumer with Multiple Roles
# ══════════════════════════════════════════════════════════════════════════════

def flash_attention_warp_specialized():
    """
    Flash Attention with warp specialization on Hopper.
    Different warps perform different operations concurrently.
    """
    warp_id = get_warp_id()
    PRODUCER_WARPS = 1   # TMA load
    CONSUMER_WARPS = 7   # Tensor Core compute
    NUM_STAGES = 2       # Double buffering

    if warp_id < PRODUCER_WARPS:
        # ══════════════════════════════════════════════════════════════════════
        # PRODUCER WARP: Asynchronous data movement via TMA
        # ══════════════════════════════════════════════════════════════════════
        for kv_block in range(num_kv_blocks):
            buffer_slot = kv_block % NUM_STAGES

            # Wait for consumers to release this buffer
            wait_for_buffer_released(buffer_slot)

            # Issue TMA loads (asynchronous, hardware-managed)
            tma_load_K(smem_K[buffer_slot], K[kv_block])
            tma_load_V(smem_V[buffer_slot], V[kv_block])

            # Signal buffer is ready for consumption
            arrive_and_expect_tx(buffer_slot, expected_bytes)

    else:
        # ══════════════════════════════════════════════════════════════════════
        # CONSUMER WARPS: Tensor Core computation
        # ══════════════════════════════════════════════════════════════════════
        O_accumulator = zeros(TILE_Q, HEAD_DIM)
        m_prev = -infinity
        l_prev = 0.0

        for kv_block in range(num_kv_blocks):
            buffer_slot = kv_block % NUM_STAGES

            # Wait for producer to fill this buffer
            wait_for_buffer_filled(buffer_slot)

            # Compute attention scores: S = Q @ K^T
            S = wgmma(Q_smem, smem_K[buffer_slot].T)

            # Online softmax update
            m_curr = max(m_prev, row_max(S))
            P = exp(S - m_curr)
            l_curr = exp(m_prev - m_curr) * l_prev + row_sum(P)

            # Rescale and accumulate: O = rescale(O_prev) + P @ V
            O_accumulator = exp(m_prev - m_curr) * O_accumulator
            O_accumulator += wgmma(P, smem_V[buffer_slot])

            m_prev, l_prev = m_curr, l_curr

            # Release buffer for producer reuse
            signal_buffer_released(buffer_slot)

        # Final normalization
        O_accumulator /= l_prev
        store(O_global, O_accumulator)
```

### 4.3 Combined Design Space

```
┌──────────────────────────────────────────────────────────────────────────┐
│           SOFTWARE PIPELINING + WARP SPECIALIZATION DESIGN SPACE         │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  DIMENSION 1: Pipeline Depth (num_stages)                          │  │
│  ├────────────────────────────────────────────────────────────────────┤  │
│  │                                                                    │  │
│  │   Stages │ SMEM Usage  │ Latency Hiding │ Register Pressure        │  │
│  │  ════════│═════════════│════════════════│═══════════════════════   │  │
│  │    1     │ Low         │ None           │ Low (baseline)           │  │
│  │    2     │ 2× baseline │ Partial        │ Moderate                 │  │
│  │    3     │ 3× baseline │ Full (typical) │ High                     │  │
│  │    4+    │ 4×+ baseline│ Diminishing    │ May spill to local mem   │  │
│  │                                                                    │  │
│  │  Tradeoff: More stages = better latency hiding, but more SMEM      │  │
│  │                                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  DIMENSION 2: Warp Specialization Ratio                            │  │
│  ├────────────────────────────────────────────────────────────────────┤  │
│  │                                                                    │  │
│  │   Config           │ Best For             │ Hardware Match         │  │
│  │  ══════════════════│══════════════════════│═══════════════════════ │  │
│  │  No WS (all same)  │ Uniform workloads    │ Pre-Hopper GPUs        │  │
│  │  1 TMA : 7 TC      │ Compute-bound GEMM   │ Hopper (typical)       │  │
│  │  2 TMA : 6 TC      │ Memory-bound kernels │ Large tile sizes       │  │
│  │  1 TMA : 1 Reduce  │ Fused ops + reduce   │ Custom patterns        │  │
│  │   : N TC           │                      │                        │  │
│  │                                                                    │  │
│  │  Tradeoff: Producer/consumer ratio must match memory/compute      │  │
│  │            bandwidth to avoid stalls                              │  │
│  │                                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  DIMENSION 3: Synchronization Mechanism                            │  │
│  ├────────────────────────────────────────────────────────────────────┤  │
│  │                                                                    │  │
│  │   Mechanism         │ Granularity  │ Overhead  │ Use Case          │  │
│  │  ═══════════════════│══════════════│═══════════│════════════════   │  │
│  │  __syncthreads()    │ Block-wide   │ High      │ All warps sync    │  │
│  │  Named barriers     │ Warp subsets │ Low       │ Producer-consumer │  │
│  │  Arrive/wait        │ Tile-level   │ Very low  │ Fine-grained pipe │  │
│  │  TMA async barriers │ Per-transfer │ Minimal   │ Hopper TMA        │  │
│  │                                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  DIMENSION 4: Memory Hierarchy Usage                               │  │
│  ├────────────────────────────────────────────────────────────────────┤  │
│  │                                                                    │  │
│  │   Pattern                │ SMEM      │ Registers │ Occupancy       │  │
│  │  ════════════════════════│═══════════│═══════════│═══════════════  │  │
│  │  Register-heavy          │ Minimal   │ 128-255   │ Low (1-2 waves) │  │
│  │  SMEM-heavy              │ 48-100KB  │ 64-96     │ Medium          │  │
│  │  Balanced                │ 32-48KB   │ 96-128    │ High (3+ waves) │  │
│  │  Persistent (megakernel) │ Varies    │ Moderate  │ 1 block/SM      │  │
│  │                                                                    │  │
│  │  Hopper enables: 227KB SMEM + 256 regs/thread for register-rich   │  │
│  │  algorithms like Flash Attention                                   │  │
│  │                                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Optimal Schedule Search (Twill)

Finding the optimal combination of SWP and WS is a challenging optimization problem. The **Twill** system (Soi et al., 2025) treats this as a constraint satisfaction problem:

```python
# ══════════════════════════════════════════════════════════════════════════════
# TWILL: Constraint-Based Schedule Optimization
# ══════════════════════════════════════════════════════════════════════════════

def twill_find_optimal_schedule(loop_body, hardware_constraints):
    """
    Twill formulates SWP + WS as a constraint satisfaction problem.
    Finds optimal schedule without heuristics.
    """
    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1: Extract loop structure and dependencies
    # ──────────────────────────────────────────────────────────────────────────
    operations = extract_operations(loop_body)
    dependencies = build_dependency_graph(operations)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2: Define decision variables
    # ──────────────────────────────────────────────────────────────────────────
    solver = ConstraintSolver()

    # For each operation: which stage? which warp role?
    for op in operations:
        op.stage = solver.IntVar(0, max_stages - 1)
        op.warp_role = solver.IntVar(0, num_warp_roles - 1)
        op.issue_cycle = solver.IntVar(0, max_cycles)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3: Add hardware constraints
    # ──────────────────────────────────────────────────────────────────────────
    # Dependency constraints: producer before consumer
    for (src, dst) in dependencies:
        solver.Add(src.issue_cycle + src.latency <= dst.issue_cycle)

    # Resource constraints: limited functional units per cycle
    for cycle in range(max_cycles):
        solver.Add(count_tensor_core_ops(cycle) <= num_tensor_cores)
        solver.Add(count_tma_ops(cycle) <= num_tma_units)

    # Shared memory constraints
    solver.Add(total_smem_usage() <= hardware_constraints.smem_size)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 4: Optimize for throughput
    # ──────────────────────────────────────────────────────────────────────────
    solver.Minimize(pipeline_initiation_interval())

    schedule = solver.Solve()
    return schedule
```

**Key insight**: Twill automatically rediscovered expert-crafted schedules for Flash Attention on Hopper and Blackwell, validating that constraint-based optimization can match human expertise.

### 4.5 Visualizing SWP + WS

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                  SOFTWARE PIPELINING + WARP SPECIALIZATION                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  WITHOUT OPTIMIZATION (Sequential execution):                                │
│  ════════════════════════════════════════════                                │
│                                                                              │
│  All warps: │ LOAD A₀ │ LOAD B₀ │ wait... │ MMA C₀ │ LOAD A₁ │ LOAD B₁ │... │
│                                                                              │
│             │◀── Memory latency exposed ──▶│                                │
│             TMA idle ─────────────────────▶ TC idle ◀─────────────────────── │
│                                                                              │
│                                                                              │
│  WITH SWP + WS (Overlapped, specialized):                                    │
│  ════════════════════════════════════════                                    │
│                                                                              │
│              Prologue      Steady State (3 iterations in flight)             │
│             ┌───────┐  ┌────────────────────────────────────────────┐        │
│             │       │  │                                            │        │
│  Producer   │LOAD A₀│  │LOAD A₂│LOAD A₃│LOAD A₄│LOAD A₅│  ...      │        │
│  Warp 0     │LOAD B₀│  │LOAD B₂│LOAD B₃│LOAD B₄│LOAD B₅│           │        │
│  (TMA)      │LOAD A₁│  │       │       │       │       │           │        │
│             │LOAD B₁│  │signal │signal │signal │signal │           │        │
│             │       │  │   │   │   │   │   │   │   │   │           │        │
│             │       │  │   ▼   │   ▼   │   ▼   │   ▼   │           │        │
│  Consumer   │       │  │ MMA   │ MMA   │ MMA   │ MMA   │  ...      │        │
│  Warps 1-7  │ wait  │  │ C₀    │ C₁    │ C₂    │ C₃    │           │        │
│  (TC)       │       │  │       │       │       │       │           │        │
│             └───────┘  └────────────────────────────────────────────┘        │
│                                                                              │
│  Timeline ──────────────────────────────────────────────────────────────────▶│
│                        │◀─── TMA and TC run concurrently ───▶│               │
│                                                                              │
│                                                                              │
│  SCHEDULE DECISION SPACE (per operation):                                    │
│  ═════════════════════════════════════════                                   │
│                                                                              │
│    op.stage ────▶ {0, 1, 2}           Which iteration offset?               │
│    op.role  ────▶ {PRODUCER, CONSUMER} Which warp group?                    │
│    op.cycle ────▶ {0, 1, 2, ...}       When to issue?                       │
│                                                                              │
│  CONSTRAINTS:                                                                │
│    • Dependencies:  load.cycle + latency ≤ compute.cycle                    │
│    • Resources:     TMA ops/cycle ≤ 1, TC ops/cycle ≤ 4                     │
│    • Memory:        buffers × stages ≤ SMEM capacity                        │
│                                                                              │
│  OBJECTIVE: Minimize Initiation Interval (II)                                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Cross-Level Optimization Design Space

### 5.1 Where MPMD-like Execution Helps

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   MPMD OPTIMIZATION OPPORTUNITIES                        │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  LEVEL        │ TECHNIQUE          │ HOW MPMD HELPS                      │
│ ══════════════│════════════════════│═════════════════════════════════════│
│               │                    │                                      │
│ INFRA         │ Overlap Scheduling │ CPU prepares batch N+1 while GPU    │
│ (mini-sglang) │ (CPU-GPU pipeline) │ executes batch N. Different         │
│               │                    │ "programs" run on different HW.     │
│               │                    │                                      │
│ ──────────────│────────────────────│──────────────────────────────────────│
│               │                    │                                      │
│ MODEL         │ Operation-level    │ GEMM, Attention, AllReduce run      │
│ (NanoFlow)    │ Pipeline Parallel  │ concurrently on partitioned SMs.    │
│               │                    │ Each category = different "program".│
│               │                    │                                      │
│ ──────────────│────────────────────│──────────────────────────────────────│
│               │                    │                                      │
│ KERNEL        │ Warp Specialization│ Producer warps (TMA) and consumer   │
│ (Megakernel)  │                    │ warps (Tensor Core) execute         │
│               │                    │ different code paths in kernel.     │
│               │                    │                                      │
│ ──────────────│────────────────────│──────────────────────────────────────│
│               │                    │                                      │
│ KERNEL        │ Megakernel Task    │ Each SM pulls different tasks from  │
│ (triton-dist) │ Scheduling         │ queue. Same code, MPMD behavior.    │
│               │                    │                                      │
└──────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Static vs Dynamic Scheduling Tradeoffs

```
┌──────────────────────────────────────────────────────────────────────────┐
│             STATIC vs DYNAMIC MPMD SCHEDULING                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                    STATIC SCHEDULING                               │  │
│  │                    (Compile-time decisions)                        │  │
│  ├────────────────────────────────────────────────────────────────────┤  │
│  │                                                                    │  │
│  │  Examples:                                                         │  │
│  │    - NanoFlow's Gurobi ILP optimization                           │  │
│  │    - CUDA Graph capture/replay                                     │  │
│  │    - Pre-computed SM allocation                                    │  │
│  │                                                                    │  │
│  │  Pros:                            Cons:                            │  │
│  │    ✓ Zero runtime overhead        ✗ Requires offline profiling    │  │
│  │    ✓ Predictable execution        ✗ Cannot adapt to variable load │  │
│  │    ✓ Optimal for fixed workloads  ✗ Fixed batch sizes only        │  │
│  │    ✓ CUDA graph compatible        ✗ Long setup time               │  │
│  │                                                                    │  │
│  │  Best for: Decode phase (fixed batch, predictable)                │  │
│  │                                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                    DYNAMIC SCHEDULING                              │  │
│  │                    (Runtime decisions)                             │  │
│  ├────────────────────────────────────────────────────────────────────┤  │
│  │                                                                    │  │
│  │  Examples:                                                         │  │
│  │    - Megakernel scoreboard-based task dispatch                    │  │
│  │    - Continuous batching in SGLang/vLLM                           │  │
│  │    - RadixAttention prefix matching                                │  │
│  │                                                                    │  │
│  │  Pros:                            Cons:                            │  │
│  │    ✓ Adapts to variable workloads ✗ Runtime scheduling overhead   │  │
│  │    ✓ Load balancing across SMs    ✗ Harder to optimize            │  │
│  │    ✓ Handles variable seq lengths ✗ Less predictable latency      │  │
│  │    ✓ No offline profiling needed  ✗ Memory for task queues        │  │
│  │                                                                    │  │
│  │  Best for: Prefill phase (variable length, unpredictable)         │  │
│  │                                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                    HYBRID SCHEDULING                               │  │
│  │                    (Best of both)                                  │  │
│  ├────────────────────────────────────────────────────────────────────┤  │
│  │                                                                    │  │
│  │  Examples:                                                         │  │
│  │    - MPK's JIT + AOT task dispatch                                │  │
│  │    - Static graphs with dynamic batch selection                   │  │
│  │    - Chunked prefill with CUDA graph decode                       │  │
│  │                                                                    │  │
│  │  Strategy:                                                         │  │
│  │    1. Pre-compute schedules for common batch sizes                │  │
│  │    2. Dynamic selection at runtime                                │  │
│  │    3. Fallback to dynamic for edge cases                          │  │
│  │                                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Complete Design Space Matrix

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   LLM SERVING OPTIMIZATION DESIGN SPACE                  │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Dimension         │ Options                   │ Systems                 │
│ ═══════════════════│═══════════════════════════│═════════════════════════│
│                    │                           │                         │
│ BATCHING           │ Static batching           │ (baseline)              │
│                    │ Continuous batching       │ vLLM, SGLang            │
│                    │ Chunked prefill           │ SGLang, mini-sglang     │
│                    │ Nano-batching             │ NanoFlow                │
│                    │                           │                         │
│ ───────────────────│───────────────────────────│─────────────────────────│
│                    │                           │                         │
│ KV CACHE           │ Pre-allocated             │ (baseline)              │
│                    │ PagedAttention            │ vLLM                    │
│                    │ RadixAttention            │ SGLang, mini-sglang     │
│                    │                           │                         │
│ ───────────────────│───────────────────────────│─────────────────────────│
│                    │                           │                         │
│ KERNEL LAUNCH      │ Many small kernels        │ vLLM, SGLang            │
│                    │ Fused operations          │ FlashAttention          │
│                    │ CUDA Graphs               │ SGLang decode           │
│                    │ Megakernel (persistent)   │ triton-distributed, MPK │
│                    │                           │                         │
│ ───────────────────│───────────────────────────│─────────────────────────│
│                    │                           │                         │
│ RESOURCE MGMT      │ Sequential kernels        │ (baseline)              │
│                    │ Multi-stream overlap      │ SGLang overlap sched    │
│                    │ SM partitioning           │ NanoFlow                │
│                    │ In-kernel task scheduling │ Megakernel              │
│                    │                           │                         │
│ ───────────────────│───────────────────────────│─────────────────────────│
│                    │                           │                         │
│ PARALLELISM        │ Data parallel (DP)        │ All                     │
│                    │ Tensor parallel (TP)      │ All multi-GPU           │
│                    │ Pipeline parallel (PP)    │ Large-scale             │
│                    │ Intra-device parallel     │ NanoFlow                │
│                    │                           │                         │
│ ───────────────────│───────────────────────────│─────────────────────────│
│                    │                           │                         │
│ SCHEDULING         │ Static (offline ILP)      │ NanoFlow                │
│                    │ Dynamic (scoreboard)      │ Megakernel              │
│                    │ Hybrid (JIT + AOT)        │ MPK                     │
│                    │                           │                         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Synthesis: Putting It All Together

### 6.1 Timeline with All Optimizations

```
Time ─────────────────────────────────────────────────────────────────────────▶

TRADITIONAL (Sequential):
┌─────────────────────────────────────────────────────────────────────────────┐
│ Recv │ Schedule │ K1 │ sync │ K2 │ sync │ K3 │ sync │ ... │ Sample │ Send   │
│      │          │    │      │    │      │    │      │     │        │        │
│ CPU  │   CPU    │GPU │      │GPU │      │GPU │      │     │  GPU   │ CPU    │
└─────────────────────────────────────────────────────────────────────────────┘
                  │◀─────────── GPU idle during CPU work ──────────────▶│


OPTIMIZED (mini-sglang Overlap):
┌─────────────────────────────────────────────────────────────────────────────┐
│ CPU: │ Process N-1 │ Recv N+1 │ Schedule N+1 │ Prepare N+1 │ Process N │    │
│      │             │          │              │             │           │    │
│ GPU: │             │◀──── Forward Batch N ─────▶ │◀── Sample ──▶│      │    │
│      │             │ K1 │ K2 │ K3 │ ... │ LMHead │              │      │    │
└─────────────────────────────────────────────────────────────────────────────┘
                     │◀─────── CPU/GPU overlap ────────▶│


OPTIMIZED (NanoFlow Intra-device):
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stream COMP: │ KQV_0 │ KQV_1 │   O_0   │ O_1 │ FC1_0 │ FC1_1 │ FC2 │ ...    │
│ Stream MEM:  │       │ DecAttn (overlapped)  │       │       │     │        │
│ Stream NET:  │             │ AllReduce (overlapped)  │       │              │
└─────────────────────────────────────────────────────────────────────────────┘
               │◀────── Three resource types run in parallel ──────▶│


OPTIMIZED (Megakernel):
┌──────────────────────────────────────────────────────────────────────────┐
│ Single Kernel Launch:                                                    │
│ SM 0: │Norm│QKV_0│Attn_0│ O_0 │FC1_0│Act_0│FC2_0│Add│Norm│...│           │
│ SM 1: │Norm│QKV_1│Attn_1│ O_1 │FC1_1│Act_1│FC2_1│Add│Norm│...│           │
│ SM 2: │Norm│QKV_2│Attn_2│ O_2 │FC1_2│Act_2│FC2_2│Add│Norm│...│           │
│       │◀──────────────── No kernel launch overhead ─────────▶│           │
└──────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Key Insights

1. **SPMD Enables MPMD**: CUDA's SPMD model achieves MPMD-like behavior through:
   - Conditional branching (warp_id checks)
   - Dynamic task queues (each SM gets different tasks)
   - Resource partitioning (different SMs assigned different roles)

2. **MPMD at Multiple Levels**:
   - **Infra**: CPU scheduler + GPU executor (different programs)
   - **Model**: Overlap scheduling with multiple CUDA streams
   - **Kernel**: Warp specialization, SM partitioning, task-based execution

3. **Static vs Dynamic Trade-off**:
   - Static: Better for predictable workloads (decode phase)
   - Dynamic: Better for variable workloads (prefill, variable lengths)
   - Hybrid: Combines benefits (pre-computed schedules + runtime selection)

4. **Optimization Target Determines Approach**:
   - **Throughput**: Maximize GPU utilization via batching + overlap
   - **Latency**: Minimize kernel launch overhead via megakernels
   - **Both**: Careful scheduling across all levels

---

## 7. References

### Academic Papers
- [NanoFlow (arXiv:2408.12757)](https://arxiv.org/abs/2408.12757) - Intra-device parallelism
- [Mirage Persistent Kernel (arXiv:2512.22219)](https://arxiv.org/html/2512.22219) - Megakernel compiler
- [Twill (arXiv:2512.18134)](https://arxiv.org/abs/2512.18134) - Optimal SWP + WS schedule search
- [CudaDMA](https://lightsighter.org/pdfs/cudadma-sc11.pdf) - Warp specialization origins
- [Singe](https://cs.stanford.edu/~sjt/pubs/ppopp14.pdf) - Warp specialization for performance

### Technical Resources
- [PyTorch Warp Specialization Blog](https://pytorch.org/blog/warp-specialization/)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Unweaving Warp Specialization](https://rohany.github.io/blog/warp-specialization/)
- [NVIDIA Developer Forums - SPMD/SIMT](https://forums.developer.nvidia.com/t/simt-simd-spmd/16931)

### Codebase Analyses (in this repo)
- `reports/implementations/nanoflow-analysis.md`
- `reports/implementations/triton-distributed-megakernel-analysis.md`
- `reports/implementations/mini-sglang-analysis.md`

---

## Tags

`#mpmd` `#spmd` `#warp-specialization` `#software-pipelining` `#megakernel` `#llm-serving` `#cuda` `#sm-partitioning` `#persistent-kernel`
