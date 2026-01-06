---
name: llm-code-analysis
description: Deep-dive analysis of LLM/ML model implementations and inference infrastructure in PyTorch, JAX, Triton, CUDA, or ML compiler frameworks. Covers serving systems (vLLM, SGLang, TensorRT-LLM), KV cache management, request scheduling, and batching strategies. Generates visual explanations with ASCII diagrams, pseudocode abstractions, and technical tutorials. Use when analyzing LLM implementations, inference engines, serving systems, ML compilers, megakernels, or distributed execution.
---

# LLM Implementation & Infrastructure Analysis Skill

Deep-dive analysis of LLM/ML model implementations and inference serving infrastructure. Covers model architectures (transformers, attention, MLP), ML compilers (Triton, CUDA, XLA), and serving systems (vLLM, SGLang, TensorRT-LLM). Generates intuitive visual explanations, pseudocode abstractions, and technical tutorials.

## When to Use

This skill auto-triggers when:
- Analyzing LLM model implementations (transformers, attention, MLP, etc.)
- Understanding ML compiler code (Triton, CUDA kernels, XLA)
- Analyzing LLM inference serving systems (vLLM, SGLang, TensorRT-LLM)
- Understanding KV cache management and PagedAttention
- Reverse-engineering training/inference pipelines
- Tracing data flow through model architectures
- Understanding megakernel or fused operator implementations
- Analyzing request scheduling and batching strategies
- Understanding distributed/parallel execution strategies

## Analysis Framework

### Phase 1: Architectural Reconnaissance

**Objective:** Map the high-level structure before diving into implementation details.

1. **System Topology Discovery**
   - Trace execution path from Python entry point to hardware kernels
   - Identify the orchestration pattern: static graph, dynamic dispatch, persistent kernel, megakernel
   - Map the module hierarchy: Model → Layers → Operators → Kernels

2. **Atomic Units of Work**
   - What constitutes a single task/tile? (e.g., attention head, token block, tensor shard)
   - How are tasks scheduled? (round-robin, dependency-driven, work-stealing)
   - What are the parallelization dimensions? (batch, sequence, heads, hidden)

3. **Configuration Extraction**
   - Identify configuration classes and hyperparameters
   - Extract architectural constants: `head_dim`, `num_heads`, `hidden_size`, `vocab_size`
   - Find constraint assertions: alignment requirements, max sequence lengths, power-of-2 restrictions

### Phase 2: Component Analysis

**For each major component (Attention, MLP, Norm, etc.):**

1. **Interface Analysis**
   ```
   INPUTS:  [tensor_name: shape, dtype, memory_location]
   OUTPUTS: [tensor_name: shape, dtype, memory_location]
   PARAMS:  [config_name: value, purpose]
   ```

2. **Algorithm Extraction**
   - Translate implementation to high-level pseudocode
   - Focus on mathematical operations, not memory management
   - Annotate loop structures with iteration dimensions

3. **Tiling & Blocking Strategy**
   - Block sizes: BLOCK_M, BLOCK_N, BLOCK_K
   - Number of tiles: how computed from input dimensions
   - Tile-to-thread mapping

4. **Memory Access Patterns**
   - Load → Compute → Store cycle
   - Memory hierarchy: Global (HBM) → Shared (SRAM) → Registers
   - Coalescing and bank conflict considerations

### Phase 3: Data Flow & Dependencies

1. **Inter-Operator Dependencies**
   - Which operators must complete before others start?
   - Is there pipelining or overlapping execution?
   - How are dependencies tracked? (scoreboard, barriers, explicit sync)

2. **Data Movement**
   - Tensor lifecycle through the model
   - In-place vs out-of-place operations
   - Buffer reuse patterns

3. **Distributed Execution** (if applicable)
   - Parallelism strategy: TP (Tensor), PP (Pipeline), SP (Sequence), DP (Data)
   - Communication primitives: all_reduce, all_gather, reduce_scatter
   - Compute-communication overlap

---

## Inference Infrastructure Analysis

For LLM serving systems (vLLM, SGLang, TensorRT-LLM, etc.), analyze these additional aspects:

### Phase A: System Architecture

1. **Service Topology**
   - Frontend API server (OpenAI-compatible, gRPC, etc.)
   - Tokenizer service (separate or integrated)
   - Backend scheduler/engine
   - Multi-GPU coordination

2. **Request Lifecycle**
   ```
   Request → Tokenize → Queue → Schedule → Prefill → Decode → Detokenize → Response
   ```

### Phase B: Scheduling & Batching

1. **Scheduling Strategies**
   - Continuous batching vs static batching
   - Prefill-decode disaggregation
   - Priority scheduling
   - Preemption policies

2. **Batch Formation**
   - How requests are grouped
   - Dynamic batch size limits
   - Chunked prefill for memory control

3. **Overlap Scheduling**
   - CPU-GPU overlap (prepare next batch while GPU busy)
   - Zero-overhead scheduling techniques

### Phase C: Memory Management

1. **KV Cache Architecture**
   ```
   ┌─────────────────────────────────────────────────┐
   │              KV CACHE MANAGEMENT                 │
   ├─────────────────────────────────────────────────┤
   │                                                  │
   │  PagedAttention (vLLM style):                   │
   │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐               │
   │  │Blk 0│ │Blk 1│ │Blk 2│ │Blk 3│  Physical     │
   │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘   Blocks      │
   │     │       │       │       │                   │
   │  Block Table (per request):                     │
   │  ┌───────────────────────┐                      │
   │  │ Req A: [0, 2, 3, _]   │                      │
   │  │ Req B: [1, _, _, _]   │                      │
   │  └───────────────────────┘                      │
   │                                                  │
   │  RadixAttention (SGLang style):                 │
   │  ┌─────────────────────────┐                    │
   │  │      Radix Tree         │ Shared prefix      │
   │  │         /\              │ reuse across       │
   │  │        /  \             │ requests           │
   │  │       A    B            │                    │
   │  └─────────────────────────┘                    │
   └─────────────────────────────────────────────────┘
   ```

2. **Memory Allocation**
   - Block size and allocation granularity
   - Pre-allocation vs on-demand
   - Eviction policies (LRU, reference counting)

3. **Prefix Caching**
   - Hash-based block deduplication
   - Automatic prefix sharing
   - Cache hit/miss handling

### Phase D: Request Flow Diagram

```
                    LLM SERVING SYSTEM ARCHITECTURE
                    ================================

┌─────────────┐     ┌──────────────┐     ┌──────────────────────────┐
│   Client    │────▶│   API Server │────▶│       Scheduler          │
│  Requests   │     │  (FastAPI)   │     │                          │
└─────────────┘     └──────────────┘     │  ┌────────────────────┐  │
                                         │  │   Waiting Queue    │  │
                                         │  └─────────┬──────────┘  │
                                         │            │             │
                                         │  ┌─────────▼──────────┐  │
                                         │  │   Running Batch    │  │
                                         │  │  (Continuous)      │  │
                                         │  └─────────┬──────────┘  │
                                         └────────────┼─────────────┘
                                                      │
                    ┌─────────────────────────────────▼─────────────────────────────────┐
                    │                         MODEL ENGINE                               │
                    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐   │
                    │  │   Prefill   │    │   Decode    │    │     KV Cache        │   │
                    │  │   (Batch)   │───▶│   (Token)   │◀──▶│   (Paged/Radix)     │   │
                    │  └─────────────┘    └─────────────┘    └─────────────────────┘   │
                    │                                                                    │
                    │  ┌────────────────────────────────────────────────────────────┐  │
                    │  │                    GPU Execution                            │  │
                    │  │  [Attention Kernel] [MLP Kernel] [AllReduce] [Sampling]    │  │
                    │  └────────────────────────────────────────────────────────────┘  │
                    └───────────────────────────────────────────────────────────────────┘
```

### Phase E: Key Metrics & Bottlenecks

1. **Latency Components**
   - Time to First Token (TTFT)
   - Inter-Token Latency (ITL)
   - End-to-end latency

2. **Throughput Factors**
   - Tokens per second (decode)
   - Requests per second
   - GPU utilization

3. **Memory Bottlenecks**
   - KV cache size vs batch size tradeoff
   - Memory fragmentation
   - Swap/offload overhead

### Inference System FAQ

Answer these for serving systems:

1. **Batching Strategy**
   - Continuous or static batching?
   - How are new requests added mid-batch?
   - Preemption support?

2. **KV Cache Design**
   - Paged or contiguous?
   - Block size?
   - Prefix caching mechanism?

3. **Scheduling**
   - FCFS, priority, or custom policy?
   - Prefill-decode separation?
   - Multi-request fairness?

4. **Scalability**
   - Multi-GPU support (TP/PP)?
   - Disaggregated architecture?
   - Load balancing?

---

## Output Deliverables

### 1. Architecture Overview (ASCII Diagram)

```
┌─────────────────────────────────────────────────────────────────┐
│                        MODEL FORWARD PASS                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Input   │───▶│ Embed +  │───▶│ Decoder  │───▶│   LM     │  │
│  │ Tokens   │    │   Norm   │    │  Layers  │    │   Head   │  │
│  └──────────┘    └──────────┘    └────┬─────┘    └──────────┘  │
│                                       │                          │
│                        ┌──────────────┴──────────────┐          │
│                        │      DECODER LAYER (×N)      │          │
│                        ├──────────────────────────────┤          │
│                        │  ┌─────────┐    ┌─────────┐ │          │
│                        │  │  Attn   │───▶│   MLP   │ │          │
│                        │  │ + Norm  │    │ + Norm  │ │          │
│                        │  └─────────┘    └─────────┘ │          │
│                        └──────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Task/Operator Breakdown Table

| Task Type | Purpose | Inputs | Outputs | Tiling | Dependencies |
|-----------|---------|--------|---------|--------|--------------|
| QKVProj | Linear projection | x:[B,S,H] | qkv:[B,S,3H] | M×K | RMSNorm |
| FlashAttn | Self-attention | qkv | attn_out | M×N blocks | QKVProj |
| ... | ... | ... | ... | ... | ... |

### 3. Pseudocode Abstraction

```python
# Example: Flash Attention (simplified)
def flash_attention(Q, K, V, causal=True):
    """
    Q, K, V: [batch, seq_len, num_heads, head_dim]

    Tiling: BLOCK_M=128 (query), BLOCK_N=128 (key/value)
    Memory: Q loaded once, K/V streamed in blocks
    """
    O = zeros_like(Q)

    for tile_m in range(0, seq_len, BLOCK_M):
        # Load Q block to SRAM (stays resident)
        q_block = Q[:, tile_m:tile_m+BLOCK_M, :, :]

        # Online softmax accumulators
        m_i = -inf  # running max
        l_i = 0     # running sum
        acc = 0     # output accumulator

        # Stream through K, V blocks
        kv_end = tile_m + BLOCK_M if causal else seq_len
        for tile_n in range(0, kv_end, BLOCK_N):
            k_block = K[:, tile_n:tile_n+BLOCK_N, :, :]
            v_block = V[:, tile_n:tile_n+BLOCK_N, :, :]

            # Compute attention scores
            scores = matmul(q_block, k_block.T) * scale

            if causal:
                scores = mask_future(scores, tile_m, tile_n)

            # Online softmax update
            m_new = max(m_i, max(scores))
            correction = exp(m_i - m_new)
            l_i = l_i * correction + sum(exp(scores - m_new))
            acc = acc * correction + matmul(softmax_local, v_block)
            m_i = m_new

        O[:, tile_m:tile_m+BLOCK_M] = acc / l_i

    return O
```

### 4. Data Flow Diagram (ASCII)

```
                    SINGLE DECODER LAYER DATA FLOW
                    ==============================

Input: hidden_states [B, S, H]
           │
           ▼
    ┌──────────────┐
    │   RMSNorm    │  ← weight: [H]
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   QKV Proj   │  ← weight: [3*H/TP, H]
    │   (Linear)   │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  QK Norm +   │  ← q_norm, k_norm weights
    │    RoPE      │  ← cos/sin cache
    └──────┬───────┘
           │
           ├────────────────┐
           │                │
           ▼                ▼
    ┌──────────┐     ┌──────────┐
    │ Q tensor │     │ KV Cache │ ← update
    └────┬─────┘     └────┬─────┘
         │                │
         └───────┬────────┘
                 │
                 ▼
          ┌──────────────┐
          │ Flash Attn   │
          │  (Tiled)     │
          └──────┬───────┘
                 │
                 ▼
          ┌──────────────┐
          │   O Proj     │  ← weight: [H, H/TP]
          └──────┬───────┘
                 │
                 ▼
          ┌──────────────┐
          │  AllReduce   │  ← TP sync (if TP > 1)
          └──────┬───────┘
                 │
                 ▼
          ┌──────────────┐
          │  Residual +  │  ← hidden_states
          └──────┬───────┘
                 │
    ┌────────────┴────────────┐
    │      (Similar flow      │
    │       for MLP block)    │
    └────────────┬────────────┘
                 │
                 ▼
Output: hidden_states [B, S, H]
```

### 5. Memory & Execution Timeline (ASCII)

```
SM Execution Timeline (Megakernel)
==================================

SM0: [RMSNorm T0][QKVProj T0][FlashAttn T0    ][OProj T0][AR wait]...
SM1: [RMSNorm T1][QKVProj T1][FlashAttn T1    ][OProj T1][AR wait]...
SM2: [RMSNorm T2][QKVProj T2][FlashAttn T2    ][OProj T2][AR wait]...
...
     ───────────────────────────────────────────────────────────▶ time

Legend:
- T0, T1, T2: Different tiles of the same operation
- AR: AllReduce synchronization point
- Tasks scheduled round-robin across SMs
- Dependencies tracked via scoreboard
```

### 6. Technical FAQ

Answer these based on code evidence:

1. **Sequence Length Support**
   - How are variable sequence lengths handled?
   - Is there padding, packing, or ragged batch support?
   - Maximum sequence length constraints?

2. **Hardware Requirements**
   - Tensor Core usage (WMMA, WGMMA)?
   - Memory alignment requirements?
   - Minimum GPU architecture?

3. **Kernel Fusion**
   - What operations are fused together?
   - QK-Norm + RoPE fusion?
   - Attention + output projection fusion?

4. **Precision & Numerics**
   - Input/output dtypes (fp16, bf16, fp8)?
   - Accumulation precision (fp32)?
   - Softmax numerical stability (online softmax)?

5. **Parallelism Configuration**
   - Supported TP/PP/SP configurations?
   - Communication patterns?
   - Load balancing strategy?

---

## Example Queries

**High-level understanding:**
- "Analyze the architecture of this LLM implementation"
- "How does this megakernel schedule tasks across SMs?"
- "What's the data flow through a single transformer layer?"

**Specific components:**
- "Explain the attention mechanism implementation with pseudocode"
- "How is the KV cache updated during decoding?"
- "What tiling strategy does the MLP use?"

**Performance & constraints:**
- "What sequence lengths are supported?"
- "Where are the synchronization points?"
- "How does tensor parallelism work in this implementation?"

**Deep dives:**
- "Trace the forward pass of a single token through the model"
- "Explain the dependency tracking mechanism"
- "How does the scoreboard synchronization work?"

---

## Analysis Checklist

When analyzing a new codebase:

- [ ] Identify entry points (model forward, inference API)
- [ ] Map module hierarchy (Model → Layer → Operator → Kernel)
- [ ] Extract configuration/hyperparameters
- [ ] Identify task/operator types
- [ ] Understand tiling and parallelization strategy
- [ ] Trace data dependencies between operators
- [ ] Document memory access patterns
- [ ] Identify fusion opportunities used
- [ ] Map distributed execution strategy (if any)
- [ ] Note hardware-specific optimizations
- [ ] Find constraint assertions and limitations
- [ ] Generate visual diagrams and pseudocode

---

## Supporting Files

- [analysis-template.md](analysis-template.md) - Report template
- [common-patterns.md](common-patterns.md) - Common ML implementation patterns
