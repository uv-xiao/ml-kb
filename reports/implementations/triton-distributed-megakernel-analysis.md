# Analysis: Triton-Distributed Megakernel (Qwen3)

**Repository:** https://github.com/ByteDance-Seed/Triton-distributed
**Framework:** Triton + PyTorch (Megakernel execution model)
**Analysis Date:** 2026-01-06

---

## 1. Executive Summary

Triton-distributed's megakernel implementation fuses an entire LLM forward pass into a single persistent kernel. Instead of launching separate kernels for each operation (RMSNorm, Attention, MLP, etc.), all operations are encoded as **tasks** scheduled to SMs via work queues, with dependencies tracked through a **scoreboard** mechanism.

**Key Features:**
- Single kernel launch for entire forward pass (eliminates launch overhead)
- Fine-grained task scheduling across SMs (round-robin or zig-zag)
- Scoreboard-based dependency tracking (producer-consumer synchronization)
- Tensor Parallel support with intra-node AllReduce
- Support for Qwen3 models with QK-Norm and RoPE

---

## 2. Architecture Overview

### 2.1 System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MEGAKERNEL EXECUTION MODEL                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Python (Host)                         GPU (Device)                         │
│  ─────────────                         ────────────                         │
│                                                                             │
│  ┌──────────────┐                     ┌──────────────────────────────────┐ │
│  │ ModelBuilder │──build_fwd()───────▶│      Work Queue (per SM)         │ │
│  │              │                      │  ┌────┬────┬────┬────┬────┐     │ │
│  │  - make_*()  │                      │  │Task│Task│Task│Task│... │     │ │
│  │  - compile() │                      │  │ 0  │ 1  │ 2  │ 3  │    │     │ │
│  │  - run()     │                      │  └────┴────┴────┴────┴────┘     │ │
│  └──────────────┘                      └──────────────┬───────────────────┘ │
│         │                                             │                     │
│         │                              ┌──────────────▼───────────────────┐ │
│         │                              │        MEGA_TRITON_KERNEL        │ │
│         │                              │  ┌─────────────────────────────┐ │ │
│         │                              │  │  SM0   SM1   SM2   ...  SMn │ │ │
│         │                              │  │   ▼     ▼     ▼         ▼   │ │ │
│         │                              │  │ ┌───┐ ┌───┐ ┌───┐     ┌───┐│ │ │
│         └──────run()──────────────────▶│  │ │WQ │ │WQ │ │WQ │ ... │WQ ││ │ │
│                                        │  │ └─┬─┘ └─┬─┘ └─┬─┘     └─┬─┘│ │ │
│                                        │  │   │     │     │         │  │ │ │
│                                        │  │   ▼     ▼     ▼         ▼  │ │ │
│                                        │  │ [Execute Tasks in Order]   │ │ │
│                                        │  │                            │ │ │
│                                        │  │   ┌─────────────────┐      │ │ │
│                                        │  │   │   SCOREBOARD    │      │ │ │
│                                        │  │   │ (Dependency Sync)│      │ │ │
│                                        │  │   └─────────────────┘      │ │ │
│                                        │  └─────────────────────────────┘ │ │
│                                        └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Module Hierarchy

```
ModelBuilder (core/model_builder.py)
├── Graph (core/graph.py)
│   └── Node → TaskBase
├── Scheduler (core/scheduler.py)
│   ├── round_robin_scheduler
│   └── zig_zag_scheduler
├── CodeGenerator (core/code_generator.py)
└── Registry (core/registry.py)
    └── TaskBuilders
        ├── RMSNormTask
        ├── QKVProjTaskBuilder
        ├── QKNormRopeUpdateKVCacheTaskBuilder
        ├── FlashAttnTaskBuilder / AttnSplitTaskBuilder + AttnCombineTaskBuilder
        ├── OProjTaskBuilder
        ├── AllReduceTaskBuilder
        ├── MLPFC1TaskBuilder
        ├── SiLUMulUpTaskBuilder
        ├── MLPFC2TaskBuilder (via OProjTaskBuilder)
        └── AddTaskBuilder

DenseModel (models/dense.py)
├── DenseLayerBuilder (×num_layers)
│   ├── TPAttnBuilder (layers/tp_attn.py)
│   └── TPMLPBuilder (layers/tp_mlp.py)
└── PagedKVCache (models/paged_kv_cache.py)
```

### 2.3 Execution Model

**Megakernel Pattern:**
1. Host builds task graph via `make_*()` calls
2. `compile()` generates Triton kernel code dynamically
3. Tasks scheduled to work queues (one per SM)
4. Single `run()` launches persistent kernel
5. Each SM pops tasks from its queue, executes, signals completion via scoreboard
6. Consumers wait on scoreboard before proceeding

---

## 3. Configuration & Parameters

### 3.1 Model Configuration (from HuggingFace config)

| Parameter | Qwen3-32B | Description |
|-----------|-----------|-------------|
| `hidden_size` | 5120 | Hidden dimension |
| `num_layers` | 64 | Decoder layers |
| `num_attention_heads` | 40 | Query heads |
| `num_key_value_heads` | 8 | KV heads (GQA) |
| `head_dim` | 128 | Per-head dimension |
| `intermediate_size` | 27648 | MLP hidden size |
| `rope_theta` | 1000000 | RoPE base frequency |

### 3.2 Kernel Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `BLOCK_M` | 128 | Attention query tile size |
| `BLOCK_N` | 128 | Attention KV tile size |
| `NUM_STAGES` | 3 | Pipeline stages for async loads |
| `num_warps` | 4 | Warps per thread block |

### 3.3 Constraints & Requirements

From code assertions:
- `MAX_NUM_TENSOR_DIMS = 4` (tensor dimensions)
- Tensor alignment: `data_ptr() % 16 == 0`
- K dimension: `K % 32 == 0` for linear ops
- Currently decode-only: `seq_len == 1`
- AllReduce requires NVSHMEM with multicast support (Hopper+)

---

## 4. Component Analysis

### 4.1 Task Base System

**Purpose:** Abstract representation of a GPU operation tile

**Interface (core/task_base.py:162):**
```
TaskBase:
  - layer_id: int          # Which layer this belongs to
  - task_id: int           # Unique ID within layer
  - tile_id_or_start: int  # Which tile of the operation
  - num_tiles: int         # Total tiles for this operation
  - config: ConfigBase     # Kernel config (BLOCK_M, etc.)
  - dependency: List[TaskDependency]  # What must complete first
  - io_tensors: List[List[Tensor]]    # [inputs, outputs]
  - inputs_dep: Dict[Tensor, InputDependencyDesc]   # Fine-grained deps
  - outs_tile_mapping: Dict[Tensor, OutputTilingDesc]
```

**Encoding for GPU:**
```python
def encoding_with_deps(self, l, r) -> Tuple[int]:
    """
    task_type | layer_id | task_id | tile_id_or_start | dep_range(l,r) | io_tensors | extra_params
    """
    # task_type: identifies which kernel function to call
    # dep_range: indices into global dependency array
    # io_tensors: data_ptr + shape for each tensor
```

### 4.2 Dependency Tracking (Scoreboard)

**Purpose:** Producer-consumer synchronization without explicit barriers

**Mechanism (core/scheduler.py:71-72):**
```python
# Scoreboard: 3D tensor [max_layer_id+1, max_task_id+1, max_num_tiles]
scoreboard = torch.zeros((max_layer_id + 1, max_task_id + 1, max_num_tiles), dtype=int32)

# Producer signals completion:
scoreboard[layer_id, task_id, tile_id] = 1

# Consumer waits:
while scoreboard[dep_layer, dep_task, dep_tile_start:dep_tile_end] != all_ones:
    spin_wait()
```

**Pseudocode:**
```python
def execute_task(task, scoreboard):
    # 1. Wait for dependencies
    for dep in task.dependencies:
        wait_until(scoreboard[dep.layer_id, dep.task_id, dep.start:dep.end] == 1)

    # 2. Execute compute
    task_compute_function(task.tile_id, task.io_tensors, task.config)

    # 3. Signal completion
    scoreboard[task.layer_id, task.task_id, task.tile_id] = 1
```

### 4.3 Flash Attention (Decode)

**Purpose:** Self-attention with paged KV cache

**Interface (tasks/flash_decode.py):**
```
INPUTS:
  - query: [B, 1, num_q_heads, head_dim], bf16
  - key_cache: [MAX_BLOCKS, PAGE_SIZE, num_kv_heads, head_dim], bf16
  - value_cache: [MAX_BLOCKS, PAGE_SIZE, num_kv_heads, head_dim], bf16
  - block_tables: [B, MAX_BLOCKS_PER_SEQ], int32
  - kv_lens: [B], int32

OUTPUTS:
  - output: [B, 1, num_q_heads, head_dim], bf16

PARAMS:
  - sm_scale: head_dim^-0.5
  - soft_cap: 0.0 (disabled)
  - NUM_KV_SPLITS: 32
```

**Pseudocode (split-k attention for decode):**
```python
def flash_decode(q, k_cache, v_cache, block_tables, kv_lens):
    """
    Split-K strategy: divide KV sequence across NUM_KV_SPLITS
    Each split computes partial attention, combined at end
    """
    B, _, H_Q, D = q.shape
    NUM_KV_SPLITS = 32

    # Phase 1: Compute partial attention per split
    partial_out = zeros(B, H_Q, NUM_KV_SPLITS, D)
    lse = zeros(B, H_Q, NUM_KV_SPLITS)  # log-sum-exp for combination

    for split_id in range(NUM_KV_SPLITS):
        kv_start = split_id * (kv_len // NUM_KV_SPLITS)
        kv_end = min(kv_start + split_size, kv_len)

        # Standard attention on this KV range
        k_split = gather_from_cache(k_cache, block_tables, kv_start, kv_end)
        v_split = gather_from_cache(v_cache, block_tables, kv_start, kv_end)

        scores = q @ k_split.T * sm_scale
        weights = softmax(scores)
        partial_out[:, :, split_id] = weights @ v_split
        lse[:, :, split_id] = logsumexp(scores)

    # Phase 2: Combine partial results
    # Numerically stable combination using LSE
    max_lse = lse.max(dim=-1)
    weights = exp(lse - max_lse)
    output = (partial_out * weights).sum(dim=2) / weights.sum(dim=2)

    return output
```

### 4.4 QK-Norm + RoPE + KV Cache Update

**Purpose:** Fused operation for Qwen3-style attention preprocessing

**Interface (tasks/norm.py):**
```
INPUTS:
  - qkv: [B, S, num_q_heads + 2*num_kv_heads, head_dim], bf16
  - block_tables: [B, MAX_BLOCKS], int32
  - kv_lens: [B], int32
  - q_rms_weight, k_rms_weight: [head_dim], bf16
  - cos_cache, sin_cache: [1, MAX_SEQ, head_dim], fp32

OUTPUTS:
  - key_cache: updated in-place
  - value_cache: updated in-place
  - q_norm_rope: [B, S, num_q_heads, head_dim], bf16
```

**Pseudocode:**
```python
def qk_norm_rope_update_kvcache(qkv, block_tables, kv_lens, q_w, k_w, cos, sin, k_cache, v_cache):
    """
    Fused: Split QKV → RMSNorm Q,K → RoPE Q,K → Update KV Cache
    """
    num_q = num_q_heads
    num_kv = num_kv_heads

    # Split packed QKV
    q = qkv[..., :num_q, :]
    k = qkv[..., num_q:num_q+num_kv, :]
    v = qkv[..., num_q+num_kv:, :]

    # RMSNorm on Q and K
    q_norm = rms_norm(q, q_w, eps)
    k_norm = rms_norm(k, k_w, eps)

    # Apply RoPE
    pos = kv_lens - 1  # current position
    q_rope = apply_rope(q_norm, cos[pos], sin[pos])
    k_rope = apply_rope(k_norm, cos[pos], sin[pos])

    # Update KV cache (paged)
    block_idx = pos // PAGE_SIZE
    offset = pos % PAGE_SIZE
    physical_block = block_tables[block_idx]
    k_cache[physical_block, offset] = k_rope
    v_cache[physical_block, offset] = v

    return q_rope
```

### 4.5 Tensor Parallel MLP

**Purpose:** Distributed MLP with column/row parallelism

**Interface (models/layers/tp_mlp.py):**
```
INPUTS:
  - x: [B, S, hidden_size], bf16

OUTPUTS:
  - output: [B, S, hidden_size], bf16

Weights (sharded):
  - gate_up_proj: [intermediate_size*2 // TP, hidden_size]
  - down_proj: [hidden_size, intermediate_size // TP]
```

**Pseudocode:**
```python
def tp_mlp_forward(x, gate_up_proj, down_proj, world_size):
    """
    TP Strategy:
    - FC1 (gate_up): Column-parallel (output sharded)
    - Activation: Local
    - FC2 (down): Row-parallel (requires AllReduce)
    """
    # FC1: gate and up projection fused
    # Each rank computes [B*S, intermediate*2 // TP]
    fc1_out = x @ gate_up_proj.T

    # SiLU(gate) * up
    gate, up = fc1_out.chunk(2, dim=-1)
    act_out = silu(gate) * up

    # FC2: down projection
    # Each rank has partial result
    fc2_out = act_out @ down_proj.T  # [B*S, hidden // TP contribution]

    # AllReduce to get full hidden
    if world_size > 1:
        output = all_reduce(fc2_out)
    else:
        output = fc2_out

    return output
```

---

## 5. Task/Operator Catalog

| ID | Task Type | Purpose | Tiling | Dependencies |
|----|-----------|---------|--------|--------------|
| 1 | `rms_norm` | RMSNorm before attention | per-row | previous layer output |
| 2 | `qkv_proj` | QKV linear projection | M×K tiles | RMSNorm |
| 3 | `qk_norm_rope_update_kvcache` | Fused norm+rope+cache | per-token | QKV proj |
| 4 | `attn_split` | Split-K attention phase 1 | per-head×split | Norm+Rope |
| 5 | `attn_combine` | Split-K attention phase 2 | per-head | Attn split |
| 6 | `o_proj` | Output projection | M×K tiles | Attn combine |
| 7 | `allreduce` | TP sync for attention | full tensor | O proj |
| 8 | `add` | Residual connection | element-wise | AllReduce, input |
| 9 | `rms_norm` | RMSNorm before MLP | per-row | Add (attn residual) |
| 10 | `mlp_fc1` | Gate+Up projection | M×K tiles | RMSNorm |
| 11 | `silu_mul_up` | SiLU(gate) * up | element-wise | FC1 |
| 12 | `mlp_fc2` | Down projection | M×K tiles | Activation |
| 13 | `allreduce` | TP sync for MLP | full tensor | FC2 |
| 14 | `add` | Residual connection | element-wise | AllReduce, attn residual |

---

## 6. Data Flow & Dependencies

### 6.1 Single Decoder Layer

```
                     DECODER LAYER DATA FLOW
                     =======================

hidden_states [B, 1, H]
       │
       ├──────────────────────────────────────────────┐
       │                                              │
       ▼                                              │
┌──────────────┐                                      │
│  RMSNorm     │◀── input_norm_w [H]                  │
└──────┬───────┘                                      │
       │                                              │
       ▼                                              │
┌──────────────┐                                      │
│  QKV Proj    │◀── qkv_weight [3H/TP, H]             │
│  (Linear)    │                                      │
└──────┬───────┘                                      │
       │                                              │
       ▼                                              │
┌──────────────┐                                      │
│ QK Norm +    │◀── q_norm_w, k_norm_w [head_dim]     │
│ RoPE +       │◀── cos, sin [max_seq, head_dim]      │
│ KV Update    │──▶ k_cache, v_cache (in-place)       │
└──────┬───────┘                                      │
       │ q_rope                                       │
       ▼                                              │
┌──────────────┐                                      │
│ Flash Attn   │◀── k_cache, v_cache, block_tables    │
│ (Split-K)    │                                      │
└──────┬───────┘                                      │
       │                                              │
       ▼                                              │
┌──────────────┐                                      │
│   O Proj     │◀── o_weight [H, H/TP]                │
└──────┬───────┘                                      │
       │                                              │
       ▼                                              │
┌──────────────┐                                      │
│  AllReduce   │  (if TP > 1)                         │
└──────┬───────┘                                      │
       │                                              │
       ▼                                              │
┌──────────────┐                                      │
│    Add       │◀─────────────────────────────────────┘
│  (residual)  │                    (residual connection)
└──────┬───────┘
       │
       ├──────────────────────────────────────────────┐
       │                                              │
       ▼                                              │
┌──────────────┐                                      │
│  RMSNorm     │◀── post_norm_w [H]                   │
└──────┬───────┘                                      │
       │                                              │
       ▼                                              │
┌──────────────┐                                      │
│  Gate+Up     │◀── gate_up_w [2*I/TP, H]             │
│  (FC1)       │                                      │
└──────┬───────┘                                      │
       │                                              │
       ▼                                              │
┌──────────────┐                                      │
│  SiLU * Up   │                                      │
└──────┬───────┘                                      │
       │                                              │
       ▼                                              │
┌──────────────┐                                      │
│   Down       │◀── down_w [H, I/TP]                  │
│   (FC2)      │                                      │
└──────┬───────┘                                      │
       │                                              │
       ▼                                              │
┌──────────────┐                                      │
│  AllReduce   │  (if TP > 1)                         │
└──────┬───────┘                                      │
       │                                              │
       ▼                                              │
┌──────────────┐                                      │
│    Add       │◀─────────────────────────────────────┘
│  (residual)  │                    (residual connection)
└──────┬───────┘
       │
       ▼
output [B, 1, H]
```

### 6.2 Scoreboard Dependency Example

```
Layer 0, Attention Block:

Task Graph:
  RMSNorm(T0,T1,T2) ──▶ QKVProj(T0,T1,T2,...) ──▶ QKNormRope(T0,T1,...) ──▶ ...

Scoreboard State Evolution:

Time   │ scoreboard[0,rms,*]  │ scoreboard[0,qkv,*]  │ scoreboard[0,rope,*]
───────┼──────────────────────┼──────────────────────┼──────────────────────
  t0   │ [0,0,0]              │ [0,0,0,0,...]        │ [0,0,...]
  t1   │ [1,0,0] ← T0 done    │ [0,0,0,0,...]        │ [0,0,...]
  t2   │ [1,1,0] ← T1 done    │ [1,0,0,0,...] ← T0   │ [0,0,...]
  t3   │ [1,1,1] ← T2 done    │ [1,1,0,0,...]        │ [1,0,...] ← T0
  ...
```

---

## 7. Parallelism & Distribution

### 7.1 Parallelism Strategy

- **Tensor Parallelism (TP):** Yes
  - Attention: QKV column-parallel, O row-parallel
  - MLP: Gate/Up column-parallel, Down row-parallel
  - AllReduce after O-proj and Down-proj

- **Pipeline Parallelism (PP):** Not implemented in this codebase
- **Sequence Parallelism (SP):** Not implemented
- **Data Parallelism (DP):** External (via distributed launcher)

### 7.2 Communication Patterns

| Operation | Primitive | Location | Data Size |
|-----------|-----------|----------|-----------|
| Post O-proj | AllReduce | After attention | B × S × H |
| Post FC2 | AllReduce | After MLP | B × S × H |

### 7.3 AllReduce Implementation

```python
def make_allreduce(input, output, double_input_buffer=False):
    """
    Uses NVSHMEM multicast for efficient intra-node AllReduce.
    Requires Hopper+ GPU architecture.
    """
    # 1. Barrier to ensure all ranks ready
    make_barrier_all_intra_node(wait_inputs=[input])

    # 2. Perform AllReduce
    _convert_op("allreduce", [[input], [output]])

    # 3. Optional barrier if input buffer reused
    if not double_input_buffer:
        make_barrier_all_intra_node(wait_inputs=[output])
```

---

## 8. Memory & Performance

### 8.1 SM Execution Timeline

```
Megakernel Execution (Round-Robin Scheduling)
=============================================

SM0: [Norm0][QKV0 ][Rope0][Attn0      ][OProj0][AR    ][Add0]...
SM1: [Norm1][QKV4 ][Rope1][Attn4      ][OProj1][AR    ][Add1]...
SM2: [Norm2][QKV8 ][Rope2][Attn8      ][OProj2][AR    ][Add2]...
SM3: [Norm3][QKV12][Rope3][Attn12     ][OProj3][wait  ][Add3]...
     ─────────────────────────────────────────────────────────▶ time

Legend:
- Each SM processes its work queue sequentially
- Tasks wait on scoreboard for dependencies
- AR = AllReduce (synchronization point across all ranks)
```

### 8.2 Dependency Optimization

From `task_dependency_opt()` in scheduler.py:
```python
# Optimization: Remove redundant dependencies
# If task T3 depends on tiles [0,1,2] and T4 depends on [0,1,2,3]
# And T3 runs before T4 on same SM, T4's dep on [0,1,2] is redundant
# (T3 already waited for them)
```

---

## 9. Technical FAQ

### Q: How are variable sequence lengths handled?
**A:** Currently decode-only (`seq_len == 1`). KV lengths tracked per-batch via `kv_lens` tensor. Paged attention supports variable KV lengths via block tables.
*Reference: dense.py:170*

### Q: What is the maximum supported sequence length?
**A:** Configurable via `max_length` parameter. Limited by KV cache allocation and `max_position_embeddings` from model config.
*Reference: dense.py:135-137*

### Q: What operations are fused?
**A:**
- QK-Norm + RoPE + KV Cache Update (single task)
- Gate + Up projection (single matmul with concatenated weights)
- SiLU + elementwise multiply
*Reference: tasks/norm.py, layers/tp_mlp.py*

### Q: What hardware is required?
**A:**
- NVIDIA GPU with Triton support
- For TP > 1: Hopper+ (H100) for NVSHMEM multicast AllReduce
- Tensor alignment: 16-byte
*Reference: model_builder.py:479-483*

### Q: How does the KV cache work?
**A:** Paged attention with fixed block size. `block_tables` maps logical positions to physical blocks. Cache updated in-place during QK-Norm+RoPE task.
*Reference: models/paged_kv_cache.py*

---

## 10. Key Files Reference

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `core/task_base.py` | Task abstraction | `TaskBase`, `TaskDependency`, `CodeGenKey` |
| `core/graph.py` | Dependency graph | `Graph`, `Node`, auto-dependency inference |
| `core/scheduler.py` | Task scheduling | `enque_tasks`, `round_robin_scheduler` |
| `core/code_generator.py` | Dynamic kernel codegen | `CodeGenerator.generate_code` |
| `core/builder.py` | Task builder base | `TaskBuilderBase` |
| `models/model_builder.py` | Main API | `ModelBuilder`, all `make_*` methods |
| `models/dense.py` | Qwen3 model | `DenseModel`, `DenseLayerBuilder` |
| `tasks/flash_attn.py` | Attention task | `FlashAttnTaskBuilder` |
| `tasks/flash_decode.py` | Decode attention | `AttnSplitTaskBuilder`, `AttnCombineTaskBuilder` |
| `tasks/norm.py` | Norm tasks | `QKNormRopeUpdateKVCacheTaskBuilder` |
| `kernels/flash_attn.py` | Triton attention | `_attn_fwd_inner`, `qkv_pack_flash_attn_task_compute` |

---

## 11. Appendix

### A. Glossary

| Term | Definition |
|------|------------|
| Megakernel | Single GPU kernel executing multiple fused operations |
| Scoreboard | Shared memory structure for tracking task completion |
| Work Queue | Per-SM list of tasks to execute |
| Tile | Subdivision of an operation for parallel execution |
| TP | Tensor Parallelism - splitting model weights across GPUs |
| GQA | Grouped Query Attention - multiple Q heads share K,V |
| Paged Attention | KV cache organized as fixed-size blocks |

### B. References

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Triton Documentation](https://triton-lang.org/)
- [NVSHMEM Documentation](https://docs.nvidia.com/hpc-sdk/nvshmem/)

---

*Generated by LLM Code Analysis Skill*
