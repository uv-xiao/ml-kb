# Mirage Persistent Kernel (MPK) Analysis

**Date:** 2026-01-11
**Paper:** arXiv:2512.22219
**Repo:** https://github.com/mirage-project/mirage (branch: mpk)
**Focus:** Megakernel architecture for LLM inference

---

## Executive Summary

Mirage Persistent Kernel (MPK) is a compiler and runtime that transforms LLM inference into a **single megakernel** - a fused GPU kernel that executes all computation and communication within one kernel launch. This eliminates kernel launch overhead and enables fine-grained scheduling across SMs.

**Key Claims:**
- Up to 1.7x end-to-end latency reduction vs kernel-per-operator systems (vLLM, SGLang)
- Pushes LLM inference close to hardware limits
- Single megakernel for entire forward pass + autoregressive loop

---

## 1. Architecture Overview

### Megakernel Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     SINGLE GPU KERNEL LAUNCH                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      WORKER CTAs (96)                            │   │
│  │  ┌──────┐ ┌──────┐ ┌──────┐     ┌──────┐                        │   │
│  │  │ SM 0 │ │ SM 1 │ │ SM 2 │ ... │SM 95 │  Execute compute tasks │   │
│  │  │Worker│ │Worker│ │Worker│     │Worker│                        │   │
│  │  └──────┘ └──────┘ └──────┘     └──────┘                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ▲                                          │
│                              │ Task Queues                              │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   SCHEDULER CTAs (12 warps)                      │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐     │   │
│  │  │Local Scheduler │  │Local Scheduler │  │Remote Scheduler│     │   │
│  │  │ (4 per CTA)    │  │ (4 per CTA)    │  │ (Multi-GPU)    │     │   │
│  │  └────────────────┘  └────────────────┘  └────────────────┘     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Design:**
- First `num_workers` CTAs execute compute tasks (e.g., 96 SMs)
- Remaining CTAs run scheduler warps (each warp = 1 scheduler)
- Total CTAs = `num_workers + (num_local_schedulers + num_remote_schedulers) / 4`

### Source Code Reference
- Main kernel: `include/mirage/persistent_kernel/persistent_kernel.cuh:1005-1013`
- Worker loop: `persistent_kernel.cuh:475-751`
- Scheduler loop: `persistent_kernel.cuh:754-1003`

---

## 2. Task Graph Representation

### Task Descriptor Structure

```cpp
// runtime_header.h:212-252
struct TaskDesc {
  TaskType task_type;           // TASK_LINEAR, TASK_ATTENTION, etc.
  unsigned variant_id;          // Kernel variant selection
  EventId trigger_event;        // Event triggered on completion
  EventId dependent_event;      // Event that must fire before execution
  void *input_ptrs[7];          // Up to 7 input tensors
  void *output_ptrs[3];         // Up to 3 output tensors
  TaskMetadata task_metadata;   // Task-specific parameters
};
```

### Task Types (Compute)

| Task | ID | Description |
|------|----|-------------|
| `TASK_EMBEDDING` | 101 | Token embedding lookup |
| `TASK_RMS_NORM` | 119 | RMSNorm layer |
| `TASK_LINEAR` | 120 | Matrix multiplication |
| `TASK_LINEAR_WITH_RESIDUAL` | 108 | Linear + residual add |
| `TASK_SILU_MUL` | 118 | SiLU activation + elementwise mul |
| `TASK_PAGED_ATTENTION_*` | 116-117 | Paged attention (decode) |
| `TASK_ALLREDUCE` | 106 | NVSHMEM all-reduce |
| `TASK_ARGMAX_*` | 109-111 | Argmax for sampling |

### Architecture-Specific Tasks

- **Hopper (SM90):** `TASK_LINEAR_HOPPER` (152), `TASK_PAGED_ATTENTION_HOPPER` (153)
- **Blackwell (SM100):** `TASK_LINEAR_SM100` (253), `TASK_ATTN_SM100` (257)

---

## 3. Decentralized Scheduling

### Worker Execution Flow

```
Worker Main Loop (persistent_kernel.cuh:475-751)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

while (true) {
    ┌──────────────────────────────────────────┐
    │ 1. FETCH TASKS (lines 528-599)           │
    │    - Poll ring buffer for next task      │
    │    - Load task descriptor to shared mem  │
    │    - Maintain 2 queues: local + remote   │
    └──────────────────────────────────────────┘
                      │
                      ▼
    ┌──────────────────────────────────────────┐
    │ 2. CHECK DEPENDENCIES (lines 601-630)    │
    │    - Thread 0 checks dependent_event     │
    │    - Busy-wait with __nanosleep(10ns)    │
    │    - NVSHMEM: nvshmem_signal_wait_until  │
    └──────────────────────────────────────────┘
                      │
                      ▼
    ┌──────────────────────────────────────────┐
    │ 3. EXECUTE TASK (lines 638-659)          │
    │    - _execute_task(task_desc, config)    │
    │    - Generated switch on task_type       │
    │    - TASK_TERMINATE → return             │
    └──────────────────────────────────────────┘
                      │
                      ▼
    ┌──────────────────────────────────────────┐
    │ 4. TRIGGER EVENTS (lines 661-748)        │
    │    - atom_add_release_gpu_u64()          │
    │    - If counter == num_triggers:         │
    │      enqueue event to scheduler          │
    └──────────────────────────────────────────┘
}
```

### Event-Based Dependencies

```
EVENT_LAUNCH_TASKS (901)
  └─> Scheduler distributes tasks to workers round-robin

EVENT_LAUNCH_MASSIVE_TASKS (902)
  └─> Split across all local schedulers for parallel distribution

EVENT_LAUNCH_DEPENDENT_TASKS (903)
  └─> Tasks wait on dependent_event before execution

EVENT_END_OF_TASK_GRAPH (910)
  └─> Batch complete, prepare next iteration or terminate
```

### Atomic Operations (mpk_atoms.cuh)

```cpp
atom_add_release_gpu_u64()  // Increment with GPU release semantics
atom_cas_release_gpu_u64()  // Compare-and-swap with release
ld_acquire_gpu_u64()        // Load with GPU acquire
ld_acquire_sys_u64()        // Load with system-wide acquire
st_relaxed_gpu_u64()        // Relaxed store (before signal)
```

---

## 4. Python API for Model Definition

### Creating a Persistent Kernel

```python
import mirage as mi

mpk = mi.PersistentKernel(
    mode="offline",              # offline/online/online_notoken
    world_size=world_size,
    mpi_rank=rank,
    num_workers=96,              # Number of worker SMs
    num_local_schedulers=48,     # Local scheduler warps
    num_remote_schedulers=0,     # Remote (multi-GPU) schedulers
    max_seq_length=4096,
    max_num_batched_requests=1,
    max_num_batched_tokens=8,
    meta_tensors={...},          # Runtime state tensors
    profiler_tensor=None,
)
```

### Defining Layers

```python
# Attach model weights
w_qkv = mpk.attach_input(torch_tensor=layer.qkv_proj.weight, name="qkv")

# Allocate intermediate tensors
hidden = mpk.new_tensor(
    dims=(batch_size, hidden_size),
    dtype=mi.bfloat16,
    name="hidden",
    io_category="cuda_tensor",   # or "nvshmem_tensor" for multi-GPU
)

# Add layers with explicit grid/block dims
mpk.rmsnorm_layer(
    input=x, weight=w_norm, output=rmsnorm_out,
    grid_dim=(batch_size, 1, 1),  # Number of tasks
    block_dim=(128, 1, 1),        # Threads per task
)

mpk.linear_layer(
    input=rmsnorm_out, weight=w_qkv, output=attn_in,
    grid_dim=(96, 1, 1),
    block_dim=(128, 1, 1),
)

mpk.paged_attention_layer(
    input=attn_in, k_cache=k_cache, v_cache=v_cache,
    q_norm=w_q_norm, k_norm=w_k_norm,
    cos_pos_embed=cos, sin_pos_embed=sin,
    output=attn_out,
    grid_dim=(max_requests, num_kv_heads, 1),
    block_dim=(128, 1, 1),
)
```

### Compilation and Execution

```python
# Generate task graph and CUDA code
results = mpk.kn_graph.generate_task_graph(num_gpus=world_size, my_gpu_id=rank)

# Compile with nvcc
mpk.compile(output_dir="./compiled")

# Execute megakernel (runs entire inference loop)
mpk()  # Single call executes all autoregressive steps!
```

---

## 5. Kernel Implementation Examples

### RMSNorm Task (Ampere)

```cpp
// tasks/ampere/rmsnorm.cuh
template <typename T, int BATCH_SIZE, int HIDDEN_DIM>
__device__ __forceinline__ void
rms_norm_impl(void const *input_ptr,
              void const *weight_ptr,
              void *output_ptr,
              float eps) {
    // Uses copy-async for memory operations
    // Tile-based computation with async pipeline
    // output = input * weight / sqrt(sum(input^2) + eps)
}
```

### Linear Layer (Hopper)

```cpp
// tasks/hopper/linear_hopper.cuh
template <typename T, int BATCH_SIZE, int OUTPUT_SIZE,
          int REDUCTION_SIZE, int O_STRIDE = OUTPUT_SIZE>
__device__ __forceinline__ void
linear_hopper_impl(void const *input_ptr,
                   void const *weight_ptr,
                   void *output_ptr,
                   int num_active_tokens) {
    // Uses TMA (Tensor Memory Accelerator) for loads
    // WGMMA (Warpgroup Matrix Multiply Accumulate)
}
```

### Paged Attention

```cpp
// tasks/ampere/multitoken_paged_attention.cuh
template <typename T, int NUM_Q_HEADS, int NUM_KV_HEADS, int HEAD_DIM>
__device__ __forceinline__ void
paged_attention_impl(void const *qkv_ptr,
                     void const *k_cache_ptr,
                     void const *v_cache_ptr,
                     void *output_ptr,
                     RuntimeConfig const &config,
                     int request_id) {
    // Paged KV cache access via indirection
    // Supports variable sequence lengths per request
}
```

---

## 6. Comparison with SGLang CUDA Graphs

| Feature | SGLang CUDA Graph | Mirage MPK |
|---------|-------------------|------------|
| **Granularity** | Graph per batch size | Single megakernel |
| **Scheduling** | GPU driver | In-kernel schedulers |
| **Dependencies** | Implicit (graph capture) | Explicit events |
| **Flexibility** | Fixed graph topology | Dynamic task dispatch |
| **Launch overhead** | 1 launch per graph | 1 launch total |
| **Memory** | Captured buffers | Ring buffer queues |
| **Multi-GPU** | Separate graphs | NVSHMEM integration |

### Key Differences

1. **CUDA Graph:** Captures kernel sequence, replays entire sequence
2. **MPK Megakernel:** Workers pull tasks dynamically from queue

```
CUDA Graph:
  launch() → K1 → K2 → K3 → ... → Kn → return

MPK Megakernel:
  launch() → Workers poll queue → Execute tasks → Signal events → Loop
```

---

## 7. Integration with SGLang

### Option A: Replace Attention Backend

```python
# New backend that uses MPK for attention
@register_attention_backend("mpk")
class MPKAttnBackend(AttentionBackend):
    def __init__(self, model_runner):
        self.mpk = mi.PersistentKernel(...)
        # Define attention layers

    def forward_decode(self, q, k, v, layer, forward_batch):
        # Populate input tensors
        self.mpk.input_tensors["q"].copy_(q)
        # Execute single attention task
        self.mpk.execute_single_task("attention")
```

**Challenge:** MPK designed for entire forward pass, not individual ops.

### Option B: Replace CUDA Graph System

```python
# In cuda_graph_runner.py
class MPKGraphRunner:
    def __init__(self, model_runner):
        self.mpk = mi.PersistentKernel(...)
        # Build entire model graph

    def capture(self):
        # Generate MPK task graph
        self.mpk.compile()

    def replay(self, forward_batch):
        # Update input tensors
        self.mpk.meta_tensors["input_tokens"].copy_(forward_batch.input_ids)
        self.mpk()  # Execute megakernel
```

**Pros:** Clean integration, single kernel for all ops
**Cons:** Requires model redefinition in MPK API

### Option C: Hybrid Approach

Use MPK for compute-heavy layers (attention, MLP), keep scheduler in SGLang:

```python
class HybridModelRunner:
    def __init__(self):
        self.mpk_attention = MPKAttentionKernel(...)
        self.mpk_mlp = MPKMLPKernel(...)

    def forward(self, batch):
        # RMSNorm (native PyTorch)
        x = self.rmsnorm(x)
        # Attention (MPK megakernel)
        x = self.mpk_attention(x, batch)
        # MLP (MPK megakernel)
        x = self.mpk_mlp(x)
```

---

## 8. Key Files Reference

| Component | File | Key Lines |
|-----------|------|-----------|
| Main kernel | `persistent_kernel.cuh` | 1005-1013 |
| Worker execution | `persistent_kernel.cuh` | 475-751 |
| Scheduler execution | `persistent_kernel.cuh` | 754-1003 |
| Task types | `runtime_header.h` | 76-142 |
| Task descriptor | `runtime_header.h` | 212-252 |
| Python API | `persistent_kernel.py` | 243-400 |
| Layer definitions | `persistent_kernel.py` | 373-1179 |
| Compilation | `persistent_kernel.py` | 1305-1463 |
| Ampere attention | `tasks/ampere/multitoken_paged_attention.cuh` | Full file |
| Hopper linear | `tasks/hopper/linear_hopper.cuh` | Full file |

---

## 9. Limitations and Considerations

1. **Static Task Graph:** Task graph generated at compile time
2. **Fixed Batch Sizes:** `max_num_batched_tokens` fixed at compilation
3. **Learning Curve:** Requires model redefinition in MPK API
4. **Debugging:** Single kernel harder to profile/debug than separate ops
5. **Compatibility:** Requires architecture-specific task implementations

---

## 10. Conclusion

Mirage MPK represents a significant advancement in LLM inference optimization by:

1. **Eliminating kernel launch overhead** through persistent megakernel
2. **Enabling fine-grained scheduling** with SM-level task distribution
3. **Supporting multi-GPU** via NVSHMEM integration

For SGLang integration, the most practical approach is **Option B (CUDA Graph replacement)** - creating an `MPKGraphRunner` that compiles the entire model into a megakernel and uses it as an alternative to traditional CUDA graphs.

The main barrier is the need to redefine models using MPK's layer API rather than standard PyTorch modules, but this provides maximum performance benefit.
