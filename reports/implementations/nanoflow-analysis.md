# NanoFlow Codebase Analysis

## Overview

NanoFlow is a throughput-oriented LLM serving framework that exploits **intra-device parallelism** through nano-batching and SM (Streaming Multiprocessor) partitioning. The key innovation is overlapping compute, memory, and network operations within a single GPU.

**Repository**: [efeslab/Nanoflow](https://github.com/efeslab/Nanoflow)
**Paper**: [arXiv:2408.12757](https://arxiv.org/abs/2408.12757)

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NanoFlow System                              │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │   Frontend  │  │  Auto Search│  │   Worker    │                  │
│  │  (Python)   │──│  (Gurobi)   │──│  Processes  │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
│         │                │                │                         │
│         ▼                ▼                ▼                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      Pipeline                                │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │   │
│  │  │ Global  │  │LayerNorm│  │   KQV   │  │  Rope   │        │   │
│  │  │ Input   │──│  Attn   │──│  GEMM   │──│ Append  │        │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │   │
│  │       │            │            │            │               │   │
│  │       ▼            ▼            ▼            ▼               │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │   │
│  │  │ Decode  │  │ Prefill │  │    O    │  │LayerNorm│        │   │
│  │  │  Attn   │  │  Attn   │  │  GEMM   │  │   FFN   │        │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │   │
│  │       │            │            │            │               │   │
│  │       ▼            ▼            ▼            ▼               │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │   │
│  │  │  UG     │  │Activation│  │    D    │  │ Sample  │        │   │
│  │  │  GEMM   │──│  SiLU   │──│  GEMM   │──│         │        │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      Executor                                │   │
│  │  • Topological sort of operations                           │   │
│  │  • CUDA event synchronization between streams               │   │
│  │  • CUDA graph capture and replay                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
nanoflow/
├── core/                    # Core execution framework
│   ├── executor.py          # DAG-based operation execution
│   ├── nanobatchSplit.py    # Nano-batch splitting logic
│   ├── bufferAllocate.py    # Memory allocation via LP
│   ├── IOWrapper.py         # Input/output tensor wrappers
│   ├── weightWrapper.py     # Weight tensor management
│   └── categoryType.py      # Operation categories (COMP/MEM/NET)
├── operations/              # Operation implementations
│   ├── operation_base.py    # Base classes for operations
│   ├── gemm/                # GEMM implementations
│   ├── attention/           # FlashAttention, vLLM attention
│   ├── norm/                # RMSNorm kernels
│   ├── allreduce/           # NCCL/MSCCL++ wrappers
│   └── virtualOp/           # Copy, Redistribute ops
├── models/                  # Model definitions
│   ├── llama3_AutoSearch.py # LLaMA-3 8B pipeline
│   └── llama3_70B_*.py      # LLaMA-3 70B variants
├── auto_search/             # Automated parameter search
│   ├── new_search.py        # Gurobi-based ILP optimizer
│   └── profileAnalysis.py   # Profile data processing
├── kvcache/                 # KV cache implementations
│   └── kv.py                # Multiple KV cache backends
├── pybind/                  # C++/CUDA backend
│   └── src/                 # CUTLASS, custom kernels
└── utils/                   # Utilities
    ├── green_ctx.py         # SM partitioning via cuBLAS
    └── prof_marker.py       # Profiling markers
```

## Key Components

### 1. Operation Base Classes (`operations/operation_base.py`)

```python
class Operations:
    """Base class for all operations"""
    def __init__(self, name: str, device: str, nano_idx=None):
        self.inputs: dict[str, IOWrapper] = {}
        self.outputs: dict[str, IOWrapper] = {}
        self.weights: dict[str, WeightWrapper] = {}
        self.stream: torch.cuda.Stream
        self.sm_count: int | None = None     # SM allocation
        self.category = None                  # COMP/MEM/NET
        self.isNanoSplit = False             # Nano-batch flag
        self.nano_ops = []                   # Child nano operations

class Operation_Layer:
    """Per-layer operation instance with scheduling variables"""
    def __init__(self, layer, base_op: Operations):
        self.cuda_event = torch.cuda.Event()
        self.prev_op_layer: list[Operation_Layer] = []
        # Gurobi variables for scheduling
        self.start_time: gp.Var
        self.end_time: gp.Var
        self.duration_map = {}  # (batch_size, sm_count) -> duration
```

Key features:
- Operations track I/O dependencies via `IOWrapper` connections
- Each operation has a `category` (COMP, MEM, NET) for scheduling
- `Operation_Layer` instances hold per-layer CUDA events and Gurobi variables

### 2. Nano-batch Splitting (`core/nanobatchSplit.py`)

```python
def split_nanobatch(op_list, op_nano_info_map, extra_links):
    """Split operations into nano-batches with virtual redistribute ops"""
    for op in op_list:
        if op.name in op_nano_info_map:
            # Create redistribution ops for inputs/outputs
            for key, value in op.inputs.items():
                op_redist = Redist(f"Nano_Dist_{op.name}_{key}", ...)
                op_redist.set_input(value)

            # Create nano-batch copies of the operation
            for info in nano_op_info_list:
                copied_op = op.copy_nano(info.batch_idx)
                copied_op.setBatchSize(info.batch_size)
```

Splitting strategy:
1. Create `Redist` (redistribute) virtual ops for tensor splitting
2. Clone base operation for each nano-batch
3. Wire up dependencies between nano-batches

### 3. Executor (`core/executor.py`)

```python
class Executor:
    def plan_layer_ordering(self):
        """Build execution DAG via topological sort"""
        G = nx.DiGraph()
        for op in self.operations_layers_list:
            G.add_node(op.name, op=op, layer=op.layer)
            # Add dependency edges
            for dep, dep_on_prev_layer in op.prerequisites:
                G.add_edge(prev_op_name, op.name)
        self.ordered_operations = list(nx.topological_sort(G))

    def execute(self, output, main_stream, plan_cuda_graph=False):
        """Execute operations with CUDA event synchronization"""
        for op_name in self.ordered_operations:
            op = self.ordered_graph.nodes[op_name]['op']
            op.wait_cuda_event()   # Wait for dependencies
            op.run()
            op.record_cuda_event() # Signal completion
```

Execution flow:
1. Build NetworkX DAG from operation dependencies
2. Topologically sort operations
3. Execute with CUDA events for cross-stream synchronization
4. Optionally capture to CUDA graph for replay

### 4. SM Partitioning (`utils/green_ctx.py`)

```python
def set_sm_count_target(sm_count: int):
    """Constrain cuBLAS GEMM to use specific SM count"""
    handle = torch.cuda.current_blas_handle()
    nvmath.bindings.cublas.set_sm_count_target(handle, sm_count)
```

This leverages NVIDIA's cuBLAS API to constrain GEMM kernels to a subset of SMs, enabling concurrent execution of different operations on partitioned resources.

### 5. Automated Search (`auto_search/new_search.py`)

Two-stage ILP optimization using Gurobi:

**Stage 1: Operation Ordering**
```python
model_stage_one = gp.Model("pipeline")
# Add timing variables
for layer_op in all_layered_ops:
    layer_op.initVariables(model, full_sm_counts)

# Non-overlapping constraints for same-category ops
for op1, op2 in combinations(same_category_ops, 2):
    model.addConstr(op1.end_time <= op2.start_time + M*(1-seq[op1,op2]))
    model.addConstr(op2.end_time <= op1.start_time + M*seq[op1,op2])

# Dependency constraints
for dep_op in layer_op.prev_op_layer:
    model.addConstr(dep_op.end_time <= layer_op.start_time)

model.setObjective(C_max, GRB.MINIMIZE)
```

**Stage 2: SM Allocation**
```python
model_stage_two = gp.Model("SecondStageOptimization")
# SM choice variables
for op in nano_ops:
    op.p_vars[sm_count] = model.addVar(vtype=GRB.BINARY)

# Resource constraint: overlapping ops must fit in total SMs
model.addConstr(
    op.p_choice + sum(other_op.p_choice * is_overlapping[op, other_op])
    <= full_sm_counts
)
```

### 6. Pipeline Definition (`models/llama3_AutoSearch.py`)

```python
class Pipeline:
    def init_operations(self):
        # Define transformer operations
        self.layerNormAttn = LayerNorm("LayerNormAttn", device)
        self.kqv = GEMM_N_Parallel("KQV", device)
        self.decAttn = DecAttnTorch("DecAttn", device)
        self.pfAttn = PFAttnTorch("PFAttn", device)
        # ... more operations

    def init_dependency(self):
        # Wire up dataflow dependencies
        self.layerNormAttn.outputs["output"] >> self.kqv.inputs["A"]
        self.kqv.outputs["D"] >> self.ropeAppend.inputs["kqv"]
        # ...

    def init_category(self):
        # Categorize operations for scheduling
        self.kqv.set_category(CategoryType.COMP)      # Compute-bound
        self.decAttn.set_category(CategoryType.MEM)   # Memory-bound
        # ...

    def nanobatch_split(self):
        # Define nano-batch configuration
        info = (
            NanoOpInfo(batch_idx=0, batch_size=decode_batch_size),
            NanoOpInfo(batch_idx=1, batch_size=global_batch_size - decode_batch_size)
        )
        op_nanobatch_info_map = {
            "LayerNormAttn": info, "KQV": info, "O": info, ...
        }
```

### 7. KV Cache (`kvcache/kv.py`)

Multiple backends supported:

```python
class KVCacheBatched:
    """FlashAttention without paging"""
    def __init__(self, num_layers, num_heads, head_dim, max_seqlen):
        self.k_cache = [torch.zeros(batch, max_seqlen, heads, dim)
                       for _ in range(num_layers)]

class KVCachevLLM:
    """Paged KV cache compatible with vLLM"""
    def __init__(self, block_size=32):
        self.block_table = torch.arange(max_blocks)
        # Reshape for vLLM's paged attention format

class BatchedDistKVCache:
    """FlashInfer-compatible paged cache"""
    def __init__(self, pool: DistKVPool):
        self.kv_indptr = torch.tensor([0])
        self.kv_indices = torch.empty(max_pages)
        self.kv_last_page_len = torch.tensor([])
```

## Execution Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                        NanoFlow Execution                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. Pipeline Initialization                                          │
│     ┌──────────────┐                                                 │
│     │ init_ops()   │─► Create operation objects                      │
│     │ init_deps()  │─► Wire I/O connections                          │
│     │ init_cat()   │─► Set COMP/MEM/NET categories                   │
│     └──────────────┘                                                 │
│            │                                                          │
│            ▼                                                          │
│  2. Offline Profiling (optional)                                     │
│     ┌──────────────┐                                                 │
│     │ profile_all()│─► Measure (batch_size, sm_count) → latency      │
│     └──────────────┘                                                 │
│            │                                                          │
│            ▼                                                          │
│  3. Auto Search (optional)                                           │
│     ┌──────────────────────────────────────────────────────┐        │
│     │  Stage 1: Ordering                                    │        │
│     │  ┌─────────┐  ┌─────────┐  ┌─────────┐              │        │
│     │  │ Topo    │─►│ Gurobi  │─►│ Extract │              │        │
│     │  │ Sort    │  │ ILP     │  │ Order   │              │        │
│     │  └─────────┘  └─────────┘  └─────────┘              │        │
│     │                                                       │        │
│     │  Stage 2: SM Allocation                               │        │
│     │  ┌─────────┐  ┌─────────┐  ┌─────────┐              │        │
│     │  │ Overlap │─►│ Gurobi  │─►│ Extract │              │        │
│     │  │ Detect  │  │ ILP     │  │ SMs     │              │        │
│     │  └─────────┘  └─────────┘  └─────────┘              │        │
│     └──────────────────────────────────────────────────────┘        │
│            │                                                          │
│            ▼                                                          │
│  4. Nano-batch Split                                                 │
│     ┌──────────────────────────────────────────────────────┐        │
│     │  Global Batch (2048 tokens)                          │        │
│     │  ┌────────────────┬───────────────────────────┐     │        │
│     │  │  Nano-batch 0  │       Nano-batch 1        │     │        │
│     │  │  (640 decode)  │  (1408 prefill tokens)    │     │        │
│     │  └────────────────┴───────────────────────────┘     │        │
│     └──────────────────────────────────────────────────────┘        │
│            │                                                          │
│            ▼                                                          │
│  5. Execution                                                         │
│     ┌──────────────────────────────────────────────────────┐        │
│     │  Stream 0 (COMP)    Stream 1 (MEM)                   │        │
│     │  ┌─────────────┐    ┌─────────────┐                 │        │
│     │  │  KQV_0      │    │             │                 │        │
│     │  │  (72 SMs)   │    │ DecAttn_0   │  ◄─ Overlap    │        │
│     │  │  KQV_1      │    │ (36 SMs)    │                 │        │
│     │  └─────────────┘    └─────────────┘                 │        │
│     │        │                  │                          │        │
│     │        └──────┬───────────┘                          │        │
│     │               ▼                                      │        │
│     │         CUDA Event Sync                              │        │
│     └──────────────────────────────────────────────────────┘        │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## Nano-batching Pipeline Visualization

```
Time →

Stream 0 (COMP - Compute Operations):
┌────────┬────────┬────────┬────────┬────────┬────────┐
│ LN_0   │ KQV_0  │ Rope_0 │ PFAttn │  O_0   │ LN_FFN │
│ 72 SMs │ 72 SMs │ 72 SMs │ 72 SMs │ 72 SMs │ 72 SMs │
├────────┼────────┼────────┼────────┼────────┼────────┤
│ LN_1   │ KQV_1  │ Rope_1 │        │  O_1   │   ...  │
│ 72 SMs │ 72 SMs │ 72 SMs │        │ 72 SMs │        │
└────────┴────────┴────────┴────────┴────────┴────────┘
                   │
                   │ Data dependency
                   ▼
Stream 1 (MEM - Memory-bound Operations):
         ┌────────────────────────┐
         │      DecAttn           │
         │       36 SMs           │
         │  (overlaps with COMP)  │
         └────────────────────────┘

Stream 2 (NET - Network Operations):
                            ┌──────────────┐
                            │  AllReduce   │
                            │    35 SMs    │
                            └──────────────┘
```

## Key Design Decisions

### 1. Operation Categories

```python
class CategoryType(Enum):
    COMP = "compute"  # Dense GEMMs: KQV, O, UG, D
    MEM = "memory"    # Decode attention (GEMV)
    NET = "network"   # AllGather, AllReduce
```

Operations in the same category run sequentially; different categories can overlap.

### 2. Dependency Tracking

```python
# IOWrapper connection syntax
self.kqv.outputs["D"] >> self.ropeAppend.inputs["kqv"]

# Results in dependency edge in execution DAG
```

### 3. CUDA Event Synchronization

```python
def record_cuda_event(self):
    if self.is_depended_on:
        self.cuda_event.record(self.stream)

def wait_cuda_event(self):
    for op_layer in self.prev_op_layer:
        if self.stream != op_layer.stream:
            self.stream.wait_event(op_layer.cuda_event)
```

### 4. SM Allocation Strategy

Non-linear utilization curve justifies partitioning:
- Network kernel: 35/108 SMs → 92% peak performance
- GEMM+GEMV concurrent without SM control → 2.5× GEMM slowdown
- Partitioned execution → Near-linear scaling

## Comparison with SGLang/vLLM

| Feature | NanoFlow | SGLang | vLLM |
|---------|----------|--------|------|
| Parallelism | Intra-device | Inter-op | Inter-request |
| Batching | Nano-batch | Chunked | Continuous |
| KV Cache | Multiple backends | RadixAttention | PagedAttention |
| Scheduling | ILP-optimized | Overlap | FCFS |
| SM Control | Yes | No | No |
| Focus | Throughput | Latency/Throughput | Memory efficiency |

## Implementation Notes

1. **Profiling Required**: NanoFlow needs offline profiling to build the `duration_map` for optimization

2. **Gurobi Dependency**: The automated search uses Gurobi ILP solver (requires license for large problems)

3. **Hardware Support**: SM partitioning via cuBLAS requires NVIDIA GPUs with `cublas.set_sm_count_target()` support

4. **FlashInfer Integration**: Uses FlashInfer's "green context" for SM-constrained attention kernels

5. **CUDA Graph Support**: Operations can be captured to CUDA graphs for reduced launch overhead

## References

- [NanoFlow Paper (arXiv:2408.12757)](https://arxiv.org/abs/2408.12757)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [MSCCL++](https://github.com/microsoft/mscclpp)
