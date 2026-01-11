# Luminal Analysis: E-Graph Based ML Compiler

**Date:** 2026-01-11
**Repository:** https://github.com/luminal-ai/luminal
**Language:** Rust
**Focus:** E-graph optimization for neural network compilation

---

## Executive Summary

Luminal is a Rust-based ML compiler that uses **e-graphs (equality saturation)** via the `egglog` library to discover optimal kernel implementations. Unlike traditional ML frameworks that rely on handwritten fusion rules, Luminal searches over the space of equivalent programs to find the best implementation.

**Key Innovation:** Multi-level kernel search using e-graph rewriting to automatically discover optimizations like Flash Attention.

---

## 1. Core Architecture

### RISC-Style Primitive Operations

Luminal reduces all neural network operations to just **12 primitive ops**:

```rust
// Unary ops
Log2, Exp2, Sin, Sqrt, Recip

// Binary ops
Add, Mul, Mod, LessThan

// Other ops
SumReduce, MaxReduce, Contiguous
```

This minimal op set enables:
1. Easy backend support (only 12 ops to implement per backend)
2. Powerful rewrite rules (small pattern space)
3. E-graph search tractability

### Compilation Flow

```
┌───────────────────────────────────────────────────────────────────────────┐
│                           LUMINAL COMPILATION                              │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  1. USER CODE                                                             │
│     ┌─────────────────────────────────────────────────────────────────┐  │
│     │ let a = cx.tensor((3, 1));                                      │  │
│     │ let b = cx.tensor((1, 4));                                      │  │
│     │ let c = a.matmul(b).output();                                   │  │
│     └─────────────────────────────────────────────────────────────────┘  │
│                              │                                            │
│                              ▼                                            │
│  2. HLIR GRAPH (primops)                                                  │
│     ┌─────────────────────────────────────────────────────────────────┐  │
│     │ Input → Expand → Mul → SumReduce → Output                       │  │
│     │   ↑                                                              │  │
│     │ Input → Expand ─────────┘                                       │  │
│     └─────────────────────────────────────────────────────────────────┘  │
│                              │                                            │
│                              ▼  build_search_space<Runtime>()             │
│  3. E-GRAPH (egglog)                                                      │
│     ┌─────────────────────────────────────────────────────────────────┐  │
│     │  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐   │  │
│     │  │ Class: Matmul   │  │ Class: KernelMM │  │ Class: BlockMM │   │  │
│     │  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌────────────┐ │   │  │
│     │  │ │Mul+SumReduce│◄├──┤►│ KernelMatmul│◄├──┤►│BlockMatmul │ │   │  │
│     │  │ └─────────────┘ │  │ └─────────────┘ │  │ └────────────┘ │   │  │
│     │  └─────────────────┘  └─────────────────┘  └────────────────┘   │  │
│     │                                                                  │  │
│     │  Equivalent implementations in same e-class                      │  │
│     └─────────────────────────────────────────────────────────────────┘  │
│                              │                                            │
│                              ▼  search(runtime, limit)                    │
│  4. LLIR GRAPHS (multiple candidates)                                    │
│     ┌───────────────┐  ┌───────────────┐  ┌───────────────┐             │
│     │ Graph 1       │  │ Graph 2       │  │ Graph 3       │             │
│     │ KernelMatmul  │  │ BlockMatmul   │  │ CublasMatmul  │             │
│     └───────────────┘  └───────────────┘  └───────────────┘             │
│                              │                                            │
│                              ▼  Profile each, select best                 │
│  5. BEST LLIR GRAPH                                                       │
│     ┌─────────────────────────────────────────────────────────────────┐  │
│     │ BlockMatmul (1.2ms) ← Selected as fastest                       │  │
│     └─────────────────────────────────────────────────────────────────┘  │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 2. E-Graph Integration (egglog)

### How E-Graphs Work

E-graphs (equality graphs) are data structures that efficiently represent many equivalent programs simultaneously. Each **e-class** contains multiple **e-nodes** that compute the same value.

```
E-Class: "attention_output"
├── E-Node: Softmax(Q@K^T) @ V          (naive)
├── E-Node: FlashAttention(Q, K, V)     (fused)
└── E-Node: BlockAttention(Q, K, V)     (tiled)
```

### Luminal's egglog Integration

Luminal uses `egglog`, a Datalog-based e-graph implementation:

**File:** `src/egglog_utils/base.egg`

```lisp
; Symbolic algebra for shape expressions
(datatype Expression
    (MNum i64)
    (MVar String)
    (MAdd Expression Expression)
    (MMul Expression Expression)
    (MDiv Expression Expression)
    ...)

; Algebraic rewrite rules
(rewrite (MAdd (MNum a) (MNum b)) (MNum (+ a b)) :ruleset expr)
(rewrite (MMul a (MNum 1)) a :ruleset expr)
(rewrite (MAdd a (MSub b a)) b :ruleset expr)
```

### EgglogOp Trait

Each operation implements `EgglogOp` to define its e-graph representation:

**File:** `src/utils.rs:14-33`

```rust
pub trait EgglogOp: Debug {
    /// Term definition: name and parameter types
    fn term(&self) -> (String, Vec<OpParam>);

    /// Rewrite rules that add this op to e-classes
    fn rewrites(&self) -> Vec<String> { vec![] }

    /// Early rewrites applied before main saturation
    fn early_rewrites(&self) -> Vec<String> { vec![] }

    /// Whether to clean up (delete) this op after extraction
    fn cleanup(&self) -> bool;

    /// Extract concrete op from e-graph
    fn extract(...) -> (LLIROp, Vec<&ENodeId>);
}
```

### Rewrite Rule Example

**File:** `crates/luminal_cuda/src/kernel/ops.rs:47-59`

```rust
impl EgglogOp for KernelMaxReduce {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("KernelMax".to_string(), vec![EList, Expr, Input, EList, Expr, EList, Dty])
    }

    fn rewrites(&self) -> Vec<String> {
        vec!["
(rule
    (
        (= ?a (Max ?out_shape ?iters ?inp ?in_stride ?iter_stride ?out_stride))
        (= ?dty (dtype ?inp))
    )
    (
        (union ?a (KernelMax ?out_shape ?iters ?inp ?in_stride ?iter_stride ?out_stride ?dty))
    )
    :name \"kernel max reduce\"
)".to_string()]
    }
}
```

This rule says: whenever we see a `Max` reduction op, add a `KernelMax` (optimized CUDA kernel) to the same e-class.

---

## 3. Search Mechanism

### Building the Search Space

**File:** `src/graph.rs:181-195`

```rust
pub fn build_search_space<Rt: Runtime + 'static>(&mut self) {
    // Collect all operations (base + runtime-specific)
    let mut ops = Rt::Ops::into_vec();
    ops.extend(<crate::op::Ops as IntoEgglogOp>::into_vec());

    // Convert HLIR graph to egglog program
    let (program, root) = hlir_to_egglog(self);

    // Run egglog saturation (equality saturation)
    self.egraph = Some(run_egglog(&program, &root, &ops, cleanup).unwrap());
    self.ops = Some(ops);
}
```

### Egglog Execution

**File:** `src/graph.rs:441-492`

```rust
fn run_egglog(program, root, ops, cleanup) -> Result<SerializedEGraph, egglog::Error> {
    // Phase 1: Early rewrites (simplification)
    let code = egglog_utils::early_egglog(program, root, ops, cleanup);
    let mut egraph = egglog::EGraph::default();
    egraph.run_program(code)?;

    // Phase 2: Full saturation with all rewrites
    let code = full_egglog(&program, ops, cleanup);
    let mut egraph = egglog::EGraph::default();
    egraph.run_program(code)?;

    // Serialize e-graph for extraction
    Ok(SerializedEGraph::new(&egraph, root_eclasses))
}
```

### Search Loop

**File:** `src/graph.rs:198-278`

```rust
pub fn search<R: Runtime>(&mut self, mut runtime: R, limit: usize) -> R {
    // Extract multiple LLIR graphs from e-graph
    let llir_graphs = egglog_to_llir(self.egraph.as_ref().unwrap(), self.ops.as_ref().unwrap(), limit);

    let mut best_graph = StableGraph::default();
    let mut best_metric: Option<R::ProfileMetric> = None;

    // Profile each candidate
    for (i, llir_graph) in llir_graphs.into_iter().enumerate() {
        let (new_metric, display_metric) = runtime.profile(&llir_graph, &self.dyn_map);

        if best_metric.is_none() || best_metric.as_ref().unwrap().gt(&new_metric) {
            best_metric = Some(new_metric);
            best_graph = llir_graph;
        }
    }

    runtime.load_llir(&best_graph);
    runtime
}
```

---

## 4. Multi-Level Kernel Search (2.0)

From the README:
> We're undergoing a large transition to "2.0", which introduces **multi-level kernel search**. This radically simplifies the compiler stack and allows us to discover complex optimizations entirely automatically while interfacing with high-performance kernel libraries.

### Kernel Levels

The CUDA backend has multiple kernel abstractions:

1. **Kernel Ops** (`crates/luminal_cuda/src/kernel/`)
   - Simple single-element operations
   - Map directly to CUDA threads

2. **Block Ops** (`crates/luminal_cuda/src/block/`)
   - Block-level operations (tiles, reductions)
   - Utilize shared memory

3. **Logical Ops** (`crates/luminal_cuda/src/logical.rs`)
   - High-level logical operations
   - May map to library calls (cuBLAS, cuDNN)

### Rewrite Hierarchy

```
HLIR (primops)
    │
    ├──rewrite──► KernelOps (thread-level)
    │
    ├──rewrite──► BlockOps (block-level, shared mem)
    │
    └──rewrite──► LogicalOps (library calls)
```

Each level adds alternative implementations to e-classes, enabling the search to find the best combination.

---

## 5. Backend Architecture

### CUDA Backend Structure

```
crates/luminal_cuda/
├── src/
│   ├── lib.rs           # Main entry, CudaRuntime
│   ├── runtime.rs       # Execution runtime
│   ├── logical.rs       # High-level ops (ToDevice, FromDevice)
│   ├── kernel/
│   │   ├── mod.rs       # KernelOp trait
│   │   └── ops.rs       # Thread-level kernels
│   └── block/
│       ├── mod.rs       # BlockOp trait
│       └── ops.rs       # Block-level kernels (matmul, attention)
```

### KernelOp Trait

**File:** `crates/luminal_cuda/src/kernel/mod.rs`

```rust
pub trait KernelOp: Debug {
    fn compile(&self, ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>)
        -> (CudaFunction, Arc<CudaModule>, String,
            (Expression, Expression, Expression),  // grid dims
            (Expression, Expression, Expression),  // block dims
            Expression,                            // shared mem
            FxHashMap<char, CudaSlice<u8>>);       // constants

    fn output_size(&self) -> Expression;
}
```

### Kernel Code Generation

Kernels are generated at compile time with symbolic shapes:

```rust
let kernel = format!(r#"
extern "C" {{
    __global__ void reduce_max_k({dtype} *out, const {dtype} *in) {{
        int const_z = blockIdx.x;
        int in_start = {in_index};  // Symbolic expression
        int iters = {iters};        // Resolved at runtime
        ...
    }}
}}"#,
    dtype = cuda_dtype(self.dtype),
    in_index = flatten_mul_strides(&self.out_shape, &self.in_stride).to_kernel(),
    iters = self.iters.to_kernel(),
);
```

---

## 6. Key Features

### Compiler-Based Architecture

From `docs/docs/why.mdx`:

> If you want to know how something is achieved in Luminal, there's a good chance the answer is the same: **compilers**.

- **Devices:** Compilers swap primops with device-specific ops
- **Datatypes:** Compilers insert conversion ops
- **Training:** Autograd compiler builds backward graph
- **Fusion:** Compilers merge elementwise ops

### Kernel Fusion

From `docs/blog/gpu.mdx`:

> Elementwise fusion generates a single kernel that does `out = exp(cos(a))`, so no intermediate reads and writes are needed.

Fusion extends across:
- Unary operations
- Binary operations
- Reshapes, permutes, expands

### Buffer Sharing

At compile time, Luminal analyzes buffer lifetimes and reuses memory:

```
Op1 writes to Buffer A
Op2 reads Buffer A, writes to Buffer B
Op3 reads Buffer B, can reuse Buffer A!
```

### Command Buffer Batching

For Metal/CUDA, multiple kernels are batched into single command buffer launches to reduce overhead.

---

## 7. Comparison with Other Frameworks

| Feature | PyTorch | JAX/XLA | TinyGrad | Luminal |
|---------|---------|---------|----------|---------|
| Execution | Eager | JIT | Lazy | Lazy |
| Optimization | torch.compile | XLA HLO | Pattern matching | E-graph search |
| Ops | ~2000 | ~500 | ~25 | 12 |
| Backends | Many | TPU/GPU | CUDA/Metal | CUDA/Metal |
| Auto-discovery | No | Limited | No | Yes |

### E-Graph Advantages

1. **Automatic Discovery:** Flash Attention-like optimizations emerge from rewrites
2. **Correctness:** E-classes guarantee semantic equivalence
3. **Extensibility:** New backends only need 12 op implementations + rewrites

---

## 8. Example Usage

```rust
use luminal::prelude::*;

// Create compute graph
let mut cx = Graph::new();
let a = cx.tensor((3, 1));
let b = cx.tensor((1, 4));
let c = a.matmul(b).output();

// Build search space with CUDA backend
cx.build_search_space::<CudaRuntime>();

// Search for best implementation (try up to 10 candidates)
let mut rt = cx.search(CudaRuntime::default(), 10);

// Set inputs and execute
rt.set_data(a, vec![1.0, 2.0, 3.0].into());
rt.set_data(b, vec![1.0, 2.0, 3.0, 3.0].into());
rt.execute(&cx.dyn_map);

// Get output
println!("Result: {:?}", rt.get_f32(c));
```

---

## 9. Key Files Reference

| Component | File | Purpose |
|-----------|------|---------|
| Graph | `src/graph.rs` | Main compute graph, search |
| E-graph utils | `src/egglog_utils/mod.rs` | egglog program generation |
| Base rewrites | `src/egglog_utils/base.egg` | Symbolic algebra rules |
| EgglogOp trait | `src/utils.rs:14-33` | Op e-graph interface |
| Primops | `src/op.rs` | 12 primitive operations |
| CUDA kernels | `crates/luminal_cuda/src/kernel/ops.rs` | Thread-level CUDA |
| CUDA blocks | `crates/luminal_cuda/src/block/ops.rs` | Block-level CUDA |
| Shape tracking | `src/shape/symbolic.rs` | Symbolic expressions |
| Serialized e-graph | `src/serialized_egraph.rs` | E-graph extraction |

---

## 10. Megakernel Architecture (The Actual Implementation)

Luminal's megakernel is NOT about e-graph rewriting for kernel fusion. Instead, it's a **persistent kernel with dynamic instruction scheduling** - a single GPU kernel that interprets a work queue of operations.

### Core Concept

From the blog post:
> A megakernel is a single GPU kernel that fuses an entire model's forward pass into one computation unit

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SINGLE MEGAKERNEL LAUNCH                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    GLOBAL WORK QUEUE                              │  │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐           │  │
│  │  │Task 0│ │Task 1│ │Task 2│ │Task 3│ │Task 4│ │ ...  │           │  │
│  │  │RMSNrm│ │Linear│ │Attn  │ │Linear│ │SiLU  │ │      │           │  │
│  │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘           │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼ SMs fetch tasks atomically               │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    SM WORKERS (1 per SM)                          │  │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐     ┌──────┐                │  │
│  │  │ SM 0 │ │ SM 1 │ │ SM 2 │ │ SM 3 │ ... │SM 107│                │  │
│  │  │1024 t│ │1024 t│ │1024 t│ │1024 t│     │1024 t│                │  │
│  │  └──────┘ └──────┘ └──────┘ └──────┘     └──────┘                │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Task Structure

**File:** `crates/luminal_cuda/src/runtime.rs:807-824`

```rust
#[repr(C)]
pub struct Task {
    pub op: i32,                    // Operation type (RMSNorm, Linear, etc.)
    pub range: i32,                 // Number of instances to execute
    pub remaining: i32,             // Countdown for work stealing
    pub in_dep_a_stride: i32,       // Barrier stride for input A
    pub in_dep_a_base: i32,         // Barrier base for input A
    pub in_dep_b_stride: i32,       // Barrier stride for input B
    pub in_dep_b_base: i32,         // Barrier base for input B
    pub in_dep_c_stride: i32,       // Barrier stride for input C
    pub in_dep_c_base: i32,         // Barrier base for input C
    pub out_dep_stride: i32,        // Barrier stride for output
    pub out_dep_base: i32,          // Barrier base for output
    pub source_ptrs: [*const f32; 3], // Input data pointers
    pub out_ptr: *mut f32,          // Output data pointer
    pub payload: PayloadBytes,      // Op-specific metadata
}
```

### Interpreter Kernel

**File:** `crates/luminal_cuda/src/block/interpreter.cu`

```cuda
__global__ void worker_kernel(Task *tasks, int num_tasks,
                              int *head, int *ready, int *queue_lock, ...) {
    __shared__ NextTask nt;
    __shared__ int done;

    while (true) {
        // 1. FETCH NEXT TASK (work-stealing from global queue)
        if (threadIdx.x == 0) {
            done = !fetch_next_task(tasks, num_tasks, head, &nt, queue_lock);
        }
        __syncthreads();
        if (done) break;

        const Task *t = &tasks[nt.task_idx];

        // 2. WAIT FOR INPUT DEPENDENCIES (barrier mechanism)
        if (threadIdx.x == 0) {
            int dep_a = eval_expression(t->in_dep_a_base, 0) + ...;
            atomicAdd(&ready[dep_out], 1);  // Signal "in-flight"

            // Wait until producers complete
            while (atomicAdd(&ready[dep_a], 0) > 0) nanosleep(64);
            while (atomicAdd(&ready[dep_b], 0) > 0) nanosleep(64);
        }
        __syncthreads();

        // 3. EXECUTE OPERATION (dispatch by op type)
        switch (t->op) {
            case RMSNormOp: RMSNorm_function(...); break;
            case LinearOp: Linear_function(...); break;
            case AttentionOp: Attention_function(...); break;
            // ... more ops
        }
        __syncthreads();

        // 4. SIGNAL COMPLETION (decrement barrier)
        if (threadIdx.x == 0) {
            atomicSub(&ready[dep_out], 1);
        }
    }
}
```

### Barrier-Based Synchronization

The key innovation is **increment-then-decrement barriers**:

```
Producer:                          Consumer:
1. atomicAdd(&barrier, 1)          1. Wait: while(barrier > 0)
2. Execute computation             2. Execute computation
3. atomicSub(&barrier, 1)
```

This allows:
- Multiple producers to signal "in-flight" work
- Consumers to wait until ALL producers complete
- No need to know producer count in advance

### Barrier Stride Computation

**File:** `crates/luminal_cuda/src/runtime.rs:828-927`

For operations with different launch dimensions, Luminal computes barrier strides:

```
Matmul A[M,K] @ B[K,N] = C[M,N]

Producer (tile of C):
  launch_range = (M/32, N/32)
  barrier_base = 100
  barrier_stride = [1, M/32]  → barrier = 100 + x + y*(M/32)

Consumer (next layer using C):
  launch_range = (M, 1)
  Wait on barriers where output was written
```

### Symbolic Work Queues

Instead of expanding all task instances:

```
# Traditional: 1024 queue entries for 32x32 matmul tiles
# Luminal: 1 symbolic task with range=(32, 32)

Task {
    op: LinearOp,
    range: 1024,    // (M/32) × (N/32)
    remaining: -1,  // Initialized on first fetch
}
```

SMs atomically decrement `remaining` to claim work units.

### Benefits Eliminated

1. **Kernel Launch Overhead:** Single launch for entire model
2. **Wave Quantization:** SMs finishing early grab next task immediately
3. **Memory Load Bubbles:** Weight loading overlaps with computation epilogue

### BlockOp Trait

**File:** `crates/luminal_cuda/src/block/mod.rs:11-36`

```rust
pub trait BlockOp: Debug + AsAny + EgglogOp {
    fn launch_range(&self) -> Vec<Expression>;      // Grid dimensions
    fn output_size(&self) -> Expression;            // Output buffer size
    fn consumer_barriers_seperate(&self) -> Vec<Vec<bool>>; // Barrier sharing
    fn cuda_op(&self) -> (String, String);          // (struct, function body)
    fn schedule_op(...) -> Vec<u8>;                 // Payload bytes
    fn expressions(&self) -> Vec<Expression>;       // Symbolic expressions
}
```

### Megakernel Compilation Flow

**File:** `crates/luminal_cuda/src/runtime.rs:380-577`

```rust
fn load_llir(&mut self, llir_graph: &LLIRGraph) {
    // 1. Partition graph into block op subgraphs
    let block_subgraphs = partition_marked_convex(llir_graph, &block_ops_in_graph);

    // 2. For each subgraph, compute barrier strides
    let (prod_strides, cons_strides, barrier_bases, n_barriers) =
        get_barrier_strides(llir_graph, &subgraph);

    // 3. Compile interpreter kernel with ops
    let (interpreter, module, expressions, constants) =
        compile_interpreter(&cuda_ctx, &cuda_stream, &block_ops, &expressions);

    // 4. Build task queue
    for node in toposort(&llir_graph, None).unwrap() {
        tasks.push(Task {
            op: op_code,
            range: expressions[&range],
            in_dep_a_base: expressions[&barrier_bases[&sources[0]]],
            // ... more fields
        });
    }

    // 5. Store as ExecutableKernel::Megakernel
    exec_graph.add_node(ExecutableKernel::Megakernel {
        interpreter,
        work_queue: tasks,
        n_barriers,
        // ...
    });
}
```

### Execution

**File:** `crates/luminal_cuda/src/runtime.rs:670-754`

```rust
ExecutableKernel::Megakernel { interpreter, work_queue, n_barriers, .. } => {
    let sm_count = cuda_context.attribute(MULTIPROCESSOR_COUNT);

    // Allocate barriers and upload task queue
    let d_barriers = cuda_stream.alloc_zeros::<i32>(n_barriers);
    let d_tasks = cuda_stream.memcpy_stod(work_queue);
    let d_head = cuda_stream.memcpy_stod(&[0i32]);

    // Launch: 1 block per SM, 1024 threads per block
    let cfg = LaunchConfig {
        grid_dim: (sm_count, 1, 1),
        block_dim: (1024, 1, 1),
        shared_mem_bytes: shared_mem_max / 2,
    };

    lb.arg(&d_tasks);
    lb.arg(&num_tasks);
    lb.arg(&d_head);
    lb.arg(&d_barriers);
    unsafe { lb.launch(cfg) };
}
```

---

## 11. E-Graph vs Megakernel: Different Purposes

| Aspect | E-Graph Search | Megakernel |
|--------|----------------|------------|
| **Purpose** | Find best kernel implementations | Execute all ops in one launch |
| **Granularity** | Per-operation alternatives | Entire model fusion |
| **Decision Time** | Compile time (profiling) | Runtime (work stealing) |
| **Overhead Reduced** | Implementation selection | Kernel launch, sync |

They work together:
1. E-graph finds best BlockOp implementations
2. BlockOps are fused into megakernel work queue
3. Megakernel interpreter executes the queue

---

## 12. Comparison with Mirage MPK

| Feature | Luminal Megakernel | Mirage MPK |
|---------|-------------------|------------|
| **Scheduling** | Global work queue + mutex | Per-worker queues + events |
| **Barriers** | Increment-then-decrement | Event triggers |
| **Work Stealing** | Atomic remaining counter | Scheduler distributes |
| **Compilation** | E-graph search + JIT | Task graph + NVCC |
| **Multi-GPU** | Not yet | NVSHMEM integration |

---

## 13. Conclusion

Luminal combines two powerful techniques:

1. **E-Graph Optimization:** Search over equivalent implementations at compile time
2. **Megakernel Execution:** Single persistent kernel with dynamic work scheduling at runtime

The megakernel eliminates kernel launch overhead and wave quantization by keeping all SMs busy with a shared work queue and barrier-based synchronization. The e-graph ensures each operation in the queue uses the best available implementation.
