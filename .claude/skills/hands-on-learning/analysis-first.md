# Analysis-First Strategy

## Why Analyze Before Running?

Running experiments without understanding the codebase is like shooting in the dark.

**Without analysis**:
- Don't know which kernels matter
- Can't interpret profiling results
- Miss configuration options that affect behavior
- Waste time on irrelevant experiments

**With analysis**:
- Know exactly what to profile
- Understand expected vs actual behavior
- Can correlate code patterns with hardware behavior
- Design targeted, informative experiments

## Pre-Run Analysis Workflow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    ANALYSIS-FIRST WORKFLOW                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐                                                        │
│  │ 1. QUICK SCAN    │  README, docs, entry points                           │
│  └────────┬─────────┘  Time: 5-10 minutes                                    │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────┐                                                        │
│  │ 2. USE SKILLS    │  Run llm-code-analysis or repo-analysis              │
│  └────────┬─────────┘  Time: Generated report                                │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────┐                                                        │
│  │ 3. IDENTIFY      │  Key components, hot paths, config options            │
│  │    FEATURES      │                                                        │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────┐                                                        │
│  │ 4. MAP TO        │  Which features → which hardware behaviors?           │
│  │    HARDWARE      │                                                        │
│  └──────────────────┘                                                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Step 1: Quick Scan

Spend 5-10 minutes understanding the project surface:

```bash
# Entry points
ls -la examples/ benchmarks/ scripts/ 2>/dev/null
grep -r "if __name__" --include="*.py" -l | head -10

# Configuration
grep -r "argparse\|click\|typer" --include="*.py" -l | head -5
cat README.md | head -100

# Key directories
find . -name "*.cuh" -o -name "*.cu" | head -20  # CUDA kernels
find . -name "*kernel*" -o -name "*attention*" | head -20
```

## Step 2: Use Analysis Skills

### For LLM/ML Projects

Use the **llm-code-analysis** skill:

```
/analyze-llm code-repos/flashinfer
```

This generates:
- Architecture overview with ASCII diagrams
- Component breakdown (attention, MLP, scheduler, etc.)
- Data flow analysis
- Tiling and parallelization strategies
- Memory access patterns

### For General Repositories

Use the **repo-analysis** skill:

```
/analyze-repo code-repos/some-project
```

This generates:
- Directory structure analysis
- Key module identification
- API surface mapping
- Dependency analysis

## Step 3: Identify Features

From the analysis, extract a **feature list**:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    PROJECT FEATURE INVENTORY                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Project: FlashInfer                                                         │
│                                                                              │
│  Core Features:                                                              │
│  ═══════════════                                                             │
│  1. Paged KV-Cache Attention                                                 │
│     - Block-sparse format with indptr/indices                                │
│     - Supports variable-length sequences                                     │
│     - Hardware: Memory-bound, benefits from HBM bandwidth                    │
│                                                                              │
│  2. JIT Kernel Compilation                                                   │
│     - Generates specialized kernels per (dtype, head_dim, ...)               │
│     - Ninja-based compilation with caching                                   │
│     - Hardware: First-call latency, then cached                              │
│                                                                              │
│  3. Plan-Run Pattern                                                         │
│     - CPU scheduling separated from GPU execution                            │
│     - Enables CUDAGraph capture                                              │
│     - Hardware: CPU-GPU overlap, static launch config                        │
│                                                                              │
│  4. Hopper FA3 Kernels                                                       │
│     - TMA for async data movement                                            │
│     - Warp specialization (producer-consumer)                                │
│     - Hardware: TMA units, named barriers, tensor cores                      │
│                                                                              │
│  5. Load-Balanced Scheduling                                                 │
│     - work_indptr distributes tiles across SMs                               │
│     - Binary search for optimal chunk size                                   │
│     - Hardware: SM utilization, load imbalance                               │
│                                                                              │
│  Configuration Options:                                                      │
│  ═══════════════════════                                                     │
│  • page_size: Affects memory fragmentation                                   │
│  • num_stages: Pipeline depth (SMEM vs latency hiding)                       │
│  • use_cuda_graph: Static vs dynamic scheduling                              │
│  • backend: fa2 vs fa3 vs cudnn                                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Step 4: Map Features to Hardware

For each feature, identify the hardware implications:

| Feature | Hardware Concern | What to Measure |
|---------|-----------------|-----------------|
| Paged attention | Memory access pattern | HBM bandwidth, cache hit rate |
| JIT compilation | First-call overhead | Compilation time, cache effectiveness |
| Warp specialization | Warp divergence | Warp stall reasons, occupancy |
| TMA loads | Async unit utilization | TMA throughput, barrier wait time |
| Load balancing | SM utilization | Per-SM work distribution |

## Using Analysis in Planning

The analysis directly informs your plan:

### Example: "Understand FlashInfer attention performance"

**From analysis, we learned**:
- FlashInfer uses plan-run pattern
- Hopper kernels use warp specialization
- Load balancing via work_indptr

**Therefore, we should track**:
1. Plan phase overhead (CPU time)
2. Kernel execution (attention kernel specifically)
3. Warp stall reasons (producer waiting vs consumer waiting)
4. SM work distribution (tiles per SM)

**NOT just**:
- "tokens per second" (too high-level)
- "latency" (doesn't tell us why)

## Checklist

- [ ] Quick scan completed (README, entry points, config)
- [ ] Analysis skill run (llm-code-analysis or repo-analysis)
- [ ] Feature inventory created
- [ ] Features mapped to hardware concerns
- [ ] Specific measurement targets identified
- [ ] Configuration options that affect performance noted
