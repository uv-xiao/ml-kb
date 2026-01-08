# Plan Creation and Requirements Interpretation

## The Art of Interpreting Goals

Users often give vague goals. Your job is to translate them into **specific, measurable tracking requirements**.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    GOAL INTERPRETATION PROCESS                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  User Goal (Vague)                                                           │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  THINK: What could they actually want to know?                          │ │
│  │                                                                         │ │
│  │  • Performance characteristics?                                         │ │
│  │  • Comparison with alternatives?                                        │ │
│  │  • Bottleneck identification?                                           │ │
│  │  • Understanding implementation details?                                │ │
│  │  • Optimization opportunities?                                          │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  CONSULT: Codebase analysis results                                     │ │
│  │                                                                         │ │
│  │  • What are the interesting features of this project?                   │ │
│  │  • What makes it different from alternatives?                           │ │
│  │  • What are the known hot paths?                                        │ │
│  │  • What configuration affects behavior?                                 │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│       │                                                                      │
│       ▼                                                                      │
│  Specific Tracking Requirements                                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Goal Interpretation Examples

### Example 1: "Understand vLLM performance"

**Think**: What aspects of vLLM performance?
- Continuous batching behavior?
- PagedAttention efficiency?
- Prefill vs decode tradeoffs?
- Multi-GPU scaling?

**Consult codebase analysis**:
- vLLM uses PagedAttention with block-based KV cache
- Scheduler implements continuous batching
- Worker uses NCCL for tensor parallelism

**Specific tracking requirements**:
1. **Scheduler behavior**: Batch formation latency, preemption frequency
2. **PagedAttention**: Block table lookup overhead, memory fragmentation
3. **Prefill/decode**: Kernel selection, iteration time distribution
4. **Multi-GPU**: AllReduce time, GPU synchronization gaps

### Example 2: "Profile SGLang RadixAttention"

**Think**: What about RadixAttention specifically?
- Cache hit rates?
- Tree structure efficiency?
- Memory savings vs vLLM?

**Consult codebase analysis**:
- RadixAttention uses prefix tree for KV cache sharing
- Hash-based lookup for cache hits
- Eviction based on reference counting

**Specific tracking requirements**:
1. **Cache operations**: Insert/lookup/evict timing
2. **Hit rate**: Prefix match ratio across requests
3. **Memory**: Actual vs theoretical with sharing
4. **Tree ops**: Node allocation, path traversal

### Example 3: "See how FlashInfer kernels behave"

**Think**: Which kernels? What behavior?
- Attention kernels specifically?
- All operators?
- Different configurations?

**Consult codebase analysis**:
- FlashInfer has decode/prefill/cascade attention
- Hopper-specific FA3 kernels with warp specialization
- JIT compilation with caching

**Specific tracking requirements**:
1. **Kernel selection**: Which kernel variant for which config
2. **Warp behavior**: Producer vs consumer execution patterns
3. **Memory access**: Tiling efficiency, SMEM usage
4. **JIT overhead**: First-call vs cached compilation

## Plan Template

```markdown
# Hands-On Learning Plan: [Project Name]

## Goal Interpretation

**Original Goal**: [User's stated goal]

**Interpreted As**:
- [Specific question 1]
- [Specific question 2]
- [Specific question 3]

**Informed by Codebase Analysis**:
- Key feature: [feature] → tracking [metric]
- Key feature: [feature] → tracking [metric]

## Environment
- GPU: [from environment.md]
- Topology: [NVLink/PCIe info]
- CUDA: [version]
- Model: [target model for experiments]

## Tracking Requirements

### Hardware-Level Tracking

| Metric | Why | How to Measure |
|--------|-----|----------------|
| SM utilization | Load balance | nsys, ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed |
| Register usage | Occupancy impact | ncu --metrics launch__registers_per_thread |
| SMEM usage | Bank conflicts, occupancy | ncu --metrics sm__sass_l1tex_data_bank_conflicts_pipe_lsu |
| Tensor core util | Compute efficiency | ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed |
| Memory bandwidth | HBM utilization | ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed |
| NVLink bandwidth | Communication efficiency | nsys with nvtx markers around AllReduce |

### Process-Level Tracking

| Event | Why | How to Capture |
|-------|-----|----------------|
| Kernel launches | Execution pattern | nsys timeline |
| Barrier waits | Synchronization overhead | nsys with named barriers |
| Memory allocs | Dynamic allocation overhead | torch.cuda.memory_stats() |
| CPU-GPU gaps | Host overhead | nsys CPU/GPU correlation |

### Application-Level Tracking

| Metric | Why | How to Measure |
|--------|-----|----------------|
| Per-layer timing | Identify slow layers | Custom timing injection |
| Batch formation | Scheduler efficiency | Log injection |
| Cache hit/miss | Cache effectiveness | Counter injection |

## Experiments

### Experiment 1: [Name]
**Purpose**: [What question does this answer?]
**Configuration**:
```bash
[exact command]
```
**Profiling**:
```bash
[profiling command]
```
**Expected Observations**:
- [What we expect to see based on analysis]
**Actual Observations**: [filled after running]

### Experiment 2: [Name]
...

## Analysis Plan

After collecting data:
1. Extract [specific metric] from [tool output]
2. Compare against [baseline/expectation]
3. Identify [anomalies/patterns]
4. Correlate [code feature] with [hardware behavior]

## Success Criteria

- [ ] Answered: [specific question 1]
- [ ] Answered: [specific question 2]
- [ ] Identified: [bottleneck/pattern]
- [ ] Understood: [process detail]
```

## Thinking Through Requirements

When creating a plan, ask yourself:

### 1. What Makes This Project Interesting?

From the codebase analysis, identify unique aspects:
- Novel algorithms (e.g., RadixAttention, PagedAttention)
- Optimization techniques (e.g., warp specialization, JIT)
- Architectural decisions (e.g., plan-run pattern)

### 2. What Would an Expert Want to Verify?

An expert would want to see:
- Does the clever algorithm actually help in practice?
- Where are the hidden overheads?
- What are the scaling characteristics?

### 3. What Hardware Behaviors Correspond to Features?

Map features to measurable hardware events:

| Code Feature | Hardware Manifestation |
|--------------|----------------------|
| Paged memory access | Irregular memory patterns, cache misses |
| Warp specialization | Divergent execution, barrier waits |
| JIT compilation | CPU overhead before first kernel |
| Tensor parallelism | AllReduce, GPU sync |
| Continuous batching | Variable kernel launch patterns |

### 4. What Would Surprise Us?

Think about unexpected behaviors:
- Memory usage higher than calculated?
- Certain configs much slower?
- Scaling not linear?

Design experiments to detect surprises.

## Anti-Patterns to Avoid

**DON'T**:
- Run benchmark and report final throughput only
- Profile without knowing what to look for
- Ignore configuration options
- Skip the analysis phase
- Focus only on averages

**DO**:
- Track specific events informed by code analysis
- Profile with targeted metrics
- Vary configurations systematically
- Understand before measuring
- Look at distributions and timelines
