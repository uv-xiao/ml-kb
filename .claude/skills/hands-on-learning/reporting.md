# Process-Focused Report Generation

## Philosophy

Reports should tell a **story** of what happens during execution, not just summarize final numbers.

**Bad report**: "Throughput: 1000 tokens/sec, Latency: 50ms"

**Good report**: "Here's what happens when a batch of 8 requests goes through the system..."

## Report Structure

```markdown
# Hands-On Learning Report: [Project Name]

## Executive Summary
[2-3 sentences: What we learned, key insight, recommendation]

## Environment
[From environment.md - GPU, topology, software]

## Codebase Analysis Summary
[Key features from pre-analysis that informed experiments]

## Goal Interpretation
**Original Goal**: [What user asked for]
**Interpreted As**: [Specific questions we answered]

## Actions Log
[Complete record of modifications, commands, analysis steps for reproducibility]

## Execution Story
[The main body - process details, events, timeline]

## Visualizations
[Flame graphs, timelines, charts that illustrate key findings]

## Hardware Behavior Analysis
[SM, register, SMEM, TC, memory utilization]

## Observations and Insights
[What we learned, surprises, patterns]

## Reproduction Guide
[Step-by-step instructions to reproduce key findings]

## Appendix
[Raw data, full command outputs, diffs, profiling artifacts]
```

## The Execution Story

This is the most important section. It tells WHAT HAPPENS step by step.

### Template: Single Request Execution Story

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    EXECUTION STORY: Single Inference Request                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Configuration:                                                              │
│  • Model: Llama-3-8B (32 layers, 32 heads, head_dim=128)                    │
│  • Input: 512 tokens, Output: 128 tokens                                     │
│  • Backend: FlashInfer FA3 on H100                                           │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════ │
│  PHASE 1: Prefill (Processing 512 input tokens)                              │
│  ═══════════════════════════════════════════════════════════════════════════ │
│                                                                              │
│  Time 0.0ms: Request arrives                                                 │
│  ├── CPU: Tokenization (2.1ms)                                               │
│  │         → 512 tokens, IDs: [128000, 791, 2822, ...]                       │
│  │                                                                           │
│  ├── CPU: Plan phase (0.3ms)                                                 │
│  │         → FlashInfer computes work distribution                           │
│  │         → 4 Q tiles × 32 heads = 128 work items                           │
│  │         → Distributed across 108 SMs                                      │
│  │                                                                           │
│  Time 2.4ms: GPU execution begins                                            │
│  │                                                                           │
│  ├── Kernel: embedding_lookup (0.08ms)                                       │
│  │         → SM util: 32%, Memory BW: 78%                                    │
│  │         → Memory-bound, simple lookup                                     │
│  │                                                                           │
│  ├── LAYER 0-31 LOOP (×32 iterations, 1.2ms each = 38.4ms total)            │
│  │   │                                                                       │
│  │   ├── Kernel: rms_norm (0.02ms)                                          │
│  │   │         → SM: 25%, Mem: 85% (memory-bound)                           │
│  │   │                                                                       │
│  │   ├── Kernel: qkv_proj_gemm (0.15ms)                                     │
│  │   │         → SM: 82%, TC: 48%                                           │
│  │   │         → Shape: [1, 512, 4096] × [4096, 12288]                      │
│  │   │                                                                       │
│  │   ├── Kernel: flash_attention_fwd (0.45ms) ← MOST TIME HERE             │
│  │   │         → SM: 68%, TC: 35%, Mem: 72%                                 │
│  │   │         → Warp specialization active:                                │
│  │   │           • Warp 0: TMA producer (12% of cycles)                    │
│  │   │           • Warp 1-7: Tensor core consumers (88%)                   │
│  │   │         → Barrier waits: 0.02ms (4% overhead)                        │
│  │   │         → Tile schedule: 128 tiles across 108 SMs                    │
│  │   │         → Load imbalance: 1.08x (good)                               │
│  │   │                                                                       │
│  │   ├── Kernel: o_proj_gemm (0.12ms)                                       │
│  │   │         → SM: 80%, TC: 45%                                           │
│  │   │                                                                       │
│  │   ├── Kernel: mlp_fused (0.35ms)                                         │
│  │   │         → gate_up + SiLU + down fused                                │
│  │   │         → SM: 85%, TC: 52%                                           │
│  │   │                                                                       │
│  │   └── KV Cache: Updated layer 0's cache                                  │
│  │               → 512 tokens × 2 × 128 dim = 128KB per layer               │
│  │                                                                           │
│  ├── Kernel: final_rms_norm (0.02ms)                                        │
│  │                                                                           │
│  └── Kernel: lm_head_gemm (0.18ms)                                          │
│            → Shape: [1, 512, 4096] × [4096, 128000]                         │
│            → SM: 75%, TC: 40%                                               │
│                                                                              │
│  Time 41.0ms: First token sampled                                            │
│  └── Sampling: top_p=0.9, temperature=0.7                                    │
│      → Next token: "The" (ID: 578)                                          │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════ │
│  PHASE 2: Decode (Generating 128 tokens, one at a time)                      │
│  ═══════════════════════════════════════════════════════════════════════════ │
│                                                                              │
│  Time 41.0ms: Decode loop begins                                             │
│                                                                              │
│  ITERATION 1 (decode token 1):                                               │
│  ├── Input: 1 token ("The")                                                  │
│  ├── Kernel execution pattern changes:                                       │
│  │   • qkv_proj: [1, 1, 4096] × [4096, 12288] - now GEMV-like               │
│  │   • attention: Query 1 token, KV cache 513 tokens                        │
│  │                                                                           │
│  ├── Kernel: flash_attention_decode (0.15ms) ← Different kernel!           │
│  │         → SM: 45%, Mem: 82%                                              │
│  │         → Now memory-bound (reading KV cache)                            │
│  │         → No warp specialization (all warps do same thing)               │
│  │                                                                           │
│  └── Total iteration: 8.2ms                                                  │
│                                                                              │
│  ITERATION 2-128: Similar pattern                                            │
│  ├── Observation: ITL slowly increases                                       │
│  │   • Iter 1:   8.2ms (KV len: 513)                                        │
│  │   • Iter 64:  9.1ms (KV len: 576)                                        │
│  │   • Iter 128: 10.3ms (KV len: 640)                                       │
│  │   → Linear growth due to KV cache read                                   │
│  │                                                                           │
│  └── Memory: KV cache grows from 4.2GB to 4.8GB                             │
│                                                                              │
│  Time 1150ms: Generation complete                                            │
│  └── Output: "The quick brown fox jumps over the lazy dog..."              │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════ │
│  SUMMARY                                                                     │
│  ═══════════════════════════════════════════════════════════════════════════ │
│                                                                              │
│  TTFT: 41.0ms                                                                │
│  ├── Tokenization: 2.1ms (5%)                                               │
│  ├── Prefill compute: 38.6ms (94%)                                          │
│  └── Sampling: 0.3ms (1%)                                                   │
│                                                                              │
│  Decode: 1109ms for 128 tokens                                              │
│  ├── Average ITL: 8.7ms                                                     │
│  ├── ITL range: 8.2ms - 10.3ms                                              │
│  └── Trend: Linear increase with KV length                                  │
│                                                                              │
│  Key Observation:                                                            │
│  • Prefill: Compute-bound (TC util 35-52%)                                   │
│  • Decode: Memory-bound (HBM util 82%, TC util 15%)                         │
│  • Transition: Kernel selection changes at decode                           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Template: Batch Execution Story

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    EXECUTION STORY: Continuous Batching                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Scenario: 8 concurrent requests, arriving over 500ms                        │
│                                                                              │
│  Time 0ms:    Request R0 arrives (input: 256 tokens)                        │
│  Time 50ms:   Request R1 arrives (input: 512 tokens)                        │
│  Time 100ms:  Request R2 arrives (input: 128 tokens)                        │
│  ...                                                                         │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════ │
│  SCHEDULER DECISIONS                                                         │
│  ═══════════════════════════════════════════════════════════════════════════ │
│                                                                              │
│  Iteration 0: Batch = {R0}                                                   │
│  ├── Running: [R0-prefill]                                                   │
│  ├── KV Cache: R0 allocated 16 blocks                                        │
│  └── GPU Memory: 4.2GB / 80GB                                               │
│                                                                              │
│  Iteration 1: Batch = {R0, R1}                                               │
│  ├── Running: [R0-decode, R1-prefill]                                        │
│  ├── Observation: Chunked prefill for R1 (256 tokens/chunk)                 │
│  │   → Reason: R0's decode shouldn't wait too long                          │
│  ├── KV Cache: R0: 17 blocks, R1: 16 blocks (partial)                        │
│  └── GPU Memory: 8.4GB / 80GB                                               │
│                                                                              │
│  Iteration 5: Batch = {R0, R1, R2, R3, R4}                                   │
│  ├── Running: [R0-decode, R1-decode, R2-decode, R3-prefill, R4-prefill]     │
│  ├── Observation: Mixed prefill-decode batch                                │
│  │   → Prefill tokens: 384                                                  │
│  │   → Decode tokens: 3                                                     │
│  │   → Kernel: Uses prefill kernel (more efficient for mixed)               │
│  └── GPU Memory: 24.1GB / 80GB                                              │
│                                                                              │
│  Iteration 20: R0 completes, R5 enters                                       │
│  ├── Event: R0 generates EOS token                                          │
│  ├── Action: R0's KV blocks freed (32 blocks returned)                      │
│  ├── Action: R5 immediately starts prefill using freed blocks               │
│  └── Observation: No memory fragmentation (contiguous reuse)                │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════ │
│  KERNEL BEHAVIOR UNDER BATCHING                                              │
│  ═══════════════════════════════════════════════════════════════════════════ │
│                                                                              │
│  Attention kernel with batch_size=5:                                         │
│  ├── Work distribution:                                                      │
│  │   • Request 0: 17 KV pages → 17 tiles                                    │
│  │   • Request 1: 32 KV pages → 32 tiles                                    │
│  │   • Request 2: 8 KV pages → 8 tiles                                      │
│  │   • Total: 57 tiles across 108 SMs                                       │
│  │                                                                           │
│  ├── Load imbalance issue:                                                   │
│  │   • Without balancing: SM 0-31 finish early, SM 32-57 overloaded         │
│  │   • FlashInfer's work_indptr: Redistributes tiles                        │
│  │   • After balancing: All SMs finish within 1.05x of each other           │
│  │                                                                           │
│  └── SM utilization: 72% average (vs 45% without balancing)                 │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════ │
│  MEMORY TIMELINE                                                             │
│  ═══════════════════════════════════════════════════════════════════════════ │
│                                                                              │
│  GPU Memory Usage Over Time:                                                 │
│                                                                              │
│  80GB ┤                                                                      │
│       │                                                                      │
│  60GB ┤            ┌────────────┐                                           │
│       │        ┌───┘            └───┐  Peak: 58GB                           │
│  40GB ┤    ┌───┘                    └───────────────                        │
│       │ ┌──┘                              ↑                                 │
│  20GB ┤─┘                           R0 completes,                           │
│       │                             memory freed                             │
│   0GB ┼────────────────────────────────────────────→ Time                   │
│       0    500   1000  1500  2000  2500  3000 (ms)                          │
│                                                                              │
│  Memory Events:                                                              │
│  • T=0: Model weights loaded (16GB)                                         │
│  • T=100: R0-R2 KV cache allocated (+12GB)                                  │
│  • T=300: R3-R7 KV cache allocated (+28GB)                                  │
│  • T=800: Peak memory (58GB) - all requests active                          │
│  • T=1200: R0 completes, memory freed (-2GB)                                │
│  • T=2500: All requests complete                                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Visualizations Section

Visual representations make findings more intuitive and memorable. Include relevant visualizations for key insights.

### Timeline / Flame Graph

For showing execution flow and identifying bottlenecks:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    EXECUTION TIMELINE (Flame Graph Style)                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Time (ms)  0      10      20      30      40      50      60      70       │
│             ├───────┼───────┼───────┼───────┼───────┼───────┼───────┤       │
│                                                                              │
│  CPU        ┌──────┐                                           ┌────┐       │
│  (main)     │tokenize                                          │sample      │
│             └──────┘                                           └────┘       │
│                                                                              │
│  GPU        ┌─────────────────────────────────────────────────────────────┐ │
│  Stream 0   │                        Forward Pass                          │ │
│             └─────────────────────────────────────────────────────────────┘ │
│                 │                                                           │
│                 ├──┬──┬───────────┬──┬──┬───────────┬──┬──┬───────────┬──  │
│  Kernels        │em│rn│  attn     │op│rn│  attn     │op│rn│  attn     │... │
│                 └──┴──┴───────────┴──┴──┴───────────┴──┴──┴───────────┴──  │
│                                                                              │
│  Legend:                                                                     │
│  • em = embedding_lookup (0.08ms)                                            │
│  • rn = rms_norm (0.02ms)                                                    │
│  • attn = flash_attention (0.45ms) ← HOTSPOT                                │
│  • op = output_proj (0.12ms)                                                 │
│                                                                              │
│  Observation: Attention dominates (65% of total GPU time)                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Kernel Breakdown Pie Chart

For showing time distribution across kernel types:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    GPU TIME BREAKDOWN                                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                        ┌───────────────┐                                     │
│                   ╱────│  Attention    │────╲                               │
│                 ╱      │    45.2%      │      ╲                             │
│               ╱        └───────────────┘        ╲                           │
│             ╱                                     ╲                         │
│           ╱    ┌──────────┐      ┌──────────┐      ╲                       │
│         ╱      │  MLP     │      │  GEMM    │        ╲                     │
│        │       │  28.1%   │      │  18.5%   │         │                    │
│        │       └──────────┘      └──────────┘         │                    │
│         ╲                                           ╱                       │
│           ╲   ┌────────┐  ┌────────┐  ┌────────┐  ╱                        │
│             ╲ │ Norm   │  │ Other  │  │ Comm   │╱                          │
│               │ 4.1%   │  │ 2.8%   │  │ 1.3%   │                           │
│               └────────┘  └────────┘  └────────┘                           │
│                                                                              │
│  Total GPU time: 38.4ms                                                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Memory Usage Over Time

For showing memory patterns and identifying leaks or spikes:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    GPU MEMORY USAGE TIMELINE                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Memory (GB)                                                                 │
│       │                                                                      │
│  60 ──┤                     ╭──────────────╮                                │
│       │                  ╭──╯              ╰──╮                             │
│  50 ──┤               ╭──╯                    ╰──╮                          │
│       │            ╭──╯                          ╰──                        │
│  40 ──┤         ╭──╯          Peak: 58GB            ╰──                     │
│       │      ╭──╯             at T=800ms               ╰──                  │
│  30 ──┤   ╭──╯                                            ╰──               │
│       │╭──╯                                                  ╰──            │
│  20 ──┼╯  ▲                  ▲                ▲                 ╰──         │
│       │   │                  │                │                    ╰──      │
│  10 ──┤   │ Model loaded     │ All requests   │ R0 completes          ╰──   │
│       │   │ (16GB)           │ active         │ memory freed             ╰──│
│   0 ──┴───┴──────────────────┴────────────────┴─────────────────────────────│
│       0   200   400   600   800   1000  1200  1400  1600  1800  2000 Time(ms)│
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Warp Stall Distribution

For understanding what warps are waiting on:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    WARP STALL REASONS (Horizontal Bar)                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Stall Reason                              % of Stall Cycles                 │
│                                                                              │
│  long_scoreboard   ████████████████████████████████████████████  42%        │
│  (waiting HBM)     ↑ Memory-bound: waiting for global memory loads          │
│                                                                              │
│  barrier           ██████████████████████████████  28%                      │
│  (__syncthreads)   ↑ Expected: warp specialization synchronization          │
│                                                                              │
│  short_scoreboard  ████████████████████  18%                                │
│  (waiting SMEM)    ↑ Some shared memory bank conflicts                      │
│                                                                              │
│  not_selected      ██████████  12%                                          │
│  (scheduler)       ↑ Warps eligible but not picked                          │
│                                                                              │
│  Interpretation:                                                             │
│  • Kernel is memory-bound (long_scoreboard dominant)                         │
│  • Warp specialization overhead acceptable (barrier = 28%)                   │
│  • Shared memory access could be optimized (short_scoreboard = 18%)          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Roofline Position

For showing compute vs memory bound status:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    ROOFLINE ANALYSIS                                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Performance                                                                 │
│  (TFLOPS)     Peak: 989 TFLOPS (FP16 Tensor Core)                           │
│       │                                                                      │
│  1000 ┤────────────────────────────────────────────────────────────────      │
│       │                                             ╱                        │
│   800 ┤                                           ╱                          │
│       │                                         ╱                            │
│   600 ┤                                       ╱                              │
│       │                                     ╱                                │
│   400 ┤                              ★    ╱   ← flash_attention (achieved)   │
│       │                                 ╱         AI = 85 FLOPS/Byte         │
│   200 ┤                          ●    ╱          Perf = 380 TFLOPS           │
│       │                             ╱                                        │
│   100 ┤               ▲           ╱                                          │
│       │                         ╱   Memory Bound     Compute Bound           │
│    50 ┤                       ╱      Region            Region                │
│       │                     ╱                                                │
│    25 ┼───────────────────╱──────────────────────────────────────────────────│
│       1        10        100       1000      10000                          │
│                  Arithmetic Intensity (FLOPS/Byte)                           │
│                                                                              │
│  Legend:                                                                     │
│  ★ flash_attention_fwd    - Near roofline, good efficiency (38% of peak)    │
│  ● decode_attention       - Memory-bound region (as expected)               │
│  ▲ rms_norm               - Memory-bound, simple reduction                  │
│                                                                              │
│  Ridge point at AI = 330 FLOPS/Byte (for H100 HBM3 @ 3TB/s)                 │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Multi-GPU Communication Timeline

For distributed scenarios:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-GPU COMMUNICATION TIMELINE                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Time (ms)   0         5        10        15        20        25            │
│              ├─────────┼─────────┼─────────┼─────────┼─────────┤            │
│                                                                              │
│  GPU 0       ▓▓▓▓▓▓▓▓▓▓░░░░░░░░░▓▓▓▓▓▓▓▓▓▓░░░░░░░░░▓▓▓▓▓▓▓▓▓▓              │
│              │ compute │AllReduce│ compute │AllReduce│ compute │             │
│                                                                              │
│  GPU 1       ▓▓▓▓▓▓▓▓░░░░░░░░░░░▓▓▓▓▓▓▓▓░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓              │
│              │compute │ wait    │compute │ wait    │ compute │              │
│                        ↑         │        ↑                                  │
│  GPU 2       ▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░▓▓▓▓▓▓▓▓▓▓              │
│                                                                              │
│  GPU 3       ▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░▓▓▓▓▓▓▓▓▓▓              │
│              │ slowest │        │ slowest │                                  │
│                        ↓                  ↓                                  │
│                   AllReduce          AllReduce                               │
│                   barrier            barrier                                 │
│                                                                              │
│  Legend: ▓ = Compute active  ░ = Waiting/Communication                      │
│                                                                              │
│  Observation:                                                                │
│  • GPU 1 finishes compute 2ms early → idle during AllReduce setup           │
│  • GPU 3 is the slowest → all others wait for it                            │
│  • Communication overhead: 18% of total time                                 │
│                                                                              │
│  Recommendation: Investigate GPU 3's workload imbalance                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Generating Visualizations from Profiling Data

```bash
# 1. Export nsys data for visualization
nsys export -t sqlite flashinfer.nsys-rep
# Then query with SQL for custom analysis

# 2. Use Perfetto for interactive flame graphs
# Upload .json trace to https://ui.perfetto.dev/

# 3. Use ncu GUI for roofline
ncu-ui flashinfer.ncu-rep
# Navigate to "Roofline" section

# 4. PyTorch TensorBoard integration
tensorboard --logdir=./profiler_logs
# Open "PyTorch Profiler" tab for flame graphs

# 5. Export Chrome trace for dev tools
prof.export_chrome_trace("trace.json")
# Open in chrome://tracing/
```

### Visualization Principles

1. **Match visualization to insight**: Flame graph for hotspots, bar chart for distribution, timeline for sequence
2. **Include annotations**: Mark key events, hotspots, anomalies directly on the visualization
3. **Show context**: Include baselines, expected values, or comparisons
4. **Use ASCII for reports**: Makes reports self-contained and reproducible in any environment
5. **Reference tools for interactive exploration**: Point users to Perfetto, TensorBoard, ncu-ui for deeper dives

## Hardware Behavior Section

### Template: Kernel Analysis Table

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    KERNEL HARDWARE ANALYSIS                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Kernel: flash_attention_fwd_sm90                                            │
│  Launch Config: grid=(128, 1, 1), block=(256, 1, 1), smem=98304             │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  OCCUPANCY ANALYSIS                                                    │  │
│  ├────────────────────────────────────────────────────────────────────────┤  │
│  │  Theoretical occupancy: 50% (limited by SMEM)                          │  │
│  │  Achieved occupancy:    47%                                            │  │
│  │                                                                        │  │
│  │  Limiters:                                                             │  │
│  │  • Registers: 96 per thread → 25% of max                              │  │
│  │  • SMEM: 96KB per block → 42% of 227KB max                            │  │
│  │  • Threads: 256 per block → 12.5% of 2048 max                         │  │
│  │                                                                        │  │
│  │  Conclusion: SMEM is the occupancy limiter                             │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  COMPUTE UTILIZATION                                                   │  │
│  ├────────────────────────────────────────────────────────────────────────┤  │
│  │  SM Throughput:        68% of peak                                     │  │
│  │  Tensor Core Active:   35% of cycles                                   │  │
│  │  FP32 Pipe Active:     12% of cycles                                   │  │
│  │                                                                        │  │
│  │  Interpretation:                                                       │  │
│  │  • Good overall utilization                                            │  │
│  │  • TC could be higher (data feeding issue?)                            │  │
│  │  • FP32 for softmax/exp operations                                     │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  MEMORY BEHAVIOR                                                       │  │
│  ├────────────────────────────────────────────────────────────────────────┤  │
│  │  HBM Throughput:       72% of peak (2.1 TB/s achieved)                 │  │
│  │  L2 Hit Rate:          45%                                             │  │
│  │  SMEM Throughput:      85% of peak                                     │  │
│  │  Bank Conflicts:       2.3 conflicts/warp (low)                        │  │
│  │                                                                        │  │
│  │  Interpretation:                                                       │  │
│  │  • Near-peak HBM utilization (memory-bound as expected)                │  │
│  │  • Moderate L2 reuse (tiling helps)                                    │  │
│  │  • SMEM well-utilized (double buffering effective)                     │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  WARP STALL REASONS                                                    │  │
│  ├────────────────────────────────────────────────────────────────────────┤  │
│  │  wait_barrier:         28% of stall cycles                             │  │
│  │  long_scoreboard:      42% of stall cycles                             │  │
│  │  short_scoreboard:     18% of stall cycles                             │  │
│  │  not_selected:         12% of stall cycles                             │  │
│  │                                                                        │  │
│  │  Interpretation:                                                       │  │
│  │  • wait_barrier: Warp specialization sync overhead                     │  │
│  │  • long_scoreboard: Waiting for HBM loads (expected)                   │  │
│  │  • Overall: Healthy stall distribution                                 │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Observations Section

### Template: Key Insights

```markdown
## Observations and Insights

### 1. Prefill vs Decode Dichotomy

**Observation**: The system uses completely different kernels for prefill and decode.

**Details**:
- Prefill: `flash_attention_fwd_sm90` (compute-bound, TC active)
- Decode: `flash_attention_decode_sm90` (memory-bound, GEMV-style)

**Implication**: Optimization strategies differ completely between phases.

### 2. Load Balancing Effectiveness

**Observation**: FlashInfer's work_indptr significantly improves SM utilization.

**Quantified**:
- Without balancing: 45% average SM utilization, 2.3x load imbalance
- With balancing: 72% average SM utilization, 1.08x load imbalance

**How it works**: Binary search finds optimal tiles-per-SM, then work_indptr assigns work ranges to each SM.

### 3. Warp Specialization Overhead

**Observation**: 28% of warp stall cycles are barrier waits.

**Trade-off**:
- Cost: Barrier synchronization overhead
- Benefit: TMA and tensor cores operate concurrently
- Net: 15% speedup vs non-specialized version (measured)

### 4. Memory Scaling Behavior

**Observation**: ITL increases linearly with KV cache length.

**Measured**:
- ITL = 6.5ms + 0.006ms × kv_length
- At 1024 tokens: 12.6ms
- At 4096 tokens: 31.1ms

**Bottleneck**: HBM bandwidth for KV cache reads (decode is memory-bound).

### 5. Surprise Finding

**Observation**: First decode iteration is 20% slower than subsequent ones.

**Root cause**: JIT compilation of decode kernel variant (cached after first call).

**Implication**: Use warmup iteration before benchmarking.
```

## Actions Log Section

The Actions Log records EVERYTHING done during the exploration for full reproducibility.

### Template: Actions Log

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    ACTIONS LOG                                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════ │
│  PHASE 1: Environment Setup                                                  │
│  ═══════════════════════════════════════════════════════════════════════════ │
│                                                                              │
│  Action 1.1: Create isolated environment                                     │
│  ─────────────────────────────────────────                                   │
│  Command:                                                                    │
│  ```bash                                                                     │
│  micromamba create -n flashinfer_exp python=3.11 -y                         │
│  micromamba activate flashinfer_exp                                          │
│  pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124│
│  pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4/        │
│  ```                                                                         │
│  Output: Environment created successfully, torch.cuda.is_available()=True   │
│                                                                              │
│  Action 1.2: Verify GPU access                                               │
│  ─────────────────────────────────────────                                   │
│  Command:                                                                    │
│  ```bash                                                                     │
│  nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv          │
│  ```                                                                         │
│  Output:                                                                     │
│  ```                                                                         │
│  name, memory.total [MiB], compute_cap                                       │
│  NVIDIA H100 80GB HBM3, 81559 MiB, 9.0                                       │
│  ```                                                                         │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════ │
│  PHASE 2: Code Modifications for Profiling                                   │
│  ═══════════════════════════════════════════════════════════════════════════ │
│                                                                              │
│  Action 2.1: Add NVTX markers to attention kernel                            │
│  ─────────────────────────────────────────                                   │
│  File: flashinfer/python/flashinfer/decode.py                                │
│  Modification:                                                               │
│  ```diff                                                                     │
│  @@ -145,6 +145,8 @@ def single_decode_with_kv_cache(                        │
│       kv_layout: str = "NHD",                                                │
│   ) -> torch.Tensor:                                                         │
│  +    import nvtx                                                            │
│  +    with nvtx.annotate("single_decode_attention", color="blue"):           │
│           return _kernels.single_decode_with_kv_cache(                       │
│               q, k, v, kv_layout, pos_encoding_mode, ...                     │
│           )                                                                  │
│  ```                                                                         │
│  Purpose: Enable NVTX tracing for decode attention                           │
│                                                                              │
│  Action 2.2: Inject timing instrumentation                                   │
│  ─────────────────────────────────────────                                   │
│  File: benchmark/bench_attention.py (new file)                               │
│  Content:                                                                    │
│  ```python                                                                   │
│  import torch                                                                │
│  import flashinfer                                                           │
│  from torch.cuda import Event                                                │
│                                                                              │
│  def timed_decode(q, k, v, num_iters=100):                                   │
│      start = Event(enable_timing=True)                                       │
│      end = Event(enable_timing=True)                                         │
│      torch.cuda.synchronize()                                                │
│      start.record()                                                          │
│      for _ in range(num_iters):                                              │
│          out = flashinfer.single_decode_with_kv_cache(q, k, v)               │
│      end.record()                                                            │
│      torch.cuda.synchronize()                                                │
│      return start.elapsed_time(end) / num_iters                              │
│  ```                                                                         │
│  Purpose: Accurate kernel timing with CUDA events                            │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════ │
│  PHASE 3: Profiling Runs                                                     │
│  ═══════════════════════════════════════════════════════════════════════════ │
│                                                                              │
│  Action 3.1: System-wide trace with nsys                                     │
│  ─────────────────────────────────────────                                   │
│  Command:                                                                    │
│  ```bash                                                                     │
│  nsys profile -o flashinfer_decode \                                         │
│      --trace=cuda,nvtx \                                                     │
│      --cuda-memory-usage=true \                                              │
│      python benchmark/bench_attention.py \                                   │
│          --batch-size 8 --seq-len 1024 --head-dim 128                        │
│  ```                                                                         │
│  Output file: flashinfer_decode.nsys-rep (142 MB)                            │
│  Duration: 45 seconds                                                        │
│                                                                              │
│  Action 3.2: Kernel-level analysis with ncu                                  │
│  ─────────────────────────────────────────                                   │
│  Command:                                                                    │
│  ```bash                                                                     │
│  ncu --set full \                                                            │
│      --kernel-name ".*decode.*" \                                            │
│      --launch-skip 10 --launch-count 5 \                                     │
│      -o flashinfer_decode_ncu \                                              │
│      python benchmark/bench_attention.py --batch-size 8                      │
│  ```                                                                         │
│  Output file: flashinfer_decode_ncu.ncu-rep (28 MB)                          │
│  Kernels profiled: 5 instances of `BatchDecodeWithPagedKVCacheKernel`        │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════ │
│  PHASE 4: Analysis Steps                                                     │
│  ═══════════════════════════════════════════════════════════════════════════ │
│                                                                              │
│  Action 4.1: Extract timeline from nsys                                      │
│  ─────────────────────────────────────────                                   │
│  Command:                                                                    │
│  ```bash                                                                     │
│  nsys stats flashinfer_decode.nsys-rep \                                     │
│      --report cuda_gpu_kern_sum > kernel_summary.txt                         │
│  ```                                                                         │
│  Key finding: Attention kernel accounts for 68% of GPU time                  │
│                                                                              │
│  Action 4.2: Extract metrics from ncu                                        │
│  ─────────────────────────────────────────                                   │
│  Command:                                                                    │
│  ```bash                                                                     │
│  ncu --import flashinfer_decode_ncu.ncu-rep \                                │
│      --csv --page raw > metrics.csv                                          │
│  ```                                                                         │
│  Key metrics extracted: SM throughput, DRAM BW, TC utilization               │
│                                                                              │
│  Action 4.3: Compare configurations                                          │
│  ─────────────────────────────────────────                                   │
│  Commands:                                                                   │
│  ```bash                                                                     │
│  # Run A: seq_len=512                                                        │
│  python benchmark/bench_attention.py --seq-len 512 > results_512.txt         │
│                                                                              │
│  # Run B: seq_len=2048                                                       │
│  python benchmark/bench_attention.py --seq-len 2048 > results_2048.txt       │
│                                                                              │
│  # Compare                                                                   │
│  diff results_512.txt results_2048.txt                                       │
│  ```                                                                         │
│  Finding: 4x sequence length → 3.8x latency (near-linear scaling)            │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════ │
│  PHASE 5: Cleanup                                                            │
│  ═══════════════════════════════════════════════════════════════════════════ │
│                                                                              │
│  Action 5.1: Revert code modifications                                       │
│  ─────────────────────────────────────────                                   │
│  Command:                                                                    │
│  ```bash                                                                     │
│  git checkout -- flashinfer/python/flashinfer/decode.py                      │
│  rm benchmark/bench_attention.py                                             │
│  ```                                                                         │
│                                                                              │
│  Action 5.2: Archive artifacts                                               │
│  ─────────────────────────────────────────                                   │
│  Command:                                                                    │
│  ```bash                                                                     │
│  mkdir -p profiling_artifacts/$(date +%Y%m%d)                                │
│  mv *.nsys-rep *.ncu-rep *.txt *.csv profiling_artifacts/$(date +%Y%m%d)/    │
│  ```                                                                         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Actions Log Principles

1. **Record EVERY command**: Include full command with all arguments
2. **Capture outputs**: At least summarize key output, include full output in Appendix
3. **Document code changes**: Use diff format for modifications
4. **Note file paths**: Always use absolute or project-relative paths
5. **Include timing**: How long each step took
6. **Track dependencies**: What needs to happen before what

## Reproduction Guide Section

This section distills the Actions Log into a step-by-step guide for reproducing key findings.

### Template: Reproduction Guide

```markdown
## Reproduction Guide

### Prerequisites

- GPU: NVIDIA H100 or similar (compute capability 9.0)
- CUDA: 12.4+
- Python: 3.11+

### Setup (One-time)

```bash
# 1. Clone and setup
git clone https://github.com/flashinfer-ai/flashinfer.git
cd flashinfer
git checkout v0.1.6  # Specific version used

# 2. Create environment
micromamba create -n flashinfer_exp python=3.11 -y
micromamba activate flashinfer_exp

# 3. Install dependencies
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4/
pip install nvtx
```

### Reproducing Finding 1: Prefill vs Decode Kernel Difference

**What we observed**: System uses `flash_attention_fwd_sm90` for prefill,
`flash_attention_decode_sm90` for decode.

**To reproduce**:

```bash
# 1. Create test script
cat > test_kernels.py << 'EOF'
import torch
import flashinfer

# Prefill: multiple query tokens
q_prefill = torch.randn(1, 512, 32, 128, device="cuda", dtype=torch.float16)
k = torch.randn(1, 512, 32, 128, device="cuda", dtype=torch.float16)
v = torch.randn(1, 512, 32, 128, device="cuda", dtype=torch.float16)

# Decode: single query token
q_decode = torch.randn(1, 1, 32, 128, device="cuda", dtype=torch.float16)

torch.cuda.cudart().cudaProfilerStart()
# This will use prefill kernel
out_prefill = flashinfer.single_prefill_with_kv_cache(q_prefill, k, v)
# This will use decode kernel
out_decode = flashinfer.single_decode_with_kv_cache(q_decode, k, v)
torch.cuda.cudart().cudaProfilerStop()
EOF

# 2. Profile
nsys profile -o kernel_compare -c cudaProfilerApi python test_kernels.py

# 3. Analyze
nsys stats kernel_compare.nsys-rep --report cuda_gpu_kern_sum
```

**Expected output**: Two different kernels in the summary.

### Reproducing Finding 2: Linear ITL Scaling

**What we observed**: ITL = 6.5ms + 0.006ms × kv_length

**To reproduce**:

```bash
# 1. Create scaling test
cat > test_scaling.py << 'EOF'
import torch
import flashinfer
from torch.cuda import Event

def measure_decode(kv_len, num_iters=100):
    q = torch.randn(1, 1, 32, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(1, kv_len, 32, 128, device="cuda", dtype=torch.float16)
    v = torch.randn(1, kv_len, 32, 128, device="cuda", dtype=torch.float16)

    # Warmup
    for _ in range(10):
        flashinfer.single_decode_with_kv_cache(q, k, v)

    # Measure
    start = Event(enable_timing=True)
    end = Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(num_iters):
        flashinfer.single_decode_with_kv_cache(q, k, v)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_iters

for kv_len in [256, 512, 1024, 2048, 4096]:
    latency = measure_decode(kv_len)
    print(f"KV length: {kv_len:5d}, Latency: {latency:.3f} ms")
EOF

# 2. Run
python test_scaling.py
```

**Expected output**:
```
KV length:   256, Latency: 8.036 ms
KV length:   512, Latency: 9.572 ms
KV length:  1024, Latency: 12.644 ms
KV length:  2048, Latency: 18.788 ms
KV length:  4096, Latency: 31.076 ms
```

### Reproducing Finding 3: Warp Stall Distribution

**What we observed**: 28% barrier, 42% long_scoreboard, 18% short_scoreboard

**To reproduce**:

```bash
# Profile with ncu for stall metrics
ncu --metrics \
    smsp__warps_issue_stalled_barrier_per_issue_active.ratio,\
    smsp__warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
    smsp__warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
    smsp__warps_issue_stalled_not_selected_per_issue_active.ratio \
    --kernel-name ".*decode.*" \
    python test_kernels.py

# Expected: Similar ratios to what we observed
```

### Artifacts Location

All profiling artifacts are archived at:
```
profiling_artifacts/20250109/
├── flashinfer_decode.nsys-rep      # System trace
├── flashinfer_decode_ncu.ncu-rep   # Kernel analysis
├── kernel_summary.txt              # Parsed timeline
├── metrics.csv                     # Raw metrics export
└── test_scripts/                   # Scripts used
    ├── test_kernels.py
    └── test_scaling.py
```
```

## Checklist

- [ ] Executive summary captures key insight
- [ ] Execution story tells step-by-step what happens
- [ ] Timeline includes timing and event correlation
- [ ] Hardware metrics interpreted (not just listed)
- [ ] Kernel analysis covers occupancy, compute, memory, stalls
- [ ] Observations are quantified where possible
- [ ] Surprises and unexpected behaviors noted
- [ ] Implications and recommendations derived
- [ ] **Actions Log complete with all commands and modifications**
- [ ] **Code diffs included for all modifications**
- [ ] **Reproduction Guide tested and verified**
- [ ] **Artifacts archived with clear location**
- [ ] **Visualizations included for key findings**:
  - [ ] Timeline/flame graph showing execution flow
  - [ ] Kernel time breakdown chart
  - [ ] Memory usage timeline (if relevant)
  - [ ] Warp stall distribution (for kernel analysis)
  - [ ] Roofline position (for performance characterization)
  - [ ] Multi-GPU timeline (if distributed)
- [ ] **Visualization tools referenced for interactive exploration**
