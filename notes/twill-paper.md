# Twill: Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs

**Paper**: [arXiv:2512.18134](https://arxiv.org/abs/2512.18134)
**Authors**: Rupanshu Soi, Rohan Yadav, Fredrik Kjolstad, Alex Aiken, Maryam Mehri Dehnavi, Michael Garland, Michael Bauer
**Date**: December 2025

## Core Contribution

Twill is the first system that **automatically derives optimal software pipelining (SWP) and warp specialization (WS) schedules** for iterative GPU programs using constraint solvers, eliminating reliance on brittle heuristics and manual expert tuning.

## Problem Statement

Modern GPUs feature heterogeneous functional units:
- **Tensor Cores**: High-throughput matrix multiplication
- **TMA (Tensor Memory Accelerator)**: Asynchronous data movement (Hopper+)
- **General-purpose CUDA cores**: Other operations

Maximizing hardware utilization requires:
1. **Software Pipelining**: Overlapping loop iterations to hide latency
2. **Warp Specialization**: Assigning different warps to different roles

These are traditionally optimized separately with hand-tuned heuristics. Twill treats them as a **unified constraint satisfaction problem**.

## Visual Explanation

### Software Pipelining (SWP)

Software pipelining overlaps multiple loop iterations to hide memory latency:

```
WITHOUT SOFTWARE PIPELINING (Sequential):
═══════════════════════════════════════════════════════════════════════════════

Iteration 0: │ LOAD A₀ │ LOAD B₀ │ wait... │ COMPUTE C₀ │ STORE C₀ │
Iteration 1:                                              │ LOAD A₁ │ LOAD B₁ │ wait... │ COMPUTE C₁ │
Iteration 2:                                                                              │ LOAD A₂ │ ...

Timeline ─────────────────────────────────────────────────────────────────────────────────────────────▶
             │◀──────── Memory latency exposed ────────▶│


WITH SOFTWARE PIPELINING (3 stages):
═══════════════════════════════════════════════════════════════════════════════

              Prologue         Steady State (overlapped)              Epilogue
             ┌────────┐  ┌─────────────────────────────────────────┐  ┌──────┐
             │        │  │                                         │  │      │
Stage 0:     │LOAD A₀ │  │LOAD A₂ │LOAD A₃ │LOAD A₄ │  ...        │  │      │
(TMA Load)   │LOAD B₀ │  │LOAD B₂ │LOAD B₃ │LOAD B₄ │             │  │      │
             │        │  │        │        │        │             │  │      │
Stage 1:     │LOAD A₁ │  │COMP C₀ │COMP C₁ │COMP C₂ │COMP C₃│... │  │COMP  │
(Compute)    │LOAD B₁ │  │        │        │        │        │   │  │Cₙ₋₁ │
             │        │  │        │        │        │        │   │  │      │
Stage 2:     │        │  │STORE   │STORE   │STORE   │STORE   │...│  │STORE │
(Store)      │        │  │C₀      │C₁      │C₂      │C₃      │   │  │Cₙ    │
             └────────┘  └─────────────────────────────────────────┘  └──────┘

Timeline ─────────────────────────────────────────────────────────────────────────────────────────────▶
                         │◀── Memory latency hidden by overlap ──▶│

Key: In steady state, each cycle issues work from 3 different iterations simultaneously
     - Iteration i:   storing results
     - Iteration i+1: computing
     - Iteration i+2: loading data
```

### Warp Specialization (WS)

Warp specialization assigns different warps to different concurrent roles:

```
WITHOUT WARP SPECIALIZATION (All warps same):
═══════════════════════════════════════════════════════════════════════════════

All 8 Warps: │ LOAD tile │ sync │ COMPUTE │ sync │ LOAD tile │ sync │ COMPUTE │ ...

             Each warp does LOAD then COMPUTE sequentially
             Tensor Cores idle during LOAD, TMA idle during COMPUTE


WITH WARP SPECIALIZATION (Producer-Consumer):
═══════════════════════════════════════════════════════════════════════════════

Warp 0       │ TMA LOAD tile₀ │ TMA LOAD tile₁ │ TMA LOAD tile₂ │ ...
(Producer)   │    to buf[0]   │    to buf[1]   │    to buf[0]   │
             │                │                │                │
             │    signal      │    signal      │    signal      │
             │      │         │      │         │      │         │
             │      ▼         │      ▼         │      ▼         │
Warps 1-7    │    wait   │ TC COMPUTE │    wait   │ TC COMPUTE │ ...
(Consumers)  │           │  buf[0]    │           │  buf[1]    │
             │           │            │           │            │

Timeline ─────────────────────────────────────────────────────────────────────▶

             TMA and Tensor Cores operate CONCURRENTLY
             Producer keeps buffers filled, consumers drain them
```

### Combined SWP + WS Schedule Space

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    TWILL'S UNIFIED OPTIMIZATION SPACE                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  For each operation op in loop body:                                     │
│                                                                          │
│    ┌─────────────────────────────────────────────────────────────────┐   │
│    │                                                                 │   │
│    │   op.stage ∈ {0, 1, 2, ...}     Which pipeline stage?          │   │
│    │        │                                                        │   │
│    │        │    Controls iteration overlap depth                    │   │
│    │        ▼                                                        │   │
│    │   ┌─────────┐  ┌─────────┐  ┌─────────┐                        │   │
│    │   │ Stage 0 │  │ Stage 1 │  │ Stage 2 │  ...                   │   │
│    │   │ (iter i)│  │(iter i-1)│ │(iter i-2)│                       │   │
│    │   └─────────┘  └─────────┘  └─────────┘                        │   │
│    │                                                                 │   │
│    │   op.role ∈ {PRODUCER, CONSUMER_0, CONSUMER_1, ...}            │   │
│    │        │                                                        │   │
│    │        │    Controls which warp group executes                  │   │
│    │        ▼                                                        │   │
│    │   ┌──────────┐  ┌────────────────────────────────┐             │   │
│    │   │ Producer │  │ Consumer Warps                 │             │   │
│    │   │ (TMA)    │  │ (Tensor Core MMA)              │             │   │
│    │   │ Warp 0   │  │ Warps 1-7                      │             │   │
│    │   └──────────┘  └────────────────────────────────┘             │   │
│    │                                                                 │   │
│    │   op.cycle ∈ {0, 1, 2, ...}    When to issue?                  │   │
│    │        │                                                        │   │
│    │        │    Controls exact timing within steady state           │   │
│    │        ▼                                                        │   │
│    │   Cycle: 0    1    2    3    4    5    ...                     │   │
│    │          │    │    │    │    │    │                            │   │
│    │          ▼    ▼    ▼    ▼    ▼    ▼                            │   │
│    │         TMA  MMA  TMA  MMA  MMA  TMA  (example schedule)       │   │
│    │                                                                 │   │
│    └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  CONSTRAINTS enforced by solver:                                         │
│  ─────────────────────────────────────────────────────────────────────   │
│  • Dependencies: op_A.cycle + latency(A) ≤ op_B.cycle  (if A → B)       │
│  • Resources:    count(TMA ops at cycle c) ≤ num_TMA_units              │
│  • Memory:       Σ(buffer sizes) ≤ shared_memory_capacity               │
│  • Sync:         Barriers placed between producer/consumer handoffs     │
│                                                                          │
│  OBJECTIVE: Minimize Initiation Interval (II)                            │
│  ────────────────────────────────────────────                            │
│  II = cycles between starting consecutive iterations in steady state     │
│  Lower II = Higher throughput                                            │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

## Technical Approach

### Formulation as Constraint Satisfaction

```
Given:
  - Loop body with operations {op₁, op₂, ..., opₙ}
  - Dependency graph G = (V, E)
  - Hardware constraints (num_tensor_cores, smem_size, ...)

Find:
  - stage(opᵢ) ∈ [0, num_stages)    # Pipeline stage assignment
  - role(opᵢ) ∈ [0, num_roles)       # Warp role assignment
  - cycle(opᵢ) ∈ [0, max_cycles)     # Issue cycle

Minimize:
  - Initiation Interval (II)         # Steady-state loop throughput
```

### Constraint Categories

1. **Dependency constraints**: Producers complete before consumers
2. **Resource constraints**: Limited functional units per cycle
3. **Memory constraints**: Shared memory doesn't exceed capacity
4. **Synchronization constraints**: Proper barrier placement

## Key Results

- **Automatic rediscovery**: Found expert-crafted Flash Attention schedules for Hopper and Blackwell
- **No heuristics**: Guaranteed optimal solutions within the search space
- **Extensible**: Easily adapts to new GPU architectures by updating constraint definitions

## Relevance to MPMD

Twill demonstrates that **intra-kernel MPMD scheduling** (different warps doing different things) can be optimally derived rather than manually crafted. This complements:

- **Cross-kernel MPMD**: Megakernel task scheduling (triton-distributed)
- **Cross-device MPMD**: SM partitioning (NanoFlow)

## Tags

`#warp-specialization` `#software-pipelining` `#gpu-optimization` `#constraint-solving` `#tensor-cores` `#hopper`
