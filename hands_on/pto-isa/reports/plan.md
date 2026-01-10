# PTO-ISA Hands-On Learning Plan

## Overview

**PTO-ISA** (Parallel Tile Operation ISA) is a virtual instruction set architecture for Ascend NPUs that provides a portable, tile-level abstraction for high-performance computing.

**Learning Goals**:
1. Understand how to program and profile PTO-ISA programs
2. Run CPU simulation to validate correctness
3. Identify opportunities for automatic optimization at the abstraction level

**Environment**: CPU simulation only (no Ascend hardware)

---

## Core Analysis Principles

### Principle 1: Understand the Abstraction Hierarchy

```
USER CODE (C++ + PTO intrinsics)
        │
        ▼
┌─────────────────────────────────┐
│      TILE ISA (84 ops)          │  ← Our focus level
│  TADD, TLOAD, TMATMUL, etc.     │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│    ABSTRACT MACHINE             │
│  Core / Device / Host           │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│    BiSheng IR → Hardware ISA    │
│    (Compiler lowering)          │
└─────────────────────────────────┘
```

### Principle 2: Process Over Metrics

Focus on HOW tile operations compose:
- Memory access patterns (TLOAD/TSTORE granularity)
- Reduction strategies (row vs column operations)
- Data reuse through tiling
- Pipeline overlap opportunities

### Principle 3: Optimization at Abstraction Level

Identify patterns that could be:
- **Fused**: Multiple ops → single op (e.g., add+exp)
- **Tiled**: Better block sizes selected automatically
- **Scheduled**: Automatic double-buffering insertion
- **Parallelized**: Work distribution across cores

---

## Experiment Plan

### Experiment 1: Environment Setup & Basic Tests

**Goal**: Verify CPU simulation works

**Steps**:
1. Check compiler requirements (C++23, GCC 14+ / Clang 16+)
2. Build and run test suite
3. Understand test structure

**Process Details**:
- How does CMake configure CPU vs NPU builds?
- What compiler flags enable CPU simulation?
- How are tiles represented in memory?

### Experiment 2: Programming Model Study

**Goal**: Understand tile operations and memory model

**Steps**:
1. Analyze vector add example (simplest case)
2. Analyze row softmax (reduction patterns)
3. Analyze GEMM (compute-intensive)

**Process Details**:
- Tile creation and layout rules
- GlobalTensor view abstraction
- Event-based synchronization model
- Memory binding (TASSIGN)

### Experiment 3: Kernel Analysis

**Goal**: Deep dive into existing kernels

**Targets**:
- `tests/cpu/st/testcase/tadd/` - Basic elementwise
- `tests/cpu/st/testcase/trowsum/` - Row reduction
- `tests/cpu/st/testcase/tmatmul/` - Matrix multiply
- `kernels/manual/a2a3/gemm_performance/` - Optimized GEMM
- `kernels/manual/a2a3/flash_atten/` - Flash Attention

**Process Details**:
- Instruction sequence analysis
- Memory access pattern tracing
- Tiling strategy examination
- Buffer management approach

### Experiment 4: Optimization Opportunity Analysis

**Goal**: Identify automatic optimization opportunities

**Focus Areas**:

1. **Operation Fusion**:
   - Common sequences: `TROWMAX → TROWEXPAND → TSUB`
   - Fused ops exist: `TROWEXPANDSUB` - when to use?
   - Missing fusions that could help

2. **Automatic Tiling**:
   - Current: manual tile size selection
   - Opportunity: cost model for tile size search
   - Constraints: on-chip memory, alignment

3. **Double Buffering**:
   - Current: manual ping-pong implementation
   - Opportunity: automatic buffer staging
   - Pattern: detect load-compute-store chains

4. **Memory Coalescing**:
   - TLOAD/TSTORE access patterns
   - Gather/Scatter optimization
   - Layout transformation overhead

5. **Event Scheduling**:
   - Current: manual event placement
   - Opportunity: automatic dependency analysis
   - Challenge: minimize synchronization overhead

### Experiment 5: Prototype Optimization Pass

**Goal**: Design a simple automatic optimization

**Options**:
1. **Fusion detector**: Find fusable op sequences
2. **Tile size suggester**: Based on problem shape
3. **Buffer allocator**: Automatic TASSIGN generation

---

## Expected Outputs

### Reports
- `environment.md` - Setup and build process
- `analysis.md` - Programming model analysis
- `kernel-analysis.md` - Kernel deep dives
- `optimization-opportunities.md` - Identified improvements
- `final-report.md` - Synthesis and recommendations

### Scripts
- `00_check_env.py` - Environment verification
- `01_run_tests.py` - Run CPU test suite
- `02_analyze_tadd.py` - Analyze vector add
- `03_analyze_softmax.py` - Analyze softmax pattern
- `04_analyze_gemm.py` - Analyze GEMM kernel
- `05_find_fusion_ops.py` - Detect fusion opportunities
- `06_tile_size_analysis.py` - Tile size impact study

---

## Key Files to Study

### Documentation
- `docs/coding/ProgrammingModel.md` - Core concepts
- `docs/coding/Tile.md` - Tile type system
- `docs/coding/opt.md` - Optimization guide
- `docs/isa/README.md` - Instruction reference

### Code
- `include/pto/common/` - Platform-independent headers
- `include/pto/cpu/` - CPU simulation implementations
- `tests/cpu/st/testcase/` - Test case implementations
- `kernels/manual/a2a3/` - Optimized kernels

### Build System
- `CMakeLists.txt` - Build configuration
- `tests/run_cpu.py` - Test runner script
