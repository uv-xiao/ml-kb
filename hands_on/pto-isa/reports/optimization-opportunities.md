# PTO-ISA Automatic Optimization Opportunities

## Overview

This report identifies opportunities for automatic optimization at the PTO-ISA abstraction level. Based on analysis of 151 source files with 886 instruction calls.

---

## 1. Operation Fusion Opportunities

### 1.1 Existing Fused Operations

PTO-ISA already provides some fused operations:

| Operation | Decomposed Form | Benefit |
|-----------|-----------------|---------|
| TROWEXPANDSUB | TROWEXPAND + TSUB | 1 tile read saved |
| TROWEXPANDDIV | TROWEXPAND + TDIV | 1 tile read saved |
| TROWEXPANDMUL | TROWEXPAND + TMUL | 1 tile read saved |

### 1.2 Missing Fused Operations (Recommended)

Based on common patterns found in Flash Attention and softmax:

#### a) TSUBEXP (Subtract + Exp)
```
Current:
    TSUB(centered, scores, max_expanded);
    TEXP(exp_scores, centered);

Proposed:
    TSUBEXP(exp_scores, scores, max_expanded);  // exp(a - b)

Benefit: Eliminates intermediate tile, common in softmax
Usage count in codebase: ~4 sequences
```

#### b) TEXPROWSUM (Exp + Row Sum)
```
Current:
    TEXP(exp_scores, centered);
    TROWSUM(row_sum, exp_scores);

Proposed:
    TEXPROWSUM(row_sum, exp_scores, centered);  // exp then row sum

Benefit: Single pass over data, memory traffic reduction
Usage count in codebase: ~3 sequences
```

#### c) TCOLEXPAND Variants
Missing column-wise fused expand operations:
- TCOLEXPANDSUB
- TCOLEXPANDDIV
- TCOLEXPANDMUL

#### d) TMAD (Multiply-Add)
```
Current:
    TMUL(tmp, a, b);
    TADD(dst, tmp, c);

Proposed:
    TMAD(dst, a, b, c);  // a * b + c

Benefit: FMA instruction, common in neural networks
```

### 1.3 Softmax Mega-Fusion

The complete row-softmax pattern appears frequently:
```cpp
// Current pattern (7 instructions):
TROWMAX(row_max, scores);           // 1. Get row max
TROWEXPAND(tmp, row_max);           // 2. Expand to matrix
TSUB(centered, scores, tmp);        // 3. Subtract max
TEXP(exp_scores, centered);         // 4. Exp
TROWSUM(row_sum, exp_scores);       // 5. Row sum
TROWEXPAND(tmp, row_sum);           // 6. Expand to matrix
TDIV(probs, exp_scores, tmp);       // 7. Normalize

// Potential fused operation:
TROWSOFTMAX(probs, scores);         // All in one!
```

**Analysis**: This is already partially fused (TROWEXPANDSUB exists). A full TROWSOFTMAX would require multi-pass algorithm which may not be efficient for all tile sizes.

---

## 2. Layout Transform Optimization

### 2.1 Problem: Double TMOV Sequences

Pattern analysis found 12 instances of `TMOV -> TMOV`:
```cpp
TMOV(tile_left, tile_mat);     // Mat -> Left format
TMOV(tile_left2, tile_left);   // Redundant?
```

**Optimization Opportunities**:

1. **TMOV Chain Elimination**: Compiler pass to detect and merge consecutive TMOVs
2. **Layout-Aware Allocation**: Choose initial tile format to minimize transforms

### 2.2 TTRANS + TMOV Pattern

Found 3 instances of `TTRANS -> TMOV`:
```cpp
TTRANS(kt_tile, k_tile);       // Transpose K
TMOV(k_right, kt_tile);        // Convert to Right format

// Could be:
TMOV_T(k_right, k_tile);       // Transpose and convert in one
```

### 2.3 TLOAD Layout Selection

Currently TLOAD loads to a specific tile format, then TMOV transforms:
```cpp
TLOAD(mat_tile, global);       // Load to Mat format
TMOV(left_tile, mat_tile);     // Convert to Left format

// Opportunity:
TLOAD<LeftLayout>(left_tile, global);  // Direct load to target format
```

---

## 3. Automatic Double Buffering

### 3.1 Current State

Manual double buffering requires explicit:
- Two buffer allocations (ping/pong)
- Event management for overlap
- Loop unrolling for pipeline fill/drain

### 3.2 Proposed Automatic System

```cpp
// User writes single-buffer code:
for (int k = 0; k < K; k += kBlock) {
    TLOAD(a_tile, a_global[k:k+kBlock]);
    TLOAD(b_tile, b_global[k:k+kBlock]);
    TMATMUL_ACC(c_acc, c_acc, a_tile, b_tile);
}

// Compiler transforms to double-buffered:
// (Auto-generated ping-pong with events)
```

**Implementation Approach**:
1. Detect load-compute-store patterns
2. Verify no cross-iteration dependencies
3. Generate ping-pong buffer allocation
4. Insert events for producer-consumer sync

### 3.3 Pragma Hint System

```cpp
#pragma pto_double_buffer(a_tile, b_tile)
for (int k = 0; k < K; k += kBlock) {
    TLOAD(a_tile, a_global[k:k+kBlock]);
    TLOAD(b_tile, b_global[k:k+kBlock]);
    TMATMUL_ACC(c_acc, c_acc, a_tile, b_tile);
}
```

---

## 4. Tile Size Auto-Tuning

### 4.1 Problem

Current approach requires manual tile size selection:
```cpp
// User must choose 64x64, 128x128, etc.
using TileT = Tile<TileType::Vec, float, 64, 64>;
```

### 4.2 Proposed: Cost Model

```cpp
// Automatic tile size selection based on:
// 1. Available on-chip memory
// 2. Problem shape (M, K, N)
// 3. Target pipeline (Cube vs Vector)
// 4. Arithmetic intensity target

auto tile_config = pto::auto_tile<GEMM>(
    M, K, N,
    /* target_reuse_factor */ 4,
    /* max_l1_usage */ 256 * 1024
);
```

### 4.3 Constraint-Based Search

For GEMM-like operations:
```
Constraints:
  - m_tile * k_tile * sizeof(T) + k_tile * n_tile * sizeof(T) <= L1_SIZE
  - m_tile * n_tile * sizeof(T) <= L0C_SIZE
  - m_tile >= 16, n_tile >= 16 (hardware minimum)
  - Prefer power-of-2 sizes for alignment

Objective:
  - Maximize arithmetic intensity: (m_tile * k_tile * n_tile) / memory_traffic
  - Balance Cube vs Vector utilization
```

---

## 5. Memory Access Pattern Optimization

### 5.1 TLOAD Coalescing

Multiple TLOADs to adjacent memory regions could be merged:
```cpp
// Current:
TLOAD(tile_a, global_a);    // Load tile A
TLOAD(tile_b, global_b);    // Load tile B (adjacent)

// If a and b are adjacent in memory:
TLOAD_MERGED(tile_ab, global_ab);  // Single DMA transfer
TEXTRACT(tile_a, tile_ab, 0, 0, rows_a, cols);
TEXTRACT(tile_b, tile_ab, rows_a, 0, rows_b, cols);
```

### 5.2 TSTORE Write Combining

Similar optimization for stores to adjacent regions.

### 5.3 Prefetch Hints

```cpp
#pragma pto_prefetch(a_global[k+kBlock:k+2*kBlock])
TLOAD(a_tile, a_global[k:k+kBlock]);
```

---

## 6. Event/Synchronization Optimization

### 6.1 Current: Manual Event Placement

```cpp
Event<Op::TLOAD, Op::TADD> e;
e = TLOAD(tile, global);
TADD(dst, tile, other, e);  // Wait for TLOAD
```

### 6.2 Proposed: Automatic Dependency Analysis

```cpp
// User writes without events:
TLOAD(a, global_a);
TLOAD(b, global_b);
TADD(c, a, b);

// Compiler inserts events automatically:
// Analyzes: TADD(c, a, b) depends on a and b
// Inserts event wait before TADD for both TLOADs
```

### 6.3 Event Coalescing

Multiple waits on same producer can be coalesced:
```cpp
// Before:
TADD(x, a, b, event_a);  // Wait for a
TSUB(y, a, c, event_a);  // Wait for a again (redundant)

// After:
wait(event_a);           // Single wait
TADD(x, a, b);
TSUB(y, a, c);
```

---

## 7. Implementation Priority

| Optimization | Complexity | Impact | Priority |
|--------------|------------|--------|----------|
| TSUBEXP fusion | Low | Medium | P1 |
| Layout chain elimination | Medium | Medium | P1 |
| Automatic events | High | High | P2 |
| Auto double-buffering | High | High | P2 |
| Tile size auto-tuning | Medium | Medium | P2 |
| TROWSOFTMAX mega-fusion | Medium | Medium | P3 |
| Memory coalescing | High | Medium | P3 |

---

## 8. Recommendations

### Short-term (Low-hanging fruit)

1. **Add missing fused ops**: TSUBEXP, TCOLEXPAND variants
2. **Static pattern detection**: Tool to identify fusion opportunities in user code
3. **Layout optimization hints**: Warn about TMOV chains

### Medium-term (Compiler passes)

1. **Event auto-insertion**: Dependency analysis pass
2. **Layout propagation**: Choose formats to minimize transforms
3. **Double-buffer pragma**: Semi-automatic buffering

### Long-term (Auto-optimization)

1. **Tile size search**: Constraint solver with cost model
2. **Full auto-scheduling**: Like Halide/TVM for PTO
3. **Profile-guided optimization**: Learn from profiler data
