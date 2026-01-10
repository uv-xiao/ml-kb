# PTO-ISA Hands-On Learning Report

## Executive Summary

**PTO-ISA** (Parallel Tile Operation ISA) is a virtual instruction set architecture for Ascend NPUs that provides portable, tile-level abstractions for high-performance computing. This hands-on learning session explored:

1. CPU simulation environment setup and testing
2. Running official demos (GEMM, Flash Attention, MLA)
3. Understanding instruction patterns through code analysis
4. Learning optimization techniques from manual performance kernels

**Key Finding**: PTO-ISA provides a well-designed tile abstraction with ~84 instructions. The main optimization opportunities are:
- Operation fusion (especially softmax patterns)
- Pipeline orchestration (cube + vector core parallelism)
- Double-buffering and preloading strategies
- Multi-core tiling and load balancing

---

## 1. Environment Setup

### Compiler Requirements
- **C++23 support required** for template metaprogramming features
- GCC 11.4.0: Does NOT support C++23 (fails with deducing-this errors)
- **Clang 20.0.0**: Successfully compiles all PTO-ISA code

### CPU Simulation Mode
- Enabled via `__CPU_SIM` preprocessor define
- Headers implement CPU fallbacks for all tile operations
- Useful for development and correctness testing without hardware

---

## 2. Running Official CPU Demos

All demos located in `code-repos/pto-isa/demos/cpu/`:

### 2.1 GEMM Demo
```bash
cd demos/cpu/gemm_demo
mkdir -p build && cd build
cmake .. -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
make && ./gemm_demo
```

**Results**:
```
gemm_demo: M=32 K=16 N=32
max_abs_diff=3.57628e-07
perf: avg_ms=0.831013 matmul_flops=32768 gflops=0.0394314
```

**Key Patterns Demonstrated**:
- GlobalTensor setup with Shape/Stride templates
- Tile types: Mat, TileLeft, TileRight, TileAcc
- Layout transforms: TMOV, TTRANS
- MatMul: TMATMUL
- Memory: TLOAD, TSTORE

### 2.2 Flash Attention Demo
```bash
cd demos/cpu/flash_attention_demo
mkdir -p build && cd build
cmake .. -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
make && ./flash_attention_demo
```

**Results**:
```
flash_attention_demo: B=1 H=2 S=64 D=32 (non-causal)
max_abs_diff(pto, ref) = 3.25963e-09
[PASS] flash_attention_demo
```

**Key Patterns Demonstrated**:
- Row-wise reductions: TROWMAX, TROWSUM
- Fused operations: TROWEXPANDSUB, TROWEXPANDDIV
- Elementwise operations: TMULS, TEXP
- Complete softmax pattern

### 2.3 MLA (Multi-Head Latent Attention) Demo
```bash
cd demos/cpu/mla_attention_demo
mkdir -p build && cd build
cmake .. -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
make && ./mla_attention_demo
```

**Results**:
```
mla_attention_demo: B=1 H=2 S=64 D=32 R=16
[PASS] mla_attention_demo
```

---

## 3. PTO-ISA Architecture

### Instruction Categories

| Category | Count | Examples | Purpose |
|----------|-------|----------|---------|
| Memory | 8 | TLOAD, TSTORE, TGATHER | GM ↔ Tile transfers |
| Layout | 6 | TMOV, TTRANS, TEXTRACT | Format conversion |
| Compute | 25+ | TADD, TMUL, TEXP, TLOG | Elementwise ops |
| Reduction | 6 | TROWSUM, TROWMAX, TCOLMAX | Axis-wise aggregation |
| MatMul | 4 | TMATMUL, TMATMUL_ACC | Cube engine ops |
| Fused | 3 | TROWEXPANDSUB/DIV/MUL | Combined ops |

### Core Abstractions

**GlobalTensor**: View into global memory
```cpp
using Shape = Shape<1, 1, 1, M, N>;
using Stride = Stride<M*N, M*N, M*N, N, 1>;
using GT = GlobalTensor<float, Shape, Stride>;
```

**Tile**: On-chip 2D buffer
```cpp
// Types: Vec (elementwise), Mat (L1), Left/Right (L0A/B), Acc (L0C)
Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor> tile;
```

### Flash Attention Code Pattern
```cpp
// From flash_attention_demo.cpp:234-255
TLOAD(qTile, qGlobal);
TLOAD(kTile, kGlobal);
TLOAD(vTile, vGlobal);

TMOV(qLeft, qTile);
TTRANS(ktTile, kTile, kTile);
TMOV(kRight, ktTile);

TMATMUL(scoresAcc, qLeft, kRight);
TMOV(scores, scoresAcc);
TMULS(scores, scores, scale);

// Softmax: max -> sub -> exp -> sum -> div
TROWMAX(rowMax, scores, scores);
TROWEXPANDSUB(scoresCentered, scores, rowMax);
TEXP(expScores, scoresCentered);
TROWSUM(rowSum, expScores, expScores);
TROWEXPANDDIV(probs, expScores, rowSum);

TMOV(pLeft, probs);
TMOV(vRight, vTile);
TMATMUL(outAcc, pLeft, vRight);
TSTORE(oGlobal, outAcc);
```

---

## 4. Optimization Techniques (from Manual Kernels)

### 4.1 GEMM Performance Kernel Optimizations

From `kernels/manual/a2a3/gemm_performance/`:

**Core Partitioning**:
- Split work across 24 cores using 4×6 grid
- `singleCoreM=1536, singleCoreK=6144, singleCoreN=1024`

**Base Block Selection**:
- Choose blocks that maximize compute-to-memory ratio
- For FP16: `[baseM, baseN, baseK] = [128, 256, 64]`
- L0A: 128×64×2 = 16 KiB (fits comfortably)
- L0B: 64×256×2 = 32 KiB (fills budget)

**L1 Caching**:
- `stepKa=stepKb=4` to cache four K-blocks per transfer
- Reduces DMA launch overhead

**Double Buffering**:
- L1, L0A, L0B all double-buffered
- 32 KiB ping/pong split per buffer

**Performance Results** (6144³):
| Metric | Value |
|--------|-------|
| TMATMUL (Cube) Ratio | 86.7% |
| TLOAD Ratio | 95.2% |
| TEXTRACT Ratio | 68.1% |
| TSTORE Ratio | 3.1% |
| Execution time | 1.506 ms |

### 4.2 Flash Attention Pipeline Orchestration

From `kernels/manual/a2a3/flash_atten/`:

**Four-Stage Pipeline**:
1. `compute_qk`: Q·K^T matmul on cube cores
2. `compute_p`: TSOFTMAXFA on vector cores
3. `compute_pv`: P·V matmul on cube cores
4. `compute_gu`: Final reduction on vector cores

**Key Optimizations**:

1. **QK Preloading**: `qkPreloadNum` (default 4) allows cube pipeline to run ahead
2. **FIFO Buffering**: `qkp_tile_fifo_size = 1 + qkPreloadNum` for producer-consumer overlap
3. **Inplace Operations**: TROWEXPANDSUB, TROWEXPANDMUL with dst==src
4. **AccTile Double-Buffering**: `assign_running_acc_tile()` for hazard avoidance

**Multi-Core Tiling**:
- Split on (B, N, S0/CUBE_S0) axes
- S1 is reduction axis, not split within core
- For flash-decoding: can split S1 with final GU kernel

---

## 5. Automatic Optimization Opportunities

### 5.1 Missing Fused Operations (P1)

| Proposed Op | Decomposed Form | Benefit |
|-------------|-----------------|---------|
| TSUBEXP | TSUB + TEXP | Softmax pattern |
| TEXPROWSUM | TEXP + TROWSUM | Single-pass reduction |
| TCOLEXPAND variants | Missing vs row | Symmetry completion |

### 5.2 Layout Transform Elimination (P1)

**Problem**: TMOV → TMOV sequences (found in pattern analysis)

**Solution**: Compiler pass to detect and merge consecutive transforms

### 5.3 Automatic Event Insertion (P2)

**Current**: Manual event placement required
```cpp
Event<Op::TLOAD, Op::TADD> e;
e = TLOAD(tile, global);
TADD(dst, tile, other, e);  // Wait for TLOAD
```

**Proposed**: Dependency analysis to auto-insert events

### 5.4 Auto Double-Buffering (P2)

**Proposed pragma**:
```cpp
#pragma pto_double_buffer(a_tile, b_tile)
for (int k = 0; k < K; k += kBlock) {
    TLOAD(a_tile, a_global[k:k+kBlock]);
    TMATMUL_ACC(c_acc, c_acc, a_tile, b_tile);
}
```

---

## 6. Key Learnings

### Programming Model
1. **Tile types matter**: Vec for elementwise, Mat for L1, Left/Right for L0A/B, Acc for L0C
2. **Layout transforms are explicit**: TMOV between tile types, TTRANS for transpose
3. **Fused ops save bandwidth**: TROWEXPANDSUB/DIV/MUL avoid intermediate storage

### Optimization Strategy
1. **Core partitioning first**: Choose 2D grid that balances work
2. **Base block for L0 fit**: 32 KiB budget per buffer
3. **Increase K reuse**: stepK caching in L1
4. **Pipeline overlap**: Double-buffer everything, use events minimally

### Performance Analysis
- **TMATMUL ratio dropping** + **TLOAD ratio rising** → memory-bound
- **TEXTRACT overhead** 40-68% → layout costs matter
- **TSTORE small** for compute-heavy kernels (many FMAs per write)

---

## 7. Files and Resources

### This Session
```
hands_on/pto-isa/
├── reports/
│   ├── INDEX.md                      # Navigation index
│   ├── plan.md                       # Learning plan
│   ├── environment.md                # Environment setup
│   ├── optimization-opportunities.md # Detailed analysis
│   └── final-report.md               # This report
└── scripts/
    ├── 00_check_env.py               # Environment verification
    └── 01_analyze_patterns.py        # Pattern extraction
```

### PTO-ISA Repository Examples
- `demos/cpu/gemm_demo/` - Basic GEMM with TMATMUL
- `demos/cpu/flash_attention_demo/` - Complete attention with softmax
- `demos/cpu/mla_attention_demo/` - Multi-head latent attention
- `kernels/manual/a2a3/gemm_performance/` - Optimized GEMM (86.7% cube util)
- `kernels/manual/a2a3/flash_atten/` - Production FA with pipeline diagrams
- `docs/coding/tutorials/` - vec-add, gemm, row-softmax tutorials

---

## 8. Conclusion

PTO-ISA provides a well-designed abstraction for Ascend NPU programming:

**Strengths**:
- Portable across A2/A3/A5 hardware generations
- Rich set of tile operations (~84 instructions)
- Good CPU simulation for development
- Existing fused ops (TROWEXPANDSUB/DIV/MUL) for common patterns

**Optimization Opportunities**:
1. **More fused operations** - TSUBEXP, TEXPROWSUM for softmax
2. **Layout chain elimination** - reduce TMOV overhead
3. **Automatic synchronization** - dependency-based event insertion
4. **Auto double-buffering** - pragma-based transformation

**Learning Path**:
1. Run CPU demos to understand basic patterns
2. Study tutorials for tiling and ping-pong concepts
3. Analyze manual kernels for production optimization techniques
4. Use utilization metrics to guide tuning decisions
