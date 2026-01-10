# PTO-ISA Environment Report

## Environment Configuration

### Compiler
- **C++ Compiler**: Clang 20.0.0 (supports C++23)
- **C Compiler**: Clang 20.0.0
- **CMake**: 4.1.2
- **Python**: 3.12.12

### Build Configuration
```bash
# CPU simulation build
python3 tests/run_cpu.py --cxx clang++ --cc clang --clean --verbose
```

## Test Results

### Unit Tests: 66/66 PASSED

All 66 CPU simulation tests passed in 426ms total:

| Category | Tests | Status |
|----------|-------|--------|
| Elementwise | tadd, tsub, tmul, tdiv, tabs, tneg, etc. | PASS |
| Logical | tand, tor, txor, tnot, tshl, tshr | PASS |
| Reductions | trowsum, trowmax, tcolsum, tcolmax | PASS |
| Comparisons | tcmp, tcmps, tsel, tsels | PASS |
| Memory | tload, tstore, tgather, tscatter | PASS |
| Matrix | tmatmul, ttrans, textract | PASS |
| Advanced | tsort32, tmrgsort, tfillpad | PASS |

### Demos: All PASSED

| Demo | Configuration | Max Error | Status |
|------|---------------|-----------|--------|
| GEMM | M=32 K=16 N=32 | 3.58e-07 | PASS |
| Flash Attention | B=1 H=2 S=64 D=32 | 3.26e-09 | PASS |

## CPU Simulation Architecture

### How It Works

```
USER CODE (C++ + PTO intrinsics)
         │
         ▼
┌────────────────────────────────┐
│   include/pto/common/          │  ← API declarations
│   pto_instr.hpp               │
│   Tile.hpp, GlobalTensor.hpp  │
└────────────────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│   include/pto/cpu/             │  ← CPU implementations
│   TAdd.hpp, TLoad.hpp, ...    │
│   parallel.hpp (threading)    │
└────────────────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│   Standard C++ execution       │
│   Tiles = in-memory arrays    │
│   Events = no-ops             │
└────────────────────────────────┘
```

### Key Differences from NPU

| Aspect | CPU Simulation | NPU Execution |
|--------|---------------|---------------|
| Tiles | In-memory arrays | On-chip SRAM |
| Events | No-ops (ignored) | Hardware synchronization |
| TASSIGN | No-op | Binds tile to buffer address |
| Parallelism | Single-threaded per core | Multi-pipeline hardware |
| Performance | ~1000x slower | Full hardware speed |

### Build System Detection

CMake automatically detects CPU vs NPU build:

```cmake
# From CMakeLists.txt
if(NOT DEFINED __CCE_AICORE__)
    # CPU simulation mode
    add_definitions(-D__CPU_SIM__)
    # Use standard C++ compiler
endif()
```

## Key Observations

### 1. Compilation Flags

CPU mode defines:
- `__CPU_SIM__` - Enables CPU stub implementations
- No `__CCE_AICORE__` - Disables NPU-specific code paths

### 2. Event Handling

In CPU mode, events are completely ignored:
```cpp
#ifdef __CCE_AICORE__
  Event<Op::TLOAD, Op::TADD> e_load_add;
  e_load_add = TLOAD(tile, tensor);
  TADD(dst, a, b, e_load_add);
#else
  // CPU: events are no-ops
  TLOAD(tile, tensor);
  TADD(dst, a, b);
#endif
```

### 3. Buffer Assignment

TASSIGN is a no-op in CPU mode:
```cpp
// NPU: assigns tile to specific L1/L0 buffer address
TASSIGN(tile, 0x0000);

// CPU: tile storage is automatic (std::vector-like)
```

### 4. Parallel Execution

CPU tests use `parallel_for_rows` for multi-threading:
```cpp
cpu::parallel_for_rows(rows, cols, [&](std::size_t r) {
    // Process row r
});
```
