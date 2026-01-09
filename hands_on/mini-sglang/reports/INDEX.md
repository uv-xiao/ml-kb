# Mini-SGLang Hands-On Learning Index

Navigation guide for all hands-on learning materials.

---

## Quick Start

1. **Check Environment**: `python scripts/00_check_env.py`
2. **Test Kernels**: `python scripts/01_test_kernels.py`
3. **Run Full Pipeline**: `./scripts/06_full_pipeline.sh --quick`

---

## Directory Structure

```
mini-sglang/
├── scripts/
│   ├── 00_check_env.py        # Environment verification
│   ├── 01_test_kernels.py     # Kernel correctness tests
│   ├── 02_profile_index.py    # Index kernel profiling
│   ├── 03_profile_store.py    # Store kernel profiling
│   ├── 04_profile_attention.py # Attention backend profiling
│   ├── 05_profile_comm.py     # NCCL communication profiling
│   └── 06_full_pipeline.sh    # Full profiling pipeline
├── results/                   # Profiling outputs (gitignored)
└── reports/
    ├── INDEX.md               # This file
    ├── environment.md         # Environment configuration
    ├── analysis.md            # Per-kernel detailed analysis
    ├── kernel-dev-guide.md    # Developer guide with code positions
    └── final-report.md        # Summary and conclusions
```

---

## Reports

| Report | Description |
|--------|-------------|
| [environment.md](environment.md) | GPU topology, CUDA setup, JIT status |
| [analysis.md](analysis.md) | Per-kernel execution stories and hardware analysis |
| [kernel-dev-guide.md](kernel-dev-guide.md) | Code positions, diagrams, profiling commands |
| [final-report.md](final-report.md) | Summary, findings, optimization opportunities |

---

## Scripts

### Environment Setup

| Script | Purpose | Usage |
|--------|---------|-------|
| `00_check_env.py` | Verify environment | `python scripts/00_check_env.py` |
| `01_test_kernels.py` | Test kernel correctness | `python scripts/01_test_kernels.py` |

### Kernel Profiling

| Script | Purpose | Usage |
|--------|---------|-------|
| `02_profile_index.py` | Index kernel analysis | `python scripts/02_profile_index.py --sweep` |
| `03_profile_store.py` | Store kernel analysis | `python scripts/03_profile_store.py --compare-patterns` |
| `04_profile_attention.py` | Attention profiling | `python scripts/04_profile_attention.py --compare` |
| `05_profile_comm.py` | NCCL analysis | `python scripts/05_profile_comm.py --analyze` |

### Full Pipeline

| Script | Purpose | Usage |
|--------|---------|-------|
| `06_full_pipeline.sh` | Run all profiling | `./scripts/06_full_pipeline.sh --all` |

---

## Kernel Catalog

| Kernel | File | Type | Analysis |
|--------|------|------|----------|
| Index | `kernel/csrc/jit/index.cu` | CUDA JIT | [analysis.md#1-index-kernel](analysis.md#1-index-kernel-analysis) |
| Store | `kernel/csrc/jit/store.cu` | CUDA JIT | [analysis.md#2-store-kernel](analysis.md#2-store-kernel-analysis) |
| PyNCCL | `kernel/csrc/src/pynccl.cu` | CUDA AOT | [analysis.md#3-pynccl-wrapper](analysis.md#3-pynccl-wrapper-analysis) |
| Radix | `kernel/csrc/src/radix.cpp` | CPU AOT | [analysis.md#4-radix-kernel](analysis.md#4-radix-kernel-analysis) |

---

## Learning Path

### Beginner

1. Read [environment.md](environment.md) to understand the setup
2. Run `00_check_env.py` to verify your environment
3. Run `01_test_kernels.py` to see kernels in action
4. Read [kernel-dev-guide.md](kernel-dev-guide.md) overview section

### Intermediate

1. Profile index kernel with `02_profile_index.py`
2. Compare store kernel patterns with `03_profile_store.py --compare-patterns`
3. Study execution stories in [analysis.md](analysis.md)
4. Run Nsight Systems: `./scripts/06_full_pipeline.sh --nsys`

### Advanced

1. Deep-dive into [kernel-dev-guide.md](kernel-dev-guide.md) optimization section
2. Profile attention with `04_profile_attention.py --sweep`
3. Analyze NCCL with `05_profile_comm.py --analyze`
4. Use Nsight Compute for detailed kernel analysis

---

## Reference Links

### Mini-SGLang Documentation

- [README.md](/home/uvxiao/mlkb/code-repos/mini-sglang/README.md)
- [features.md](/home/uvxiao/mlkb/code-repos/mini-sglang/docs/features.md)
- [structures.md](/home/uvxiao/mlkb/code-repos/mini-sglang/docs/structures.md)

### External Resources

- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [FlashInfer Documentation](https://docs.flashinfer.ai/)
- [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/)

---

## Results Interpretation

### Expected Performance Ranges (A100)

| Kernel | Metric | Expected Range |
|--------|--------|----------------|
| Index | BW Efficiency | 70-80% |
| Store (seq) | BW Efficiency | 70-75% |
| Store (rand) | BW Efficiency | 55-65% |
| Prefill Attention | TFLOPs | 50-80 |
| Decode Attention | BW | 1.4-1.7 TB/s |
| NCCL AllReduce | NVLink Util | 85-95% |

### Bottleneck Identification

| Observation | Likely Cause |
|-------------|--------------|
| Low BW, high SM util | Register spilling |
| High BW, low SM util | Memory-bound (expected) |
| High barrier stalls | Warp specialization sync |
| High long_scoreboard | HBM latency |

---

*Generated for Mini-SGLang Hands-On Learning*
