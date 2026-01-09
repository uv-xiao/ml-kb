# FlashInfer Hands-On Learning Index

**Project**: FlashInfer Kernel Analysis
**Generated**: 2026-01-09
**Status**: Complete

---

## Directory Structure

```
hands_on/flashinfer/
+-- scripts/                    # Profiling and analysis scripts
|   +-- 00_check_env.py         # Environment detection
|   +-- 02_profile_prefill.py   # Prefill attention profiler
|   +-- 03_profile_decode.py    # Decode attention profiler
|   +-- 04_profile_rmsnorm.py   # RMSNorm profiler
|   +-- 05_profile_rope.py      # RoPE profiler
|   +-- 06_jit_analysis.py      # JIT system analysis
|
+-- results/                    # Profiling outputs (JSON)
|   +-- prefill_results.json
|   +-- decode_results.json
|   +-- rmsnorm_results.json
|   +-- rope_results.json
|   +-- jit_analysis.json
|   +-- environment_info.json
|
+-- reports/                    # Analysis reports
    +-- INDEX.md                # This file
    +-- environment.md          # Hardware and software environment
    +-- analysis.md             # Kernel analysis with profiling data
    +-- kernel-dev-guide.md     # Development guide for each kernel
    +-- final-report.md         # Executive summary and conclusions
```

---

## Report Navigation

### Quick Start

1. **[environment.md](environment.md)** - Start here to understand the hardware setup
2. **[analysis.md](analysis.md)** - Detailed kernel profiling results
3. **[kernel-dev-guide.md](kernel-dev-guide.md)** - Development and customization guide
4. **[final-report.md](final-report.md)** - Executive summary

### By Topic

| Topic | Document | Section |
|-------|----------|---------|
| GPU specs | [environment.md](environment.md) | Hardware Configuration |
| JIT setup | [environment.md](environment.md) | JIT System |
| Prefill attention | [analysis.md](analysis.md) | Section 1 |
| Decode attention | [analysis.md](analysis.md) | Section 2 |
| RMSNorm | [analysis.md](analysis.md) | Section 3 |
| RoPE | [analysis.md](analysis.md) | Section 4 |
| Kernel implementation | [kernel-dev-guide.md](kernel-dev-guide.md) | All |
| Performance tuning | [final-report.md](final-report.md) | Recommendations |

### By Kernel

| Kernel | Python API | Analysis | Dev Guide |
|--------|------------|----------|-----------|
| Prefill Attention | `BatchPrefillWithPagedKVCacheWrapper` | [Link](analysis.md#1-prefill-attention-analysis) | [Link](kernel-dev-guide.md#1-prefill-attention-kernel) |
| Decode Attention | `BatchDecodeWithPagedKVCacheWrapper` | [Link](analysis.md#2-decode-attention-analysis) | [Link](kernel-dev-guide.md#2-decode-attention-kernel) |
| RMSNorm | `flashinfer.rmsnorm()` | [Link](analysis.md#3-rmsnorm-analysis) | [Link](kernel-dev-guide.md#3-rmsnorm-kernel) |
| Fused Add RMSNorm | `flashinfer.fused_add_rmsnorm()` | [Link](analysis.md#3-rmsnorm-analysis) | [Link](kernel-dev-guide.md#3-rmsnorm-kernel) |
| RoPE | `flashinfer.apply_rope*()` | [Link](analysis.md#4-rope-analysis) | [Link](kernel-dev-guide.md#4-rope-kernel) |

---

## Profiling Scripts Usage

### Environment Check

```bash
# Check FlashInfer installation and GPU
export FLASHINFER_DISABLE_VERSION_CHECK=1
python scripts/00_check_env.py
```

### Run All Profilers

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1

# Profile each kernel
python scripts/02_profile_prefill.py
python scripts/03_profile_decode.py
python scripts/04_profile_rmsnorm.py
python scripts/05_profile_rope.py
python scripts/06_jit_analysis.py
```

### Advanced Profiling

```bash
# Detailed timeline with nsys
nsys profile -o prefill_trace --trace=cuda,nvtx \
    python scripts/02_profile_prefill.py

# Kernel analysis with ncu
ncu --set full -o decode_kernel \
    python scripts/03_profile_decode.py
```

---

## Key Findings Summary

| Kernel | Type | Peak Utilization | Primary Bottleneck |
|--------|------|------------------|-------------------|
| Prefill Attention | Compute-bound | 98.5% TC | SM work distribution |
| Decode Attention | Memory-bound | 87% HBM | Memory bandwidth |
| RMSNorm | Memory-bound | 26% HBM | Launch overhead |
| RoPE | Memory-bound | 39% HBM | Compute overhead |

---

## Tags

`#flashinfer` `#index` `#hands-on` `#profiling`
