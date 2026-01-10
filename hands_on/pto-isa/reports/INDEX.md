# PTO-ISA Hands-On Learning Index

## Overview

**Project**: PTO-ISA (Parallel Tile Operation ISA)
**Repository**: `code-repos/pto-isa/`
**Purpose**: Learn programming and optimization for Ascend NPU tile operations

## Reports

| Report | Description |
|--------|-------------|
| [plan.md](plan.md) | Learning objectives and experiment plan |
| [environment.md](environment.md) | Environment setup and test results |
| [optimization-opportunities.md](optimization-opportunities.md) | Detailed automatic optimization analysis |
| [final-report.md](final-report.md) | Comprehensive findings and recommendations |

## Scripts

| Script | Description |
|--------|-------------|
| [00_check_env.py](../scripts/00_check_env.py) | Verify environment and run quick test |
| [01_analyze_patterns.py](../scripts/01_analyze_patterns.py) | Extract and analyze instruction patterns |

## Key Findings

### 1. CPU Demos Work
All three official demos pass with CPU simulation:
- **GEMM Demo**: max_error=3.58e-07
- **Flash Attention**: max_error=3.26e-09
- **MLA Attention**: max_error=1.14e-11

### 2. Optimization Techniques
From manual kernels (GEMM 86.7% cube utilization):
- Core partitioning (4×6 grid for 24 cores)
- Base block selection (128×256×64 for FP16)
- L1 caching (stepK=4)
- Double buffering (L1, L0A, L0B)

### 3. Automation Opportunities
- Operation fusion (TSUBEXP, TEXPROWSUM)
- Layout chain elimination (TMOV → TMOV)
- Automatic event insertion
- Auto double-buffering pragma

## Quick Start

```bash
# Run official CPU demos
cd code-repos/pto-isa/demos/cpu/gemm_demo
mkdir -p build && cd build
cmake .. -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
make && ./gemm_demo

# Run Flash Attention demo
cd ../../flash_attention_demo
mkdir -p build && cd build
cmake .. -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
make && ./flash_attention_demo

# Analyze instruction patterns (from this directory)
python3 scripts/01_analyze_patterns.py
```

## PTO-ISA Repository Examples

| Example | Location | Purpose |
|---------|----------|---------|
| GEMM Demo | demos/cpu/gemm_demo/ | Basic TMATMUL patterns |
| Flash Attention | demos/cpu/flash_attention_demo/ | Softmax + attention |
| MLA Demo | demos/cpu/mla_attention_demo/ | Latent attention |
| GEMM Performance | kernels/manual/a2a3/gemm_performance/ | 86.7% cube utilization |
| Flash Attention | kernels/manual/a2a3/flash_atten/ | Pipeline orchestration |

## Related Resources

- [PTO-ISA Documentation](../../code-repos/pto-isa/docs/)
- [ISA Reference](../../code-repos/pto-isa/docs/isa/)
- [Tutorials](../../code-repos/pto-isa/docs/coding/tutorials/)
- [Optimization Guide](../../code-repos/pto-isa/docs/coding/opt.md)
