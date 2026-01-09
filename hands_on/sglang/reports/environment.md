# SGLang Hands-On Learning: Environment Report

**Generated:** 2026-01-09
**Status:** Hardware Documentation

---

## Hardware Configuration

### GPU Inventory

| GPU | Model | Memory | Form Factor | Compute | NUMA |
|-----|-------|--------|-------------|---------|------|
| 0 | NVIDIA A100 | 80 GB | PCIe | SM 8.0 | Node 0 |
| 1 | NVIDIA A100 | 80 GB | PCIe | SM 8.0 | Node 0 |
| 2 | NVIDIA A100 | 80 GB | PCIe | SM 8.0 | Node 0 |
| 3 | NVIDIA A100 | 80 GB | PCIe | SM 8.0 | Node 0 |
| 4 | NVIDIA A100 | 80 GB | PCIe | SM 8.0 | Node 1 |
| 5 | NVIDIA A100 | 80 GB | SXM4 | SM 8.0 | Node 1 |
| 6 | NVIDIA A100 | 80 GB | SXM4 | SM 8.0 | Node 1 |

**Total GPU Memory:** 560 GB HBM2e

### A100 Hardware Specifications (SM 8.0)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NVIDIA A100 80GB ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Streaming Multiprocessors: 108 SMs                                          │
│  ├── FP32 CUDA Cores: 6,912                                                  │
│  ├── FP16 Tensor Cores: 432 (3rd generation)                                │
│  └── INT8 Tensor Cores: 432                                                  │
│                                                                              │
│  Memory System:                                                              │
│  ├── HBM2e: 80 GB @ 2.0 TB/s peak bandwidth                                 │
│  ├── L2 Cache: 40 MB                                                         │
│  ├── Shared Memory: 164 KB per SM (configurable)                            │
│  └── Register File: 256 KB per SM                                           │
│                                                                              │
│  Compute Capability:                                                         │
│  ├── Peak FP16 Tensor: 312 TFLOPS                                           │
│  ├── Peak FP32: 19.5 TFLOPS                                                 │
│  ├── Peak INT8 Tensor: 624 TOPS                                             │
│  └── Peak TF32 Tensor: 156 TFLOPS                                           │
│                                                                              │
│  Key Features:                                                               │
│  ├── Sparsity support (2:4 structured) for 2x speedup                       │
│  ├── MIG (Multi-Instance GPU) partitioning                                  │
│  ├── NVLink 3.0: 600 GB/s bidirectional (12 links)                         │
│  └── PCIe Gen4: 64 GB/s bidirectional                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### GPU Topology Matrix

```
        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6
GPU0     X    NV12  PXB   PXB   SYS   SYS   SYS
GPU1    NV12   X    PXB   PXB   SYS   SYS   SYS
GPU2    PXB   PXB    X    NV12  SYS   SYS   SYS
GPU3    PXB   PXB   NV12   X    SYS   SYS   SYS
GPU4    SYS   SYS   SYS   SYS    X    PXB   PXB
GPU5    SYS   SYS   SYS   SYS   PXB    X    NV12
GPU6    SYS   SYS   SYS   SYS   PXB   NV12   X

Legend:
  NV12 = NVLink (12 links, 600 GB/s bidirectional)
  PXB  = Same PCIe switch (~25 GB/s)
  SYS  = Cross-NUMA via CPU (~10-15 GB/s)
```

### Topology Visualization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM TOPOLOGY                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────┐   ┌─────────────────────────────────┐  │
│  │          NUMA NODE 0            │   │          NUMA NODE 1            │  │
│  │                                 │   │                                 │  │
│  │   ┌───────┐     ┌───────┐      │   │   ┌───────┐                     │  │
│  │   │ GPU0  │<───>│ GPU1  │      │   │   │ GPU4  │                     │  │
│  │   │ PCIe  │ NV  │ PCIe  │      │   │   │ PCIe  │                     │  │
│  │   │ 80GB  │     │ 80GB  │      │   │   │ 80GB  │                     │  │
│  │   └───┬───┘     └───┬───┘      │   │   └───┬───┘                     │  │
│  │       │   PXB       │   PXB    │   │       │   PXB                   │  │
│  │   ┌───┴───┐     ┌───┴───┐      │   │   ┌───┴───┐     ┌───────┐      │  │
│  │   │ GPU2  │<───>│ GPU3  │      │   │   │ GPU5  │<───>│ GPU6  │      │  │
│  │   │ PCIe  │ NV  │ PCIe  │      │   │   │ SXM4  │ NV  │ SXM4  │      │  │
│  │   │ 80GB  │     │ 80GB  │      │   │   │ 80GB  │     │ 80GB  │      │  │
│  │   └───────┘     └───────┘      │   │   └───────┘     └───────┘      │  │
│  │                                 │   │                                 │  │
│  └────────────────┬────────────────┘   └────────────────┬────────────────┘  │
│                   │                                     │                    │
│                   └─────────── QPI/UPI ─────────────────┘                    │
│                          (~40 GB/s per direction)                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

NVLink Pairs:
  • GPU 0 ↔ GPU 1 (NV12, NUMA 0)
  • GPU 2 ↔ GPU 3 (NV12, NUMA 0)
  • GPU 5 ↔ GPU 6 (NV12, NUMA 1)
```

---

## Recommended Configurations

### Tensor Parallelism (TP) Configurations

| Config | GPUs | Interconnect | Use Case |
|--------|------|--------------|----------|
| TP=2 (Best) | [0,1] or [2,3] or [5,6] | NVLink | Latency-sensitive, 70B models |
| TP=4 (NUMA-local) | [0,1,2,3] | NVLink + PXB | 70B-140B models, NUMA 0 |
| TP=4 (Alternative) | [4,5,6] + 1 more | Mixed | Requires cross-NUMA |
| TP=7 (Full) | [0-6] | Mixed | Very large models only |

### Bandwidth Estimates

```
Communication Bandwidth by Path:
────────────────────────────────────────────────────
Path Type        Unidirectional    Bidirectional
────────────────────────────────────────────────────
NVLink (12x)     300 GB/s          600 GB/s
PCIe Gen4 x16    32 GB/s           64 GB/s
PXB (same sw)    ~25 GB/s          ~50 GB/s
SYS (cross-NUMA) ~10-15 GB/s       ~20-30 GB/s
────────────────────────────────────────────────────

AllReduce Efficiency (TP=4 on NUMA 0):
  • Best case (ring): ~200 GB/s effective
  • With PCIe hops: ~50-80 GB/s effective
```

### Memory Budget per GPU (A100 80GB)

```
Memory Allocation Strategy:
────────────────────────────────────────────────────
Component                    Typical (70B)   Max
────────────────────────────────────────────────────
Model Weights                17.5 GB/GPU     35 GB/GPU
KV Cache                     40-50 GB        60 GB
Activation Memory            2-5 GB          10 GB
CUDA Workspace               1-2 GB          4 GB
SGLang Overhead              0.5-1 GB        2 GB
────────────────────────────────────────────────────
Total                        ~62 GB          ~80 GB
────────────────────────────────────────────────────

KV Cache Token Capacity (Llama 70B, TP=4):
  • Per GPU: ~50 GB available for KV
  • Tokens per layer: ~150K @ FP16
  • With paged attention: Efficient dynamic allocation
```

---

## Software Environment

### Required Packages

| Package | Version | Status |
|---------|---------|--------|
| CUDA | 12.5 | Required |
| PyTorch | 2.4+ | Required |
| FlashInfer | 0.2+ | Required |
| SGLang | Latest | Required |
| Triton | 3.0+ | Optional |

### Installation Verification

```bash
# Run environment check script
python scripts/00_check_env.py --verbose

# Quick validation
python -c "import sglang; print(sglang.__version__)"
python -c "import flashinfer; print(flashinfer.__version__)"
```

---

## Profiling Tool Availability

### NVIDIA Tools

| Tool | Purpose | Command |
|------|---------|---------|
| nvidia-smi | GPU monitoring | `nvidia-smi dmon -s pucvmet` |
| nsys | System-level tracing | `nsys profile -o trace python ...` |
| ncu | Kernel-level profiling | `ncu --set full -o report python ...` |
| nvtx | Custom annotations | Integrated in code |

### NSight Systems Configuration

```bash
# Recommended nsys profile command for SGLang
nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --cuda-memory-usage=true \
  --gpuctxsw=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  -o sglang_trace \
  python benchmark_script.py
```

### NSight Compute Configuration

```bash
# Attention kernel deep-dive
ncu \
  --set full \
  --import-source yes \
  --kernel-regex ".*attention.*|.*flash.*" \
  --launch-count 10 \
  -o attention_profile \
  python attention_benchmark.py
```

---

## Hardware-Aware Profiling Targets

### Per-Kernel Metrics to Collect

| Metric | What It Tells You | Target Range (A100) |
|--------|-------------------|---------------------|
| SM Utilization | Kernel parallelism | 70-95% |
| Tensor Core Utilization | TC usage | 30-50% (prefill), 10-20% (decode) |
| Memory Throughput | HBM bandwidth | 1.5-1.8 TB/s |
| L2 Hit Rate | Cache efficiency | >50% for reused data |
| Occupancy | Warps per SM | 50-100% |
| Warp Stall Reasons | Bottleneck ID | See analysis |

### Warp Stall Interpretation Guide

```
Warp Stall Category         Interpretation               Action
────────────────────────────────────────────────────────────────────
long_scoreboard             Waiting for HBM loads        Prefetch, tiling
barrier                     Sync overhead                Reduce sync points
short_scoreboard            SMEM bank conflicts          Pad SMEM arrays
not_selected                Low occupancy                Increase parallelism
wait                        Memory dependency            Pipeline loads
tex_throttle                Texture cache full           Reduce tex accesses
────────────────────────────────────────────────────────────────────
```

---

## Quick Reference

### Environment Variables

```bash
# SGLang configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3      # Use NUMA 0 GPUs
export SGLANG_ATTENTION_BACKEND=flashinfer
export SGLANG_LOG_LEVEL=INFO

# Profiling enablement
export NVTX_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0  # Keep 0 for accurate timing
```

### Validation Commands

```bash
# GPU health check
nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu --format=csv

# Memory status
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# NVLink status
nvidia-smi nvlink --status
```

---

## Next Steps

1. Run `scripts/00_check_env.py` to validate environment
2. Review `reports/analysis.md` for codebase understanding
3. Follow `reports/plan.md` for experiment execution
4. Use profiling scripts in `scripts/` for detailed analysis
