# Mini-SGLang Environment Report

Environment configuration and JIT compilation status for hands-on learning.

---

## System Environment

### GPU Configuration

```
GPU Configuration:
  7 x NVIDIA A100 80GB (mix PCIe + SXM4)
  Compute Capability: 8.0 (Ampere)
  Total GPU Memory: 560 GB
  Peak Memory BW: ~2 TB/s (HBM2e)
```

### GPU Topology

```
       GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6
GPU0    X    NV12  PXB   PXB   SYS   SYS   SYS
GPU1   NV12   X    PXB   PXB   SYS   SYS   SYS
GPU2   PXB   PXB    X    NV12  SYS   SYS   SYS
GPU3   PXB   PXB   NV12   X    SYS   SYS   SYS
GPU4   SYS   SYS   SYS   SYS    X    PXB   PXB
GPU5   SYS   SYS   SYS   SYS   PXB    X    NV12
GPU6   SYS   SYS   SYS   SYS   PXB   NV12   X

Legend:
  NV12 = NVLink (12 links, bidirectional)
  PXB  = PCIe through switch
  SYS  = Cross-NUMA via system interconnect
```

### NVLink Pairs

Recommended for Tensor Parallelism:
- **TP=2 Options**: (0,1), (2,3), (5,6) - Full NVLink bandwidth
- **TP=4 Option**: (0,1,2,3) - Same NUMA node, mixed NVLink+PCIe

### CUDA Environment

| Component | Version | Status |
|-----------|---------|--------|
| CUDA Toolkit | 12.5 | Required |
| Driver | Compatible | Required |
| NVCC | 12.5.x | Required for JIT |
| CUDA_HOME | Set | Required |

---

## JIT Compilation System

### Overview

Mini-SGLang uses TVM-FFI for CUDA kernel JIT compilation:

```
┌─────────────────────────────────────────────────────────────┐
│                    JIT COMPILATION FLOW                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  First Call:                                                 │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │ Python  │───►│ TVM-FFI │───►│  NVCC   │───►│  Cache  │  │
│  │   API   │    │load_jit │    │ Compile │    │   .so   │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│       │                                             │       │
│       │         Subsequent Calls:                   │       │
│       │         ┌─────────────────────────────────┐ │       │
│       └─────────►│      Load from cache          │◄┘       │
│                 └─────────────────────────────────┘         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### JIT Cache Location

```
~/.cache/tvm_ffi/
├── minisgl__index_8192_4_128_1_false/
│   └── libminisgl__index_8192_4_128_1_false.so
├── minisgl__store_2048_128_1_false/
│   └── libminisgl__store_2048_128_1_false.so
└── ...
```

### Kernel Template Parameters

**Index Kernel:**
```cpp
IndexKernel<element_size, num_splits, num_threads, max_occupancy, use_pdl>
// Example: IndexKernel<8192, 4, 128, 1, false>
```

**Store Kernel:**
```cpp
StoreKernel<element_size, num_threads, max_occupancy, use_pdl>
// Example: StoreKernel<2048, 128, 1, false>
```

### Compilation Flags

```bash
# C++ flags
-std=c++20 -O3

# CUDA flags
-std=c++20 -O3 --expt-relaxed-constexpr
```

---

## Dependencies

### Python Packages

| Package | Purpose | Required |
|---------|---------|----------|
| torch | Tensor operations | Yes |
| tvm_ffi | Kernel JIT compilation | Yes |
| flashinfer | Attention kernels | Yes |
| flash_attn | Alternative attention | Optional |

### Kernel Dependencies

| Kernel | Dependencies |
|--------|--------------|
| Index | tvm_ffi, CUDA |
| Store | tvm_ffi, CUDA |
| PyNCCL | tvm_ffi, CUDA, libnccl |
| Radix | tvm_ffi (CPU only) |

---

## Profiling Tools

### Required

| Tool | Version | Purpose |
|------|---------|---------|
| nsys | 2024.x+ | System-level profiling |
| ncu | 2024.x+ | Kernel-level analysis |

### Installation Check

```bash
# Check nsys
nsys --version

# Check ncu
ncu --version
```

### Typical nsys Output

```
CUDA GPU Trace:
Time(%)  Total Time (ns)  Instances  Avg (ns)   Kernel Name
-------  ---------------  ---------  --------   -----------
  45.2%        1,234,567         64    19,290   index_kernel<...>
  32.1%          876,543         64    13,696   store_kv_cache<...>
  22.7%          619,345        128     4,838   flash_attention_...
```

---

## Environment Verification Script

Run the environment check:

```bash
python scripts/00_check_env.py
```

### Expected Output

```
============================================================
GPU CONFIGURATION
============================================================
  GPU 0: NVIDIA A100 80GB PCIe (81251 MiB, CC 8.0)
  GPU 1: NVIDIA A100 80GB PCIe (81251 MiB, CC 8.0)
  ...

============================================================
CUDA TOOLKIT
============================================================
  nvcc: OK (release 12.5)
  CUDA_HOME: /usr/local/cuda

============================================================
PYTHON DEPENDENCIES
============================================================
  PyTorch: OK (v2.x.x)
  TVM-FFI: OK (vx.x.x)
  FlashInfer: OK (vx.x.x)
  Mini-SGLang: OK

============================================================
JIT COMPILATION STATUS
============================================================
  JIT Cache Directory: ~/.cache/tvm_ffi
  Cached Kernels: N (or 0 if first run)
  JIT Ready: YES

============================================================
PROFILING TOOLS
============================================================
  nsys: OK
  ncu: OK

============================================================
KERNEL COMPILATION TEST
============================================================
  Index Kernel: OK (JIT compiled and tested)
  Store Kernel: OK (JIT compiled and tested)
  Radix Kernel: OK (AOT compiled and tested)
  NCCL Module: OK (AOT compiled)

============================================================
ENVIRONMENT SUMMARY
============================================================
  Environment is READY for Mini-SGLang hands-on learning!
```

---

## Troubleshooting

### Common Issues

**1. nvcc not found**
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**2. JIT compilation fails**
```bash
# Check CUDA version matches
nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

**3. tvm_ffi import error**
```bash
# Reinstall tvm_ffi
pip install tvm-ffi --upgrade
```

**4. NCCL not found**
```bash
# Install NCCL
apt-get install libnccl2 libnccl-dev
```

---

*Generated for Mini-SGLang Hands-On Learning*
