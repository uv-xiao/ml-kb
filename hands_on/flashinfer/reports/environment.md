# FlashInfer Environment Report

**Generated**: 2026-01-09
**Purpose**: Document hardware configuration and FlashInfer installation for hands-on learning

---

## Hardware Configuration

### GPU Inventory

| GPU Index | Model | Memory | Compute Capability | Type |
|-----------|-------|--------|-------------------|------|
| 0 | NVIDIA A100 80GB PCIe | 80 GB | SM 8.0 | PCIe |
| 1 | NVIDIA A100 80GB PCIe | 80 GB | SM 8.0 | PCIe |
| 2 | NVIDIA A100 80GB PCIe | 80 GB | SM 8.0 | PCIe |
| 3 | NVIDIA A100 80GB PCIe | 80 GB | SM 8.0 | PCIe |
| 4 | NVIDIA A100-SXM4-80GB | 80 GB | SM 8.0 | SXM4 |
| 5 | NVIDIA A100 80GB PCIe | 80 GB | SM 8.0 | PCIe |
| 6 | NVIDIA A100 80GB PCIe | 80 GB | SM 8.0 | PCIe |

**Total GPU Memory**: 560 GB (7 x 80 GB)

### A100 Key Specifications (SM 8.0)

| Specification | Value |
|---------------|-------|
| SMs | 108 |
| FP16 Tensor Core TFLOPS | 312 |
| FP32 TFLOPS | 19.5 |
| Memory Bandwidth | 2,039 GB/s (HBM2e) |
| L2 Cache | 40 MB |
| Shared Memory per SM | 164 KB (configurable) |
| Registers per SM | 65,536 |

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
```

**Legend**:
- **NV12**: NVLink with 12 lanes (~600 GB/s bidirectional for A100)
- **PXB**: Connection via PCIe bridge (within same PCIe root complex)
- **SYS**: Connection across NUMA nodes (via QPI/UPI)

### Topology Analysis

```
NUMA NODE 0 (CPU 0-27, 56-83)    NUMA NODE 1 (CPU 28-55, 84-111)
================================  ================================

  GPU0 <=NV12=> GPU1                    GPU4 (SXM4)
    |            |                        |
   PXB          PXB                      PXB
    |            |                        |
  GPU2 <=NV12=> GPU3                GPU5 <=NV12=> GPU6

================================  ================================
             |__________ SYS (QPI/UPI) ____________|
```

**NVLink Pairs** (Best for TP=2):
- GPU 0-1: NV12 (~600 GB/s)
- GPU 2-3: NV12 (~600 GB/s)
- GPU 5-6: NV12 (~600 GB/s)

**Recommended Configurations**:
- **TP=2**: Use NVLink pairs (0-1, 2-3, or 5-6)
- **TP=4**: GPUs 0-3 (same NUMA, mixed NV12+PXB)
- **Avoid**: Cross-NUMA for latency-sensitive workloads

---

## Software Environment

### CUDA Stack

| Component | Version |
|-----------|---------|
| NVIDIA Driver | 570.195.03 |
| CUDA Toolkit | 12.5 |
| CUDA (PyTorch) | 12.8 |
| cuDNN | 91002 (9.1.0) |

### Python Environment

| Package | Version |
|---------|---------|
| Python | 3.12 |
| PyTorch | 2.9.1+cu128 |
| FlashInfer | 0.5.3 |

---

## FlashInfer JIT System

### JIT Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| Workspace Base | `~/.cache/flashinfer` | Default location |
| JIT Verbose | Not set | Set `FLASHINFER_JIT_VERBOSE=1` for debug |
| JIT Debug | Not set | Set `FLASHINFER_JIT_DEBUG=1` for debug builds |
| CUDA Arch List | Not set | Auto-detected as SM 8.0 |

### JIT Directory Structure

```
~/.cache/flashinfer/0.5.3/80/
├── cached_ops/                 # Compiled .so files
│   ├── <kernel_uri_hash>/      # Per-kernel directory
│   │   ├── build.ninja         # Ninja build file
│   │   └── <module>.so         # Compiled module
│   └── ...
├── generated/                  # Generated source files
│   ├── <kernel_uri_hash>/      # Per-kernel directory
│   │   ├── config.inc          # Type configuration
│   │   └── *.cu                # CUDA source
│   └── ...
└── flashinfer_jit.log          # JIT compilation log
```

### Compilation Triggers

JIT compilation is triggered when:

1. **First API call** with new parameter combination:
   - dtype (float16, bfloat16, float8)
   - head_dim (64, 128, 256)
   - pos_encoding_mode (None, RoPE, ALiBi)
   - sliding window, logits soft cap

2. **Cache miss** (no cached .so file):
   - First run after installation
   - After clearing cache (`rm -rf ~/.cache/flashinfer/`)
   - After FlashInfer version upgrade

3. **Source change detected**:
   - Modified kernel source in `include/flashinfer/`
   - Updated compilation flags

### Observed Compilation Times

| Kernel | First Call (JIT) | Cached Call |
|--------|------------------|-------------|
| RMSNorm | ~258 ms | 0.032 ms |
| RoPE | ~13 ms | <0.1 ms |
| Attention Wrapper Init | ~4 ms | N/A |

---

## Kernel Availability

### Tested Kernels

| Kernel | Status | Notes |
|--------|--------|-------|
| RMSNorm | OK | JIT compiled successfully |
| Fused Add RMSNorm | OK | Available |
| RoPE (standard) | OK | JIT compiled successfully |
| RoPE (Llama 3.1) | OK | Available |
| BatchPrefillWithPagedKVCache | OK | Wrapper initialized |
| BatchDecodeWithPagedKVCache | OK | Wrapper initialized |

### Architecture-Specific Features

| Feature | Min SM | A100 (SM 8.0) |
|---------|--------|---------------|
| FlashAttention-2 | SM 8.0 | Supported |
| FlashAttention-3 (TMA) | SM 9.0 | Not available |
| MLA Attention | SM 10.0 | Not available |
| FP8 GEMM | SM 8.9 | Not available |

---

## Performance Baselines

### Memory Bandwidth

A100 80GB HBM2e theoretical bandwidth: **2,039 GB/s**

Expected practical limits:
- Memory-bound kernels: 80-90% of peak (~1.6-1.8 TB/s)
- Compute-bound kernels: Limited by TC utilization

### Roofline Model Reference

```
Arithmetic Intensity (AI) = FLOPs / Bytes

                      -------- Compute Bound --------
                     /
                    / <- Ridge Point (AI = 162 FLOPs/Byte)
                   /
Performance      /
(TFLOPS)       /
              /
Memory ------
Bound

For A100:
- Peak FP16 TC: 312 TFLOPS
- Peak HBM BW: 2.04 TB/s
- Ridge Point: 312 / 2.04 = ~153 FLOPs/Byte
```

---

## Environment Variables Reference

### Recommended for Development

```bash
# Enable verbose JIT logging
export FLASHINFER_JIT_VERBOSE=1

# Enable debug builds (for cuda-gdb, nsight)
export FLASHINFER_JIT_DEBUG=1

# API logging (0=off, 1=basic, 3=detailed, 5=with stats)
export FLASHINFER_LOGLEVEL=3

# Parallel compilation
export FLASHINFER_NVCC_THREADS=4

# Bypass version check (if using mismatched packages)
export FLASHINFER_DISABLE_VERSION_CHECK=1
```

### Profiling Setup

```bash
# For nsys profiling
nsys profile -o profile --trace=cuda,nvtx python script.py

# For ncu kernel analysis
ncu --set full -o kernel_profile python script.py

# For Python profiling with PyTorch
python -m torch.utils.bottleneck script.py
```

---

## Next Steps

1. Run profiling scripts to capture kernel behavior
2. Analyze attention kernels (prefill/decode) in detail
3. Study JIT compilation patterns with different configurations
4. Generate kernel development guide

---

## Tags

`#flashinfer` `#environment` `#a100` `#jit` `#cuda`
