# NanoFlow: Towards Optimal Large Language Model Serving Throughput

**Paper**: [arXiv:2408.12757](https://arxiv.org/abs/2408.12757)
**Repository**: [efeslab/Nanoflow](https://github.com/efeslab/Nanoflow)
**Authors**: Kan Zhu, Yilong Zhao, Liangyu Zhao, Gefei Zuo, Yile Gu, Dedong Xie, et al.
**Date**: August 2024 (revised May 2025)

## Key Insight

Traditional LLM serving systems underutilize GPU resources because heterogeneous operations (compute, memory, network) execute **sequentially** within a device. NanoFlow challenges the assumption that LLM serving is memory-bound, showing that **end-to-end LLM serving is compute-bound** for most common workloads.

## Core Contributions

### 1. Nano-batching

Split requests at **operation granularity** (not layer granularity):
- Global batch divided into smaller nano-batches
- Breaks dependency of sequential operations
- Enables overlapping of heterogeneous operations

Example for LLaMA-2-70B:
- KQV and decode attention: 4 nano-batches
- O/UGD projections: 2 nano-batches

### 2. Operation-Level Pipeline Parallelism

Overlap three types of operations within a single device:
- **Compute-bound**: Dense GEMMs (KQV, O, UG, D projections)
- **Memory-bound**: Decode attention (GEMV)
- **Network-bound**: AllGather/AllReduce for tensor parallelism

Critical path contains only compute operations; memory and network operations are hidden.

### 3. SM Partitioning & Execution Unit Scheduling

Custom SM (Streaming Multiprocessor) allocation:
- A100 has 108 SMs that can be partitioned
- Network kernels achieve 92% peak with only 35 SMs (32%)
- Controlled allocation mitigates kernel interference
- Uses `cublas.set_sm_count_target()` for compute ops
- Uses FlashInfer's green context for attention ops

### 4. Automated Parameter Search

Two-stage optimization using Gurobi ILP solver:
1. **Stage 1**: Determine nano-batch ordering via topological sort
2. **Stage 2**: Optimize SM allocation across concurrent operations

Search space: nano-batch sizes × SM assignments
Runtime: <10 minutes using offline profiling

### 5. Asynchronous CPU Scheduling

- Batch formation runs parallel to GPU execution
- EOS detection delayed by one iteration
- Reduces scheduler overhead from 10% to 2%

## Roofline Analysis

Key metric: **TR = T_Mem / T_Compute**
- TR > 1: Memory-bound
- TR < 1: Compute-bound

Finding: Most real workloads with GQA on multi-GPU are compute-bound.

## Performance Results

**Hardware**: 8× A100 80GB SXM

| Model | Throughput vs SOTA | Optimal % |
|-------|-------------------|-----------|
| LLaMA-2-70B | 1.91× | 68.5% |
| LLaMA-3-70B | - | 59-72% |
| Mixtral 8×7B | - | 59-72% |
| LLaMA-3-8B | - | 59-72% |

Baselines: vLLM v0.5.3, DeepSpeed-FastGen v0.2.3, TensorRT-LLM v0.8.0

## Implementation Stack

- **GEMM**: CUTLASS with SM-constrained variants
- **Attention**: FlashInfer with thread block mapping
- **Network**: MSCCL++ for AllGather/AllReduce
- **Backend**: C++ (~4K lines)
- **Frontend**: Python demo

## Key Equations

**Compute Time**:
```
T_compute = (2 × M × K × N) / (GPU_FLOPS)
```

**Memory Time**:
```
T_mem = (data_size) / (memory_bandwidth)
```

**Network Time**:
```
T_net = (data_size) / (network_bandwidth)
```

## Comparison with Other Techniques

| Technique | Level | Focus |
|-----------|-------|-------|
| Data Parallelism | Inter-device | Throughput via replication |
| Tensor Parallelism | Inter-device | Memory via partitioning |
| Pipeline Parallelism | Inter-device | Memory via layer distribution |
| **NanoFlow** | **Intra-device** | **Resource overlap** |

## Related Concepts

- **Continuous Batching**: Dynamic batch formation (Orca, vLLM)
- **PagedAttention**: Memory-efficient KV cache (vLLM)
- **Chunked Prefill**: Memory control for long sequences (SGLang)
- **Overlap Scheduling**: CPU-GPU pipelining (SGLang)

NanoFlow complements these techniques by focusing on **intra-device** parallelism while others focus on **inter-device** or **memory management**.

## Limitations & Future Work

1. Requires extensive offline profiling
2. SM partitioning support varies by hardware
3. Automated search adds complexity
4. Currently focused on throughput (not latency optimization)

## Tags

`#llm-serving` `#intra-device-parallelism` `#nano-batching` `#sm-partitioning` `#throughput-optimization`
