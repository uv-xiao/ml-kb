# Hardware-Level Profiling and Tracing

## Philosophy

**Everything runs on hardware.** Understanding performance means understanding:
- How SMs execute your kernels
- How registers and shared memory are utilized
- How tensor cores are fed with data
- How memory bandwidth is consumed
- How communication happens across GPUs

## Profiling Tools Hierarchy

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    PROFILING TOOLS HIERARCHY                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Level 1: System-Wide Tracing (Timeline)                                     │
│  ════════════════════════════════════════                                    │
│  Tool: NVIDIA Nsight Systems (nsys)                                          │
│  Shows: Kernel launches, CPU-GPU interaction, memory transfers               │
│  Use for: Overall execution flow, finding gaps, correlation                  │
│                                                                              │
│  Level 2: Kernel-Level Analysis (Per-Kernel)                                 │
│  ═══════════════════════════════════════════                                 │
│  Tool: NVIDIA Nsight Compute (ncu)                                           │
│  Shows: SM utilization, register/SMEM usage, memory throughput               │
│  Use for: Deep-dive into specific kernels                                    │
│                                                                              │
│  Level 3: Source-Level Correlation                                           │
│  ═════════════════════════════════                                           │
│  Tool: ncu with source info, CUDA debugger                                   │
│  Shows: Which lines cause which stalls                                       │
│  Use for: Optimization at code level                                         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## NVIDIA Nsight Systems (nsys)

### Basic Usage

```bash
# Full system trace
nsys profile -o output_name \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    python script.py

# With GPU metrics sampling
nsys profile -o output_name \
    --trace=cuda,nvtx \
    --gpu-metrics-device=all \
    --gpu-metrics-frequency=10000 \
    python script.py

# Multi-process (for distributed)
nsys profile -o output_%q{RANK} \
    --trace=cuda,nvtx,mpi \
    mpirun -np 8 python distributed_script.py
```

### Key Traces to Enable

| Trace | Flag | What It Shows |
|-------|------|---------------|
| CUDA API | `--trace=cuda` | Kernel launches, memcpy, sync |
| NVTX markers | `--trace=nvtx` | Custom annotations in code |
| OS runtime | `--trace=osrt` | Thread scheduling, I/O |
| cuDNN | `--trace=cudnn` | cuDNN API calls |
| NCCL | `--trace=nccl` | Collective communication |

### Interpreting Nsys Output

Look for:
1. **Gaps between kernels**: CPU overhead, synchronization
2. **Kernel overlap**: Are streams being used effectively?
3. **Memory transfer patterns**: H2D/D2H timing
4. **NVTX regions**: Custom markers for code sections

## NVIDIA Nsight Compute (ncu)

### Basic Usage

```bash
# Full analysis (slow but comprehensive)
ncu --set full -o output_name python script.py

# Target specific kernels (faster)
ncu --kernel-name "flash_attn.*" \
    --launch-skip 10 --launch-count 5 \
    -o output_name python script.py

# Specific metrics only
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
    python script.py
```

### Critical Metrics to Collect

#### SM Utilization

```bash
# Overall SM throughput
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed

# Warp occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active

# Achieved occupancy vs theoretical
ncu --metrics sm__maximum_warps_per_active_cycle_pct
```

#### Register Usage

```bash
# Registers per thread
ncu --metrics launch__registers_per_thread

# Register spilling
ncu --metrics sm__sass_average_data_bytes_per_wavefront_spill
```

#### Shared Memory

```bash
# SMEM usage
ncu --metrics sm__sass_data_bytes_per_wavefront_mem_shared

# Bank conflicts
ncu --metrics sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared
```

#### Tensor Core Utilization

```bash
# Tensor core active cycles
ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed

# Tensor operations
ncu --metrics sm__sass_thread_inst_executed_op_hmma_pred_on.sum
```

#### Memory Bandwidth

```bash
# HBM throughput
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed

# L2 cache
ncu --metrics lts__throughput.avg.pct_of_peak_sustained_elapsed

# SMEM throughput
ncu --metrics sm__memory_throughput.avg.pct_of_peak_sustained_elapsed
```

#### Warp Stalls

```bash
# Stall reasons breakdown
ncu --metrics sm__sass_average_warps_issue_stalled_*
```

### Metric Interpretation Guide

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    METRIC INTERPRETATION GUIDE                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SM Utilization:                                                             │
│  • < 30%: Kernel likely memory-bound or has low occupancy                    │
│  • 30-60%: Typical for memory-bound kernels                                  │
│  • > 60%: Good compute utilization                                           │
│  • > 80%: Excellent (usually compute-bound)                                  │
│                                                                              │
│  Tensor Core Utilization:                                                    │
│  • < 10%: Tensor cores not being used effectively                            │
│  • 10-30%: Partial utilization (data movement bottleneck?)                   │
│  • > 30%: Good tensor core usage                                             │
│  • > 50%: Excellent (matrix math dominated)                                  │
│                                                                              │
│  Memory Bandwidth (DRAM):                                                    │
│  • < 30%: Not memory-bound, or inefficient access patterns                   │
│  • 30-60%: Moderate memory pressure                                          │
│  • > 60%: Memory-bound (good for memory-bound kernels)                       │
│  • > 80%: Near peak (excellent for pure memory kernels)                      │
│                                                                              │
│  Occupancy:                                                                  │
│  • Low occupancy + high SM util = compute-bound, efficient                   │
│  • Low occupancy + low SM util = register/SMEM limited                       │
│  • High occupancy + low SM util = memory-bound                               │
│                                                                              │
│  Warp Stalls:                                                                │
│  • stalled_barrier: Waiting for sync (warp specialization artifact)         │
│  • stalled_membar: Memory barrier wait                                       │
│  • stalled_long_scoreboard: Waiting for memory load                          │
│  • stalled_math_pipe_throttle: Compute units saturated                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Custom Instrumentation

### NVTX Markers

```python
import torch
import nvtx

# Range markers
with nvtx.annotate("attention_layer", color="blue"):
    output = attention(q, k, v)

# Or as decorator
@nvtx.annotate(color="green")
def mlp_layer(x):
    return down_proj(silu(gate_proj(x)) * up_proj(x))
```

### PyTorch CUDA Events

```python
import torch

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
# Your operation here
model_forward(input)
end_event.record()

torch.cuda.synchronize()
elapsed_ms = start_event.elapsed_time(end_event)
```

### Memory Tracking

```python
import torch

# Snapshot at specific point
torch.cuda.memory._record_memory_history(max_entries=100000)

# Your code
output = model(input)

# Dump snapshot
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

# Analyze with torch.cuda.memory._snapshot_to_tree()
```

## Multi-GPU Profiling

### NCCL Tracing

```bash
# Enable NCCL debug
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Profile with nsys
nsys profile -o multi_gpu \
    --trace=cuda,nvtx,nccl \
    torchrun --nproc_per_node=8 script.py
```

### Per-GPU Analysis

```python
# Add per-rank profiling
import os
rank = int(os.environ.get("RANK", 0))

if rank == 0:  # Profile only rank 0
    with torch.profiler.profile(...) as prof:
        train_step()
```

## Process Details Capture

### Event Timeline Template

```
Time (ms)    Event                          Duration    Notes
═══════════════════════════════════════════════════════════════════════════════
0.00         [START] Forward pass
0.12         kernel: embedding_lookup        0.08 ms    SM: 45%, Mem: 72%
0.22         kernel: rms_norm               0.03 ms    SM: 28%, Mem: 85%
0.28         kernel: qkv_gemm               0.45 ms    SM: 78%, TC: 42%
0.75         kernel: flash_attention_fwd    1.23 ms    SM: 65%, Mem: 81%
             └── warp_0: TMA loads                      12 iterations
             └── warp_1-7: tensor core MMA              11 iterations
2.00         kernel: o_proj_gemm            0.38 ms    SM: 75%, TC: 45%
2.40         barrier: nccl_allreduce        0.85 ms    GPU0 waits for GPU3
3.25         kernel: mlp_gate_up_gemm       0.52 ms    SM: 80%, TC: 48%
...
```

### Capturing Intermediate States

```python
class DetailedProfiler:
    def __init__(self):
        self.events = []

    def record(self, name, **kwargs):
        torch.cuda.synchronize()
        self.events.append({
            "time": time.perf_counter(),
            "name": name,
            "gpu_memory": torch.cuda.memory_allocated(),
            "gpu_reserved": torch.cuda.memory_reserved(),
            **kwargs
        })

    def dump(self, path):
        import json
        with open(path, "w") as f:
            json.dump(self.events, f, indent=2)

# Usage
profiler = DetailedProfiler()
profiler.record("start_forward")
output = layer1(x)
profiler.record("after_layer1", output_shape=list(output.shape))
output = layer2(output)
profiler.record("after_layer2", output_shape=list(output.shape))
profiler.dump("detailed_trace.json")
```

## Advanced ncu Profiling Methodologies

### Replay Modes

ncu uses replay to collect metrics across multiple passes:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    NCU REPLAY MODES                                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Kernel Replay (Default):                                                    │
│  ════════════════════════                                                    │
│  • Saves GPU memory before kernel, restores between passes                   │
│  • Best for: Independent kernels, detailed per-kernel analysis              │
│  • Overhead: Memory save/restore per kernel                                  │
│                                                                              │
│  Application Replay:                                                         │
│  ═══════════════════                                                         │
│  • Runs entire application multiple times                                    │
│  • Best for: Kernels with host interdependencies                            │
│  • Flag: --replay-mode application                                          │
│                                                                              │
│  Range Replay:                                                               │
│  ══════════════                                                              │
│  • Captures sequences of CUDA API calls as unified workloads                │
│  • Best for: Concurrent kernels that must run together                      │
│  • Flag: --replay-mode range                                                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Speed of Light Analysis

Speed of Light (SOL) measures achieved vs theoretical peak:

```bash
# Collect SOL metrics
ncu --set full --section SpeedOfLight python script.py
```

Interpretation:
- **Compute SOL**: How close to peak FLOPS
- **Memory SOL**: How close to peak bandwidth
- High compute + low memory = compute-bound
- Low compute + high memory = memory-bound
- Low both = latency-bound or low occupancy

### Roofline Analysis

Roofline charts visualize performance bounds:

```
              Peak FLOPS
                  │
Performance       │────────────────── Compute Bound
(FLOPS)          │                /
                 │              /
                 │            /  Memory Bound
                 │          /
                 │        /
                 │      /
                 │    / ← Ridge Point
                 │  /
                 │/
                 └────────────────────────────────
                   Arithmetic Intensity (FLOPS/Byte)
```

- **Below roofline**: Room for optimization
- **At roofline**: Hitting hardware limits
- **Ridge point**: Where memory-bound meets compute-bound

```bash
# Generate roofline data
ncu --set roofline -o roofline_output python script.py
```

### Detailed Warp Stall Analysis

| Stall Reason | Cause | Fix |
|--------------|-------|-----|
| `long_scoreboard` | Waiting for L1TEX/memory load | Increase cache hits, prefetch, better locality |
| `short_scoreboard` | Waiting for MIO (shared mem) | Reduce bank conflicts, optimize layout |
| `math_pipe_throttle` | Compute pipe oversubscribed | Increase warps, reduce math intensity |
| `memory_throttle` | Too many memory instructions | Combine operations, reduce frequency |
| `barrier` | Waiting at __syncthreads() | Expected for warp specialization |
| `not_selected` | Eligible but not scheduled | Increase occupancy |
| `wait` | Waiting for fixed-latency op | Normal, no action needed |

```bash
# Detailed stall breakdown
ncu --metrics \
    smsp__warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
    smsp__warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
    smsp__warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,\
    smsp__warps_issue_stalled_barrier_per_issue_active.ratio \
    python script.py
```

## LLM Inference-Specific Profiling

### TensorRT-LLM Methodology

```bash
# Control iteration range to reduce profile size
export TLLM_PROFILE_START_STOP=100-150
export TLLM_NVTX_DEBUG=1

# Profile with cudaProfilerApi for precise control
nsys profile -o trace -f true \
    -t 'cuda,nvtx,python-gil' \
    -c cudaProfilerApi \
    --cuda-graph-trace node \
    python trtllm_script.py
```

Key environment variables:
- `TLLM_PROFILE_START_STOP=A-B`: Profile only iterations A to B
- `TLLM_NVTX_DEBUG=1`: Extra NVTX markers
- `TLLM_TORCH_PROFILE_TRACE=<path>`: Save PyTorch profiler data

### vLLM Profiling

```bash
# Enable vLLM profiler
export VLLM_TORCH_PROFILER_DIR=/path/to/traces

# Start server, then control via HTTP
curl -X POST http://localhost:8000/start_profile
# ... send inference requests ...
curl -X POST http://localhost:8000/stop_profile

# Analyze with Perfetto UI (ui.perfetto.dev)
```

### PyTorch Profiler Integration

```python
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,      # Skip first iteration
        warmup=1,    # Warmup iteration
        active=3,    # Profile 3 iterations
        repeat=1
    ),
    on_trace_ready=tensorboard_trace_handler('./logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, batch in enumerate(dataloader):
        model(batch)
        prof.step()

# View with: tensorboard --logdir=./logs
# Or export: prof.export_chrome_trace("trace.json")
```

Key metrics from PyTorch profiler:
- `self_cuda_time_total`: Time spent in CUDA kernels
- `cuda_memory_usage`: Peak memory per operation
- `cpu_time_total`: Host-side overhead

## Best Practices

### Reproducibility

```bash
# Lock GPU clocks for consistent measurements
sudo nvidia-smi -lgc <freq>  # Lock to specific frequency

# Enable persistence mode
sudo nvidia-smi -pm 1

# Disable cache control for realistic behavior
ncu --cache-control none ...
```

### Overhead Management

- **Software-patched metrics**: Highest overhead, use sparingly
- **Hardware counters**: Very low overhead
- **Limit kernel targets**: Use `--kernel-name` regex
- **Short kernels**: Need >20μs to reach steady state

### Profile Size Management

```bash
# For LLM inference with many iterations
# Only profile specific iteration range
nsys profile -c cudaProfilerApi ...

# In code:
torch.cuda.cudart().cudaProfilerStart()
# ... iterations to profile ...
torch.cuda.cudart().cudaProfilerStop()
```

## Checklist

- [ ] nsys trace collected (timeline, GPU-CPU correlation)
- [ ] ncu analysis for hot kernels (SM, register, SMEM, TC, memory)
- [ ] Speed of Light analysis performed
- [ ] Roofline position determined (memory-bound vs compute-bound)
- [ ] Warp stall reasons analyzed with specific fixes identified
- [ ] NVTX markers added to code regions of interest
- [ ] Multi-GPU communication profiled (if applicable)
- [ ] Intermediate states captured (not just final metrics)
- [ ] Memory usage timeline captured
- [ ] Event timeline extracted
- [ ] PyTorch profiler traces exported for TensorBoard/Perfetto

## References

- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [Nsight Systems Documentation](https://developer.nvidia.com/nsight-systems)
- [TensorRT-LLM Performance Analysis](https://nvidia.github.io/TensorRT-LLM/performance/perf-analysis.html)
- [Profiling vLLM Inference Server](https://developers.redhat.com/articles/2025/10/16/profiling-vllm-inference-server-gpu-acceleration-rhel)
- [PyTorch Profiler Tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
