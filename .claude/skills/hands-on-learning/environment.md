# Environment Detection and Setup

## Overview

Before running anything, understand your hardware and create an isolated environment.

## GPU Detection

### Basic Information

```bash
# GPU models and memory
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv

# Detailed GPU info
nvidia-smi -q
```

### GPU Topology (Critical for Multi-GPU)

```bash
# GPU interconnect topology
nvidia-smi topo -m

# Example output interpretation:
#         GPU0  GPU1  GPU2  GPU3  CPU Affinity
# GPU0     X    NV4   NV4   NV4   0-23
# GPU1    NV4    X    NV4   NV4   0-23
# GPU2    NV4   NV4    X    NV4   24-47
# GPU3    NV4   NV4   NV4    X    24-47
#
# Legend:
#   X    = Self
#   NV#  = NVLink (# = generation, higher = faster)
#   PIX  = PCIe switch
#   PXB  = PCIe bridge
#   PHB  = CPU/PCIe host bridge
#   SYS  = System (slowest)
```

### NVLink Details

```bash
# NVLink status and bandwidth
nvidia-smi nvlink -s

# NVLink topology
nvidia-smi nvlink -t

# Per-link bandwidth
nvidia-smi nvlink -c
```

### PCIe Information

```bash
# PCIe bandwidth and generation
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv

# Detailed PCIe info
lspci -vvv | grep -A 30 "NVIDIA"
```

## Environment Summary Template

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         ENVIRONMENT SUMMARY                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  GPU Configuration:                                                          │
│  ═══════════════════                                                         │
│  • GPU Model:        NVIDIA H100 80GB HBM3                                   │
│  • GPU Count:        8                                                       │
│  • Memory per GPU:   80 GB HBM3                                              │
│  • Compute Cap:      9.0 (Hopper)                                            │
│  • Total GPU Memory: 640 GB                                                  │
│                                                                              │
│  GPU Topology:                                                               │
│  ═════════════                                                               │
│  • Interconnect:     NVLink 4.0 (900 GB/s bidirectional)                     │
│  • Topology:         All-to-all NVLink (full mesh)                           │
│  • NVSwitch:         Yes (4th generation)                                    │
│  • PCIe:             Gen5 x16                                                │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  GPU Topology Matrix:                                                   │ │
│  │         GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7                 │ │
│  │  GPU0    X    NV4   NV4   NV4   NV4   NV4   NV4   NV4                  │ │
│  │  GPU1   NV4    X    NV4   NV4   NV4   NV4   NV4   NV4                  │ │
│  │  ...    (all NV4 = full NVLink mesh)                                   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Software Stack:                                                             │
│  ═══════════════                                                             │
│  • CUDA Version:     12.4                                                    │
│  • cuDNN Version:    8.9.7                                                   │
│  • NCCL Version:     2.20.3                                                  │
│  • Driver Version:   550.54.15                                               │
│                                                                              │
│  System Resources:                                                           │
│  ═════════════════                                                           │
│  • CPU:              AMD EPYC 9654 96-Core (2 sockets)                       │
│  • RAM:              1.5 TB DDR5                                             │
│  • NUMA Nodes:       2 (GPUs 0-3 on node 0, GPUs 4-7 on node 1)              │
│                                                                              │
│  Communication Implications:                                                 │
│  ═══════════════════════════                                                 │
│  • TP across all 8 GPUs: Efficient (NVLink)                                  │
│  • Cross-NUMA traffic: May have higher latency                               │
│  • Recommended TP config: TP=8 (all NVLink) or TP=4 (same NUMA)             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Isolated Environment Creation

### Option 1: Micromamba (Recommended)

```bash
# Install micromamba if not present
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
export PATH="$PWD/bin:$PATH"

# Create environment
micromamba create -n project_env python=3.11 -y
micromamba activate project_env

# Install CUDA-aware PyTorch
micromamba install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Install project
pip install -e /path/to/project
```

### Option 2: UV (Fast, Modern)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv project_env --python 3.11

# Activate
source project_env/bin/activate

# Install dependencies
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -e /path/to/project
```

### Option 3: Conda

```bash
# Create environment
conda create -n project_env python=3.11 -y
conda activate project_env

# Install CUDA toolkit
conda install cuda-toolkit=12.4 -c nvidia -y

# Install project
pip install -e /path/to/project
```

### Option 4: Docker/Container

```bash
# Use NVIDIA NGC container
docker run --gpus all -it --rm \
    -v /path/to/project:/workspace \
    nvcr.io/nvidia/pytorch:24.04-py3 \
    bash

# Inside container
cd /workspace
pip install -e .
```

## Verification Commands

```bash
# Verify CUDA is accessible
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'Device count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'Current device: {torch.cuda.get_device_name(0)}')"

# Verify NCCL (for multi-GPU)
python -c "import torch.distributed as dist; print(f'NCCL available: {dist.is_nccl_available()}')"

# Check installed packages
pip list | grep -E "torch|cuda|flash|triton"
```

## Environment Checklist

- [ ] GPU count and models identified
- [ ] GPU topology mapped (NVLink, PCIe, NUMA)
- [ ] Communication bandwidth understood
- [ ] CUDA/cuDNN versions verified
- [ ] Isolated environment created
- [ ] Project installed in environment
- [ ] CUDA accessibility verified
- [ ] Multi-GPU communication tested (if applicable)
