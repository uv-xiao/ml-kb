---
name: hands-on-learning
description: Hands-on exploration and profiling of LLM serving projects. Explores codebase with docs, detects environment (GPUs, CUDA), creates try-using plans, executes with profiling/tracing, analyzes results, and generates comprehensive reports. Mimics expert learning through practice. Use for deep-diving into vLLM, SGLang, FlashInfer, TensorRT-LLM, or any ML serving system.
---

# Hands-On Learning Skill for LLM Serving Projects

## Philosophy

This skill mimics how a **human expert learns a new project**:

1. **Understand before doing** - Analyze codebase first, then plan experiments
2. **Process over metrics** - Care about HOW things run, not just final numbers
3. **Hardware is truth** - Everything runs on real hardware; understand SM/register/memory behavior
4. **Interpret goals** - Translate vague objectives into specific tracking requirements

## Skill Structure

This skill consists of multiple focused modules:

| File | Purpose |
|------|---------|
| `SKILL.md` | Overview and workflow (this file) |
| `environment.md` | Environment detection (GPU, topology, dependencies) |
| `analysis-first.md` | Pre-run codebase analysis strategy |
| `planning.md` | Plan creation with requirements interpretation |
| `profiling.md` | Hardware-level profiling and tracing |
| `reporting.md` | Process-focused report generation |

## Core Workflow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    HANDS-ON LEARNING WORKFLOW                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐                                                         │
│  │ 1. ENVIRONMENT  │  Detect GPU, topology, create isolated env             │
│  └────────┬────────┘  → See environment.md                                   │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                         │
│  │ 2. ANALYZE FIRST│  Run llm-code-analysis or repo-analysis               │
│  └────────┬────────┘  → See analysis-first.md                               │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                         │
│  │ 3. PLAN         │  Interpret goals → specific tracking requirements      │
│  └────────┬────────┘  → See planning.md                                      │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                         │
│  │ 4. EXECUTE      │  Run with hardware-level profiling                     │
│  └────────┬────────┘  → See profiling.md                                     │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                         │
│  │ 5. REPORT       │  Process details, events, hardware behavior            │
│  └─────────────────┘  → See reporting.md                                     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Key Principles

### Principle 1: Analyze Before Running

**DO NOT** jump into running experiments immediately.

**FIRST**: Use `llm-code-analysis` or `repo-analysis` to understand:
- What kernels/operators exist?
- What are the key data structures?
- Where are the hot paths?
- What configuration options affect performance?

This analysis informs WHAT to profile and WHERE to look.

### Principle 2: Process Details Over Final Metrics

**Bad**: "Throughput is 1000 tokens/sec"

**Good**:
- "At batch size 8, the attention kernel launches 32 times per forward pass"
- "SM occupancy drops to 45% during the first 3 decode steps due to KV cache warmup"
- "AllReduce stalls for 2.3ms waiting for the slowest GPU"

We care about **events, transitions, and intermediate states**.

### Principle 3: Hardware is the Ground Truth

Everything runs on physical hardware. Always track:
- **SM utilization**: How many SMs are active? When?
- **Register pressure**: Are we spilling to local memory?
- **Shared memory**: Bank conflicts? Occupancy limits?
- **Tensor cores**: Are they actually being used? Utilization %?
- **Memory bandwidth**: Are we hitting HBM limits?
- **NVLink/PCIe**: Communication bottlenecks?

### Principle 4: Interpret User Goals

User says: "I want to understand FlashInfer performance"

This is vague. Think about what they might actually want:
- Attention kernel behavior at different sequence lengths?
- Comparison with other attention implementations?
- Memory efficiency of paged KV cache?
- Decode vs prefill characteristics?

**Use codebase analysis** to identify what's INTERESTING about this project, then propose specific tracking targets.

## Quick Start

### Step 1: Environment Setup
```bash
# Detect environment (see environment.md for details)
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
nvidia-smi topo -m

# Create isolated environment
micromamba create -n project_env python=3.11 -y
micromamba activate project_env
```

### Step 2: Pre-Analysis
```
Use /analyze-llm or repo-analysis skill to generate:
- Architecture overview
- Key component identification
- Performance-critical paths
```

### Step 3: Plan Creation
```
Based on analysis + user goal, define:
- What specific events to track
- Which kernels to profile
- What hardware metrics matter
- Expected behavior vs actual
```

### Step 4: Execute with Profiling
```bash
# Hardware-level profiling (see profiling.md)
ncu --set full -o profile_output python script.py

# Or system-level tracing
nsys profile --trace=cuda,nvtx python script.py
```

### Step 5: Generate Report
```
Focus on:
- Timeline of events (not just averages)
- Hardware utilization patterns
- Anomalies and interesting observations
- "Story" of what happens during execution
```

## Module Index

- [environment.md](environment.md) - GPU topology, isolated environments
- [analysis-first.md](analysis-first.md) - Pre-run codebase analysis
- [planning.md](planning.md) - Goal interpretation and plan creation
- [profiling.md](profiling.md) - Hardware-level profiling strategies
- [reporting.md](reporting.md) - Process-focused report generation

## Tags

`#hands-on` `#profiling` `#hardware` `#process-tracing` `#llm-serving`
