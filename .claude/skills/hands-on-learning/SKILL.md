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

## Directory Structure

**IMPORTANT**: All hands-on learning sessions MUST use this directory structure:

```
hands_on/
└── <project_name>/
    ├── scripts/                    # Execution and profiling scripts
    │   ├── setup.sh                # Environment setup
    │   ├── profile.sh              # Profiling commands
    │   └── run_experiments.py      # Experiment runner
    ├── results/                    # Profiling outputs and logs (.gitignored)
    │   ├── nsys/                   # Nsight Systems traces
    │   ├── ncu/                    # Nsight Compute reports
    │   └── logs/                   # Execution logs
    └── reports/                    # Analysis reports and development guides
        ├── INDEX.md                # Navigation and summary
        ├── plan.md                 # Goals and tracking requirements
        ├── environment.md          # Environment detection results
        ├── analysis.md             # Codebase analysis findings
        ├── experiments.md          # Per-experiment findings
        ├── kernel-dev-guide.md     # Kernel development guide (if applicable)
        └── final-report.md         # Synthesis and conclusions
```

### Environment Variables

When working with a project, use these variables:

```bash
export PROJECT_NAME="<your-project-name>"
export HANDS_ON_BASE="$(pwd)/hands_on"
export PROJECT_DIR="${HANDS_ON_BASE}/${PROJECT_NAME}"
export SCRIPTS_DIR="${PROJECT_DIR}/scripts"
export RESULTS_DIR="${PROJECT_DIR}/results"
export REPORTS_DIR="${PROJECT_DIR}/reports"
```

### Initialization

Before starting any hands-on learning session:

```bash
PROJECT_NAME="<project-name>"
mkdir -p "hands_on/${PROJECT_NAME}/scripts"
mkdir -p "hands_on/${PROJECT_NAME}/results"
mkdir -p "hands_on/${PROJECT_NAME}/reports"
```

## Skill Structure

This skill consists of multiple focused modules:

| File | Purpose |
|------|---------|
| `SKILL.md` | Overview, directory structure, workflow (this file) |
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
│  │ 0. INITIALIZE   │  Create project directory structure                    │
│  └────────┬────────┘  → hands_on/<project>/scripts,results,reports          │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                         │
│  │ 1. ENVIRONMENT  │  Detect GPU, topology, create isolated env             │
│  └────────┬────────┘  → Save to reports/environment.md                      │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                         │
│  │ 2. ANALYZE FIRST│  Run llm-code-analysis or repo-analysis               │
│  └────────┬────────┘  → Save to reports/analysis.md                         │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                         │
│  │ 3. PLAN         │  Interpret goals → specific tracking requirements      │
│  └────────┬────────┘  → Save to reports/plan.md                             │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                         │
│  │ 4. EXECUTE      │  Run with hardware-level profiling                     │
│  └────────┬────────┘  → Scripts in scripts/, outputs in results/            │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                         │
│  │ 5. REPORT       │  Process details, events, hardware behavior            │
│  └─────────────────┘  → Save to reports/final-report.md                     │
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

### Principle 4: Joint Process+Hardware Analysis (CRITICAL)

**Principles 2 and 3 must be applied TOGETHER, not separately.**

**Bad** (process only): "The attention kernel takes 0.45ms"
**Bad** (metrics only): "SM utilization is 68%, HBM is 72%"

**Good** (combined):
```
Event: Flash Attention (prefill)
├── Duration: 0.45ms (37% of layer time)
├── Algorithm: FlashAttention-2 with tiling
├── Hardware state:
│   ├── SM util: 68%, TC: 35%, HBM: 72%
│   └── Warp stalls: long_scoreboard 42%, barrier 28%
├── Interpretation: Mixed compute/memory bound
└── Bottleneck: TC limited by HBM read rate
```

Every observation must answer: **"What is happening AND how is the hardware behaving during this event?"**

### Principle 5: Interpret User Goals

User says: "I want to understand FlashInfer performance"

This is vague. Think about what they might actually want:
- Attention kernel behavior at different sequence lengths?
- Comparison with other attention implementations?
- Memory efficiency of paged KV cache?
- Decode vs prefill characteristics?

**Use codebase analysis** to identify what's INTERESTING about this project, then propose specific tracking targets.

## Quick Start

### Step 0: Initialize Project

```bash
PROJECT_NAME="my-project"
mkdir -p "hands_on/${PROJECT_NAME}/scripts"
mkdir -p "hands_on/${PROJECT_NAME}/results"
mkdir -p "hands_on/${PROJECT_NAME}/reports"
```

### Step 1: Environment Setup

```bash
# Detect environment (see environment.md for details)
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
nvidia-smi topo -m

# Create isolated environment
micromamba create -n project_env python=3.11 -y
micromamba activate project_env

# Save environment info
nvidia-smi > "hands_on/${PROJECT_NAME}/reports/environment.md"
```

### Step 2: Pre-Analysis

```
Use /analyze-llm or repo-analysis skill to generate:
- Architecture overview
- Key component identification
- Performance-critical paths

Save output to: hands_on/${PROJECT_NAME}/reports/analysis.md
```

### Step 3: Plan Creation

```
Based on analysis + user goal, define:
- What specific events to track
- Which kernels to profile
- What hardware metrics matter
- Expected behavior vs actual

Save plan to: hands_on/${PROJECT_NAME}/reports/plan.md
```

### Step 4: Execute with Profiling

```bash
# Create profiling script
cat > "hands_on/${PROJECT_NAME}/scripts/profile.sh" << 'EOF'
#!/bin/bash
PROJECT_DIR="$(dirname "$0")/.."
RESULTS_DIR="${PROJECT_DIR}/results"

# Hardware-level profiling
ncu --set full -o "${RESULTS_DIR}/ncu/profile" python script.py

# Or system-level tracing
nsys profile -o "${RESULTS_DIR}/nsys/trace" --trace=cuda,nvtx python script.py
EOF
chmod +x "hands_on/${PROJECT_NAME}/scripts/profile.sh"
```

### Step 5: Generate Report

```
Focus on:
- Timeline of events (not just averages)
- Hardware utilization patterns
- Anomalies and interesting observations
- "Story" of what happens during execution

Save to: hands_on/${PROJECT_NAME}/reports/final-report.md
```

## Output File Locations

| Output Type | Location |
|-------------|----------|
| Environment info | `hands_on/<project>/reports/environment.md` |
| Codebase analysis | `hands_on/<project>/reports/analysis.md` |
| Experiment plan | `hands_on/<project>/reports/plan.md` |
| Profiling scripts | `hands_on/<project>/scripts/` |
| Profiling outputs | `hands_on/<project>/results/` (gitignored) |
| Experiment results | `hands_on/<project>/reports/experiments.md` |
| Kernel dev guide | `hands_on/<project>/reports/kernel-dev-guide.md` |
| Final report | `hands_on/<project>/reports/final-report.md` |
| Index/navigation | `hands_on/<project>/reports/INDEX.md` |

## Module Index

- [environment.md](environment.md) - GPU topology, isolated environments
- [analysis-first.md](analysis-first.md) - Pre-run codebase analysis
- [planning.md](planning.md) - Goal interpretation and plan creation
- [profiling.md](profiling.md) - Hardware-level profiling strategies
- [reporting.md](reporting.md) - Process-focused report generation

## Tags

`#hands-on` `#profiling` `#hardware` `#process-tracing` `#llm-serving`
