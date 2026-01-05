---
description: Deep-dive analysis of LLM/ML model implementation code
allowed-tools: Read, Glob, Grep, Bash, WebFetch
---

Perform comprehensive analysis of the specified LLM/ML implementation.

**Target:** $ARGUMENTS

## Analysis Protocol

### Step 1: Reconnaissance

1. **Identify codebase structure:**
   - Find entry points (model forward, inference API)
   - Map directory structure (models/, kernels/, layers/, ops/)
   - Identify configuration files and classes

2. **Determine framework:**
   - PyTorch, JAX, Triton, CUDA, or hybrid
   - Execution model: eager, compiled, megakernel

### Step 2: Architecture Mapping

1. **Module hierarchy:**
   - Model → Layers → Operators → Kernels
   - Create ASCII diagram of structure

2. **Configuration extraction:**
   - Model hyperparameters (hidden_size, num_heads, etc.)
   - Kernel configs (BLOCK_M, BLOCK_N, NUM_STAGES)
   - Constraints and assertions

### Step 3: Component Deep-Dive

For each major component (Attention, MLP, Norm, etc.):

1. **Interface:** inputs, outputs, parameters
2. **Pseudocode:** algorithm abstraction (ignore boilerplate)
3. **Tiling:** block sizes, number of tiles, mapping
4. **Memory:** load/store patterns, hierarchy usage

### Step 4: Data Flow Analysis

1. **Dependencies:** which ops wait for which
2. **Data movement:** tensor lifecycle through model
3. **Parallelism:** TP/PP/SP strategy, communication

### Step 5: Generate Report

Produce a report with:

1. **Architecture ASCII Diagram**
   - High-level model structure
   - Kernel/task mapping

2. **Task/Operator Table**
   | Task | Purpose | Inputs | Outputs | Tiling | Dependencies |

3. **Pseudocode for Key Components**
   - Focus on algorithm, not memory management
   - Annotate tiling and memory patterns

4. **Data Flow Diagram (ASCII)**
   - Single layer or forward pass
   - Memory hierarchy visualization

5. **Technical FAQ**
   - Sequence length support
   - Hardware requirements
   - Kernel fusion strategy
   - Parallelism configuration

6. **Key Files Reference**
   | File | Purpose | Key Functions |

## Output Format

Save detailed analysis to: `reports/implementations/[name]-analysis.md`

Return a summary with:
- Key architectural insights
- Notable optimizations
- Limitations discovered
- Links to detailed sections
