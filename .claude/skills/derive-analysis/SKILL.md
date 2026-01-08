---
name: derive-analysis
description: Derives comprehensive analysis reports from existing notes, papers, and codebase analyses. Synthesizes multiple sources, searches online for complementary information, and generates visual explanations with ASCII diagrams, pseudocode, and intuitive frameworks. Use when creating research syntheses, comprehensive studies, or deriving new insights from existing materials in the knowledge base.
---

# Deriving Comprehensive Analysis Reports

## Overview

This skill guides creating well-researched, visually-clear analysis reports by:
1. Reading and synthesizing existing notes in the knowledge base
2. Searching online for complementary technical information
3. Synthesizing disparate sources into coherent narratives
4. Creating visual explanations with ASCII diagrams, pseudocode, and tables

## Workflow

### Phase 1: Understand the Purpose

Read the target file to understand:
- What questions need to be answered?
- What references are cited?
- What level of detail is expected?
- What visual elements would help?

### Phase 2: Gather Existing Materials

1. Read all referenced materials in the knowledge base
2. Use Grep/Glob to find related notes and reports
3. Identify concepts that need deeper explanation
4. Note gaps that require external research

### Phase 3: Research Online

For each knowledge gap:
- Search for authoritative technical sources
- Find recent developments and implementations
- Locate expert explanations and tutorials
- Clarify technical definitions and terminology

### Phase 4: Synthesize and Structure

Create a report that:
- Answers the stated purpose clearly
- Explains concepts progressively (simple to complex)
- Uses visual frameworks at each level of abstraction
- Includes pseudocode with key parameters explained
- Connects theory to practice with concrete examples

### Phase 5: Visual Explanation Patterns

#### Multi-Level Architecture Diagrams
```
┌─────────────────────────────────────────────────────┐
│  Level 1: High-Level View                           │
├─────────────────────────────────────────────────────┤
│  ┌─────────┐    ┌─────────┐    ┌─────────┐        │
│  │ Component│───▶│Component│───▶│Component│        │
│  └─────────┘    └─────────┘    └─────────┘        │
├─────────────────────────────────────────────────────┤
│  Level 2: Detailed View                             │
│  ┌─────────────────────────────────────┐           │
│  │  Sub-component details...           │           │
│  └─────────────────────────────────────┘           │
└─────────────────────────────────────────────────────┘
```

#### Timeline-Based Execution Flow
```
Time ───────────────────────────────────────────────▶

Thread 0: ┌────┐     ┌────┐          ┌────┐
          │ Op1│     │ Op3│          │ Op5│
          └────┘     └────┘          └────┘

Thread 1:      ┌────┐     ┌────┐
               │ Op2│     │ Op4│
               └────┘     └────┘
                    ▲
                    │ Synchronization
```

#### Loop-Based Pseudocode (Preferred)

Use loops to represent iterative processes - this is the most accurate way to show
how systems actually execute. Configurations alone don't capture runtime behavior.

**Multi-Level Loop Structure** (for LLM serving):
```python
# LEVEL 1: Serving Loop (seconds scale)
def serving_loop():
    while server_running:
        # Receive requests and form batches
        requests = receive_requests()
        batch = scheduler.form_batch(requests)  # continuous batching

        # LEVEL 2: Generation Loop (milliseconds scale)
        for step in range(max_new_tokens):
            # Forward pass through all layers
            hidden = embed(batch.token_ids)

            # LEVEL 3: Layer Loop (microseconds scale)
            for layer_idx in range(num_layers):
                hidden = transformer_layer(hidden, layer_idx)

            # Sample next tokens
            next_tokens = sample(lm_head(hidden))
            batch.append_tokens(next_tokens)

            # Check completion
            if batch.all_finished():
                break

        # Send responses
        send_responses(batch)
```

**Kernel-Level Loop** (for megakernels):
```python
# Persistent kernel - single launch, runs entire inference
def megakernel(task_queues, scoreboard):
    sm_id = get_sm_id()

    while not all_done:
        # Each SM pulls from its own queue
        task = task_queues[sm_id].dequeue()

        # Wait for dependencies
        for dep in task.dependencies:
            while scoreboard[dep] == 0:
                spin_wait()

        # Execute task (different task types = MPMD behavior)
        execute_task(task)

        # Signal completion
        scoreboard[task.id] = 1
```

**Loop with Timing Annotations**:
```python
def decode_step(batch):
    """Single decode iteration - ~10-50ms for 70B model"""

    for layer in range(num_layers):           # 80 iterations
        # Attention: ~200µs (memory-bound, GEMV)
        q, k, v = qkv_proj(hidden)            # GEMM: ~50µs
        k_cache[layer].append(k)
        v_cache[layer].append(v)
        attn_out = flash_attention(q, k_cache, v_cache)  # ~100µs
        hidden = o_proj(attn_out)             # GEMM: ~50µs

        # MLP: ~400µs (compute-bound, GEMM)
        hidden = down_proj(silu(gate_proj(hidden)) * up_proj(hidden))

    return lm_head(hidden)  # ~100µs
```

#### Comparison Tables
```
| Aspect        | Approach A    | Approach B    | Tradeoff           |
|---------------|---------------|---------------|-------------------|
| Latency       | Low           | High          | A better for RT   |
| Throughput    | Medium        | High          | B better for batch|
| Complexity    | Simple        | Complex       | A easier to impl  |
```

## Quality Checklist

- [ ] All referenced materials have been read
- [ ] Knowledge gaps filled with online research
- [ ] Concepts explained progressively
- [ ] Visual diagrams at each abstraction level
- [ ] **Loop-based pseudocode shows actual execution flow** (not just configs)
- [ ] Timing annotations where helpful
- [ ] Tradeoffs and design space clearly analyzed
- [ ] Sources cited appropriately
