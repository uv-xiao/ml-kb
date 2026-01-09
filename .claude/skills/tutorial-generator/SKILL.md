---
name: tutorial-generator
description: Creates and updates comprehensive tutorials from existing analysis notes, reports, and codebase documentation. Synthesizes technical materials into structured educational content with examples, ASCII diagrams, and practical exercises. Use when generating tutorials, creating learning materials, updating tutorial content from analysis reports, or converting technical documentation into teaching materials.
allowed-tools: Read, Grep, Write, Glob, Edit, Bash
user-invocable: update-tutorial
---

# Tutorial Generator Skill

## Overview

This skill helps create and update comprehensive tutorials from existing technical analysis, research notes, and codebase documentation in the knowledge base.

## When to Use

- Creating new tutorial chapters from analysis reports
- Updating existing tutorials with new insights from hands-on learning
- Converting technical documentation into step-by-step guides
- Synthesizing multiple sources into educational content

## Tutorial Structure

Tutorials in this knowledge base follow a consistent structure:

```
tutorials/<topic>/
├── README.md                 # Overview, prerequisites, learning path
├── chapters/
│   └── XX_chapter_name/
│       ├── README.md         # Chapter overview and learning goals
│       ├── examples/         # Runnable code examples
│       │   └── 01_example/
│       │       ├── README.md
│       │       └── code.*
│       ├── exercises/        # Practice problems
│       │   └── 01_exercise/
│       │       ├── problem.md
│       │       └── solution.*
│       └── profiling/        # Performance analysis (if applicable)
└── common/                   # Shared utilities
```

## Source Materials

When generating tutorials, draw from:

1. **Analysis Reports** (`reports/implementations/`)
   - Hands-on learning reports
   - Kernel development guides
   - Codebase analysis

2. **Research Notes** (`notes/topics/`)
   - Technical summaries
   - Quick references

3. **Code Repositories** (`code-repos/`)
   - Actual implementations
   - Test cases
   - Documentation

## Content Guidelines

### Structure Each Tutorial Section With:

1. **Learning Objectives**
   - What the reader will learn
   - Prerequisites
   - Estimated time

2. **Conceptual Overview**
   - ASCII diagrams explaining key concepts
   - Analogies for complex ideas
   - Visual explanations of data flow

3. **Code Examples**
   - Complete, runnable code
   - Inline comments explaining key lines
   - Expected output

4. **Hands-On Exercises**
   - Progressive difficulty
   - Clear problem statements
   - Solution hints

5. **Profiling & Analysis** (for performance-focused content)
   - NCU/NSys commands
   - Key metrics to observe
   - Optimization opportunities

### ASCII Diagram Style

Use consistent ASCII art for visualizations:

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPONENT OVERVIEW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input                    Process                  Output       │
│  ┌─────┐                 ┌─────┐                 ┌─────┐       │
│  │     │────────────────▶│     │────────────────▶│     │       │
│  └─────┘                 └─────┘                 └─────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Code Position References

Always include file paths and line numbers:

```markdown
**Code Location:** `path/to/file.py:123-145`

Key implementation details:
- Function `xyz()` handles ...
- The kernel launches at line 130
```

### Profiling Commands

Include ready-to-run profiling commands:

```bash
# Profile with NCU
ncu --set full --kernel-name "kernel_name" -o profile python script.py

# System trace with NSys
nsys profile --trace=cuda,nvtx -o trace python script.py
```

## Workflow

### Creating New Tutorial Content

1. **Identify Source Materials**
   ```bash
   # Find relevant reports
   ls reports/implementations/

   # Search for analysis
   grep -r "topic_keyword" notes/ reports/
   ```

2. **Extract Key Content**
   - Read analysis reports
   - Identify core concepts to teach
   - Gather code examples

3. **Structure the Tutorial**
   - Create chapter directory
   - Write README with learning goals
   - Create example subdirectories

4. **Write Content**
   - Transform analysis into teaching material
   - Add ASCII diagrams
   - Include runnable examples

5. **Add Exercises**
   - Create problem statements
   - Write solutions
   - Add validation tests

### Updating Existing Tutorials

1. **Review New Analysis**
   - Read the latest reports
   - Identify new insights

2. **Map to Tutorial Structure**
   - Determine which chapters to update
   - Identify gaps to fill

3. **Integrate Content**
   - Add new sections
   - Update existing examples
   - Add new exercises if appropriate

## Example: Updating Kernels Tutorial

To update the kernels tutorial with mini-sglang analysis:

1. Source materials:
   - `reports/implementations/mini-sglang-hands-on-learning.md`
   - `reports/implementations/mini-sglang-kernel-dev-guide.md`
   - `reports/implementations/flashinfer-kernel-dev-guide.md`

2. Target location:
   - `tutorials/kernels/chapters/12_llm_serving/` (new chapter)
   - Or integrate into existing Chapter 09 (attention) and Chapter 10 (MoE)

3. Content to extract:
   - Kernel architecture diagrams
   - Code position references
   - Profiling commands
   - Optimization opportunities

## Validation

After creating or updating tutorial content:

1. Verify all code examples compile/run
2. Check that exercises have solutions
3. Ensure ASCII diagrams render correctly
4. Validate all file paths exist
5. Test profiling commands

## Reference Files

For additional context on tutorial standards:
- [reference.md](reference.md) - Detailed formatting guidelines
- [templates/](templates/) - Starter templates for new content
