---
name: technical-reports
description: Generates comprehensive technical reports including literature surveys, comparative analyses, and implementation guides. Use when creating surveys, comparing approaches or implementations, writing technical guides, or synthesizing knowledge from multiple sources.
---

# Technical Reports Skill

Generate comprehensive technical reports including surveys, comparisons, and implementation guides.

## When to Use

This skill auto-triggers when:
- Creating literature surveys or state-of-the-art reviews
- Comparing multiple approaches, papers, or implementations
- Writing implementation guides or tutorials
- Synthesizing knowledge from multiple sources

## Report Types

### 1. Literature Survey
Comprehensive review of a research topic.

**Structure**:
```markdown
# Survey: [Topic]

## Abstract
Brief overview of the survey scope

## Introduction
- Background and motivation
- Scope and methodology
- Organization of the survey

## Background
Key concepts and preliminaries

## Taxonomy
Classification of approaches

## Methods Review
### Category A
- Method 1
- Method 2

### Category B
- Method 3

## Comparison
| Method | Approach | Strengths | Weaknesses |
|--------|----------|-----------|------------|

## Open Problems
Current challenges and gaps

## Future Directions
Promising research directions

## References
```

### 2. Comparative Analysis
Side-by-side comparison of approaches.

**Structure**:
```markdown
# Comparison: [Topic]

## Overview
What is being compared and why

## Candidates
Brief intro to each approach

## Comparison Dimensions
- Dimension 1: criteria
- Dimension 2: criteria

## Detailed Comparison

### Dimension 1
| Approach | Score/Rating | Notes |
|----------|--------------|-------|

### Dimension 2
...

## Summary Table
| Approach | Dim1 | Dim2 | Dim3 | Overall |
|----------|------|------|------|---------|

## Recommendations
When to use each approach

## Conclusion
```

### 3. Implementation Guide
Step-by-step technical implementation.

**Structure**:
```markdown
# Implementation Guide: [Topic]

## Overview
What we're implementing and why

## Prerequisites
- Required knowledge
- Dependencies
- Environment setup

## Architecture
High-level design

## Implementation

### Step 1: [Component]
```code
implementation
```
Explanation

### Step 2: [Component]
...

## Complete Code
Full working example

## Usage
How to use the implementation

## Testing
How to verify correctness

## Performance
Optimization tips

## Troubleshooting
Common issues and solutions

## References
```

## Example Queries

- "Create a survey of attention mechanisms in transformers"
- "Compare BERT, GPT, and T5 architectures"
- "Write an implementation guide for LoRA fine-tuning"
- "Summarize the state of the art in LLM alignment"
- "Compare training strategies for large language models"

## Output Guidelines

1. **Depth**: Match depth to the request (brief overview vs. comprehensive)
2. **Citations**: Include references to source papers/repos
3. **Tables**: Use comparison tables for clarity
4. **Code**: Include code examples in implementation guides
5. **Structure**: Follow consistent section organization

## Best Practices

1. Specify the report type and scope upfront
2. Provide source materials (papers, repos, notes)
3. Request specific depth (overview vs. deep dive)
4. Ask for particular sections if needed
5. Iterate on drafts for refinement
