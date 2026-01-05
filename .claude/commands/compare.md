---
description: Compare multiple papers, methods, or approaches
allowed-tools: Read, Glob, Grep
---

Create a comparison of the specified items.

**Items to compare**: $ARGUMENTS

## Instructions

1. Identify what is being compared:
   - Papers (by title, file, or reference)
   - Methods/algorithms
   - Implementations/repositories
   - Frameworks/tools

2. Determine comparison dimensions:
   - For papers: contributions, methods, results, datasets
   - For methods: approach, complexity, performance, use cases
   - For repos: features, architecture, dependencies, activity

3. Generate a structured comparison:

### Comparison Table

| Item | Aspect 1 | Aspect 2 | Aspect 3 |
|------|----------|----------|----------|
| A    | ...      | ...      | ...      |
| B    | ...      | ...      | ...      |

### Detailed Analysis

For each comparison dimension, provide:
- How each item approaches it
- Relative strengths and weaknesses
- Trade-offs

### Recommendations

When to use each approach based on different scenarios.

## Output Format

Return a markdown document suitable for saving to `reports/comparisons/`.
