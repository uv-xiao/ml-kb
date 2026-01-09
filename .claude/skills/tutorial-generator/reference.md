# Tutorial Generator Reference

## Detailed Formatting Guidelines

### Markdown Standards

#### Headers
- Use `#` for chapter titles
- Use `##` for major sections
- Use `###` for subsections
- Use `####` sparingly for minor divisions

#### Code Blocks
Always specify language for syntax highlighting:

```python
# Python example
def function():
    pass
```

```cuda
// CUDA example
__global__ void kernel() {
    // ...
}
```

```bash
# Shell commands
ncu --set full kernel.cu
```

### ASCII Diagram Components

#### Boxes
```
┌────────┐     ╔════════╗
│ Normal │     ║ Double ║
└────────┘     ╚════════╝
```

#### Arrows
```
──────▶  (right arrow)
◀──────  (left arrow)
   │
   │     (down arrow)
   ▼

   ▲
   │     (up arrow)
   │
```

#### Flow Connections
```
├── (branch)
└── (end branch)
│   (vertical line)
───  (horizontal line)
```

### Table Format

Use GFM-style tables:

| Component | File | Description |
|-----------|------|-------------|
| Kernel A  | `a.cu` | Description |
| Kernel B  | `b.cu` | Description |

### Code Position Format

Always use this format for code references:

```markdown
**File:** `path/to/file.extension`
**Lines:** 123-145

Key implementation:
- Line 125: Main computation
- Line 130: Kernel launch
```

## Template: New Chapter

```markdown
# Chapter XX: Topic Name

## Learning Objectives

By the end of this chapter, you will:
- Understand [concept 1]
- Implement [skill 1]
- Profile and optimize [component]

## Prerequisites

- Completed Chapter XX-1
- Familiarity with [technology]
- Hardware: [requirements]

## Overview

[2-3 paragraphs introducing the topic with ASCII diagram]

## Section 1: Concept Name

### Theory

[Explanation with diagrams]

### Implementation

[Code with explanations]

### Hands-On: Exercise Name

**Objective:** [What to accomplish]

**Starter Code:** `exercises/01_name/starter.cu`

**Hints:**
1. Consider using [approach]
2. Pay attention to [aspect]

## Section 2: ...

## Summary

Key takeaways:
- [Point 1]
- [Point 2]

## Next Steps

- Chapter XX+1: [Next topic]
- Additional reading: [Resources]
```

## Template: Example README

```markdown
# Example: [Name]

## Overview

This example demonstrates [functionality].

## Key Concepts

- [Concept 1]: Brief explanation
- [Concept 2]: Brief explanation

## Code Structure

```
example_name/
├── main.cu         # Entry point
├── kernel.cuh      # Kernel implementation
└── CMakeLists.txt  # Build configuration
```

## Building

```bash
mkdir build && cd build
cmake ..
make
```

## Running

```bash
./example_name
```

## Expected Output

```
[Sample output here]
```

## Profiling

```bash
ncu --set full ./example_name
```

## Key Metrics to Observe

- Memory bandwidth utilization
- SM occupancy
- [Other relevant metrics]

## Further Exploration

Try modifying:
- [Suggestion 1]
- [Suggestion 2]
```

## Template: Exercise Problem

```markdown
# Exercise: [Name]

## Difficulty: [Easy/Medium/Hard]

## Learning Goal

Practice [skill] by implementing [task].

## Problem Statement

[Clear description of what to implement]

## Input Specification

- [Input 1]: [Type and description]
- [Input 2]: [Type and description]

## Output Specification

- [Output]: [Type and description]

## Constraints

- [Constraint 1]
- [Constraint 2]

## Starter Code

See `starter.cu` for the template.

## Hints

<details>
<summary>Hint 1</summary>
[First hint]
</details>

<details>
<summary>Hint 2</summary>
[Second hint]
</details>

## Testing

Run the test script:
```bash
python test.py
```

## Performance Target

Achieve at least [X]% of [baseline] performance.
```

## Source Material Mapping

When converting analysis reports to tutorials:

| Report Section | Tutorial Section |
|----------------|------------------|
| Architecture Overview | Chapter Introduction |
| Kernel Catalog | Reference Tables |
| ASCII Diagrams | Concept Explanations |
| Code Positions | Example References |
| Profiling Commands | Hands-On Exercises |
| Optimization Opportunities | Advanced Exercises |

## Quality Checklist

Before finalizing tutorial content:

- [ ] All code examples compile without errors
- [ ] Expected outputs are accurate
- [ ] ASCII diagrams render correctly in GitHub
- [ ] Links to files are valid
- [ ] Exercises have working solutions
- [ ] Profiling commands are tested
- [ ] Prerequisites are clearly stated
- [ ] Learning objectives are specific and measurable
