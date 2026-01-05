---
description: Create or update a note on a topic
allowed-tools: Read, Write, Edit, Glob
---

Create or update a note on the specified topic.

**Topic/Title**: $ARGUMENTS

## Instructions

1. Check if a note already exists:
   - Search `notes/topics/` for existing note
   - Search `notes/quick-ref/` for quick references

2. If creating a new note:
   - Determine appropriate location:
     - `notes/topics/` for comprehensive topic notes
     - `notes/quick-ref/` for quick references, cheat sheets
   - Use kebab-case filename: `topic-name.md`

3. Note structure for topics:

```markdown
# [Topic Name]

## Overview
Brief introduction

## Key Concepts
- Concept 1: explanation
- Concept 2: explanation

## Details
In-depth information

## Examples
Practical examples or code

## Related
- Links to related notes
- Links to papers in knowledge base

## References
Sources and citations
```

4. Note structure for quick-ref:

```markdown
# [Topic] Quick Reference

## Common Commands/Patterns
```code
examples
```

## Key Points
- Point 1
- Point 2

## Gotchas
Common mistakes to avoid

## See Also
Links to detailed notes
```

5. If updating an existing note:
   - Add new information to appropriate section
   - Update "Last updated" if present
   - Maintain existing structure

## Output

Confirm the note location and summarize what was added.
