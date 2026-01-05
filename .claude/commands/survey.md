---
description: Generate a literature survey on a topic
allowed-tools: Read, Glob, Grep, WebSearch, WebFetch
---

Generate a literature survey on the specified topic.

**Topic**: $ARGUMENTS

## Instructions

1. Scope the survey:
   - Define the topic boundaries
   - Identify key subtopics
   - Determine time range (recent vs. historical)

2. Gather sources:
   - Search existing references in REFERENCES.md
   - Look for related notes in papers/ and notes/
   - Optionally search web for recent developments

3. Organize the survey:

```markdown
# Survey: [Topic]

## Overview
Brief introduction to the topic and survey scope

## Background
Key concepts and terminology

## Taxonomy
Classification of approaches/methods

## Key Developments

### [Subtopic 1]
- Key papers and contributions
- Evolution of approaches

### [Subtopic 2]
...

## Current State of the Art
Best performing methods/latest developments

## Open Challenges
Unsolved problems and research gaps

## Future Directions
Promising research avenues

## References
List of cited works
```

4. Include:
   - Citations to papers in the knowledge base
   - Comparison tables where helpful
   - Timeline of key developments if relevant

## Output

Save to `reports/surveys/YYYY-MM-{topic}-survey.md`
