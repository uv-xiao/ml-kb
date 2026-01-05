---
description: Search the knowledge base for a topic or keyword
allowed-tools: Read, Glob, Grep
---

Search the knowledge base for the specified query.

**Search query**: $ARGUMENTS

## Instructions

1. Search across all knowledge base locations:
   - `REFERENCES.md` - master reference list
   - `papers/` - paper notes and summaries
   - `notes/` - topic notes and quick references
   - `code-repos/` - repository analyses
   - `reports/` - generated reports

2. Search methods:
   - Keyword matching in titles and content
   - Topic/tag matching
   - Author name matching (for papers)

3. Return results organized by type:

```markdown
## Search Results: "[query]"

### References
- [Title](link) - brief description

### Papers
- [Paper title](file) - key finding

### Notes
- [Note title](file) - relevant section

### Reports
- [Report title](file) - relevant content

### Code Repos
- [Repo name](file) - relevant info
```

4. For each result, include:
   - Title/name with link to file
   - Brief context showing why it matched
   - Most relevant snippet if applicable

5. Suggest related searches if results are limited

## Output

Return formatted search results with links to files.
