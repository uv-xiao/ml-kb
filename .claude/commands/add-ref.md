---
description: Add a new reference to the knowledge base
allowed-tools: Read, Write, Edit, WebFetch
---

Add a new reference to REFERENCES.md with proper metadata.

**Reference to add**: $ARGUMENTS

## Instructions

1. Determine the reference type:
   - `paper`: Academic paper, preprint, technical report
   - `repo`: Code repository (GitHub, GitLab, etc.)
   - `blog`: Blog post, tutorial, article
   - `doc`: Documentation, API reference
   - `book`: Book or book chapter
   - `dataset`: Dataset resource

2. Extract metadata:
   - Title
   - Authors/creators
   - Year/date
   - URL or DOI
   - Topics/tags
   - Brief description (1-2 sentences)

3. If a URL is provided, fetch the page to extract accurate metadata

4. Add to the appropriate section in REFERENCES.md

5. Create a note file if requested or if the reference is significant:
   - Papers: `papers/YYYY-author-short-title.md`
   - Repos: `code-repos/repo-name.md`

## Output

Confirm what was added with the extracted metadata.
