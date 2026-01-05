---
description: Add a code repository as a git submodule for analysis
allowed-tools: Bash, Read, Write, Edit
---

Add a code repository to the knowledge base as a git submodule.

**Repository:** $ARGUMENTS

## Instructions

1. Parse the repository URL/identifier:
   - Full URL: `https://github.com/org/repo`
   - Short form: `org/repo` (assumes GitHub)

2. Determine submodule name:
   - Use repo name by default
   - Or extract from user input if specified

3. Add as git submodule:
   ```bash
   git submodule add <url> code-repos/<name>
   ```

4. Update REFERENCES.md with entry in Code Repositories section:
   - Repository name
   - URL
   - Language/framework
   - Brief description (fetch from README if available)
   - Topics/tags

5. Optionally initialize for analysis:
   - If large repo, suggest specific subdirectories to focus on
   - Mention `/analyze-llm code-repos/<name>` for deep analysis

## Output

Confirm:
- Submodule added successfully
- Path: `code-repos/<name>`
- Next steps for analysis
