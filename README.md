# ML/LLM Knowledge Base

A personal research knowledge base for Machine Learning and Large Language Models, powered by Claude Code.

## Overview

This repository organizes research materials (papers, code, documentation) and provides Claude Code skills and commands to facilitate exploration, analysis, and report generation.

## Directory Structure

```
mlkb/
├── papers/                # Research papers and paper notes
├── notes/
│   ├── topics/            # In-depth topic notes
│   └── quick-ref/         # Quick reference sheets
├── code-repos/            # Code repository analyses
├── reports/
│   ├── surveys/           # Literature surveys
│   ├── comparisons/       # Comparative analyses
│   └── implementations/   # Implementation guides
├── REFERENCES.md          # Master reference index
├── CLAUDE.md              # Claude Code project config
└── .claude/
    ├── skills/            # Auto-triggering skills
    └── commands/          # Slash commands
```

## Quick Start

### Adding References

```
/add-ref https://arxiv.org/abs/1706.03762
/add-ref paper "Attention Is All You Need"
/add-ref repo https://github.com/huggingface/transformers
```

### Analyzing Papers

Drop a PDF or provide a URL, then ask:
- "Summarize the key contributions"
- "What methods does this paper use?"
- "Extract the main results"

### Analyzing Code Repositories

Provide a repo URL or path:
- "What's the architecture of this repository?"
- "Where is the attention mechanism implemented?"
- "Generate a technical overview"

### Creating Notes

```
/note transformers
/note quick-ref pytorch-tips
```

### Generating Reports

```
/survey attention mechanisms
/compare BERT GPT-2 T5
```

### Searching

```
/search attention
/search transformer architecture
```

## Slash Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/add-ref` | Add reference to REFERENCES.md | `/add-ref https://arxiv.org/...` |
| `/compare` | Compare papers/methods/repos | `/compare BERT GPT T5` |
| `/survey` | Generate literature survey | `/survey LLM alignment` |
| `/note` | Create or update a note | `/note fine-tuning` |
| `/search` | Search the knowledge base | `/search attention` |

## Skills (Auto-Triggered)

These skills activate automatically based on context:

### paper-analysis
Triggers when analyzing research papers. Extracts:
- Problem statement and contributions
- Methods and algorithms
- Results and benchmarks
- Limitations and future work

### repo-analysis
Triggers when exploring code repositories. Provides:
- Architecture mapping
- Implementation tracing
- Dependency analysis
- Documentation generation

### technical-reports
Triggers when generating reports. Creates:
- Literature surveys
- Comparative analyses
- Implementation guides

## File Naming Conventions

### Papers
```
papers/YYYY-author-short-title.md
papers/2017-vaswani-attention-is-all-you-need.md
```

### Notes
```
notes/topics/transformer-architecture.md
notes/quick-ref/pytorch-distributed.md
```

### Reports
```
reports/surveys/2026-01-llm-alignment-survey.md
reports/comparisons/attention-mechanisms.md
reports/implementations/lora-fine-tuning-guide.md
```

### Code Repos
```
code-repos/transformers-analysis.md
code-repos/llama-architecture.md
```

## Reference Types

The knowledge base tracks these reference types in REFERENCES.md:

| Type | Description | Example |
|------|-------------|---------|
| paper | Academic papers, preprints | arXiv, conferences |
| repo | Code repositories | GitHub projects |
| blog | Blog posts, tutorials | Technical blogs |
| doc | Documentation, API refs | Official docs |
| book | Books, chapters | Textbooks |
| dataset | Dataset resources | Benchmarks |

## Workflows

### Literature Review Workflow

1. Add papers: `/add-ref [paper-url]`
2. Analyze each: "Summarize contributions of [paper]"
3. Compare: `/compare [paper1] [paper2]`
4. Synthesize: `/survey [topic]`

### Code Understanding Workflow

1. Add repo: `/add-ref repo [url]`
2. Explore: "What's the structure of this repo?"
3. Trace: "How does the training loop work?"
4. Document: "Generate architecture documentation"

### Research Note-Taking Workflow

1. Create topic note: `/note [topic]`
2. Add findings as you research
3. Link to papers and repos
4. Build quick-ref sheets for common patterns

## Tips

- Keep REFERENCES.md updated as your master index
- Use consistent file naming with dates
- Link related materials across notes
- Tag entries with topics for searchability
- Generate reports incrementally as you learn

## License

Personal research use.
