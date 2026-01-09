# ML/LLM Knowledge Base

A research knowledge base for Machine Learning and Large Language Models, managed with Claude Code.

## Project Structure

```
mlkb/
├── CLAUDE.md              # Project instructions (this file)
├── REFERENCES.md          # Master reference list
├── papers/                # Research papers (PDFs, notes, links)
├── notes/                 # Technical notes and summaries
│   ├── topics/            # Topic-organized notes
│   └── quick-ref/         # Quick reference materials
├── code-repos/            # Code repositories (git submodules)
├── tutorials/             # Learning tutorials (git submodules)
│   └── kernels/           # CUDA kernel development tutorial
├── hands_on/              # Hands-on learning sessions
│   └── <project>/         # Per-project learning materials
│       ├── scripts/       # Execution and profiling scripts
│       ├── results/       # Profiling outputs (gitignored)
│       └── reports/       # Analysis reports and guides
├── reports/               # Generated technical reports
│   ├── surveys/           # Literature surveys
│   ├── comparisons/       # Comparative analyses
│   └── implementations/   # Implementation guides
└── .claude/
    ├── skills/            # Custom Claude Code skills
    │   ├── paper-analysis/
    │   ├── repo-analysis/
    │   ├── technical-reports/
    │   ├── tutorial-generator/
    │   └── hands-on-learning/
    └── commands/          # Slash commands for quick tasks
```

## Reference Types

This knowledge base supports:
- **Papers**: Academic papers, preprints (arXiv), technical reports
- **Documents**: Blog posts, tutorials, documentation
- **Web pages**: Reference pages, API docs, online resources
- **Code repositories**: GitHub repos, implementations, frameworks

## Skills Available

### paper-analysis
Analyzes research papers to extract key contributions, methods, results, and limitations. Auto-triggers when discussing papers.

### repo-analysis
Analyzes ML/LLM code repositories to understand architecture, trace implementations, and generate documentation.

### llm-code-analysis
Deep-dive analysis of LLM/ML implementations (PyTorch, Triton, CUDA). Generates visual explanations, pseudocode abstractions, and technical tutorials with ASCII diagrams.

### technical-reports
Generates comprehensive technical reports including surveys, comparisons, and implementation guides.

### tutorial-generator
Creates comprehensive tutorials and learning guides from existing analysis notes, reports, and codebase documentation. Synthesizes technical materials into structured educational content with examples, ASCII diagrams, and practical exercises.

### hands-on-learning
Hands-on exploration and profiling of LLM serving projects. Analyzes codebases, detects environment (GPUs, CUDA), creates experiment plans, executes with hardware-level profiling (NSys, NCU), and generates comprehensive reports. All outputs organized in `hands_on/<project>/` with scripts, results (gitignored), and reports subdirectories.

## Tutorials

The `tutorials/` directory contains hands-on learning materials organized by topic:

### kernels (tutorials/kernels/)
Comprehensive CUDA kernel development tutorial covering:
- GPU architecture & CUDA basics (Chapters 1-3)
- CUTLASS & CuTe/CuteDSL (Chapters 4-5)
- Advanced CUDA & DSLs: Triton, TileLang (Chapters 6-8)
- LLM Kernels: Attention, MoE accelerators (Chapters 9-10)
- Capstone projects (Chapter 11)

### Adding/Updating Tutorials
Use `/update-tutorial` to update tutorials from existing analysis and reports, or create tutorials manually following the structure in each tutorial's README.

## Hands-On Learning

The `hands_on/` directory contains structured learning sessions for LLM serving projects:

```
hands_on/
├── sglang/              # SGLang inference framework
├── mini-sglang/         # Lightweight inference framework
└── flashinfer/          # FlashInfer attention kernels
```

### Running Hands-On Learning
Use the `hands-on-learning` skill to start a new session:
```
/hands-on-learning <project-name>
```

Each session creates:
- `scripts/` - Profiling and experiment scripts
- `results/` - Profiling outputs (gitignored)
- `reports/` - Analysis reports, development guides, and findings

## Common Tasks

### Adding a Reference
Use `/add-ref` to add a new reference with metadata to REFERENCES.md

### Analyzing a Paper
1. Add paper to `papers/` directory or reference by URL
2. Ask Claude to analyze: "Summarize the key contributions of [paper]"
3. The paper-analysis skill will auto-trigger

### Analyzing a Code Repository
1. Add repo as submodule: `git submodule add <url> code-repos/<name>`
2. Use `/analyze-llm code-repos/<name>` for deep analysis
3. Ask specific questions about architecture or implementations
4. Analysis reports saved to `reports/implementations/`

### Generating Reports
1. Reference source materials (papers, code, notes)
2. Specify report type: survey, comparison, or implementation guide
3. Get structured technical report

### Comparing Papers/Approaches
Use `/compare` to create comparison tables across multiple papers or methods

## File Naming Conventions

### Papers
- Format: `YYYY-author-short-title.pdf` or `.md` for notes
- Example: `2017-vaswani-attention-is-all-you-need.md`

### Notes
- Topic notes: `notes/topics/transformers.md`
- Quick refs: `notes/quick-ref/pytorch-tips.md`

### Reports
- Surveys: `reports/surveys/YYYY-MM-topic-survey.md`
- Comparisons: `reports/comparisons/topic-comparison.md`

## Best Practices

1. Keep REFERENCES.md updated as the master index
2. Use descriptive filenames with dates
3. Link related papers and implementations
4. Tag entries with topics for easy searching
5. Version reports with dates
