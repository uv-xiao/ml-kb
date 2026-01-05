# Repository Analysis Skill

Analyze ML/LLM code repositories to understand architecture, trace implementations, and generate technical documentation.

## When to Use

This skill auto-triggers when:
- Exploring a code repository structure
- Understanding model implementations
- Tracing training or inference pipelines
- Generating documentation for codebases
- Finding specific implementations (attention, loss functions, etc.)

## Capabilities

### Architecture Analysis
- Map repository structure and organization
- Identify key modules and their relationships
- Trace data flow and dependencies
- Document design patterns used

### Implementation Tracing
- Find specific algorithm implementations
- Trace model forward/backward passes
- Understand training loops and optimizers
- Map configuration to code

### Documentation Generation
- Create README overviews
- Generate API documentation
- Write implementation guides
- Produce architecture diagrams (as text)

## Common Repository Patterns

### Training Pipeline
```
repo/
├── config/           # Hyperparameters, model configs
├── data/             # Dataset loading, preprocessing
├── models/           # Model architectures
├── train.py          # Training entry point
├── eval.py           # Evaluation scripts
└── utils/            # Helper functions
```

### Hugging Face Style
```
repo/
├── src/model_name/
│   ├── modeling_*.py     # Model implementations
│   ├── configuration_*.py # Config classes
│   └── tokenization_*.py  # Tokenizers
└── examples/              # Usage examples
```

### Research Code
```
repo/
├── experiments/      # Experiment configs
├── src/              # Core implementation
├── scripts/          # Training/eval scripts
└── notebooks/        # Analysis notebooks
```

## Key Files to Examine

| File Pattern | Purpose |
|-------------|---------|
| `train.py`, `main.py` | Entry points |
| `model*.py`, `models/` | Model definitions |
| `config*.py`, `*.yaml` | Configuration |
| `data*.py`, `dataset*.py` | Data loading |
| `requirements.txt`, `setup.py` | Dependencies |
| `README.md` | Documentation |

## Example Queries

- "What's the structure of this repository?"
- "Where is the attention mechanism implemented?"
- "How does the training loop work?"
- "What dependencies does this project use?"
- "Generate a technical overview of this codebase"
- "Trace the forward pass of the model"
- "What hyperparameters are configurable?"

## Output Formats

**Architecture Overview**:
```markdown
# Repository: name

## Structure
- Module A: purpose
- Module B: purpose

## Key Components
1. Component description

## Data Flow
input -> processing -> output

## Dependencies
- dep1: purpose
- dep2: purpose
```

**Implementation Trace**:
```markdown
# Function/Class Name

## Location
file:line

## Purpose
What it does

## Key Logic
Step-by-step explanation

## Dependencies
What it calls/uses
```

## Best Practices

1. Start with high-level structure before diving into details
2. Ask about specific components for targeted analysis
3. Request traces for understanding complex logic
4. Generate docs incrementally for large codebases
