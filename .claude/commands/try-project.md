# Try Project Command

Hands-on exploration of an LLM serving project with profiling and analysis.

## Usage

```
/try-project <project-path-or-name> [target]
```

**Arguments:**
- `project-path-or-name`: Path to project (e.g., `code-repos/flashinfer`) or name (e.g., `vLLM`)
- `target`: Optional learning objective (e.g., "understand attention kernel", "benchmark decode latency")

## What This Command Does

1. **Detects environment** (GPUs, CUDA, dependencies)
2. **Explores codebase** (entry points, architecture, docs)
3. **Creates try-using plan** tailored to your hardware and target
4. **Executes experiments** with profiling
5. **Analyzes results** (traces, metrics)
6. **Generates report** with findings

## Example

```
/try-project code-repos/flashinfer "understand paged attention performance"
```

This will:
- Detect your GPU configuration
- Read FlashInfer docs and code
- Create experiments for paged attention
- Run benchmarks with profiling
- Analyze kernel traces
- Generate a comprehensive report

## Output

The command produces:
- `reports/hands-on/<project>-<date>.md` - Comprehensive learning report
- Profiling artifacts in temporary directory
- Console output with progress and findings
