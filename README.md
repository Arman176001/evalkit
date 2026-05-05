# evalkit

A production-grade LLM evaluation and benchmark harness. Point it at a YAML test suite, get a rich terminal table, a self-contained HTML report, a JSON results file for regression tracking, and a diff against the last run.

## Install

```bash
pip install -e .
```

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

## Quick start

```bash
evalkit run evals/example_factual.yaml
```

Override the model at runtime:

```bash
evalkit run evals/example_factual.yaml --model claude-haiku-4-5-20251001
evalkit run evals/example_code.yaml    --model gpt-4o
```

The command prints a live progress ticker, then a results table, saves a JSON run file, and opens an HTML report in your browser.

## YAML reference

```yaml
name: "my-suite"                        # required — used in filenames and reports
description: "Optional one-liner"
model: "gpt-4o-mini"                    # default model for this suite
judge_model: "claude-haiku-4-5-20251001"  # model used by the llm_judge scorer
temperature: 0.0
max_tokens: 512
system: null                            # optional system prompt for every case

cases:
  - id: my-case                         # required, must be unique within the suite
    prompt: "What is {input}?"          # supports {input} substitution
    input: "the capital of France"      # replaces {input} in prompt
    expected: "Paris"                   # reference answer (required by exact/contains/embed)
    scorers: [exact]                    # list of scorer names (see below)
    rubric: "Is the answer correct?"    # guidance for llm_judge
    tags: [geography, easy]             # arbitrary labels for filtering
    pattern: null                       # regex pattern for ExactScorer
```

## Scorers

| Name | Passes when | Use for |
|------|-------------|---------|
| `exact` | Output exactly matches expected (case-insensitive). Substring match gives partial credit (0.5) but doesn't pass at default threshold. | Short, deterministic answers |
| `contains` | Expected string appears anywhere in output | Keywords, code snippets |
| `llm_judge` | Claude scores the output ≥ 3.5/5 | Open-ended, subjective quality |
| `embed` | Cosine similarity of sentence embeddings ≥ 0.7 | Semantic equivalence |

Multiple scorers per case are AND-ed: a case passes only if **all** scorers pass.

## CLI reference

```
evalkit run <suite.yaml> [--model MODEL] [--output-dir DIR] [--no-report]
evalkit diff <run_a.json> <run_b.json>
evalkit list-runs [--runs-dir DIR]
evalkit show <run_id> [--runs-dir DIR]
```

`evalkit run` exit codes: `0` = all passed, `1` = any failed — plugs straight into CI.

## Regression tracking

Every run is saved to `.evalkit/runs/<timestamp>_<suite_name>.json`. After each run, evalkit automatically diffs against the previous run for the same suite and prints:

```
Diff vs previous: 2 regressions, 1 improvement
```

Compare any two runs manually:

```bash
evalkit diff .evalkit/runs/20260505T090000_factual-qa-v1.json \
             .evalkit/runs/20260505T100000_factual-qa-v1.json
```

List all saved runs:

```bash
evalkit list-runs
```

## Roadmap

- Web UI dashboard for run history
- Gemini and local (Ollama) providers
- Custom scorer plugins via entry points
- Parallel case execution
- CI GitHub Action
