<div align="center">

# Evalkit-bench

**Production-grade LLM evaluation and benchmark harness**

[![Tests](https://github.com/Arman176001/evalkit/actions/workflows/test.yml/badge.svg)](https://github.com/Arman176001/evalkit/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/evalkit-bench)](https://pypi.org/project/evalkit-bench/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Point it at a YAML test suite → get a live terminal table, a self-contained HTML report, regression tracking, and side-by-side model comparisons.

</div>

---

## What it does

```
$ evalkit run evals/factual.yaml --model gpt-4o-mini

  ✓  [1/5]  capital-france
  ✓  [2/5]  capital-japan
  ✓  [3/5]  cell-powerhouse
  ✗  [4/5]  speed-of-light
  ✓  [5/5]  pythagorean

╭──────────────────┬──────────────────────────┬──────────────┬──────────┬────────╮
│ ID               │ Prompt                   │ Output       │ Scorers  │ Result │
├──────────────────┼──────────────────────────┼──────────────┼──────────┼────────┤
│ capital-france   │ What is the capital of…  │ Paris        │ exact ✓  │  PASS  │
│ speed-of-light   │ What is the speed of…    │ 299,792 km/s │ exact ✗  │  FAIL  │
╰──────────────────┴──────────────────────────┴──────────────┴──────────┴────────╯

  ████████████████░░░░  80%  4/5 passed · avg 340ms · 1,200 tokens

  Diff vs previous: 1 regression
  HTML report → .evalkit/runs/20260505T120000_factual-v1_report.html
```

---

## Install

```bash
pip install evalkit-bench
```

Add the embedding scorer (optional, ~1 GB download):

```bash
pip install "evalkit-bench[embed]"
```

Set your API keys — create a `.env` file in your project:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Quick start

**Run a suite:**
```bash
evalkit run evals/example_factual.yaml
```

**Override model at runtime:**
```bash
evalkit run evals/example_factual.yaml --model gpt-4o
evalkit run evals/example_factual.yaml --model claude-haiku-4-5-20251001
```

**Compare multiple models side-by-side:**
```bash
evalkit compare evals/example_factual.yaml \
  --models gpt-4o-mini,gpt-4o,claude-haiku-4-5-20251001
```

```
╭──────────────────┬─────────────┬─────────┬──────────────╮
│ Case             │ gpt-4o-mini │  gpt-4o │  haiku-4-5   │
├──────────────────┼─────────────┼─────────┼──────────────┤
│ capital-france   │    PASS     │  PASS   │     PASS     │
│ cell-powerhouse  │    PASS     │  PASS   │     FAIL     │
│ speed-of-light   │    PASS     │  PASS   │     PASS     │
├──────────────────┼─────────────┼─────────┼──────────────┤
│ Pass rate        │    100%     │  100%   │     80%      │
│ Avg latency      │    340ms    │  820ms  │    280ms     │
│ Total tokens     │   1,200     │  2,400  │     950      │
╰──────────────────┴─────────────┴─────────┴──────────────╯
```

---

## Writing a benchmark suite

Suites are plain YAML files:

```yaml
name: "factual-qa-v1"
description: "Tests basic factual recall"
model: "gpt-4o-mini"             # default model
judge_model: "claude-haiku-4-5-20251001"  # used by llm_judge scorer
temperature: 0.0
max_tokens: 512

cases:
  - id: capital-france
    prompt: "What is the capital of France? Reply with just the city name."
    expected: "Paris"
    scorers: [exact]
    tags: [geography, easy]

  - id: summarise-article
    prompt: "Summarise in 2 sentences: {input}"
    input: "The Eiffel Tower was built between 1887 and 1889..."
    expected: "The Eiffel Tower was built in the late 1880s."
    scorers: [llm_judge, embed]
    rubric: "Does the summary capture the key facts without hallucinating?"
    tags: [summarisation, medium]
```

### All case fields

| Field | Required | Description |
|-------|----------|-------------|
| `id` | ✅ | Unique identifier within the suite |
| `prompt` | ✅ | Prompt sent to the model. Supports `{input}` substitution |
| `expected` | — | Reference answer (required by `exact`, `contains`, `embed`) |
| `input` | — | Replaces `{input}` in the prompt |
| `scorers` | — | List of scorers (default: `[exact]`) |
| `rubric` | — | Evaluation guidance for `llm_judge` |
| `tags` | — | Arbitrary labels for filtering |
| `pattern` | — | Regex pattern for `exact` scorer |

---

## Scorers

| Scorer | Passes when | Best for |
|--------|-------------|----------|
| `exact` | Output matches expected (case-insensitive). Exact → 1.0, substring → 0.5 | Short deterministic answers |
| `contains` | Expected string appears anywhere in output | Keywords, function names |
| `llm_judge` | Claude rates the output ≥ 3.5 / 5 | Open-ended quality, summarisation |
| `embed` | Cosine similarity of sentence embeddings ≥ 0.7 | Semantic equivalence |

Multiple scorers are **AND-ed** — a case passes only if every scorer passes.

---

## Regression tracking

Every run is saved automatically to `.evalkit/runs/`. After each run, evalkit diffs against the previous run for the same suite:

```
Diff vs previous: 2 regressions, 1 improvement
```

**Compare any two runs manually:**
```bash
evalkit diff .evalkit/runs/20260505T090000_factual-v1.json \
             .evalkit/runs/20260505T100000_factual-v1.json
```

**List all runs:**
```bash
evalkit list-runs
```

**Inspect a specific run:**
```bash
evalkit show 20260505T120000
```

---

## CLI reference

```
evalkit run     <suite.yaml>  [--model MODEL] [--output-dir DIR] [--no-report]
evalkit compare <suite.yaml>  --models MODEL1,MODEL2,...
evalkit diff    <run_a.json>  <run_b.json>
evalkit list-runs             [--runs-dir DIR]
evalkit show    <run_id>      [--runs-dir DIR]
```

> **CI-friendly:** `evalkit run` exits with `0` if all cases pass, `1` if any fail.

---

## HTML report

Every run generates a self-contained HTML report — no CDN, works offline, sendable as an attachment.

- Pass/fail per case with color coding
- Click any row to expand full prompt, output, and judge reasoning
- Regression and improvement badges when diffing against a previous run
- Overall pass-rate bar and summary stats

---

## Roadmap

- [ ] Gemini and Ollama (local) providers
- [ ] Parallel case execution
- [ ] Web UI dashboard for run history
- [ ] Custom scorer plugins via entry points
- [ ] `evalkit init` — interactive suite generator

---

## Contributing

```bash
git clone https://github.com/Arman176001/evalkit
cd evalkit
pip install -e ".[dev]"
python -m pytest tests/ -v
```

All 44 tests run in under 3 seconds with no API calls required.

---

<div align="center">

Made by [Arman](https://github.com/Arman176001) · MIT License

</div>
