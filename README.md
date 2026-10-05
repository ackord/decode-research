# Decode Research

Autonomous research on exact autoregressive decode optimization.

The goal is to make decode of a fixed target model faster without changing its greedy output.

## Scope

- Decoder-only LLM inference
- Autoregressive decode only
- Exact final output required
- Fixed target-model weights, tokenizer, architecture, and training
- Inference-time auxiliary components are allowed
- No hardware-specific, compiler, kernel, or benchmark tricks
- Ideas should plausibly transfer to frontier-scale models

The current testbed uses `mlx-community/SmolLM2-135M-Instruct` at a pinned revision.

Prefill is not scored. It may process the prompt and construct prompt-derived state, but it may not move future-token generation or verification ahead of the decode timer. All inference work that depends on generated tokens counts as decode work.

Each benchmark decode is an independent request. Request-specific generated state may not be reused across requests.

## Research loop

```text
research one experiment
    ↓
independent review and refinement
    ↓
implement
    ↓
exact-token evaluation
    ↓
benchmark against current best
    ↓
accept or reject
    ↓
record evidence for the next round
```

Each round starts with one GPT-6 Astra researcher at max reasoning. It inspects the current implementation and accumulated evidence, searches broadly, and returns at most one worthwhile experiment.

A fresh GPT-6 Astra reviewer at max reasoning then tries to falsify and improve that result. It checks correctness assumptions, boundaries, evidence, hidden work, generalization, and implementability. It can rewrite the proposal and plan or abandon a fundamentally unsound mechanism. It does not predict benchmark outcome.

GPT-6.1 Sol implements the reviewed experiment. The orchestrator performs the authoritative exact-token evaluation and performance measurements.

Accepted changes become the next baseline.

## Evaluation

Correctness is a hard gate. Exact evaluation checks the fixed prompts and, for candidates that pass them, the full benchmark workload before any timing.

The benchmark measures decode only and compares candidate and reference implementations back-to-back with alternating order. A candidate is accepted only when it is at least **3% faster in both confirmation runs**.

## Research history

Each completed round is stored under `experiments/`:

```text
experiments/0001/
├── report.json
└── patch.diff
```

`report.json` records the researcher output, reviewer output, implementation result, exact evaluation, benchmark measurements, and final decision. `patch.diff` preserves tested code when implementation was attempted.

Agent results are sanitized before handoff or persistence: repository paths become relative, and other absolute filesystem paths are redacted.

History is evidence, not a blanket exclusion list. Tested failures and fundamental review or implementation findings should prevent repetition of the same resolved mechanism without excluding broader untested directions.

## Run

```bash
uv sync --frozen
uv run --frozen python -m orchestration.orchestrator
```

Use `--once` to run a single research round.
