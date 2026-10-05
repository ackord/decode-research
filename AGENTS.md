# AGENTS.md

## Goal

Make autoregressive decode of the fixed target model faster while producing exactly the same greedy output as the reference.

Decode throughput is the metric. Correctness is a hard gate. Prefill is not scored.

Prefer methods that could matter for frontier-scale decoder-only LLMs.

## Research boundary

The target model is fixed. Do not change its weights, tokenizer, architecture, or training.

The inference procedure may change inside `candidate/`, including auxiliary inference-time components, provided the target model stays fixed and the final tokens exactly match reference greedy decoding.

Auxiliary components may be separately trained or pretrained, but must not use benchmark prompts or benchmark results for training or tuning.

Internal approximation is allowed only when the emitted tokens remain exact.

Each benchmark decode is an independent request. Do not reuse generated outputs, KV state, hidden state, or other request-specific computation across requests.

Prefill may process the prompt and construct prompt-derived state. Do not generate, verify, or compute later output tokens before decode begins. All inference work that depends on generated tokens counts as decode work.

The claimed optimization must be hardware-independent. MLX may implement it, but the speedup must not depend on MLX, Metal, Apple Silicon, compiler behavior, custom kernels, or benchmark-specific tricks.

Only the implementer may modify `candidate/` during a round. Do not modify `reference/`, `harness/`, benchmark inputs, or benchmark rules.

## Research loop

Each round:

1. Research the strongest worthwhile experiment, or abandon.
2. Independently review and improve the research result.
3. Implement the reviewed experiment.
4. Pass exact-token evaluation.
5. Benchmark against the current best.
6. Keep only a clear improvement.
7. Record the round as evidence for future research.

The researcher searches broadly, inspects the actual implementation, considers alternatives internally, and returns at most one experiment.

The reviewer checks the proposal against the research boundary, current code, and prior evidence. It should repair or sharpen a sound mechanism rather than replace it for novelty. It may abandon a proposal only for a fundamental flaw that cannot be repaired soundly.

Neither research nor review is a substitute benchmark. When a sound, practical mechanism removes substantial decode work and its performance is genuinely uncertain, let evaluation and benchmarking decide.

The implementer builds the reviewed experiment. The orchestrator owns authoritative correctness and performance validation. Implementation-environment limitations are not experiment results.

## Research history

Use prior rounds as evidence, not as a blacklist.

A mechanism is strong negative history when it was actually tested and failed, or when review or implementation established a fundamental flaw. Do not generalize a narrow failure into a ban on a broader untested direction.

Preserve promising directions that have not actually been resolved. Use accumulated evidence to choose the strongest next experiment.

## Style

Keep research, code, and reports short and clear.

Prefer the simplest sound experiment.
