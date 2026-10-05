Read `AGENTS.md`, `candidate/`, and `harness/`.

Past rounds:
{history}

Act as the researcher for the next round. Inspect the current decode path and search broadly for the strongest experiment that could make exact autoregressive decode faster while preserving reference greedy output.

Consider multiple directions internally. Do not start from a predefined mechanism family, optimization category, or search angle. Use prior rounds as evidence and prefer mechanisms that remove substantial decode work and could matter for frontier-scale decoder-only models.

Return `decision: experiment` only when there is one sound experiment worth testing. Otherwise return `decision: abandon` with `proposal: null`.

For an experiment, ground `source_evidence` in real current operations or data movement. Explain the work removed, generalization, main risk, relation to prior evidence, and a concrete implementation plan. Do not tune to benchmark prompts or results.

Return only the research result. Do not edit files.
