Read `AGENTS.md`.

Implement the reviewed experiment below.

Your role is to build the experiment, not to decide whether it passes evaluation or benchmarking. The orchestrator performs authoritative correctness and performance validation after you return.

You may edit only `candidate/`. Do not commit or change git history. Keep the implementation small and clear, and preserve the reviewed mechanism.

Return `outcome: abandon` only if the mechanism cannot be implemented soundly within the project boundaries after investigation. Limitations of your execution environment are not experiment results and are not reasons to abandon.

If implementation is sound, return `outcome: implemented` even if you cannot execute the model or benchmark locally. Do not make placeholder edits.

An implemented experiment must change `candidate/inference.py` so the orchestrator's eval and benchmark exercise it. Helper files are allowed only if `candidate/inference.py` uses them. Do not add experiment reports or standalone demo files; the orchestrator records the round.

You may use the network when needed for implementation research or dependencies.

Experiment:
{proposal}

Plan:
{plan}
