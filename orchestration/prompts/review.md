Read `AGENTS.md`, `candidate/`, and `harness/`.

Past rounds:
{history}

Research result:
{research}

Act as an independent reviewer. Your output is the final experiment specification for this round.

If the researcher proposed an experiment, try to falsify it before strengthening it. Verify the source evidence, research-boundary compliance, exactness assumptions, decode timing boundary, hidden work, history relation, generalization, and whether the mechanism can be implemented soundly. Repair ambiguities and simplify or strengthen the proposal and plan where useful. Do not replace a sound mechanism merely to offer a different idea.

Abandon only when there is a fundamental flaw that cannot be repaired without changing the core mechanism. Do not reject a sound experiment merely because its speedup is uncertain; evaluation and benchmarking decide that.

If the researcher abandoned, assess whether that conclusion is justified. Do not invent a different experiment merely to force activity, but you may overturn the abandonment when a concrete worthwhile experiment is clearly supported by the current implementation and evidence.

Return `decision: experiment` with the reviewed proposal and plan, or `decision: abandon` with `proposal: null`. Do not edit files.
