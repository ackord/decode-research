import json

from candidate.inference import generate as candidate_generate
from harness.config import (
    BENCH_CONTEXT_TOKENS,
    BENCH_NEW_TOKENS,
    BENCH_TEXT,
    EVAL_NEW_TOKENS,
    EVAL_PROMPTS,
)
from harness.model import load_model
from reference.inference import generate as reference_generate


def compare(reference_model, candidate_model, token_ids, max_new_tokens):
    expected = reference_generate(
        reference_model,
        token_ids,
        max_new_tokens,
    )
    actual = candidate_generate(
        candidate_model,
        token_ids,
        max_new_tokens,
    )
    return actual == expected


def run_eval():
    reference_model, tokenizer = load_model()
    candidate_model, _ = load_model()

    cases = []

    for prompt in EVAL_PROMPTS:
        token_ids = tokenizer.encode(prompt)
        cases.append({
            "prompt": prompt,
            "correct": compare(
                reference_model,
                candidate_model,
                token_ids,
                EVAL_NEW_TOKENS,
            ),
        })

    if all(case["correct"] for case in cases):
        benchmark_ids = tokenizer.encode(BENCH_TEXT)
        if len(benchmark_ids) < BENCH_CONTEXT_TOKENS:
            raise RuntimeError("Benchmark text is too short")
        cases.append({
            "prompt": "<benchmark>",
            "correct": compare(
                reference_model,
                candidate_model,
                benchmark_ids[:BENCH_CONTEXT_TOKENS],
                BENCH_NEW_TOKENS,
            ),
        })

    return {
        "correct": all(case["correct"] for case in cases),
        "cases": cases,
    }


def main():
    result = run_eval()

    print(json.dumps(result, indent=2))

    if not result["correct"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
