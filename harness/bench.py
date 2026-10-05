import json
import statistics
import time

import mlx.core as mx

import candidate.inference as candidate
from harness.config import BENCH_CONTEXT_TOKENS, BENCH_NEW_TOKENS, BENCH_TEXT, RUNS, WARMUPS
from harness.model import load_model
import reference.inference as reference


def benchmark_tokens(tokenizer):
    token_ids = tokenizer.encode(BENCH_TEXT)
    if len(token_ids) < BENCH_CONTEXT_TOKENS:
        raise RuntimeError("Benchmark text is too short")
    return token_ids[:BENCH_CONTEXT_TOKENS]


def run_once(implementation, model, token_ids):
    token, cache = implementation.prefill(model, token_ids)
    mx.synchronize()

    start = time.perf_counter()
    implementation.decode(model, token, cache, BENCH_NEW_TOKENS)
    mx.synchronize()
    seconds = time.perf_counter() - start

    return (BENCH_NEW_TOKENS - 1) / seconds


def main():
    model, tokenizer = load_model()
    token_ids = benchmark_tokens(tokenizer)

    for _ in range(WARMUPS):
        run_once(reference, model, token_ids)
        run_once(candidate, model, token_ids)

    ratios = []
    for index in range(RUNS):
        if index % 2:
            candidate_speed = run_once(candidate, model, token_ids)
            reference_speed = run_once(reference, model, token_ids)
        else:
            reference_speed = run_once(reference, model, token_ids)
            candidate_speed = run_once(candidate, model, token_ids)
        ratios.append(candidate_speed / reference_speed)

    print(json.dumps({
        "relative_decode_speed": statistics.median(ratios),
    }))


if __name__ == "__main__":
    main()
