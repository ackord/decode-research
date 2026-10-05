MODEL_ID = "mlx-community/SmolLM2-135M-Instruct"
MODEL_REVISION = "422de227b90002f443a21a58b1087f6ee7632731"

EVAL_NEW_TOKENS = 64
EVAL_PROMPTS = [
    "Explain why the sky is blue.",
    "Write a short Python function that reverses a list.",
    "What are the main causes of inflation?",
    "Describe how photosynthesis works.",
    "Continue this story: The door opened and nobody was there.",
]

BENCH_CONTEXT_TOKENS = 512
BENCH_NEW_TOKENS = 128
WARMUPS = 3
RUNS = 20

BENCH_TEXT = (
    "Large language models process sequences of tokens using repeated "
    "transformer layers. During inference the prompt is processed first, "
    "followed by autoregressive generation of new tokens. "
) * 100
