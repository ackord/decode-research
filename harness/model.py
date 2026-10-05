from mlx_lm import load

from harness.config import MODEL_ID, MODEL_REVISION


def load_model():
    return load(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
