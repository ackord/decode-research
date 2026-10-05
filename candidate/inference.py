import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache


def prefill(model, token_ids):
    tokens = mx.array(token_ids)[None]
    cache = make_prompt_cache(model)

    logits = model(tokens, cache=cache)
    token = mx.argmax(logits[:, -1, :], axis=-1)

    mx.eval(token)

    return token, cache


def decode(model, first_token, cache, max_new_tokens):
    if max_new_tokens < 1:
        return []

    token = first_token
    output = [token.item()]
    pending = []

    for _ in range(max_new_tokens - 1):
        logits = model(token.reshape(1, 1), cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1)

        mx.async_eval(token)
        pending.append(token)
        # Submit the successor before reading the preceding token on the host.
        if len(pending) == 2:
            output.append(pending.pop(0).item())

    for token in pending:
        output.append(token.item())

    return output


def generate(model, token_ids, max_new_tokens):
    token, cache = prefill(model, token_ids)
    return decode(model, token, cache, max_new_tokens)
