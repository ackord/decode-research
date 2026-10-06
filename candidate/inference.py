import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.models import llama


MAX_CONTEXT = 16
MAX_DRAFTS = 16
INDEX_CONTEXT = MAX_CONTEXT + MAX_DRAFTS - 1


class _Successors:
    """Exact occurrence counts with the most recent winner on frequency ties."""

    __slots__ = ("counts", "preferred")

    def __init__(self, token):
        self.counts = {token: 1}
        self.preferred = token

    def observe(self, token):
        count = self.counts.get(token, 0) + 1
        self.counts[token] = count
        if count >= self.counts[self.preferred]:
            self.preferred = token


class RequestCache(list):
    """Ordinary caches plus request-local successor counts and decode counters."""

    def __init__(self, caches, token_ids):
        super().__init__(caches)
        self.history = []
        self.index = {}
        self.stats = {
            "scalar_steps": 0,
            "verifications": 0,
            "bulk_verifications": 0,
            "bulk_updates": 0,
            "draft_lengths": [0] * (MAX_DRAFTS + 1),
            "accepted_lengths": [0] * (MAX_DRAFTS + 1),
            "rejected_positions": 0,
        }
        for token in token_ids:
            self.append_token(int(token))

    def append_token(self, token):
        j = len(self.history)
        for n in range(1, min(INDEX_CONTEXT, j) + 1):
            context = tuple(self.history[j - n : j])
            successors = self.index.get(context)
            if successors is None:
                self.index[context] = _Successors(token)
            else:
                successors.observe(token)
        self.history.append(token)

    def match(self):
        for n in range(min(MAX_CONTEXT, len(self.history)), 0, -1):
            context = tuple(self.history[-n:])
            if context in self.index:
                return context
        return None


def prefill(model, token_ids):
    tokens = mx.array(token_ids)[None]
    cache = make_prompt_cache(model)
    logits = model(tokens, cache=cache)
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)
    return token, RequestCache(cache, token_ids)


def _supported(model, cache):
    # Mirror these native classes and rounding boundaries. Other architectures,
    # quantization and sharding retain the scalar fallback.
    if type(model) is not llama.Model or type(model.model) is not llama.LlamaModel:
        return False
    if len(cache) != len(model.layers) or not cache:
        return False
    for layer, kv in zip(model.layers, cache):
        if (type(layer) is not llama.TransformerBlock or layer.use_sliding
                or type(kv) is not KVCache
                or type(layer.self_attn) is not llama.Attention
                or type(layer.mlp) is not llama.MLP):
            return False
        projections = (layer.self_attn.q_proj, layer.self_attn.k_proj,
                       layer.self_attn.v_proj, layer.self_attn.o_proj,
                       layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj)
        if any(type(p) is not nn.Linear for p in projections):
            return False
    return (type(model.model.embed_tokens) is nn.Embedding
            and (model.args.tie_word_embeddings or type(model.lm_head) is nn.Linear))


def _draft(cache, remaining):
    context = cache.match()
    if context is None or remaining < 2:
        return []
    c = cache[0].offset
    if any(kv.offset != c or kv.keys is None or kv.values is None for kv in cache):
        return []
    if c != len(cache.history) - 1:
        raise RuntimeError("Speculation requires drained scalar outputs")
    capacity = min(min(kv.keys.shape[2], kv.values.shape[2]) for kv in cache)
    limit = min(MAX_DRAFTS, len(context), remaining - 1, capacity - c - 1)
    drafts = []
    # Extend the entire observed context, without mutating the index or backing
    # off to a shorter suffix. Every extension occurred in committed history.
    for _ in range(max(0, limit)):
        successors = cache.index.get(context)
        if successors is None:
            break
        token = successors.preferred
        drafts.append(token)
        context += (token,)
    return drafts


def _project(projection, inputs, dependencies):
    """Gate consumed inputs, so a whole projection phase precedes the next."""
    outputs = []
    for x in inputs:
        deps = dependencies if not outputs else [*dependencies, outputs[-1]]
        outputs.append(projection(mx.depends(x, deps)))
    return outputs


def _bulk_start(cache, count):
    """Choose bulk insertion only with drained history and existing capacity."""
    if count < 3 or not isinstance(cache, RequestCache) or not cache:
        return None
    start = cache[0].offset
    if start != len(cache.history) - 1:
        return None
    for kv in cache:
        if (kv.offset != start or kv.keys is None or kv.values is None
                or min(kv.keys.shape[2], kv.values.shape[2]) < start + count):
            return None
    return start


def _verify(model, cache, tokens):
    """Teacher-forced singleton forwards, interchanged only across projections.

    tokens contains pending followed by drafts, each with native shape (1, 1).
    Each attention reads its own native causal prefix. Eligible blocks append
    all singleton K/V results before the read-only attention phase; shorter
    blocks retain dependencies protecting reads from subsequent cache writes.
    """
    start = _bulk_start(cache, len(tokens))
    if start is not None:
        cache.stats["bulk_verifications"] += 1
    body = model.model
    xs = [body.embed_tokens(t) for t in tokens]
    for layer, kv in zip(body.layers, cache):
        norm = [layer.input_layernorm(mx.depends(x, xs)) for x in xs]
        attn = layer.self_attn
        qs = _project(attn.q_proj, norm, norm)
        ks = _project(attn.k_proj, norm, qs)
        vs = _project(attn.v_proj, norm, ks)
        attention = []
        if start is not None:
            rotated_qs, rotated_ks, reshaped_vs = [], [], []
            for i in range(len(xs)):
                q, k, v = mx.depends([qs[i], ks[i], vs[i]], vs)
                q = q.reshape(1, 1, attn.n_heads, -1).transpose(0, 2, 1, 3)
                k = k.reshape(1, 1, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
                v = v.reshape(1, 1, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
                rotated_qs.append(attn.rope(q, offset=start + i))
                rotated_ks.append(attn.rope(k, offset=start + i))
                reshaped_vs.append(v)
            kv.update_and_fetch(mx.concatenate(rotated_ks, axis=2),
                                mx.concatenate(reshaped_vs, axis=2))
            cache.stats["bulk_updates"] += 1
            for i, q in enumerate(rotated_qs):
                end = start + i + 1
                if end < kv.keys.shape[2]:
                    k, v = kv.keys[..., :end, :], kv.values[..., :end, :]
                else:
                    k, v = kv.keys, kv.values
                deps = [kv.keys, kv.values]
                if attention:
                    deps.append(attention[-1])
                out = llama.scaled_dot_product_attention(
                    mx.depends(q, deps), k, v, cache=kv, scale=attn.scale, mask=None
                )
                attention.append(out.transpose(0, 2, 1, 3).reshape(1, 1, -1))
        else:
            for i in range(len(xs)):
                deps = vs if not attention else [*vs, attention[-1]]
                q, k, v = mx.depends([qs[i], ks[i], vs[i]], deps)
                q = q.reshape(1, 1, attn.n_heads, -1).transpose(0, 2, 1, 3)
                k = k.reshape(1, 1, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
                v = v.reshape(1, 1, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
                q = attn.rope(q, offset=kv.offset)
                k = attn.rope(k, offset=kv.offset)
                if attention:
                    kv.keys, kv.values = mx.depends(
                        [kv.keys, kv.values], attention[-1]
                    )
                k, v = kv.update_and_fetch(k, v)
                out = llama.scaled_dot_product_attention(
                    q, k, v, cache=kv, scale=attn.scale, mask=None
                )
                attention.append(out.transpose(0, 2, 1, 3).reshape(1, 1, -1))
        projected = _project(attn.o_proj, attention, attention)
        hs = [x + r for x, r in zip(xs, projected)]
        norm = [layer.post_attention_layernorm(h) for h in hs]
        gates = _project(layer.mlp.gate_proj, norm, projected)
        ups = _project(layer.mlp.up_proj, norm, gates)
        activated = [llama.swiglu(g, mx.depends(u, ups)) for g, u in zip(gates, ups)]
        downs = _project(layer.mlp.down_proj, activated, activated)
        xs = [h + r for h, r in zip(hs, downs)]
    norm = [body.norm(mx.depends(x, xs)) for x in xs]
    head = body.embed_tokens.as_linear if model.args.tie_word_embeddings else model.lm_head
    logits = _project(head, norm, norm)
    return [mx.argmax(x[:, -1, :], axis=-1) for x in logits]


def decode(model, first_token, cache, max_new_tokens):
    if max_new_tokens < 1:
        return []

    state = cache if isinstance(cache, RequestCache) else None
    supported = state is not None and _supported(model, cache)
    token = first_token
    output = []
    pending = []

    def commit(value):
        output.append(value)
        if state is not None:
            state.append_token(value)

    commit(token.item())
    while len(output) + len(pending) < max_new_tokens:
        # While outputs are unread, token and KV are ahead of host history.
        # Recompute speculative lookup and capacity decisions after draining.
        if supported and not pending:
            drafts = _draft(state, max_new_tokens - len(output))
            if drafts:
                inputs = [token.reshape(1, 1)] + [
                    mx.array([[d]], dtype=token.dtype) for d in drafts
                ]
                predictions = _verify(model, cache, inputs)
                mx.async_eval(*predictions)
                ids = mx.stack(predictions).tolist()
                accepted = 0
                while accepted < len(drafts) and drafts[accepted] == ids[accepted][0]:
                    accepted += 1
                for value in drafts[:accepted]:
                    commit(value)
                commit(ids[accepted][0])
                token = predictions[accepted]
                rejected = len(drafts) - accepted
                for kv in cache:
                    kv.trim(rejected)
                del inputs, predictions, ids
                stats = state.stats
                stats["verifications"] += 1
                stats["draft_lengths"][len(drafts)] += 1
                stats["accepted_lengths"][accepted] += 1
                stats["rejected_positions"] += rejected
                continue

        logits = model(token.reshape(1, 1), cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.async_eval(token)
        pending.append(token)
        if state is not None:
            state.stats["scalar_steps"] += 1
        # Preserve round 0004's submit-successor-before-host-read pipeline.
        if len(pending) == 2:
            commit(pending.pop(0).item())
            if supported and (state.history[-1],) in state.index:
                for unread in pending:
                    commit(unread.item())
                pending.clear()

    for unread in pending:
        commit(unread.item())
    return output


def generate(model, token_ids, max_new_tokens):
    token, cache = prefill(model, token_ids)
    return decode(model, token, cache, max_new_tokens)
