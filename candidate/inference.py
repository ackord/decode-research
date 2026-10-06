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
        self.batch_rows_eligible = None
        self.grouped_eligible = None
        self.rotary_eligible = None
        self.attention_eligible = None
        self.stats = {
            "scalar_steps": 0,
            "verifications": 0,
            "bulk_verifications": 0,
            "bulk_updates": 0,
            "batched_blocks": 0,
            "batched_positions": 0,
            "grouped_blocks": 0,
            "grouped_positions": 0,
            "grouped_projection_calls": 0,
            "attention_blocks": 0,
            "attention_calls": 0,
            "causal_chunks": 0,
            "causal_positions": 0,
            "rotary_blocks": 0,
            "grouped_rotary_calls": 0,
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


def _batch_supported(model, cache):
    """Freeze the audited native BF16 activation path, without evaluating it."""
    if not _supported(model, cache):
        return False
    body = model.model
    # These widths bound the audited native normalization/SwiGLU layouts.
    if model.args.hidden_size != 576 or model.args.intermediate_size != 1536:
        return False
    if body.embed_tokens.weight.dtype != mx.bfloat16:
        return False
    norms = [body.norm]
    projections = []
    for layer in body.layers:
        norms.extend((layer.input_layernorm, layer.post_attention_layernorm))
        attn, mlp = layer.self_attn, layer.mlp
        projections.extend((attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj,
                            mlp.gate_proj, mlp.up_proj, mlp.down_proj))
    if not model.args.tie_word_embeddings:
        projections.append(model.lm_head)
    return (all(type(n) is nn.RMSNorm and n.weight.shape == (576,)
                and n.weight.dtype == mx.bfloat16 for n in norms)
            and all(p.weight.dtype == mx.bfloat16 and "bias" not in p
                    for p in projections))


def _grouped_supported(model, cache):
    """Metadata-only gate for the five audited native projection geometries."""
    if not _batch_supported(model, cache) or not model.args.tie_word_embeddings:
        return False
    body = model.model
    if len(body.layers) != 30 or body.embed_tokens.weight.shape != (49152, 576):
        return False
    shapes = ((576, 576), (192, 576), (192, 576), (576, 576),
              (1536, 576), (1536, 576), (576, 1536))
    for layer in body.layers:
        a, f = layer.self_attn, layer.mlp
        if a.n_heads != 9 or a.n_kv_heads != 3 or type(a.rope) is not nn.RoPE:
            return False
        projections = (a.q_proj, a.k_proj, a.v_proj, a.o_proj,
                       f.gate_proj, f.up_proj, f.down_proj)
        if any(p.weight.shape != shape for p, shape in zip(projections, shapes)):
            return False
    return True


def _rotary_supported(model):
    """Separate native rotary gate; failure preserves projection grouping."""
    return all(type(a.rope) is nn.RoPE and a.rope.dims == 64
               and a.rope.traditional is False and a.rope.base == 100000
               and a.rope.scale == 1.0
               for a in (layer.self_attn for layer in model.layers))


def _attention_supported(model, cache):
    """Independent gate; no change to the accepted projection/RoPE gates."""
    return (_grouped_supported(model, cache) and _rotary_supported(model)
            and all(type(kv) is KVCache and not layer.use_sliding
                    and layer.self_attn.head_dim == 64
                    and getattr(layer.self_attn, "sinks", None) is None
                    and getattr(layer.self_attn, "mask", None) is None
                    for layer, kv in zip(model.layers, cache)))


def _attention_schedule(start, count):
    eligible = min(count, max(0, 1023 - start))
    chunks = [(lo, min(lo + 8, eligible)) for lo in range(0, eligible, 8)]
    return chunks + [(i, i + 1) for i in range(eligible, count)]


def _chunk_attention(attn, kv, queries, start, schedule, stats):
    """Read shared, independently bounded K/V prefixes; assemble only once."""
    fragments = []
    for lo, hi in schedule:
        r, end = hi - lo, start + hi
        if r == 1:
            q = queries[lo:hi]
        else:
            q = queries[lo:hi].reshape(1, r, 9, 64).transpose(0, 2, 1, 3)
        # Both updated buffers and the preceding read precede consumed queries.
        deps = [kv.keys, kv.values]
        if fragments:
            deps.append(fragments[-1])
        out = llama.scaled_dot_product_attention(
            mx.depends(q, deps), kv.keys[..., :end, :], kv.values[..., :end, :],
            cache=kv, scale=attn.scale, mask="causal" if r > 1 else None)
        fragments.append(out.transpose(0, 2, 1, 3))
        stats["attention_calls"] += 1
        if r > 1:
            stats["causal_chunks"] += 1
            stats["causal_positions"] += r
    block = fragments[0] if len(fragments) == 1 else mx.concatenate(fragments, axis=1)
    return mx.contiguous(block).reshape(1, queries.shape[0], 576), fragments


def _rotary_offsets(start, count):
    # Avoid a float position conversion and an overflowing exclusive endpoint.
    return mx.arange(count, dtype=mx.int32) + mx.array(start, dtype=mx.int32)


def _grouped_project(weight, block, dependencies, lhs_indices, rhs_indices):
    """Independent M=1 products; keep the weight transpose column-major.

    contiguous establishes row/feature strides, including for unusual storage;
    on native checkpoint weights and row operations it is a storage-preserving
    no-op. Never make the transposed weight contiguous or broadcast its data.
    """
    m, k = block.shape[1:]
    x = mx.depends(mx.contiguous(block), dependencies).reshape(m, 1, k)
    w = mx.contiguous(weight).T[None]
    return mx.gather_mm(x, w, lhs_indices=lhs_indices, rhs_indices=rhs_indices,
                        sorted_indices=False).reshape(1, m, weight.shape[0])


def _rows(block):
    # Native row operations produce contiguous blocks. Enforce that contract
    # without copying suitable storage; split keeps the reduction stride at 1
    # and the final matmul input strides at (K, 1), including every row offset.
    return mx.split(mx.contiguous(block), block.shape[1], axis=1)


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


def _verify(model, cache, tokens, *, _trace=None):
    """Teacher-forced forwards with grouped projections, rotary and short attention.

    tokens contains pending followed by drafts, each with native shape (1, 1).
    Each attention reads its own native causal prefix. Eligible blocks append
    all singleton K/V results before the read-only attention phase; shorter
    blocks retain dependencies protecting reads from subsequent cache writes.
    """
    start = _bulk_start(cache, len(tokens))
    if isinstance(cache, RequestCache) and cache.batch_rows_eligible is None:
        cache.batch_rows_eligible = _batch_supported(model, cache)
    batched = (start is not None and len(tokens) <= MAX_DRAFTS + 1
               and cache.batch_rows_eligible
               and all(t.shape == (1, 1) for t in tokens)
               and all(kv.keys.dtype == mx.bfloat16
                       and kv.values.dtype == mx.bfloat16 for kv in cache))
    if isinstance(cache, RequestCache) and cache.grouped_eligible is None:
        cache.grouped_eligible = _grouped_supported(model, cache)
    grouped = (batched and cache.grouped_eligible
               and all(kv.keys.shape[:2] == (1, 3)
                       and kv.values.shape[:2] == (1, 3)
                       and kv.keys.ndim == kv.values.ndim == 4
                       and kv.keys.shape[3] == kv.values.shape[3] == 64
                       for kv in cache))
    if isinstance(cache, RequestCache) and cache.rotary_eligible is None:
        cache.rotary_eligible = _rotary_supported(model) if grouped else None
    rotary = (grouped and cache.rotary_eligible and 0 <= start
              and start + len(tokens) - 1 <= 2147483647)
    if isinstance(cache, RequestCache) and cache.attention_eligible is None:
        cache.attention_eligible = _attention_supported(model, cache) if rotary else None
    schedule = (_attention_schedule(start, len(tokens))
                if rotary and cache.attention_eligible else [])
    chunked = any(hi - lo > 1 for lo, hi in schedule)
    if chunked:
        cache.stats["attention_blocks"] += 1
    if rotary:
        rotary_offsets = _rotary_offsets(start, len(tokens))
        cache.stats["rotary_blocks"] += 1
    if grouped:
        lhs_indices = mx.arange(len(tokens), dtype=mx.uint32)
        rhs_indices = mx.zeros((len(tokens),), dtype=mx.uint32)
        cache.stats["grouped_blocks"] += 1
        cache.stats["grouped_positions"] += len(tokens)

    def group(weight, block, deps):
        result = _grouped_project(weight, block, deps, lhs_indices, rhs_indices)
        cache.stats["grouped_projection_calls"] += 1
        return result

    def trace(name, arrays):
        if _trace is not None:
            _trace.append((name, arrays if isinstance(arrays, list) else [arrays]))

    if start is not None:
        cache.stats["bulk_verifications"] += 1
    if batched:
        cache.stats["batched_blocks"] += 1
        cache.stats["batched_positions"] += len(tokens)
    body = model.model
    xs = [body.embed_tokens(t) for t in tokens]
    if batched:
        x_block = mx.concatenate(xs, axis=1)
    for layer, kv in zip(body.layers, cache):
        if batched:
            norm_block = layer.input_layernorm(x_block)
            norm = _rows(norm_block) if not grouped else None
        else:
            norm = [layer.input_layernorm(mx.depends(x, xs)) for x in xs]
        attn = layer.self_attn
        if grouped:
            q_block = group(attn.q_proj.weight, norm_block, [norm_block])
            k_block = group(attn.k_proj.weight, norm_block, [q_block])
            v_block = group(attn.v_proj.weight, norm_block, [k_block])
            if not rotary or _trace is not None:
                qs, ks, vs = _rows(q_block), _rows(k_block), _rows(v_block)
        else:
            qs = _project(attn.q_proj, norm, norm)
            ks = _project(attn.k_proj, norm, qs)
            vs = _project(attn.v_proj, norm, ks)
        if _trace is not None:
            trace("q", qs)
            trace("k", ks)
            trace("v", vs)
        attention = []
        attention_block = None
        if start is not None:
            if rotary:
                # Gate both rotary inputs on the final projection phase. Batch
                # elements are singleton sequences with independent offsets.
                q, k, v = mx.depends([q_block, k_block, v_block], [v_block])
                q = attn.rope(q.reshape(len(tokens), 9, 1, 64),
                              offset=rotary_offsets)
                k = attn.rope(k.reshape(len(tokens), 3, 1, 64),
                              offset=rotary_offsets)
                cache.stats["grouped_rotary_calls"] += 2
                q = mx.contiguous(q)
                if not chunked or _trace is not None:
                    rotated_qs = mx.split(q, len(tokens), axis=0)
                if _trace is not None:
                    trace("rope_q", rotated_qs)
                    trace("rope_k", mx.split(k, len(tokens), axis=0))
                kv.update_and_fetch(k.transpose(2, 1, 0, 3),
                    v.reshape(len(tokens), 3, 1, 64).transpose(2, 1, 0, 3))
            else:
                rotated_qs, rotated_ks, reshaped_vs = [], [], []
                for i in range(len(xs)):
                    q, k, v = mx.depends([qs[i], ks[i], vs[i]], vs)
                    q = q.reshape(1, 1, attn.n_heads, -1).transpose(0, 2, 1, 3)
                    k = k.reshape(1, 1, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
                    v = v.reshape(1, 1, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
                    rotated_qs.append(attn.rope(q, offset=start + i))
                    rotated_ks.append(attn.rope(k, offset=start + i))
                    reshaped_vs.append(v)
                if _trace is not None:
                    trace("rope_q", rotated_qs)
                    trace("rope_k", rotated_ks)
                kv.update_and_fetch(mx.concatenate(rotated_ks, axis=2),
                                    mx.concatenate(reshaped_vs, axis=2))
            cache.stats["bulk_updates"] += 1
            if chunked:
                attention_block, fragments = _chunk_attention(
                    attn, kv, q, start, schedule, cache.stats)
                if _trace is not None:
                    attention = _rows(attention_block)
            else:
                for i, q in enumerate(rotated_qs):
                    end = start + i + 1
                    k, v = kv.keys[..., :end, :], kv.values[..., :end, :]
                    deps = [kv.keys, kv.values]
                    if attention:
                        deps.append(attention[-1])
                    out = llama.scaled_dot_product_attention(
                        mx.depends(q, deps), k, v, cache=kv, scale=attn.scale, mask=None
                    )
                    cache.stats["attention_calls"] += 1
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
                if isinstance(cache, RequestCache):
                    cache.stats["attention_calls"] += 1
                attention.append(out.transpose(0, 2, 1, 3).reshape(1, 1, -1))
        if _trace is not None:
            trace("attention", attention)
        if grouped:
            if attention_block is None:
                attention_block = mx.concatenate(attention, axis=1)
                fragments = attention
            projected_block = group(attn.o_proj.weight, attention_block, fragments)
            h_block = x_block + projected_block
            norm_block = layer.post_attention_layernorm(h_block)
            gate_block = group(layer.mlp.gate_proj.weight, norm_block, [projected_block])
            up_block = group(layer.mlp.up_proj.weight, norm_block, [gate_block])
            activated_block = llama.swiglu(mx.depends(gate_block, up_block), up_block)
            down_block = group(layer.mlp.down_proj.weight, activated_block, [activated_block])
            x_block = h_block + down_block
            if _trace is not None:
                trace("projected", _rows(projected_block))
                trace("h", _rows(h_block))
                trace("gate", _rows(gate_block))
                trace("up", _rows(up_block))
                trace("activated", _rows(activated_block))
                trace("down", _rows(down_block))
        else:
            projected = _project(attn.o_proj, attention, attention)
            if batched:
                h_block = x_block + mx.concatenate(projected, axis=1)
                norm = _rows(layer.post_attention_layernorm(h_block))
            else:
                hs = [x + r for x, r in zip(xs, projected)]
                norm = [layer.post_attention_layernorm(h) for h in hs]
            gates = _project(layer.mlp.gate_proj, norm, projected)
            ups = _project(layer.mlp.up_proj, norm, gates)
            if batched:
                gate_block = mx.concatenate(mx.depends(gates, ups), axis=1)
                up_block = mx.concatenate(ups, axis=1)
                activated = _rows(llama.swiglu(gate_block, up_block))
            else:
                activated = [llama.swiglu(g, mx.depends(u, ups)) for g, u in zip(gates, ups)]
            downs = _project(layer.mlp.down_proj, activated, activated)
            if batched:
                x_block = h_block + mx.concatenate(downs, axis=1)
            else:
                xs = [h + r for h, r in zip(hs, downs)]
            if _trace is not None:
                trace("projected", projected)
                trace("h", _rows(h_block) if batched else hs)
                trace("gate", gates)
                trace("up", ups)
                trace("activated", activated)
                trace("down", downs)
        if _trace is not None:
            trace("x", _rows(x_block) if batched else xs)
            trace("kv", [kv.keys[..., :kv.offset, :], kv.values[..., :kv.offset, :]])
    head = body.embed_tokens.as_linear if model.args.tie_word_embeddings else model.lm_head
    if grouped:
        norm_block = body.norm(x_block)
        logits = _rows(group(body.embed_tokens.weight, norm_block, [norm_block]))
    else:
        if batched:
            norm = _rows(body.norm(x_block))
        else:
            norm = [body.norm(mx.depends(x, xs)) for x in xs]
        logits = _project(head, norm, norm)
    if _trace is not None:
        trace("logits", logits)
    return [mx.argmax(x[:, -1, :], axis=-1) for x in logits]


def decode(model, first_token, cache, max_new_tokens):
    if max_new_tokens < 1:
        return []

    state = cache if isinstance(cache, RequestCache) else None
    supported = state is not None and _supported(model, cache)
    if state is not None and state.batch_rows_eligible is None:
        state.batch_rows_eligible = _batch_supported(model, cache)
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


def check_grouped(model):
    """Run the optional native differential suite, outside decode/benchmarking.

    Example: from harness.model import load_model; from candidate.inference
    import check_grouped; check_grouped(load_model()[0])
    """
    from .checks import check_grouped as run_checks
    return run_checks(model)
