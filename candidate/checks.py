"""Native checks invoked by inference.check_grouped; no benchmark inputs."""
from contextlib import ExitStack
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import llama
from mlx_lm.models.cache import make_prompt_cache

from . import inference as ci


def _equal(a, b, label):
    assert a.shape == b.shape and a.dtype == b.dtype, (label, a.shape, b.shape)
    dtype = mx.uint16 if a.dtype == mx.bfloat16 else mx.uint32
    different = a.view(dtype) != b.view(dtype)
    if mx.any(different).item():
        index = mx.argmax(different.reshape(-1)).item()
        raise AssertionError(f"First bitwise discrepancy: {label}, flat index {index}")


def _prefixes(a, b, label):
    assert len(a) == len(b), label
    for i, (ka, kb) in enumerate(zip(a, b)):
        assert ka.offset == kb.offset, (label, i, ka.offset, kb.offset)
        for name in ("keys", "values"):
            _equal(getattr(ka, name)[..., :ka.offset, :],
                   getattr(kb, name)[..., :kb.offset, :], f"{label}/layer{i}/{name}")


def _clone(model, cache, capacity=None):
    copies = make_prompt_cache(model)
    assert len(cache) == len(copies)
    for source, target in zip(cache, copies):
        target.offset = source.offset
        for name in ("keys", "values"):
            array = mx.asarray(getattr(source, name), copy=True)
            if capacity is not None:
                array = mx.concatenate([array[..., :source.offset, :],
                    mx.zeros((*array.shape[:2], capacity - source.offset, array.shape[3]),
                             dtype=array.dtype)], axis=2)
            setattr(target, name, array)
    result = ci.RequestCache(copies, cache.history)
    return result


def _projection_checks(model):
    a, f = model.layers[0].self_attn, model.layers[0].mlp
    weights = (a.q_proj.weight, a.k_proj.weight, f.gate_proj.weight,
               f.down_proj.weight, model.model.embed_tokens.weight)
    for w in weights:
        n, k = w.shape
        # Independent synthetic weights make cancellation and BF16 halfway
        # cases explicit; also test the unmodified target weights.
        j = mx.arange(k, dtype=mx.float32)
        row = mx.arange(n, dtype=mx.float32)[:, None]
        cancellation = ((j // 2 % 7 - 3)[None] + (row % 5) * 0.125).astype(mx.bfloat16)
        boundary = (1 + ((row + j[None]) % 5 - 2) / 256
                    + ((row + j[None]) % 3 - 1) / 262144).astype(mx.bfloat16)
        for m in range(3, 18):
            lhs = mx.arange(m, dtype=mx.uint32)
            rhs = mx.zeros((m,), dtype=mx.uint32)
            r = mx.arange(m, dtype=mx.float32)[:, None]
            base = mx.random.normal((1, m, k), key=mx.random.key(101 + m + k))
            cases = ((w, base), (w, base * 0.0001), (w, base * 1000),
                     (cancellation, ((j % 2 * 2 - 1)[None] * (1 + r / 16))[None]),
                     (boundary, (1 + ((j[None] + r) % 5 - 2) / 256
                                  + ((j[None] + r) % 3 - 1) / 262144)[None]))
            for case, (weight, inputs) in enumerate(cases):
                x = mx.contiguous(inputs.astype(mx.bfloat16))
                # Exercise actual gating, then a second phase dependent on
                # the first complete grouped output, using shared indices.
                deps = [mx.sum(x.astype(mx.float32))]
                for phase in range(2):
                    grouped = ci._grouped_project(weight, x, deps, lhs, rhs)
                    single = ci._project(lambda t: t @ weight.T, ci._rows(x), deps)
                    for i, output in enumerate(ci._rows(grouped)):
                        _equal(output, single[i], f"projection/{n}x{k}/m{m}/case{case}/phase{phase}/row{i}")
                    deps = [grouped]


def _rotary_checks(model):
    """Isolate vector dispatch from the verifier, including int32 endpoints."""
    rope = model.layers[0].self_attn.rope
    for m in range(3, 18):
        starts = (0, 1, 255, 4095, 2**24 - m, 2**24 - 1,
                  2**24, 2**24 + 1, 2**30 - 1, 2**31 - m)
        for heads in (3, 9):
            j = mx.arange(m * heads * 64, dtype=mx.float32).reshape(m, heads, 1, 64)
            random = mx.random.normal(j.shape, key=mx.random.key(701 + heads + m))
            signs = j % 2 * 2 - 1
            cases = (random, random * 1e-30, random * 1e30,
                     signs * (1 + (j % 7) / 128),
                     1 + (j % 5 - 2) / 256 + (j % 3 - 1) / 262144,
                     mx.where(j % 2 == 0, mx.array(0.0), mx.array(-0.0)))
            for case, values in enumerate(cases):
                x = mx.contiguous(values.astype(mx.bfloat16))
                for start in starts:
                    offsets = ci._rotary_offsets(start, m)
                    assert offsets.dtype == mx.int32
                    assert offsets.tolist() == list(range(start, start + m))
                    grouped = rope(x, offset=offsets)
                    rows = mx.split(grouped, m, axis=0)
                    inputs = mx.split(x, m, axis=0)
                    assert len(rows) == len(inputs) == m
                    for i in range(m):
                        single = rope(inputs[i], offset=start + i)
                        _equal(rows[i], single,
                               f"rotary/m{m}/heads{heads}/case{case}/C{start}/row{i}")


def _reference_trace(model, cache, tokens):
    """Capture the actual native singleton model calls, without replacing math."""
    captured = {}
    linears, ropes, norms, blocks = {}, {}, {}, {}
    for i, layer in enumerate(model.layers):
        a, f = layer.self_attn, layer.mlp
        for name, module in (("q", a.q_proj), ("k", a.k_proj), ("v", a.v_proj),
                             ("projected", a.o_proj), ("gate", f.gate_proj),
                             ("up", f.up_proj), ("down", f.down_proj)):
            linears[id(module)] = (i, name)
        ropes[id(a.rope)] = i
        norms[id(layer.post_attention_layernorm)] = i
        blocks[id(layer)] = i
    rope_calls = {}

    def record(key, value):
        captured.setdefault(key, []).append(value)

    linear_call, rope_call = nn.Linear.__call__, nn.RoPE.__call__
    norm_call, block_call = nn.RMSNorm.__call__, llama.TransformerBlock.__call__

    def linear(module, x):
        result = linear_call(module, x)
        if id(module) in linears:
            i, name = linears[id(module)]
            record((i, name), result)
            if name == "projected":
                record((i, "attention"), x)
            if name == "down":
                record((i, "activated"), x)
        return result

    def rope(module, x, *args, **kwargs):
        i = ropes[id(module)]
        call = rope_calls.get(i, 0)
        rope_calls[i] = call + 1
        assert x.shape[2] == 1, ("reference RoPE must be singleton", x.shape)
        result = rope_call(module, x, *args, **kwargs)
        record((i, "rope_q" if call % 2 == 0 else "rope_k"), result)
        return result

    def norm(module, x):
        if id(module) in norms:
            record((norms[id(module)], "h"), x)
        return norm_call(module, x)

    def block(module, *args, **kwargs):
        result = block_call(module, *args, **kwargs)
        record((blocks[id(module)], "x"), result)
        return result

    with ExitStack() as stack:
        for cls, fn in ((nn.Linear, linear), (nn.RoPE, rope),
                        (nn.RMSNorm, norm), (llama.TransformerBlock, block)):
            stack.enter_context(patch.object(cls, "__call__", fn))
        logits = []
        for token in tokens:
            logits.append(model(token, cache=cache))
        captured[(len(model.layers), "logits")] = logits
        mx.eval(*logits)
    return captured


def _trace_dict(trace):
    result, layer = {}, -1
    for name, arrays in trace:
        if name == "q":
            layer += 1
        if name == "logits":
            layer += 1
        if name != "kv":
            result[(layer, name)] = arrays
    return result


def _verification_checks(model, base, pending):
    for m in range(3, 18):
        # Unrelated forced tokens exercise rejected tails, not sampled matches.
        tokens = [pending.reshape(1, 1)] + [mx.array([[31 + 13 * i]]) for i in range(m - 1)]
        for capacity in (base[0].keys.shape[2], base[0].offset + m):
            grouped, previous, reference = [_clone(model, base, capacity) for _ in range(3)]
            grouped.append_token(pending.item())
            previous.append_token(pending.item())
            previous.rotary_eligible = False
            gt, pt = [], []
            rope_call = nn.RoPE.__call__
            offsets = {}

            def grouped_rope(module, x, *, offset=0):
                assert x.shape in ((m, 9, 1, 64), (m, 3, 1, 64)), x.shape
                assert offset.dtype == mx.int32 and offset.shape == (m,)
                assert offset.tolist() == list(range(base[0].offset, base[0].offset + m))
                i = offsets.get(id(module), 0)
                assert x.shape[1] == (9 if i == 0 else 3)
                offsets[id(module)] = i + 1
                return rope_call(module, x, offset=offset)

            with patch.object(nn.RoPE, "__call__", grouped_rope):
                gp = ci._verify(model, grouped, tokens, _trace=gt)
            assert len(offsets) == 30 and all(n == 2 for n in offsets.values())
            assert grouped.stats["rotary_blocks"] == 1
            assert grouped.stats["grouped_rotary_calls"] == 60
            pp = ci._verify(model, previous, tokens, _trace=pt)
            mx.eval(*gp, *pp)
            assert grouped.stats["grouped_projection_calls"] == 211
            assert previous.stats["grouped_projection_calls"] == 211
            assert previous.stats["rotary_blocks"] == 0
            assert previous.stats["grouped_rotary_calls"] == 0
            gd, pd = _trace_dict(gt), _trace_dict(pt)
            rd = _reference_trace(model, reference, tokens)
            assert gd.keys() == pd.keys() == rd.keys()
            for key in gd:
                assert len(gd[key]) == len(pd[key]) == len(rd[key]) == m, key
                for row, (g, p, r) in enumerate(zip(gd[key], pd[key], rd[key])):
                    label = f"verifier/m{m}/capacity{capacity}/{key}/row{row}"
                    _equal(g, p, label + "/previous")
                    _equal(g, r, label + "/reference")
            _prefixes(grouped, previous, "previous verifier")
            _prefixes(grouped, reference, "singleton reference")
            untraced = _clone(model, base, capacity)
            untraced.append_token(pending.item())
            untraced_logits = []
            rows = ci._rows

            def capture_logits(block):
                result = rows(block)
                if block.shape[-1] == 49152:
                    untraced_logits.extend(result)
                return result

            with patch.object(ci, "_rows", capture_logits):
                up = ci._verify(model, untraced, tokens)
            mx.eval(*up)
            traced_logits = gd[(len(model.layers), "logits")]
            assert len(up) == len(gp) == len(untraced_logits) == len(traced_logits) == m
            for i in range(m):
                _equal(up[i], gp[i], f"untraced/m{m}/row{i}")
                _equal(untraced_logits[i], traced_logits[i],
                       f"untraced logits/m{m}/row{i}")
            _prefixes(untraced, grouped, f"untraced/m{m}")
            assert untraced.stats["grouped_rotary_calls"] == 60
            # Rollback each possible acceptance length, then overwrite stale
            # tails and grow past capacity through actual scalar forwards.
            for accepted in range(m):
                g = _clone(model, grouped, base[0].offset + m)
                r = _clone(model, reference, base[0].offset + m)
                for kv in [*g, *r]:
                    kv.trim(m - 1 - accepted)
                token = gp[accepted].reshape(1, 1)
                for step in range(m + 1 - accepted):
                    gl, rl = model(token, cache=g), model(token, cache=r)
                    _equal(gl, rl, f"rollback/m{m}/a{accepted}/step{step}")
                    token = mx.argmax(rl[:, -1, :], axis=-1).reshape(1, 1)
                _prefixes(g, r, f"rollback/m{m}/a{accepted}")


def _decode_checks(model, base, pending):
    # Independent greedy oracle on synthetic prompt tokens, never benchmark
    # data. Force exactly one speculative block, then scalar continuation.
    oracle_cache = _clone(model, base)
    token, oracle = pending, [pending.item()]
    for _ in range(23):
        logits = model(token.reshape(1, 1), cache=oracle_cache)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        oracle.append(token.item())
    for accepted in range(17):
        cache = _clone(model, base, base[0].offset + 17)
        drafts = oracle[1:17]
        if accepted < 16:
            drafts[accepted] = (drafts[accepted] + 1) % 49152
        calls = []

        def draft(state, remaining):
            if calls:
                return []
            calls.append(True)
            assert remaining >= 17
            return drafts

        with patch.object(ci, "_draft", draft):
            output = ci.decode(model, pending, cache, 24)
        assert output == oracle
        assert cache.stats["accepted_lengths"][accepted] == 1
        assert cache.stats["grouped_blocks"] == 1
        assert cache.stats["grouped_positions"] == 17
        assert cache.stats["grouped_projection_calls"] == 211
        assert cache.stats["rotary_blocks"] == 1
        assert cache.stats["grouped_rotary_calls"] == 60
        assert all(kv.offset == base[0].offset + 23 for kv in cache)
        _prefixes(cache, oracle_cache, f"decode/accept{accepted}")
    # Two consecutive fully accepted blocks, then the scalar pipeline.
    cache = _clone(model, base, base[0].offset + 7)
    blocks = []

    def consecutive_draft(state, remaining):
        if len(blocks) == 2:
            return []
        index = len(state.history) - len(base.history)
        blocks.append(index)
        return oracle[index:index + 2]

    with patch.object(ci, "_draft", consecutive_draft):
        assert ci.decode(model, pending, cache, 24) == oracle
    assert blocks == [1, 4]
    assert cache.stats["rotary_blocks"] == 2
    assert cache.stats["grouped_rotary_calls"] == 120
    _prefixes(cache, oracle_cache, "consecutive blocks then scalar growth")

    # A rotary-only gate failure must leave accepted projection grouping on.
    g, r = _clone(model, base), _clone(model, base)
    g.append_token(pending.item())
    tokens = [pending.reshape(1, 1)] * 3
    trace = []
    with patch.object(ci, "_rotary_supported", return_value=False):
        predictions = ci._verify(model, g, tokens, _trace=trace)
    ref = [model(t, cache=r) for t in tokens]
    mx.eval(*predictions)
    logits = dict(trace)["logits"]
    assert len(logits) == len(ref) == 3
    for i in range(3):
        _equal(logits[i], ref[i], f"rotary-only fallback/row{i}")
    assert g.stats["grouped_projection_calls"] == 211
    assert g.stats["rotary_blocks"] == g.stats["grouped_rotary_calls"] == 0
    _prefixes(g, r, "rotary-only fallback")

    for limit in (0, 1, 2, 3, 5):
        cache = _clone(model, base)
        assert ci.decode(model, pending, cache, limit) == oracle[:limit]
        if limit <= 1:
            assert cache.stats["grouped_blocks"] == 0
    for limit in (3, 4, 5, 17):
        cache = _clone(model, base)
        drafted = []

        def bounded_draft(state, remaining):
            assert not drafted
            drafted.append(remaining)
            return oracle[1:remaining]

        with patch.object(ci, "_draft", bounded_draft):
            output = ci.decode(model, pending, cache, limit)
        assert output == oracle[:limit]
        assert all(kv.offset == base[0].offset + limit - 1 for kv in cache)
        assert cache.stats["grouped_blocks"] == int(limit >= 4)
    # Capacity failure, too-short blocks, and explicitly unsupported metadata
    # all preserve the original verifier, including eligible batched rows.
    for m, capacity, disable in ((2, base[0].offset + 2, False),
                                 (3, base[0].offset + 2, False),
                                 (3, base[0].offset + 3, True),
                                 (18, base[0].offset + 18, False)):
        g, r = _clone(model, base, capacity), _clone(model, base, capacity)
        g.append_token(pending.item())
        if disable:
            g.grouped_eligible = False
        tokens = [pending.reshape(1, 1)] * m
        trace = []
        predictions = ci._verify(model, g, tokens, _trace=trace)
        ref = [model(t, cache=r) for t in tokens]
        logits = dict(trace)["logits"]
        mx.eval(*predictions)
        assert len(logits) == len(ref) == m
        for i, (a, b) in enumerate(zip(logits, ref)):
            _equal(a, b, f"fallback/m{m}/row{i}")
        assert g.stats["grouped_blocks"] == 0
        if disable:
            assert g.stats["batched_blocks"] == 1
        _prefixes(g, r, "fallback")


def check_grouped(model):
    """Fail at the first numerical discrepancy; never certify throughput."""
    prompt = [1] + [41 + (i * 17) % 311 for i in range(23)]
    pending, base = ci.prefill(model, prompt)
    assert ci._grouped_supported(model, base), "Loaded model is outside the audited gate"
    assert ci._rotary_supported(model), "Loaded rotary parameters are outside the gate"
    # Each native parameter is independently required by the rotary gate.
    rope = model.layers[0].self_attn.rope
    for name, unsupported in (("dims", 32), ("traditional", True),
                              ("base", 10000), ("scale", 0.5)):
        with patch.object(rope, name, unsupported):
            assert not ci._rotary_supported(model), name
            assert ci._grouped_supported(model, base), name
    _rotary_checks(model)
    _projection_checks(model)
    _verification_checks(model, base, pending)
    _decode_checks(model, base, pending)
    # A fresh prefill must recreate all request state, including counters.
    second_pending, second = ci.prefill(model, prompt)
    _equal(pending, second_pending, "independent request")
    assert second.stats["grouped_blocks"] == second.stats["grouped_projection_calls"] == 0
    assert second.grouped_eligible is None and second.rotary_eligible is None
    assert second.stats["rotary_blocks"] == second.stats["grouped_rotary_calls"] == 0
    assert second.index is not base.index and second.history is not base.history
    _prefixes(base, second, "independent request")
    print("Native grouped rotary/projections, intermediates, logits, KV and transitions passed.")
