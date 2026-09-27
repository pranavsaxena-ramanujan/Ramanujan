"""Generate Ramanujan DSL programs for any model described by a ``ModelSpec``.

Each distinct layer kind gets one program: RMSNorm, a token mixer (grouped-query attention
or Gated DeltaNet), then an FFN, all expressed as small ``*_GPU_1`` kernels that the
translator compiles to OpenCL. Optional features (fused QKV, biases, Q/K norm, attention
output gate, RoPE style and frequency factors, FFN layout, extra norms) are switched on by
the layer's KindSpec flags, so no architecture name appears here.
"""
import ast
import math

from .llm_spec import quant


def _matvec(encoding, columns):
    kind = quant(encoding)
    name = "{0}mv{1}".format(encoding.split("-")[1], columns)
    if kind.intrinsic is None:
        body = """def {name}_GPU_1(w, x, y, row):
    total = 0.0
    column = 0
    base = row * {columns}
    while column < {columns}:
        total = total + w[base + column] * x[column]
        column = column + 1
    y[row] = total
"""
    else:
        body = """def {name}_GPU_1(w, x, y, row):
    total = 0.0
    weight = 0.0
    column = 0
    block_base = row * {blocks}
    block_index = 0
    position = 0
    while column < {columns}:
        block_index = block_base + column / {block}
        position = column % {block}
        weight = {intrinsic}(w, block_index, position)
        total = total + x[column] * weight
        column = column + 1
    y[row] = total
"""
    return name, body.format(name=name, columns=columns, blocks=columns // kind.block,
                             block=kind.block, intrinsic=kind.intrinsic)


def _rmsnorm(dim, eps):
    name = "rmsnorm{0}".format(dim)
    return name, """def {name}_GPU_1(x, w, y, head):
    base = head * {dim}
    squares = 0.0
    i = 0
    while i < {dim}:
        squares = squares + x[base + i] * x[base + i]
        i = i + 1
    inv = 1.0 / sqrt(squares / {dimf} + {eps})
    i = 0
    while i < {dim}:
        y[base + i] = x[base + i] * inv * w[i]
        i = i + 1
""".format(name=name, dim=dim, dimf=float(dim), eps=repr(eps))


def _rmsnorm_inplace(dim, eps):
    name = "rmsnorm_inplace{0}".format(dim)
    return name, """def {name}_GPU_1(x, w, head):
    base = head * {dim}
    squares = 0.0
    i = 0
    while i < {dim}:
        squares = squares + x[base + i] * x[base + i]
        i = i + 1
    inv = 1.0 / sqrt(squares / {dimf} + {eps})
    i = 0
    while i < {dim}:
        x[base + i] = x[base + i] * inv * w[i]
        i = i + 1
""".format(name=name, dim=dim, dimf=float(dim), eps=repr(eps))


def _copy(offset):
    name = "copy_from{0}".format(offset)
    return name, """def {name}_GPU_1(src, dst, i):
    dst[i] = src[i + {offset}]
""".format(name=name, offset=offset)


_ADD = ("add", """def add_GPU_1(x, y, i):
    x[i] = x[i] + y[i]
""")


def _activation(name, value):
    if name == "silu":
        return "{0} / (1.0 + exp(0.0 - {0}))".format(value)
    # GELU, tanh approximation (ggml_gelu); tanh(y) = 1 - 2 / (exp(2y) + 1).
    return ("0.5 * {0} * (2.0 - 2.0 / (exp(1.5957691216057308 * ({0} + 0.044715 * {0} * {0} * {0}))"
            " + 1.0))").format(value)


def _act_mul(activation):
    name = "{0}_mul".format(activation)
    return name, """def {name}_GPU_1(g, u, out, i):
    v = g[i]
    out[i] = {act} * u[i]
""".format(name=name, act=_activation(activation, "v"))


def _act_mul_fused(activation, ffn):
    name = "{0}_mul_fused{1}".format(activation, ffn)
    return name, """def {name}_GPU_1(gu, out, i):
    v = gu[i]
    out[i] = {act} * gu[i + {ffn}]
""".format(name=name, act=_activation(activation, "v"), ffn=ffn)


def _act(activation):
    name = "{0}_inplace".format(activation)
    return name, """def {name}_GPU_1(x, i):
    v = x[i]
    x[i] = {act}
""".format(name=name, act=_activation(activation, "v"))


class _Program:
    def __init__(self, header):
        self.lines = ["# " + header, ""]
        self.kernels = {}
        self.arrays = []
        self.body = []

    def kernel(self, generated):
        name, source = generated
        if self.kernels.setdefault(name, source) != source:
            raise ValueError("conflicting kernel definitions for {0}".format(name))
        return name + "_GPU_1"

    def scratch(self, name, size):
        self._add_array(name)
        self.lines.append("{0} = [0 for _ in range({1})]".format(name, size))

    def inputs(self, *names):
        for name in names:
            self._add_array(name)

    def _add_array(self, name):
        if name in self.arrays:
            raise ValueError("duplicate program array: {0}".format(name))
        self.arrays.append(name)

    def call(self, kernel, *args):
        self.body.append("{0}({1})".format(self.kernel(kernel), ", ".join(str(arg) for arg in args)))

    def render(self, returned):
        lines = self.lines + [""] + list(self.kernels.values())
        lines += ["LOAD_MEM({0})".format(name) for name in self.arrays]
        lines += self.body
        lines += ["GPU_SYNC({0})".format(name) for name in returned]
        lines += ["RELEASE_MEM({0})".format(name) for name in self.arrays]
        lines.append("RETURN({0})".format(", ".join(returned)))
        source = "\n".join(lines) + "\n"
        ast.parse(source)
        return source


def _linear(program, kind, role, x, y, rows, columns):
    program.call(_matvec(kind.encoding(role), columns), role, x, y, rows)
    if kind.has(role + "_bias"):
        program.call(_ADD, y, role + "_bias", rows)


def layer_states(spec, kind, context):
    """(name, float count, csv columns) for the mutable state a layer carries between tokens."""
    if kind.mixer == "attention":
        width = spec.kv_heads * spec.head_dim
        return [("attn_k_cache", context * width, width), ("attn_v_cache", context * width, width)]
    s = spec.ssm
    conv_dim = 2 * s["group_count"] * s["state_size"] + s["inner_size"]
    return [("ssm_s_state", s["time_step_rank"] * s["state_size"] ** 2, s["state_size"]),
            ("ssm_conv_state", conv_dim * (s["conv_kernel"] - 1), s["conv_kernel"] - 1)]


def layer_roles(kind):
    return [role for role, _ in kind.encodings]


def layer_inputs(spec, kind):
    """Global (non-layer) tensors a layer program reads, as {array name: tensor name}."""
    if kind.mixer == "attention" and spec.rope_freqs:
        return {"rope_freqs": spec.rope_freqs}
    return {}


def generate_layer_program(spec, kind, context):
    program = _Program("Generated {0} layer ({1}, {2}). Do not edit.".format(
        spec.architecture, kind.id, kind.mixer))
    states = [name for name, _, _ in layer_states(spec, kind, context)]
    program.inputs("h_state", *states)
    if kind.mixer == "attention":
        program.inputs("pos_arr")
    program.inputs(*layer_roles(kind))
    program.inputs(*layer_inputs(spec, kind))
    program.scratch("t_xn", spec.dim)
    program.scratch("t_mixed", spec.dim)
    program.call(_rmsnorm(spec.dim, spec.eps), "h_state", "attn_norm", "t_xn", 1)
    if kind.mixer == "attention":
        _attention(program, spec, kind, context)
    else:
        _gated_deltanet(program, spec, kind)
    if kind.flag("post_attn_norm"):
        program.call(_rmsnorm_inplace(spec.dim, spec.eps), "t_mixed", "post_attention_norm", 1)
    program.call(_ADD, "h_state", "t_mixed", spec.dim)
    _ffn(program, spec, kind)
    return program.render(["h_state"] + states)


def _attention(program, spec, kind, context):
    if context <= 0 or context * spec.kv_heads * spec.head_dim > 2 ** 24:
        raise ValueError("context must be positive and keep KV indexes exact in float32")
    if spec.sliding_window and context > spec.sliding_window:
        raise ValueError("context {0} exceeds the sliding window {1}".format(context, spec.sliding_window))
    d, hd = spec.dim, spec.head_dim
    q_width, kv_width = spec.heads * hd, spec.kv_heads * hd
    group = spec.heads // spec.kv_heads
    for name, size in (("t_q", q_width), ("t_k", kv_width), ("t_v", kv_width),
                       ("t_scores", spec.heads * context), ("t_attn", q_width)):
        program.scratch(name, size)
    if kind.flag("qkv") == "fused":
        program.scratch("t_qkv", q_width + 2 * kv_width)
        _linear(program, kind, "attn_qkv", "t_xn", "t_qkv", q_width + 2 * kv_width, d)
        program.call(_copy(0), "t_qkv", "t_q", q_width)
        program.call(_copy(q_width), "t_qkv", "t_k", kv_width)
        program.call(_copy(q_width + kv_width), "t_qkv", "t_v", kv_width)
    else:
        q_rows = 2 * q_width if kind.flag("q_gate") else q_width
        q_target = "t_qfull" if kind.flag("q_gate") else "t_q"
        if kind.flag("q_gate"):
            program.scratch("t_qfull", q_rows)
            program.scratch("t_qgate", q_width)
        _linear(program, kind, "attn_q", "t_xn", q_target, q_rows, d)
        _linear(program, kind, "attn_k", "t_xn", "t_k", kv_width, d)
        _linear(program, kind, "attn_v", "t_xn", "t_v", kv_width, d)
    if kind.flag("q_gate"):
        # Q rows are interleaved per head as [query(head_dim), gate(head_dim)].
        norm = kind.flag("qk_norm")
        name = "attn_qsplit{0}".format("_norm" if norm else "")
        source = """def {name}_GPU_1(qfull, {norm_arg}qn, qgate, head):
    src = head * {two_hd}
    dst = head * {hd}
    inv = 1.0
    squares = 0.0
    i = 0
""".format(name=name, norm_arg="q_norm, " if norm else "", two_hd=2 * hd, hd=hd)
        if norm:
            source += """    while i < {hd}:
        squares = squares + qfull[src + i] * qfull[src + i]
        i = i + 1
    inv = 1.0 / sqrt(squares / {hdf} + {eps})
    i = 0
""".format(hd=hd, hdf=float(hd), eps=repr(spec.eps))
        source += """    while i < {hd}:
        qn[dst + i] = qfull[src + i] * inv{weight}
        qgate[dst + i] = qfull[src + {hd} + i]
        i = i + 1
""".format(hd=hd, weight=" * q_norm[i]" if norm else "")
        program.call((name, source), "t_qfull", *(["attn_q_norm"] if norm else []), "t_q", "t_qgate",
                     spec.heads)
    elif kind.flag("qk_norm"):
        program.call(_rmsnorm_inplace(hd, spec.eps), "t_q", "attn_q_norm", spec.heads)
    if kind.flag("qk_norm"):
        program.call(_rmsnorm_inplace(hd, spec.eps), "t_k", "attn_k_norm", spec.kv_heads)
    rope = _rope(spec)
    extra = ["rope_freqs"] if spec.rope_freqs else []
    program.call(rope, "t_q", "pos_arr", *extra, spec.heads * (spec.rope_dims // 2))
    program.call(rope, "t_k", "pos_arr", *extra, spec.kv_heads * (spec.rope_dims // 2))
    program.call(("kv_store", """def kv_store_GPU_1(k, v, k_cache, v_cache, pos_arr, i):
    pos = pos_arr[0]
    k_cache[pos * {w} + i] = k[i]
    v_cache[pos * {w} + i] = v[i]
""".format(w=kv_width)), "t_k", "t_v", "attn_k_cache", "attn_v_cache", "pos_arr", kv_width)
    program.call(("attn_scores", """def attn_scores_GPU_1(q, k_cache, pos_arr, scores, idx):
    head = idx / {ctx}
    t = idx % {ctx}
    kv = head / {group}
    pos = pos_arr[0]
    total = 0.0
    i = 0
    if t <= pos:
        while i < {hd}:
            total = total + q[head * {hd} + i] * k_cache[t * {w} + kv * {hd} + i]
            i = i + 1
    scores[idx] = total * {scale}
""".format(ctx=context, group=group, hd=hd, w=kv_width, scale=repr(spec.attn_scale))),
                 "t_q", "attn_k_cache", "pos_arr", "t_scores", spec.heads * context)
    gated = kind.flag("q_gate")
    mix = """def attn_mix{suffix}_GPU_1(scores, v_cache, {gate_arg}pos_arr, attn, idx):
    head = idx / {hd}
    i = idx % {hd}
    kv = head / {group}
    pos = pos_arr[0]
    base = head * {ctx}
    best = scores[base]
    t = 1
    while t <= pos:
        best = fmax(best, scores[base + t])
        t = t + 1
    total = 0.0
    acc = 0.0
    wt = 0.0
    t = 0
    while t <= pos:
        wt = exp(scores[base + t] - best)
        total = total + wt
        acc = acc + wt * v_cache[t * {w} + kv * {hd} + i]
        t = t + 1
""".format(suffix="_gated" if gated else "", gate_arg="qgate, " if gated else "", hd=hd, group=group,
           ctx=context, w=kv_width)
    if gated:
        mix += """    g = qgate[idx]
    attn[idx] = acc / total / (1.0 + exp(0.0 - g))
"""
    else:
        mix += """    attn[idx] = acc / total
"""
    program.call(("attn_mix_gated" if gated else "attn_mix", mix), "t_scores", "attn_v_cache",
                 *(["t_qgate"] if gated else []), "pos_arr", "t_attn", q_width)
    _linear(program, kind, "attn_output", "t_attn", "t_mixed", d, q_width)


def _rope(spec):
    half = spec.rope_dims // 2
    freqs = bool(spec.rope_freqs)
    name = "rope_{0}{1}".format(spec.semantics.rope_style, "_freqs" if freqs else "")
    if spec.semantics.rope_style == "neox":
        first, second = "base + i", "base + i + {0}".format(half)
    else:
        first, second = "base + 2 * i", "base + 2 * i + 1"
    return name, """def {name}_GPU_1(vec, pos_arr, {freq_arg}p):
    head = p / {half}
    i = p % {half}
    base = head * {hd}
    pos = pos_arr[0] * {scale}
    ang = pos * pow({theta}, (0.0 - 2.0 * i) / {dims}){freq_div}
    c = cos(ang)
    s = sin(ang)
    a = vec[{first}]
    b = vec[{second}]
    vec[{first}] = a * c - b * s
    vec[{second}] = b * c + a * s
""".format(name=name, freq_arg="rope_freqs, " if freqs else "", half=half, hd=spec.head_dim,
           scale=repr(spec.rope_position_scale), theta=repr(spec.rope_theta),
           dims=float(spec.rope_dims), freq_div=" / rope_freqs[i]" if freqs else "",
           first=first, second=second)


def _ffn(program, spec, kind):
    d, ffn = spec.dim, kind.flag("ffn_dim")
    activation = spec.semantics.activation
    program.scratch("t_xn2", d)
    program.scratch("t_ffn_h", ffn)
    program.scratch("t_ffn_o", d)
    program.call(_rmsnorm(d, spec.eps), "h_state", kind.flag("ffn_norm"), "t_xn2", 1)
    layout = kind.flag("ffn")
    if layout == "gated":
        program.scratch("t_ffn_g", ffn)
        program.scratch("t_ffn_u", ffn)
        _linear(program, kind, "ffn_gate", "t_xn2", "t_ffn_g", ffn, d)
        _linear(program, kind, "ffn_up", "t_xn2", "t_ffn_u", ffn, d)
        program.call(_act_mul(activation), "t_ffn_g", "t_ffn_u", "t_ffn_h", ffn)
    elif layout == "fused_gate_up":
        # ffn_up produces [gate(ffn), up(ffn)] (Phi-3 gate_up_proj).
        program.scratch("t_ffn_gu", 2 * ffn)
        _linear(program, kind, "ffn_up", "t_xn2", "t_ffn_gu", 2 * ffn, d)
        program.call(_act_mul_fused(activation, ffn), "t_ffn_gu", "t_ffn_h", ffn)
    else:
        _linear(program, kind, "ffn_up", "t_xn2", "t_ffn_h", ffn, d)
        program.call(_act(activation), "t_ffn_h", ffn)
    _linear(program, kind, "ffn_down", "t_ffn_h", "t_ffn_o", d, ffn)
    if kind.flag("post_ffn_norm"):
        program.call(_rmsnorm_inplace(d, spec.eps), "t_ffn_o", "post_ffw_norm", 1)
    program.call(_ADD, "h_state", "t_ffn_o", d)


def _gated_deltanet(program, spec, kind):
    s = spec.ssm
    size, k_heads, v_heads, inner = s["state_size"], s["group_count"], s["time_step_rank"], s["inner_size"]
    key_dim = k_heads * size
    conv_dim = 2 * key_dim + inner
    k, k1 = s["conv_kernel"], s["conv_kernel"] - 1
    for name, length in (("t_qkv", conv_dim), ("t_z", inner), ("t_beta_raw", v_heads),
                         ("t_alpha", v_heads), ("t_gate", v_heads), ("t_beta", v_heads),
                         ("t_co", conv_dim), ("t_qk", 2 * key_dim), ("t_core", inner),
                         ("t_gated", inner)):
        program.scratch(name, length)
    conv = ("dn_conv", """def dn_conv_GPU_1(qkv, conv_w, conv_state, co, c):
    acc = conv_w[c * {k} + {k1}] * qkv[c]
    j = 0
    while j < {k1}:
        acc = acc + conv_w[c * {k} + j] * conv_state[c * {k1} + j]
        j = j + 1
    j = 0
    while j < {k1} - 1:
        conv_state[c * {k1} + j] = conv_state[c * {k1} + j + 1]
        j = j + 1
    conv_state[c * {k1} + {k1} - 1] = qkv[c]
    co[c] = acc / (1.0 + exp(0.0 - acc))
""".format(k=k, k1=k1))
    l2 = ("dn_l2", """def dn_l2_GPU_1(co, qk, head):
    base = head * {s}
    squares = 0.0
    i = 0
    while i < {s}:
        squares = squares + co[base + i] * co[base + i]
        i = i + 1
    inv = 1.0 / fmax(sqrt(squares), {eps})
    i = 0
    while i < {s}:
        qk[base + i] = co[base + i] * inv
        i = i + 1
""".format(s=size, eps=repr(spec.eps)))
    gate = ("dn_gate", """def dn_gate_GPU_1(alpha, beta_raw, dt_bias, ssm_a, gate, beta, h):
    v = alpha[h] + dt_bias[h]
    sp = v
    if v <= 20.0:
        sp = log1p(exp(v))
    gate[h] = sp * ssm_a[h]
    beta[h] = 1.0 / (1.0 + exp(0.0 - beta_raw[h]))
""")
    delta = ("dn_delta", """def dn_delta_GPU_1(s_state, qk, co, gate, beta, core, idx):
    h = idx / {s}
    j = idx % {s}
    kh = h % {kh}
    q_base = kh * {s}
    k_base = {key_dim} + kh * {s}
    s_base = h * {ss} + j
    decay = exp(gate[h])
    vhat = 0.0
    i = 0
    while i < {s}:
        s_state[s_base + i * {s}] = s_state[s_base + i * {s}] * decay
        vhat = vhat + s_state[s_base + i * {s}] * qk[k_base + i]
        i = i + 1
    d = (co[{v_base} + h * {s} + j] - vhat) * beta[h]
    acc = 0.0
    i = 0
    while i < {s}:
        s_state[s_base + i * {s}] = s_state[s_base + i * {s}] + qk[k_base + i] * d
        acc = acc + s_state[s_base + i * {s}] * qk[q_base + i]
        i = i + 1
    core[h * {s} + j] = acc * {scale}
""".format(s=size, kh=k_heads, key_dim=key_dim, ss=size * size, v_base=2 * key_dim,
           scale=repr(1.0 / math.sqrt(size))))
    gatenorm = ("dn_gatenorm", """def dn_gatenorm_GPU_1(core, ssm_norm, z, gated, h):
    base = h * {s}
    squares = 0.0
    i = 0
    while i < {s}:
        squares = squares + core[base + i] * core[base + i]
        i = i + 1
    inv = 1.0 / sqrt(squares / {sf} + {eps})
    zv = 0.0
    i = 0
    while i < {s}:
        zv = z[base + i]
        gated[base + i] = core[base + i] * inv * ssm_norm[i] * (zv / (1.0 + exp(0.0 - zv)))
        i = i + 1
""".format(s=size, sf=float(size), eps=repr(spec.eps)))
    d = spec.dim
    _linear(program, kind, "attn_qkv", "t_xn", "t_qkv", conv_dim, d)
    _linear(program, kind, "attn_gate", "t_xn", "t_z", inner, d)
    _linear(program, kind, "ssm_beta", "t_xn", "t_beta_raw", v_heads, d)
    _linear(program, kind, "ssm_alpha", "t_xn", "t_alpha", v_heads, d)
    program.call(gate, "t_alpha", "t_beta_raw", "ssm_dt_bias", "ssm_a", "t_gate", "t_beta", v_heads)
    program.call(conv, "t_qkv", "ssm_conv1d", "ssm_conv_state", "t_co", conv_dim)
    program.call(l2, "t_co", "t_qk", 2 * k_heads)
    program.call(delta, "ssm_s_state", "t_qk", "t_co", "t_gate", "t_beta", "t_core", inner)
    program.call(gatenorm, "t_core", "ssm_norm", "t_z", "t_gated", v_heads)
    _linear(program, kind, "ssm_out", "t_gated", "t_mixed", d, inner)


def generate_embed_program(spec, encoding):
    kind = quant(encoding)
    if spec.dim % kind.block:
        raise ValueError("embedding width is not a multiple of the quant block")
    program = _Program("Generated {0} token embedding row decode. Do not edit.".format(spec.architecture))
    program.inputs("emb_row")
    program.scratch("h_state", spec.dim)
    value = "emb_row[i]" if kind.intrinsic is None else "{0}(emb_row, i / {1}, i % {1})".format(
        kind.intrinsic, kind.block)
    scale = spec.embed_multiplier
    if scale != 1.0:
        value = "{0} * {1}".format(value, repr(scale))
    program.call(("embed", """def embed_GPU_1(emb_row, out, i):
    out[i] = {value}
""".format(value=value)), "emb_row", "h_state", spec.dim)
    return program.render(["h_state"])


def generate_head_program(spec, output_encoding):
    program = _Program("Generated {0} final norm, output head and argmax. Do not edit.".format(
        spec.architecture))
    program.inputs("h_state", "output_norm", "output")
    program.scratch("t_xn", spec.dim)
    program.scratch("logits", spec.vocab)
    program.scratch("argmax_arr", 1)
    program.call(_rmsnorm(spec.dim, spec.eps), "h_state", "output_norm", "t_xn", 1)
    program.call(_matvec(output_encoding, spec.dim), "output", "t_xn", "logits", spec.vocab)
    program.call(("argmax", """def argmax_GPU_1(logits, argmax_arr, gid):
    best = logits[0]
    best_i = 0.0
    j = 1
    while j < {vocab}:
        if logits[j] > best:
            best = logits[j]
            best_i = j
        j = j + 1
    argmax_arr[0] = best_i
""".format(vocab=spec.vocab)), "logits", "argmax_arr", 1)
    return program.render(["logits", "argmax_arr"])
