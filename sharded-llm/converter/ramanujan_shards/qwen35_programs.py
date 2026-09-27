import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path


_QUANT = {
    "gguf-f32": (1, 4, None),
    "gguf-q4_1": (32, 20, "GGUF_Q4_1_VALUE"),
    "gguf-q5_k": (256, 176, "GGUF_Q5_K_VALUE"),
    "gguf-q6_k": (256, 210, "GGUF_Q6_K_VALUE"),
}

DELTA_TENSORS = {
    "attn_norm": "attn_norm.weight", "w_qkv": "attn_qkv.weight", "w_z": "attn_gate.weight",
    "w_beta": "ssm_beta.weight", "w_alpha": "ssm_alpha.weight", "dt_bias": "ssm_dt.bias",
    "ssm_a": "ssm_a", "conv_w": "ssm_conv1d.weight", "ssm_norm": "ssm_norm.weight",
    "w_out": "ssm_out.weight", "post_norm": "post_attention_norm.weight",
    "w_ffn_gate": "ffn_gate.weight", "w_ffn_up": "ffn_up.weight", "w_ffn_down": "ffn_down.weight",
}
ATTN_TENSORS = {
    "attn_norm": "attn_norm.weight", "w_q": "attn_q.weight", "w_k": "attn_k.weight",
    "w_v": "attn_v.weight", "q_norm": "attn_q_norm.weight", "k_norm": "attn_k_norm.weight",
    "w_o": "attn_output.weight", "post_norm": "post_attention_norm.weight",
    "w_ffn_gate": "ffn_gate.weight", "w_ffn_up": "ffn_up.weight", "w_ffn_down": "ffn_down.weight",
}
HEAD_TENSORS = {"output_norm": "output_norm.weight", "w_output": "output.weight"}


@dataclass(frozen=True)
class Qwen35Config:
    dim: int
    eps: float
    conv_kernel: int
    state_size: int
    k_heads: int
    v_heads: int
    inner: int
    heads: int
    kv_heads: int
    head_dim: int
    rope_dims: int
    rope_theta: float
    ffn: int
    vocab: int
    layers: int
    interval: int

    @property
    def key_dim(self):
        return self.k_heads * self.state_size

    @property
    def conv_dim(self):
        return 2 * self.key_dim + self.inner

    def is_attention(self, layer):
        return (layer + 1) % self.interval == 0

    @classmethod
    def from_metadata(cls, metadata):
        if metadata.get("general.architecture") != "qwen35":
            raise ValueError("expected qwen35 GGUF metadata")
        m = lambda key: metadata["qwen35." + key]
        config = cls(
            dim=int(m("embedding_length")), eps=float(m("attention.layer_norm_rms_epsilon")),
            conv_kernel=int(m("ssm.conv_kernel")), state_size=int(m("ssm.state_size")),
            k_heads=int(m("ssm.group_count")), v_heads=int(m("ssm.time_step_rank")),
            inner=int(m("ssm.inner_size")), heads=int(m("attention.head_count")),
            kv_heads=int(m("attention.head_count_kv")), head_dim=int(m("attention.key_length")),
            rope_dims=int(m("rope.dimension_count")), rope_theta=float(m("rope.freq_base")),
            ffn=int(m("feed_forward_length")), vocab=len(metadata["tokenizer.ggml.tokens"]),
            layers=int(m("block_count")) - int(metadata.get("qwen35.nextn_predict_layers", 0)),
            interval=int(m("full_attention_interval")),
        )
        if (config.inner != config.v_heads * config.state_size or config.v_heads % config.k_heads
                or config.heads % config.kv_heads or config.rope_dims % 2
                or config.rope_dims > config.head_dim or config.conv_kernel < 2
                or int(metadata["qwen35.attention.value_length"]) != config.head_dim):
            raise ValueError("unsupported qwen35 hyperparameters")
        return config


def load_package(package_dir):
    package_dir = Path(package_dir)
    package = json.loads((package_dir / "model-manifest.json").read_text(encoding="utf-8"))
    if package.get("sourceFormat") != "gguf" or package.get("architectureId") != "qwen35":
        raise ValueError("expected a qwen35 GGUF shard package")
    if package.get("partial"):
        raise ValueError("inference requires every shard")
    shards = []
    for entry in package["shards"]:
        manifest = json.loads((package_dir / entry["manifestPath"]).read_text(encoding="utf-8"))
        root = (package_dir / entry["manifestPath"]).parent.resolve()
        tensors = {}
        for name, tensor in manifest["tensorFiles"].items():
            path = (root / tensor["path"]).resolve()
            if root not in path.parents:
                raise ValueError("tensor path escapes shard: {0}".format(name))
            tensors[name] = dict(tensor, file=path)
        shards.append({"id": entry["shardId"], "root": root, "tensors": tensors,
                       "layerStart": manifest.get("layerStart"), "layerEnd": manifest.get("layerEnd")})
    return shards


def tensor_columns(tensor):
    block, block_bytes, _ = _encoding(tensor["encoding"])
    shape = tensor["shape"]
    if len(shape) == 1:
        return shape[0]
    row_bytes = shape[1] // block * block_bytes
    if shape[1] % block or row_bytes % 4:
        raise ValueError("tensor rows are not float-word aligned")
    return row_bytes // 4


def _encoding(encoding):
    if encoding not in _QUANT:
        raise ValueError("unsupported tensor encoding: {0}".format(encoding))
    return _QUANT[encoding]


def _check_layer(config, tensors, layer, roles):
    c = config
    expected = {
        "attn_norm": [c.dim], "post_norm": [c.dim],
        "w_ffn_gate": [c.ffn, c.dim], "w_ffn_up": [c.ffn, c.dim], "w_ffn_down": [c.dim, c.ffn],
        "w_qkv": [c.conv_dim, c.dim], "w_z": [c.inner, c.dim], "w_beta": [c.v_heads, c.dim],
        "w_alpha": [c.v_heads, c.dim], "dt_bias": [c.v_heads], "ssm_a": [c.v_heads],
        "conv_w": [c.conv_dim, c.conv_kernel], "ssm_norm": [c.state_size], "w_out": [c.dim, c.inner],
        "w_q": [2 * c.heads * c.head_dim, c.dim], "w_k": [c.kv_heads * c.head_dim, c.dim],
        "w_v": [c.kv_heads * c.head_dim, c.dim], "q_norm": [c.head_dim], "k_norm": [c.head_dim],
        "w_o": [c.dim, c.heads * c.head_dim],
    }
    encodings = {}
    for role, suffix in roles.items():
        name = "blk.{0}.{1}".format(layer, suffix)
        tensor = tensors.get(name)
        if tensor is None or tensor["shape"] != expected[role]:
            raise ValueError("missing or misshaped tensor: {0}".format(name))
        if len(expected[role]) == 1 and tensor["encoding"] != "gguf-f32":
            raise ValueError("vector tensor must be F32: {0}".format(name))
        _encoding(tensor["encoding"])
        encodings[role] = tensor["encoding"]
    return encodings


def layer_encodings(config, shards):
    tensors = {name: tensor for shard in shards for name, tensor in shard["tensors"].items()}
    kinds = {}
    for layer in range(config.layers):
        kind = "attn" if config.is_attention(layer) else "delta"
        encodings = _check_layer(config, tensors, layer, ATTN_TENSORS if kind == "attn" else DELTA_TENSORS)
        if kinds.setdefault(kind, encodings) != encodings:
            raise ValueError("layer {0} encodings differ from other {1} layers".format(layer, kind))
    for role, name in HEAD_TENSORS.items():
        if name not in tensors:
            raise ValueError("missing tensor: {0}".format(name))
    if (tensors["output.weight"]["shape"] != [config.vocab, config.dim]
            or tensors["token_embd.weight"]["shape"] != [config.vocab, config.dim]
            or tensors["output_norm.weight"]["shape"] != [config.dim]):
        raise ValueError("unexpected embedding or output head shape")
    kinds["head"] = {role: tensors[name]["encoding"] for role, name in HEAD_TENSORS.items()}
    kinds["embed"] = {"token_embd": tensors["token_embd.weight"]["encoding"]}
    return kinds


def _matvec(encoding, columns):
    block, _, intrinsic = _encoding(encoding)
    name = "{0}mv{1}".format(encoding.split("-")[1], columns)
    if intrinsic is None:
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
    return name, body.format(name=name, columns=columns, blocks=columns // block,
                             block=block, intrinsic=intrinsic)


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


_ADD = """def add_GPU_1(x, y, i):
    x[i] = x[i] + y[i]
"""

_SILU_MUL = """def silu_mul_GPU_1(g, u, out, i):
    v = g[i]
    out[i] = v / (1.0 + exp(0.0 - v)) * u[i]
"""


class _Program:
    def __init__(self, header):
        self.lines = ["# " + header, ""]
        self.kernels = {}
        self.arrays = []
        self.body = []

    def kernel(self, generated):
        name, source = generated
        self.kernels.setdefault(name, source)
        return name + "_GPU_1"

    def scratch(self, name, size):
        self.lines.append("{0} = [0 for _ in range({1})]".format(name, size))
        self.arrays.append(name)

    def inputs(self, *names):
        self.arrays.extend(names)

    def call(self, kernel, *args):
        self.body.append("{0}({1})".format(kernel, ", ".join(str(arg) for arg in args)))

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


def _ffn(program, config, encodings):
    c = config
    program.scratch("xn2", c.dim)
    program.scratch("ffn_g", c.ffn)
    program.scratch("ffn_u", c.ffn)
    program.scratch("ffn_h", c.ffn)
    program.scratch("ffn_o", c.dim)
    program.call(program.kernel(_rmsnorm(c.dim, c.eps)), "h_state", "post_norm", "xn2", 1)
    program.call(program.kernel(_matvec(encodings["w_ffn_gate"], c.dim)), "w_ffn_gate", "xn2", "ffn_g", c.ffn)
    program.call(program.kernel(_matvec(encodings["w_ffn_up"], c.dim)), "w_ffn_up", "xn2", "ffn_u", c.ffn)
    program.call(program.kernel(("silu_mul", _SILU_MUL)), "ffn_g", "ffn_u", "ffn_h", c.ffn)
    program.call(program.kernel(_matvec(encodings["w_ffn_down"], c.ffn)), "w_ffn_down", "ffn_h", "ffn_o", c.dim)
    program.call(program.kernel(("add", _ADD)), "h_state", "ffn_o", c.dim)


def generate_delta_program(config, encodings):
    c = config
    k1 = c.conv_kernel - 1
    program = _Program("Generated Qwen35 Gated DeltaNet layer. Do not edit.")
    program.inputs("h_state", "ssm_s_state", "ssm_conv_state", *DELTA_TENSORS)
    for name, size in (("xn", c.dim), ("qkv", c.conv_dim), ("zbuf", c.inner), ("beta_raw", c.v_heads),
                       ("alpha", c.v_heads), ("gate", c.v_heads), ("beta", c.v_heads),
                       ("co", c.conv_dim), ("qk", 2 * c.key_dim), ("core", c.inner),
                       ("gated", c.inner), ("attn_out", c.dim)):
        program.scratch(name, size)
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
""".format(k=c.conv_kernel, k1=k1))
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
""".format(s=c.state_size, eps=repr(c.eps)))
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
""".format(s=c.state_size, kh=c.k_heads, key_dim=c.key_dim, ss=c.state_size * c.state_size,
           v_base=2 * c.key_dim, scale=repr(1.0 / math.sqrt(c.state_size))))
    gatenorm = ("dn_gatenorm", """def dn_gatenorm_GPU_1(core, ssm_norm, zbuf, gated, h):
    base = h * {s}
    squares = 0.0
    i = 0
    while i < {s}:
        squares = squares + core[base + i] * core[base + i]
        i = i + 1
    inv = 1.0 / sqrt(squares / {sf} + {eps})
    z = 0.0
    i = 0
    while i < {s}:
        z = zbuf[base + i]
        gated[base + i] = core[base + i] * inv * ssm_norm[i] * (z / (1.0 + exp(0.0 - z)))
        i = i + 1
""".format(s=c.state_size, sf=float(c.state_size), eps=repr(c.eps)))
    program.call(program.kernel(_rmsnorm(c.dim, c.eps)), "h_state", "attn_norm", "xn", 1)
    program.call(program.kernel(_matvec(encodings["w_qkv"], c.dim)), "w_qkv", "xn", "qkv", c.conv_dim)
    program.call(program.kernel(_matvec(encodings["w_z"], c.dim)), "w_z", "xn", "zbuf", c.inner)
    program.call(program.kernel(_matvec(encodings["w_beta"], c.dim)), "w_beta", "xn", "beta_raw", c.v_heads)
    program.call(program.kernel(_matvec(encodings["w_alpha"], c.dim)), "w_alpha", "xn", "alpha", c.v_heads)
    program.call(program.kernel(gate), "alpha", "beta_raw", "dt_bias", "ssm_a", "gate", "beta", c.v_heads)
    program.call(program.kernel(conv), "qkv", "conv_w", "ssm_conv_state", "co", c.conv_dim)
    program.call(program.kernel(l2), "co", "qk", 2 * c.k_heads)
    program.call(program.kernel(delta), "ssm_s_state", "qk", "co", "gate", "beta", "core", c.inner)
    program.call(program.kernel(gatenorm), "core", "ssm_norm", "zbuf", "gated", c.v_heads)
    program.call(program.kernel(_matvec(encodings["w_out"], c.inner)), "w_out", "gated", "attn_out", c.dim)
    program.call(program.kernel(("add", _ADD)), "h_state", "attn_out", c.dim)
    _ffn(program, c, encodings)
    return program.render(["h_state", "ssm_s_state", "ssm_conv_state"])


def generate_attention_program(config, encodings, context):
    c = config
    if context <= 0 or context * c.kv_heads * c.head_dim > 2**24:
        raise ValueError("context must be positive and keep KV indexes exact in float32")
    group = c.heads // c.kv_heads
    half = c.rope_dims // 2
    kv_width = c.kv_heads * c.head_dim
    program = _Program("Generated Qwen35 gated attention layer. Do not edit.")
    program.inputs("h_state", "attn_k_cache", "attn_v_cache", "pos_arr", *ATTN_TENSORS)
    for name, size in (("xn", c.dim), ("qfull", 2 * c.heads * c.head_dim), ("kraw", kv_width),
                       ("vbuf", kv_width), ("qn", c.heads * c.head_dim), ("qgate", c.heads * c.head_dim),
                       ("kn", kv_width), ("scores", c.heads * context), ("attn", c.heads * c.head_dim),
                       ("attn_out", c.dim)):
        program.scratch(name, size)
    qnorm = ("attn_qnorm", """def attn_qnorm_GPU_1(qfull, q_norm, qn, qgate, head):
    src = head * {two_hd}
    dst = head * {hd}
    squares = 0.0
    i = 0
    while i < {hd}:
        squares = squares + qfull[src + i] * qfull[src + i]
        i = i + 1
    inv = 1.0 / sqrt(squares / {hdf} + {eps})
    i = 0
    while i < {hd}:
        qn[dst + i] = qfull[src + i] * inv * q_norm[i]
        qgate[dst + i] = qfull[src + {hd} + i]
        i = i + 1
""".format(two_hd=2 * c.head_dim, hd=c.head_dim, hdf=float(c.head_dim), eps=repr(c.eps)))
    rope = ("rope", """def rope_GPU_1(vec, pos_arr, p):
    head = p / {half}
    i = p % {half}
    base = head * {hd}
    pos = pos_arr[0]
    ang = pos * pow({theta}, (0.0 - 2.0 * i) / {dims})
    c = cos(ang)
    s = sin(ang)
    a = vec[base + i]
    b = vec[base + i + {half}]
    vec[base + i] = a * c - b * s
    vec[base + i + {half}] = b * c + a * s
""".format(half=half, hd=c.head_dim, theta=repr(c.rope_theta), dims=float(c.rope_dims)))
    store = ("kv_store", """def kv_store_GPU_1(kn, vbuf, k_cache, v_cache, pos_arr, i):
    pos = pos_arr[0]
    k_cache[pos * {w} + i] = kn[i]
    v_cache[pos * {w} + i] = vbuf[i]
""".format(w=kv_width))
    scores = ("attn_scores", """def attn_scores_GPU_1(qn, k_cache, pos_arr, scores, idx):
    head = idx / {ctx}
    t = idx % {ctx}
    kv = head / {group}
    pos = pos_arr[0]
    total = 0.0
    i = 0
    if t <= pos:
        while i < {hd}:
            total = total + qn[head * {hd} + i] * k_cache[t * {w} + kv * {hd} + i]
            i = i + 1
    scores[idx] = total * {scale}
""".format(ctx=context, group=group, hd=c.head_dim, w=kv_width, scale=repr(1.0 / math.sqrt(c.head_dim))))
    mix = ("attn_mix", """def attn_mix_GPU_1(scores, v_cache, qgate, pos_arr, attn, idx):
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
    g = qgate[idx]
    attn[idx] = acc / total / (1.0 + exp(0.0 - g))
""".format(hd=c.head_dim, group=group, ctx=context, w=kv_width))
    program.call(program.kernel(_rmsnorm(c.dim, c.eps)), "h_state", "attn_norm", "xn", 1)
    program.call(program.kernel(_matvec(encodings["w_q"], c.dim)), "w_q", "xn", "qfull", 2 * c.heads * c.head_dim)
    program.call(program.kernel(_matvec(encodings["w_k"], c.dim)), "w_k", "xn", "kraw", kv_width)
    program.call(program.kernel(_matvec(encodings["w_v"], c.dim)), "w_v", "xn", "vbuf", kv_width)
    program.call(program.kernel(qnorm), "qfull", "q_norm", "qn", "qgate", c.heads)
    program.call(program.kernel(_rmsnorm(c.head_dim, c.eps)), "kraw", "k_norm", "kn", c.kv_heads)
    program.call(program.kernel(rope), "qn", "pos_arr", c.heads * half)
    program.call(program.kernel(rope), "kn", "pos_arr", c.kv_heads * half)
    program.call(program.kernel(store), "kn", "vbuf", "attn_k_cache", "attn_v_cache", "pos_arr", kv_width)
    program.call(program.kernel(scores), "qn", "attn_k_cache", "pos_arr", "scores", c.heads * context)
    program.call(program.kernel(mix), "scores", "attn_v_cache", "qgate", "pos_arr", "attn", c.heads * c.head_dim)
    program.call(program.kernel(_matvec(encodings["w_o"], c.heads * c.head_dim)), "w_o", "attn", "attn_out", c.dim)
    program.call(program.kernel(("add", _ADD)), "h_state", "attn_out", c.dim)
    _ffn(program, c, encodings)
    return program.render(["h_state", "attn_k_cache", "attn_v_cache"])


def generate_embed_program(config, encoding):
    block, _, intrinsic = _encoding(encoding)
    if intrinsic is None or config.dim % block:
        raise ValueError("token embedding must be a block-quantized row")
    program = _Program("Generated Qwen35 token embedding row decode. Do not edit.")
    program.inputs("emb_row")
    program.scratch("h_state", config.dim)
    program.call(program.kernel(("embed", """def embed_GPU_1(emb_row, out, i):
    out[i] = {intrinsic}(emb_row, i / {block}, i % {block})
""".format(intrinsic=intrinsic, block=block))), "emb_row", "h_state", config.dim)
    return program.render(["h_state"])


def generate_head_program(config, encodings):
    c = config
    program = _Program("Generated Qwen35 final norm, output head and argmax. Do not edit.")
    program.inputs("h_state", "output_norm", "w_output")
    program.scratch("xn", c.dim)
    program.scratch("logits", c.vocab)
    program.scratch("argmax_arr", 1)
    program.call(program.kernel(_rmsnorm(c.dim, c.eps)), "h_state", "output_norm", "xn", 1)
    program.call(program.kernel(_matvec(encodings["w_output"], c.dim)), "w_output", "xn", "logits", c.vocab)
    program.call(program.kernel(("argmax", """def argmax_GPU_1(logits, argmax_arr, gid):
    best = logits[0]
    best_i = 0.0
    j = 1
    while j < {vocab}:
        if logits[j] > best:
            best = logits[j]
            best_i = j
        j = j + 1
    argmax_arr[0] = best_i
""".format(vocab=c.vocab))), "logits", "argmax_arr", 1)
    return program.render(["logits", "argmax_arr"])
