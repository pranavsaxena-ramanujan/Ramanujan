"""Architecture-agnostic description of a decoder-only GGUF model.

GGUF standardizes tensor names (``blk.N.attn_q.weight``) and hyperparameter keys
(``{arch}.embedding_length``), so most of a model's structure can be read from the file:
which projections are fused, which biases and norms exist, whether attention has an
output gate, the FFN layout, tied embeddings, and recurrent (Gated DeltaNet) layers.
A few facts are hardcoded per architecture by llama.cpp instead of stored in GGUF; those
live in ``ARCHITECTURES`` and can be overridden for unknown architectures.
"""
import math
import re
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class QuantType:
    block: int
    block_bytes: int
    intrinsic: Optional[str]


QUANT = {
    "gguf-f32": QuantType(1, 4, None),
    "gguf-f16": QuantType(2, 4, "GGUF_F16_VALUE"),
    "gguf-q4_0": QuantType(32, 18, "GGUF_Q4_0_VALUE"),
    "gguf-q4_1": QuantType(32, 20, "GGUF_Q4_1_VALUE"),
    "gguf-q5_0": QuantType(32, 22, "GGUF_Q5_0_VALUE"),
    "gguf-q5_1": QuantType(32, 24, "GGUF_Q5_1_VALUE"),
    "gguf-q8_0": QuantType(32, 34, "GGUF_Q8_0_VALUE"),
    "gguf-q4_k": QuantType(256, 144, "GGUF_Q4_K_VALUE"),
    "gguf-q5_k": QuantType(256, 176, "GGUF_Q5_K_VALUE"),
    "gguf-q6_k": QuantType(256, 210, "GGUF_Q6_K_VALUE"),
}


def quant(encoding):
    if encoding not in QUANT:
        raise ValueError("unsupported tensor encoding: {0} (supported: {1})".format(
            encoding, ", ".join(sorted(QUANT))))
    return QUANT[encoding]


def tensor_columns(tensor):
    """Float words per CSV row for a tensor file (the native runtime loads raw bytes as floats).

    Kernels index weights by global block, so rows need not be word aligned; when a row is
    not, several rows share one CSV row so the column count still divides the file.
    """
    kind = quant(tensor["encoding"])
    shape = tensor["shape"]
    if len(shape) == 1:
        if tensor["encoding"] != "gguf-f32":
            raise ValueError("vector tensors must be F32")
        return shape[0]
    if shape[-1] % kind.block:
        raise ValueError("tensor row length is not a multiple of the quant block")
    row_bytes = shape[-1] // kind.block * kind.block_bytes
    rows = 1
    for dimension in shape[:-1]:
        rows *= dimension
    for group in (1, 2, 4):
        if row_bytes * group % 4 == 0 and rows % group == 0:
            return row_bytes * group // 4
    raise ValueError("tensor file is not float-word aligned")


@dataclass(frozen=True)
class Semantics:
    """Per-architecture behavior that GGUF metadata does not record."""
    rope_style: str = "norm"         # "norm": rotate adjacent pairs; "neox": rotate halves
    activation: str = "silu"         # FFN activation: "silu" or "gelu" (tanh approximation)
    embed_scale: str = "none"        # "none" or "sqrt_dim" (Gemma)


ARCHITECTURES = {
    "llama": Semantics(rope_style="norm"),
    "qwen2": Semantics(rope_style="neox"),
    "qwen3": Semantics(rope_style="neox"),
    "phi3": Semantics(rope_style="neox"),
    "gemma": Semantics(rope_style="neox", activation="gelu", embed_scale="sqrt_dim"),
    "qwen35": Semantics(rope_style="neox"),
}


@dataclass(frozen=True)
class KindSpec:
    """One distinct layer shape; every layer of a kind runs the same program."""
    id: str
    mixer: str                                  # "attention" or "gated_deltanet"
    encodings: Tuple[Tuple[str, str], ...]      # (role, encoding), sorted
    flags: Tuple[Tuple[str, object], ...]       # (name, value), sorted

    def encoding(self, role):
        return dict(self.encodings)[role]

    def has(self, role):
        return role in dict(self.encodings)

    def flag(self, name):
        return dict(self.flags)[name]


@dataclass(frozen=True)
class LayerSpec:
    index: int
    kind: str
    roles: Dict[str, str]                       # role -> tensor name


@dataclass
class ModelSpec:
    architecture: str
    semantics: Semantics
    dim: int
    vocab: int
    eps: float
    heads: int
    kv_heads: int
    head_dim: int
    rope_dims: int
    rope_theta: float
    rope_position_scale: float
    attn_scale: float
    sliding_window: Optional[int]
    layers: List[LayerSpec]
    kinds: Dict[str, KindSpec]
    embed_tensor: str
    head_roles: Dict[str, str]
    rope_freqs: Optional[str] = None
    ssm: Optional[Dict[str, int]] = None
    assumptions: List[str] = field(default_factory=list)

    @property
    def embed_multiplier(self):
        return math.sqrt(self.dim) if self.semantics.embed_scale == "sqrt_dim" else 1.0

    def kind(self, layer):
        return self.kinds[self.layers[layer].kind]


def role_name(suffix):
    """GGUF tensor suffix -> program array name, e.g. ``ssm_dt.bias`` -> ``ssm_dt_bias``."""
    if suffix.endswith(".weight"):
        suffix = suffix[:-len(".weight")]
    return suffix.replace(".", "_")


_LAYER = re.compile(r"^blk\.([0-9]+)\.(.+)$")
_UNSUPPORTED_LAYER_TENSORS = {
    "ffn_gate_inp": "mixture-of-experts layers",
    "attn_norm_b": "LayerNorm layers", "attn_norm.bias": "LayerNorm layers",
    "ssm_in": "Mamba layers", "ssm_x": "Mamba layers",
    "attn_kv_a_mqa": "multi-head latent attention", "attn_q_a": "multi-head latent attention",
    "time_mix_first": "RWKV layers",
}


def mixer_for(suffixes):
    """Classify a layer by the GGUF tensor suffixes it contains."""
    for marker, feature in _UNSUPPORTED_LAYER_TENSORS.items():
        if any(suffix == marker or suffix.startswith(marker + ".") for suffix in suffixes):
            return "unsupported:" + feature
    if "ssm_conv1d.weight" in suffixes and "ssm_a" in suffixes and "ssm_beta.weight" in suffixes:
        return "gated_deltanet"
    if "attn_q.weight" in suffixes or "attn_qkv.weight" in suffixes:
        return "attention"
    if any(suffix.startswith("nextn.") for suffix in suffixes):
        return "auxiliary"
    return "unsupported:unrecognized layer tensors"


def _int(value, name):
    if isinstance(value, list):
        if not value or any(item != value[0] for item in value):
            raise ValueError("per-layer {0} values are not supported".format(name))
        value = value[0]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("invalid {0}: {1!r}".format(name, value))
    return value


def build_spec(metadata, tensors, overrides=None):
    """Build a ModelSpec from GGUF metadata and a {name: {"shape", "encoding"}} tensor map.

    ``overrides`` may set Semantics fields (rope_style, activation, embed_scale); they are
    required knowledge for architectures that are not in ARCHITECTURES.
    """
    arch = metadata.get("general.architecture")
    if not arch:
        raise ValueError("GGUF metadata has no general.architecture")
    key = lambda name: "{0}.{1}".format(arch, name)
    get = lambda name, default=None: metadata.get(key(name), default)
    assumptions = []
    semantics = ARCHITECTURES.get(arch)
    if semantics is None:
        semantics = Semantics()
        assumptions.append("architecture {0!r} is not in the registry; assuming llama semantics "
                           "(norm RoPE, SiLU, no embedding scale) unless overridden".format(arch))
    if overrides:
        semantics = replace(semantics, **{k: v for k, v in overrides.items() if v is not None})
    if semantics.rope_style not in ("norm", "neox") or semantics.activation not in ("silu", "gelu") \
            or semantics.embed_scale not in ("none", "sqrt_dim"):
        raise ValueError("invalid architecture semantics: {0}".format(semantics))
    if get("attention.layer_norm_rms_epsilon") is None:
        if get("attention.layer_norm_epsilon") is not None:
            raise ValueError("{0} uses LayerNorm, which is not supported (RMSNorm only)".format(arch))
        raise ValueError("missing {0}".format(key("attention.layer_norm_rms_epsilon")))
    for name in ("final_logit_softcapping", "attn_logit_softcapping"):
        if get(name) is not None:
            raise ValueError("{0} (logit soft-capping) is not supported".format(key(name)))
    for name in ("embedding_scale", "residual_scale", "logit_scale"):
        if get(name) is not None and float(get(name)) != 1.0:
            raise ValueError("{0} (scalar multipliers, e.g. Granite) is not supported".format(key(name)))
    scaling = get("rope.scaling.type", "none")
    position_scale = 1.0
    if scaling == "linear":
        position_scale = 1.0 / float(get("rope.scaling.factor", 1.0))
    elif scaling not in ("none", None):
        raise ValueError("RoPE scaling {0!r} is not supported".format(scaling))

    dim = _int(get("embedding_length"), "embedding_length")
    heads = _int(get("attention.head_count"), "attention.head_count")
    kv_heads = _int(get("attention.head_count_kv", heads), "attention.head_count_kv")
    head_dim = _int(get("attention.key_length", dim // heads), "attention.key_length")
    if int(get("attention.value_length", head_dim)) != head_dim:
        raise ValueError("attention value_length must equal key_length")
    if heads % kv_heads:
        raise ValueError("attention head_count must be a multiple of head_count_kv")
    rope_dims = _int(get("rope.dimension_count", head_dim), "rope.dimension_count")
    if rope_dims % 2 or rope_dims > head_dim:
        raise ValueError("rope.dimension_count must be even and at most the head size")
    block_count = _int(get("block_count"), "block_count")
    layer_count = block_count - int(get("nextn_predict_layers", 0))

    by_layer = {}
    for name in tensors:
        match = _LAYER.fullmatch(name)
        if match:
            by_layer.setdefault(int(match.group(1)), {})[match.group(2)] = name
    if set(by_layer) != set(range(block_count)):
        raise ValueError("layer tensors do not cover block_count={0}".format(block_count))
    ssm = None
    if any(mixer_for(by_layer[layer]) == "gated_deltanet" for layer in range(layer_count)):
        ssm = {name: _int(get("ssm." + name), "ssm." + name)
               for name in ("conv_kernel", "state_size", "group_count", "time_step_rank", "inner_size")}
        if ssm["inner_size"] != ssm["time_step_rank"] * ssm["state_size"] or \
                ssm["time_step_rank"] % ssm["group_count"] or ssm["conv_kernel"] < 2:
            raise ValueError("unsupported Gated DeltaNet hyperparameters")

    vocab_tokens = metadata.get("tokenizer.ggml.tokens")
    embed = tensors.get("token_embd.weight")
    if embed is None or len(embed["shape"]) != 2 or embed["shape"][1] != dim:
        raise ValueError("missing or misshaped token_embd.weight")
    vocab = embed["shape"][0]
    if vocab_tokens is not None and len(vocab_tokens) > vocab:
        raise ValueError("tokenizer vocabulary is larger than the embedding table")
    spec = ModelSpec(
        architecture=arch, semantics=semantics, dim=dim, vocab=vocab,
        eps=float(get("attention.layer_norm_rms_epsilon")), heads=heads, kv_heads=kv_heads,
        head_dim=head_dim, rope_dims=rope_dims, rope_theta=float(get("rope.freq_base", 10000.0)),
        rope_position_scale=position_scale,
        attn_scale=float(get("attention.scale", 1.0 / math.sqrt(head_dim))),
        sliding_window=get("attention.sliding_window"), layers=[], kinds={},
        embed_tensor="token_embd.weight", head_roles={}, ssm=ssm, assumptions=assumptions)
    if "rope_factors_long.weight" in tensors:
        raise ValueError("LongRoPE (rope_factors_long) models are not supported")
    if "rope_freqs.weight" in tensors:
        if tensors["rope_freqs.weight"]["shape"] != [rope_dims // 2]:
            raise ValueError("rope_freqs.weight must have rope_dims/2 entries")
        spec.rope_freqs = "rope_freqs.weight"

    signatures = {}
    for layer in range(layer_count):
        names = by_layer[layer]
        mixer = mixer_for(names)
        if mixer.startswith("unsupported:") or mixer == "auxiliary":
            raise ValueError("layer {0}: {1} are not supported".format(
                layer, mixer.split(":", 1)[-1] if ":" in mixer else "auxiliary tensors"))
        roles = {role_name(suffix): name for suffix, name in names.items()}
        flags = _layer_flags(spec, mixer, roles, tensors, layer)
        encodings = tuple(sorted((role, tensors[name]["encoding"]) for role, name in roles.items()))
        signature = (mixer, encodings, tuple(sorted(flags.items())))
        if signature not in signatures:
            kind_id = "{0}{1}".format("attn" if mixer == "attention" else "delta",
                                      sum(1 for s in signatures if s[0] == mixer))
            signatures[signature] = kind_id
            spec.kinds[kind_id] = KindSpec(kind_id, mixer, encodings, signature[2])
        spec.layers.append(LayerSpec(layer, signatures[signature], roles))

    output = "output.weight" if "output.weight" in tensors else "token_embd.weight"
    for name in ("output_norm.weight", output):
        if name not in tensors:
            raise ValueError("missing tensor: {0}".format(name))
    if tensors[output]["shape"] != [vocab, dim] or tensors["output_norm.weight"]["shape"] != [dim]:
        raise ValueError("unexpected output head shape")
    spec.head_roles = {"output_norm": "output_norm.weight", "output": output}
    used = set(spec.head_roles.values()) | {spec.embed_tensor}
    used.update(name for layer in spec.layers for name in layer.roles.values())
    if spec.rope_freqs:
        used.add(spec.rope_freqs)
    for name in sorted(used):
        try:
            tensor_columns(tensors[name])
        except ValueError as error:
            raise ValueError("{0}: {1}".format(name, error))
    return spec


def _layer_flags(spec, mixer, roles, tensors, layer):
    shape = lambda role: tensors[roles[role]]["shape"] if role in roles else None
    require = lambda role, expected: _require(layer, roles, tensors, role, expected)
    for role, name in roles.items():
        quant(tensors[name]["encoding"])
        if len(tensors[name]["shape"]) == 1 and tensors[name]["encoding"] != "gguf-f32":
            raise ValueError("vector tensor must be F32: {0}".format(name))
    d, hd = spec.dim, spec.head_dim
    flags = {}
    require("attn_norm", [d])
    if mixer == "attention":
        q_width, kv_width = spec.heads * hd, spec.kv_heads * hd
        if "attn_qkv" in roles:
            require("attn_qkv", [q_width + 2 * kv_width, d])
            flags["qkv"] = "fused"
            flags["q_gate"] = False
        else:
            q_rows = shape("attn_q")[0] if "attn_q" in roles else None
            if q_rows not in (q_width, 2 * q_width):
                raise ValueError("layer {0}: attn_q has {1} rows, expected {2}".format(layer, q_rows, q_width))
            require("attn_q", [q_rows, d])
            require("attn_k", [kv_width, d])
            require("attn_v", [kv_width, d])
            flags["qkv"] = "separate"
            flags["q_gate"] = q_rows == 2 * q_width
        for role in ("attn_q_norm", "attn_k_norm"):
            if role in roles and shape(role) != [hd]:
                raise ValueError("layer {0}: only per-head {1} is supported".format(layer, role))
        if ("attn_q_norm" in roles) != ("attn_k_norm" in roles):
            raise ValueError("layer {0}: Q and K norms must both be present".format(layer))
        flags["qk_norm"] = "attn_q_norm" in roles
        require("attn_output", [d, q_width])
    else:
        s = spec.ssm
        key_dim = s["group_count"] * s["state_size"]
        conv_dim = 2 * key_dim + s["inner_size"]
        v_heads = s["time_step_rank"]
        for role, expected in (("attn_qkv", [conv_dim, d]), ("attn_gate", [s["inner_size"], d]),
                               ("ssm_beta", [v_heads, d]), ("ssm_alpha", [v_heads, d]),
                               ("ssm_dt_bias", [v_heads]), ("ssm_a", [v_heads]),
                               ("ssm_conv1d", [conv_dim, s["conv_kernel"]]),
                               ("ssm_norm", [s["state_size"]]), ("ssm_out", [d, s["inner_size"]])):
            require(role, expected)
    for role in roles:
        if role.endswith("_bias") and role != "ssm_dt_bias":
            base = role[:-len("_bias")]
            if base not in roles or shape(role) != [shape(base)[0]]:
                raise ValueError("layer {0}: bias {1} does not match its weight".format(layer, role))
    # llama/qwen name the pre-FFN norm ffn_norm; qwen35 names it post_attention_norm. When
    # both exist (Gemma 2 style), post_attention_norm normalizes the attention output.
    if "ffn_norm" in roles:
        flags["ffn_norm"] = "ffn_norm"
        flags["post_attn_norm"] = "post_attention_norm" in roles
    elif "post_attention_norm" in roles:
        flags["ffn_norm"] = "post_attention_norm"
        flags["post_attn_norm"] = False
    else:
        raise ValueError("layer {0}: no pre-FFN norm (ffn_norm or post_attention_norm)".format(layer))
    for role in ("ffn_norm", "post_attention_norm", "post_ffw_norm"):
        if role in roles:
            require(role, [d])
    flags["post_ffn_norm"] = "post_ffw_norm" in roles
    if "ffn_down" not in roles or "ffn_up" not in roles:
        raise ValueError("layer {0}: missing ffn_up/ffn_down".format(layer))
    ffn = shape("ffn_down")[1]
    require("ffn_down", [d, ffn])
    if "ffn_gate" in roles:
        require("ffn_gate", [ffn, d])
        require("ffn_up", [ffn, d])
        flags["ffn"] = "gated"
    elif shape("ffn_up") == [2 * ffn, d]:
        flags["ffn"] = "fused_gate_up"
    else:
        require("ffn_up", [ffn, d])
        flags["ffn"] = "plain"
    flags["ffn_dim"] = ffn
    known = {"attn_norm", "attn_q", "attn_k", "attn_v", "attn_qkv", "attn_q_norm", "attn_k_norm",
             "attn_output", "attn_gate", "ssm_beta", "ssm_alpha", "ssm_dt_bias", "ssm_a", "ssm_conv1d",
             "ssm_norm", "ssm_out", "ffn_norm", "post_attention_norm", "post_ffw_norm",
             "ffn_gate", "ffn_up", "ffn_down"}
    unknown = sorted(role for role in roles if role not in known and not (
        role.endswith("_bias") and role[:-len("_bias")] in known))
    if unknown:
        raise ValueError("layer {0}: unsupported tensors {1}".format(layer, ", ".join(unknown)))
    return flags


def _require(layer, roles, tensors, role, expected):
    if role not in roles:
        raise ValueError("layer {0}: missing tensor role {1}".format(layer, role))
    actual = tensors[roles[role]]["shape"]
    if actual != expected:
        raise ValueError("layer {0}: {1} has shape {2}, expected {3}".format(layer, roles[role], actual, expected))
