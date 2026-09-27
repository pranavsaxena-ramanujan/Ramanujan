"""NumPy reference forward pass for any ``ModelSpec`` (the correctness oracle for programs)."""
import math

import numpy as np

from .llm_spec import quant


def _halves(raw, start):
    return raw[:, start:start + 2].copy().view("<f2").astype(np.float32)


def _nibbles(qs):
    """ggml 32-value nibble layout: low nibbles are values 0-15, high nibbles 16-31."""
    return np.concatenate([qs & 15, qs >> 4], axis=1).astype(np.int32)


def _fifth_bits(qh):
    bits = qh.copy().view("<u4").astype(np.int64)
    return (((bits >> np.arange(32)) & 1) << 4).astype(np.int32)


def _k_scales(blocks):
    scales = blocks[:, 4:16].astype(np.int32)
    sc = np.empty((len(blocks), 8), np.float32)
    mn = np.empty((len(blocks), 8), np.float32)
    for g in range(4):
        sc[:, g] = scales[:, g] & 63
        mn[:, g] = scales[:, g + 4] & 63
        sc[:, g + 4] = (scales[:, g + 8] & 15) | ((scales[:, g] >> 6) << 4)
        mn[:, g + 4] = (scales[:, g + 8] >> 4) | ((scales[:, g + 4] >> 6) << 4)
    return sc, mn


def dequantize(encoding, raw, columns):
    """Decode a (rows, row_bytes) uint8 matrix into (rows, columns) float32."""
    rows = raw.shape[0]
    kind = quant(encoding)
    if encoding == "gguf-f32":
        return raw.copy().view("<f4").reshape(rows, columns)
    if encoding == "gguf-f16":
        return raw.copy().view("<f2").astype(np.float32).reshape(rows, columns)
    blocks = raw.reshape(-1, kind.block_bytes)
    if encoding == "gguf-q4_0":
        values = (_nibbles(blocks[:, 2:18]) - 8) * _halves(blocks, 0)
    elif encoding == "gguf-q4_1":
        values = _nibbles(blocks[:, 4:20]) * _halves(blocks, 0) + _halves(blocks, 2)
    elif encoding == "gguf-q5_0":
        values = ((_nibbles(blocks[:, 6:22]) | _fifth_bits(blocks[:, 2:6])) - 16) * _halves(blocks, 0)
    elif encoding == "gguf-q5_1":
        values = (_nibbles(blocks[:, 8:24]) | _fifth_bits(blocks[:, 4:8])) * _halves(blocks, 0) + _halves(blocks, 2)
    elif encoding == "gguf-q8_0":
        values = blocks[:, 2:34].copy().view(np.int8).astype(np.float32) * _halves(blocks, 0)
    elif encoding == "gguf-q4_k":
        sc, mn = _k_scales(blocks)
        qs = blocks[:, 16:144].reshape(-1, 4, 32)
        quants = np.empty((len(blocks), 8, 32), np.float32)
        for g in range(8):
            quants[:, g] = (qs[:, g // 2] >> (4 * (g % 2))) & 15
        values = (_halves(blocks, 0)[:, :, None] * sc[:, :, None] * quants
                  - _halves(blocks, 2)[:, :, None] * mn[:, :, None])
    elif encoding == "gguf-q5_k":
        sc, mn = _k_scales(blocks)
        high = blocks[:, 16:48]
        low = blocks[:, 48:176].reshape(-1, 4, 32)
        quants = np.empty((len(blocks), 8, 32), np.float32)
        for g in range(8):
            quants[:, g] = ((low[:, g // 2] >> (4 * (g % 2))) & 15) + ((high >> g) & 1) * 16
        values = (_halves(blocks, 0)[:, :, None] * sc[:, :, None] * quants
                  - _halves(blocks, 2)[:, :, None] * mn[:, :, None])
    elif encoding == "gguf-q6_k":
        low = blocks[:, :128].reshape(-1, 2, 2, 32)
        high = blocks[:, 128:192].reshape(-1, 2, 32)
        subscales = blocks[:, 192:208].copy().view(np.int8).reshape(-1, 2, 8).astype(np.float32)
        scale = _halves(blocks, 208)[:, :, None]
        values = np.empty((len(blocks), 2, 4, 32), np.float32)
        for segment in range(4):
            q = ((low[:, :, segment % 2] >> ((segment // 2) * 4)) & 15) | (((high >> (2 * segment)) & 3) << 4)
            sub = np.repeat(subscales[:, :, 2 * segment:2 * segment + 2], 16, axis=2)
            values[:, :, segment] = scale * sub * (q.astype(np.float32) - 32)
    else:
        raise ValueError("unsupported encoding: {0}".format(encoding))
    return np.asarray(values, np.float32).reshape(rows, columns)


def row_bytes(tensor):
    kind = quant(tensor["encoding"])
    return tensor["shape"][-1] // kind.block * kind.block_bytes


def read_rows(tensor, start, count):
    size = row_bytes(tensor)
    raw = np.fromfile(tensor["file"], dtype=np.uint8, count=count * size, offset=start * size)
    if raw.size != count * size:
        raise ValueError("truncated tensor file: {0}".format(tensor["file"]))
    return dequantize(tensor["encoding"], raw.reshape(count, size), tensor["shape"][-1])


def matvec(tensor, x, chunk_bytes=64 << 20):
    rows = tensor["shape"][0]
    step = max(1, chunk_bytes // row_bytes(tensor))
    out = np.empty(rows, np.float32)
    for start in range(0, rows, step):
        count = min(step, rows - start)
        out[start:start + count] = read_rows(tensor, start, count) @ x
    return out


def vector(tensor):
    return np.fromfile(tensor["file"], dtype="<f4")


def _silu(v):
    return v / (1.0 + np.exp(-v))


def _gelu(v):
    return 0.5 * v * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (v + 0.044715 * v ** 3)))


class LlmReference:
    """Single-token decoder forward pass driven entirely by the ModelSpec."""

    def __init__(self, spec, tensors):
        self.spec = spec
        self.tensors = tensors

    def _rms(self, x, weight):
        return x / np.sqrt(np.mean(x.astype(np.float64) ** 2) + self.spec.eps).astype(np.float32) * weight

    def _act(self, v):
        return _gelu(v) if self.spec.semantics.activation == "gelu" else _silu(v)

    def new_state(self, layer):
        s = self.spec
        if s.kind(layer).mixer == "attention":
            return {"k": [], "v": []}
        m = s.ssm
        conv_dim = 2 * m["group_count"] * m["state_size"] + m["inner_size"]
        return {"conv": np.zeros((conv_dim, m["conv_kernel"] - 1), np.float32),
                "S": np.zeros((m["time_step_rank"], m["state_size"], m["state_size"]), np.float32)}

    def embed(self, token):
        return read_rows(self.tensors[self.spec.embed_tensor], token, 1)[0] * np.float32(self.spec.embed_multiplier)

    def layer(self, layer, x, state, pos):
        spec, kind = self.spec, self.spec.kind(layer)
        t = {role: self.tensors[name] for role, name in spec.layers[layer].roles.items()}

        def linear(role, value):
            out = matvec(t[role], value)
            return out + vector(t[role + "_bias"]) if role + "_bias" in t else out

        xn = self._rms(x, vector(t["attn_norm"]))
        if kind.mixer == "attention":
            mixed = self._attention(kind, t, linear, xn, state, pos)
        else:
            mixed = self._delta(t, linear, xn, state)
        if kind.flag("post_attn_norm"):
            mixed = self._rms(mixed, vector(t["post_attention_norm"]))
        h = x + mixed
        hn = self._rms(h, vector(t[kind.flag("ffn_norm")]))
        ffn = kind.flag("ffn_dim")
        if kind.flag("ffn") == "gated":
            hidden = self._act(linear("ffn_gate", hn)) * linear("ffn_up", hn)
        elif kind.flag("ffn") == "fused_gate_up":
            gate_up = linear("ffn_up", hn)
            hidden = self._act(gate_up[:ffn]) * gate_up[ffn:]
        else:
            hidden = self._act(linear("ffn_up", hn))
        out = linear("ffn_down", hidden)
        if kind.flag("post_ffn_norm"):
            out = self._rms(out, vector(t["post_ffw_norm"]))
        return h + out

    def _rope(self, vec, pos):
        s = self.spec
        half = s.rope_dims // 2
        angles = pos * s.rope_position_scale * np.power(s.rope_theta, -2.0 * np.arange(half) / s.rope_dims)
        if s.rope_freqs:
            angles = angles / vector(self.tensors[s.rope_freqs])
        cos, sin = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
        if s.semantics.rope_style == "neox":
            first, second = slice(0, half), slice(half, 2 * half)
        else:
            first, second = slice(0, 2 * half, 2), slice(1, 2 * half, 2)
        a, b = vec[:, first].copy(), vec[:, second].copy()
        vec[:, first] = a * cos - b * sin
        vec[:, second] = b * cos + a * sin

    def _attention(self, kind, t, linear, xn, state, pos):
        s = self.spec
        hd, q_width, kv_width = s.head_dim, s.heads * s.head_dim, s.kv_heads * s.head_dim
        gate = None
        if kind.flag("qkv") == "fused":
            qkv = linear("attn_qkv", xn)
            q, k, v = qkv[:q_width], qkv[q_width:q_width + kv_width], qkv[q_width + kv_width:]
        else:
            q, k, v = linear("attn_q", xn), linear("attn_k", xn), linear("attn_v", xn)
        if kind.flag("q_gate"):
            full = q.reshape(s.heads, 2, hd)
            q, gate = full[:, 0].copy(), full[:, 1].copy()
        q = q.reshape(s.heads, hd).copy()
        k = k.reshape(s.kv_heads, hd).copy()
        v = v.reshape(s.kv_heads, hd)
        if kind.flag("qk_norm"):
            q = np.stack([self._rms(row, vector(t["attn_q_norm"])) for row in q])
            k = np.stack([self._rms(row, vector(t["attn_k_norm"])) for row in k])
        self._rope(q, pos)
        self._rope(k, pos)
        if len(state["k"]) != pos:
            raise ValueError("KV cache length does not match position")
        state["k"].append(k)
        state["v"].append(v)
        keys, values = np.stack(state["k"]), np.stack(state["v"])
        group = s.heads // s.kv_heads
        out = np.empty((s.heads, hd), np.float32)
        for h in range(s.heads):
            scores = keys[:, h // group] @ q[h] * s.attn_scale
            weights = np.exp(scores - scores.max())
            out[h] = (weights / weights.sum()) @ values[:, h // group]
        if gate is not None:
            out *= 1.0 / (1.0 + np.exp(-gate))
        return linear("attn_output", out.reshape(-1))

    def _delta(self, t, linear, xn, state):
        m = self.spec.ssm
        size, k_heads, v_heads = m["state_size"], m["group_count"], m["time_step_rank"]
        key_dim = k_heads * size
        qkv = linear("attn_qkv", xn)
        z = linear("attn_gate", xn)
        beta = 1.0 / (1.0 + np.exp(-linear("ssm_beta", xn)))
        v = linear("ssm_alpha", xn) + vector(t["ssm_dt_bias"])
        gate = np.where(v > 20, v, np.log1p(np.exp(np.minimum(v, 20)))) * vector(t["ssm_a"])
        conv = vector(t["ssm_conv1d"]).reshape(-1, m["conv_kernel"])
        co = _silu(conv[:, -1] * qkv + np.sum(conv[:, :-1] * state["conv"], axis=1))
        state["conv"] = np.concatenate([state["conv"][:, 1:], qkv[:, None]], axis=1)
        q = co[:key_dim].reshape(k_heads, size)
        k = co[key_dim:2 * key_dim].reshape(k_heads, size)
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), self.spec.eps)
        k = k / np.maximum(np.linalg.norm(k, axis=1, keepdims=True), self.spec.eps)
        values = co[2 * key_dim:].reshape(v_heads, size)
        core = np.empty((v_heads, size), np.float32)
        for h in range(v_heads):
            kh = h % k_heads
            S = state["S"][h]
            S *= np.float32(math.exp(gate[h]))
            d = (values[h] - k[kh] @ S) * beta[h]
            S += np.outer(k[kh], d)
            core[h] = (q[kh] / math.sqrt(size)) @ S
        norm = vector(t["ssm_norm"])
        gated = np.stack([self._rms(core[h], norm) for h in range(v_heads)]) * _silu(z.reshape(v_heads, size))
        return linear("ssm_out", gated.reshape(-1))

    def logits(self, x):
        s = self.spec
        return matvec(self.tensors[s.head_roles["output"]],
                      self._rms(x, vector(self.tensors[s.head_roles["output_norm"]])))
