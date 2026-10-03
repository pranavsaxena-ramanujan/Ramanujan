import math

import numpy as np

from .qwen35_programs import ATTN_TENSORS, DELTA_TENSORS


def _halves(raw, start):
    return raw[:, start:start + 2].copy().view("<f2").astype(np.float32)


def dequantize(encoding, raw, columns):
    """Decode a (rows, row_bytes) uint8 matrix into (rows, columns) float32."""
    rows = raw.shape[0]
    if encoding == "gguf-f32":
        return raw.copy().view("<f4").reshape(rows, columns)
    if encoding == "gguf-q4_1":
        blocks = raw.reshape(-1, 20)
        qs = blocks[:, 4:]
        values = np.concatenate([qs & 15, qs >> 4], axis=1).astype(np.float32)
        return (values * _halves(blocks, 0) + _halves(blocks, 2)).reshape(rows, columns)
    if encoding == "gguf-q5_k":
        blocks = raw.reshape(-1, 176)
        scales = blocks[:, 4:16].astype(np.int32)
        sc = np.empty((len(blocks), 8), np.float32)
        mn = np.empty((len(blocks), 8), np.float32)
        for g in range(4):
            sc[:, g] = scales[:, g] & 63
            mn[:, g] = scales[:, g + 4] & 63
            sc[:, g + 4] = (scales[:, g + 8] & 15) | ((scales[:, g] >> 6) << 4)
            mn[:, g + 4] = (scales[:, g + 8] >> 4) | ((scales[:, g + 4] >> 6) << 4)
        high = blocks[:, 16:48]
        low = blocks[:, 48:176].reshape(-1, 4, 32)
        quants = np.empty((len(blocks), 8, 32), np.float32)
        for g in range(8):
            quants[:, g] = ((low[:, g // 2] >> (4 * (g % 2))) & 15) + ((high >> g) & 1) * 16
        values = (_halves(blocks, 0)[:, :, None] * sc[:, :, None] * quants
                  - _halves(blocks, 2)[:, :, None] * mn[:, :, None])
        return values.reshape(rows, columns)
    if encoding == "gguf-q6_k":
        blocks = raw.reshape(-1, 210)
        low = blocks[:, :128].reshape(-1, 2, 2, 32)
        high = blocks[:, 128:192].reshape(-1, 2, 32)
        subscales = blocks[:, 192:208].copy().view(np.int8).reshape(-1, 2, 8).astype(np.float32)
        scale = _halves(blocks, 208)[:, :, None]
        values = np.empty((len(blocks), 2, 4, 32), np.float32)
        for segment in range(4):
            quant = ((low[:, :, segment % 2] >> ((segment // 2) * 4)) & 15) | (((high >> (2 * segment)) & 3) << 4)
            sub = np.repeat(subscales[:, :, 2 * segment:2 * segment + 2], 16, axis=2)
            values[:, :, segment] = scale * sub * (quant.astype(np.float32) - 32)
        return values.reshape(rows, columns)
    raise ValueError("unsupported encoding: {0}".format(encoding))


def _row_bytes(tensor):
    columns = tensor["shape"][-1]
    block, block_bytes = {"gguf-f32": (1, 4), "gguf-q4_1": (32, 20),
                          "gguf-q5_k": (256, 176), "gguf-q6_k": (256, 210)}[tensor["encoding"]]
    return columns // block * block_bytes


def read_rows(tensor, start, count):
    row_bytes = _row_bytes(tensor)
    raw = np.fromfile(tensor["file"], dtype=np.uint8, count=count * row_bytes, offset=start * row_bytes)
    if raw.size != count * row_bytes:
        raise ValueError("truncated tensor file: {0}".format(tensor["file"]))
    return dequantize(tensor["encoding"], raw.reshape(count, row_bytes), tensor["shape"][-1])


def matvec(tensor, x, chunk_bytes=64 << 20):
    rows = tensor["shape"][0]
    step = max(1, chunk_bytes // _row_bytes(tensor))
    out = np.empty(rows, np.float32)
    for start in range(0, rows, step):
        count = min(step, rows - start)
        out[start:start + count] = read_rows(tensor, start, count) @ x
    return out


def vector(tensor):
    return np.fromfile(tensor["file"], dtype="<f4")


def _silu(v):
    return v / (1.0 + np.exp(-v))


class Qwen35Reference:
    """NumPy reference implementation for Qwen3.8 inference math."""

    def __init__(self, config, shards):
        self.c = config
        self.tensors = {name: t for shard in shards for name, t in shard["tensors"].items()}

    def _t(self, name):
        return self.tensors[name]

    def _rms(self, x, weight):
        return x / np.sqrt(np.mean(x.astype(np.float64) ** 2) + self.c.eps).astype(np.float32) * weight

    def new_state(self, layer):
        c = self.c
        if c.is_attention(layer):
            return {"k": [], "v": []}
        return {"conv": np.zeros((c.conv_dim, c.conv_kernel - 1), np.float32),
                "S": np.zeros((c.v_heads, c.state_size, c.state_size), np.float32)}

    def embed(self, token):
        return read_rows(self._t("token_embd.weight"), token, 1)[0]

    def layer(self, layer, x, state, pos):
        p = "blk.{0}.".format(layer)
        roles = ATTN_TENSORS if self.c.is_attention(layer) else DELTA_TENSORS
        t = {role: self._t(p + suffix) for role, suffix in roles.items()}
        xn = self._rms(x, vector(t["attn_norm"]))
        mixed = self._attention(t, xn, state, pos) if "w_q" in t else self._delta(t, xn, state)
        h = x + mixed
        hn = self._rms(h, vector(t["post_norm"]))
        g = matvec(t["w_ffn_gate"], hn)
        u = matvec(t["w_ffn_up"], hn)
        return h + matvec(t["w_ffn_down"], _silu(g) * u)

    def _delta(self, t, xn, state):
        c = self.c
        qkv = matvec(t["w_qkv"], xn)
        z = matvec(t["w_z"], xn)
        beta = 1.0 / (1.0 + np.exp(-matvec(t["w_beta"], xn)))
        v = matvec(t["w_alpha"], xn) + vector(t["dt_bias"])
        gate = np.where(v > 20, v, np.log1p(np.exp(np.minimum(v, 20)))) * vector(t["ssm_a"])
        conv = vector(t["conv_w"]).reshape(c.conv_dim, c.conv_kernel)
        co = _silu(conv[:, -1] * qkv + np.sum(conv[:, :-1] * state["conv"], axis=1))
        state["conv"] = np.concatenate([state["conv"][:, 1:], qkv[:, None]], axis=1)
        s = c.state_size
        q = co[:c.key_dim].reshape(c.k_heads, s)
        k = co[c.key_dim:2 * c.key_dim].reshape(c.k_heads, s)
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), c.eps)
        k = k / np.maximum(np.linalg.norm(k, axis=1, keepdims=True), c.eps)
        values = co[2 * c.key_dim:].reshape(c.v_heads, s)
        core = np.empty((c.v_heads, s), np.float32)
        for h in range(c.v_heads):
            kh = h % c.k_heads
            S = state["S"][h]
            S *= np.float32(math.exp(gate[h]))
            d = (values[h] - k[kh] @ S) * beta[h]
            S += np.outer(k[kh], d)
            core[h] = (q[kh] / math.sqrt(s)) @ S
        norm = vector(t["ssm_norm"])
        gated = np.stack([self._rms(core[h], norm) for h in range(c.v_heads)]) * _silu(z.reshape(c.v_heads, s))
        return matvec(t["w_out"], gated.reshape(-1))

    def _rope(self, vec, pos):
        c = self.c
        half = c.rope_dims // 2
        angles = pos * np.power(c.rope_theta, -2.0 * np.arange(half) / c.rope_dims)
        cos, sin = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
        a, b = vec[:, :half].copy(), vec[:, half:2 * half].copy()
        vec[:, :half] = a * cos - b * sin
        vec[:, half:2 * half] = b * cos + a * sin

    def _attention(self, t, xn, state, pos):
        c = self.c
        hd = c.head_dim
        full = matvec(t["w_q"], xn).reshape(c.heads, 2, hd)
        q, g = full[:, 0].copy(), full[:, 1].copy()
        k = matvec(t["w_k"], xn).reshape(c.kv_heads, hd)
        v = matvec(t["w_v"], xn).reshape(c.kv_heads, hd)
        q = np.stack([self._rms(row, vector(t["q_norm"])) for row in q])
        k = np.stack([self._rms(row, vector(t["k_norm"])) for row in k])
        self._rope(q, pos)
        self._rope(k, pos)
        if len(state["k"]) != pos:
            raise ValueError("KV cache length does not match position")
        state["k"].append(k)
        state["v"].append(v)
        keys, values = np.stack(state["k"]), np.stack(state["v"])
        group = c.heads // c.kv_heads
        out = np.empty((c.heads, hd), np.float32)
        for h in range(c.heads):
            scores = keys[:, h // group] @ q[h] / math.sqrt(hd)
            weights = np.exp(scores - scores.max())
            out[h] = (weights / weights.sum()) @ values[:, h // group]
        out *= 1.0 / (1.0 + np.exp(-g))
        return matvec(t["w_o"], out.reshape(-1))

    def logits(self, x):
        return matvec(self._t("output.weight"), self._rms(x, vector(self._t("output_norm.weight"))))
