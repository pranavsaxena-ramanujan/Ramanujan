// OpenCL kernels for the Ramanujan LLM runtime (libramanujan_llm).
//
// Built once per device with -DWG=<work-group size>. Weights stay in raw GGUF block layout;
// every quant type exposes one decoder that expands a 32-value chunk of a row into two
// float16 vectors, so matvec, fused gate/up and embedding kernels are shared by all types.
// All kernels are enqueued on one in-order queue with no host synchronization between them.

#ifndef WG
#define WG 64
#endif

#define HALF(p, offset) vload_half(0, (__global const half *)((p) + (offset)))

__constant uint16 BIT_LO = (uint16)(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
__constant uint16 BIT_HI = (uint16)(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);

inline uint qh32(__global const uchar *p) {
    return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
}

inline float sum16(float16 v) {
    float8 s8 = v.lo + v.hi;
    float4 s4 = s8.lo + s8.hi;
    return s4.x + s4.y + s4.z + s4.w;
}

// ---- 32-value chunk decoders: a = values 0..15, b = values 16..31 of chunk c of a row ----

inline void decode_f32(__global const uchar *row, int c, float16 *a, float16 *b) {
    __global const float *w = (__global const float *)row + c * 32;
    *a = vload16(0, w);
    *b = vload16(1, w);
}

inline void decode_f16(__global const uchar *row, int c, float16 *a, float16 *b) {
    __global const half *w = (__global const half *)row + c * 32;
    *a = vload_half16(0, w);
    *b = vload_half16(1, w);
}

inline void decode_q4_0(__global const uchar *row, int c, float16 *a, float16 *b) {
    __global const uchar *blk = row + c * 18;
    float d = HALF(blk, 0);
    uchar16 q = vload16(0, blk + 2);
    *a = (convert_float16(q & (uchar)15) - 8.0f) * d;
    *b = (convert_float16(q >> (uchar)4) - 8.0f) * d;
}

inline void decode_q4_1(__global const uchar *row, int c, float16 *a, float16 *b) {
    __global const uchar *blk = row + c * 20;
    float d = HALF(blk, 0), m = HALF(blk, 2);
    uchar16 q = vload16(0, blk + 4);
    *a = convert_float16(q & (uchar)15) * d + m;
    *b = convert_float16(q >> (uchar)4) * d + m;
}

inline void decode_q5_0(__global const uchar *row, int c, float16 *a, float16 *b) {
    __global const uchar *blk = row + c * 22;
    float d = HALF(blk, 0);
    uint16 qh = (uint16)(qh32(blk + 2));
    uchar16 q = vload16(0, blk + 6);
    uint16 lo = convert_uint16(q & (uchar)15) | (((qh >> BIT_LO) & 1u) << 4);
    uint16 hi = convert_uint16(q >> (uchar)4) | (((qh >> BIT_HI) & 1u) << 4);
    *a = (convert_float16(lo) - 16.0f) * d;
    *b = (convert_float16(hi) - 16.0f) * d;
}

inline void decode_q5_1(__global const uchar *row, int c, float16 *a, float16 *b) {
    __global const uchar *blk = row + c * 24;
    float d = HALF(blk, 0), m = HALF(blk, 2);
    uint16 qh = (uint16)(qh32(blk + 4));
    uchar16 q = vload16(0, blk + 8);
    uint16 lo = convert_uint16(q & (uchar)15) | (((qh >> BIT_LO) & 1u) << 4);
    uint16 hi = convert_uint16(q >> (uchar)4) | (((qh >> BIT_HI) & 1u) << 4);
    *a = convert_float16(lo) * d + m;
    *b = convert_float16(hi) * d + m;
}

inline void decode_q8_0(__global const uchar *row, int c, float16 *a, float16 *b) {
    __global const uchar *blk = row + c * 34;
    float d = HALF(blk, 0);
    *a = convert_float16(as_char16(vload16(0, blk + 2))) * d;
    *b = convert_float16(as_char16(vload16(1, blk + 2))) * d;
}

// K-quant 6-bit scale/min for group g (0..7) of a 12-byte packed scale array.
inline void k_scale_min(__global const uchar *s, int g, float *sc, float *mn) {
    if (g < 4) {
        *sc = (float)(s[g] & 63);
        *mn = (float)(s[g + 4] & 63);
    } else {
        *sc = (float)((s[g + 4] & 15) | ((s[g - 4] >> 6) << 4));
        *mn = (float)((s[g + 4] >> 4) | ((s[g] >> 6) << 4));
    }
}

inline void decode_q4_k(__global const uchar *row, int c, float16 *a, float16 *b) {
    __global const uchar *blk = row + (c >> 3) * 144;
    int g = c & 7;
    float sc, mn;
    k_scale_min(blk + 4, g, &sc, &mn);
    float d = HALF(blk, 0) * sc, m = HALF(blk, 2) * mn;
    __global const uchar *qs = blk + 16 + (g >> 1) * 32;
    uchar shift = (uchar)((g & 1) * 4);
    *a = convert_float16((vload16(0, qs) >> shift) & (uchar)15) * d - m;
    *b = convert_float16((vload16(1, qs) >> shift) & (uchar)15) * d - m;
}

inline void decode_q5_k(__global const uchar *row, int c, float16 *a, float16 *b) {
    __global const uchar *blk = row + (c >> 3) * 176;
    int g = c & 7;
    float sc, mn;
    k_scale_min(blk + 4, g, &sc, &mn);
    float d = HALF(blk, 0) * sc, m = HALF(blk, 2) * mn;
    __global const uchar *qh = blk + 16;
    __global const uchar *qs = blk + 48 + (g >> 1) * 32;
    uchar shift = (uchar)((g & 1) * 4);
    uchar16 lo_a = (vload16(0, qs) >> shift) & (uchar)15;
    uchar16 lo_b = (vload16(1, qs) >> shift) & (uchar)15;
    uchar16 hb_a = ((vload16(0, qh) >> (uchar)g) & (uchar)1) << (uchar)4;
    uchar16 hb_b = ((vload16(1, qh) >> (uchar)g) & (uchar)1) << (uchar)4;
    *a = convert_float16(lo_a | hb_a) * d - m;
    *b = convert_float16(lo_b | hb_b) * d - m;
}

inline void decode_q6_k(__global const uchar *row, int c, float16 *a, float16 *b) {
    __global const uchar *blk = row + (c >> 3) * 210;
    int g = c & 7, n = g >> 2, s = g & 3;
    float d = HALF(blk, 208);
    __global const uchar *ql = blk + n * 64 + (s & 1) * 32;
    __global const uchar *qh = blk + 128 + n * 32;
    __global const char *sub = (__global const char *)(blk + 192 + n * 8 + 2 * s);
    uchar lshift = (uchar)((s >> 1) * 4), hshift = (uchar)(2 * s);
    uchar16 qa = ((vload16(0, ql) >> lshift) & (uchar)15) | (((vload16(0, qh) >> hshift) & (uchar)3) << (uchar)4);
    uchar16 qb = ((vload16(1, ql) >> lshift) & (uchar)15) | (((vload16(1, qh) >> hshift) & (uchar)3) << (uchar)4);
    *a = (convert_float16(qa) - 32.0f) * (d * (float)sub[0]);
    *b = (convert_float16(qb) - 32.0f) * (d * (float)sub[1]);
}

inline float activation(float v, int act) {
    if (act == 1) {
        return 0.5f * v * (1.0f + tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    }
    return v / (1.0f + exp(-v));
}

// Segmented tree reduction of part[] in groups of `tpr` lanes (tpr is a power of two).
#define SEGMENT_REDUCE(part, lid, tpr)                                                         \
    barrier(CLK_LOCAL_MEM_FENCE);                                                              \
    for (int _s = (tpr) >> 1; _s > 0; _s >>= 1) {                                               \
        if (((lid) & ((tpr) - 1)) < _s) (part)[lid] += (part)[(lid) + _s];                      \
        barrier(CLK_LOCAL_MEM_FENCE);                                                          \
    }

// y[y_off + row] = W[row] . x[x_off..] (+ bias[row]) (+ residual[res_off + row])
// `tpr` lanes cooperate on one row; a work group covers WG / tpr rows.
#define DEFINE_KERNELS(T)                                                                      \
__kernel void matvec_##T(__global const uchar *w, int row_bytes, int chunks, int rows,        \
                         __global const float *x, int x_off, __global float *y, int y_off,     \
                         __global const float *bias, __global const float *residual,          \
                         int res_off, int tpr) {                                               \
    __local float part[WG];                                                                    \
    int lid = get_local_id(0);                                                                 \
    int row = get_group_id(0) * (WG / tpr) + lid / tpr;                                        \
    int lane = lid & (tpr - 1);                                                                \
    float acc = 0.0f;                                                                          \
    if (row < rows) {                                                                          \
        __global const uchar *r = w + (size_t)row * (size_t)row_bytes;                         \
        __global const float *xv = x + x_off;                                                  \
        for (int c = lane; c < chunks; c += tpr) {                                             \
            float16 a, b;                                                                      \
            decode_##T(r, c, &a, &b);                                                          \
            acc += sum16(a * vload16(0, xv + c * 32) + b * vload16(1, xv + c * 32));           \
        }                                                                                      \
    }                                                                                          \
    part[lid] = acc;                                                                           \
    SEGMENT_REDUCE(part, lid, tpr)                                                             \
    if (lane == 0 && row < rows) {                                                             \
        float v = part[lid];                                                                   \
        if (bias) v += bias[row];                                                              \
        if (residual) v += residual[res_off + row];                                            \
        y[y_off + row] = v;                                                                    \
    }                                                                                          \
}                                                                                              \
                                                                                               \
/* out[row] = act(Wg[g_row0 + row] . x) * (Wu[u_row0 + row] . x) */                           \
__kernel void gateup_##T(__global const uchar *wg, int g_row0, __global const uchar *wu,      \
                         int u_row0, int row_bytes, int chunks, int rows,                      \
                         __global const float *x, __global float *out, int act, int tpr) {     \
    __local float pg[WG];                                                                      \
    __local float pu[WG];                                                                      \
    int lid = get_local_id(0);                                                                 \
    int row = get_group_id(0) * (WG / tpr) + lid / tpr;                                        \
    int lane = lid & (tpr - 1);                                                                \
    float ag = 0.0f, au = 0.0f;                                                                \
    if (row < rows) {                                                                          \
        __global const uchar *rg = wg + (size_t)(g_row0 + row) * (size_t)row_bytes;            \
        __global const uchar *ru = wu + (size_t)(u_row0 + row) * (size_t)row_bytes;            \
        for (int c = lane; c < chunks; c += tpr) {                                             \
            float16 x0 = vload16(0, x + c * 32), x1 = vload16(1, x + c * 32);                  \
            float16 a, b;                                                                      \
            decode_##T(rg, c, &a, &b);                                                         \
            ag += sum16(a * x0 + b * x1);                                                      \
            decode_##T(ru, c, &a, &b);                                                         \
            au += sum16(a * x0 + b * x1);                                                      \
        }                                                                                      \
    }                                                                                          \
    pg[lid] = ag;                                                                              \
    pu[lid] = au;                                                                              \
    barrier(CLK_LOCAL_MEM_FENCE);                                                              \
    for (int s = tpr >> 1; s > 0; s >>= 1) {                                                   \
        if (lane < s) {                                                                        \
            pg[lid] += pg[lid + s];                                                            \
            pu[lid] += pu[lid + s];                                                            \
        }                                                                                      \
        barrier(CLK_LOCAL_MEM_FENCE);                                                          \
    }                                                                                          \
    if (lane == 0 && row < rows) out[row] = activation(pg[lid], act) * pu[lid];                \
}                                                                                              \
                                                                                               \
/* out[out_off + i] = row[i] * scale for one decoded row (global size = chunks). */          \
__kernel void embed_##T(__global const uchar *row, __global float *out, int out_off,          \
                        float scale) {                                                         \
    int c = get_global_id(0);                                                                  \
    float16 a, b;                                                                              \
    decode_##T(row, c, &a, &b);                                                                \
    vstore16(a * scale, 0, out + out_off + c * 32);                                            \
    vstore16(b * scale, 1, out + out_off + c * 32);                                            \
}

DEFINE_KERNELS(f32)
DEFINE_KERNELS(f16)
DEFINE_KERNELS(q4_0)
DEFINE_KERNELS(q4_1)
DEFINE_KERNELS(q5_0)
DEFINE_KERNELS(q5_1)
DEFINE_KERNELS(q8_0)
DEFINE_KERNELS(q4_k)
DEFINE_KERNELS(q5_k)
DEFINE_KERNELS(q6_k)

// out[row] = x[row] / rms(x[row]) * w   (one work group per row; in place when out == x)
__kernel void rmsnorm_rows(__global const float *x, int x_off, int x_stride, __global const float *w,
                           __global float *out, int out_off, int out_stride, int cols, float eps) {
    __local float part[WG];
    int lid = get_local_id(0), row = get_group_id(0);
    __global const float *xr = x + x_off + row * x_stride;
    __global float *yr = out + out_off + row * out_stride;
    float acc = 0.0f;
    for (int i = lid; i < cols; i += WG) acc += xr[i] * xr[i];
    part[lid] = acc;
    SEGMENT_REDUCE(part, lid, WG)
    float inv = rsqrt(part[0] / (float)cols + eps);
    for (int i = lid; i < cols; i += WG) yr[i] = xr[i] * inv * w[i];
}

// x[row] /= max(||x[row]||, eps), in place (Gated DeltaNet q/k normalization)
__kernel void l2norm_rows(__global float *x, int x_off, int stride, int cols, float eps) {
    __local float part[WG];
    int lid = get_local_id(0), row = get_group_id(0);
    __global float *xr = x + x_off + row * stride;
    float acc = 0.0f;
    for (int i = lid; i < cols; i += WG) acc += xr[i] * xr[i];
    part[lid] = acc;
    SEGMENT_REDUCE(part, lid, WG)
    float inv = 1.0f / fmax(sqrt(part[0]), eps);
    for (int i = lid; i < cols; i += WG) xr[i] *= inv;
}

__kernel void add_inplace(__global float *a, __global const float *b, int n) {
    int i = get_global_id(0);
    if (i < n) a[i] += b[i];
}

__kernel void add_bias(__global float *a, int a_off, __global const float *b, int n) {
    int i = get_global_id(0);
    if (i < n) a[a_off + i] += b[i];
}

// out[i] = act(x[g_off + i]) * x[u_off + i], or act(x[g_off + i]) when u_off < 0
__kernel void act_mul(__global const float *x, int g_off, int u_off, __global float *out, int n, int act) {
    int i = get_global_id(0);
    if (i >= n) return;
    float v = activation(x[g_off + i], act);
    out[i] = u_off < 0 ? v : v * x[u_off + i];
}

// Rotate the first rope_dims of every head. table row `pos` holds cos[half] then sin[half].
// style 0: adjacent pairs (llama); style 1: halves (NeoX).
__kernel void rope(__global float *v, int off, int head_stride, int heads, int half_dims,
                   __global const float *table, int pos, int style) {
    int gid = get_global_id(0);
    if (gid >= heads * half_dims) return;
    int h = gid / half_dims, i = gid % half_dims;
    __global float *p = v + off + h * head_stride;
    float c = table[pos * 2 * half_dims + i], s = table[pos * 2 * half_dims + half_dims + i];
    int i0 = style == 1 ? i : 2 * i, i1 = style == 1 ? i + half_dims : 2 * i + 1;
    float a = p[i0], b = p[i1];
    p[i0] = a * c - b * s;
    p[i1] = b * c + a * s;
}

__kernel void kv_store(__global const float *k, int k_off, __global const float *v, int v_off,
                       __global float *kc, __global float *vc, int pos, int width) {
    int i = get_global_id(0);
    if (i >= width) return;
    kc[pos * width + i] = k[k_off + i];
    vc[pos * width + i] = v[v_off + i];
}

// One work group per query head: softmax(q K^T * scale) V over positions 0..n_pos-1,
// optionally multiplied by sigmoid(gate) (gate stored at q + gate_off within the head).
__kernel void attention(__global const float *q, int q_off, int q_stride, __global const float *kc,
                        __global const float *vc, __global float *scores, int ctx, __global float *out,
                        int n_pos, int hd, int kv_width, int group, float scale, int gate_off) {
    __local float part[WG];
    int lid = get_local_id(0), h = get_group_id(0), kvh = h / group;
    __global const float *qh = q + q_off + h * q_stride;
    __global float *sc = scores + h * ctx;
    float m = -INFINITY;
    for (int t = lid; t < n_pos; t += WG) {
        __global const float *kr = kc + t * kv_width + kvh * hd;
        float s = 0.0f;
        for (int d = 0; d < hd; d++) s += qh[d] * kr[d];
        s *= scale;
        sc[t] = s;
        m = fmax(m, s);
    }
    part[lid] = m;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = WG >> 1; s > 0; s >>= 1) {
        if (lid < s) part[lid] = fmax(part[lid], part[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mx = part[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    float sum = 0.0f;
    for (int t = lid; t < n_pos; t += WG) {
        float e = exp(sc[t] - mx);
        sc[t] = e;
        sum += e;
    }
    part[lid] = sum;
    SEGMENT_REDUCE(part, lid, WG)
    float inv = 1.0f / part[0];
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (int d = lid; d < hd; d += WG) {
        float acc = 0.0f;
        for (int t = 0; t < n_pos; t++) acc += sc[t] * vc[t * kv_width + kvh * hd + d];
        acc *= inv;
        if (gate_off >= 0) acc *= 1.0f / (1.0f + exp(-qh[gate_off + d]));
        out[h * hd + d] = acc;
    }
}

// ---- Gated DeltaNet (qwen35 linear attention) ----

__kernel void dn_gate(__global float *beta, __global const float *alpha, __global const float *dt_bias,
                      __global const float *a, __global float *decay, int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    beta[i] = 1.0f / (1.0f + exp(-beta[i]));
    float v = alpha[i] + dt_bias[i];
    float sp = v > 20.0f ? v : log1p(exp(fmin(v, 20.0f)));
    decay[i] = exp(sp * a[i]);
}

// Causal depthwise conv1d over [state..., x] per channel, SiLU, then shift x into the state.
__kernel void dn_conv(__global const float *x, __global const float *w, __global float *state,
                      __global float *out, int channels, int taps) {
    int c = get_global_id(0);
    if (c >= channels) return;
    int k1 = taps - 1;
    __global const float *wc = w + c * taps;
    __global float *st = state + c * k1;
    float xc = x[c];
    float acc = wc[k1] * xc;
    for (int j = 0; j < k1; j++) acc += wc[j] * st[j];
    for (int j = 0; j < k1 - 1; j++) st[j] = st[j + 1];
    st[k1 - 1] = xc;
    out[c] = acc / (1.0f + exp(-acc));
}

// One work group per value head: decay S, delta-rule update with k/v/beta, read out with q.
// S[h] is size x size, row i = key dim, column j = value dim.
__kernel void dn_delta(__global float *S, __global const float *co, __global const float *beta,
                       __global const float *decay, __global float *core, int size, int k_heads,
                       int key_dim) {
    int h = get_group_id(0), kh = h % k_heads;
    __global const float *q = co + kh * size;
    __global const float *k = co + key_dim + kh * size;
    __global const float *v = co + 2 * key_dim + h * size;
    __global float *Sh = S + (size_t)h * size * size;
    float dec = decay[h], b = beta[h], inv = rsqrt((float)size);
    for (int j = get_local_id(0); j < size; j += get_local_size(0)) {
        float ks = 0.0f;
        for (int i = 0; i < size; i++) ks += k[i] * (Sh[i * size + j] * dec);
        float dj = (v[j] - ks) * b;
        float acc = 0.0f;
        for (int i = 0; i < size; i++) {
            float s = Sh[i * size + j] * dec + k[i] * dj;
            Sh[i * size + j] = s;
            acc += (q[i] * inv) * s;
        }
        core[h * size + j] = acc;
    }
}

// out[h] = rmsnorm(core[h]) * w * silu(z[h]), one work group per value head.
__kernel void dn_gatenorm(__global const float *core, __global const float *w, __global const float *z,
                          __global float *out, int size, float eps) {
    __local float part[WG];
    int lid = get_local_id(0), h = get_group_id(0);
    __global const float *cr = core + h * size;
    float acc = 0.0f;
    for (int i = lid; i < size; i += WG) acc += cr[i] * cr[i];
    part[lid] = acc;
    SEGMENT_REDUCE(part, lid, WG)
    float inv = rsqrt(part[0] / (float)size + eps);
    for (int i = lid; i < size; i += WG) {
        float zv = z[h * size + i];
        out[h * size + i] = cr[i] * inv * w[i] * (zv / (1.0f + exp(-zv)));
    }
}

__kernel void copy_f32(__global const float *src, int src_off, __global float *dst, int dst_off, int n) {
    int i = get_global_id(0);
    if (i < n) dst[dst_off + i] = src[src_off + i];
}

__kernel void fill_zero(__global float *x, int n) {
    int i = get_global_id(0);
    if (i < n) x[i] = 0.0f;
}
