import ast
from pathlib import Path
from typing import Iterable, List


_RULE_LINE = "# " + "═" * 14


def generate_phi3_prefill_kernel(reference_kernel: Path, layer_start: int,
                                 layer_end: int, include_output: bool) -> str:
    source = reference_kernel.read_text(encoding="utf-8")
    functions = _function_source(source, [
        "matmul_4bit_GPU_2",
        "rmsnorm_GPU_1",
        "rope_and_cache_GPU_2",
        "causal_attn_GPU_2",
        "silu_GPU_2",
        "residual_add_GPU_2",
        "copy_GPU_2",
        "logits_compute_GPU_1",
        "argmax_GPU_1",
    ])
    functions = _use_integer_decode_residual_row(functions)
    layer_blocks = _prefill_layer_blocks(source, layer_start, layer_end)
    cache_names = [name for layer in range(layer_start, layer_end)
                   for name in ("l{0}_k_cache".format(layer), "l{0}_v_cache".format(layer))]
    weight_names = [name for layer in range(layer_start, layer_end) for name in _layer_weight_names(layer)]

    declarations = [
        "n_seq = params[0]",
        "h_ln1 = [0 for _ in range(3145728)]",
        "h_ln2 = [0 for _ in range(3145728)]",
        "attn_out = [0 for _ in range(3145728)]",
        "scores_2d = [0 for _ in range(33554432)]",
        "qkv_buf = [0 for _ in range(9437184)]",
        "h_attn_buf = [0 for _ in range(3145728)]",
        "h_ff_buf = [0 for _ in range(16777216)]",
        "h_out_buf = [0 for _ in range(3145728)]",
        "h_state = [0 for _ in range(3145728)]",
        "kp_qkv = [0 for _ in range(3)]",
        "kp_proj = [0 for _ in range(3)]",
        "kp_fc = [0 for _ in range(3)]",
        "kp_fcp = [0 for _ in range(3)]",
        "kp_qkv[0] = 3072.0",
        "kp_qkv[1] = 9216.0",
        "kp_qkv[2] = 512.0",
        "kp_proj[0] = 3072.0",
        "kp_proj[1] = 3072.0",
        "kp_proj[2] = 512.0",
        "kp_fc[0] = 3072.0",
        "kp_fc[1] = 16384.0",
        "kp_fc[2] = 512.0",
        "kp_fcp[0] = 16384.0",
        "kp_fcp[1] = 3072.0",
        "kp_fcp[2] = 1366.0",
    ]
    if include_output:
        declarations.extend(["logits = [0 for _ in range(32064)]", "argmax_arr = [0 for _ in range(1)]"])
    declarations.extend("{0} = [0 for _ in range(3145728)]".format(name) for name in cache_names)

    shared_buffers = [
        "hidden", "h_state", "h_ln1", "h_ln2", "attn_out", "scores_2d", "qkv_buf",
        "h_attn_buf", "h_ff_buf", "h_out_buf", "cos_cache", "sin_cache",
        "cur_n_seq_arr", "kp_qkv", "kp_proj", "kp_fc", "kp_fcp",
    ]
    if include_output:
        shared_buffers.extend(["logits", "argmax_arr", "ln_f_g", "lm_head_1", "lm_head_2"])
    loads = ["LOAD_MEM({0})".format(name) for name in shared_buffers]
    releases = ["RELEASE_MEM({0})".format(name) for name in weight_names + shared_buffers]
    output_lines: List[str] = []
    return_names = ["h_state"] + cache_names
    if include_output:
        output_lines.extend([
            "rmsnorm_GPU_1(h_state, ln_f_g, h_ln1, n_seq)",
            "logits_compute_GPU_1(h_ln1, lm_head_1, lm_head_2, logits, cur_n_seq_arr, 32064)",
            "argmax_GPU_1(logits, argmax_arr, 1)",
        ])
        return_names.append("argmax_arr")
    syncs = ["GPU_SYNC({0})".format(name) for name in return_names]
    releases.extend("RELEASE_MEM({0})".format(name) for name in cache_names)

    generated = "\n\n".join([
        "# Generated Phi-3 Ramanujan prefill shard. Do not edit.",
        "\n".join(declarations),
        functions,
        "\n".join(loads),
        "copy_GPU_2(hidden, h_state, n_seq, 3072)",
        layer_blocks,
        "\n".join(output_lines),
        "\n".join(syncs),
        "\n".join(releases),
        "RETURN({0})".format(", ".join(return_names)),
        "",
    ])
    ast.parse(generated)
    return generated


def generate_phi3_decode_kernel(reference_kernel: Path, layer_start: int,
                                layer_end: int, include_output: bool) -> str:
    source = reference_kernel.read_text(encoding="utf-8")
    functions = _function_source(source, [
        "matmul_4bit_decode_GPU_1",
        "rmsnorm_decode_GPU_1",
        "rope_and_cache_decode_GPU_1",
        "causal_attn_k_decode_GPU_2",
        "causal_attn_softmax_decode_GPU_1",
        "causal_attn_v_decode_GPU_2",
        "silu_decode_GPU_1",
        "residual_add_decode_GPU_1",
        "logits_compute_GPU_1",
        "argmax_GPU_1",
    ])
    layer_blocks = _decode_layer_blocks(source, layer_start, layer_end)
    cache_names = [name for layer in range(layer_start, layer_end)
                   for name in ("l{0}_k_cache".format(layer), "l{0}_v_cache".format(layer))]
    weight_names = [name for layer in range(layer_start, layer_end) for name in _layer_weight_names(layer)]
    declarations = [
        "h_ln1 = [0 for _ in range(3145728)]",
        "h_ln2 = [0 for _ in range(3145728)]",
        "attn_out = [0 for _ in range(3145728)]",
        "scores_2d = [0 for _ in range(33554432)]",
        "qkv_buf = [0 for _ in range(9437184)]",
        "h_attn_buf = [0 for _ in range(3145728)]",
        "h_ff_buf = [0 for _ in range(16777216)]",
        "h_out_buf = [0 for _ in range(3145728)]",
        "kp_qkv = [0 for _ in range(3)]",
        "kp_proj = [0 for _ in range(3)]",
        "kp_fc = [0 for _ in range(3)]",
        "kp_fcp = [0 for _ in range(3)]",
        "kp_qkv[0] = 3072.0", "kp_qkv[1] = 9216.0", "kp_qkv[2] = 512.0",
        "kp_proj[0] = 3072.0", "kp_proj[1] = 3072.0", "kp_proj[2] = 512.0",
        "kp_fc[0] = 3072.0", "kp_fc[1] = 16384.0", "kp_fc[2] = 512.0",
        "kp_fcp[0] = 16384.0", "kp_fcp[1] = 3072.0", "kp_fcp[2] = 1366.0",
    ]
    if include_output:
        declarations.extend(["logits = [0 for _ in range(32064)]", "argmax_arr = [0 for _ in range(1)]"])
    shared_buffers = [
        "h_state", "h_ln1", "h_ln2", "attn_out", "scores_2d", "qkv_buf",
        "h_attn_buf", "h_ff_buf", "h_out_buf", "cos_cache", "sin_cache",
        "cur_n_seq_arr", "kp_qkv", "kp_proj", "kp_fc", "kp_fcp",
    ]
    if include_output:
        shared_buffers.extend(["logits", "argmax_arr", "ln_f_g", "lm_head_1", "lm_head_2"])
    loads = [
        "LOAD_MEM({0})".format(name)
        for name in weight_names + shared_buffers + cache_names
    ]
    output_lines: List[str] = []
    return_names = ["h_state"] + cache_names
    if include_output:
        output_lines.extend([
            "rmsnorm_decode_GPU_1(h_state, ln_f_g, h_ln1, cur_n_seq_arr, 3072)",
            "logits_compute_GPU_1(h_ln1, lm_head_1, lm_head_2, logits, cur_n_seq_arr, 32064)",
            "argmax_GPU_1(logits, argmax_arr, 1)",
        ])
        return_names.append("argmax_arr")
    syncs = ["GPU_SYNC({0})".format(name) for name in return_names]
    releases = ["RELEASE_MEM({0})".format(name) for name in weight_names + shared_buffers + cache_names]
    generated = "\n\n".join([
        "# Generated Phi-3 Ramanujan decode shard. Do not edit.",
        "\n".join(declarations), functions, "\n".join(loads), layer_blocks,
        "\n".join(output_lines), "\n".join(syncs), "\n".join(releases),
        "RETURN({0})".format(", ".join(return_names)), "",
    ])
    ast.parse(generated)
    return generated


def _function_source(source: str, names: Iterable[str]) -> str:
    lines = source.splitlines()
    tree = ast.parse(source)
    by_name = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    blocks = []
    for name in names:
        node = by_name.get(name)
        if node is None or node.end_lineno is None:
            raise ValueError("reference kernel is missing function: {0}".format(name))
        blocks.append("\n".join(lines[node.lineno - 1:node.end_lineno]))
    return "\n\n".join(blocks)


def _use_integer_decode_residual_row(source: str) -> str:
    old = """    row = cur_n_seq_arr[0] - 1.0
    idx = row * 3072 + col
    hidden[idx] = hidden[idx] + buf[idx]"""
    new = """    row = cur_n_seq_arr[0] - 1.0
    row_int = 0
    r_f = row + 0.1
    while r_f >= 1024.0:
        row_int = row_int + 1024
        r_f = r_f - 1024.0
    while r_f >= 1.0:
        row_int = row_int + 1
        r_f = r_f - 1.0
    idx = row_int * 3072 + col
    hidden[idx] = hidden[idx] + buf[idx]"""
    if old not in source:
        raise ValueError("decode residual function has an unexpected shape")
    return source.replace(old, new, 1)


def _prefill_layer_blocks(source: str, layer_start: int, layer_end: int) -> str:
    prefill_marker = "PREFILL"
    decode_marker = "DECODE LOOP"
    prefill_start = source.index(prefill_marker)
    prefill_end = source.index(decode_marker, prefill_start)
    prefill = source[prefill_start:prefill_end]
    blocks = []
    for layer in range(layer_start, layer_end):
        marker = "# ── Layer {0} ──".format(layer)
        start = prefill.index(marker)
        next_marker = "# ── Layer {0} ──".format(layer + 1)
        next_index = prefill.find(next_marker, start + len(marker))
        final_index = prefill.find("# Final norm", start + len(marker))
        candidates = [index for index in (next_index, final_index) if index >= 0]
        end = min(candidates) if candidates else len(prefill)
        blocks.append(prefill[start:end].strip())
    return "\n\n".join(blocks)


def _decode_layer_blocks(source: str, layer_start: int, layer_end: int) -> str:
    decode = source[source.index("DECODE LOOP"):]
    blocks = []
    for layer in range(layer_start, layer_end):
        marker = "    # ── Layer {0} ──".format(layer)
        start = decode.index(marker)
        next_marker = "    # ── Layer {0} ──".format(layer + 1)
        next_index = decode.find(next_marker, start + len(marker))
        final_index = decode.find("    rmsnorm_decode_GPU_1(h_state, ln_f_g", start + len(marker))
        candidates = [index for index in (next_index, final_index) if index >= 0]
        end = min(candidates) if candidates else len(decode)
        lines = decode[start:end].strip().splitlines()
        blocks.append("\n".join(line[4:] if line.startswith("    ") else line for line in lines))
    return "\n\n".join(blocks)


def _layer_weight_names(layer: int) -> List[str]:
    prefix = "l{0}_".format(layer)
    return [prefix + suffix for suffix in (
        "qkv_packed", "o_packed", "gate_up_packed", "down_packed",
        "qkv_scales", "o_scales", "gate_up_scales", "down_scales", "ln1_g", "ln2_g",
    )]