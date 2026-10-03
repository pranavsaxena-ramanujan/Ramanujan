"""Manifest-backed, conservative admission budgets for the native LLM graph format.

These are combined host/device upper bounds, not just weight file sizes. Backend
admission must preserve the budgets for every live session and check individual
native buffers against the device's maximum allocation (file-transfer sizes
are not allocation sizes for row-streamed embeddings). No cluster-total
resident-weight check is made: streamed models may exceed total cluster memory.
"""
import copy
import math
from pathlib import Path

from .llm_graph import graph_files
from .llm_spec import ARCHITECTURES, quant

UPLOAD_STAGING_BYTES = 32 << 20
RUNTIME_MARGIN_BYTES = 8 << 20
MAX_BYTES = (1 << 63) - 1
MAX_INDEX = (1 << 31) - 1


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= MAX_BYTES:
        raise ValueError("capacity metadata requires positive integer " + label)
    return value


def _row_bytes(tensor):
    kind = quant(tensor["encoding"])
    shape = tensor["shape"]
    if not shape or len(shape) > 2:
        raise ValueError("capacity metadata requires a vector or matrix tensor")
    for size in shape:
        _positive(size, "tensor dimension")
    if shape[-1] % kind.block:
        raise ValueError("capacity metadata requires quant-block-aligned tensor rows")
    return shape[-1] // kind.block * kind.block_bytes


def stage_capacity(spec, tensors, graph, affinity, session):
    """Build one /llm/plan stage, without opening any local or remote sessions.

    Weight and file sizes come from tensorFiles.bytes and must match local files.
    Stream slots include the consumed item plus depth prefetched cyclic items
    (including repeated copies when a session contains only one layer).
    State/workspace follows native/llm/runtime.cpp allocate(), with doubled
    host/device copies, full-context wire buffers, and explicit upload staging.
    """
    if spec.architecture not in ARCHITECTURES:
        raise ValueError("capacity estimates are unavailable for unknown architecture " + spec.architecture)
    context = _positive(graph["max_context"], "max_context")
    depth = _positive(graph["stream_depth"], "stream_depth")
    threads = _positive(graph["stream_threads"], "stream_threads")
    for name in ("dim", "vocab", "heads", "kv_heads", "head_dim", "rope_dims"):
        _positive(getattr(spec, name), name)
        if getattr(spec, name) > MAX_INDEX:
            raise ValueError("capacity dimension exceeds native integer indexing: " + name)
    if max(context, depth, threads, context * spec.kv_heads * spec.head_dim,
           context * spec.heads, context * spec.rope_dims, context * spec.dim,
           2 * spec.heads * spec.head_dim) > MAX_INDEX:
        raise ValueError("capacity context exceeds native integer indexing")
    by_file = {}
    for tensor in tensors.values():
        path = str(tensor["file"])
        if path in by_file and by_file[path] != tensor:
            raise ValueError("conflicting capacity metadata for " + path)
        by_file[path] = tensor
    files = []
    sizes = {}
    for path in graph_files(graph):
        if path not in by_file:
            raise ValueError("capacity metadata missing for " + path)
        tensor = by_file[path]
        size = _positive(tensor.get("bytes"), "tensorFiles.bytes for " + path)
        expected = _row_bytes(tensor) * math.prod(tensor["shape"][:-1])
        if size < expected or (size != expected and not tensor.get("lengthIncludesPadding")):
            raise ValueError("capacity tensor size does not match shape/encoding: " + path)
        stat = Path(path).stat()
        if stat.st_size != size:
            raise ValueError("capacity manifest/file size mismatch: " + path)
        sizes[path] = size
        files.append({"path": path, "bytes": size, "mtime": stat.st_mtime_ns // 1_000_000})

    item_sizes = []
    state_floats = 0
    allocation_bytes = [4 * spec.dim, 4 * context * spec.dim, 4 * spec.heads * context]
    qkv = att = ffn = hidden = 1
    kv_width = spec.kv_heads * spec.head_dim
    ssm = spec.ssm
    if ssm is not None:
        for name in ("state_size", "group_count", "inner_size", "time_step_rank", "conv_kernel"):
            _positive(ssm.get(name), "ssm." + name)
        conv_dim = 2 * ssm["group_count"] * ssm["state_size"] + ssm["inner_size"]
        if max(ssm.values()) > MAX_INDEX or max(
                ssm["time_step_rank"] * ssm["state_size"] ** 2,
                conv_dim * ssm["conv_kernel"]) > MAX_INDEX:
            raise ValueError("capacity recurrent state exceeds native integer indexing")
    for layer in graph["layers"]:
        item_sizes.append(sum(sizes[tensor["file"]] for tensor in layer["tensors"].values()))
        allocation_bytes.extend(_row_bytes(tensor) * math.prod(tensor["shape"][:-1])
                                for tensor in layer["tensors"].values())
        flags = layer["flags"]
        width = _positive(flags.get("ffn_dim"), "ffn_dim")
        if 2 * width > MAX_INDEX:
            raise ValueError("capacity FFN dimension exceeds native integer indexing")
        if flags.get("ffn") not in ("gated", "fused_gate_up", "plain"):
            raise ValueError("capacity estimates unavailable for FFN layout")
        ffn = max(ffn, 2 * width)
        hidden = max(hidden, width)
        if layer["mixer"] == "attention":
            q_width = spec.heads * spec.head_dim
            q_rows = q_width if flags.get("qkv") == "fused" else layer["tensors"]["attn_q"]["shape"][0]
            qkv = max(qkv, q_rows + 2 * kv_width)
            att = max(att, q_width)
            state_floats += 2 * context * kv_width
            allocation_bytes.append(4 * context * kv_width)
        elif layer["mixer"] == "gated_deltanet" and ssm is not None:
            qkv = max(qkv, conv_dim)
            att = max(att, ssm["inner_size"])
            state_floats += (ssm["time_step_rank"] * ssm["state_size"] ** 2
                             + conv_dim * (ssm["conv_kernel"] - 1))
            allocation_bytes.extend((4 * ssm["time_step_rank"] * ssm["state_size"] ** 2,
                                     4 * conv_dim * (ssm["conv_kernel"] - 1)))
        else:
            raise ValueError("capacity estimates unavailable for mixer " + layer["mixer"])
    if "head" in graph:
        item_sizes.append(sum(sizes[tensor["file"]] for tensor in graph["head"].values()))
        allocation_bytes.extend(_row_bytes(tensor) * math.prod(tensor["shape"][:-1])
                                for tensor in graph["head"].values())
        allocation_bytes.append(4 * spec.vocab)

    scratch_floats = (3 * spec.dim + qkv + 2 * att + spec.heads * context + ffn + hidden)
    allocation_bytes.extend(4 * size for size in (qkv, att, ffn, hidden))
    if ssm is not None:
        # The runtime allocates these for every session of an SSM-capable model.
        scratch_floats += ssm["inner_size"] + 3 * ssm["time_step_rank"] + conv_dim
        allocation_bytes.extend(4 * size for size in (ssm["inner_size"], ssm["time_step_rank"], conv_dim))
    if graph["layers"]:
        scratch_floats += context * spec.rope_dims
        allocation_bytes.append(4 * context * spec.rope_dims)
    if "head" in graph:
        scratch_floats += spec.vocab
    embed_bytes = _row_bytes(graph["embed"]) if "embed" in graph else 0
    allocation_bytes.append(embed_bytes)
    rope_bytes = sizes[graph["rope_freqs"]["file"]] if "rope_freqs" in graph else 0
    stream_item = max(item_sizes, default=0) * (depth + 1)
    stream_shared = embed_bytes + rope_bytes + (min(threads, depth) * UPLOAD_STAGING_BYTES if item_sizes else 0)
    # Allocated once per session whatever its layers: the orchestrator counts these once when it
    # merges consecutive pieces into one session (see CapacityDao.VramPlacement).
    shared_scratch = 64 * context * spec.dim + (UPLOAD_STAGING_BYTES if item_sizes else 0) + RUNTIME_MARGIN_BYTES
    scratch_bytes = (8 * scratch_floats + 2 * embed_bytes
                     + (64 * spec.vocab if "head" in graph else 0)
                     + shared_scratch + 4096 * len(files))
    result = {"affinity": affinity, "session": session,
              "weightBytes": sum(sizes.values()), "streamWorkingBytes": stream_item + stream_shared,
              "streamItemBytes": stream_item, "streamSharedBytes": stream_shared,
              "stateBytes": 8 * state_floats, "scratchBytes": scratch_bytes, "sharedScratchBytes": shared_scratch,
              "files": files, "graph": copy.deepcopy(graph), "maxAllocationBytes": max(allocation_bytes)}
    if any(result[key] > MAX_BYTES for key in ("weightBytes", "streamWorkingBytes", "stateBytes",
                                              "scratchBytes", "maxAllocationBytes")):
        raise ValueError("capacity estimate exceeds signed 64-bit byte budget")
    return result


def validate_plan(payload, requested, weights):
    """Fail closed on incomplete, reordered, or non-contiguous assignments."""
    if not isinstance(payload, dict) or payload.get("status") != "SUCCESS":
        raise RuntimeError("capacity plan response is not SUCCESS; reservations may remain pinned")
    plan_id = payload.get("planId")
    if not isinstance(plan_id, str) or not plan_id.strip():
        raise RuntimeError("capacity plan response has no valid planId; reservations may remain pinned")
    assignments = payload.get("stages")
    if not isinstance(assignments, list) or len(assignments) != len(requested):
        raise RuntimeError("capacity plan " + plan_id + " has incomplete stage assignments; retained")
    completed_hosts = set()
    previous_host = None
    for expected, actual in zip(requested, assignments):
        if not isinstance(actual, dict) or any(actual.get(key) != expected[key] for key in ("affinity", "session")):
            raise RuntimeError("capacity plan " + plan_id + " has mismatched stage identities; retained")
        host = actual.get("hostId")
        mode = actual.get("weights")
        if not isinstance(host, str) or not host.strip() or mode not in ("resident", "stream"):
            raise RuntimeError("capacity plan " + plan_id + " has invalid host/mode; retained")
        if weights != "auto" and mode != weights:
            raise RuntimeError("capacity plan " + plan_id + " did not honor --weights; retained")
        if host != previous_host:
            if host in completed_hosts:
                raise RuntimeError("capacity plan " + plan_id + " is not contiguous; retained")
            if previous_host is not None:
                completed_hosts.add(previous_host)
            previous_host = host
    return plan_id, assignments


def merge_pieces(pieces):
    """One stage covering consecutive plan_stages(split=True) pieces (whole layers only)."""
    return {"shard": pieces[0]["shard"], "embed": any(piece["embed"] for piece in pieces),
            "layers": [index for piece in pieces for index in piece["layers"]],
            "head": any(piece["head"] for piece in pieces)}


def validate_placement(payload, requested, weights, dry_run=False):
    """Check an orchestrator VRAM placement: contiguous whole-piece runs covering every piece in
    order, each on one device, with every device used for one contiguous range of the model."""
    if not isinstance(payload, dict) or payload.get("status") != "SUCCESS":
        raise RuntimeError("capacity placement response is not SUCCESS")
    plan_id = payload.get("planId")
    if not dry_run and (not isinstance(plan_id, str) or not plan_id.strip()):
        raise RuntimeError("capacity placement response has no valid planId; reservations may remain pinned")
    label = "capacity placement" if dry_run else "capacity plan " + plan_id
    groups = payload.get("stages")
    if not isinstance(groups, list) or not groups:
        raise RuntimeError(label + " has no stages" + ("" if dry_run else "; retained"))
    cursor = 0
    completed_hosts = set()
    previous_host = None
    for group in groups:
        pieces = group.get("pieces") if isinstance(group, dict) else None
        if (not isinstance(pieces, list) or len(pieces) != 2 or pieces[0] != cursor
                or not isinstance(pieces[1], int) or pieces[1] <= cursor or pieces[1] > len(requested)):
            raise RuntimeError(label + " does not cover the model contiguously" + ("" if dry_run else "; retained"))
        first = requested[cursor]
        if group.get("affinity") != first["affinity"] or group.get("session") != first["session"]:
            raise RuntimeError(label + " has mismatched stage identities" + ("" if dry_run else "; retained"))
        host, mode = group.get("hostId"), group.get("weights")
        if (not isinstance(host, str) or not host.strip() or mode not in ("resident", "stream")
                or not isinstance(group.get("graph"), dict) or group["graph"].get("weights") != mode):
            raise RuntimeError(label + " has invalid host/mode/graph" + ("" if dry_run else "; retained"))
        if weights == "resident" and mode != "resident":
            raise RuntimeError(label + " did not honor --weights" + ("" if dry_run else "; retained"))
        if host != previous_host:
            if host in completed_hosts:
                raise RuntimeError(label + " is not contiguous" + ("" if dry_run else "; retained"))
            if previous_host is not None:
                completed_hosts.add(previous_host)
            previous_host = host
        cursor = pieces[1]
    if cursor != len(requested):
        raise RuntimeError(label + " does not cover the model" + ("" if dry_run else "; retained"))
    return plan_id, groups


def describe_placement(groups, pieces):
    """Human-readable per-device shard sizes from an orchestrator placement."""
    devices = []
    for group in groups:
        stage = merge_pieces(pieces[group["pieces"][0]:group["pieces"][1]])
        layers = stage["layers"]
        devices.append({
            "hostId": group["hostId"], "weights": group["weights"],
            "embed": stage["embed"], "head": stage["head"],
            "layers": [layers[0], layers[-1]] if layers else [], "layerCount": len(layers),
            "shardBytes": group.get("weightBytes"), "deviceBytes": group.get("deviceBytes"),
            "gpuBudgetBytes": group.get("gpuBudgetBytes"), "gpuTotalBytes": group.get("gpuTotalBytes"),
            "unifiedMemory": group.get("unifiedMemory"),
            # Bytes this device still has to download; 0 means its cached shard is reused.
            "downloadBytes": group.get("downloadBytes")})
    return devices
