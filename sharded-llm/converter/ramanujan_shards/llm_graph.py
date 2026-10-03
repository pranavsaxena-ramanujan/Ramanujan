"""Stage graphs for the native LLM runtime (ramanujan-native/native/llm, libramanujan_llm).

A model is split into consecutive pieces (the token embedding, each decoder layer, the output
head). Consecutive pieces whose weights live on the same shard form one pipeline stage; the
native runtime executes a stage from a self-contained JSON graph that names the tensor files,
their raw GGUF encodings and every per-layer structural flag from the ModelSpec.
"""
import json

GRAPH_FORMAT = "ramanujan-llm-graph/1"


def _tensor(tensors, name):
    tensor = tensors[name]
    return {"file": str(tensor["file"]), "encoding": tensor["encoding"], "shape": list(tensor["shape"])}


def _shard(tensors, name):
    return tensors[name].get("shard", 0)


def _pieces(spec, tensors, last_layer=None):
    last = len(spec.layers) - 1 if last_layer is None else min(last_layer, len(spec.layers) - 1)
    pieces = [("embed", None, _shard(tensors, spec.embed_tensor))]
    for layer in spec.layers[:last + 1]:
        owners = {_shard(tensors, name) for name in layer.roles.values()}
        if len(owners) != 1:
            raise ValueError("layer {0} is split across shards".format(layer.index))
        pieces.append(("layer", layer.index, owners.pop()))
    if last_layer is None:
        # The head runs where the (large) output matrix lives; the norm is tiny.
        pieces.append(("head", None, _shard(tensors, spec.head_roles["output"])))
    return pieces


def plan_stages(spec, tensors, last_layer=None, split=False):
    """Ordered stages: {"shard", "embed", "layers", "head"}. ``split`` gives every piece its own
    stage (used to compare each layer's output with the reference)."""
    stages = []
    for kind, index, shard in _pieces(spec, tensors, last_layer):
        if split or not stages or stages[-1]["shard"] != shard:
            stages.append({"shard": shard, "embed": False, "layers": [], "head": False})
        stage = stages[-1]
        if kind == "embed":
            stage["embed"] = True
        elif kind == "layer":
            stage["layers"].append(index)
        else:
            stage["head"] = True
    return stages


def hyper(spec):
    return {"architecture": spec.architecture, "dim": spec.dim, "vocab": spec.vocab, "eps": spec.eps,
            "heads": spec.heads, "kv_heads": spec.kv_heads, "head_dim": spec.head_dim,
            "rope_dims": spec.rope_dims, "rope_theta": spec.rope_theta,
            "rope_position_scale": spec.rope_position_scale, "rope_style": spec.semantics.rope_style,
            "activation": spec.semantics.activation, "attn_scale": spec.attn_scale,
            "embed_multiplier": spec.embed_multiplier, "ssm": spec.ssm}


def stage_graph(spec, tensors, stage, max_context, weights="auto", stream_depth=2, stream_threads=2):
    """JSON-ready graph for one stage from plan_stages."""
    if spec.sliding_window and max_context > spec.sliding_window:
        # The runtime attends over the whole context, which is exact only within the window.
        raise ValueError("context {0} exceeds the sliding window {1}".format(max_context, spec.sliding_window))
    graph = {"format": GRAPH_FORMAT, "hyper": hyper(spec), "max_context": int(max_context),
             "weights": weights, "stream_depth": int(stream_depth), "stream_threads": int(stream_threads),
             "layers": []}
    if stage["embed"]:
        graph["embed"] = _tensor(tensors, spec.embed_tensor)
    if stage["layers"] and spec.rope_freqs:
        graph["rope_freqs"] = _tensor(tensors, spec.rope_freqs)
    for index in stage["layers"]:
        layer = spec.layers[index]
        kind = spec.kinds[layer.kind]
        graph["layers"].append({
            "index": index, "mixer": kind.mixer, "flags": dict(kind.flags),
            "tensors": {role: _tensor(tensors, name) for role, name in sorted(layer.roles.items())}})
    if stage["head"]:
        graph["head"] = {role: _tensor(tensors, name) for role, name in spec.head_roles.items()}
    return graph


def graph_files(graph):
    """Every weight file a stage graph reads."""
    files = []
    for key in ("embed", "rope_freqs"):
        if key in graph:
            files.append(graph[key]["file"])
    for layer in graph["layers"]:
        files.extend(tensor["file"] for tensor in layer["tensors"].values())
    for tensor in graph.get("head", {}).values():
        files.append(tensor["file"])
    return list(dict.fromkeys(files))


def dumps(graph):
    return json.dumps(graph, sort_keys=True, separators=(",", ":"))
