from pathlib import Path
from typing import Any, Dict, Iterable

from .contracts import AdapterGraph, ShardPlan, StageDefinition
from .interfaces import ArchitectureAdapter, SourceReader


class Phi3ArchitectureAdapter(ArchitectureAdapter):
    _LAYER_TENSORS = (
        "self_attn.qkv_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    )

    def __init__(self, target_shards: int = 4):
        if target_shards <= 0:
            raise ValueError("target_shards must be positive")
        self._target_shards = target_shards

    def build_graph(self, source: SourceReader) -> AdapterGraph:
        config = source.metadata().get("config", {})
        if config.get("model_type") != "phi3":
            raise ValueError("Phi3ArchitectureAdapter requires model_type=phi3")
        layer_count = _positive_int(config, "num_hidden_layers")

        stages = []
        previous = None
        group_count = min(self._target_shards, layer_count)
        for group_index in range(group_count):
            layer_start = group_index * layer_count // group_count
            layer_end = (group_index + 1) * layer_count // group_count
            resident_bytes = 0
            for layer_index in range(layer_start, layer_end):
                prefix = "model.layers.{0}.".format(layer_index)
                resident_bytes += sum(
                    _runtime_tensor_bytes(source, prefix + suffix)
                    for suffix in self._LAYER_TENSORS
                )
            roles = ["decoder-blocks"]
            entrypoints = ["prefill", "decode"]
            if group_index == 0:
                resident_bytes += _float32_bytes(source, "model.embed_tokens.weight")
                roles.append("token-input")
            if group_index == group_count - 1:
                resident_bytes += _float32_bytes(source, "model.norm.weight")
                resident_bytes += _float32_bytes(source, "lm_head.weight")
                roles.append("logit-output")
                entrypoints.append("sample")
            stage_id = "decoder-group-{0:02d}".format(group_index)
            stages.append(
                StageDefinition(
                    stage_id=stage_id,
                    estimated_resident_bytes=resident_bytes,
                    dependencies=[] if previous is None else [previous],
                    entrypoints=entrypoints,
                    metadata={
                        "roles": ",".join(roles),
                        "layer_start": str(layer_start),
                        "layer_end": str(layer_end),
                    },
                )
            )
            previous = stage_id
        return AdapterGraph(
            architecture_id="phi3",
            architecture_version=str(config.get("_name_or_path", "unknown")),
            stages=stages,
            metadata={
                "hidden_size": str(_positive_int(config, "hidden_size")),
                "vocab_size": str(_positive_int(config, "vocab_size")),
            },
        )

    def emit_shard(self, source: SourceReader, plan: ShardPlan, output_dir: Path) -> None:
        raise NotImplementedError("Phi-3 shard emission will be added after IR template extraction")


def _positive_int(config: Dict[str, Any], name: str) -> int:
    value = config.get(name)
    if not isinstance(value, int) or value <= 0:
        raise ValueError("Phi-3 config requires positive integer {0}".format(name))
    return value


def _float32_bytes(source: SourceReader, name: str) -> int:
    return _element_count(source.tensor_metadata(name).get("shape", [])) * 4


def _runtime_tensor_bytes(source: SourceReader, name: str) -> int:
    metadata = source.tensor_metadata(name)
    shape = metadata.get("shape", [])
    if name.endswith("layernorm.weight"):
        return _element_count(shape) * 4
    if len(shape) != 2:
        raise ValueError("quantized Phi-3 weight must be rank 2: {0}".format(name))
    output_features, input_features = shape
    packed_values = output_features * ((input_features + 5) // 6)
    scales = output_features
    return (packed_values + scales) * 4


def _element_count(shape: Iterable[int]) -> int:
    count = 1
    found = False
    for dimension in shape:
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("tensor dimensions must be positive integers")
        count *= dimension
        found = True
    if not found:
        raise ValueError("tensor shape is required")
    return count