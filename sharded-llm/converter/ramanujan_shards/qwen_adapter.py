import re
from pathlib import Path
from typing import Dict, List

from .contracts import AdapterGraph, ShardPlan, StageDefinition
from .interfaces import ArchitectureAdapter, SourceReader


_LAYER_NAME = re.compile(r"^blk\.([0-9]+)\..+$")
_GLOBAL_TENSORS = {"token_embd.weight", "output_norm.weight", "output.weight", "rope_freqs.weight"}


class Qwen38ArchitectureAdapter(ArchitectureAdapter):
    def __init__(self, target_shards: int = 4):
        if target_shards <= 0:
            raise ValueError("target_shards must be positive")
        self._target_shards = target_shards

    def build_graph(self, source: SourceReader) -> AdapterGraph:
        metadata = source.metadata()
        architecture = metadata.get("general.architecture")
        if architecture != "qwen35":
            raise ValueError("Qwen 3.8 GGUF requires general.architecture=qwen35")
        layer_count = _positive_int(metadata, "qwen35.block_count")
        hidden_size = _positive_int(metadata, "qwen35.embedding_length")
        interval = metadata.get("qwen35.full_attention_interval", 4)
        if not isinstance(interval, int) or interval <= 0:
            raise ValueError("invalid full attention interval")
        names = set(source.tensor_names())
        required = {"token_embd.weight", "output_norm.weight", "output.weight"}
        if not required.issubset(names):
            raise ValueError("missing Qwen global tensors: {0}".format(sorted(required - names)))
        layer_tensors: Dict[int, List[str]] = {index: [] for index in range(layer_count)}
        for name in names:
            match = _LAYER_NAME.fullmatch(name)
            if match:
                layer_index = int(match.group(1))
                if layer_index >= layer_count:
                    raise ValueError("tensor outside declared Qwen layer range: {0}".format(name))
                layer_tensors[layer_index].append(name)
            elif name not in _GLOBAL_TENSORS:
                raise ValueError("unassigned GGUF tensor: {0}".format(name))
        for layer_index, tensors in layer_tensors.items():
            if not tensors:
                raise ValueError("missing Qwen layer: {0}".format(layer_index))
        auxiliary_blocks = {layer for layer, tensors in layer_tensors.items()
                            if any(".nextn." in name for name in tensors)}

        group_count = min(self._target_shards, layer_count)
        stages = []
        for group_index in range(group_count):
            layer_start = group_index * layer_count // group_count
            layer_end = (group_index + 1) * layer_count // group_count
            selected = [name for layer in range(layer_start, layer_end)
                        for name in layer_tensors[layer]]
            if group_index == 0:
                selected += [name for name in ("token_embd.weight", "rope_freqs.weight") if name in names]
            if group_index == group_count - 1:
                selected += ["output_norm.weight", "output.weight"]
            stage_id = "decoder-group-{0:02d}".format(group_index)
            layer_types = ["auxiliary" if layer in auxiliary_blocks else
                           "full_attention" if (layer + 1) % interval == 0 else "linear_attention"
                           for layer in range(layer_start, layer_end)]
            stages.append(StageDefinition(
                stage_id=stage_id,
                estimated_resident_bytes=sum(source.tensor_metadata(name)["length"] for name in selected),
                dependencies=[] if not stages else [stages[-1].stage_id],
                metadata={
                    "layer_start": str(layer_start),
                    "layer_end": str(layer_end),
                    "layer_types": ",".join(layer_types),
                },
            ))
        return AdapterGraph(
            architecture_id="qwen35",
            architecture_version=str(metadata.get("general.name", "Qwen3.8")),
            stages=stages,
            metadata={"hidden_size": str(hidden_size), "layer_count": str(layer_count),
                      "decoder_layer_count": str(layer_count - len(auxiliary_blocks))},
        )

    def emit_shard(self, source: SourceReader, plan: ShardPlan, output_dir: Path) -> None:
        from .qwen_emitter import emit_qwen_shard

        emit_qwen_shard(source, self.build_graph(source), plan, output_dir)


def _positive_int(metadata: Dict, name: str) -> int:
    value = metadata.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("Qwen GGUF requires positive integer {0}".format(name))
    return value