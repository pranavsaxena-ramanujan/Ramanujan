import json
import re
from pathlib import Path

from .contracts import AdapterGraph, ShardPlan, StageDefinition
from .interfaces import ArchitectureAdapter, SourceReader


_BLOCK_TENSOR = re.compile(r"^blk\.([0-9]+)\..+$")


class GGUFArchitectureAdapter(ArchitectureAdapter):
    def __init__(self, target_shards: int = 4):
        if target_shards <= 0:
            raise ValueError("target_shards must be positive")
        self._target_shards = target_shards

    def build_graph(self, source: SourceReader) -> AdapterGraph:
        metadata = source.metadata()
        architecture = metadata.get("general.architecture", "unknown")
        names = sorted(source.tensor_names())
        if not names:
            raise ValueError("GGUF contains no tensors")
        blocks = {}
        globals_ = []
        for name in names:
            match = _BLOCK_TENSOR.fullmatch(name)
            if match:
                blocks.setdefault(int(match.group(1)), []).append(name)
            else:
                globals_.append(name)
        if blocks:
            count = metadata.get(str(architecture) + ".block_count", max(blocks) + 1)
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise ValueError("invalid GGUF block count")
            if set(blocks) != set(range(count)):
                raise ValueError("GGUF layer tensor coverage does not match block count")
            group_count = min(self._target_shards, count)
            groups = []
            for index in range(group_count):
                start = index * count // group_count
                end = (index + 1) * count // group_count
                group_names = [name for layer in range(start, end) for name in blocks[layer]]
                if index == 0:
                    group_names += [name for name in globals_ if not name.startswith("output")]
                if index == group_count - 1:
                    group_names += [name for name in globals_ if name.startswith("output")]
                groups.append((group_names, {"layer_start": str(start), "layer_end": str(end)}))
        else:
            group_count = min(self._target_shards, len(names))
            groups = [
                (names[index * len(names) // group_count:(index + 1) * len(names) // group_count], {})
                for index in range(group_count)
            ]
        stages = []
        for index, (group_names, group_metadata) in enumerate(groups):
            stage_id = "tensor-group-{0:02d}".format(index)
            stages.append(StageDefinition(
                stage_id=stage_id,
                estimated_resident_bytes=sum(source.tensor_metadata(name)["length"] for name in group_names),
                dependencies=[] if not stages else [stages[-1].stage_id],
                metadata=dict(group_metadata, tensor_names=json.dumps(group_names)),
            ))
        return AdapterGraph(
            architecture_id=str(architecture),
            architecture_version=str(metadata.get("general.name", "unknown")),
            stages=stages,
            metadata={"tensor_count": str(len(names)), "layer_count": str(len(blocks))},
        )

    def emit_shard(self, source: SourceReader, plan: ShardPlan, output_dir: Path) -> None:
        from .gguf_emitter import emit_gguf_shard

        emit_gguf_shard(source, self.build_graph(source), plan, output_dir)