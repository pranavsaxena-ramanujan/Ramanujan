import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union

from .contracts import AdapterGraph, ShardPlan
from .gguf_adapter import GGUFArchitectureAdapter
from .gguf_source import GGUFSourceReader
from .interfaces import SourceReader
from .planner import plan_shards


_SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_CHUNK_SIZE = 4 * 1024 * 1024


def emit_gguf_package(gguf_path: Union[str, Path], output_dir: Path, shards: int = 4,
                      shard_index: Optional[int] = None, per_layer: bool = False) -> Dict:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("output directory already exists: {0}".format(output_dir))
    reader = GGUFSourceReader(gguf_path)
    graph = GGUFArchitectureAdapter(shards, per_layer=per_layer).build_graph(reader)
    plans = plan_shards(graph, len(graph.stages) if per_layer else shards)
    if shard_index is not None:
        if shard_index < 0 or shard_index >= len(plans):
            raise ValueError("shard_index must be between 0 and {0}".format(len(plans) - 1))
        selected_plans = [plans[shard_index]]
    else:
        selected_plans = plans
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output_dir.name + ".partial-",
                                     dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        entries = []
        for plan in selected_plans:
            manifest = emit_gguf_shard(reader, graph, plan, staging / plan.shard_id)
            entries.append({
                "shardId": plan.shard_id,
                "manifestPath": plan.shard_id + "/manifest.json",
                "estimatedResidentBytes": manifest["estimatedResidentBytes"],
                "tensorCount": len(manifest["tensorFiles"]),
            })
        package = {
            "schemaVersion": "1.0",
            "packageId": output_dir.name,
            "architectureId": graph.architecture_id,
            "architectureVersion": graph.architecture_version,
            "sourceFormat": "gguf",
            "status": "weights-only",
            "partial": shard_index is not None,
            "totalShards": len(plans),
            "metadata": graph.metadata,
            "shards": entries,
        }
        if per_layer:
            package["artifactLayout"] = "per-layer"
        (staging / "model-manifest.json").write_text(
            json.dumps(package, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output_dir)
    return package


def emit_gguf_shard(source: SourceReader, graph: AdapterGraph, plan: ShardPlan,
                    output_dir: Path) -> Dict:
    by_id = {stage.stage_id: stage for stage in graph.stages}
    indexes = [next((index for index, stage in enumerate(graph.stages) if stage.stage_id == stage_id), -1)
               for stage_id in plan.stage_ids]
    if not indexes or indexes != list(range(indexes[0], indexes[0] + len(indexes))) or indexes[0] < 0:
        raise ValueError("plan contains unknown or non-contiguous GGUF stages")
    stages = [by_id[stage_id] for stage_id in plan.stage_ids]
    names = [name for stage in stages for name in json.loads(stage.metadata["tensor_names"])]
    if len(names) != len(set(names)):
        raise ValueError("plan assigns a GGUF tensor more than once")
    output_dir = Path(output_dir)
    (output_dir / "weights").mkdir(parents=True, exist_ok=False)
    tensor_files = {}
    checksums = {}
    for name in names:
        descriptor = source.tensor_metadata(name)
        file_name = name if len(name) <= 240 and _SAFE_NAME.fullmatch(name) and ".." not in name else (
            hashlib.sha256(name.encode("utf-8")).hexdigest()
        )
        relative_path = "weights/" + file_name + ".bin"
        digest = hashlib.sha256()
        with source.open_tensor(name) as input_stream, (output_dir / relative_path).open("wb") as output_stream:
            remaining = descriptor["length"]
            while remaining:
                chunk = input_stream.read(min(_CHUNK_SIZE, remaining))
                if not chunk:
                    raise ValueError("truncated GGUF tensor during copy: {0}".format(name))
                output_stream.write(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
        tensor_files[name] = {
            "encoding": "gguf-" + descriptor["dtype"].lower(),
            "path": relative_path,
            "bytes": descriptor["length"],
            "shape": descriptor["shape"],
            "ggmlType": descriptor["ggml_type"],
            "sourceOffset": descriptor["offset"],
        }
        if descriptor.get("length_includes_padding"):
            tensor_files[name]["lengthIncludesPadding"] = True
        checksums[relative_path] = "sha256:" + digest.hexdigest()
    manifest = {
        "schemaVersion": "1.0",
        "shardId": plan.shard_id,
        "status": "weights-only",
        "stageIds": list(plan.stage_ids),
        "tensorFiles": tensor_files,
        "checksums": checksums,
        "estimatedResidentBytes": sum(item["bytes"] for item in tensor_files.values()),
    }
    if "layer_start" in stages[0].metadata:
        manifest["layerStart"] = int(stages[0].metadata["layer_start"])
        manifest["layerEnd"] = int(stages[-1].metadata["layer_end"])
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest