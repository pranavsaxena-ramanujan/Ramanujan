import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Dict

from .contracts import AdapterGraph, ShardPlan
from .gguf_source import GGUFSourceReader
from .interfaces import SourceReader
from .planner import plan_shards
from .qwen_adapter import Qwen38ArchitectureAdapter, _LAYER_NAME


_SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_CHUNK_SIZE = 4 * 1024 * 1024


def emit_qwen_package(gguf_path: Path, output_dir: Path, shards: int = 4) -> Dict:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("output directory already exists: {0}".format(output_dir))
    reader = GGUFSourceReader(gguf_path)
    adapter = Qwen38ArchitectureAdapter(shards)
    graph = adapter.build_graph(reader)
    plans = plan_shards(graph, shards)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output_dir.name + ".partial-",
                                     dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        shard_entries = []
        for plan in plans:
            shard_dir = staging / plan.shard_id
            manifest = emit_qwen_shard(reader, graph, plan, shard_dir)
            shard_entries.append({
                "shardId": plan.shard_id,
                "manifestPath": plan.shard_id + "/manifest.json",
                "estimatedResidentBytes": manifest["estimatedResidentBytes"],
            })
        package = {
            "schemaVersion": "1.0",
            "packageId": output_dir.name,
            "architectureId": graph.architecture_id,
            "architectureVersion": graph.architecture_version,
            "sourceFormat": "gguf",
            "status": "weights-only",
            "metadata": graph.metadata,
            "shards": shard_entries,
        }
        (staging / "model-manifest.json").write_text(
            json.dumps(package, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output_dir)
    return package


def emit_qwen_shard(source: SourceReader, graph: AdapterGraph, plan: ShardPlan,
                    output_dir: Path) -> Dict:
    by_id = {stage.stage_id: stage for stage in graph.stages}
    if not plan.stage_ids or any(stage_id not in by_id for stage_id in plan.stage_ids):
        raise ValueError("plan contains unknown or empty Qwen stages")
    stages = [by_id[stage_id] for stage_id in plan.stage_ids]
    layer_start = int(stages[0].metadata["layer_start"])
    layer_end = int(stages[-1].metadata["layer_end"])
    if any(int(current.metadata["layer_end"]) != int(following.metadata["layer_start"])
           for current, following in zip(stages, stages[1:])):
        raise ValueError("Qwen plan stages are not contiguous")
    first = stages[0].stage_id == graph.stages[0].stage_id
    last = stages[-1].stage_id == graph.stages[-1].stage_id
    output_dir = Path(output_dir)
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=False)
    tensor_files = {}
    checksums = {}
    for name in sorted(source.tensor_names()):
        match = _LAYER_NAME.fullmatch(name)
        selected = (layer_start <= int(match.group(1)) < layer_end) if match else (
            (first and name in ("token_embd.weight", "rope_freqs.weight")) or
            (last and name in ("output_norm.weight", "output.weight"))
        )
        if not selected:
            continue
        if len(name) > 240 or not _SAFE_NAME.fullmatch(name) or ".." in name:
            raise ValueError("unsafe GGUF tensor name: {0}".format(name))
        descriptor = source.tensor_metadata(name)
        relative_path = "weights/" + name + ".bin"
        destination = output_dir / relative_path
        digest = hashlib.sha256()
        with source.open_tensor(name) as input_stream, destination.open("wb") as output_stream:
            remaining = descriptor["length"]
            while remaining:
                chunk = input_stream.read(min(remaining, _CHUNK_SIZE))
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
        checksums[relative_path] = "sha256:" + digest.hexdigest()
    manifest = {
        "schemaVersion": "1.0",
        "shardId": plan.shard_id,
        "status": "weights-only",
        "layerStart": layer_start,
        "layerEnd": layer_end,
        "layerTypes": [kind for stage in stages for kind in stage.metadata["layer_types"].split(",")],
        "tensorFiles": tensor_files,
        "checksums": checksums,
        "estimatedResidentBytes": sum(item["bytes"] for item in tensor_files.values()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest