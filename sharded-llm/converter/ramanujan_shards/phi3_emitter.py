import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .kernel_generator import generate_phi3_decode_kernel, generate_phi3_prefill_kernel
from .phi3_adapter import Phi3ArchitectureAdapter
from .safetensors_source import SafeTensorsSourceReader
from .tensor_writer import write_float32_tensor, write_packed_4bit_matrix


_QUANTIZED_TENSORS = {
    "self_attn.qkv_proj.weight": "qkv",
    "self_attn.o_proj.weight": "o",
    "mlp.gate_up_proj.weight": "gate_up",
    "mlp.down_proj.weight": "down",
}


def emit_phi3_package(model_dir: Path, output_dir: Path, reference_kernel: Path) -> Dict:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("output directory already exists: {0}".format(output_dir))
    reader = SafeTensorsSourceReader(Path(model_dir))
    graph = Phi3ArchitectureAdapter(target_shards=4).build_graph(reader)
    staging = output_dir.with_name(output_dir.name + ".partial")
    if staging.exists():
        shutil.rmtree(str(staging))
    staging.mkdir(parents=True)
    try:
        shard_manifests = []
        for shard_index, stage in enumerate(graph.stages):
            shard_dir = staging / "shard-{0:02d}".format(shard_index)
            weights_dir = shard_dir / "weights"
            programs_dir = shard_dir / "programs"
            weights_dir.mkdir(parents=True)
            programs_dir.mkdir()
            layer_start = int(stage.metadata["layer_start"])
            layer_end = int(stage.metadata["layer_end"])
            tensor_files: Dict[str, Dict] = {}

            if shard_index == 0:
                _write_float(reader, "model.embed_tokens.weight", weights_dir / "embed_tokens.bin",
                             tensor_files)
            for layer in range(layer_start, layer_end):
                _write_layer(reader, layer, weights_dir, tensor_files)
            _write_rope(reader.metadata()["config"], weights_dir, tensor_files)
            if shard_index == len(graph.stages) - 1:
                _write_float(reader, "model.norm.weight", weights_dir / "ln_f_g.bin",
                             tensor_files)
                _write_split_rows(reader, "lm_head.weight", weights_dir, "lm_head", 16000,
                                  tensor_files)

            program_path = programs_dir / "prefill.py"
            program_path.write_text(
                generate_phi3_prefill_kernel(reference_kernel, layer_start, layer_end,
                                             shard_index == len(graph.stages) - 1),
                encoding="utf-8",
            )
            (programs_dir / "decode.py").write_text(
                generate_phi3_decode_kernel(reference_kernel, layer_start, layer_end,
                                            shard_index == len(graph.stages) - 1),
                encoding="utf-8",
            )
            (programs_dir / "decode_resident.py").write_text(
                generate_phi3_decode_kernel(reference_kernel, layer_start, layer_end,
                                            shard_index == len(graph.stages) - 1,
                                            resident_kv=True),
                encoding="utf-8",
            )
            if shard_index == 0:
                ranges = [(int(item.metadata["layer_start"]), int(item.metadata["layer_end"]))
                          for item in graph.stages]
                (programs_dir / "decode_fused.py").write_text(
                    generate_phi3_decode_kernel(reference_kernel, ranges[0][0],
                                                ranges[-1][1], True, resident_kv=True,
                                                shard_ranges=ranges),
                    encoding="utf-8",
                )
            checksums = _checksums(shard_dir)
            state = [
                {
                    "name": "layer-{0:02d}-kv".format(layer),
                    "role": "model-state",
                    "mutable": True,
                    "checkpointed": True,
                    "shape": ["2", "max_sequence", str(graph.metadata["hidden_size"])],
                    "dataType": "float32",
                }
                for layer in range(layer_start, layer_end)
            ]
            manifest = {
                "schemaVersion": "1.0",
                "shardId": "shard-{0:02d}".format(shard_index),
                "irPath": "programs/prefill.py",
                "capabilities": ["inference.prefill", "inference.decode"],
                "entrypoints": [
                    {
                        "name": "prefill",
                        "commandId": "source:prefill",
                        "capability": "inference.prefill",
                        "inputs": ["hidden", "params", "cur_n_seq_arr"],
                        "outputs": ["h_state"] + (["argmax_arr"] if shard_index == len(graph.stages) - 1 else []),
                    },
                    {
                        "name": "decode",
                        "commandId": "source:decode",
                        "capability": "inference.decode",
                        "inputs": ["h_state", "cur_n_seq_arr"] + [
                            "l{0}_{1}_cache".format(layer, kind)
                            for layer in range(layer_start, layer_end)
                            for kind in ("k", "v")
                        ],
                        "outputs": ["h_state"] + (["argmax_arr"] if shard_index == len(graph.stages) - 1 else []),
                    },
                ],
                "states": state,
                "tensorFiles": tensor_files,
                "checksums": checksums,
                "adapterMetadata": dict(stage.metadata),
                "estimatedResidentBytes": stage.estimated_resident_bytes,
            }
            (shard_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            shard_manifests.append(manifest)

        package_manifest = {
            "schemaVersion": "1.0",
            "packageId": "phi3-mini-4k-instruct-rj-q4",
            "architectureId": graph.architecture_id,
            "architectureVersion": graph.architecture_version,
            "sourceFormat": "safetensors",
            "capabilities": ["inference.prefill", "inference.decode"],
            "executionGraph": [
                {
                    "nodeId": stage.stage_id,
                    "shardId": "shard-{0:02d}".format(index),
                    "entrypoint": "prefill",
                    "dependsOn": list(stage.dependencies),
                }
                for index, stage in enumerate(graph.stages)
            ],
            "shards": [
                {
                    "shardId": manifest["shardId"],
                    "manifestPath": manifest["shardId"] + "/manifest.json",
                    "estimatedResidentBytes": manifest["estimatedResidentBytes"],
                }
                for manifest in shard_manifests
            ],
        }
        (staging / "model-manifest.json").write_text(
            json.dumps(package_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(output_dir)
        return package_manifest
    except BaseException:
        shutil.rmtree(str(staging), ignore_errors=True)
        raise


def _write_layer(reader, layer: int, output_dir: Path, files: Dict) -> None:
    prefix = "model.layers.{0}.".format(layer)
    for suffix, stem in _QUANTIZED_TENSORS.items():
        tensor_name = prefix + suffix
        packed_path = output_dir / "l{0}_{1}_packed.bin".format(layer, stem)
        scales_path = output_dir / "l{0}_{1}_scales.bin".format(layer, stem)
        with reader.open_tensor(tensor_name) as source:
            packed_bytes, scale_bytes = write_packed_4bit_matrix(
                source, reader.tensor_metadata(tensor_name), packed_path, scales_path
            )
        files[tensor_name] = {
            "encoding": "rj-row-q4x6-f32",
            "packedPath": "weights/" + packed_path.name,
            "scalesPath": "weights/" + scales_path.name,
            "packedBytes": packed_bytes,
            "scaleBytes": scale_bytes,
            "shape": reader.tensor_metadata(tensor_name)["shape"],
        }
    for suffix, stem in (("input_layernorm.weight", "ln1_g"),
                         ("post_attention_layernorm.weight", "ln2_g")):
        _write_float(reader, prefix + suffix, output_dir / "l{0}_{1}.bin".format(layer, stem),
                     files)


def _write_float(reader, tensor_name: str, output: Path, files: Dict) -> None:
    metadata = reader.tensor_metadata(tensor_name)
    with reader.open_tensor(tensor_name) as source:
        byte_count = write_float32_tensor(source, metadata, output)
    files[tensor_name] = {
        "encoding": "float32-le",
        "path": "weights/" + output.name,
        "bytes": byte_count,
        "shape": metadata["shape"],
    }


def _write_split_rows(reader, tensor_name: str, output_dir: Path, stem: str, split_row: int,
                      files: Dict) -> None:
    metadata = reader.tensor_metadata(tensor_name)
    rows, columns = metadata["shape"]
    if split_row <= 0 or split_row >= rows:
        raise ValueError("split row must be within tensor")
    with reader.open_tensor(tensor_name) as source:
        first_metadata = dict(metadata, shape=[split_row, columns])
        second_metadata = dict(metadata, shape=[rows - split_row, columns])
        first = output_dir / (stem + "_1.bin")
        second = output_dir / (stem + "_2.bin")
        first_bytes = write_float32_tensor(source, first_metadata, first)
        second_bytes = write_float32_tensor(source, second_metadata, second)
    files[tensor_name] = {
        "encoding": "float32-le",
        "paths": ["weights/" + first.name, "weights/" + second.name],
        "bytes": [first_bytes, second_bytes],
        "shape": metadata["shape"],
        "splitRows": [split_row, rows - split_row],
    }


def _write_rope(config: Dict, output_dir: Path, files: Dict) -> None:
    head_dimension = int(config["hidden_size"]) // int(config["num_attention_heads"])
    half_dimension = head_dimension // 2
    max_sequence = int(config["max_position_embeddings"])
    theta = float(config.get("rope_theta", 10000.0))
    frequencies = 1.0 / (theta ** (np.arange(0, head_dimension, 2, dtype=np.float32) / head_dimension))
    positions = np.arange(max_sequence, dtype=np.float32)
    angles = np.outer(positions, frequencies)
    for name, values in (("cos_cache", np.cos(angles)), ("sin_cache", np.sin(angles))):
        path = output_dir / (name + ".bin")
        encoded = values.astype("<f4")
        path.write_bytes(encoded.tobytes())
        files[name] = {
            "encoding": "float32-le",
            "path": "weights/" + path.name,
            "bytes": encoded.nbytes,
            "shape": [max_sequence, half_dimension],
        }
def _checksums(shard_dir: Path) -> Dict[str, str]:
    checksums = {}
    for path in sorted(shard_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            checksums[str(path.relative_to(shard_dir))] = "sha256:" + digest.hexdigest()
    return checksums