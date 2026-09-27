import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Optional, Union

from .gguf_source import GGUFSourceReader
from .llm_spec import mixer_for


_LAYER = re.compile(r"^blk\.([0-9]+)\.(.+)$")


def build_ir_plan(package_dir: Path) -> dict:
    package_dir = Path(package_dir)
    package = json.loads((package_dir / "model-manifest.json").read_text(encoding="utf-8"))
    if package.get("sourceFormat") != "gguf" or package.get("status") != "weights-only":
        raise ValueError("expected a GGUF weights-only package")
    if package.get("partial"):
        raise ValueError("IR planning requires all shards")

    stages = []
    bindings = {}
    required = set()
    next_layer = 0
    for shard_index, shard in enumerate(package["shards"]):
        if not re.fullmatch(r"shard-[0-9]+", shard["shardId"]) or (
                shard["manifestPath"] != shard["shardId"] + "/manifest.json"):
            raise ValueError("invalid shard manifest reference")
        manifest = json.loads((package_dir / shard["manifestPath"]).read_text(encoding="utf-8"))
        if manifest["shardId"] != shard["shardId"]:
            raise ValueError("shard id mismatch")
        tensors = manifest["tensorFiles"]
        by_layer = {}
        for name, tensor in tensors.items():
            relative_path = tensor["path"]
            if (not re.fullmatch(r"weights/[A-Za-z0-9_][A-Za-z0-9_.-]*\.bin", relative_path)
                    or ".." in relative_path or name in bindings):
                raise ValueError("invalid or duplicate tensor binding: {0}".format(name))
            bindings[name] = {
                "shardId": shard["shardId"], "path": shard["shardId"] + "/" + relative_path,
                "encoding": tensor["encoding"], "shape": tensor["shape"],
                "bytes": tensor["bytes"], "ggmlType": tensor["ggmlType"],
                "checksum": manifest["checksums"][relative_path],
            }
            match = _LAYER.fullmatch(name)
            if match:
                by_layer.setdefault(int(match.group(1)), set()).add(match.group(2))
        layers = []
        if by_layer:
            if manifest["layerStart"] != next_layer or set(by_layer) != set(
                    range(manifest["layerStart"], manifest["layerEnd"])):
                raise ValueError("non-contiguous layer tensors in {0}".format(shard["shardId"]))
            next_layer = manifest["layerEnd"]
        for index, names in sorted(by_layer.items()):
            layer = {"index": index,
                     "tensors": ["blk.{0}.{1}".format(index, name) for name in sorted(names)]}
            mixer = ("auxiliary" if any(name.startswith("nextn.") for name in names)
                     else mixer_for(names))
            if mixer == "gated_deltanet":
                layer.update(operator="gated_deltanet", state="recurrent_and_conv")
                required.add("llm.gated_deltanet")
            elif mixer == "attention":
                layer.update(operator="causal_attention", state="kv_cache")
                required.add("llm.causal_attention")
            elif mixer == "auxiliary":
                layer.update(operator="auxiliary", state="none")
            else:
                reason = mixer.split(":", 1)[1]
                layer.update(operator="unsupported", state="none", reason=reason)
                required.add("architecture.{0}.{1}".format(
                    package["architectureId"], re.sub(r"[^a-z0-9]+", "_", reason.lower()).strip("_")))
            layers.append(layer)
        if any(tensor["encoding"] != "gguf-f32" for tensor in tensors.values()):
            required.add("gguf.quantized_tensor_decode")
        stages.append({"shardId": shard["shardId"],
                       "dependsOn": [] if shard_index == 0 else [stages[-1]["shardId"]],
                       "input": "token_ids" if shard_index == 0 else "hidden_state",
                       "output": "token_logits" if shard_index == len(package["shards"]) - 1 else "hidden_state",
                       "layers": layers,
                       "globalTensors": sorted(name for name in tensors if not _LAYER.fullmatch(name))})
    if next_layer != int(package["metadata"]["layer_count"]):
        raise ValueError("incomplete layer coverage")
    if len(bindings) != int(package["metadata"]["tensor_count"]):
        raise ValueError("incomplete tensor coverage")
    required.add("gguf.tokenizer_and_sampling")
    return {"schemaVersion": "1.0", "sourceFormat": "gguf", "architectureId": package["architectureId"],
            "sourcePackage": str(package_dir.resolve()), "status": "planning-only",
            "stages": stages, "tensorBindings": bindings, "requiredCapabilities": sorted(required)}


def require_executable(plan: dict) -> None:
    raise RuntimeError(
        "no executable Ramanujan GGUF backend is registered; missing: {0}".format(
            ", ".join(plan["requiredCapabilities"])
        )
    )


def write_ir_plan(package_dir: Path, output_dir: Path, run: bool = False,
                  gguf_path: Optional[Union[str, Path]] = None) -> dict:
    plan = build_ir_plan(package_dir)
    if run:
        require_executable(plan)
    source_metadata = None
    if gguf_path is not None:
        reader = GGUFSourceReader(gguf_path)
        source_metadata = reader.metadata()
        if source_metadata.get("general.architecture") != plan["architectureId"]:
            raise ValueError("GGUF architecture does not match shard package")
        if set(reader.tensor_names()) != set(plan["tensorBindings"]):
            raise ValueError("GGUF tensor names do not match shard package")
        for name, binding in plan["tensorBindings"].items():
            descriptor = reader.tensor_metadata(name)
            if (descriptor["shape"] != binding["shape"] or
                    descriptor["length"] != binding["bytes"] or
                    "gguf-" + descriptor["dtype"].lower() != binding["encoding"]):
                raise ValueError("GGUF tensor descriptor does not match shard: {0}".format(name))
        plan["sourceMetadataPath"] = "gguf-metadata.json"
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("output directory already exists: {0}".format(output_dir))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output_dir.name + ".partial-", dir=output_dir.parent) as directory:
        temporary = Path(directory)
        (temporary / "model-plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n",
                                                  encoding="utf-8")
        if source_metadata is not None:
            (temporary / "gguf-metadata.json").write_text(
                json.dumps(source_metadata, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
            )
        for stage in plan["stages"]:
            stage_dir = temporary / stage["shardId"]
            stage_dir.mkdir()
            stage_bindings = {name: binding for name, binding in plan["tensorBindings"].items()
                              if binding["shardId"] == stage["shardId"]}
            (stage_dir / "plan.json").write_text(json.dumps({"status": "planning-only", "stage": stage,
                                                              "tensorBindings": stage_bindings},
                                                             indent=2, sort_keys=True) + "\n",
                                                 encoding="utf-8")
        temporary.rename(output_dir)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Produce a non-executable IR plan from GGUF weight shards")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gguf", help="Original GGUF path or Range-capable URL for metadata and tensor checks")
    parser.add_argument("--run", action="store_true", help="Require an executable backend (fails if unavailable)")
    args = parser.parse_args()
    try:
        plan = write_ir_plan(args.package, args.output_dir, run=args.run, gguf_path=args.gguf)
    except (RuntimeError, ValueError) as error:
        parser.exit(1, "error: {0}\n".format(error))
    print(json.dumps({"status": plan["status"], "shards": len(plan["stages"]),
                      "tensorBindings": len(plan["tensorBindings"]),
                      "requiredCapabilities": plan["requiredCapabilities"]}, sort_keys=True))


if __name__ == "__main__":
    main()