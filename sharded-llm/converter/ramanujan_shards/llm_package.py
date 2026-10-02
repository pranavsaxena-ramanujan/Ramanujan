"""Load a Ramanujan GGUF shard package for any architecture."""
import json
from pathlib import Path

from .llm_spec import build_spec


def load_package(package_dir):
    """Returns (package manifest, shards); every tensor entry carries an absolute ``file``."""
    package_dir = Path(package_dir)
    package = json.loads((package_dir / "model-manifest.json").read_text(encoding="utf-8"))
    if package.get("sourceFormat") != "gguf":
        raise ValueError("expected a GGUF shard package")
    if package.get("partial"):
        raise ValueError("inference requires every shard")
    shards = []
    for entry in package["shards"]:
        manifest = json.loads((package_dir / entry["manifestPath"]).read_text(encoding="utf-8"))
        root = (package_dir / entry["manifestPath"]).parent.resolve()
        tensors = {}
        for name, tensor in manifest["tensorFiles"].items():
            path = (root / tensor["path"]).resolve()
            if root not in path.parents:
                raise ValueError("tensor path escapes shard: {0}".format(name))
            tensors[name] = dict(tensor, file=path, shard=len(shards))
        shards.append({"id": entry["shardId"], "root": root, "tensors": tensors,
                       "layerStart": manifest.get("layerStart"), "layerEnd": manifest.get("layerEnd")})
    return package, shards


def all_tensors(shards):
    merged = {}
    for shard in shards:
        for name, tensor in shard["tensors"].items():
            if name in merged:
                raise ValueError("tensor appears in two shards: {0}".format(name))
            merged[name] = tensor
    return merged


def add_model_arguments(parser):
    """CLI flags for facts GGUF does not store (shared by the runner and the llama.cpp check)."""
    group = parser.add_argument_group(
        "architecture semantics", "facts GGUF does not store; needed only for architectures outside "
        "the llm_spec.ARCHITECTURES registry, or to override it")
    group.add_argument("--rope-style", choices=["norm", "neox"],
                       help="norm rotates adjacent pairs (llama); neox rotates halves (qwen, phi3, gemma)")
    group.add_argument("--activation", choices=["silu", "gelu"])
    group.add_argument("--embed-scale", choices=["none", "sqrt_dim"])
    group.add_argument("--accept-metadata", action="append", default=[], metavar="KEY",
                       help="accept an unrecognized '<arch>.*' GGUF metadata key after verifying with "
                            "compare_llama_cpp.py that it does not change the output (repeatable)")


def load_model_from_args(args):
    overrides = {"rope_style": args.rope_style, "activation": args.activation, "embed_scale": args.embed_scale}
    return load_model(args.package, args.metadata, overrides, args.accept_metadata)


def load_model(package_dir, metadata_path, overrides=None, accept_metadata=()):
    """Returns (spec, shards, merged tensors, metadata) for a package and its GGUF metadata JSON."""
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    package, shards = load_package(package_dir)
    architecture = package.get("architectureId")
    if architecture and metadata.get("general.architecture") != architecture:
        raise ValueError("metadata architecture {0!r} does not match the package ({1!r})".format(
            metadata.get("general.architecture"), architecture))
    tensors = all_tensors(shards)
    return build_spec(metadata, tensors, overrides, accept_metadata), shards, tensors, metadata
