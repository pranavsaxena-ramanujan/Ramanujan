import argparse
import hashlib
import json
import re
from pathlib import Path


def verify_gguf_package(package_dir: Path) -> int:
    package_dir = Path(package_dir)
    package = json.loads((package_dir / "model-manifest.json").read_text(encoding="utf-8"))
    if package.get("sourceFormat") != "gguf" or package.get("status") != "weights-only":
        raise ValueError("not a GGUF weights-only package")
    tensor_count = int(package["metadata"]["tensor_count"])
    layer_count = int(package["metadata"]["layer_count"])
    partial = package.get("partial", False)
    if tensor_count <= 0 or layer_count < 0 or not package["shards"]:
        raise ValueError("invalid GGUF package metadata")
    if partial and len(package["shards"]) != 1:
        raise ValueError("partial package must contain exactly one shard")
    seen_tensors = set()
    seen_shards = set()
    next_layer = 0
    for entry in package["shards"]:
        shard_id = entry["shardId"]
        if not re.fullmatch(r"shard-[0-9]+", shard_id) or shard_id in seen_shards:
            raise ValueError("invalid or duplicate shard id: {0}".format(shard_id))
        seen_shards.add(shard_id)
        if entry["manifestPath"] != shard_id + "/manifest.json":
            raise ValueError("invalid shard manifest path: {0}".format(shard_id))
        shard_dir = package_dir / shard_id
        manifest = json.loads((shard_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["shardId"] != shard_id or manifest.get("status") != "weights-only":
            raise ValueError("invalid shard manifest: {0}".format(shard_id))
        if layer_count:
            start, end = manifest["layerStart"], manifest["layerEnd"]
            if (not partial and start != next_layer) or end <= start or start < 0 or end > layer_count:
                raise ValueError("invalid layer coverage: {0}".format(shard_id))
            next_layer = end
        elif "layerStart" in manifest or "layerEnd" in manifest:
            raise ValueError("unexpected layer range: {0}".format(shard_id))
        descriptors = manifest["tensorFiles"]
        checksums = manifest["checksums"]
        if entry.get("tensorCount") != len(descriptors):
            raise ValueError("shard tensor count mismatch: {0}".format(shard_id))
        if set(checksums) != {item["path"] for item in descriptors.values()} or len(checksums) != len(descriptors):
            raise ValueError("checksum list does not match tensors: {0}".format(shard_id))
        total_bytes = 0
        for name, descriptor in descriptors.items():
            if name in seen_tensors:
                raise ValueError("duplicate tensor: {0}".format(name))
            seen_tensors.add(name)
            match = re.fullmatch(r"blk\.([0-9]+)\..+", name)
            if layer_count and match and not start <= int(match.group(1)) < end:
                raise ValueError("tensor in wrong shard: {0}".format(name))
            relative_path = descriptor["path"]
            if not re.fullmatch(r"weights/[A-Za-z0-9_][A-Za-z0-9_.-]*\.bin", relative_path):
                raise ValueError("invalid tensor path: {0}".format(name))
            tensor_path = shard_dir / relative_path
            if not tensor_path.resolve().is_relative_to(shard_dir.resolve()):
                raise ValueError("tensor path escapes shard: {0}".format(name))
            digest = hashlib.sha256()
            byte_count = 0
            with tensor_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
                    byte_count += len(chunk)
            if byte_count != descriptor["bytes"] or checksums[relative_path] != "sha256:" + digest.hexdigest():
                raise ValueError("tensor checksum or size mismatch: {0}".format(name))
            total_bytes += byte_count
        if total_bytes != entry["estimatedResidentBytes"] or total_bytes != manifest["estimatedResidentBytes"]:
            raise ValueError("shard byte estimate mismatch: {0}".format(shard_id))
    if not seen_tensors or (not partial and (len(seen_tensors) != tensor_count or next_layer != layer_count)):
        raise ValueError("incomplete GGUF tensor or layer coverage")
    return len(seen_tensors)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a GGUF tensor shard package")
    parser.add_argument("--package", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps({"tensorsVerified": verify_gguf_package(args.package)}, sort_keys=True))


if __name__ == "__main__":
    main()