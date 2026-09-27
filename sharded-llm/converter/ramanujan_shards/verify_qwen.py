import argparse
import hashlib
import json
import re
from pathlib import Path


def verify_qwen_package(package_dir: Path) -> int:
    package_dir = Path(package_dir)
    package = json.loads((package_dir / "model-manifest.json").read_text(encoding="utf-8"))
    if package.get("architectureId") != "qwen35" or package.get("status") != "weights-only":
        raise ValueError("not a Qwen weights-only package")
    layer_count = int(package["metadata"]["layer_count"])
    next_layer = 0
    seen_tensors = set()
    seen_shards = set()
    for entry in package["shards"]:
        shard_id = entry["shardId"]
        if not re.fullmatch(r"shard-[0-9]+", shard_id) or shard_id in seen_shards:
            raise ValueError("invalid or duplicate shard id: {0}".format(shard_id))
        seen_shards.add(shard_id)
        if entry["manifestPath"] != shard_id + "/manifest.json":
            raise ValueError("invalid shard manifest path: {0}".format(shard_id))
        shard_dir = package_dir / shard_id
        manifest = json.loads((shard_dir / "manifest.json").read_text(encoding="utf-8"))
        start, end = manifest["layerStart"], manifest["layerEnd"]
        if (manifest["shardId"] != shard_id or start != next_layer or end <= start
                or end > layer_count or len(manifest["layerTypes"]) != end - start):
            raise ValueError("invalid layer coverage: {0}".format(shard_id))
        next_layer = end
        descriptors = manifest["tensorFiles"]
        checksums = manifest["checksums"]
        if set(checksums) != {descriptor["path"] for descriptor in descriptors.values()}:
            raise ValueError("checksum list does not match tensors: {0}".format(shard_id))
        total_bytes = 0
        for name, descriptor in descriptors.items():
            if name in seen_tensors:
                raise ValueError("duplicate tensor: {0}".format(name))
            seen_tensors.add(name)
            match = re.fullmatch(r"blk\.([0-9]+)\..+", name)
            if match and not start <= int(match.group(1)) < end:
                raise ValueError("tensor in wrong shard: {0}".format(name))
            relative_path = descriptor["path"]
            if relative_path != "weights/" + name + ".bin":
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
    if next_layer != layer_count or not {"token_embd.weight", "output.weight", "output_norm.weight"}.issubset(seen_tensors):
        raise ValueError("incomplete Qwen layer or global tensor coverage")
    return len(seen_tensors)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a Qwen GGUF weights-only shard package")
    parser.add_argument("--package", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps({"tensorsVerified": verify_qwen_package(args.package)}, sort_keys=True))


if __name__ == "__main__":
    main()