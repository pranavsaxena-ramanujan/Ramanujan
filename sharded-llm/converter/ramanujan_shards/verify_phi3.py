import argparse
import ast
import hashlib
import json
from pathlib import Path


def verify_phi3_package(package_dir: Path):
    package_dir = Path(package_dir)
    package_manifest = _read_json(package_dir / "model-manifest.json")
    if package_manifest.get("architectureId") != "phi3":
        raise ValueError("expected architectureId=phi3")
    shards = package_manifest.get("shards", [])
    if len(shards) != 4:
        raise ValueError("expected four Phi-3 shards, found {0}".format(len(shards)))

    expected_start = 0
    checked_files = 0
    total_bytes = 0
    for shard_summary in shards:
        manifest_path = package_dir / shard_summary["manifestPath"]
        shard_dir = manifest_path.parent
        manifest = _read_json(manifest_path)
        metadata = manifest.get("adapterMetadata", {})
        layer_start = int(metadata["layer_start"])
        layer_end = int(metadata["layer_end"])
        if layer_start != expected_start or layer_end <= layer_start:
            raise ValueError("non-contiguous layer range in {0}".format(manifest_path))
        expected_start = layer_end

        program_path = shard_dir / manifest["irPath"]
        ast.parse(program_path.read_text(encoding="utf-8"))
        for relative_path, expected in manifest.get("checksums", {}).items():
            artifact = shard_dir / relative_path
            actual = _sha256(artifact)
            if actual != expected:
                raise ValueError("checksum mismatch: {0}".format(artifact))
            checked_files += 1
            total_bytes += artifact.stat().st_size

    if expected_start != 32:
        raise ValueError("Phi-3 layer coverage ended at {0}, expected 32".format(expected_start))
    return {
        "shards": len(shards),
        "checkedFiles": checked_files,
        "checkedBytes": total_bytes,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a generated Phi-3 Ramanujan shard package")
    parser.add_argument("--package-dir", required=True, type=Path)
    args = parser.parse_args()
    result = verify_phi3_package(args.package_dir)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()