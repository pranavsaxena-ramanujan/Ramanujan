import argparse
import hashlib
import json
from pathlib import Path

from .kernel_generator import generate_phi3_decode_kernel, generate_phi3_prefill_kernel


def refresh_phi3_programs(package_dir: Path, reference_kernel: Path):
    package_dir = Path(package_dir)
    package_manifest_path = package_dir / "model-manifest.json"
    package_manifest = _read_json(package_manifest_path)
    capabilities = package_manifest.setdefault("capabilities", [])
    if "inference.decode" not in capabilities:
        capabilities.append("inference.decode")

    refreshed = []
    shards = package_manifest["shards"]
    for shard_index, shard_summary in enumerate(shards):
        manifest_path = package_dir / shard_summary["manifestPath"]
        shard_dir = manifest_path.parent
        manifest = _read_json(manifest_path)
        metadata = manifest["adapterMetadata"]
        layer_start = int(metadata["layer_start"])
        layer_end = int(metadata["layer_end"])
        include_output = shard_index == len(shards) - 1

        programs = {
            "programs/prefill.py": generate_phi3_prefill_kernel(
                reference_kernel, layer_start, layer_end, include_output
            ),
            "programs/decode.py": generate_phi3_decode_kernel(
                reference_kernel, layer_start, layer_end, include_output
            ),
            "programs/decode_resident.py": generate_phi3_decode_kernel(
                reference_kernel, layer_start, layer_end, include_output, resident_kv=True
            ),
        }
        for relative_path, source in programs.items():
            path = shard_dir / relative_path
            path.write_text(source, encoding="utf-8")
            digest = "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()
            manifest.setdefault("checksums", {})[relative_path] = digest
            refreshed.append({"path": str(path), "bytes": len(source.encode("utf-8")), "sha256": digest})

        shard_capabilities = manifest.setdefault("capabilities", [])
        if "inference.decode" not in shard_capabilities:
            shard_capabilities.append("inference.decode")
        manifest["entrypoints"] = [
            entrypoint for entrypoint in manifest.get("entrypoints", [])
            if entrypoint.get("name") != "decode"
        ]
        cache_names = [
            "l{0}_{1}_cache".format(layer, kind)
            for layer in range(layer_start, layer_end)
            for kind in ("k", "v")
        ]
        manifest["entrypoints"].append({
            "name": "decode",
            "commandId": "source:decode",
            "capability": "inference.decode",
            "inputs": ["h_state", "cur_n_seq_arr"] + cache_names,
            "outputs": ["h_state"] + (["argmax_arr"] if include_output else []),
        })
        _write_json(manifest_path, manifest)

    _write_json(package_manifest_path, package_manifest)
    return refreshed


def _read_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Refresh programs in an existing Phi-3 shard package")
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--reference-kernel", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(refresh_phi3_programs(args.package_dir, args.reference_kernel), sort_keys=True))


if __name__ == "__main__":
    main()