import argparse
import json
from pathlib import Path

from .phi3_emitter import emit_phi3_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Phi-3 safetensors to Ramanujan IR shards")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-kernel", required=True, type=Path)
    args = parser.parse_args()
    manifest = emit_phi3_package(
        args.model_dir,
        args.output_dir,
        args.reference_kernel,
    )
    print(json.dumps({
        "output": str(args.output_dir),
        "shards": len(manifest["shards"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()