import argparse
import json
from pathlib import Path

from .qwen_emitter import emit_qwen_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream Qwen 3.8 GGUF into weights-only layer shards")
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()
    manifest = emit_qwen_package(args.gguf, args.output_dir, args.shards)
    print(json.dumps({"output": str(args.output_dir), "shards": len(manifest["shards"]),
                      "status": manifest["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()