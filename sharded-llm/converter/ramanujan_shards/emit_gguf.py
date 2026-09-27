import argparse
import json
from pathlib import Path

from .gguf_emitter import emit_gguf_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Shard any GGUF into raw, checksum-verified tensor packages")
    parser.add_argument("--gguf", required=True, help="Local path or HTTP(S) URL")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--shard-index", type=int, help="Only fetch and store this zero-based shard")
    args = parser.parse_args()
    package = emit_gguf_package(args.gguf, args.output_dir, args.shards, args.shard_index)
    print(json.dumps({"output": str(args.output_dir), "architecture": package["architectureId"],
                      "shards": len(package["shards"]), "status": package["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()