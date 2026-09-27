import argparse
import json
from pathlib import Path

from .emit_q4_1_operator import emit_f32_operator


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit an F32 Ramanujan matvec operator from GGUF shards")
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        metadata = emit_f32_operator(args.package, args.tensor, args.output_dir)
    except ValueError as error:
        parser.exit(1, "error: {0}\n".format(error))
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()