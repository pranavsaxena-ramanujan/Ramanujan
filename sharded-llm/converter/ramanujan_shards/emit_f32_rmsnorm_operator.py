import argparse
import json
import tempfile
from pathlib import Path

from .gguf_f32_norm import generate_f32_rmsnorm_program
from .gguf_ir_plan import build_ir_plan


def emit_f32_rmsnorm_operator(package_dir: Path, tensor_name: str, output_dir: Path,
                              epsilon: float) -> dict:
    plan = build_ir_plan(package_dir)
    binding = plan["tensorBindings"].get(tensor_name)
    if binding is None:
        raise ValueError("tensor not found in package: {0}".format(tensor_name))
    if binding["encoding"] != "gguf-f32" or len(binding["shape"]) != 1:
        raise ValueError("RMSNorm requires a rank-1 F32 tensor")
    dimension = binding["shape"][0]
    if binding["bytes"] != dimension * 4:
        raise ValueError("F32 tensor byte length does not match its shape")
    program = generate_f32_rmsnorm_program(dimension, epsilon)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("output directory already exists: {0}".format(output_dir))
    metadata = {
        "status": "operator-probe", "tensor": tensor_name,
        "shardId": binding["shardId"],
        "weightPath": str(Path(plan["sourcePackage"]) / binding["path"]),
        "weightChecksum": binding["checksum"],
        "shape": [dimension], "epsilon": epsilon,
        "programPath": "f32_rmsnorm.py",
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output_dir.name + ".partial-", dir=output_dir.parent) as directory:
        temporary = Path(directory)
        (temporary / metadata["programPath"]).write_text(program, encoding="utf-8")
        (temporary / "binding.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                                                encoding="utf-8")
        temporary.rename(output_dir)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit an F32 RMSNorm operator from GGUF shards")
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--epsilon", required=True, type=float)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        metadata = emit_f32_rmsnorm_operator(args.package, args.tensor, args.output_dir, args.epsilon)
    except ValueError as error:
        parser.exit(1, "error: {0}\n".format(error))
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()