import argparse
import json
import tempfile
from pathlib import Path

from .gguf_ir_plan import build_ir_plan
from .gguf_f32_matvec import generate_f32_matvec_program
from .gguf_q4_1 import generate_q4_1_matvec_program
from .gguf_q5_k import generate_q5_k_matvec_program
from .gguf_q6_k import generate_q6_k_matvec_program


def emit_q4_1_operator(package_dir: Path, tensor_name: str, output_dir: Path) -> dict:
    return _emit_operator(package_dir, tensor_name, output_dir, "q4_1", 32, 20,
                          generate_q4_1_matvec_program)


def emit_q5_k_operator(package_dir: Path, tensor_name: str, output_dir: Path) -> dict:
    return _emit_operator(package_dir, tensor_name, output_dir, "q5_k", 256, 176,
                          generate_q5_k_matvec_program)


def emit_q6_k_operator(package_dir: Path, tensor_name: str, output_dir: Path) -> dict:
    return _emit_operator(package_dir, tensor_name, output_dir, "q6_k", 256, 210,
                          generate_q6_k_matvec_program)


def emit_f32_operator(package_dir: Path, tensor_name: str, output_dir: Path) -> dict:
    return _emit_operator(package_dir, tensor_name, output_dir, "f32", 1, 4,
                          generate_f32_matvec_program)


def _emit_operator(package_dir, tensor_name, output_dir, encoding, block_size, block_bytes,
                   generate_program):
    plan = build_ir_plan(package_dir)
    binding = plan["tensorBindings"].get(tensor_name)
    if binding is None:
        raise ValueError("tensor not found in package: {0}".format(tensor_name))
    if binding["encoding"] != "gguf-" + encoding or len(binding["shape"]) != 2:
        raise ValueError("operator requires a rank-2 {0} tensor".format(encoding.upper()))
    rows, columns = binding["shape"]
    source = generate_program(rows, columns)
    row_bytes = columns // block_size * block_bytes
    if row_bytes % 4 or binding["bytes"] != rows * row_bytes:
        raise ValueError("{0} tensor byte length does not match its shape".format(encoding.upper()))
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("output directory already exists: {0}".format(output_dir))
    metadata = {
        "status": "operator-probe",
        "tensor": tensor_name,
        "shardId": binding["shardId"],
        "weightPath": str(Path(plan["sourcePackage"]) / binding["path"]),
        "weightChecksum": binding["checksum"],
        "weightShape": [rows, row_bytes // 4],
        "activationShape": [columns],
        "outputShape": [rows],
        "programPath": encoding + "_matvec.py",
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output_dir.name + ".partial-", dir=output_dir.parent) as directory:
        temporary = Path(directory)
        (temporary / metadata["programPath"]).write_text(source, encoding="utf-8")
        (temporary / "binding.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                                                encoding="utf-8")
        temporary.rename(output_dir)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit a Q4_1 Ramanujan matvec operator from GGUF shards")
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        metadata = emit_q4_1_operator(args.package, args.tensor, args.output_dir)
    except ValueError as error:
        parser.exit(1, "error: {0}\n".format(error))
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()