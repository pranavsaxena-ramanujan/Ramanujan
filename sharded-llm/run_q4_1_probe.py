#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from converter.ramanujan_shards.gguf_f32_matvec import dot_f32_row, generate_f32_matvec_program
from converter.ramanujan_shards.gguf_q4_1 import dot_q4_1_row, generate_q4_1_matvec_program
from converter.ramanujan_shards.gguf_q5_k import dot_q5_k_row, generate_q5_k_matvec_program
from converter.ramanujan_shards.gguf_q6_k import dot_q6_k_row, generate_q6_k_matvec_program
from run_phi3_shards import (RamanujanServer, RssMonitor, _mark_csv_older_than_binary,
                             _write_csv, _write_values)


def run_probe(binding_path, rows, java, jar, native_dir, timeout=180):
    binding = json.loads(Path(binding_path).read_text(encoding="utf-8"))
    programs = {
        "f32_matvec.py": (1, 4, generate_f32_matvec_program, dot_f32_row),
        "q4_1_matvec.py": (32, 20, generate_q4_1_matvec_program, dot_q4_1_row),
        "q5_k_matvec.py": (256, 176, generate_q5_k_matvec_program, dot_q5_k_row),
        "q6_k_matvec.py": (256, 210, generate_q6_k_matvec_program, dot_q6_k_row),
    }
    if binding["programPath"] not in programs:
        raise ValueError("unsupported quantized program: {0}".format(binding["programPath"]))
    block_size, block_bytes, generate_program, dot_row = programs[binding["programPath"]]
    weight = Path(binding["weightPath"])
    total_rows = binding["outputShape"][0]
    columns = binding["activationShape"][0]
    packed_words = binding["weightShape"][1]
    if (rows < 1 or rows > total_rows or columns % block_size or
            columns // block_size * block_bytes != packed_words * 4 or
            weight.stat().st_size != total_rows * packed_words * 4):
        raise ValueError("invalid quantized binding shape or weight length")

    activation = [((index % 7) - 3) / 8.0 for index in range(columns)]
    with tempfile.TemporaryDirectory(prefix="q4_1-probe-") as directory:
        work = Path(directory)
        program = work / binding["programPath"]
        program.write_text(generate_program(rows, columns), encoding="utf-8")
        (work / "weights.bin").symlink_to(weight.resolve())
        weights_csv = work / "weights.csv"
        _write_csv(weights_csv, rows * packed_words)
        _mark_csv_older_than_binary(weights_csv, weight)
        activation_csv = work / "activation.csv"
        _write_values(activation_csv, activation)
        output_input = work / "output-input" / "output.csv"
        output_input.parent.mkdir()
        _write_csv(output_input, rows)
        monitor = RssMonitor(7 * 1024 ** 3)
        server = RamanujanServer(java, jar, native_dir, work / "worker", monitor)
        monitor.start()
        try:
            server.start()
            server.process.stdin.write("run {0} {1} {2} {3} --dump output\n".format(
                program, weights_csv, activation_csv, output_input))
            server.process.stdin.flush()
            server._wait_for("KERNEL_DONE", timeout)
            output_csv = work / "output.csv"
            server.dump("output", output_csv)
            actual = np.fromfile(output_csv.with_suffix(".bin"), dtype="<f4")
        finally:
            server.close()
            monitor.stop()

    if len(actual) != rows:
        raise RuntimeError("worker returned {0} values, expected {1}".format(len(actual), rows))
    if not np.isfinite(actual).all():
        raise AssertionError("worker returned non-finite quantized output")
    with weight.open("rb") as source:
        expected = [dot_row(source, row, activation) for row in range(rows)]
    errors = [abs(float(value) - reference) for value, reference in zip(actual, expected)]
    if any(error > 0.002 + abs(reference) * 0.0005
           for error, reference in zip(errors, expected)):
        raise AssertionError("quantized GPU result differs from CPU reference: {0}".format(errors))
    return {"tensor": binding["tensor"], "rows": rows, "columns": columns,
            "maxAbsoluteError": max(errors), "actualSample": actual[:4].tolist(),
            "referenceSample": expected[:4], "status": "verified-{0}-matvec".format(
                binding["programPath"].split("_matvec.py")[0])}


def main():
    parser = argparse.ArgumentParser(description="Run a quantized shard matvec on Ramanujan and check CPU parity")
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--java", type=Path, default=Path(
        "/Users/pranav/Library/Java/JavaVirtualMachines/corretto-1.8.0_402/Contents/Home/bin/java"))
    parser.add_argument("--jar", type=Path, default=Path.home() / "Desktop/ws/developer-console-1.0-SNAPSHOT-fat.jar")
    parser.add_argument("--native-dir", type=Path, default=Path.home() / "Desktop/ws")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    print(json.dumps(run_probe(args.binding, args.rows, args.java, args.jar,
                               args.native_dir, args.timeout), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()