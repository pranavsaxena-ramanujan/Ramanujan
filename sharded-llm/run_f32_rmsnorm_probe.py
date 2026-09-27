#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from converter.ramanujan_shards.gguf_f32_norm import generate_f32_rmsnorm_program
from run_phi3_shards import (RamanujanServer, RssMonitor, _mark_csv_older_than_binary,
                             _write_csv, _write_values)


def run_probe(binding_path, java, jar, native_dir, timeout=180):
    binding = json.loads(Path(binding_path).read_text(encoding="utf-8"))
    if binding["programPath"] != "f32_rmsnorm.py":
        raise ValueError("unsupported norm program")
    dimension = binding["shape"][0]
    epsilon = binding["epsilon"]
    weight = Path(binding["weightPath"])
    if weight.stat().st_size != dimension * 4:
        raise ValueError("invalid F32 norm weight length")
    hidden = [((index % 19) - 9) / 8.0 for index in range(dimension)]
    with tempfile.TemporaryDirectory(prefix="f32-rmsnorm-probe-") as directory:
        work = Path(directory)
        program = work / binding["programPath"]
        program.write_text(generate_f32_rmsnorm_program(dimension, epsilon), encoding="utf-8")
        (work / "gamma.bin").symlink_to(weight.resolve())
        gamma_csv = work / "gamma.csv"
        _write_csv(gamma_csv, dimension)
        _mark_csv_older_than_binary(gamma_csv, weight)
        hidden_csv = work / "hidden.csv"
        _write_values(hidden_csv, hidden)
        output_input = work / "output-input" / "output.csv"
        output_input.parent.mkdir()
        _write_csv(output_input, dimension)
        monitor = RssMonitor(7 * 1024 ** 3)
        server = RamanujanServer(java, jar, native_dir, work / "worker", monitor)
        monitor.start()
        try:
            server.start()
            server.process.stdin.write("run {0} {1} {2} {3} --dump output\n".format(
                program, hidden_csv, gamma_csv, output_input))
            server.process.stdin.flush()
            server._wait_for("KERNEL_DONE", timeout)
            output_csv = work / "output.csv"
            server.dump("output", output_csv)
            actual = np.fromfile(output_csv.with_suffix(".bin"), dtype="<f4")
        finally:
            server.close()
            monitor.stop()

    if len(actual) != dimension or not np.isfinite(actual).all():
        raise AssertionError("worker returned invalid RMSNorm output")
    gamma = np.fromfile(weight, dtype="<f4")
    scale = 1.0 / np.sqrt(sum(value * value for value in hidden) / dimension + epsilon)
    expected = np.array(hidden, dtype=np.float32) * scale * gamma
    error = np.max(np.abs(actual - expected))
    if error > 1e-4:
        raise AssertionError("RMSNorm GPU differs from CPU reference: {0}".format(error))
    return {"tensor": binding["tensor"], "dimension": dimension, "maxAbsoluteError": float(error),
            "actualSample": actual[:4].tolist(), "referenceSample": expected[:4].tolist(),
            "status": "verified-f32-rmsnorm"}


def main():
    parser = argparse.ArgumentParser(description="Check F32 GGUF RMSNorm on the Ramanujan worker")
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--java", type=Path, default=Path(
        "/Users/pranav/Library/Java/JavaVirtualMachines/corretto-1.8.0_402/Contents/Home/bin/java"))
    parser.add_argument("--jar", type=Path, default=Path.home() / "Desktop/ws/developer-console-1.0-SNAPSHOT-fat.jar")
    parser.add_argument("--native-dir", type=Path, default=Path.home() / "Desktop/ws")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    print(json.dumps(run_probe(args.binding, args.java, args.jar,
                               args.native_dir, args.timeout), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()