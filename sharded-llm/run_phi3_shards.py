#!/usr/bin/env python3
import argparse
import collections
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import tempfile
import threading
import time

import numpy as np
from transformers import AutoTokenizer


GIB = 1024 ** 3


class MemoryLimitExceeded(RuntimeError):
    pass


class RssMonitor:
    def __init__(self, limit_bytes):
        self.limit_bytes = limit_bytes
        self.peak_bytes = {}
        self.failure = None
        self._pids = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def add(self, label, pid):
        self._pids[label] = pid

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def check(self):
        if self.failure is not None:
            raise self.failure

    def _run(self):
        while not self._stop.wait(0.25):
            for label, pid in list(self._pids.items()):
                rss = _rss_bytes(pid)
                self.peak_bytes[label] = max(self.peak_bytes.get(label, 0), rss)
                if rss > self.limit_bytes:
                    self.failure = MemoryLimitExceeded(
                        "{0} RSS {1} exceeded limit {2}".format(
                            label, rss, self.limit_bytes
                        )
                    )
                    try:
                        os.kill(pid, 9)
                    except ProcessLookupError:
                        pass
                    self._stop.set()
                    return


def _rss_bytes(pid):
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    value = result.stdout.strip()
    return int(value) * 1024 if value else 0


class RamanujanServer:
    def __init__(self, java, jar, native_dir, workspace, monitor, diagnostics=False):
        self.java = java
        self.jar = jar
        self.native_dir = native_dir
        self.workspace = workspace
        self.monitor = monitor
        self.diagnostics = diagnostics
        self.process = None
        self.stderr_tail = collections.deque(maxlen=80)

    def start(self):
        self.workspace.mkdir(parents=True, exist_ok=True)
        native_library = self.native_dir / "libnative.dylib"
        if not native_library.is_file():
            raise FileNotFoundError("native library not found: {0}".format(native_library))
        shutil.copy2(native_library, self.workspace / native_library.name)
        env = os.environ.copy()
        env["RAMANUJAN_WS"] = str(self.workspace)
        env["RAMANUJAN_SEQUENTIAL"] = "true"
        env["RAMANUJAN_MAX_PARALLELISM"] = "1"
        command = [
            str(self.java),
            "-Xmx6g",
            "-XX:+UseG1GC",
            "-Djava.library.path=" + str(self.native_dir),
            "-jar",
            str(self.jar),
            "server",
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self.monitor.add("ramanujan-jvm", self.process.pid)
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        self._wait_for("SERVER_READY", 90)

    def run(self, kernel, csv_paths, timeout):
        command = "run " + " ".join([str(kernel)] + [str(path) for path in csv_paths])
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        self._wait_for("KERNEL_DONE", timeout)

    def dump(self, name, path, timeout=180):
        self.process.stdin.write("dump {0} {1}\n".format(name, path))
        self.process.stdin.flush()
        self._wait_for("Dumped " + name, timeout, prefix=True)

    def close(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                self.process.stdin.write("quit\n")
                self.process.stdin.flush()
                self.process.wait(timeout=15)
            except Exception:
                self.process.kill()
                self.process.wait(timeout=15)

    def _wait_for(self, expected, timeout, prefix=False):
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.monitor.check()
            if self.process.poll() is not None:
                raise RuntimeError(
                    "Ramanujan JVM exited with {0}:\n{1}".format(
                        self.process.returncode, "".join(self.stderr_tail)
                    )
                )
            ready, _, _ = select.select([self.process.stdout], [], [], 0.25)
            if not ready:
                continue
            line = self.process.stdout.readline().rstrip()
            if line.startswith("KERNEL_ERROR"):
                raise RuntimeError(line + "\n" + "".join(self.stderr_tail))
            if (prefix and line.startswith(expected)) or line == expected:
                return
        raise TimeoutError("timed out waiting for {0}".format(expected))

    def _drain_stderr(self):
        for line in self.process.stderr:
            self.stderr_tail.append(line)
            if self.diagnostics and (
                "SKIPPING dispatch" in line
                or "clEnqueueNDRangeKernel" in line
                or "has no GPU buffer" in line
            ):
                print(line.rstrip(), flush=True)


def _write_csv(path, columns):
    path.write_text(",".join("0" for _ in range(columns)) + "\n", encoding="ascii")


def _mark_csv_older_than_binary(csv_path, binary_path):
    binary_mtime = binary_path.stat().st_mtime
    os.utime(csv_path, (binary_mtime - 1, binary_mtime - 1))


def _write_values(path, values):
    path.write_text(",".join(str(float(value)) for value in values) + "\n", encoding="ascii")


def _columns_for_weight(name, float_count):
    if name in ("cos_cache", "sin_cache"):
        return 48
    if name.startswith("lm_head_"):
        return 3072
    if name.endswith("_qkv_packed") or name.endswith("_o_packed"):
        return 512
    if name.endswith("_gate_up_packed"):
        return 512
    if name.endswith("_down_packed"):
        return 1366
    return int(float_count)


def _cache_names(shard_dir):
    manifest = json.loads((shard_dir / "manifest.json").read_text())
    metadata = manifest["adapterMetadata"]
    return [
        "l{0}_{1}_cache".format(layer, kind)
        for layer in range(int(metadata["layer_start"]), int(metadata["layer_end"]))
        for kind in ("k", "v")
    ]


def _bind_shard_inputs(shard_dir, run_dir, hidden_bin, sequence_length,
                       hidden_name="hidden", cache_bins=None):
    input_dir = run_dir / "inputs"
    input_dir.mkdir(parents=True)
    csv_paths = []

    hidden_link = input_dir / (hidden_name + ".bin")
    hidden_link.symlink_to(hidden_bin)
    hidden_csv = input_dir / (hidden_name + ".csv")
    _write_csv(hidden_csv, 3072)
    _mark_csv_older_than_binary(hidden_csv, hidden_bin)
    csv_paths.append(hidden_csv)

    params_csv = input_dir / "params.csv"
    current_sequence_csv = input_dir / "cur_n_seq_arr.csv"
    _write_values(params_csv, [sequence_length])
    _write_values(current_sequence_csv, [sequence_length])
    csv_paths.extend([params_csv, current_sequence_csv])

    for name, source in sorted((cache_bins or {}).items()):
        destination = input_dir / (name + ".bin")
        destination.symlink_to(source)
        csv_path = input_dir / (name + ".csv")
        _write_csv(csv_path, 3072)
        _mark_csv_older_than_binary(csv_path, source)
        csv_paths.append(csv_path)

    for source in sorted((shard_dir / "weights").glob("*.bin")):
        if source.name == "embed_tokens.bin":
            continue
        name = source.stem
        destination = input_dir / source.name
        destination.symlink_to(source)
        csv_path = input_dir / (name + ".csv")
        _write_csv(csv_path, _columns_for_weight(name, source.stat().st_size // 4))
        _mark_csv_older_than_binary(csv_path, source)
        csv_paths.append(csv_path)
    return csv_paths


def _dump_caches(server, shard_dir, state_dir):
    state_dir.mkdir(parents=True, exist_ok=True)
    cache_bins = {}
    for name in _cache_names(shard_dir):
        csv_path = state_dir / (name + ".csv")
        server.dump(name, csv_path)
        cache_bins[name] = csv_path.with_suffix(".bin")
    return cache_bins


def _write_token_embedding(path, embeddings, token_id, position):
    with path.open("wb") as stream:
        stream.truncate(1024 * 3072 * 4)
    hidden = np.memmap(path, dtype="<f4", mode="r+", shape=(1024, 3072))
    hidden[position] = embeddings[token_id]
    hidden.flush()
    del hidden


def _cache_continuity(previous_path, current_path, previous_rows, current_row):
    shape = (1024, 3072)
    previous = np.memmap(previous_path, dtype="<f4", mode="r", shape=shape)
    current = np.memmap(current_path, dtype="<f4", mode="r", shape=shape)
    result = {
        "previousRowsUnchanged": bool(np.array_equal(
            previous[:previous_rows], current[:previous_rows]
        )),
        "currentRowNorm": float(np.linalg.norm(current[current_row])),
        "nextRowNorm": float(np.linalg.norm(current[current_row + 1])),
    }
    del previous
    del current
    return result


def _row_norm(path, row):
    values = np.memmap(path, dtype="<f4", mode="r")
    start = row * 3072
    result = float(np.linalg.norm(values[start:start + 3072]))
    del values
    return result


def _row_delta(previous_path, current_path, row):
    start = row * 3072
    previous = np.memmap(previous_path, dtype="<f4", mode="r")
    current = np.memmap(current_path, dtype="<f4", mode="r")
    delta = np.abs(current[start:start + 3072] - previous[start:start + 3072])
    result = {
        "unchanged": bool(np.count_nonzero(delta) == 0),
        "maxAbsoluteDelta": float(np.max(delta)),
    }
    del previous
    del current
    return result


def _changed_rows(previous_path, current_path):
    shape = (1024, 3072)
    previous = np.memmap(previous_path, dtype="<f4", mode="r", shape=shape)
    current = np.memmap(current_path, dtype="<f4", mode="r", shape=shape)
    result = np.flatnonzero(np.any(previous != current, axis=1)).tolist()
    del previous
    del current
    return result


def _diagnostic_decode_kernel(source_path, destination_path):
    source = source_path.read_text(encoding="utf-8")
    source = source.replace(
        "GPU_SYNC(h_state)",
        "GPU_SYNC(h_attn_buf)\nGPU_SYNC(h_out_buf)\nGPU_SYNC(h_state)",
        1,
    )
    source = source.replace(
        "RETURN(h_state,", "RETURN(h_attn_buf, h_out_buf, h_state,", 1
    )
    destination_path.write_text(source, encoding="utf-8")
    return destination_path


def run_inference(args):
    package_dir = args.package_dir.resolve()
    model_dir = args.model_dir.resolve()
    manifest = json.loads((package_dir / "model-manifest.json").read_text())
    shards = manifest["shards"]
    if len(shards) != 4:
        raise ValueError("expected four shards")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True
    )
    formatted = "<|user|>\n{0}<|end|>\n<|assistant|>\n".format(args.prompt)
    token_ids = tokenizer.encode(formatted, add_special_tokens=False)
    if not token_ids or len(token_ids) > 1024:
        raise ValueError("prompt token count must be between 1 and 1024")

    monitor = RssMonitor(int(args.max_rss_gb * GIB))
    monitor.add("shard-runner", os.getpid())
    monitor.start()
    server = None
    started = time.time()
    work_dir = Path(tempfile.mkdtemp(prefix="phi3_rj_shards_"))
    try:
        embedding_path = package_dir / "shard-00" / "weights" / "embed_tokens.bin"
        embeddings = np.memmap(embedding_path, dtype="<f4", mode="r", shape=(32064, 3072))
        hidden_bin = work_dir / "prompt-hidden.bin"
        np.asarray(embeddings[token_ids], dtype="<f4").tofile(hidden_bin)
        server = RamanujanServer(
            args.java,
            args.jar,
            args.native_dir,
            work_dir / "rj-workspace",
            monitor,
            diagnostics=args.diagnostics,
        )
        server.start()

        shard_timings = []
        cache_bins = {}
        for index, shard_summary in enumerate(shards):
            monitor.check()
            shard_dir = (package_dir / shard_summary["manifestPath"]).parent
            shard_run_dir = work_dir / shard_summary["shardId"]
            csv_paths = _bind_shard_inputs(
                shard_dir, shard_run_dir, hidden_bin, len(token_ids)
            )
            before = time.time()
            server.run(shard_dir / "programs" / "prefill.py", csv_paths, args.timeout)
            next_hidden_csv = shard_run_dir / "h_state.csv"
            server.dump("h_state", next_hidden_csv)
            hidden_bin = next_hidden_csv.with_suffix(".bin")
            cache_bins.update(_dump_caches(
                server, shard_dir, work_dir / "state-prefill" / shard_summary["shardId"]
            ))
            if args.diagnostics:
                cache_norms = [
                    _row_norm(cache_bins[name], len(token_ids) - 1)
                    for name in _cache_names(shard_dir)
                ]
                print(json.dumps({
                    "event": "prefill-state",
                    "shard": shard_summary["shardId"],
                    "hiddenRowNorm": _row_norm(hidden_bin, len(token_ids) - 1),
                    "minCacheRowNorm": min(cache_norms),
                    "maxCacheRowNorm": max(cache_norms),
                }, sort_keys=True), flush=True)
            shard_timings.append(time.time() - before)
            print(
                json.dumps(
                    {
                        "event": "shard-complete",
                        "shard": shard_summary["shardId"],
                        "seconds": round(shard_timings[-1], 3),
                        "jvmPeakRssBytes": monitor.peak_bytes.get("ramanujan-jvm", 0),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        token_csv = work_dir / "next-token.csv"
        server.dump("argmax_arr", token_csv)
        token_values = np.fromfile(token_csv.with_suffix(".bin"), dtype="<f4")
        if token_values.size == 0:
            raise RuntimeError("argmax_arr was empty")
        generated_tokens = [int(token_values[0])]

        decode_timings = []
        while len(generated_tokens) < args.n_tokens:
            position = len(token_ids) + len(generated_tokens) - 1
            if position >= 1024:
                raise ValueError("prompt plus generated tokens exceeds 1024")
            decode_hidden = work_dir / "decode-{0:02d}-hidden.bin".format(len(generated_tokens))
            _write_token_embedding(decode_hidden, embeddings, generated_tokens[-1], position)
            hidden_bin = decode_hidden
            step_started = time.time()
            next_cache_bins = {}
            for shard_summary in shards:
                monitor.check()
                shard_dir = (package_dir / shard_summary["manifestPath"]).parent
                shard_run_dir = work_dir / "decode-{0:02d}".format(len(generated_tokens)) / shard_summary["shardId"]
                shard_caches = {
                    name: cache_bins[name] for name in _cache_names(shard_dir)
                }
                csv_paths = _bind_shard_inputs(
                    shard_dir,
                    shard_run_dir,
                    hidden_bin,
                    position + 1,
                    hidden_name="h_state",
                    cache_bins=shard_caches,
                )
                input_hidden_path = hidden_bin
                input_hidden_norm = _row_norm(hidden_bin, position)
                decode_kernel = shard_dir / "programs" / "decode.py"
                if args.diagnostics:
                    decode_kernel = _diagnostic_decode_kernel(
                        decode_kernel, shard_run_dir / "decode-diagnostic.py"
                    )
                server.run(decode_kernel, csv_paths, args.timeout)
                next_hidden_csv = shard_run_dir / "h_state-output.csv"
                server.dump("h_state", next_hidden_csv)
                hidden_bin = next_hidden_csv.with_suffix(".bin")
                next_cache_bins.update(_dump_caches(
                    server, shard_dir, shard_run_dir / "state"
                ))
                if args.diagnostics:
                    attention_csv = shard_run_dir / "h_attn_buf.csv"
                    output_csv = shard_run_dir / "h_out_buf.csv"
                    server.dump("h_attn_buf", attention_csv)
                    server.dump("h_out_buf", output_csv)
                    hidden_delta = _row_delta(input_hidden_path, hidden_bin, position)
                    cache_checks = {
                        name: _cache_continuity(
                            shard_caches[name], next_cache_bins[name], position, position
                        )
                        for name in _cache_names(shard_dir)
                    }
                    print(json.dumps({
                        "event": "cache-continuity",
                        "shard": shard_summary["shardId"],
                        "position": position,
                        "inputHiddenRowNorm": input_hidden_norm,
                        "outputHiddenRowNorm": _row_norm(hidden_bin, position),
                        "hiddenRowUnchanged": hidden_delta["unchanged"],
                        "hiddenRowMaxAbsoluteDelta": hidden_delta["maxAbsoluteDelta"],
                        "changedHiddenRows": _changed_rows(
                            input_hidden_path, hidden_bin
                        ),
                        "attentionProjectionRowNorm": _row_norm(
                            attention_csv.with_suffix(".bin"), position
                        ),
                        "mlpProjectionRowNorm": _row_norm(
                            output_csv.with_suffix(".bin"), position
                        ),
                        "allPreviousRowsUnchanged": all(
                            check["previousRowsUnchanged"]
                            for check in cache_checks.values()
                        ),
                        "minCurrentRowNorm": min(
                            check["currentRowNorm"] for check in cache_checks.values()
                        ),
                        "maxNextRowNorm": max(
                            check["nextRowNorm"] for check in cache_checks.values()
                        ),
                    }, sort_keys=True), flush=True)
            cache_bins = next_cache_bins
            token_csv = work_dir / "next-token-{0:02d}.csv".format(len(generated_tokens))
            server.dump("argmax_arr", token_csv)
            token_values = np.fromfile(token_csv.with_suffix(".bin"), dtype="<f4")
            if token_values.size == 0:
                raise RuntimeError("argmax_arr was empty during decode")
            generated_tokens.append(int(token_values[0]))
            decode_timings.append(time.time() - step_started)
            print(json.dumps({
                "event": "token-complete",
                "tokenIndex": len(generated_tokens) - 1,
                "tokenId": generated_tokens[-1],
                "token": tokenizer.decode([generated_tokens[-1]]),
                "seconds": round(decode_timings[-1], 3),
                "jvmPeakRssBytes": monitor.peak_bytes.get("ramanujan-jvm", 0),
            }, sort_keys=True), flush=True)

        monitor.check()
        return {
            "prompt": args.prompt,
            "promptTokens": len(token_ids),
            "generatedTokenIds": generated_tokens,
            "generatedText": tokenizer.decode(generated_tokens),
            "seconds": round(time.time() - started, 3),
            "shardSeconds": [round(value, 3) for value in shard_timings],
            "decodeSeconds": [round(value, 3) for value in decode_timings],
            "peakRssBytes": dict(monitor.peak_bytes),
        }
    finally:
        if server is not None:
            server.close()
        monitor.stop()
        shutil.rmtree(work_dir, ignore_errors=True)


def parse_args():
    workspace = Path(__file__).resolve().parents[2]
    default_java_home = Path(
        "/Users/pranav/Library/Java/JavaVirtualMachines/corretto-1.8.0_402/Contents/Home"
    )
    parser = argparse.ArgumentParser(
        description="Run guarded autoregressive Phi-3 inference through four Ramanujan shards"
    )
    parser.add_argument("prompt")
    parser.add_argument("--package-dir", type=Path, default=workspace / "phi3_rj_ir_shards")
    parser.add_argument("--model-dir", type=Path, default=workspace / "Phi-3-mini-4k-instruct")
    parser.add_argument("--java", type=Path, default=default_java_home / "bin" / "java")
    parser.add_argument(
        "--jar", type=Path, default=Path.home() / "Desktop/ws/developer-console-1.0-SNAPSHOT-fat.jar"
    )
    parser.add_argument("--native-dir", type=Path, default=Path.home() / "Desktop/ws")
    parser.add_argument("--max-rss-gb", type=float, default=8.0)
    parser.add_argument("--n-tokens", type=int, default=1)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run_inference(parse_args()), sort_keys=True), flush=True)