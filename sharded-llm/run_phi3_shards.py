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
    def __init__(self, label, rss, limit):
        self.label = label
        super().__init__("{0} RSS {1} exceeded limit {2}".format(label, rss, limit))


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

    def remove(self, label):
        self._pids.pop(label, None)
        if self.failure is not None and self.failure.label == label:
            self.failure = None

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
                if rss > self.limit_bytes and self.failure is None:
                    self.failure = MemoryLimitExceeded(label, rss, self.limit_bytes)
                    if label == "ramanujan-jvm":
                        try:
                            os.kill(pid, 9)
                        except ProcessLookupError:
                            pass
                    break


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
    def __init__(self, java, jar, native_dir, workspace, monitor, diagnostics=False,
                 package_manifest=None, resident_kv=False, profile=False, gpu_pool=False):
        self.java = java
        self.jar = jar
        self.native_dir = native_dir
        self.workspace = workspace
        self.monitor = monitor
        self.diagnostics = diagnostics
        self.package_manifest = package_manifest
        self.resident_kv = resident_kv
        self.profile = profile
        self.gpu_pool = gpu_pool
        self.process = None
        self.stderr_tail = collections.deque(maxlen=80)

    def start(self):
        self.workspace.mkdir(parents=True, exist_ok=True)
        native_library = self.native_dir / "libnative_llm.dylib"
        if not native_library.is_file():
            raise FileNotFoundError("native library not found: {0}".format(native_library))
        shutil.copy2(native_library, self.workspace / native_library.name)
        env = os.environ.copy()
        env["RAMANUJAN_WS"] = str(self.workspace)
        env["RAMANUJAN_SEQUENTIAL"] = "true"
        env["RAMANUJAN_MAX_PARALLELISM"] = "1"
        env["RAMANUJAN_RESIDENT_KV"] = "true" if self.resident_kv else "false"
        env["RAMANUJAN_GPU_POOL"] = "true" if self.gpu_pool else "false"
        command = [
            str(self.java),
            "-Xmx6g",
            "-XX:+UseG1GC",
            "-Dramanujan.nativeLibrary=native_llm",
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
        if self.package_manifest is not None:
            self.process.stdin.write("REGISTER_SHARDS {0}\n".format(self.package_manifest))
            self.process.stdin.flush()
            self._wait_for("SHARDS_REGISTERED", 30)

    def run(self, kernel, csv_paths, timeout):
        command = "run " + " ".join([str(kernel)] + [str(path) for path in csv_paths])
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        self._wait_for("KERNEL_DONE", timeout)

    def change_shard(self, kernel, timeout):
        self.process.stdin.write("CHANGE_SHARD {0}\n".format(kernel))
        self.process.stdin.flush()
        self._wait_for("SHARD_READY", timeout)

    def dump(self, name, path, timeout=180):
        self.process.stdin.write("dump {0} {1}\n".format(name, path))
        self.process.stdin.flush()
        self._wait_for("Dumped " + name, timeout, prefix=True)

    def take(self, name, path, timeout=180):
        self.process.stdin.write("take {0} {1}\n".format(name, path))
        self.process.stdin.flush()
        self._wait_for("Taken " + name, timeout)

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
        self.monitor.remove("ramanujan-jvm")
        self.process = None

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
            if (line.startswith("KERNEL_ERROR") or line.startswith("SHARD_ERROR")
                    or line.startswith("Unknown command")):
                raise RuntimeError(line + "\n" + "".join(self.stderr_tail))
            if (prefix and line.startswith(expected)) or line == expected:
                return
        raise TimeoutError("timed out waiting for {0}".format(expected))

    def _drain_stderr(self):
        for line in self.process.stderr:
            self.stderr_tail.append(line)
            if (self.profile and line.startswith("[Server]")) or (self.diagnostics and (
                "SKIPPING dispatch" in line
                or "clEnqueueNDRangeKernel" in line
                or "has no GPU buffer" in line
            )):
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


def _link_state(source, destination):
    # Native decides mutability from the resolved file name, so a symlink would expose the target's name.
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def _bind_shard_weights(shard_dir, binding_dir):
    binding_dir.mkdir(parents=True)
    csv_paths = []
    for source in sorted((shard_dir / "weights").glob("*.bin")):
        if source.name == "embed_tokens.bin":
            continue
        name = source.stem
        (binding_dir / source.name).symlink_to(source)
        csv_path = binding_dir / (name + ".csv")
        _write_csv(csv_path, _columns_for_weight(name, source.stat().st_size // 4))
        _mark_csv_older_than_binary(csv_path, source)
        csv_paths.append(csv_path)
    return csv_paths


def _bind_shard_inputs(weight_csvs, run_dir, hidden_bin, sequence_length,
                       hidden_name="hidden", cache_bins=None):
    input_dir = run_dir / "inputs"
    input_dir.mkdir(parents=True)
    csv_paths = []

    hidden_link = input_dir / (hidden_name + ".bin")
    _link_state(hidden_bin, hidden_link)
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
        _link_state(source, destination)
        csv_path = input_dir / (name + ".csv")
        _write_csv(csv_path, 3072)
        _mark_csv_older_than_binary(csv_path, source)
        csv_paths.append(csv_path)
    return csv_paths + weight_csvs


def _dump_caches(server, shard_dir, state_dir, take=False):
    state_dir.mkdir(parents=True, exist_ok=True)
    cache_bins = {}
    for name in _cache_names(shard_dir):
        csv_path = state_dir / (name + ".csv")
        if take:
            server.take(name, csv_path.with_suffix(".bin"))
        else:
            server.dump(name, csv_path)
        cache_bins[name] = csv_path.with_suffix(".bin")
    return cache_bins


def _execute_shard(server, kernel, csv_paths, shard_dir, run_dir, state_dir,
                   timeout, diagnostics=False, token_csv=None, reuse_worker=False,
                   resident_kv=False, allow_retry=True, profile=False):
    for attempt in range(2):
        completed = False
        try:
            if reuse_worker and server.process is not None:
                server.change_shard(kernel, timeout)
            else:
                server.start()
            run_started = time.monotonic()
            server.run(kernel, csv_paths, timeout)
            run_seconds = time.monotonic() - run_started
            transfer_started = time.monotonic()
            hidden_csv = run_dir / "h_state-output.csv"
            if resident_kv:
                server.take("h_state", hidden_csv.with_suffix(".bin"))
            else:
                server.dump("h_state", hidden_csv)
            caches = (_dump_caches(server, shard_dir, state_dir, take=resident_kv)
                      if not resident_kv or kernel.name == "prefill.py" else {})
            if diagnostics:
                server.dump("h_attn_buf", run_dir / "h_attn_buf.csv")
                server.dump("h_out_buf", run_dir / "h_out_buf.csv")
            if token_csv is not None:
                server.dump("argmax_arr", token_csv)
            if profile:
                print(json.dumps({"event": "shard-profile", "shard": shard_dir.name,
                                  "program": kernel.name, "runSeconds": round(run_seconds, 3),
                                  "transferSeconds": round(time.monotonic() - transfer_started, 3)}),
                      flush=True)
            server.monitor.check()
            completed = True
            return hidden_csv.with_suffix(".bin"), caches
        except MemoryLimitExceeded as exc:
            if exc.label != "ramanujan-jvm" or attempt == 1 or not allow_retry:
                raise
            print(json.dumps({"event": "worker-restart", "shard": shard_dir.name,
                              "reason": "rss-limit"}), flush=True)
        finally:
            if not reuse_worker or not completed:
                server.close()


def _write_token_embedding(path, previous_hidden, embeddings, token_id, position):
    shutil.copyfile(previous_hidden, path)
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
    if (manifest.get("architectureId") != "phi3" or
            manifest.get("sourceFormat") != "safetensors" or
            manifest.get("status") == "weights-only"):
        raise ValueError("run_phi3_shards requires an executable Phi-3 package; "
                         "GGUF weight shards and IR plans are not runnable by this worker")
    shards = manifest["shards"]
    if len(shards) != 4:
        raise ValueError("expected four shards")
    if args.resident_kv and (args.worker_per_shard or args.diagnostics):
        raise ValueError("resident KV requires the persistent worker without diagnostics")
    if args.gpu_pool and not args.resident_kv:
        raise ValueError("GPU pool requires resident KV mode")
    if args.native_loop and not args.resident_kv:
        raise ValueError("native loop requires resident KV mode")
    if args.resident_kv:
        for shard_summary in shards:
            shard_dir = (package_dir / shard_summary["manifestPath"]).parent
            if not (shard_dir / "programs" / "decode_resident.py").is_file():
                raise ValueError("refresh the shard programs before enabling resident KV")
    if args.native_loop and not (package_dir / "shard-00/programs/decode_fused.py").is_file():
        raise ValueError("refresh the shard programs before enabling the native loop")
    # Diagnostic programs live outside the registered package, so they need per-shard workers.
    reuse_worker = not (args.worker_per_shard or args.diagnostics)

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
            package_manifest=package_dir / "model-manifest.json" if reuse_worker else None,
            resident_kv=args.resident_kv,
            profile=args.profile,
            gpu_pool=args.gpu_pool,
        )
        weight_csvs = {
            shard_summary["shardId"]: _bind_shard_weights(
                (package_dir / shard_summary["manifestPath"]).parent,
                work_dir / "weights" / shard_summary["shardId"],
            )
            for shard_summary in shards
        }
        fused_weight_csvs = []
        if args.native_loop:
            fused_weight_csvs = list({path.stem: path for shard_summary in shards
                                      for path in weight_csvs[shard_summary["shardId"]]}.values())
        shard_timings = []
        cache_bins = {}
        for index, shard_summary in enumerate(shards):
            monitor.check()
            shard_dir = (package_dir / shard_summary["manifestPath"]).parent
            shard_run_dir = work_dir / shard_summary["shardId"]
            csv_paths = _bind_shard_inputs(
                weight_csvs[shard_summary["shardId"]], shard_run_dir, hidden_bin, len(token_ids)
            )
            before = time.time()
            hidden_bin, shard_caches = _execute_shard(
                server, shard_dir / "programs" / "prefill.py", csv_paths,
                shard_dir, shard_run_dir,
                work_dir / "state-prefill" / shard_summary["shardId"],
                args.timeout,
                token_csv=work_dir / "next-token.csv" if index == len(shards) - 1 else None,
                reuse_worker=reuse_worker,
                resident_kv=args.resident_kv,
                profile=args.profile,
            )
            cache_bins.update(shard_caches)
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
            _write_token_embedding(decode_hidden, hidden_bin, embeddings, generated_tokens[-1], position)
            hidden_bin = decode_hidden
            step_started = time.time()
            next_cache_bins = cache_bins.copy() if args.resident_kv else {}
            if args.native_loop:
                fused_dir = work_dir / "decode-{0:02d}".format(len(generated_tokens)) / "fused"
                csv_paths = _bind_shard_inputs(
                    fused_weight_csvs, fused_dir, hidden_bin, position + 1,
                    hidden_name="h_state", cache_bins=cache_bins,
                )
                hidden_bin, _ = _execute_shard(
                    server, package_dir / "shard-00/programs/decode_fused.py",
                    csv_paths, package_dir / "shard-00", fused_dir, fused_dir / "state",
                    args.timeout, token_csv=work_dir / "next-token-{0:02d}.csv".format(len(generated_tokens)),
                    reuse_worker=True, resident_kv=True, allow_retry=False,
                    profile=args.profile,
                )
                monitor.check()
                token_csv = work_dir / "next-token-{0:02d}.csv".format(len(generated_tokens))
                token_values = np.fromfile(token_csv.with_suffix(".bin"), dtype="<f4")
                if token_values.size == 0:
                    raise RuntimeError("argmax_arr was empty during decode")
                generated_tokens.append(int(token_values[0]))
                decode_timings.append(time.time() - step_started)
                print(json.dumps({
                    "event": "token-complete", "tokenIndex": len(generated_tokens) - 1,
                    "tokenId": generated_tokens[-1],
                    "token": tokenizer.decode([generated_tokens[-1]]),
                    "seconds": round(decode_timings[-1], 3),
                    "jvmPeakRssBytes": monitor.peak_bytes.get("ramanujan-jvm", 0),
                }, sort_keys=True), flush=True)
                continue
            for index, shard_summary in enumerate(shards):
                monitor.check()
                shard_dir = (package_dir / shard_summary["manifestPath"]).parent
                shard_run_dir = work_dir / "decode-{0:02d}".format(len(generated_tokens)) / shard_summary["shardId"]
                shard_caches = {
                    name: cache_bins[name] for name in _cache_names(shard_dir)
                }
                csv_paths = _bind_shard_inputs(
                    weight_csvs[shard_summary["shardId"]],
                    shard_run_dir,
                    hidden_bin,
                    position + 1,
                    hidden_name="h_state",
                    cache_bins=shard_caches,
                )
                input_hidden_path = hidden_bin
                input_hidden_norm = _row_norm(hidden_bin, position)
                decode_kernel = shard_dir / "programs" / (
                    "decode_resident.py" if args.resident_kv else "decode.py"
                )
                if args.diagnostics:
                    decode_kernel = _diagnostic_decode_kernel(
                        decode_kernel, shard_run_dir / "decode-diagnostic.py"
                    )
                token_csv = (work_dir / "next-token-{0:02d}.csv".format(len(generated_tokens))
                             if index == len(shards) - 1 else None)
                hidden_bin, dumped_caches = _execute_shard(
                    server, decode_kernel, csv_paths, shard_dir, shard_run_dir,
                    shard_run_dir / "state", args.timeout,
                    diagnostics=args.diagnostics, token_csv=token_csv,
                    reuse_worker=reuse_worker,
                    resident_kv=args.resident_kv,
                    allow_retry=not args.resident_kv,
                    profile=args.profile,
                )
                next_cache_bins.update(dumped_caches)
                if args.diagnostics:
                    attention_csv = shard_run_dir / "h_attn_buf.csv"
                    output_csv = shard_run_dir / "h_out_buf.csv"
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
    parser.add_argument("--max-rss-gb", type=float, default=7.0)
    parser.add_argument("--n-tokens", type=int, default=1)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--worker-per-shard", action="store_true",
                        help="start a fresh JVM for every shard run instead of one worker")
    parser.add_argument("--resident-kv", action="store_true",
                        help="keep KV buffers in native memory across decode; disable retry after prefill")
    parser.add_argument("--gpu-pool", action="store_true",
                        help="reuse bounded OpenCL buffers between shards (requires --resident-kv)")
    parser.add_argument("--native-loop", action="store_true",
                        help="decode all four shards in one worker run (requires --resident-kv)")
    parser.add_argument("--profile", action="store_true",
                        help="print per-shard run and result transfer timings")
    parser.add_argument("--timeout", type=int, default=1800)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run_inference(parse_args()), sort_keys=True), flush=True)