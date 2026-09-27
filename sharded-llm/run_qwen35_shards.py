#!/usr/bin/env python3
"""Run a four-shard Qwen35 GGUF package on Ramanujan workers.

By default each shard gets a local persistent JVM. With --homelab, every shard step is
submitted to an `rj homelab` server and executed by `rj worker` processes; the shard id is
the task affinity, so each worker caches and runs only the shards it is assigned.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np

from converter.ramanujan_shards.qwen35_programs import (
    ATTN_TENSORS, DELTA_TENSORS, HEAD_TENSORS, Qwen35Config, generate_attention_program,
    generate_delta_program, generate_embed_program, generate_head_program, layer_encodings,
    load_package, tensor_columns)
from converter.ramanujan_shards.qwen35_reference import Qwen35Reference
from converter.ramanujan_shards.qwen35_tokenizer import load_tokenizer
from run_phi3_shards import RamanujanServer, RssMonitor, _mark_csv_older_than_binary, _write_csv, _write_values


def _stub(csv_path, columns, binary):
    _write_csv(csv_path, columns)
    _mark_csv_older_than_binary(csv_path, binary)
    return csv_path


def _zeros(path, count, columns):
    np.zeros(count, np.float32).tofile(path)
    return _stub(path.with_suffix(".csv"), columns, path)


class WeightPrefetcher:
    """Warms the OS page cache for upcoming shard steps while the current step computes.

    The native runtime mmaps weight files and faults them in page by page, and eviction
    only unmaps them. Large sequential reads on background threads fill the page cache
    ahead of the JVM, so SSD reads overlap with compute and run at sequential speed.
    """
    CHUNK = 8 << 20

    def __init__(self, threads):
        self.pool = concurrent.futures.ThreadPoolExecutor(threads, thread_name_prefix="prefetch")
        self.pending = set()
        self.stopped = threading.Event()
        self.local = threading.local()

    def warm(self, key, files):
        if key in self.pending:
            return
        self.pending.add(key)
        for path in files:
            self.pool.submit(self._read, path)

    def consumed(self, key):
        self.pending.discard(key)

    def _read(self, path):
        buffer = getattr(self.local, "buffer", None)
        if buffer is None:
            buffer = self.local.buffer = bytearray(self.CHUNK)
        try:
            with open(path, "rb", buffering=0) as source:
                if hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_WILLNEED)
                    return
                while not self.stopped.is_set() and source.readinto(buffer):
                    pass
        except OSError:
            pass

    def close(self):
        self.stopped.set()
        self.pool.shutdown(wait=True, cancel_futures=True)


class Worker:
    def __init__(self, shard, args, workspace):
        self.shard = shard
        self.monitor = RssMonitor(int(args.rss_limit_gb * 1024 ** 3))
        self.server = RamanujanServer(args.java, args.jar, args.native_dir, workspace, self.monitor)
        self.timeout = args.timeout

    def start(self):
        self.monitor.start()
        self.server.start()

    def run(self, program, csvs, takes, evict=False):
        self.server.run(program, csvs, self.timeout)
        for name, destination in takes.items():
            temporary = destination.with_name(destination.name + ".take")
            self.server.take(name, temporary)
            os.replace(temporary, destination)

    def evict(self):
        self.server.process.stdin.write("EVICT_WEIGHTS\n")
        self.server.process.stdin.flush()
        self.server._wait_for("WEIGHTS_EVICTED", self.timeout)

    def close(self):
        peak = self.monitor.peak_bytes.get("ramanujan-jvm", 0)
        self.server.close()
        self.monitor.stop()
        return peak


class HomelabWorker:
    """One shard's steps on an `rj homelab` server; the shard id pins them to one `rj worker`."""

    def __init__(self, shard, args):
        self.shard = shard
        self.url = args.homelab.rstrip("/")
        self.timeout = args.timeout
        self.resident = args.resident_weights

    def start(self):
        pass

    def _post(self, route, body):
        request = urllib.request.Request(self.url + route, json.dumps(body).encode("utf-8"),
                                         {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            raise RuntimeError("homelab {0} failed with HTTP {1}: {2}".format(route, error.code, detail)) from None
        if payload.get("status") != "SUCCESS":
            raise RuntimeError("homelab {0} failed: {1}".format(route, payload))
        return payload

    def run(self, program, csvs, takes, evict=False):
        request_id = "{0}-{1}".format(self.shard["id"], uuid.uuid4().hex)
        self._post("/orchestrator/run", {
            "args": [str(program)] + [str(csv) for csv in csvs],
            "requestId": request_id,
            "affinity": self.shard["id"],
            "evictWeights": bool(evict and not self.resident)})
        for name, destination in takes.items():
            temporary = destination.with_name(destination.name + ".take")
            self._post("/orchestrator/dump", {"name": name, "path": str(temporary),
                                              "requestId": request_id, "raw": True})
            os.replace(temporary, destination)

    def evict(self):
        pass

    def close(self):
        return 0


class Qwen35Runner:
    def __init__(self, args):
        self.args = args
        metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
        self.config = Qwen35Config.from_metadata(metadata)
        self.tokenizer = load_tokenizer(metadata)
        self.eos = {int(metadata["tokenizer.ggml.eos_token_id"])}
        if "tokenizer.ggml.bos_token_id" in metadata:
            self.eos.add(int(metadata["tokenizer.ggml.bos_token_id"]))
        self.shards = load_package(args.package)
        if len(self.shards) != 4:
            raise ValueError("expected a four-shard package, found {0}".format(len(self.shards)))
        self.encodings = layer_encodings(self.config, self.shards)
        self.work = Path(args.work_dir).resolve()
        if self.work.exists():
            raise ValueError("work directory already exists: {0}".format(self.work))
        self.work.mkdir(parents=True)
        self.owner = {}
        for index, shard in enumerate(self.shards):
            for name in shard["tensors"]:
                self.owner[name] = index
        self._write_programs()
        self._bind_layers()
        self.workers = [None] * len(self.shards)
        self.position = 0
        self.prefetcher = WeightPrefetcher(args.prefetch_threads) if args.prefetch_steps > 0 else None
        last = args.stop_after_layer if args.stop_after_layer is not None else self.config.layers - 1
        self.cycle = [("layer", layer) for layer in range(last + 1)]
        if args.stop_after_layer is None:
            self.cycle.append(("head", None))

    def _write_programs(self):
        c, e, directory = self.config, self.encodings, self.work / "programs"
        directory.mkdir()
        sources = {
            "embed": generate_embed_program(c, e["embed"]["token_embd"]),
            "delta": generate_delta_program(c, e["delta"]),
            "attn": generate_attention_program(c, e["attn"], self.args.max_context),
            "head": generate_head_program(c, e["head"]),
        }
        self.programs = {}
        for name, source in sources.items():
            path = directory / (name + ".py")
            path.write_text(source, encoding="utf-8")
            self.programs[name] = path

    def _tensor(self, name):
        return self.shards[self.owner[name]]["tensors"][name]

    def _bind_layers(self):
        c = self.config
        self.layer_inputs = []
        for layer in range(c.layers):
            directory = self.work / "bind" / "blk.{0}".format(layer)
            state = self.work / "state" / "blk.{0}".format(layer)
            directory.mkdir(parents=True)
            state.mkdir(parents=True)
            attention = c.is_attention(layer)
            roles = ATTN_TENSORS if attention else DELTA_TENSORS
            csvs, weights = [], []
            for role, suffix in roles.items():
                tensor = self._tensor("blk.{0}.{1}".format(layer, suffix))
                weights.append(tensor["file"])
                (directory / (role + ".bin")).symlink_to(tensor["file"])
                csvs.append(_stub(directory / (role + ".csv"), tensor_columns(tensor), tensor["file"]))
            if attention:
                width = c.kv_heads * c.head_dim
                states = {"attn_k_cache": (self.args.max_context * width, width),
                          "attn_v_cache": (self.args.max_context * width, width)}
            else:
                states = {"ssm_s_state": (c.v_heads * c.state_size * c.state_size, c.state_size),
                          "ssm_conv_state": (c.conv_dim * (c.conv_kernel - 1), c.conv_kernel - 1)}
            for name, (count, columns) in states.items():
                csvs.append(_zeros(state / (name + ".bin"), count, columns))
            owners = {self.owner["blk.{0}.{1}".format(layer, suffix)] for suffix in roles.values()}
            if len(owners) != 1:
                raise ValueError("layer {0} is split across shards".format(layer))
            self.layer_inputs.append({"worker": owners.pop(), "csvs": csvs, "state": state,
                                      "program": self.programs["attn" if attention else "delta"],
                                      "takes": list(states), "weights": weights})
        head = self.work / "bind" / "head"
        head.mkdir()
        self.head_csvs, self.head_weights = [], []
        for role, name in HEAD_TENSORS.items():
            tensor = self._tensor(name)
            self.head_weights.append(tensor["file"])
            (head / (role + ".bin")).symlink_to(tensor["file"])
            self.head_csvs.append(_stub(head / (role + ".csv"), tensor_columns(tensor), tensor["file"]))
        owners = {self.owner[name] for name in HEAD_TENSORS.values()}
        if len(owners) != 1:
            raise ValueError("output head is split across shards")
        self.head_worker = owners.pop()
        self.embed_worker = self.owner["token_embd.weight"]

    def _step_weights(self, step):
        return self.head_weights if step[0] == "head" else self.layer_inputs[step[1]]["weights"]

    def _prefetch(self, step):
        """Warm `step` and the next --prefetch-steps steps of the layer/head cycle."""
        if self.prefetcher is None:
            return
        index = self.cycle.index(step)
        for offset in range(self.args.prefetch_steps + 1):
            upcoming = self.cycle[(index + offset) % len(self.cycle)]
            self.prefetcher.warm(upcoming, self._step_weights(upcoming))

    def _consumed(self, step):
        if self.prefetcher is not None:
            self.prefetcher.consumed(step)

    def _worker(self, index):
        if self.workers[index] is None:
            if self.args.homelab:
                worker = HomelabWorker(self.shards[index], self.args)
            else:
                worker = Worker(self.shards[index], self.args, self.work / "workers" / self.shards[index]["id"])
            worker.start()
            self.workers[index] = worker
        return self.workers[index]

    def _embed(self, token, directory):
        tensor = self._tensor("token_embd.weight")
        row_bytes = tensor_columns(tensor) * 4
        rows = self._row_cache(tensor["file"])
        row = rows / "tok{0}.bin".format(token)
        if not row.exists():
            with open(tensor["file"], "rb") as source:
                source.seek(token * row_bytes)
                data = source.read(row_bytes)
            if len(data) != row_bytes:
                raise ValueError("token id outside embedding table: {0}".format(token))
            temporary = row.with_name("{0}.{1}.partial".format(row.name, os.getpid()))
            temporary.write_bytes(data)
            os.replace(temporary, row)
        bind = directory / "embed"
        bind.mkdir()
        (bind / "emb_row.bin").symlink_to(row)
        csv = _stub(bind / "emb_row.csv", row_bytes // 4, row)
        hidden = directory / "h_state.bin"
        self._worker(self.embed_worker).run(self.programs["embed"], [csv], {"h_state": hidden}, evict=True)
        _stub(directory / "h_state.csv", self.config.dim, hidden)

    def _row_cache(self, table):
        # Rows are immutable slices of the embedding table. A stable per-table path lets
        # workers cache each token row once across runs instead of once per work dir.
        stat = Path(table).stat()
        key = hashlib.sha256("{0}\0{1}\0{2}".format(Path(table).resolve(), stat.st_size,
                                                     stat.st_mtime_ns).encode("utf-8")).hexdigest()[:24]
        rows = Path(self.args.row_cache).expanduser().resolve() / key
        rows.mkdir(parents=True, exist_ok=True)
        return rows

    def forward(self, tokens, on_layer=None):
        """Advance every layer's state by `tokens`; returns per-token hidden directories."""
        directories = []
        for offset, token in enumerate(tokens):
            directory = self.work / "hidden" / str(self.position + offset)
            directory.mkdir(parents=True)
            _write_values(directory / "pos_arr.csv", [self.position + offset])
            self._embed(token, directory)
            directories.append(directory)
        if on_layer is not None:
            on_layer(-1, directories)
        last_layer = self.args.stop_after_layer if self.args.stop_after_layer is not None else self.config.layers - 1
        for layer in range(last_layer + 1):
            spec = self.layer_inputs[layer]
            self._prefetch(("layer", layer))
            worker = self._worker(spec["worker"])
            started = time.monotonic()
            for directory in directories:
                csvs = [directory / "h_state.csv"] + spec["csvs"]
                if self.config.is_attention(layer):
                    csvs.append(directory / "pos_arr.csv")
                takes = {"h_state": directory / "h_state.bin"}
                takes.update({name: spec["state"] / (name + ".bin") for name in spec["takes"]})
                worker.run(spec["program"], csvs, takes, evict=directory is directories[-1])
            worker.evict()
            self._consumed(("layer", layer))
            if self.args.verbose:
                print(json.dumps({"event": "layer", "layer": layer, "shard": self.shards[spec["worker"]]["id"],
                                  "tokens": len(tokens), "seconds": round(time.monotonic() - started, 2)}),
                      flush=True)
            if on_layer is not None:
                on_layer(layer, directories)
        self.position += len(tokens)
        return directories

    def head(self, directory):
        self._prefetch(("head", None))
        worker = self._worker(self.head_worker)
        logits, argmax = directory / "logits.bin", directory / "argmax.bin"
        worker.run(self.programs["head"], [directory / "h_state.csv"] + self.head_csvs,
                   {"logits": logits, "argmax_arr": argmax}, evict=True)
        worker.evict()
        self._consumed(("head", None))
        return int(np.fromfile(argmax, "<f4")[0]), np.fromfile(logits, "<f4")

    def close(self):
        if self.prefetcher is not None:
            self.prefetcher.close()
        peaks = {}
        for worker in self.workers:
            if worker is not None:
                peaks[worker.shard["id"]] = worker.close()
        return peaks


def _compare(reference, runner, tokens):
    states = {}
    hidden = [reference.embed(token) for token in tokens]
    report = []

    def on_layer(layer, directories):
        if layer >= 0:
            states.setdefault(layer, reference.new_state(layer))
            for offset in range(len(tokens)):
                hidden[offset] = reference.layer(layer, hidden[offset], states[layer], offset)
        errors = []
        for offset, directory in enumerate(directories):
            actual = np.fromfile(directory / "h_state.bin", "<f4")
            errors.append(float(np.max(np.abs(actual - hidden[offset])) / max(1e-6, np.max(np.abs(hidden[offset])))))
        entry = {"event": "reference-check", "layer": layer, "maxRelativeError": max(errors)}
        report.append(entry)
        print(json.dumps(entry), flush=True)

    return on_layer, hidden, report


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen35 four-shard inference on Ramanujan")
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path, help="gguf-metadata.json from gguf_ir_plan")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-context", type=int, default=128)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--java", type=Path, default=Path(
        "/Users/pranav/Library/Java/JavaVirtualMachines/corretto-1.8.0_402/Contents/Home/bin/java"))
    parser.add_argument("--jar", type=Path, default=Path("../developer-console/target/developer-console-1.0-SNAPSHOT-fat.jar"))
    parser.add_argument("--native-dir", type=Path, default=Path("../ramanujan-native/native/build"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--rss-limit-gb", type=float, default=5.0)
    parser.add_argument("--check-layers", type=int, help="compare the first N layers with the NumPy reference, then stop")
    parser.add_argument("--reference-token", action="store_true",
                        help="also run the full NumPy reference for the first generated token")
    parser.add_argument("--row-cache", default="~/.cache/ramanujan/qwen35-embedding-rows",
                        help="shared directory for token embedding rows (reused across runs)")
    parser.add_argument("--homelab", metavar="URL",
                        help="submit shard steps to an `rj homelab` server instead of local JVMs")
    parser.add_argument("--resident-weights", action="store_true",
                        help="with --homelab, keep each worker's shard weights mapped between steps")
    parser.add_argument("--prefetch-steps", type=int,
                        help="layer/head steps to read ahead into the page cache while the current "
                             "step computes (default 1 locally, 0 with --homelab)")
    parser.add_argument("--prefetch-threads", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.prefetch_steps is None:
        args.prefetch_steps = 0 if args.homelab else 1
    if args.resident_weights and not args.homelab:
        parser.error("--resident-weights requires --homelab")
    args.stop_after_layer = args.check_layers - 1 if args.check_layers else None
    return args


def main():
    args = parse_args()
    runner = Qwen35Runner(args)
    tokens = runner.tokenizer.encode(args.prompt).ids
    if not tokens or len(tokens) + args.max_new_tokens > args.max_context:
        raise SystemExit("prompt plus new tokens must fit in --max-context")
    print(json.dumps({"event": "prompt", "tokens": tokens}), flush=True)
    try:
        if args.check_layers:
            reference = Qwen35Reference(runner.config, runner.shards)
            on_layer, _, report = _compare(reference, runner, tokens)
            runner.forward(tokens, on_layer)
            worst = max(entry["maxRelativeError"] for entry in report)
            print(json.dumps({"event": "reference-summary", "layers": args.check_layers,
                              "maxRelativeError": worst}), flush=True)
            if worst > 2e-2:
                raise SystemExit("Ramanujan hidden state diverged from the NumPy reference")
            return
        generated = []
        started = time.monotonic()
        directories = runner.forward(tokens)
        for step in range(args.max_new_tokens):
            token, logits = runner.head(directories[-1])
            if step == 0 and args.reference_token:
                reference = Qwen35Reference(runner.config, runner.shards)
                states = [reference.new_state(layer) for layer in range(runner.config.layers)]
                hidden = None
                for offset, prompt_token in enumerate(tokens):
                    hidden = reference.embed(prompt_token)
                    for layer in range(runner.config.layers):
                        hidden = reference.layer(layer, hidden, states[layer], offset)
                expected = reference.logits(hidden)
                print(json.dumps({"event": "reference-token", "ramanujan": token,
                                  "reference": int(np.argmax(expected)),
                                  "maxLogitError": float(np.max(np.abs(expected - logits)))}), flush=True)
            generated.append(token)
            print(json.dumps({"event": "token", "id": token,
                              "text": runner.tokenizer.decode([token], skip_special_tokens=False),
                              "seconds": round(time.monotonic() - started, 1)}), flush=True)
            if token in runner.eos or step == args.max_new_tokens - 1:
                break
            directories = runner.forward([token])
        print(json.dumps({"event": "completion", "prompt": args.prompt,
                          "text": runner.tokenizer.decode(generated, skip_special_tokens=False)}), flush=True)
    finally:
        print(json.dumps({"event": "worker-peak-rss", "bytes": runner.close()}), flush=True)


if __name__ == "__main__":
    main()
