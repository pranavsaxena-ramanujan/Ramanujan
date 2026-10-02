"""Pipeline execution of a sharded GGUF model on the native LLM runtime (libramanujan_llm).

Consecutive pieces (embedding, layers, head) on the same shard form one stage. Locally every
stage is an in-process NativeStage; with --homelab each stage call is POSTed to `rj homelab`
(/llm/step), whose `rj worker` for that shard keeps the stage session (weights, KV cache and
recurrent state) alive between tokens, so only token ids and one hidden vector per token
cross the network.
"""
import base64
import json
import time
import urllib.error
import urllib.request
import uuid

import numpy as np

from converter.ramanujan_shards.llm_graph import graph_files, plan_stages, stage_graph
from converter.ramanujan_shards.llm_package import load_model_from_args
from converter.ramanujan_shards.llm_tokenizer import load_tokenizer
from converter.ramanujan_shards.native_llm import NativeStage


class HomelabStage:
    """One stage executed by the `rj worker` that owns its shard."""

    def __init__(self, url, shard_id, graph, timeout):
        self.url = url.rstrip("/")
        self.shard_id = shard_id
        self.graph = graph
        self.embed = "embed" in graph
        self.head = "head" in graph
        self.dim = graph["hyper"]["dim"]
        self.timeout = timeout
        self.session = "{0}-{1}".format(shard_id, uuid.uuid4().hex)
        self.position = 0
        self.last = {}

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
            raise RuntimeError("homelab {0} failed: {1}".format(route, payload.get("error", payload)))
        return payload

    def step(self, tokens=None, hidden=None):
        body = {"affinity": self.shard_id, "session": self.session, "pos": self.position,
                "timeout": int(self.timeout)}
        if self.position == 0:
            # The worker opens (or reopens) the session only at position 0; later steps reuse it.
            body["graph"] = self.graph
            body["files"] = graph_files(self.graph)
        if self.embed:
            body["tokens"] = [int(token) for token in tokens]
            n = len(body["tokens"])
        else:
            states = np.ascontiguousarray(hidden, dtype="<f4").reshape(-1, self.dim)
            body["hidden"] = base64.b64encode(states.tobytes()).decode("ascii")
            n = len(states)
        body["n"] = n
        payload = self._post("/llm/step", body)
        out = np.frombuffer(base64.b64decode(payload["output"]), "<f4")
        self.last = {key: payload[key] for key in ("worker", "info") if key in payload}
        self.position += n
        return out if self.head else out.reshape(n, self.dim)

    def reset(self):
        self._post("/llm/close", {"affinity": self.shard_id, "session": self.session})
        self.position = 0

    def info(self):
        return self.last.get("info", {})

    def close(self):
        try:
            self._post("/llm/close", {"affinity": self.shard_id, "session": self.session})
        except (RuntimeError, OSError):
            pass


class NativeGgufRunner:
    """Drop-in for GgufRunner (forward/head/close) backed by the native runtime."""

    def __init__(self, args):
        self.args = args
        self.spec, self.shards, self.tensors, metadata = load_model_from_args(args)
        self.tokenizer = load_tokenizer(metadata)
        self.stop_ids = set(self.tokenizer.stop_ids)
        last = args.stop_after_layer
        self.plan = plan_stages(self.spec, self.tensors, last_layer=last, split=bool(args.check_layers))
        self.stages = []
        for stage in self.plan:
            graph = stage_graph(self.spec, self.tensors, stage, args.max_context, weights=args.weights,
                                stream_depth=args.stream_depth, stream_threads=args.stream_threads)
            shard_id = self.shards[stage["shard"]]["id"]
            started = time.monotonic()
            if args.homelab:
                executor = HomelabStage(args.homelab, shard_id, graph, args.timeout)
            else:
                executor = NativeStage(graph)
            self.stages.append(executor)
            if args.verbose:
                info = executor.info() if not args.homelab else {}
                print(json.dumps({"event": "stage-open", "shard": shard_id, "embed": stage["embed"],
                                  "layers": stage["layers"], "head": stage["head"],
                                  "seconds": round(time.monotonic() - started, 2),
                                  "weights": info.get("weights"), "device": info.get("device")}), flush=True)
        self.position = 0
        self.logits = None

    def forward(self, tokens, on_layer=None):
        hidden = None
        for stage, executor in zip(self.plan, self.stages):
            started = time.monotonic()
            if executor.embed:
                out = executor.step(tokens=tokens)
            else:
                out = executor.step(hidden=hidden)
            if executor.head:
                self.logits = out
            else:
                hidden = out
            if self.args.verbose:
                info = executor.info()
                print(json.dumps({"event": "stage", "shard": self.shards[stage["shard"]]["id"],
                                  "layers": stage["layers"], "tokens": len(tokens),
                                  "seconds": round(time.monotonic() - started, 3),
                                  "deviceMs": round(info.get("last_step_ms", 0), 1),
                                  "loadWaitMs": round(info.get("last_wait_ms", 0), 1)}), flush=True)
            if on_layer is not None and not executor.head:
                if stage["embed"] and not stage["layers"]:
                    on_layer(-1, list(out))
                elif stage["layers"]:
                    on_layer(stage["layers"][-1], list(out))
        self.position += len(tokens)
        return [self.logits]

    def head(self, logits):
        if logits is None:
            raise ValueError("the stage plan has no output head")
        # Lowest index on ties, matching numpy.argmax.
        return int(np.argmax(logits)), logits

    def close(self):
        peaks = {}
        for executor in self.stages:
            executor.close()
        return peaks
