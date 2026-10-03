"""Pipeline execution of a sharded GGUF model on the native LLM runtime (libramanujan_llm).

Consecutive pieces (embedding, layers, head) on the same shard form one stage. Locally every
stage is an in-process NativeStage. With --homelab the `rj worker` that owns a stage's shard
keeps the stage session (weights, KV cache and recurrent state) alive between tokens, so only
token ids and one hidden vector per token cross the network. A token goes through all stages
in one /llm/chain call: the homelab hands consecutive stages owned by the same worker to that
worker as a single task, and the head returns only the chosen token unless the logits are
needed. --check-layers needs every hidden state, so it calls /llm/step once per stage.
"""
import base64
import json
import os
import time
import urllib.error
import urllib.request
import uuid

import numpy as np

from converter.ramanujan_shards.llm_graph import coalesce_stages, graph_files, plan_stages, stage_graph
from converter.ramanujan_shards.llm_capacity import (describe_placement, merge_pieces, stage_capacity,
                                                      validate_placement, validate_plan)
from converter.ramanujan_shards.llm_package import load_model_from_args
from converter.ramanujan_shards.llm_tokenizer import load_tokenizer
from converter.ramanujan_shards.native_llm import NativeStage


def _post(url, route, body, timeout):
    headers = {"Content-Type": "application/json"}
    portal_token = os.environ.get("RAMANUJAN_PORTAL_TOKEN")
    if portal_token:
        if not url.startswith("https://") and not url.startswith(("http://localhost:", "http://127.0.0.1:")):
            raise ValueError("portal authentication requires HTTPS (except localhost)")
        headers["Authorization"] = "Bearer " + portal_token
    data = None if body is None else json.dumps(body).encode("utf-8")
    if data is None:
        del headers["Content-Type"]
    request = urllib.request.Request(url + route, data, headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise RuntimeError("homelab {0} failed with HTTP {1}: {2}".format(route, error.code, detail)) from None
    if payload.get("status") != "SUCCESS":
        raise RuntimeError("homelab {0} failed: {1}".format(route, payload.get("error", payload)))
    return payload


class HomelabStage:
    """One stage executed by the `rj worker` that owns its shard."""

    def __init__(self, url, shard_id, graph, timeout, plan_id=None):
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
        self.plan_id = plan_id
        self.closed = False

    def _post(self, route, body):
        if self.plan_id is not None:
            body = dict(body, planId=self.plan_id)
        return _post(self.url, route, body, self.timeout)

    def request(self):
        """This stage's part of a /llm/chain body; the graph and files go only with position 0."""
        if self.closed:
            raise RuntimeError("LLM stage is closed")
        stage = {"affinity": self.shard_id, "session": self.session}
        if self.position == 0:
            stage["graph"] = self.graph
            stage["files"] = graph_files(self.graph)
        return stage

    def step(self, tokens=None, hidden=None):
        if self.closed:
            raise RuntimeError("LLM stage is closed")
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
        if self.closed:
            return
        if self.plan_id is not None:
            # A failed or timed-out close must keep the pinned capacity reservation.
            self._post("/llm/close", {"affinity": self.shard_id, "session": self.session})
            self.closed = True
            return
        try:
            self._post("/llm/close", {"affinity": self.shard_id, "session": self.session})
        except (RuntimeError, OSError):
            pass


class HomelabChain:
    """Runs one token through every HomelabStage with a single /llm/chain call."""

    def __init__(self, stages):
        self.stages = stages
        self.url = stages[0].url
        self.timeout = stages[0].timeout
        self.last = {}
        plan_ids = {stage.plan_id for stage in stages}
        if len(plan_ids) != 1:
            raise ValueError("a homelab chain must use one capacity plan")
        self.plan_id = stages[0].plan_id

    def step(self, tokens, logits=False):
        """Returns the head's logits when `logits` is set, otherwise only the chosen token id."""
        n = len(tokens)
        body = {"stages": [stage.request() for stage in self.stages],
                "tokens": [int(token) for token in tokens], "n": n, "pos": self.stages[0].position,
                "timeout": int(self.timeout)}
        if self.plan_id is not None:
            body["planId"] = self.plan_id
        if not logits:
            body["output"] = "argmax"
        payload = _post(self.url, "/llm/chain", body, self.timeout)
        infos = payload.get("infos", [])
        for index, stage in enumerate(self.stages):
            if index < len(infos):
                stage.last = {"info": infos[index]}
            stage.position += n
        self.last = {key: payload[key] for key in ("tasks", "workers") if key in payload}
        if logits:
            return np.frombuffer(base64.b64decode(payload["output"]), "<f4")
        return int(payload["token"])


def on_chain(plan):
    """A chain needs the whole model: it starts from token ids and ends at the head."""
    return bool(plan) and plan[0]["embed"] and plan[-1]["head"]


class NativeGgufRunner:
    """Drop-in for GgufRunner (forward/head/close) backed by the native runtime."""

    def __init__(self, args):
        self.args = args
        self.capacity_aware = bool(getattr(args, "capacity_aware", False))
        if self.capacity_aware and not args.homelab:
            raise ValueError("--capacity-aware requires --homelab")
        self.plan_id = None
        self.closed = False
        self.spec, self.shards, self.tensors, metadata = load_model_from_args(args)
        self.tokenizer = load_tokenizer(metadata)
        self.stop_ids = set(self.tokenizer.stop_ids)
        last = args.stop_after_layer
        canonical = bool(self.shards) and all(shard.get("artifactLayout") == "per-layer"
                                             for shard in self.shards)
        if (self.capacity_aware and not args.check_layers
                and getattr(args, "capacity_placement", "search") == "vram"):
            self._place_by_vram(last)
        else:
            self._open_stages(last, canonical)
        self.chain = HomelabChain(self.stages) if args.homelab and on_chain(self.plan) else None
        self.position = 0
        self.logits = None

    def _open_stages(self, last, canonical):
        args = self.args
        self.plan = plan_stages(self.spec, self.tensors, last_layer=last,
                                split=(self.capacity_aware and canonical) or bool(args.check_layers))
        if self.capacity_aware and canonical and not args.check_layers:
            self.plan = coalesce_stages(self.plan, getattr(args, "capacity_max_stages", 8))
        self.stages = []
        run_id = uuid.uuid4().hex if self.capacity_aware else None
        for index, stage in enumerate(self.plan):
            graph = stage_graph(self.spec, self.tensors, stage, args.max_context, weights=args.weights,
                                stream_depth=args.stream_depth, stream_threads=args.stream_threads)
            shard_id = self.shards[stage["shard"]]["id"]
            started = time.monotonic()
            if args.homelab:
                affinity = "{0}-stage-{1}-{2}".format(shard_id, index, run_id) if run_id else shard_id
                executor = HomelabStage(args.homelab, affinity, graph, args.timeout)
            else:
                executor = NativeStage(graph)
            self.stages.append(executor)
            if args.verbose:
                info = executor.info() if not args.homelab else {}
                print(json.dumps({"event": "stage-open", "shard": shard_id, "embed": stage["embed"],
                                  "layers": stage["layers"], "head": stage["head"],
                                  "seconds": round(time.monotonic() - started, 2),
                                  "weights": info.get("weights"), "device": info.get("device")}), flush=True)
        if self.capacity_aware:
            self._reserve_capacity()

    def _place_by_vram(self, last):
        """The orchestrator shards whole layers by live VRAM: biggest device, biggest shard.

        Every embedding/layer/head piece is sent with its capacity estimate; /llm/plan returns
        contiguous runs merged into one session per device range, with their graphs and the
        resident/stream choice. Nothing is opened or downloaded before the placement is printed.
        """
        args = self.args
        if args.weights == "stream":
            raise ValueError("--capacity-placement vram sizes resident shards; use --weights auto or resident")
        dry_run = bool(getattr(args, "capacity_dry_run", False))
        pieces = plan_stages(self.spec, self.tensors, last_layer=last, split=True)
        run_id = uuid.uuid4().hex
        requested = []
        for index, piece in enumerate(pieces):
            graph = stage_graph(self.spec, self.tensors, piece, args.max_context, weights="auto",
                                stream_depth=args.stream_depth, stream_threads=args.stream_threads)
            affinity = "{0}-piece-{1}-{2}".format(self.shards[piece["shard"]]["id"], index, run_id)
            requested.append(stage_capacity(self.spec, self.tensors, graph, affinity,
                                            "{0}-{1}".format(affinity, uuid.uuid4().hex)))
        body = {"stages": requested, "weights": args.weights, "placement": "vram"}
        if dry_run:
            body["dryRun"] = True
        payload = _post(args.homelab.rstrip("/"), "/llm/plan", body, args.timeout)
        self.plan_id, groups = validate_placement(payload, requested, args.weights, dry_run)
        print(json.dumps({"event": "capacity-placement", "planId": self.plan_id,
                          "devices": describe_placement(groups, pieces)}), flush=True)
        if dry_run:
            raise SystemExit(0)
        self.plan = []
        self.stages = []
        for group in groups:
            start, end = group["pieces"]
            stage = merge_pieces(pieces[start:end])
            expected = stage_graph(self.spec, self.tensors, stage, args.max_context, weights=group["weights"],
                                   stream_depth=args.stream_depth, stream_threads=args.stream_threads)
            if json.loads(json.dumps(expected)) != group["graph"]:
                raise RuntimeError("capacity plan " + self.plan_id + " returned an unexpected graph; retained")
            executor = HomelabStage(args.homelab, group["affinity"], group["graph"], args.timeout, self.plan_id)
            # The orchestrator pinned this session id; the worker must open exactly it.
            executor.session = group["session"]
            executor.host_id = group["hostId"]
            self.plan.append(stage)
            self.stages.append(executor)
        if args.verbose:
            print(json.dumps({"event": "capacity-plan", "planId": self.plan_id,
                              "stages": [{key: group[key] for key in ("affinity", "hostId", "weights", "pieces")}
                                         for group in groups]}), flush=True)

    def _reserve_capacity(self):
        requested = [stage_capacity(self.spec, self.tensors, executor.graph,
                                    executor.shard_id, executor.session) for executor in self.stages]
        payload = _post(self.args.homelab.rstrip("/"), "/llm/plan",
                        {"stages": requested, "weights": self.args.weights}, self.args.timeout)
        self.plan_id, assignments = validate_plan(payload, requested, self.args.weights)
        for executor, assignment in zip(self.stages, assignments):
            executor.plan_id = self.plan_id
            executor.graph["weights"] = assignment["weights"]
            executor.host_id = assignment["hostId"]
        if self.args.verbose:
            print(json.dumps({"event": "capacity-plan", "planId": self.plan_id,
                              "stages": assignments}), flush=True)

    def forward(self, tokens, on_layer=None):
        if self.chain is not None and on_layer is None:
            return self._forward_chain(tokens)
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

    def _forward_chain(self, tokens):
        # Only --reference-token needs logits, and only for the first generated token.
        want_logits = bool(self.args.reference_token) and self.position == 0
        started = time.monotonic()
        out = self.chain.step(tokens, logits=want_logits)
        if self.args.verbose:
            for stage, executor in zip(self.plan, self.stages):
                info = executor.info()
                print(json.dumps({"event": "stage", "shard": self.shards[stage["shard"]]["id"],
                                  "layers": stage["layers"], "tokens": len(tokens),
                                  "deviceMs": round(info.get("last_step_ms", 0), 1),
                                  "loadWaitMs": round(info.get("last_wait_ms", 0), 1)}), flush=True)
            print(json.dumps({"event": "chain", "tokens": len(tokens),
                              "seconds": round(time.monotonic() - started, 3),
                              "tasks": self.chain.last.get("tasks"),
                              "workers": self.chain.last.get("workers")}), flush=True)
        self.position += len(tokens)
        self.logits = out if want_logits else None
        return [out]

    def head(self, logits):
        if isinstance(logits, int):
            # The homelab chain already picked the token.
            return logits, None
        if logits is None:
            raise ValueError("the stage plan has no output head")
        # Lowest index on ties, matching numpy.argmax.
        return int(np.argmax(logits)), logits

    def close(self):
        peaks = {}
        if self.closed:
            return peaks
        errors = []
        for executor in self.stages:
            try:
                executor.close()
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError("LLM close failed; capacity plan {0} retained: {1}".format(
                self.plan_id, "; ".join(errors)))
        if self.plan_id is not None:
            _post(self.args.homelab.rstrip("/"), "/llm/plan/release",
                  {"planId": self.plan_id}, self.args.timeout)
        self.closed = True
        return peaks
