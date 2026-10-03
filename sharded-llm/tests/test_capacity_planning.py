"""Capacity admission tests use only local fixture files and mocked HTTP/native calls."""
import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "converter"), str(ROOT / "converter" / "tests")]

from native_runner import HomelabChain, HomelabStage, NativeGgufRunner  # noqa: E402
from converter.ramanujan_shards.llm_capacity import (  # noqa: E402
    describe_placement,
    UPLOAD_STAGING_BYTES, stage_capacity, validate_placement, validate_plan)
from converter.ramanujan_shards.llm_graph import coalesce_stages, plan_stages, stage_graph  # noqa: E402
from converter.ramanujan_shards.llm_spec import QUANT, build_spec  # noqa: E402
from test_llm_programs import _Model, _attention_layout, _globals, _hyper  # noqa: E402
from run_gguf_shards import main, parse_args  # noqa: E402


class _CapacityFixture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.directory.name)
        self.model = self._model()
        self.spec = build_spec(self.model.metadata, self.model.tensors)

    def tearDown(self):
        self.directory.cleanup()

    def _model(self, arch="llama", layers=2, tied=False):
        layout = _globals(64, 40, tied=tied)
        for index in range(layers):
            layout.update(_attention_layout(index, 64, 4, 2, 16, 64))
        model = _Model(self.root, arch, _hyper(64, 4, 2, layers), layout, "gguf-f32")
        for tensor in model.tensors.values():
            tensor.update(bytes=tensor["file"].stat().st_size, shard=0)
        return model

    def _graph(self, stage=None, context=128, depth=2, threads=2):
        if stage is None:
            stage = plan_stages(self.spec, self.model.tensors)[0]
        return stage_graph(self.spec, self.model.tensors, stage, context,
                           stream_depth=depth, stream_threads=threads)

    def _budget(self, graph):
        return stage_capacity(self.spec, self.model.tensors, graph, "a", "s")


class CapacityTest(_CapacityFixture):
    def test_manifest_files_weight_bytes_and_stream_slots_are_exact(self):
        stage = plan_stages(self.spec, self.model.tensors, split=True)[1]
        budget = self._budget(self._graph(stage, depth=3, threads=2))
        layer_bytes = sum(tensor["bytes"] for name, tensor in self.model.tensors.items()
                          if name.startswith("blk.0."))
        self.assertEqual(budget["weightBytes"], layer_bytes)
        self.assertEqual(budget["streamWorkingBytes"], 4 * layer_bytes + 2 * UPLOAD_STAGING_BYTES)
        self.assertEqual(budget["stateBytes"], 8 * 2 * 128 * 2 * 16)
        self.assertEqual(set(budget), {"affinity", "session", "weightBytes", "streamWorkingBytes",
                                       "stateBytes", "scratchBytes", "files", "graph", "maxAllocationBytes",
                                       "streamItemBytes", "streamSharedBytes", "sharedScratchBytes"})
        self.assertEqual(budget["streamItemBytes"], 4 * layer_bytes)
        self.assertEqual(sum(item["bytes"] for item in budget["files"]), layer_bytes)
        self.assertTrue(all(set(item) == {"path", "bytes", "mtime"} for item in budget["files"]))
        self.assertTrue(all(item["mtime"] == Path(item["path"]).stat().st_mtime_ns // 1_000_000
                            for item in budget["files"]))

    def test_context_increases_state_and_workspace_not_weight_files(self):
        small = self._budget(self._graph(context=8))
        large = self._budget(self._graph(context=1024))
        self.assertEqual(large["weightBytes"], small["weightBytes"])
        self.assertEqual(large["streamWorkingBytes"], small["streamWorkingBytes"])
        self.assertEqual(large["stateBytes"], small["stateBytes"] * 128)
        self.assertGreater(large["scratchBytes"], small["scratchBytes"])
        native_workspace = 4 * (3 * 64 + 128 + 2 * 64 + 4 * 1024 + 128 + 64
                                + 1024 * 16 + 40)
        self.assertGreaterEqual(large["scratchBytes"], 2 * native_workspace)

    def test_embedding_streams_one_row_not_whole_table(self):
        stage = plan_stages(self.spec, self.model.tensors, split=True)[0]
        budget = self._budget(self._graph(stage))
        self.assertEqual(budget["weightBytes"], self.model.tensors["token_embd.weight"]["bytes"])
        self.assertEqual(budget["streamWorkingBytes"], 64 * 4)
        self.assertEqual(budget["stateBytes"], 0)
        self.assertEqual(budget["maxAllocationBytes"], 4 * 128 * 64)

    def test_graph_snapshot_and_single_buffer_limit_are_context_aware(self):
        graph = self._graph(context=1 << 20)
        budget = self._budget(graph)
        self.assertGreaterEqual(budget["maxAllocationBytes"], 4 * (1 << 20) * self.spec.kv_heads * self.spec.head_dim)
        self.assertGreaterEqual(budget["maxAllocationBytes"], 4 * (1 << 20) * self.spec.rope_dims)
        self.assertGreaterEqual(budget["maxAllocationBytes"], 4 * (1 << 20) * self.spec.dim)
        graph["weights"] = "stream"
        self.assertEqual(budget["graph"]["weights"], "auto")

    def test_supported_architectures_use_validated_shapes(self):
        for arch in ("llama", "qwen2", "qwen3", "phi3", "gemma", "qwen35"):
            with self.subTest(arch=arch):
                self.model = self._model(arch)
                self.spec = build_spec(self.model.metadata, self.model.tensors)
                self.assertGreater(self._budget(self._graph())["scratchBytes"], 0)

    def test_hybrid_recurrent_state_and_global_ssm_workspace(self):
        ssm = {"conv_kernel": 4, "state_size": 8, "group_count": 2,
               "time_step_rank": 4, "inner_size": 32}
        layout = _globals(64, 40)
        layout.update({
            "blk.0.attn_norm.weight": ([64], None),
            "blk.0.attn_qkv.weight": ([64, 64], None),
            "blk.0.attn_gate.weight": ([32, 64], None),
            "blk.0.ssm_beta.weight": ([4, 64], None),
            "blk.0.ssm_alpha.weight": ([4, 64], None),
            "blk.0.ssm_dt.bias": ([4], None), "blk.0.ssm_a": ([4], None),
            "blk.0.ssm_conv1d.weight": ([64, 4], "gguf-f32"),
            "blk.0.ssm_norm.weight": ([8], None), "blk.0.ssm_out.weight": ([64, 32], None),
            "blk.0.post_attention_norm.weight": ([64], None),
            "blk.0.ffn_gate.weight": ([64, 64], None),
            "blk.0.ffn_up.weight": ([64, 64], None),
            "blk.0.ffn_down.weight": ([64, 64], None)})
        layout.update(_attention_layout(1, 64, 4, 2, 16, 64, q_gate=True, qk_norm=True,
                                        norms=("post_attention_norm",)))
        metadata = _hyper(64, 4, 2, 2, **{"attention.key_length": 16, "rope.dimension_count": 8})
        metadata.update({"ssm." + name: value for name, value in ssm.items()})
        self.model = _Model(self.root, "qwen35", metadata, layout, "gguf-q4_1")
        for tensor in self.model.tensors.values():
            tensor.update(bytes=tensor["file"].stat().st_size, shard=0)
        self.spec = build_spec(self.model.metadata, self.model.tensors)
        graph = self._graph()
        budget = self._budget(graph)
        recurrent = 4 * 8 ** 2 + (2 * 2 * 8 + 32) * 3
        attention = 2 * 128 * 2 * 16
        self.assertEqual(budget["stateBytes"], 8 * (recurrent + attention))
        bigger = self._budget(self._graph(context=256))
        self.assertEqual(bigger["stateBytes"] - budget["stateBytes"], 8 * attention)
        graph["layers"] = []
        ssm_scratch = self._budget(graph)["scratchBytes"]
        self.spec.ssm = None
        self.assertGreater(ssm_scratch, self._budget(graph)["scratchBytes"])

    def test_all_supported_quantized_shapes_produce_manifest_byte_counts(self):
        layout = _globals(256, 16)
        layout.update(_attention_layout(0, 256, 4, 2, 64, 512))
        for encoding in QUANT:
            with self.subTest(encoding=encoding):
                self.model = _Model(self.root, "llama", _hyper(256, 4, 2, 1),
                                    layout, encoding)
                for tensor in self.model.tensors.values():
                    tensor.update(bytes=tensor["file"].stat().st_size, shard=0)
                self.spec = build_spec(self.model.metadata, self.model.tensors)
                self.assertEqual(self._budget(self._graph())["weightBytes"],
                                 sum(tensor["bytes"] for tensor in self.model.tensors.values()))

    def test_missing_stale_or_invalid_manifest_metadata_fails_closed(self):
        tensor = self.model.tensors["token_embd.weight"]
        for value in (None, True, -1, 0, tensor["bytes"] + 4):
            with self.subTest(value=value), patch.dict(tensor, {"bytes": value}):
                with self.assertRaises(ValueError):
                    self._budget(self._graph())
        with tensor["file"].open("ab") as stream:
            stream.write(b"x")
        with self.assertRaisesRegex(ValueError, "manifest/file size mismatch"):
            self._budget(self._graph())

    def test_unknown_architecture_mixer_and_context_are_not_guessed(self):
        self.spec.architecture = "custom"
        with self.assertRaisesRegex(ValueError, "unknown architecture"):
            self._budget(self._graph())
        self.spec.architecture = "llama"
        graph = self._graph()
        graph["layers"][0]["mixer"] = "mamba"
        with self.assertRaisesRegex(ValueError, "mixer"):
            self._budget(graph)
        for context in (0, -1, 1 << 32):
            with self.subTest(context=context), self.assertRaises(ValueError):
                self._budget(self._graph(context=context))


class CapacityRunnerTest(_CapacityFixture):
    def _args(self, **overrides):
        args = dict(capacity_aware=True, homelab="http://fixture.invalid/", timeout=1,
                    stop_after_layer=None, check_layers=None, max_context=128, weights="auto",
                    stream_depth=2, stream_threads=2, verbose=False, reference_token=False,
                    capacity_max_stages=8)
        args.update(overrides)
        return SimpleNamespace(**args)

    def _load(self):
        return (self.spec, [{"id": "shard-00", "artifactLayout": "per-layer"}],
                self.model.tensors, self.model.metadata)

    def _post(self, url, route, body, timeout):
        self.requests.append((route, copy.deepcopy(body)))
        if route == "/llm/plan":
            return {"status": "SUCCESS", "planId": "plan-1", "stages": [
                {"affinity": stage["affinity"], "session": stage["session"],
                 "hostId": "host-{0}".format(index // getattr(self, "host_group_size", 2)),
                 "weights": "stream" if self.requested_weights == "stream" or index % 2 else "resident"}
                for index, stage in enumerate(body["stages"])]}
        if route == "/llm/chain":
            if getattr(self, "chain_error", False):
                raise TimeoutError("chain still in flight")
            return {"status": "SUCCESS", "token": 7}
        if route == "/llm/close" and body["session"] == getattr(self, "failed_close", None):
            raise TimeoutError("close not acknowledged")
        if route == "/llm/plan/release" and getattr(self, "release_error", False):
            raise RuntimeError("release failed")
        return {"status": "SUCCESS"}

    def _runner(self, **overrides):
        self.requests = []
        self.requested_weights = overrides.get("weights", "auto")
        self.load_patch = patch("native_runner.load_model_from_args", return_value=self._load())
        self.tokenizer_patch = patch("native_runner.load_tokenizer", return_value=SimpleNamespace(stop_ids=[]))
        self.post_patch = patch("native_runner._post", side_effect=self._post)
        self.native_patch = patch("native_runner.NativeStage", side_effect=AssertionError("no live native resources"))
        for manager in (self.load_patch, self.tokenizer_patch, self.post_patch, self.native_patch):
            manager.start()
            self.addCleanup(manager.stop)
        return NativeGgufRunner(self._args(**overrides))

    def test_plan_is_only_request_before_open_and_modes_drive_graphs(self):
        runner = self._runner()
        self.assertEqual([route for route, _ in self.requests], ["/llm/plan"])
        route, body = self.requests[0]
        self.assertEqual(set(body), {"stages", "weights"})
        self.assertEqual(len(body["stages"]), 4)
        self.assertEqual(len({s["affinity"] for s in body["stages"]}), 4)
        self.assertEqual(len({s["session"] for s in body["stages"]}), 4)
        self.assertEqual([s.graph["weights"] for s in runner.stages],
                         ["resident", "stream", "resident", "stream"])
        self.assertEqual(runner.forward([1]), [7])
        chain = self.requests[-1][1]
        self.assertEqual(chain["planId"], "plan-1")
        self.assertEqual([s["graph"]["weights"] for s in chain["stages"]],
                         ["resident", "stream", "resident", "stream"])
        self.assertTrue(all(stage["graph"]["weights"] == "auto" for stage in body["stages"]))
        for requested, actual in zip(body["stages"], chain["stages"]):
            expected = copy.deepcopy(requested["graph"])
            expected["weights"] = actual["graph"]["weights"]
            self.assertEqual(actual["graph"], expected)
        self.assertEqual([s.position for s in runner.stages], [1, 1, 1, 1])
        runner.close()
        self.assertEqual([route for route, _ in self.requests][-5:],
                         ["/llm/close"] * 4 + ["/llm/plan/release"])
        self.assertTrue(all(body["planId"] == "plan-1" for _, body in self.requests[1:]))
        before = len(self.requests)
        runner.close()
        self.assertEqual(len(self.requests), before)

    def test_step_reset_close_carry_plan_id(self):
        stage = HomelabStage("http://fixture.invalid", "a",
                            {"hyper": {"dim": 1}, "embed": {"file": "/fixture"},
                             "layers": []}, 1, plan_id="p")
        with patch("native_runner._post", return_value={"status": "SUCCESS", "output": "AACAPw=="}) as post:
            stage.step(tokens=[1])
            stage.reset()
            stage.close()
        self.assertEqual([call.args[1] for call in post.call_args_list],
                         ["/llm/step", "/llm/close", "/llm/close"])
        self.assertTrue(all(call.args[2]["planId"] == "p" for call in post.call_args_list))

    def test_failed_close_attempts_all_stages_but_never_releases(self):
        runner = self._runner()
        self.failed_close = runner.stages[1].session
        with self.assertRaisesRegex(RuntimeError, "plan-1 retained"):
            runner.close()
        self.assertEqual([route for route, _ in self.requests][1:], ["/llm/close"] * 4)
        self.failed_close = None
        runner.close()
        self.assertEqual([route for route, _ in self.requests][-2:], ["/llm/close", "/llm/plan/release"])

    def test_inflight_error_does_not_release_without_confirmed_closes(self):
        runner = self._runner()
        self.chain_error = True
        with self.assertRaises(TimeoutError):
            runner.forward([1])
        self.assertNotIn("/llm/plan/release", [route for route, _ in self.requests])
        self.failed_close = runner.stages[0].session
        with self.assertRaises(RuntimeError):
            runner.close()
        self.assertNotIn("/llm/plan/release", [route for route, _ in self.requests])

    def test_release_failure_keeps_retryable_plan(self):
        runner = self._runner()
        self.release_error = True
        with self.assertRaisesRegex(RuntimeError, "release failed"):
            runner.close()
        self.assertFalse(runner.closed)
        self.release_error = False
        runner.close()
        self.assertEqual([route for route, _ in self.requests].count("/llm/close"), 4)
        self.assertTrue(runner.closed)

    def test_stream_request_has_no_combined_cluster_weight_limit(self):
        # A sparse, multi-GB embedding does not enter RAM and is read one row at a time.
        tensor = self.model.tensors["token_embd.weight"]
        tensor["shape"][0] = 1 << 24
        tensor["bytes"] = (1 << 24) * 64 * 4
        with tensor["file"].open("r+b") as stream:
            stream.truncate(tensor["bytes"])
        self.spec.vocab = 1 << 24
        runner = self._runner(weights="stream", stop_after_layer=1)
        body = self.requests[0][1]
        self.assertEqual(body["weights"], "stream")
        self.assertGreater(sum(s["weightBytes"] for s in body["stages"]), 4 << 30)
        self.assertLess(sum(s["streamWorkingBytes"] + s["scratchBytes"] + s["stateBytes"]
                            for s in body["stages"]), 1 << 30)
        self.assertLess(body["stages"][0]["maxAllocationBytes"], 1 << 20)
        self.assertTrue(all(stage.graph["weights"] == "stream" for stage in runner.stages))
        runner.close()

    def test_complete_model_can_exceed_two_devices_while_each_stream_stage_fits(self):
        # Admission is mocked: this checks the driver's contract, not idle-buffer
        # reuse in the native runtime. Sparse files never materialize full weights.
        layout = _globals(4096, 32768)
        for index in range(65):
            layout.update(_attention_layout(index, 4096, 32, 8, 128, 11008))
        tensors = {}
        for name, (shape, _) in layout.items():
            path = self.root / (name + ".bin")
            size = 4 * math.prod(shape)
            with path.open("wb") as stream:
                stream.truncate(size)
            tensors[name] = {"file": path, "shape": shape, "encoding": "gguf-f32",
                             "bytes": size, "shard": 0}
        metadata = {"general.architecture": "llama"}
        metadata.update({"llama." + name: value for name, value in _hyper(4096, 32, 8, 65).items()})
        self.model = SimpleNamespace(tensors=tensors, metadata=metadata)
        self.spec = build_spec(metadata, tensors)
        self.host_group_size = 4
        runner = self._runner(weights="stream", stream_depth=1, stream_threads=1)
        body = self.requests[0][1]
        self.assertEqual(len(body["stages"]), 8)
        self.assertGreater(sum(stage["weightBytes"] for stage in body["stages"]), 16 << 30)
        working = [stage["streamWorkingBytes"] + stage["stateBytes"] + stage["scratchBytes"]
                   for stage in body["stages"]]
        self.assertLess(sum(working), 16 << 30)
        for host in {stage.host_id for stage in runner.stages}:
            self.assertLess(sum(size for size, stage in zip(working, runner.stages)
                                if stage.host_id == host), 8 << 30)
        self.assertTrue(runner.plan[0]["embed"])
        self.assertTrue(runner.plan[-1]["head"])
        self.assertEqual(len({stage.host_id for stage in runner.stages}), 2)
        self.assertEqual(self.requests[0][0], "/llm/plan")
        runner.close()

    def test_canonical_65_layer_package_is_reserved_as_at_most_eight_sessions(self):
        self.model = self._model(layers=65)
        for name, tensor in self.model.tensors.items():
            tensor["shard"] = (int(name.split(".")[1]) if name.startswith("blk.")
                               else 0 if name == "token_embd.weight" else 64)
        self.spec = build_spec(self.model.metadata, self.model.tensors)
        self.host_group_size = 99
        shards = [{"id": "shard-" + str(index), "artifactLayout": "per-layer"} for index in range(65)]
        with patch.object(self, "_load", return_value=(self.spec, shards, self.model.tensors, self.model.metadata)):
            runner = self._runner(weights="stream", stream_depth=1, stream_threads=1)
        requested = self.requests[0][1]["stages"]
        self.assertEqual(len(requested), 8)
        layers = [layer["index"] for stage in requested for layer in stage["graph"]["layers"]]
        self.assertEqual(layers, list(range(65)))
        self.assertIn("embed", requested[0]["graph"])
        self.assertIn("head", requested[-1]["graph"])
        self.assertEqual(len({stage.host_id for stage in runner.stages}), 1)
        self.assertEqual(runner.forward([1]), [7])
        runner.close()
        self.assertEqual([route for route, _ in self.requests].count("/llm/close"), 8)

    def test_check_layers_keeps_individual_graphs_even_with_grouping_limit(self):
        runner = self._runner(check_layers=2, capacity_max_stages=1)
        self.assertEqual(len(self.requests[0][1]["stages"]), 4)
        self.assertTrue(all(len(stage.graph["layers"]) <= 1 for stage in runner.stages))
        runner.close()

    def test_grouping_is_configurable_and_never_changes_reserved_bindings(self):
        fine = plan_stages(self.spec, self.model.tensors, split=True)
        self.assertEqual(coalesce_stages(fine, 0), fine)
        for limit in (-1, True, None):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                coalesce_stages(fine, limit)
        grouped = coalesce_stages(fine, 1)
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0]["layers"], [0, 1])
        runner = self._runner(weights="stream", capacity_max_stages=1)
        self.assertEqual(len(self.requests[0][1]["stages"]), 1)
        runner.forward([1])
        reserved = self.requests[0][1]["stages"][0]
        executed = self.requests[1][1]["stages"][0]
        self.assertEqual((executed["affinity"], executed["session"]),
                         (reserved["affinity"], reserved["session"]))
        runner.close()

    def test_default_does_not_call_plan_or_release_and_preserves_stage_grouping(self):
        runner = self._runner(capacity_aware=False)
        self.assertEqual(self.requests, [])
        self.assertEqual(len(runner.stages), 1)
        self.assertEqual(runner.stages[0].shard_id, "shard-00")
        runner.close()
        self.assertEqual([route for route, _ in self.requests], ["/llm/close"])
        self.assertNotIn("planId", self.requests[0][1])

    def test_grouped_package_keeps_coarse_graphs_under_capacity_opt_in(self):
        loaded = (self.spec, [{"id": "shard-00"}], self.model.tensors, self.model.metadata)
        with patch.object(self, "_load", return_value=loaded):
            runner = self._runner()
        self.assertEqual(len(self.requests[0][1]["stages"]), 1)
        self.assertEqual(len(runner.stages), 1)
        graph = runner.stages[0].graph
        self.assertIn("embed", graph)
        self.assertIn("head", graph)
        self.assertEqual(len(graph["layers"]), 2)
        runner.close()

    def test_admission_failure_does_not_open_or_release_sessions(self):
        self.requests = []
        with patch("native_runner.load_model_from_args", return_value=self._load()), \
                patch("native_runner.load_tokenizer", return_value=SimpleNamespace(stop_ids=[])), \
                patch("native_runner._post", side_effect=RuntimeError("insufficient capacity")) as post, \
                patch("native_runner.NativeStage", side_effect=AssertionError("live resource")):
            with self.assertRaisesRegex(RuntimeError, "insufficient capacity"):
                NativeGgufRunner(self._args())
        self.assertEqual([call.args[1] for call in post.call_args_list], ["/llm/plan"])

    def test_invalid_plan_does_not_open_or_release_sessions(self):
        with patch.object(self, "_post", return_value={"status": "SUCCESS", "planId": "retained",
                                                       "stages": []}) as post:
            with self.assertRaisesRegex(RuntimeError, "retained"):
                self._runner()
        self.assertEqual([call.args[1] for call in post.call_args_list], ["/llm/plan"])

    def test_bad_manifest_fails_before_admission(self):
        del self.model.tensors["token_embd.weight"]["bytes"]
        with self.assertRaisesRegex(ValueError, "tensorFiles.bytes"):
            self._runner()
        self.assertEqual(self.requests, [])

    def test_invalid_prompt_after_admission_closes_before_releasing(self):
        runner = self._runner()
        runner.args.runtime = "native"
        runner.args.prompt_turns = None
        runner.args.prompt = "fixture"
        runner.args.max_new_tokens = 129
        runner.tokenizer.encode = lambda prompt: [1]
        with patch("run_gguf_shards.parse_args", return_value=runner.args), \
                patch("native_runner.NativeGgufRunner", return_value=runner), patch("builtins.print"):
            with self.assertRaisesRegex(SystemExit, "fit in --max-context"):
                main()
        self.assertEqual([route for route, _ in self.requests],
                         ["/llm/plan"] + ["/llm/close"] * 4 + ["/llm/plan/release"])

    def test_tokenizer_failure_after_admission_closes_before_releasing(self):
        runner = self._runner()
        runner.args.runtime = "native"
        runner.args.prompt_turns = None
        runner.args.prompt = "fixture"
        runner.args.max_new_tokens = 8
        runner.tokenizer.encode = Mock(side_effect=ValueError("bad tokenization"))
        with patch("run_gguf_shards.parse_args", return_value=runner.args), \
                patch("native_runner.NativeGgufRunner", return_value=runner), patch("builtins.print"):
            with self.assertRaisesRegex(ValueError, "bad tokenization"):
                main()
        self.assertEqual(self.requests[-1][0], "/llm/plan/release")


def _merged(pieces, start, end, host, weights):
    """What CapacityDao.VramPlacement returns for pieces[start:end] on one device."""
    graph = copy.deepcopy(pieces[start]["graph"])
    for key in ("embed", "head", "rope_freqs"):
        graph.pop(key, None)
    graph["layers"] = []
    for piece in pieces[start:end]:
        for key in ("embed", "head", "rope_freqs"):
            if key in piece["graph"]:
                graph[key] = piece["graph"][key]
        graph["layers"].extend(piece["graph"]["layers"])
    graph["weights"] = weights
    return {"affinity": pieces[start]["affinity"], "session": pieces[start]["session"], "hostId": host,
            "weights": weights, "pieces": [start, end], "graph": graph,
            "weightBytes": sum(piece["weightBytes"] for piece in pieces[start:end]), "deviceBytes": 1,
            "gpuBudgetBytes": 2, "gpuTotalBytes": 3, "unifiedMemory": False}


class VramPlacementTest(_CapacityFixture):
    def test_pieces_carry_parts_that_sum_to_their_budgets(self):
        for piece in plan_stages(self.spec, self.model.tensors, split=True):
            budget = self._budget(self._graph(piece))
            self.assertEqual(budget["streamItemBytes"] + budget["streamSharedBytes"], budget["streamWorkingBytes"])
            self.assertLessEqual(budget["sharedScratchBytes"], budget["scratchBytes"])
        whole = self._budget(self._graph(plan_stages(self.spec, self.model.tensors)[0]))
        pieces = [self._budget(self._graph(piece)) for piece in plan_stages(self.spec, self.model.tensors, split=True)]
        # The orchestrator's merged bound never understates one real merged session.
        merged = (sum(p["scratchBytes"] - p["sharedScratchBytes"] for p in pieces)
                  + max(p["sharedScratchBytes"] for p in pieces))
        self.assertGreaterEqual(merged, whole["scratchBytes"])
        self.assertEqual(sum(p["stateBytes"] for p in pieces), whole["stateBytes"])

    def test_description_shows_what_each_device_must_download(self):
        pieces = plan_stages(self.spec, self.model.tensors, split=True)
        groups = [{"hostId": "mac", "weights": "resident", "pieces": [0, 1], "weightBytes": 5, "downloadBytes": 0},
                  {"hostId": "t4", "weights": "resident", "pieces": [1, len(pieces)], "weightBytes": 7,
                   "downloadBytes": 7}]
        described = describe_placement(groups, pieces)
        self.assertEqual([device["downloadBytes"] for device in described], [0, 7])
        self.assertTrue(described[0]["embed"])
        self.assertTrue(described[1]["head"])

    def test_placement_must_cover_every_piece_contiguously(self):
        requested = [{"affinity": "a" + str(i), "session": "s" + str(i), "weightBytes": 1,
                      "graph": {"layers": [], "weights": "auto"}} for i in range(4)]
        good = {"status": "SUCCESS", "planId": "p", "stages": [
            _merged(requested, 0, 3, "big", "resident"), _merged(requested, 3, 4, "small", "resident")]}
        self.assertEqual(validate_placement(good, requested, "auto")[0], "p")
        dry = dict(good, planId=None)
        self.assertEqual(len(validate_placement(dry, requested, "auto", dry_run=True)[1]), 2)
        bad = [
            dict(good, stages=good["stages"][:1]),
            dict(good, stages=[_merged(requested, 0, 2, "big", "resident"), good["stages"][1]]),
            dict(good, stages=[_merged(requested, 0, 1, "a", "resident"), _merged(requested, 1, 3, "b", "resident"),
                               _merged(requested, 3, 4, "a", "resident")]),
            dict(good, stages=[dict(good["stages"][0], session="x"), good["stages"][1]]),
            dict(good, stages=[dict(good["stages"][0], weights="stream"), good["stages"][1]]),
            dict(good, planId=None),
        ]
        for payload in bad:
            with self.assertRaises(RuntimeError):
                validate_placement(payload, requested, "auto")
        streamed = dict(good, stages=[_merged(requested, 0, 3, "big", "stream"), good["stages"][1]])
        with self.assertRaisesRegex(RuntimeError, "--weights"):
            validate_placement(streamed, requested, "resident")


class VramRunnerTest(_CapacityFixture):
    _args = CapacityRunnerTest._args
    _load = CapacityRunnerTest._load
    _runner = CapacityRunnerTest._runner

    def _post(self, url, route, body, timeout):
        if route == "/llm/plan":
            self.requests.append((route, copy.deepcopy(body)))
            if self.groups is None:
                raise RuntimeError("homelab /llm/plan failed: Connected devices lack fresh capacity")
            stages = [_merged(body["stages"], *group) for group in self.groups]
            return {"status": "SUCCESS", "planId": None if body.get("dryRun") else "plan-v", "stages": stages}
        return CapacityRunnerTest._post(self, url, route, body, timeout)

    def test_orchestrator_places_pieces_and_runner_opens_one_session_per_run(self):
        self.groups = [(0, 3, "big", "resident"), (3, 4, "small", "stream")]
        runner = self._runner(capacity_placement="vram")
        self.assertEqual([route for route, _ in self.requests], ["/llm/plan"])
        body = self.requests[0][1]
        self.assertEqual(body["placement"], "vram")
        self.assertNotIn("dryRun", body)
        self.assertEqual(len(body["stages"]), 4)
        for stage in body["stages"]:
            self.assertNotIn("hostId", stage)
            self.assertEqual(stage["streamItemBytes"] + stage["streamSharedBytes"], stage["streamWorkingBytes"])
        self.assertEqual([(s.shard_id, s.session) for s in runner.stages],
                         [(body["stages"][0]["affinity"], body["stages"][0]["session"]),
                          (body["stages"][3]["affinity"], body["stages"][3]["session"])])
        self.assertEqual([s.graph["weights"] for s in runner.stages], ["resident", "stream"])
        self.assertEqual([(p["embed"], p["layers"], p["head"]) for p in runner.plan],
                         [(True, [0, 1], False), (False, [], True)])
        self.assertEqual(runner.forward([1]), [7])
        self.assertEqual(self.requests[-1][1]["planId"], "plan-v")
        runner.close()
        self.assertEqual(self.requests[-1][0], "/llm/plan/release")

    def test_dry_run_prints_shards_without_reserving_or_opening(self):
        self.groups = [(0, 4, "big", "resident")]
        with patch("builtins.print") as printed, self.assertRaises(SystemExit):
            self._runner(capacity_placement="vram", capacity_dry_run=True)
        self.assertEqual([route for route, _ in self.requests], ["/llm/plan"])
        self.assertTrue(self.requests[0][1]["dryRun"])
        self.assertIn("capacity-placement", printed.call_args.args[0])
        self.assertIn('"layerCount": 2', printed.call_args.args[0])

    def test_no_fitting_device_fails_before_opening(self):
        self.groups = None
        with self.assertRaisesRegex(RuntimeError, "lack fresh capacity"):
            self._runner(capacity_placement="vram")
        self.assertEqual([route for route, _ in self.requests], ["/llm/plan"])

    def test_unexpected_graph_is_rejected_before_opening(self):
        self.groups = [(0, 4, "big", "resident")]
        original = _merged

        def tampered(*args):
            result = original(*args)
            result["graph"]["max_context"] = 1
            return result
        with patch(__name__ + "._merged", side_effect=tampered), self.assertRaisesRegex(RuntimeError, "unexpected graph"):
            self._runner(capacity_placement="vram")
        self.assertEqual([route for route, _ in self.requests], ["/llm/plan"])


class PlanValidationTest(unittest.TestCase):
    def test_invalid_plans_reject_without_releasing_allocations(self):
        requested = [{"affinity": "a" + str(i), "session": "s" + str(i)} for i in range(3)]
        valid = {"status": "SUCCESS", "planId": "p", "stages": [
            dict(stage, hostId="h" + str(i), weights="stream") for i, stage in enumerate(requested)]}
        self.assertEqual(validate_plan(valid, requested, "auto")[0], "p")
        bad = []
        for field in ("planId", "stages"):
            payload = copy.deepcopy(valid)
            del payload[field]
            bad.append(payload)
        for field, value in (("weights", "auto"), ("hostId", ""), ("session", "wrong"), ("affinity", "wrong")):
            payload = copy.deepcopy(valid)
            payload["stages"][0][field] = value
            bad.append(payload)
        payload = copy.deepcopy(valid)
        payload["stages"][-1]["hostId"] = payload["stages"][0]["hostId"]
        bad.append(payload)
        payload = copy.deepcopy(valid)
        payload["stages"].reverse()
        bad.append(payload)
        for payload in bad:
            with self.subTest(payload=payload), self.assertRaises(RuntimeError):
                validate_plan(payload, requested, "auto")
        with self.assertRaisesRegex(RuntimeError, "honor --weights"):
            validate_plan(valid, requested, "resident")

    def test_cli_default_and_opt_in_validation(self):
        common = ["runner", "--package", "fixture", "--metadata", "fixture"]
        with patch.object(sys, "argv", common + ["--runtime", "native"]):
            args = parse_args()
            self.assertFalse(args.capacity_aware)
            self.assertEqual(args.capacity_max_stages, 8)
        with patch.object(sys, "argv", common + ["--capacity-aware", "--runtime", "native",
                                               "--homelab", "http://fixture.invalid"]):
            self.assertTrue(parse_args().capacity_aware)
        for extra in (["--runtime", "native"], ["--runtime", "dsl", "--work-dir", "fixture",
                                              "--homelab", "http://fixture.invalid"]):
            with patch.object(sys, "argv", common + ["--capacity-aware"] + extra), \
                    patch("sys.stderr"), self.assertRaises(SystemExit):
                parse_args()


if __name__ == "__main__":
    unittest.main()
