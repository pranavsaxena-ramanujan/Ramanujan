"""Parity of the native OpenCL LLM runtime (libramanujan_llm) with the NumPy reference.

Runs the same synthetic model zoo as test_llm_programs (llama, qwen2, qwen3, phi3, gemma,
an unknown architecture, qwen35 Gated DeltaNet, K-quants). Skipped when the library is not
built or no OpenCL device is available.
"""
import unittest

import numpy as np

import test_llm_programs as programs
from ramanujan_shards.llm_graph import graph_files, plan_stages, stage_graph
from ramanujan_shards.llm_reference import LlmReference
from ramanujan_shards.llm_spec import build_spec
from ramanujan_shards.native_llm import NativeStage, find_library

_AVAILABLE = None


def _available():
    global _AVAILABLE
    if _AVAILABLE is None:
        _AVAILABLE = find_library() is not None
    return _AVAILABLE


def _close(test, actual, expected, what):
    scale = max(1e-6, float(np.max(np.abs(expected))))
    error = float(np.max(np.abs(np.asarray(actual) - expected))) / scale
    test.assertLess(error, 2e-4, "{0}: relative error {1}".format(what, error))


@unittest.skipUnless(_available(), "libramanujan_llm is not built")
class NativeRuntimeParityTest(programs.GenericProgramSimulationTest):
    weights = "resident"

    def check(self, model, tokens=(3, 7), context=8, overrides=None):
        spec = build_spec(model.metadata, model.tensors, overrides)
        try:
            self._per_piece(spec, model.tensors, tokens, context)
        except RuntimeError as error:
            if "no OpenCL" in str(error) or "OpenCL library" in str(error):
                self.skipTest(str(error))
            raise
        self._whole_model(spec, model.tensors, tokens, context)
        return spec

    def _open(self, spec, tensors, stage, context):
        return NativeStage(stage_graph(spec, tensors, stage, context, weights=self.weights))

    def _per_piece(self, spec, tensors, tokens, context):
        """Every piece in its own stage: compare each layer's hidden output per token."""
        reference = LlmReference(spec, tensors)
        stages = [self._open(spec, tensors, stage, context) for stage in plan_stages(spec, tensors, split=True)]
        states = [reference.new_state(layer) for layer in range(len(spec.layers))]
        try:
            for pos, token in enumerate(tokens):
                expected = reference.embed(token)
                hidden = stages[0].step(tokens=[token])
                _close(self, hidden[0], expected, "embed")
                for layer in range(len(spec.layers)):
                    expected = reference.layer(layer, expected, states[layer], pos)
                    hidden = stages[1 + layer].step(hidden=hidden)
                    _close(self, hidden[0], expected, "token {0} layer {1}".format(pos, layer))
                logits = stages[-1].step(hidden=hidden)
                expected_logits = reference.logits(expected)
                _close(self, logits, expected_logits, "logits")
                self.assertEqual(int(np.argmax(expected_logits)), int(np.argmax(logits)))
        finally:
            for stage in stages:
                stage.close()

    def _whole_model(self, spec, tensors, tokens, context):
        """One stage: batched prefill, one decode step, position checks and reset."""
        reference = LlmReference(spec, tensors)
        states = [reference.new_state(layer) for layer in range(len(spec.layers))]
        expected = []
        sequence = list(tokens) + [5]
        for pos, token in enumerate(sequence):
            x = reference.embed(token)
            for layer in range(len(spec.layers)):
                x = reference.layer(layer, x, states[layer], pos)
            expected.append(reference.logits(x))
        (stage,) = plan_stages(spec, tensors)
        with self._open(spec, tensors, stage, context) as native:
            _close(self, native.step(tokens=list(tokens)), expected[len(tokens) - 1], "prefill logits")
            _close(self, native.step(tokens=[5]), expected[-1], "decode logits")
            native.position = 0
            with self.assertRaisesRegex(RuntimeError, "does not match session position"):
                native.step(tokens=[1])
            native.reset()
            _close(self, native.step(tokens=list(tokens)), expected[len(tokens) - 1], "logits after reset")
            info = native.info()
            self.assertEqual(self.weights, info["weights"])
            self.assertEqual(len(tokens), info["position"])


@unittest.skipUnless(_available(), "libramanujan_llm is not built")
class NativeRuntimeStreamingParityTest(NativeRuntimeParityTest):
    weights = "stream"


class StagePlanTest(unittest.TestCase):
    def test_consecutive_pieces_on_one_shard_merge_and_tied_head_returns_to_embedding_shard(self):
        class _Layer:
            def __init__(self, index, shard):
                self.index = index
                self.roles = {"attn_norm": "blk.{0}.attn_norm.weight".format(index)}
                self.kind = "attn0"
        class _Spec:
            embed_tensor = "token_embd.weight"
            head_roles = {"output_norm": "output_norm.weight", "output": "token_embd.weight"}
            layers = [_Layer(0, 0), _Layer(1, 1), _Layer(2, 1)]
        tensors = {"token_embd.weight": {"shard": 0}, "output_norm.weight": {"shard": 1},
                   "blk.0.attn_norm.weight": {"shard": 0}, "blk.1.attn_norm.weight": {"shard": 1},
                   "blk.2.attn_norm.weight": {"shard": 1}}
        stages = plan_stages(_Spec, tensors)
        self.assertEqual([(0, True, [0], False), (1, False, [1, 2], False), (0, False, [], True)],
                         [(s["shard"], s["embed"], s["layers"], s["head"]) for s in stages])
        self.assertEqual(5, len(plan_stages(_Spec, tensors, split=True)))
        self.assertEqual([(0, True, [0], False), (1, False, [1], False)],
                         [(s["shard"], s["embed"], s["layers"], s["head"])
                          for s in plan_stages(_Spec, tensors, last_layer=1)])

    def test_graph_files_lists_each_weight_once(self):
        graph = {"embed": {"file": "/a"}, "layers": [{"tensors": {"x": {"file": "/b"}, "y": {"file": "/a"}}}],
                 "head": {"output": {"file": "/a"}, "output_norm": {"file": "/c"}}}
        self.assertEqual(["/a", "/b", "/c"], graph_files(graph))


if __name__ == "__main__":
    unittest.main()
