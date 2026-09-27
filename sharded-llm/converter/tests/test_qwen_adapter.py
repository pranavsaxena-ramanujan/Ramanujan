import unittest

from ramanujan_shards import plan_shards
from ramanujan_shards.qwen_adapter import Qwen38ArchitectureAdapter


class FakeQwenSource:
    def __init__(self, layers=8):
        self.info = {
            "general.architecture": "qwen35",
            "qwen35.block_count": layers,
            "qwen35.embedding_length": 5120,
        }
        self.tensors = {name: {"length": 18} for name in
                        ("token_embd.weight", "output_norm.weight", "output.weight")}
        self.tensors.update({"blk.{0}.attn_q.weight".format(index): {"length": 18}
                             for index in range(layers)})

    def metadata(self):
        return self.info

    def tensor_names(self):
        return self.tensors.keys()

    def tensor_metadata(self, name):
        return self.tensors[name]


class QwenAdapterTest(unittest.TestCase):
    def test_balanced_contiguous_stages_and_plan(self):
        source = FakeQwenSource()
        graph = Qwen38ArchitectureAdapter(4).build_graph(source)
        self.assertEqual("qwen35", graph.architecture_id)
        self.assertEqual([(0, 2), (2, 4), (4, 6), (6, 8)], [
            (int(stage.metadata["layer_start"]), int(stage.metadata["layer_end"]))
            for stage in graph.stages
        ])
        self.assertEqual("linear_attention,full_attention", graph.stages[1].metadata["layer_types"])
        self.assertEqual(54, graph.stages[0].estimated_resident_bytes)
        self.assertEqual(72, graph.stages[-1].estimated_resident_bytes)
        self.assertEqual(4, len(plan_shards(graph, 4)))

    def test_rejects_missing_or_unassigned_tensors(self):
        source = FakeQwenSource()
        source.tensors.pop("blk.3.attn_q.weight")
        with self.assertRaisesRegex(ValueError, "missing Qwen layer: 3"):
            Qwen38ArchitectureAdapter().build_graph(source)
        source.tensors["blk.3.attn_q.weight"] = {"length": 18}
        source.tensors["vision.weight"] = {"length": 18}
        with self.assertRaisesRegex(ValueError, "unassigned GGUF tensor"):
            Qwen38ArchitectureAdapter().build_graph(source)

    def test_detects_auxiliary_block_and_actual_interval_key(self):
        source = FakeQwenSource(5)
        source.info["qwen35.full_attention_interval"] = 2
        source.tensors["blk.4.nextn.eh_proj.weight"] = {"length": 18}
        graph = Qwen38ArchitectureAdapter(2).build_graph(source)
        self.assertEqual("4", graph.metadata["decoder_layer_count"])
        self.assertEqual("auxiliary", graph.stages[-1].metadata["layer_types"].split(",")[-1])
        self.assertIn("full_attention", graph.stages[0].metadata["layer_types"])


if __name__ == "__main__":
    unittest.main()