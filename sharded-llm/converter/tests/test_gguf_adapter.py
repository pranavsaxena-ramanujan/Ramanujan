import json
import unittest

from ramanujan_shards.gguf_adapter import GGUFArchitectureAdapter


class FakeSource:
    def __init__(self):
        self.info = {"general.architecture": "llama", "llama.block_count": 4}
        self.tensors = {name: {"length": 18} for name in
                        ["token_embd.weight", "output_norm.weight", "output.weight"] +
                        ["blk.{0}.attn_q.weight".format(layer) for layer in range(4)]}

    def metadata(self):
        return self.info

    def tensor_names(self):
        return self.tensors.keys()

    def tensor_metadata(self, name):
        return self.tensors[name]


class GGUFAdapterTest(unittest.TestCase):
    def test_groups_any_architecture_without_dropping_tensors(self):
        source = FakeSource()
        graph = GGUFArchitectureAdapter(2).build_graph(source)
        self.assertEqual("llama", graph.architecture_id)
        self.assertEqual([(0, 2), (2, 4)], [
            (int(stage.metadata["layer_start"]), int(stage.metadata["layer_end"]))
            for stage in graph.stages
        ])
        assigned = [name for stage in graph.stages
                    for name in json.loads(stage.metadata["tensor_names"])]
        self.assertEqual(sorted(source.tensors), sorted(assigned))
        self.assertEqual(len(assigned), len(set(assigned)))

    def test_nonstandard_names_fall_back_to_tensor_groups(self):
        source = FakeSource()
        source.info = {"general.architecture": "custom"}
        source.tensors = {name: {"length": 18} for name in ("encoder.0", "encoder.1", "decoder.0")}
        graph = GGUFArchitectureAdapter(2).build_graph(source)
        self.assertEqual("0", graph.metadata["layer_count"])
        self.assertEqual(2, len(graph.stages))
        self.assertEqual(set(source.tensors), {name for stage in graph.stages
                         for name in json.loads(stage.metadata["tensor_names"])})

    def test_rejects_missing_declared_layer(self):
        source = FakeSource()
        source.tensors.pop("blk.2.attn_q.weight")
        with self.assertRaisesRegex(ValueError, "coverage"):
            GGUFArchitectureAdapter().build_graph(source)


if __name__ == "__main__":
    unittest.main()