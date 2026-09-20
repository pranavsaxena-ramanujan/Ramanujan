import json
import struct
import tempfile
import unittest
from pathlib import Path

from ramanujan_shards import Phi3ArchitectureAdapter, SafeTensorsSourceReader


class SafeTensorsPhi3Test(unittest.TestCase):
    def test_reads_only_requested_tensor_slice(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            tensors = {
                "first": ("F32", [2], struct.pack("<ff", 1.0, 2.0)),
                "second": ("F32", [1], struct.pack("<f", 3.0)),
            }
            _write_model(model_dir, {}, tensors)
            reader = SafeTensorsSourceReader(model_dir)

            with reader.open_tensor("second") as stream:
                self.assertEqual(struct.pack("<f", 3.0), stream.read())
                self.assertEqual(b"", stream.read())
            self.assertEqual([1], reader.tensor_metadata("second")["shape"])

    def test_phi3_adapter_builds_metadata_driven_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            config = {
                "_name_or_path": "tiny-phi3",
                "model_type": "phi3",
                "num_hidden_layers": 2,
                "hidden_size": 12,
                "vocab_size": 32,
            }
            tensors = {
                "model.embed_tokens.weight": ("F32", [32, 12], b""),
                "model.norm.weight": ("F32", [12], b""),
                "lm_head.weight": ("F32", [32, 12], b""),
            }
            for layer in range(2):
                prefix = "model.layers.{0}.".format(layer)
                tensors.update({
                    prefix + "self_attn.qkv_proj.weight": ("F32", [36, 12], b""),
                    prefix + "self_attn.o_proj.weight": ("F32", [12, 12], b""),
                    prefix + "mlp.gate_up_proj.weight": ("F32", [32, 12], b""),
                    prefix + "mlp.down_proj.weight": ("F32", [12, 16], b""),
                    prefix + "input_layernorm.weight": ("F32", [12], b""),
                    prefix + "post_attention_layernorm.weight": ("F32", [12], b""),
                })
            _write_model(model_dir, config, tensors)

            graph = Phi3ArchitectureAdapter().build_graph(SafeTensorsSourceReader(model_dir))

            self.assertEqual("phi3", graph.architecture_id)
            self.assertEqual(["decoder-group-00", "decoder-group-01"],
                             [stage.stage_id for stage in graph.stages])
            self.assertEqual(["decoder-group-00"], graph.stages[-1].dependencies)
            self.assertEqual("0", graph.stages[0].metadata["layer_start"])
            self.assertEqual("1", graph.stages[0].metadata["layer_end"])
            self.assertIn("token-input", graph.stages[0].metadata["roles"])
            self.assertIn("logit-output", graph.stages[-1].metadata["roles"])
            self.assertEqual("12", graph.metadata["hidden_size"])
            self.assertGreater(graph.stages[0].estimated_resident_bytes, 0)


def _write_model(model_dir, config, tensors):
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    offsets = {}
    payload = bytearray()
    for name, (data_type, shape, data) in tensors.items():
        element_count = 1
        for dimension in shape:
            element_count *= dimension
        expected_bytes = element_count * 4
        tensor_data = data if data else bytes(expected_bytes)
        start = len(payload)
        payload.extend(tensor_data)
        offsets[name] = {
            "dtype": data_type,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    header = json.dumps(offsets, separators=(",", ":")).encode("utf-8")
    file_name = "model.safetensors"
    with (model_dir / file_name).open("wb") as stream:
        stream.write(struct.pack("<Q", len(header)))
        stream.write(header)
        stream.write(payload)
    index = {"metadata": {"total_size": len(payload)}, "weight_map": {name: file_name for name in tensors}}
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()