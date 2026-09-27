import json
import struct
import tempfile
import unittest
from pathlib import Path

from ramanujan_shards.gguf_emitter import emit_gguf_package
from ramanujan_shards.gguf_ir_plan import build_ir_plan, require_executable, write_ir_plan


class GGUFIRPlanTest(unittest.TestCase):
    def test_detects_hybrid_blocks_and_rejects_executable_ir(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model-manifest.json").write_text(json.dumps({
                "sourceFormat": "gguf", "status": "weights-only", "partial": False,
                "architectureId": "qwen35", "metadata": {"layer_count": "3", "tensor_count": "6"},
                "shards": [{"shardId": "shard-00", "manifestPath": "shard-00/manifest.json"}],
            }), encoding="utf-8")
            shard = root / "shard-00"
            shard.mkdir()
            (shard / "manifest.json").write_text(json.dumps({
                "shardId": "shard-00", "layerStart": 0, "layerEnd": 3,
                "tensorFiles": {
                    name: {"encoding": encoding, "path": "weights/" + name + ".bin",
                           "shape": [32], "bytes": 18, "ggmlType": 2}
                    for name, encoding in (
                        ("token_embd.weight", "gguf-q4_1"),
                        ("blk.0.ssm_conv1d.weight", "gguf-q4_1"),
                        ("blk.0.ssm_a", "gguf-f32"),
                        ("blk.0.ssm_beta.weight", "gguf-q4_1"),
                        ("blk.1.attn_q.weight", "gguf-q4_1"),
                        ("blk.2.nextn.eh_proj.weight", "gguf-f32"),
                    )
                },
                "checksums": {"weights/" + name + ".bin": "sha256:" + "0" * 64
                              for name in ("token_embd.weight", "blk.0.ssm_conv1d.weight",
                                           "blk.0.ssm_a", "blk.0.ssm_beta.weight", "blk.1.attn_q.weight", "blk.2.nextn.eh_proj.weight")},
            }), encoding="utf-8")

            plan = build_ir_plan(root)

            self.assertEqual("planning-only", plan["status"])
            self.assertEqual(["gated_deltanet", "causal_attention", "auxiliary"],
                             [layer["operator"] for layer in plan["stages"][0]["layers"]])
            self.assertEqual(6, len(plan["tensorBindings"]))
            self.assertIn("llm.gated_deltanet", plan["requiredCapabilities"])
            self.assertIn("llm.causal_attention", plan["requiredCapabilities"])
            with self.assertRaisesRegex(RuntimeError, "no executable Ramanujan GGUF backend"):
                require_executable(plan)
            with self.assertRaisesRegex(RuntimeError, "no executable Ramanujan GGUF backend"):
                write_ir_plan(root, root / "rejected", run=True)
            self.assertFalse((root / "rejected").exists())

            written = write_ir_plan(root, root / "plan")
            self.assertEqual(plan, written)
            self.assertEqual("planning-only", json.loads(
                (root / "plan" / "shard-00" / "plan.json").read_text(encoding="utf-8"))["status"])
            with self.assertRaisesRegex(ValueError, "already exists"):
                write_ir_plan(root, root / "plan")

    def test_classifies_layers_by_tensors_for_any_architecture(self):
        names = ("token_embd.weight", "blk.0.attn_q.weight", "blk.1.attn_q.weight", "blk.1.ffn_gate_inp.weight")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model-manifest.json").write_text(json.dumps({
                "sourceFormat": "gguf", "status": "weights-only", "partial": False,
                "architectureId": "llama", "metadata": {"layer_count": "2", "tensor_count": "4"},
                "shards": [{"shardId": "shard-00", "manifestPath": "shard-00/manifest.json"}],
            }), encoding="utf-8")
            (root / "shard-00").mkdir()
            (root / "shard-00" / "manifest.json").write_text(json.dumps({
                "shardId": "shard-00", "layerStart": 0, "layerEnd": 2,
                "tensorFiles": {name: {"encoding": "gguf-q4_0", "path": "weights/" + name + ".bin",
                                       "shape": [32], "bytes": 18, "ggmlType": 2} for name in names},
                "checksums": {"weights/" + name + ".bin": "sha256:" + "0" * 64 for name in names},
            }), encoding="utf-8")

            layers = build_ir_plan(root)["stages"][0]["layers"]
            required = build_ir_plan(root)["requiredCapabilities"]

            self.assertEqual(["causal_attention", "unsupported"], [layer["operator"] for layer in layers])
            self.assertEqual("mixture-of-experts layers", layers[1]["reason"])
            self.assertIn("llm.causal_attention", required)
            self.assertIn("architecture.llama.mixture_of_experts_layers", required)

    def test_rejects_unrelated_gguf_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gguf = root / "wrong.gguf"
            architecture = b"llama"
            key = b"general.architecture"
            header = (b"GGUF" + struct.pack("<IQQ", 3, 0, 1)
                      + struct.pack("<Q", len(key)) + key + struct.pack("<I", 8)
                      + struct.pack("<Q", len(architecture)) + architecture)
            gguf.write_bytes(header)
            (root / "model-manifest.json").write_text(json.dumps({
                "sourceFormat": "gguf", "status": "weights-only", "architectureId": "qwen35",
                "metadata": {"tensor_count": "1", "layer_count": "0"},
                "shards": [{"shardId": "shard-00", "manifestPath": "shard-00/manifest.json"}],
            }), encoding="utf-8")
            (root / "shard-00").mkdir()
            (root / "shard-00" / "manifest.json").write_text(json.dumps({
                "shardId": "shard-00", "tensorFiles": {"x": {
                    "path": "weights/x.bin", "shape": [1], "bytes": 4,
                    "ggmlType": 0, "encoding": "gguf-f32"}},
                "checksums": {"weights/x.bin": "sha256:" + "0" * 64},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "architecture does not match"):
                write_ir_plan(root, root / "plan", gguf_path=gguf)
            self.assertFalse((root / "plan").exists())

    def test_exports_original_header_from_sharded_gguf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gguf = root / "tiny.gguf"
            architecture = b"custom"
            key = b"general.architecture"
            name = b"weight"
            header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 1, 1))
            header += struct.pack("<Q", len(key)) + key + struct.pack("<I", 8)
            header += struct.pack("<Q", len(architecture)) + architecture
            header += struct.pack("<Q", len(name)) + name + struct.pack("<IQIQ", 1, 8, 0, 0)
            header += bytes(-len(header) % 32)
            gguf.write_bytes(header + struct.pack("<8f", *range(8)))
            emit_gguf_package(gguf, root / "shards", shards=1)

            plan = write_ir_plan(root / "shards", root / "plan", gguf_path=gguf)

            self.assertEqual("planning-only", plan["status"])
            self.assertEqual("gguf-metadata.json", plan["sourceMetadataPath"])
            self.assertEqual("custom", json.loads((root / "plan" / "gguf-metadata.json")
                                                  .read_text(encoding="utf-8"))["general.architecture"])
            self.assertEqual("shard-00/weights/weight.bin", plan["tensorBindings"]["weight"]["path"])


if __name__ == "__main__":
    unittest.main()