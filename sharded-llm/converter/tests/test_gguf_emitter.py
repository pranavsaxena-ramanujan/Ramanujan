import json
import struct
import tempfile
import unittest
from pathlib import Path

from ramanujan_shards.gguf_emitter import emit_gguf_package
from ramanujan_shards.verify_gguf import verify_gguf_package


def _string(value):
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _write_gguf(path, architecture, tensors, block_count=None):
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1 + (block_count is not None)))
    header += _string("general.architecture") + struct.pack("<I", 8) + _string(architecture)
    if block_count is not None:
        header += _string(architecture + ".block_count") + struct.pack("<II", 4, block_count)
    payload = bytearray()
    for name, type_id, dimensions, data in tensors:
        payload += bytes(-len(payload) % 32)
        header += _string(name) + struct.pack("<I", len(dimensions))
        header += b"".join(struct.pack("<Q", dimension) for dimension in dimensions)
        header += struct.pack("<IQ", type_id, len(payload))
        payload += data
    header += bytes(-len(header) % 32)
    path.write_bytes(header + payload)


class GenericGGUFEmitterTest(unittest.TestCase):
    def test_shards_non_qwen_layers_without_output_head(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensors = [("token_embd.weight", 0, [8], b"e" * 32)]
            tensors += [("blk.{0}.attn_q.weight".format(index), 2, [32], bytes([index]) * 18)
                        for index in range(4)]
            tensors += [("output_norm.weight", 0, [8], b"n" * 32)]
            _write_gguf(root / "model.gguf", "llama", tensors, block_count=4)

            package = emit_gguf_package(root / "model.gguf", root / "shards", shards=2)

            self.assertEqual("llama", package["architectureId"])
            self.assertEqual("weights-only", package["status"])
            self.assertEqual(2, len(package["shards"]))
            discovered = {}
            for index in range(2):
                shard = root / "shards" / "shard-{0:02d}".format(index)
                manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual((index * 2, index * 2 + 2),
                                 (manifest["layerStart"], manifest["layerEnd"]))
                for name, entry in manifest["tensorFiles"].items():
                    self.assertNotIn(name, discovered)
                    discovered[name] = (shard / entry["path"]).read_bytes()
            self.assertEqual({name: data for name, _, _, data in tensors}, discovered)
            self.assertEqual(len(tensors), verify_gguf_package(root / "shards"))

            manifest_path = root / "shards" / "shard-01" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["layerStart"] = 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "layer coverage"):
                verify_gguf_package(root / "shards")

    def test_unknown_architecture_uses_tensor_fallback_and_safe_filenames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensors = [("encoder/weight", 100, [32], b"x" * 18),
                       ("decoder.weight", 0, [8], b"y" * 32)]
            _write_gguf(root / "model.gguf", "custom", tensors)

            package = emit_gguf_package(root / "model.gguf", root / "shards", shards=2)

            self.assertEqual("0", package["metadata"]["layer_count"])
            entries = {}
            for shard_id in ("shard-00", "shard-01"):
                shard = root / "shards" / shard_id
                manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
                self.assertNotIn("layerStart", manifest)
                for name, descriptor in manifest["tensorFiles"].items():
                    self.assertNotIn("/", Path(descriptor["path"]).name)
                    entries[name] = (descriptor, (shard / descriptor["path"]).read_bytes())
            self.assertEqual(b"x" * 18 + b"\0" * 14, entries["encoder/weight"][1])
            self.assertTrue(entries["encoder/weight"][0]["lengthIncludesPadding"])
            self.assertEqual(b"y" * 32, entries["decoder.weight"][1])
            self.assertEqual(2, verify_gguf_package(root / "shards"))
            opaque_path = next((root / "shards" / shard_id / entry["path"])
                               for shard_id in ("shard-00", "shard-01")
                               for name, (entry, _) in entries.items()
                               if name == "encoder/weight" and (root / "shards" / shard_id / entry["path"]).exists())
            with opaque_path.open("r+b") as stream:
                stream.write(b"z")
            with self.assertRaisesRegex(ValueError, "checksum or size mismatch"):
                verify_gguf_package(root / "shards")

    def test_single_shard_fetches_only_assigned_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensors = [("token_embd.weight", 0, [8], b"e" * 32)]
            tensors += [("blk.{0}.attn_q.weight".format(index), 2, [32], bytes([index]) * 18)
                        for index in range(4)]
            _write_gguf(root / "model.gguf", "llama", tensors, block_count=4)
            package = emit_gguf_package(root / "model.gguf", root / "shards", shards=2,
                                        shard_index=1)
            self.assertTrue(package["partial"])
            self.assertEqual(2, package["totalShards"])
            self.assertEqual(["shard-01"], [entry["shardId"] for entry in package["shards"]])
            self.assertEqual(2, verify_gguf_package(root / "shards"))
            self.assertFalse((root / "shards" / "shard-00").exists())
            with self.assertRaisesRegex(ValueError, "shard_index"):
                emit_gguf_package(root / "model.gguf", root / "invalid", shards=2,
                                  shard_index=2)


if __name__ == "__main__":
    unittest.main()