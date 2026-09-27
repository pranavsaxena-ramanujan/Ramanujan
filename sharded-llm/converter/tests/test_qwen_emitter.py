import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from ramanujan_shards.qwen_emitter import emit_qwen_package
from ramanujan_shards.verify_qwen import verify_qwen_package


def _string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _write_qwen_gguf(path):
    tensors = [("token_embd.weight", 0, b"E" * 32)]
    tensors += [("blk.{0}.attn_q.weight".format(layer), 2, bytes([layer + 1]) * 18)
                for layer in range(8)]
    tensors += [("output_norm.weight", 0, b"N" * 32),
                ("output.weight", 2, b"O" * 18)]
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 3))
    header += _string("general.architecture") + struct.pack("<I", 8) + _string("qwen35")
    header += _string("qwen35.block_count") + struct.pack("<II", 4, 8)
    header += _string("qwen35.embedding_length") + struct.pack("<II", 4, 8)
    payload = bytearray()
    for name, type_id, data in tensors:
        payload += bytes(-len(payload) % 32)
        header += _string(name) + struct.pack("<IQIQ", 1, 32 if type_id == 2 else 8,
                                              type_id, len(payload))
        payload += data
    header += bytes(-len(header) % 32)
    path.write_bytes(header + payload)
    return dict((name, data) for name, _, data in tensors)


class QwenEmitterTest(unittest.TestCase):
    def test_streams_raw_weights_to_four_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensors = _write_qwen_gguf(root / "tiny.gguf")
            output = root / "shards"

            package = emit_qwen_package(root / "tiny.gguf", output)

            self.assertEqual("weights-only", package["status"])
            self.assertEqual(4, len(package["shards"]))
            covered = []
            emitted = set()
            for entry in package["shards"]:
                shard = output / entry["shardId"]
                manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
                covered += list(range(manifest["layerStart"], manifest["layerEnd"]))
                self.assertEqual(2, len(manifest["layerTypes"]))
                self.assertEqual(sum(item["bytes"] for item in manifest["tensorFiles"].values()),
                                 entry["estimatedResidentBytes"])
                for name, descriptor in manifest["tensorFiles"].items():
                    self.assertNotIn(name, emitted)
                    emitted.add(name)
                    data = (shard / descriptor["path"]).read_bytes()
                    self.assertEqual(tensors[name], data)
                    self.assertEqual("sha256:" + hashlib.sha256(data).hexdigest(),
                                     manifest["checksums"][descriptor["path"]])
            self.assertEqual(list(range(8)), covered)
            self.assertEqual(set(tensors), emitted)
            self.assertEqual(len(tensors), verify_qwen_package(output))
            self.assertEqual("gguf-q4_0", json.loads(
                (output / "shard-00" / "manifest.json").read_text(encoding="utf-8")
            )["tensorFiles"]["blk.0.attn_q.weight"]["encoding"])

            with (output / "shard-01" / "weights" / "blk.2.attn_q.weight.bin").open("r+b") as stream:
                stream.write(b"X")
            with self.assertRaisesRegex(ValueError, "checksum or size mismatch"):
                verify_qwen_package(output)

    def test_does_not_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_qwen_gguf(root / "tiny.gguf")
            output = root / "shards"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                emit_qwen_package(root / "tiny.gguf", output)
            self.assertEqual([], list(output.iterdir()))


if __name__ == "__main__":
    unittest.main()