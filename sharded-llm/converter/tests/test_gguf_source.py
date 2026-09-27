import struct
import tempfile
import unittest
from pathlib import Path

from ramanujan_shards.gguf_source import GGUFSourceReader


def _string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


class GGUFSourceTest(unittest.TestCase):
    def test_reads_aligned_q4_tensor_without_crossing_next_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.gguf"
            header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 2, 2))
            header += _string("general.architecture") + struct.pack("<I", 8) + _string("qwen35")
            header += _string("general.alignment") + struct.pack("<II", 4, 32)
            header += _string("blk.0.weight") + struct.pack("<IQIQ", 1, 32, 2, 0)
            header += _string("output_norm.weight") + struct.pack("<IQIQ", 1, 2, 0, 32)
            header += bytes(-len(header) % 32)
            path.write_bytes(header + b"a" * 18 + b"\0" * 14 + struct.pack("<ff", 1.0, 2.0))

            reader = GGUFSourceReader(path)
            self.assertEqual("qwen35", reader.metadata()["general.architecture"])
            self.assertEqual([32], reader.tensor_metadata("blk.0.weight")["shape"])
            self.assertEqual(18, reader.tensor_metadata("blk.0.weight")["length"])
            self.assertEqual(len(header) + 32, reader.tensor_metadata("output_norm.weight")["offset"])
            with reader.open_tensor("blk.0.weight") as stream:
                self.assertTrue(stream.seekable())
                self.assertEqual(b"a" * 18, stream.read())
                self.assertEqual(b"", stream.read())
                self.assertEqual(b"a", stream.seek(17) and stream.read(1))

    def test_rejects_truncated_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.gguf"
            header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 1, 0))
            header += _string("weight") + struct.pack("<IQIQ", 1, 32, 2, 0)
            header += bytes(-len(header) % 32)
            path.write_bytes(header + b"a" * 17)
            with self.assertRaisesRegex(ValueError, "truncated GGUF tensor"):
                GGUFSourceReader(path)

    def test_preserves_unknown_quantization_as_opaque_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "opaque.gguf"
            header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 2, 0))
            header += _string("first") + struct.pack("<IQIQ", 1, 32, 100, 0)
            header += _string("second") + struct.pack("<IQIQ", 1, 2, 0, 32)
            header += bytes(-len(header) % 32)
            path.write_bytes(header + b"x" * 18 + b"\0" * 14 + struct.pack("<ff", 1.0, 2.0))
            reader = GGUFSourceReader(path)
            self.assertEqual("GGML_TYPE_100", reader.tensor_metadata("first")["dtype"])
            self.assertTrue(reader.tensor_metadata("first")["length_includes_padding"])
            with reader.open_tensor("first") as stream:
                self.assertEqual(b"x" * 18 + b"\0" * 14, stream.read())

    def test_knows_exact_k_quant_payload_length(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kquant.gguf"
            header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 1, 0))
            header += _string("weight") + struct.pack("<IQIQ", 1, 256, 13, 0)
            header += bytes(-len(header) % 32)
            path.write_bytes(header + b"q" * 176 + b"unused tail")
            reader = GGUFSourceReader(path)
            self.assertEqual("Q5_K", reader.tensor_metadata("weight")["dtype"])
            self.assertEqual(176, reader.tensor_metadata("weight")["length"])
            with reader.open_tensor("weight") as stream:
                self.assertEqual(b"q" * 176, stream.read())


if __name__ == "__main__":
    unittest.main()