import io
import struct
import tempfile
import unittest
from pathlib import Path

from ramanujan_shards.emit_q4_1_operator import emit_q5_k_operator
from ramanujan_shards.gguf_emitter import emit_gguf_package
from ramanujan_shards.gguf_q5_k import (decode_q5_k_row, dot_q5_k_row,
                                       generate_q5_k_matvec_program)


class Q5KReferenceTest(unittest.TestCase):
    def test_high_bits_nibbles_and_upper_scale_bits(self):
        scales = bytes([0x41, 2, 3, 4, 0x85, 6, 7, 8, 0x1f, 0x22, 0x33, 0x44])
        high_bits = bytes([0x55] * 32)
        low_bits = bytes([0x10] * 128)
        block = struct.pack("<ee", 0.5, 0.25) + scales + high_bits + low_bits
        source = io.BytesIO(block * 2)

        values = decode_q5_k_row(source, 1, 256)

        self.assertEqual(256, len(values))
        self.assertEqual(0.5 * 1 * 16 - 0.25 * 5, values[0])
        self.assertEqual(0.5 * 2 * 1 - 0.25 * 6, values[32])
        self.assertEqual(0.5 * 31 * 16 - 0.25 * 33, values[128])
        self.assertEqual(0.5 * 2 * 1 - 0.25 * 2, values[160])
        self.assertEqual(sum(values), dot_q5_k_row(source, 1, [1.0] * 256))

    def test_rejects_truncated_or_unaligned_rows(self):
        with self.assertRaisesRegex(ValueError, "multiple of 256"):
            decode_q5_k_row(io.BytesIO(), 0, 255)
        with self.assertRaisesRegex(ValueError, "truncated"):
            decode_q5_k_row(io.BytesIO(bytes(175)), 0, 256)

    def test_generated_gpu_program_uses_q5_k_blocks(self):
        program = generate_q5_k_matvec_program(5120, 6144)
        self.assertIn("blocks_per_row = 24", program)
        self.assertIn("GGUF_Q5_K_VALUE(weights, block_index, position)", program)
        self.assertIn("q5_k_matvec_GPU_1(weights, activation, output, 5120)", program)
        self.assertNotIn("weights = [", program)
        with self.assertRaisesRegex(ValueError, "block-aligned"):
            generate_q5_k_matvec_program(2, 255)

    def test_operator_binding_from_sharded_gguf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = b"general.architecture"
            architecture = b"custom"
            name = b"matrix.weight"
            header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 1, 1))
            header += struct.pack("<Q", len(key)) + key + struct.pack("<I", 8)
            header += struct.pack("<Q", len(architecture)) + architecture
            header += struct.pack("<Q", len(name)) + name + struct.pack("<IQQIQ", 2, 256, 2, 13, 0)
            header += bytes(-len(header) % 32)
            block = struct.pack("<ee", 0.5, 0.25) + bytes(12 + 32 + 128)
            (root / "source.gguf").write_bytes(header + block * 2)
            emit_gguf_package(root / "source.gguf", root / "shards", 1)

            binding = emit_q5_k_operator(root / "shards", "matrix.weight", root / "probe")

            self.assertEqual([2, 44], binding["weightShape"])
            self.assertEqual([256], binding["activationShape"])
            self.assertEqual([2], binding["outputShape"])
            self.assertIn("GGUF_Q5_K_VALUE", (root / "probe" / binding["programPath"]).read_text())
            with self.assertRaisesRegex(ValueError, "already exists"):
                emit_q5_k_operator(root / "shards", "matrix.weight", root / "probe")


if __name__ == "__main__":
    unittest.main()