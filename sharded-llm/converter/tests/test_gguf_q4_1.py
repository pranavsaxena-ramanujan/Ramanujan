import io
import json
import struct
import tempfile
import unittest
from pathlib import Path

from ramanujan_shards.emit_q4_1_operator import emit_q4_1_operator
from ramanujan_shards.gguf_emitter import emit_gguf_package
from ramanujan_shards.gguf_q4_1 import (
    decode_q4_1_row, dot_q4_1_row, generate_q4_1_matvec_program,
)


class Q41ReferenceTest(unittest.TestCase):
    def test_low_nibbles_precede_high_nibbles_with_per_block_scale_and_offset(self):
        first_block = struct.pack("<ee", 0.5, -2.0) + bytes(range(16))
        second_block = struct.pack("<ee", 0.25, 1.0) + bytes([0xF0] * 16)
        source = io.BytesIO(first_block + second_block + first_block + second_block)

        decoded = decode_q4_1_row(source, 0, 64)

        self.assertEqual([-2.0 + 0.5 * index for index in range(16)], decoded[:16])
        self.assertEqual([-2.0] * 16, decoded[16:32])
        self.assertEqual([1.0] * 16, decoded[32:48])
        self.assertEqual([4.75] * 16, decoded[48:64])
        self.assertAlmostEqual(sum(decoded), dot_q4_1_row(source, 0, [1.0] * 64))
        self.assertEqual(decoded, decode_q4_1_row(source, 1, 64))

    def test_rejects_truncated_and_invalid_rows(self):
        with self.assertRaisesRegex(ValueError, "multiple of 32"):
            decode_q4_1_row(io.BytesIO(), 0, 33)
        with self.assertRaisesRegex(ValueError, "truncated"):
            decode_q4_1_row(io.BytesIO(b"\0" * 19), 0, 32)

    def test_generated_gpu_program_uses_raw_q4_1_words(self):
        program = generate_q4_1_matvec_program(12288, 5120)
        self.assertTrue(program.startswith("def q4_1_matvec_GPU_1("))
        self.assertNotIn("weights = [0 for", program)
        self.assertNotIn("activation = [0 for", program)
        self.assertNotIn("output = [0 for", program)
        self.assertIn("blocks_per_row = 160", program)
        self.assertIn("GGUF_Q4_1_VALUE(weights, block_index, position)", program)
        self.assertIn("q4_1_matvec_GPU_1(weights, activation, output, 12288)", program)
        with self.assertRaisesRegex(ValueError, "block-aligned"):
            generate_q4_1_matvec_program(2, 33)

    def test_operator_binding_uses_verified_tensor_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = b"general.architecture"
            arch = b"custom"
            name = b"matrix.weight"
            header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 1, 1))
            header += struct.pack("<Q", len(key)) + key + struct.pack("<I", 8)
            header += struct.pack("<Q", len(arch)) + arch
            header += struct.pack("<Q", len(name)) + name + struct.pack("<IQQIQ", 2, 32, 2, 3, 0)
            header += bytes(-len(header) % 32)
            (root / "source.gguf").write_bytes(header + (struct.pack("<ee", 0.5, -2.0)
                                                       + bytes(range(16))) * 2)
            emit_gguf_package(root / "source.gguf", root / "shards", 1)

            metadata = emit_q4_1_operator(root / "shards", "matrix.weight", root / "probe")

            self.assertEqual("operator-probe", metadata["status"])
            self.assertEqual([2, 5], metadata["weightShape"])
            self.assertEqual([32], metadata["activationShape"])
            self.assertEqual([2], metadata["outputShape"])
            self.assertEqual(metadata, json.loads((root / "probe" / "binding.json").read_text()))
            self.assertIn("blocks_per_row = 1", (root / "probe" / "q4_1_matvec.py").read_text())
            with self.assertRaisesRegex(ValueError, "already exists"):
                emit_q4_1_operator(root / "shards", "matrix.weight", root / "probe")


if __name__ == "__main__":
    unittest.main()