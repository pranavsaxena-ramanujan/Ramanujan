import io
import struct
import tempfile
import unittest
from pathlib import Path

from ramanujan_shards.emit_q4_1_operator import emit_q6_k_operator
from ramanujan_shards.gguf_emitter import emit_gguf_package
from ramanujan_shards.gguf_q6_k import (decode_q6_k_row, dot_q6_k_row,
                                       generate_q6_k_matvec_program)


class Q6KReferenceTest(unittest.TestCase):
    def test_signed_subscales_and_four_segments(self):
        block = (bytes([0x21] * 128) + bytes([0xe4] * 64) +
                 struct.pack("<16b", 1, -2, 3, -4, 5, -6, 7, -8,
                             9, -10, 11, -12, 13, -14, 15, -16) + struct.pack("<e", 0.5))
        source = io.BytesIO(block * 2)

        values = decode_q6_k_row(source, 1, 256)

        self.assertEqual(256, len(values))
        self.assertEqual(0.5 * 1 * -31, values[0])
        self.assertEqual(0.5 * -2 * -31, values[16])
        self.assertEqual(0.5 * 3 * -15, values[32])
        self.assertEqual(0.5 * 5 * 2, values[64])
        self.assertEqual(0.5 * 7 * 18, values[96])
        self.assertEqual(0.5 * 9 * -31, values[128])
        self.assertEqual(sum(values), dot_q6_k_row(source, 1, [1.0] * 256))

    def test_rejects_truncated_and_invalid_rows(self):
        with self.assertRaisesRegex(ValueError, "multiple of 256"):
            decode_q6_k_row(io.BytesIO(), 0, 255)
        with self.assertRaisesRegex(ValueError, "truncated"):
            decode_q6_k_row(io.BytesIO(bytes(209)), 0, 256)

    def test_generated_matvec_uses_integer_block_index(self):
        program = generate_q6_k_matvec_program(248320, 5120)
        self.assertIn("blocks_per_row = 20", program)
        self.assertIn("GGUF_Q6_K_VALUE(weights, block_index, position)", program)
        self.assertNotIn("weights = [", program)

    def test_operator_binding_for_half_word_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = b"general.architecture"
            architecture = b"custom"
            name = b"matrix.weight"
            header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 1, 1))
            header += struct.pack("<Q", len(key)) + key + struct.pack("<I", 8)
            header += struct.pack("<Q", len(architecture)) + architecture
            header += struct.pack("<Q", len(name)) + name + struct.pack("<IQQIQ", 2, 512, 2, 14, 0)
            header += bytes(-len(header) % 32)
            (root / "source.gguf").write_bytes(header + bytes(210 * 4))
            emit_gguf_package(root / "source.gguf", root / "shards", 1)

            binding = emit_q6_k_operator(root / "shards", "matrix.weight", root / "probe")

            self.assertEqual([2, 105], binding["weightShape"])
            self.assertEqual([512], binding["activationShape"])
            self.assertEqual([2], binding["outputShape"])
            self.assertIn("GGUF_Q6_K_VALUE", (root / "probe" / binding["programPath"]).read_text())


if __name__ == "__main__":
    unittest.main()