import io
import struct
import tempfile
import unittest
from pathlib import Path

from ramanujan_shards.emit_q4_1_operator import emit_f32_operator
from ramanujan_shards.gguf_emitter import emit_gguf_package
from ramanujan_shards.gguf_f32_matvec import dot_f32_row, generate_f32_matvec_program


class F32MatvecTest(unittest.TestCase):
    def test_rows_and_index_bounds(self):
        source = io.BytesIO(struct.pack("<6f", 1, 2, 3, 4, -2, 1))
        self.assertEqual(7, dot_f32_row(source, 1, [2, 1, 1]))
        with self.assertRaisesRegex(ValueError, "truncated"):
            dot_f32_row(source, 2, [1, 1, 1])
        program = generate_f32_matvec_program(48, 5120)
        self.assertIn("weights[row * columns + column]", program)
        with self.assertRaises(ValueError):
            generate_f32_matvec_program(2**30, 4)

    def test_binding_f32_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = b"general.architecture"
            architecture = b"custom"
            name = b"matrix.weight"
            header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 1, 1))
            header += struct.pack("<Q", len(key)) + key + struct.pack("<I", 8)
            header += struct.pack("<Q", len(architecture)) + architecture
            header += struct.pack("<Q", len(name)) + name + struct.pack("<IQQIQ", 2, 128, 2, 0, 0)
            header += bytes(-len(header) % 32)
            (root / "source.gguf").write_bytes(header + bytes(128 * 2 * 4))
            emit_gguf_package(root / "source.gguf", root / "shards", 1)
            binding = emit_f32_operator(root / "shards", name.decode(), root / "probe")
            self.assertEqual([2, 128], binding["weightShape"])
            self.assertEqual([128], binding["activationShape"])
            self.assertIn("f32_matvec_GPU_1", (root / "probe" / binding["programPath"]).read_text())