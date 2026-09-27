import struct
import tempfile
import unittest
from pathlib import Path

from ramanujan_shards.emit_f32_rmsnorm_operator import emit_f32_rmsnorm_operator
from ramanujan_shards.gguf_emitter import emit_gguf_package
from ramanujan_shards.gguf_f32_norm import generate_f32_rmsnorm_program


class F32NormTest(unittest.TestCase):
    def test_model_dimensions_and_epsilon(self):
        program = generate_f32_rmsnorm_program(5120, 1e-6)
        self.assertIn("dimension = 5120", program)
        self.assertIn("sqrt(squares / dimension + 1e-06)", program)
        self.assertIn("output[index] = hidden[index] * inv_rms * gamma[index]", program)

    def test_rejects_invalid_parameters(self):
        for dimension, epsilon in ((0, 1e-6), (5120, 0), (5120, float("nan"))):
            with self.assertRaises(ValueError):
                generate_f32_rmsnorm_program(dimension, epsilon)

    def test_binding_f32_norm_from_gguf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = b"general.architecture"
            architecture = b"custom"
            name = b"output_norm.weight"
            header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 1, 1))
            header += struct.pack("<Q", len(key)) + key + struct.pack("<I", 8)
            header += struct.pack("<Q", len(architecture)) + architecture
            header += struct.pack("<Q", len(name)) + name + struct.pack("<IQIQ", 1, 128, 0, 0)
            header += bytes(-len(header) % 32)
            (root / "source.gguf").write_bytes(header + struct.pack("<128f", *([1.0] * 128)))
            emit_gguf_package(root / "source.gguf", root / "shards", 1)

            binding = emit_f32_rmsnorm_operator(root / "shards", name.decode(), root / "probe", 1e-6)

            self.assertEqual([128], binding["shape"])
            self.assertEqual(1e-6, binding["epsilon"])
            self.assertIn("f32_rmsnorm_GPU_1", (root / "probe" / binding["programPath"]).read_text())


if __name__ == "__main__":
    unittest.main()