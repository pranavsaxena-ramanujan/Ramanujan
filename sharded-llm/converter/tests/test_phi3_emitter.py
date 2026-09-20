import ast
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ramanujan_shards.kernel_generator import generate_phi3_decode_kernel, generate_phi3_prefill_kernel
from ramanujan_shards.tensor_writer import write_packed_4bit_matrix


class Phi3EmitterTest(unittest.TestCase):
    def test_generated_kernel_uses_supported_array_declarations(self):
        reference = Path(__file__).parents[3] / "ramanujan-test-codes" / "phi3" / "phi3_transformer_stack_4bit.py"
        source = generate_phi3_prefill_kernel(reference, 0, 8, False)
        tree = ast.parse(source)
        unsupported = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.List)
        ]
        self.assertEqual([], unsupported)

    def test_generated_decode_kernel_is_supported_python(self):
        reference = Path(__file__).parents[3] / "ramanujan-test-codes" / "phi3" / "phi3_transformer_stack_4bit.py"
        source = generate_phi3_decode_kernel(reference, 24, 32, True)
        tree = ast.parse(source)
        self.assertIn("rope_and_cache_decode_GPU_1", source)
        self.assertIn("LOAD_MEM(l24_qkv_packed)", source)
        self.assertIn("LOAD_MEM(l31_ln2_g)", source)
        self.assertIn("idx = row_int * 3072 + col", source)
        self.assertIn("RETURN(h_state, l24_k_cache", source)
        self.assertFalse(any(
            isinstance(node, ast.Assign) and isinstance(node.value, ast.List)
            for node in ast.walk(tree)
        ))

    def test_chunked_quantization_round_trip_and_temp_cleanup(self):
        values = np.array([
            [-7.0, -3.0, 0.0, 1.0, 4.0, 7.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype="<f4")
        metadata = {"dtype": "F32", "shape": [2, 7]}
        with tempfile.TemporaryDirectory(prefix="rj_phi3_emit_test_") as directory:
            root = Path(directory)
            packed_path = root / "packed.bin"
            scales_path = root / "scales.bin"
            with (root / "source.bin").open("wb") as stream:
                stream.write(values.tobytes())
            with (root / "source.bin").open("rb") as source:
                write_packed_4bit_matrix(
                    source, metadata, packed_path, scales_path,
                    chunk_rows=1,
                )

            packed = np.fromfile(packed_path, dtype="<f4").reshape(2, 2).astype(np.uint32)
            scales = np.fromfile(scales_path, dtype="<f4")
            restored = np.empty((2, 12), dtype=np.float32)
            for nibble in range(6):
                restored[:, nibble::6] = (((packed >> (4 * nibble)) & 15).astype(np.float32) - 8) * scales[:, None]
            np.testing.assert_allclose(values, restored[:, :7], atol=1.0)
        self.assertFalse(root.exists())

if __name__ == "__main__":
    unittest.main()