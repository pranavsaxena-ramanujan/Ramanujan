import unittest

import numpy as np

from ramanujan_shards.gguf_q4_1 import decode_q4_1_row
from ramanujan_shards.gguf_q5_k import decode_q5_k_row
from ramanujan_shards.gguf_q6_k import decode_q6_k_row
from ramanujan_shards.qwen35_programs import (
    Qwen35Config, generate_attention_program, generate_delta_program, generate_embed_program,
    generate_head_program)
from ramanujan_shards.qwen35_reference import dequantize

import io


def _metadata(**overrides):
    metadata = {
        "general.architecture": "qwen35", "qwen35.embedding_length": 512,
        "qwen35.attention.layer_norm_rms_epsilon": 1e-6, "qwen35.ssm.conv_kernel": 4,
        "qwen35.ssm.state_size": 16, "qwen35.ssm.group_count": 2, "qwen35.ssm.time_step_rank": 4,
        "qwen35.ssm.inner_size": 64, "qwen35.attention.head_count": 4,
        "qwen35.attention.head_count_kv": 2, "qwen35.attention.key_length": 32,
        "qwen35.attention.value_length": 32, "qwen35.rope.dimension_count": 16,
        "qwen35.rope.freq_base": 1e7, "qwen35.feed_forward_length": 1024,
        "qwen35.block_count": 5, "qwen35.nextn_predict_layers": 1,
        "qwen35.full_attention_interval": 4, "tokenizer.ggml.tokens": ["a"] * 300,
    }
    metadata.update(overrides)
    return metadata


class Qwen35ProgramTest(unittest.TestCase):
    def setUp(self):
        self.config = Qwen35Config.from_metadata(_metadata())
        self.delta = {"attn_norm": "gguf-f32", "post_norm": "gguf-f32", "w_qkv": "gguf-q4_1",
                      "w_z": "gguf-q4_1", "w_beta": "gguf-f32", "w_alpha": "gguf-f32",
                      "dt_bias": "gguf-f32", "ssm_a": "gguf-f32", "conv_w": "gguf-f32",
                      "ssm_norm": "gguf-f32", "w_out": "gguf-q5_k", "w_ffn_gate": "gguf-q4_1",
                      "w_ffn_up": "gguf-q4_1", "w_ffn_down": "gguf-q4_1"}
        self.attn = {"attn_norm": "gguf-f32", "post_norm": "gguf-f32", "w_q": "gguf-q4_1",
                     "w_k": "gguf-q4_1", "w_v": "gguf-q4_1", "q_norm": "gguf-f32", "k_norm": "gguf-f32",
                     "w_o": "gguf-q4_1", "w_ffn_gate": "gguf-q4_1", "w_ffn_up": "gguf-q4_1",
                     "w_ffn_down": "gguf-q4_1"}

    def test_config_excludes_nextn_block_and_marks_attention_layers(self):
        self.assertEqual(4, self.config.layers)
        self.assertEqual([False, False, False, True], [self.config.is_attention(i) for i in range(4)])
        with self.assertRaises(ValueError):
            Qwen35Config.from_metadata(_metadata(**{"qwen35.ssm.inner_size": 63}))

    def test_layer_programs_bind_state_and_return_it(self):
        delta = generate_delta_program(self.config, self.delta)
        self.assertIn("RETURN(h_state, ssm_s_state, ssm_conv_state)", delta)
        self.assertIn("GGUF_Q5_K_VALUE(w, block_index, position)", delta)
        self.assertIn("dn_delta_GPU_1(ssm_s_state, qk, co, gate, beta, core, 64)", delta)
        attn = generate_attention_program(self.config, self.attn, 8)
        self.assertIn("RETURN(h_state, attn_k_cache, attn_v_cache)", attn)
        self.assertIn("attn_scores_GPU_1(qn, attn_k_cache, pos_arr, scores, 32)", attn)
        for source in (delta, attn):
            self.assertEqual(source.count("LOAD_MEM("), source.count("RELEASE_MEM("))
        with self.assertRaises(ValueError):
            generate_attention_program(self.config, self.attn, 2**24)

    def test_embed_and_head_programs(self):
        self.assertIn("GGUF_Q4_1_VALUE(emb_row, i / 32, i % 32)",
                      generate_embed_program(self.config, "gguf-q4_1"))
        head = generate_head_program(self.config, {"output_norm": "gguf-f32", "w_output": "gguf-q6_k"})
        self.assertIn("q6_kmv512_GPU_1(w_output, xn, logits, 300)", head)
        self.assertIn("RETURN(logits, argmax_arr)", head)
        with self.assertRaises(ValueError):
            generate_embed_program(self.config, "gguf-f32")


class VectorizedDequantTest(unittest.TestCase):
    def test_matches_row_oracles(self):
        rng = np.random.default_rng(3)
        for decoder, encoding, block_bytes in ((decode_q4_1_row, "gguf-q4_1", 20),
                                                (decode_q5_k_row, "gguf-q5_k", 176),
                                                (decode_q6_k_row, "gguf-q6_k", 210)):
            columns = 512
            blocks = columns // (32 if encoding == "gguf-q4_1" else 256)
            raw = rng.integers(0, 256, size=(2, blocks * block_bytes), dtype=np.uint8)
            scale_offsets = (0, 2) if encoding != "gguf-q6_k" else (208,)
            for block in range(blocks):
                for offset in scale_offsets:
                    raw[:, block * block_bytes + offset:block * block_bytes + offset + 2] = (
                        np.frombuffer(np.float16(0.01).tobytes(), np.uint8))
            expected = np.array([decoder(io.BytesIO(raw.tobytes()), row, columns) for row in range(2)],
                                np.float32)
            np.testing.assert_allclose(dequantize(encoding, raw, columns), expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
