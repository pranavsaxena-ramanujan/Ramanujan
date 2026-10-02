import unittest

from ramanujan_shards.llm_tokenizer import GgufTokenizer


def _bpe(pre, **extra):
    metadata = {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": pre, "tokenizer.ggml.merges": [],
                "tokenizer.ggml.tokens": ["<s>", "a", "b"], "tokenizer.ggml.bos_token_id": 0}
    metadata.update(extra)
    return GgufTokenizer(metadata)


class AddBosDefaultTest(unittest.TestCase):
    """Without tokenizer.ggml.add_bos_token, follow llama.cpp's per-vocabulary default."""

    def test_llama3_bpe_adds_bos_by_default(self):
        self.assertTrue(_bpe("llama-bpe").add_bos)
        self.assertTrue(_bpe("llama3").add_bos)

    def test_other_bpe_does_not_add_bos_by_default(self):
        self.assertFalse(_bpe("qwen2").add_bos)
        self.assertFalse(_bpe("default").add_bos)

    def test_explicit_metadata_wins(self):
        self.assertFalse(_bpe("llama-bpe", **{"tokenizer.ggml.add_bos_token": False}).add_bos)
        self.assertTrue(_bpe("qwen2", **{"tokenizer.ggml.add_bos_token": True}).add_bos)


if __name__ == "__main__":
    unittest.main()
