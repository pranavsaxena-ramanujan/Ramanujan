import json
import unittest
from pathlib import Path

from ramanujan_shards.chat_prompt import fit_chat_prompt, read_chat_turns


class WordTokenizer:
    def encode(self, text):
        return text.split()


class ChatPromptTest(unittest.TestCase):
    def test_keeps_everything_that_fits(self):
        prompt, tokens, dropped = fit_chat_prompt(WordTokenizer(), ["u1 ", "a1 ", "u2 "], "A:", "S ", budget=10)
        self.assertEqual(prompt, "S u1 a1 u2 A:")
        self.assertEqual(dropped, 0)
        self.assertEqual(len(tokens), 5)

    def test_drops_oldest_pairs(self):
        turns = ["one two ", "three four ", "five six ", "seven eight ", "nine "]
        prompt, _, dropped = fit_chat_prompt(WordTokenizer(), turns, budget=5)
        self.assertEqual((prompt, dropped), ("five six seven eight nine ", 2))

    def test_rejects_when_last_turn_is_too_long(self):
        with self.assertRaisesRegex(ValueError, "max-context"):
            fit_chat_prompt(WordTokenizer(), ["a ", "b ", "c d e "], budget=2)

    def test_rejects_even_turn_count(self):
        with self.assertRaises(ValueError):
            fit_chat_prompt(WordTokenizer(), ["a ", "b "], budget=10)

    def test_read_turns(self):
        self.assertEqual(read_chat_turns(json.dumps({"turns": ["x"], "suffix": "y"})), (["x"], "y", ""))
        with self.assertRaises(ValueError):
            read_chat_turns(json.dumps({"turns": [1]}))

    def test_real_tokenizer_adds_bos_once(self):
        metadata = Path(__file__).resolve().parents[4] / "gguf-models" / "qwen2.5-0.5b-instruct-q4_k_m-ir-plan" / "gguf-metadata.json"
        if not metadata.exists():
            self.skipTest("qwen2.5 metadata not available")
        from ramanujan_shards.llm_tokenizer import load_tokenizer
        tokenizer = load_tokenizer(json.loads(metadata.read_text()))
        turns = ["<|im_start|>user\nMy name is Ada.<|im_end|>\n", "<|im_start|>assistant\nHi Ada!<|im_end|>\n",
                 "<|im_start|>user\nWhat is my name?<|im_end|>\n"]
        prompt, tokens, dropped = fit_chat_prompt(tokenizer, turns, "<|im_start|>assistant\n", budget=200)
        self.assertEqual(dropped, 0)
        self.assertEqual(tokens, tokenizer.encode(prompt))
        self.assertIn("Ada", tokenizer.decode(tokens))


if __name__ == "__main__":
    unittest.main()
