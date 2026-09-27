from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

# llama.cpp LLAMA_VOCAB_PRE_TYPE_QWEN35 split pattern.
QWEN35_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
                  r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
_CONTROL, _USER_DEFINED = 3, 4


def load_tokenizer(metadata):
    if metadata.get("tokenizer.ggml.model") != "gpt2" or metadata.get("tokenizer.ggml.pre") != "qwen35":
        raise ValueError("expected a qwen35 byte-level BPE tokenizer")
    tokens = metadata["tokenizer.ggml.tokens"]
    types = metadata["tokenizer.ggml.token_type"]
    merges = [tuple(merge.split(" ", 1)) for merge in metadata["tokenizer.ggml.merges"]]
    tokenizer = Tokenizer(models.BPE(vocab={token: i for i, token in enumerate(tokens)}, merges=merges))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN35_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens([AddedToken(tokens[i], special=True, normalized=False)
                                  for i, kind in enumerate(types) if kind == _CONTROL])
    tokenizer.add_tokens([AddedToken(tokens[i], normalized=False)
                          for i, kind in enumerate(types) if kind == _USER_DEFINED])
    return tokenizer
