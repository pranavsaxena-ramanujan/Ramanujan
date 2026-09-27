"""Tokenizers built from GGUF ``tokenizer.ggml.*`` metadata (llama.cpp semantics).

``gpt2`` vocabularies are byte-level BPE; the ``tokenizer.ggml.pre`` value selects the
pre-tokenizer split regex. ``llama`` vocabularies are SentencePiece BPE, tokenized with
llama.cpp's score-ordered bigram merge and ``<0xXX>`` byte fallback.
"""
import heapq
import re

_NORMAL, _UNKNOWN, _CONTROL, _USER_DEFINED, _UNUSED, _BYTE = 1, 2, 3, 4, 5, 6

_GPT2 = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
_LLAMA3 = (r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
           r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
_QWEN2 = (r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
          r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
_QWEN35 = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
           r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
# tokenizer.ggml.pre -> (split regex, NFC normalize, ignore_merges); from llama.cpp llama-vocab.cpp.
BPE_PRE_TOKENIZERS = {
    "default": (_GPT2, False, False),
    "gpt-2": (_GPT2, False, False),
    "llama3": (_LLAMA3, False, True),
    "llama-bpe": (_LLAMA3, False, True),
    "smollm": (_GPT2, False, False),
    "qwen2": (_QWEN2, True, False),
    "qwen35": (_QWEN35, True, False),
}
# llama.cpp treats these control tokens as end-of-generation in addition to eos/eot.
_END_OF_GENERATION = {"<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>", "<EOT>",
                      "<|end_of_text|>", "<|eom_id|>", "<|return|>", "<|call|>"}


class GgufTokenizer:
    def __init__(self, metadata):
        self.model = metadata.get("tokenizer.ggml.model")
        self.tokens = metadata["tokenizer.ggml.tokens"]
        self.types = metadata.get("tokenizer.ggml.token_type") or [_NORMAL] * len(self.tokens)
        self.ids = {token: i for i, token in enumerate(self.tokens)}
        self.bos = metadata.get("tokenizer.ggml.bos_token_id")
        self.eos = metadata.get("tokenizer.ggml.eos_token_id")
        self.add_bos = bool(metadata.get("tokenizer.ggml.add_bos_token", self.model == "llama"))
        self.add_eos = bool(metadata.get("tokenizer.ggml.add_eos_token", False))
        self.stop_ids = {i for i in (self.eos, metadata.get("tokenizer.ggml.eot_token_id"),
                                     metadata.get("tokenizer.ggml.eom_token_id")) if i is not None}
        self.stop_ids.update(i for i, token in enumerate(self.tokens)
                             if token in _END_OF_GENERATION and self.types[i] in (_CONTROL, _USER_DEFINED))
        specials = [token for i, token in enumerate(self.tokens) if self.types[i] in (_CONTROL, _USER_DEFINED)
                    and token]
        self._special = re.compile("|".join(re.escape(t) for t in sorted(specials, key=len, reverse=True))) \
            if specials else None
        if self.model == "gpt2":
            self._init_bpe(metadata)
        elif self.model == "llama":
            self.scores = self._spm_scores(metadata)
            self.add_space_prefix = bool(metadata.get("tokenizer.ggml.add_space_prefix", True))
        else:
            raise ValueError("unsupported tokenizer.ggml.model {0!r} (supported: gpt2, llama)".format(self.model))

    def _init_bpe(self, metadata):
        from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

        pre = metadata.get("tokenizer.ggml.pre", "default")
        if pre not in BPE_PRE_TOKENIZERS:
            raise ValueError("unsupported tokenizer.ggml.pre {0!r} (supported: {1})".format(
                pre, ", ".join(sorted(BPE_PRE_TOKENIZERS))))
        pattern, nfc, ignore_merges = BPE_PRE_TOKENIZERS[pre]
        merges = [tuple(merge.split(" ", 1)) for merge in metadata["tokenizer.ggml.merges"]]
        bpe = Tokenizer(models.BPE(vocab=dict(self.ids), merges=merges, ignore_merges=ignore_merges))
        if nfc:
            bpe.normalizer = normalizers.NFC()
        bpe.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(pattern), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        bpe.decoder = decoders.ByteLevel()
        self._bpe = bpe

    def _spm_scores(self, metadata):
        scores = metadata.get("tokenizer.ggml.scores")
        merges = metadata.get("tokenizer.ggml.merges")
        if scores and len(set(scores)) > 1:
            return scores
        if not merges:
            return scores or [0.0] * len(self.tokens)
        # Some HF conversions store all-zero scores plus BPE merges; SentencePiece BPE merge
        # priority equals merge rank, and pieces no merge produces are never formed.
        derived = [None] * len(self.tokens)
        for rank, merge in enumerate(merges):
            token = self.ids.get(merge.replace(" ", "", 1))
            if token is not None and derived[token] is None:
                derived[token] = -float(rank)
        return derived

    def _fragments(self, text):
        """Split text around special-token strings: yields (is_special, text or id)."""
        position = 0
        if self._special is not None:
            for match in self._special.finditer(text):
                if match.start() > position:
                    yield False, text[position:match.start()]
                yield True, self.ids[match.group(0)]
                position = match.end()
        if position < len(text):
            yield False, text[position:]

    def encode(self, text, add_special=True):
        ids = [self.bos] if add_special and self.add_bos and self.bos is not None else []
        previous_special = True
        for special, fragment in self._fragments(text):
            if special:
                ids.append(fragment)
            elif self.model == "gpt2":
                ids.extend(self._bpe.encode(fragment, add_special_tokens=False).ids)
            else:
                if self.add_space_prefix and previous_special:
                    fragment = " " + fragment
                ids.extend(self._spm(fragment.replace(" ", "\u2581")))
            previous_special = special
        if add_special and self.add_eos and self.eos is not None:
            ids.append(self.eos)
        return ids

    def _spm(self, text):
        # llama.cpp llm_tokenizer_spm: merge the adjacent pair whose merged piece has the best score.
        symbols = [[i - 1, i + 1, ch] for i, ch in enumerate(text)]
        if not symbols:
            return []
        symbols[-1][1] = -1
        queue, merged_from = [], {}

        def add_bigram(left, right):
            if left < 0 or right < 0:
                return
            piece = symbols[left][2] + symbols[right][2]
            token = self.ids.get(piece)
            if token is not None and self.scores[token] is not None:
                heapq.heappush(queue, (-self.scores[token], left, piece))
                merged_from[piece] = (symbols[left][2], symbols[right][2])

        for i in range(1, len(symbols)):
            add_bigram(i - 1, i)
        while queue:
            _, left, piece = heapq.heappop(queue)
            right = symbols[left][1]
            if right < 0 or symbols[left][2] + symbols[right][2] != piece:
                continue
            symbols[left][2] = piece
            symbols[right][2] = ""
            symbols[left][1] = symbols[right][1]
            if symbols[right][1] >= 0:
                symbols[symbols[right][1]][0] = left
            add_bigram(symbols[left][0], left)
            add_bigram(left, symbols[left][1])
        output = []

        def resegment(piece):
            token = self.ids.get(piece)
            if token is not None:
                output.append(token)
            elif piece in merged_from:
                for part in merged_from[piece]:
                    resegment(part)
            else:
                for byte in piece.encode("utf-8"):
                    output.append(self.ids["<0x{0:02X}>".format(byte)])

        index = 0
        while index >= 0:
            if symbols[index][2]:
                resegment(symbols[index][2])
            index = symbols[index][1]
        return output

    def decode(self, ids):
        if self.model == "gpt2":
            return self._bpe.decode(list(ids), skip_special_tokens=False)
        pieces = bytearray()
        for token in ids:
            kind = self.types[token]
            if kind == _BYTE:
                pieces.append(int(self.tokens[token][3:-1], 16))
            elif kind in (_NORMAL, _USER_DEFINED):
                pieces.extend(self.tokens[token].replace("\u2581", " ").encode("utf-8"))
        return pieces.decode("utf-8", errors="replace")


def load_tokenizer(metadata):
    return GgufTokenizer(metadata)
