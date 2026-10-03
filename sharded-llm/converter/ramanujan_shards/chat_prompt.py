"""Fits a multi-turn chat into the model's context window."""
import json


def fit_chat_prompt(tokenizer, turns, suffix="", prefix="", budget=1024):
    """Joins prefix + turns + suffix, dropping the oldest user/assistant pairs until it fits.

    `turns` are already-rendered chat turns alternating user/assistant and ending with the
    current user turn. Returns (prompt, token ids, dropped turn count); raises ValueError
    when even the last user turn does not fit in `budget` tokens.
    """
    if not turns or len(turns) % 2 == 0:
        raise ValueError("chat turns must alternate user/assistant and end with a user turn")
    start = 0
    while True:
        prompt = prefix + "".join(turns[start:]) + suffix
        tokens = tokenizer.encode(prompt)
        if tokens and len(tokens) <= budget:
            return prompt, tokens, start
        if start + 1 >= len(turns):
            raise ValueError("prompt plus new tokens must fit in --max-context")
        start += 2


def read_chat_turns(source):
    request = json.loads(source)
    turns = request.get("turns")
    if not isinstance(turns, list) or not all(isinstance(turn, str) for turn in turns):
        raise ValueError("--prompt-turns expects JSON {\"turns\": [strings], \"suffix\": str, \"prefix\": str}")
    return turns, str(request.get("suffix", "")), str(request.get("prefix", ""))
