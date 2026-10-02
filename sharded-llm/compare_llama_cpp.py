#!/usr/bin/env python3
"""Compare the generic NumPy reference (llm_reference.py) with llama.cpp on a GGUF model.

Needs `pip install llama-cpp-python gguf`. Checks tokenization, per-position logits over the
prompt, and optionally llama.cpp's greedy continuation (to compare with run_gguf_shards.py).
Ramanujan's OpenCL output is checked against the same NumPy reference with
`run_gguf_shards.py --check-layers` / `--reference-token`.
"""
import argparse
import json
import time

import numpy as np

from converter.ramanujan_shards.llm_package import add_model_arguments, load_model_from_args
from converter.ramanujan_shards.llm_reference import LlmReference
from converter.ramanujan_shards.llm_tokenizer import load_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gguf", required=True, help="original .gguf file")
    parser.add_argument("--package", required=True, help="shard package emitted from that file")
    parser.add_argument("--metadata", required=True, help="gguf-metadata.json from gguf_ir_plan")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--greedy-tokens", type=int, default=0,
                        help="also print llama.cpp's greedy continuation of this many tokens")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gpu-layers", type=int, default=0,
                        help="llama.cpp layers to offload (-1 = all, e.g. Metal). The CPU backend rounds "
                             "activations to 8 bits, so GPU logits are a much tighter comparison")
    add_model_arguments(parser)
    args = parser.parse_args()
    from llama_cpp import Llama

    spec, _, tensors, metadata = load_model_from_args(args)
    tokenizer = load_tokenizer(metadata)
    ids = tokenizer.encode(args.prompt)
    llm = Llama(model_path=args.gguf, n_ctx=max(64, len(ids) + args.greedy_tokens + 1), n_gpu_layers=args.gpu_layers,
                logits_all=True, verbose=False, n_threads=args.threads)
    # add_bos=True lets llama.cpp apply the model's own BOS rule, so a wrong tokenizer default shows up here.
    expected_ids = llm.tokenize(args.prompt.encode("utf-8"), add_bos=True, special=True)
    print(json.dumps({"event": "tokens", "architecture": spec.architecture, "ours": ids,
                      "llama.cpp": expected_ids, "match": ids == expected_ids}), flush=True)
    llm.eval(ids)
    expected = np.array(llm.scores[:len(ids)])
    reference = LlmReference(spec, tensors)
    states = [reference.new_state(layer) for layer in range(len(spec.layers))]
    started = time.monotonic()
    worst_correlation, argmax_matches = 1.0, 0
    for position, token in enumerate(ids):
        hidden = reference.embed(token)
        for layer in range(len(spec.layers)):
            hidden = reference.layer(layer, hidden, states[layer], position)
        logits, target = reference.logits(hidden), expected[position]
        correlation = float(np.corrcoef(logits, target)[0, 1])
        worst_correlation = min(worst_correlation, correlation)
        argmax_matches += int(np.argmax(logits) == np.argmax(target))
        print(json.dumps({"event": "position", "position": position, "argmax": int(np.argmax(logits)),
                          "llama.cpp": int(np.argmax(target)), "correlation": round(correlation, 6)}),
              flush=True)
    print(json.dumps({"event": "summary", "argmaxMatches": argmax_matches, "positions": len(ids),
                      "worstCorrelation": round(worst_correlation, 6),
                      "next": tokenizer.decode([int(np.argmax(logits))]),
                      "referenceSeconds": round(time.monotonic() - started, 1)}), flush=True)
    if args.greedy_tokens:
        generated = []
        for _ in range(args.greedy_tokens):
            token = int(np.argmax(llm.scores[llm.n_tokens - 1]))
            generated.append(token)
            if token in tokenizer.stop_ids:
                break
            llm.eval([token])
        print(json.dumps({"event": "llama.cpp-greedy", "ids": generated,
                          "text": tokenizer.decode(generated)}), flush=True)


if __name__ == "__main__":
    main()
