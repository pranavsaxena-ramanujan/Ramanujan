#!/usr/bin/env python3
"""Compatibility entry point: Qwen35 packages now run through the generic run_gguf_shards.py."""
from run_gguf_shards import *  # noqa: F401,F403
from run_gguf_shards import GgufRunner, main

Qwen35Runner = GgufRunner

if __name__ == "__main__":
    main()
