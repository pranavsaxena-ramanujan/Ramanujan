"""ctypes binding for libramanujan_llm (the native OpenCL LLM runtime)."""
import ctypes
import json
import os
import sys
from pathlib import Path

import numpy as np

from .llm_graph import dumps

_LIBRARY_NAMES = {"darwin": "libramanujan_llm.dylib", "win32": "ramanujan_llm.dll"}


def library_name():
    return _LIBRARY_NAMES.get(sys.platform, "libramanujan_llm.so")


def find_library(search=None):
    """Path to libramanujan_llm: $RJLLM_LIBRARY, then `search` dirs, then the in-repo build dir."""
    if os.environ.get("RJLLM_LIBRARY"):
        return Path(os.environ["RJLLM_LIBRARY"])
    candidates = [Path(directory) for directory in (search or [])]
    candidates.append(Path(__file__).resolve().parents[3] / "ramanujan-native" / "native" / "build")
    for directory in candidates:
        path = directory / library_name()
        if path.exists():
            return path
    return None


_lib = None


def load_library(path=None):
    global _lib
    if _lib is not None:
        return _lib
    path = Path(path) if path else find_library()
    if path is None or not path.exists():
        raise OSError("libramanujan_llm not found; build it with `make ramanujan_llm` in ramanujan-native/native/build "
                      "or set RJLLM_LIBRARY")
    lib = ctypes.CDLL(str(path))
    lib.rjllm_open.restype = ctypes.c_void_p
    lib.rjllm_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.rjllm_output_size.restype = ctypes.c_size_t
    lib.rjllm_output_size.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.rjllm_step.restype = ctypes.c_int
    lib.rjllm_step.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                               ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
    lib.rjllm_reset.argtypes = [ctypes.c_void_p]
    lib.rjllm_info.restype = ctypes.c_char_p
    lib.rjllm_info.argtypes = [ctypes.c_void_p]
    lib.rjllm_close.argtypes = [ctypes.c_void_p]
    _lib = lib
    return lib


class NativeStage:
    """One pipeline stage executing in-process on the native runtime."""

    def __init__(self, graph, library=None):
        self.lib = load_library(library)
        self.graph = graph
        self.embed = "embed" in graph
        self.head = "head" in graph
        self.dim = graph["hyper"]["dim"]
        error = ctypes.create_string_buffer(4096)
        self.handle = self.lib.rjllm_open(dumps(graph).encode("utf-8"), error, len(error))
        if not self.handle:
            raise RuntimeError("rjllm_open failed: " + error.value.decode("utf-8", "replace"))
        self.position = 0

    def step(self, tokens=None, hidden=None):
        """Advance by a list of token ids (embedding stage) or an (n, dim) hidden array; returns
        the last token's logits for a head stage, else (n, dim) hidden states."""
        if self.embed:
            ids = np.ascontiguousarray(tokens, dtype=np.int32)
            n = len(ids)
            tokens_ptr, hidden_ptr = ids.ctypes.data, None
        else:
            states = np.ascontiguousarray(hidden, dtype=np.float32).reshape(-1, self.dim)
            n = len(states)
            tokens_ptr, hidden_ptr = None, states.ctypes.data
        out = np.empty(self.lib.rjllm_output_size(self.handle, n), np.float32)
        error = ctypes.create_string_buffer(4096)
        status = self.lib.rjllm_step(self.handle, tokens_ptr, hidden_ptr, n, self.position, out.ctypes.data, out.size,
                                     error, len(error))
        if status:
            raise RuntimeError("rjllm_step failed: " + error.value.decode("utf-8", "replace"))
        self.position += n
        return out if self.head else out.reshape(n, self.dim)

    def reset(self):
        self.lib.rjllm_reset(self.handle)
        self.position = 0

    def info(self):
        return json.loads(self.lib.rjllm_info(self.handle).decode("utf-8"))

    def close(self):
        if self.handle:
            self.lib.rjllm_close(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
