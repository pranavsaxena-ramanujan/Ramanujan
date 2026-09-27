from .contracts import AdapterGraph, ShardPlan, StageDefinition
from .gguf_adapter import GGUFArchitectureAdapter
from .gguf_emitter import emit_gguf_package
from .gguf_source import GGUFSourceReader
from .interfaces import ArchitectureAdapter, SourceReader, TensorEncoder
from .phi3_adapter import Phi3ArchitectureAdapter
from .planner import plan_shards
from .qwen_adapter import Qwen38ArchitectureAdapter
from .qwen_emitter import emit_qwen_package
from .safetensors_source import SafeTensorsSourceReader


def verify_gguf_package(package_dir):
    from .verify_gguf import verify_gguf_package as verify

    return verify(package_dir)

__all__ = [
    "AdapterGraph",
    "ArchitectureAdapter",
    "GGUFArchitectureAdapter",
    "GGUFSourceReader",
    "Phi3ArchitectureAdapter",
    "Qwen38ArchitectureAdapter",
    "SafeTensorsSourceReader",
    "ShardPlan",
    "SourceReader",
    "StageDefinition",
    "TensorEncoder",
    "emit_gguf_package",
    "emit_qwen_package",
    "plan_shards",
    "verify_gguf_package",
]