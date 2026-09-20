from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable

from .contracts import AdapterGraph, ShardPlan


class SourceReader(ABC):
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def tensor_names(self) -> Iterable[str]:
        raise NotImplementedError

    @abstractmethod
    def tensor_metadata(self, name: str) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def open_tensor(self, name: str) -> BinaryIO:
        raise NotImplementedError


class ArchitectureAdapter(ABC):
    @abstractmethod
    def build_graph(self, source: SourceReader) -> AdapterGraph:
        raise NotImplementedError

    @abstractmethod
    def emit_shard(self, source: SourceReader, plan: ShardPlan, output_dir: Path) -> None:
        raise NotImplementedError


class TensorEncoder(ABC):
    @property
    @abstractmethod
    def encoding_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def encode(self, source: BinaryIO, destination: BinaryIO, metadata: Dict[str, Any]) -> None:
        raise NotImplementedError