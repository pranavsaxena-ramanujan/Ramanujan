import io
import json
import struct
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, Tuple

from .interfaces import SourceReader


class _TensorSlice(io.RawIOBase):
    def __init__(self, path: Path, offset: int, length: int):
        self._file = path.open("rb")
        self._file.seek(offset)
        self._remaining = length

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if self.closed or self._remaining == 0:
            return b""
        requested = self._remaining if size is None or size < 0 else min(size, self._remaining)
        data = self._file.read(requested)
        self._remaining -= len(data)
        return data

    def close(self) -> None:
        if not self.closed:
            self._file.close()
        super().close()


class SafeTensorsSourceReader(SourceReader):
    def __init__(self, model_dir: Path):
        self._model_dir = Path(model_dir)
        with (self._model_dir / "config.json").open("r", encoding="utf-8") as stream:
            self._config = json.load(stream)
        with (self._model_dir / "model.safetensors.index.json").open("r", encoding="utf-8") as stream:
            index = json.load(stream)
        self._weight_map = index["weight_map"]
        self._index_metadata = index.get("metadata", {})
        self._headers: Dict[str, Tuple[Dict[str, Any], int]] = {}

    def metadata(self) -> Dict[str, Any]:
        return {
            "config": dict(self._config),
            "index": dict(self._index_metadata),
            "source_format": "safetensors",
        }

    def tensor_names(self) -> Iterable[str]:
        return self._weight_map.keys()

    def tensor_metadata(self, name: str) -> Dict[str, Any]:
        header, _ = self._header_for_tensor(name)
        return dict(header[name])

    def open_tensor(self, name: str) -> BinaryIO:
        header, data_offset = self._header_for_tensor(name)
        tensor = header[name]
        start, end = tensor["data_offsets"]
        return _TensorSlice(
            self._model_dir / self._weight_map[name],
            data_offset + start,
            end - start,
        )

    def _header_for_tensor(self, name: str) -> Tuple[Dict[str, Any], int]:
        if name not in self._weight_map:
            raise KeyError(name)
        file_name = self._weight_map[name]
        cached = self._headers.get(file_name)
        if cached is not None:
            return cached

        with (self._model_dir / file_name).open("rb") as stream:
            header_size_data = stream.read(8)
            if len(header_size_data) != 8:
                raise ValueError("invalid safetensors header in {0}".format(file_name))
            header_size = struct.unpack("<Q", header_size_data)[0]
            header = json.loads(stream.read(header_size).decode("utf-8"))
        cached = (header, 8 + header_size)
        self._headers[file_name] = cached
        return cached