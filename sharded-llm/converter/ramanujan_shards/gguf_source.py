import io
import re
import struct
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, Union
from urllib.request import Request, urlopen

from .interfaces import SourceReader


_VALUE_TYPES = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
    7: "?", 10: "Q", 11: "q", 12: "d",
}
_BLOCK_TYPES = {
    0: (1, 4, "F32"),
    1: (1, 2, "F16"),
    2: (32, 18, "Q4_0"),
    3: (32, 20, "Q4_1"),
    6: (32, 22, "Q5_0"),
    7: (32, 24, "Q5_1"),
    8: (32, 34, "Q8_0"),
    10: (256, 84, "Q2_K"),
    11: (256, 110, "Q3_K"),
    12: (256, 144, "Q4_K"),
    13: (256, 176, "Q5_K"),
    14: (256, 210, "Q6_K"),
    15: (256, 292, "Q8_K"),
    30: (1, 2, "BF16"),
}
_MAX_STRING = 16 * 1024 * 1024
_HEADER_CHUNK = 1024 * 1024


class _HTTPRangeSource:
    def __init__(self, url: str):
        self._url = url
        request = Request(url, headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"})
        with urlopen(request, timeout=30) as response:
            match = re.fullmatch(r"bytes 0-0/([0-9]+)", response.headers.get("Content-Range", ""))
            if response.status != 206 or not match:
                raise ValueError("GGUF HTTP source must support byte ranges")
            self.size = int(match.group(1))
            if self.size <= 0 or len(response.read(2)) != 1:
                raise ValueError("invalid GGUF HTTP source size")
        self._cache_offset = -1
        self._cache = b""

    def fetch(self, offset: int, length: int) -> bytes:
        if length <= 0 or offset < 0 or offset + length > self.size:
            raise ValueError("GGUF range outside source")
        end = offset + length - 1
        request = Request(self._url, headers={
            "Range": "bytes={0}-{1}".format(offset, end), "Accept-Encoding": "identity"
        })
        with urlopen(request, timeout=60) as response:
            expected = "bytes {0}-{1}/{2}".format(offset, end, self.size)
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError("GGUF HTTP source returned an invalid byte range")
            data = response.read(length + 1)
        if len(data) != length:
            raise ValueError("truncated GGUF HTTP range")
        return data

    def header_bytes(self, offset: int, length: int) -> bytes:
        result = bytearray()
        while length:
            block_start = offset // _HEADER_CHUNK * _HEADER_CHUNK
            if block_start != self._cache_offset:
                self._cache_offset = block_start
                self._cache = self.fetch(block_start, min(_HEADER_CHUNK, self.size - block_start))
            count = min(length, len(self._cache) - (offset - block_start))
            result.extend(self._cache[offset - block_start:offset - block_start + count])
            offset += count
            length -= count
        return bytes(result)


class _HTTPStream(io.RawIOBase):
    def __init__(self, remote: _HTTPRangeSource, start: int = 0, length: int = -1):
        self._remote = remote
        self._start = start
        self._length = remote.size - start if length < 0 else length
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if self.closed:
            raise ValueError("I/O operation on closed GGUF stream")
        origin = {io.SEEK_SET: 0, io.SEEK_CUR: self._position, io.SEEK_END: self._length}.get(whence)
        if origin is None or not 0 <= origin + offset <= self._length:
            raise ValueError("seek outside GGUF stream")
        self._position = origin + offset
        return self._position

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed GGUF stream")
        remaining = self._length - self._position
        count = remaining if size is None or size < 0 else min(remaining, size)
        if not count:
            return b""
        offset = self._start + self._position
        data = (self._remote.header_bytes(offset, count) if self._start == 0
                else self._remote.fetch(offset, count))
        self._position += len(data)
        return data


class _TensorSlice(io.RawIOBase):
    def __init__(self, path: Path, offset: int, length: int):
        self._file = path.open("rb")
        self._start = offset
        self._length = length
        self._position = 0
        self._file.seek(offset)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if self.closed:
            raise ValueError("I/O operation on closed tensor")
        origin = {io.SEEK_SET: 0, io.SEEK_CUR: self._position, io.SEEK_END: self._length}.get(whence)
        if origin is None:
            raise ValueError("invalid seek mode")
        position = origin + offset
        if position < 0 or position > self._length:
            raise ValueError("seek outside tensor")
        self._file.seek(self._start + position)
        self._position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed tensor")
        remaining = self._length - self._position
        data = self._file.read(remaining if size is None or size < 0 else min(size, remaining))
        self._position += len(data)
        return data

    def close(self) -> None:
        if not self.closed:
            self._file.close()
        super().close()


class GGUFSourceReader(SourceReader):
    def __init__(self, path: Union[str, Path]):
        location = str(path)
        self._remote = _HTTPRangeSource(location) if location.startswith(("http://", "https://")) else None
        self._path = None if self._remote else Path(path)
        self._tensors: Dict[str, Dict[str, Any]] = {}
        with (_HTTPStream(self._remote) if self._remote else self._path.open("rb")) as stream:
            if _read_exact(stream, 4) != b"GGUF":
                raise ValueError("invalid GGUF magic")
            version = _read_number(stream, "I")
            if version not in (2, 3):
                raise ValueError("unsupported GGUF version: {0}".format(version))
            tensor_count = _read_number(stream, "Q")
            metadata_count = _read_number(stream, "Q")
            self._metadata = {"gguf_version": version, "source_format": "gguf"}
            for _ in range(metadata_count):
                key = _read_string(stream)
                if key in self._metadata:
                    raise ValueError("duplicate GGUF metadata key: {0}".format(key))
                self._metadata[key] = _read_value(stream, _read_number(stream, "I"))
            descriptors = []
            for _ in range(tensor_count):
                name = _read_string(stream)
                dimension_count = _read_number(stream, "I")
                if dimension_count == 0 or dimension_count > 4:
                    raise ValueError("invalid GGUF tensor rank: {0}".format(name))
                dimensions = [_read_number(stream, "Q") for _ in range(dimension_count)]
                type_id = _read_number(stream, "I")
                relative_offset = _read_number(stream, "Q")
                if name in self._tensors:
                    raise ValueError("duplicate GGUF tensor: {0}".format(name))
                block_size, block_bytes, type_name = _BLOCK_TYPES.get(
                    type_id, (1, 0, "GGML_TYPE_{0}".format(type_id))
                )
                elements = 1
                for dimension in dimensions:
                    if dimension == 0:
                        raise ValueError("zero-sized GGUF tensor: {0}".format(name))
                    elements *= dimension
                if dimensions[0] % block_size:
                    raise ValueError("GGUF tensor row is not block-aligned: {0}".format(name))
                length = elements // block_size * block_bytes if block_bytes else None
                self._tensors[name] = {
                    "shape": list(reversed(dimensions)),
                    "gguf_dimensions": dimensions,
                    "dtype": type_name,
                    "ggml_type": type_id,
                    "length": length,
                    "relative_offset": relative_offset,
                }
                descriptors.append((relative_offset, name))
            alignment = self._metadata.get("general.alignment", 32)
            if not isinstance(alignment, int) or alignment <= 0 or alignment & (alignment - 1):
                raise ValueError("invalid GGUF alignment")
            data_start = (stream.tell() + alignment - 1) // alignment * alignment
            stream.seek(0, io.SEEK_END)
            file_size = stream.tell()
            previous_end = 0
            ordered = sorted(descriptors)
            for index, (relative_offset, name) in enumerate(ordered):
                tensor = self._tensors[name]
                next_offset = (ordered[index + 1][0] if index + 1 < len(ordered)
                               else file_size - data_start)
                if tensor["length"] is None:
                    tensor["length"] = next_offset - relative_offset
                    tensor["length_includes_padding"] = True
                length = tensor["length"]
                if relative_offset % alignment or relative_offset < previous_end:
                    raise ValueError("overlapping or unaligned GGUF tensor: {0}".format(name))
                if length < 0 or (index + 1 < len(ordered) and relative_offset + length > next_offset):
                    raise ValueError("overlapping GGUF tensor: {0}".format(name))
                absolute_offset = data_start + relative_offset
                if absolute_offset + length > file_size:
                    raise ValueError("truncated GGUF tensor: {0}".format(name))
                tensor["offset"] = absolute_offset
                previous_end = relative_offset + length

    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    def tensor_names(self) -> Iterable[str]:
        return self._tensors.keys()

    def tensor_metadata(self, name: str) -> Dict[str, Any]:
        return dict(self._tensors[name])

    def open_tensor(self, name: str) -> BinaryIO:
        tensor = self._tensors[name]
        if self._remote:
            return _HTTPStream(self._remote, tensor["offset"], tensor["length"])
        return _TensorSlice(self._path, tensor["offset"], tensor["length"])


def _read_exact(stream: BinaryIO, length: int) -> bytes:
    if length > _MAX_STRING:
        raise ValueError("GGUF field exceeds maximum size")
    data = stream.read(length)
    if len(data) != length:
        raise ValueError("truncated GGUF header")
    return data


def _read_number(stream: BinaryIO, fmt: str) -> Any:
    return struct.unpack("<" + fmt, _read_exact(stream, struct.calcsize(fmt)))[0]


def _read_string(stream: BinaryIO) -> str:
    return _read_exact(stream, _read_number(stream, "Q")).decode("utf-8")


def _read_value(stream: BinaryIO, value_type: int) -> Any:
    if value_type in _VALUE_TYPES:
        return _read_number(stream, _VALUE_TYPES[value_type])
    if value_type == 8:
        return _read_string(stream)
    if value_type == 9:
        item_type = _read_number(stream, "I")
        count = _read_number(stream, "Q")
        if item_type == 9 or count > 1_000_000:
            raise ValueError("invalid GGUF metadata array")
        return [_read_value(stream, item_type) for _ in range(count)]
    raise ValueError("unsupported GGUF metadata type: {0}".format(value_type))