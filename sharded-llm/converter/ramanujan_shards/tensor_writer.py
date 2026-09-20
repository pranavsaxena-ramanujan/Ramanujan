from pathlib import Path
from typing import BinaryIO, Dict, Tuple

import numpy as np


_SOURCE_DTYPES = {
    "BF16": (np.dtype("<u2"), 2),
    "F16": (np.dtype("<f2"), 2),
    "F32": (np.dtype("<f4"), 4),
}


def write_float32_tensor(source: BinaryIO, metadata: Dict, destination: Path,
                         chunk_elements: int = 1_048_576) -> int:
    source_dtype, source_bytes = _source_dtype(metadata)
    remaining = _element_count(metadata["shape"])
    written = 0
    with destination.open("wb") as output:
        while remaining:
            count = min(chunk_elements, remaining)
            raw = _read_exact(source, count * source_bytes)
            values = _decode(raw, source_dtype, metadata["dtype"])
            output.write(values.astype("<f4", copy=False).tobytes())
            written += values.size * 4
            remaining -= count
    return written


def write_packed_4bit_matrix(source: BinaryIO, metadata: Dict, packed_destination: Path,
                             scales_destination: Path, chunk_rows: int = 128) -> Tuple[int, int]:
    shape = metadata.get("shape", [])
    if len(shape) != 2:
        raise ValueError("4-bit packing requires a rank-2 tensor")
    rows, columns = shape
    source_dtype, source_bytes = _source_dtype(metadata)
    packed_columns = (columns + 5) // 6
    packed_bytes = 0
    scale_bytes = 0

    with packed_destination.open("wb") as packed_output, scales_destination.open("wb") as scale_output:
        for row_start in range(0, rows, chunk_rows):
            row_count = min(chunk_rows, rows - row_start)
            raw = _read_exact(source, row_count * columns * source_bytes)
            values = _decode(raw, source_dtype, metadata["dtype"]).reshape(row_count, columns)
            scales = np.max(np.abs(values), axis=1).astype(np.float32) / np.float32(7.0)
            scales[scales == 0] = np.float32(1.0)
            quantized = np.rint(values / scales[:, None])
            np.clip(quantized, -8, 7, out=quantized)
            quantized += np.float32(8.0)
            quantized = quantized.astype(np.uint32)
            if columns % 6:
                quantized = np.pad(quantized, ((0, 0), (0, 6 - columns % 6)), constant_values=8)
            quantized = quantized.reshape(row_count, packed_columns, 6)
            packed = np.zeros((row_count, packed_columns), dtype=np.uint32)
            for nibble in range(6):
                packed += quantized[:, :, nibble] << np.uint32(4 * nibble)
            packed_float = packed.astype("<f4")
            scale_float = scales.astype("<f4", copy=False)
            packed_output.write(packed_float.tobytes())
            scale_output.write(scale_float.tobytes())
            packed_bytes += packed_float.nbytes
            scale_bytes += scale_float.nbytes
    return packed_bytes, scale_bytes


def _source_dtype(metadata: Dict):
    data_type = metadata.get("dtype")
    if data_type not in _SOURCE_DTYPES:
        raise ValueError("unsupported safetensors dtype: {0}".format(data_type))
    return _SOURCE_DTYPES[data_type]


def _decode(raw: bytes, source_dtype: np.dtype, data_type: str) -> np.ndarray:
    values = np.frombuffer(raw, dtype=source_dtype)
    if data_type == "BF16":
        expanded = values.astype(np.uint32)
        expanded <<= np.uint32(16)
        return expanded.view(np.float32)
    return values.astype(np.float32, copy=data_type != "F32")


def _read_exact(source: BinaryIO, size: int) -> bytes:
    data = source.read(size)
    if len(data) != size:
        raise ValueError("unexpected end of tensor: expected {0} bytes, got {1}".format(size, len(data)))
    return data


def _element_count(shape) -> int:
    count = 1
    for dimension in shape:
        count *= dimension
    return count