import ast
import struct
from typing import BinaryIO, Sequence


_VALUES_PER_BLOCK = 256
_BYTES_PER_BLOCK = 210


def decode_q6_k_row(source: BinaryIO, row: int, columns: int) -> list:
    if row < 0 or columns <= 0 or columns % _VALUES_PER_BLOCK:
        raise ValueError("Q6_K requires a nonnegative row and a multiple of 256 columns")
    source.seek(row * (columns // _VALUES_PER_BLOCK) * _BYTES_PER_BLOCK)
    values = []
    for _ in range(columns // _VALUES_PER_BLOCK):
        block = source.read(_BYTES_PER_BLOCK)
        if len(block) != _BYTES_PER_BLOCK:
            raise ValueError("truncated Q6_K row")
        scale = struct.unpack_from("<e", block, 208)[0]
        subscales = struct.unpack_from("<16b", block, 192)
        for position in range(_VALUES_PER_BLOCK):
            half = position // 128
            segment = (position % 128) // 32
            index = position % 32
            low_byte = block[half * 64 + (segment % 2) * 32 + index]
            high_byte = block[128 + half * 32 + index]
            quant = ((low_byte >> ((segment // 2) * 4)) & 15)
            quant |= ((high_byte >> (segment * 2)) & 3) << 4
            subscale = subscales[half * 8 + index // 16 + segment * 2]
            values.append(scale * subscale * (quant - 32))
    return values


def dot_q6_k_row(source: BinaryIO, row: int, vector: Sequence[float]) -> float:
    values = decode_q6_k_row(source, row, len(vector))
    return sum(value * activation for value, activation in zip(values, vector))


def generate_q6_k_matvec_program(rows: int, columns: int) -> str:
    if (rows <= 0 or columns <= 0 or columns % _VALUES_PER_BLOCK or
            rows * columns // _VALUES_PER_BLOCK > 2**31 - 1):
        raise ValueError("Q6_K matvec requires positive, block-aligned dimensions and 32-bit indexes")
    blocks_per_row = columns // _VALUES_PER_BLOCK
    source = """def q6_k_matvec_GPU_1(weights, activation, output, row):
    columns = {columns}
    blocks_per_row = {blocks_per_row}
    total = 0.0
    column = 0
    while column < columns:
        block_index = row * blocks_per_row + column / 256
        position = column % 256
        weight = GGUF_Q6_K_VALUE(weights, block_index, position)
        total = total + activation[column] * weight
        column = column + 1
    output[row] = total

LOAD_MEM(weights)
LOAD_MEM(activation)
LOAD_MEM(output)
q6_k_matvec_GPU_1(weights, activation, output, {rows})
GPU_SYNC(output)
RELEASE_MEM(weights)
RELEASE_MEM(activation)
RELEASE_MEM(output)
RETURN(output)
""".format(rows=rows, columns=columns, blocks_per_row=blocks_per_row)
    ast.parse(source)
    return source