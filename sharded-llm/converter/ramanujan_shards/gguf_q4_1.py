import ast
import struct
from typing import BinaryIO, Sequence


_VALUES_PER_BLOCK = 32
_BYTES_PER_BLOCK = 20


def decode_q4_1_row(source: BinaryIO, row: int, columns: int) -> list:
    if row < 0 or columns <= 0 or columns % _VALUES_PER_BLOCK:
        raise ValueError("Q4_1 requires a nonnegative row and a multiple of 32 columns")
    blocks_per_row = columns // _VALUES_PER_BLOCK
    source.seek(row * blocks_per_row * _BYTES_PER_BLOCK)
    values = []
    for _ in range(blocks_per_row):
        block = source.read(_BYTES_PER_BLOCK)
        if len(block) != _BYTES_PER_BLOCK:
            raise ValueError("truncated Q4_1 row")
        scale, minimum = struct.unpack_from("<ee", block)
        for nibble_shift in (0, 4):
            values.extend(minimum + scale * ((byte >> nibble_shift) & 15)
                          for byte in block[4:])
    return values


def dot_q4_1_row(source: BinaryIO, row: int, vector: Sequence[float]) -> float:
    values = decode_q4_1_row(source, row, len(vector))
    return sum(value * activation for value, activation in zip(values, vector))


def generate_q4_1_matvec_program(rows: int, columns: int) -> str:
    if rows <= 0 or columns <= 0 or columns % _VALUES_PER_BLOCK or rows * columns // _VALUES_PER_BLOCK > 2**31 - 1:
        raise ValueError("Q4_1 matvec requires positive, block-aligned dimensions and 32-bit indexes")
    blocks_per_row = columns // _VALUES_PER_BLOCK
    source = """def q4_1_matvec_GPU_1(weights, activation, output, row):
    columns = {columns}
    blocks_per_row = {blocks_per_row}
    total = 0.0
    column = 0
    while column < columns:
        block_index = row * blocks_per_row + column / 32
        position = column % 32
        weight = GGUF_Q4_1_VALUE(weights, block_index, position)
        total = total + activation[column] * weight
        column = column + 1
    output[row] = total

LOAD_MEM(weights)
LOAD_MEM(activation)
LOAD_MEM(output)
q4_1_matvec_GPU_1(weights, activation, output, {rows})
GPU_SYNC(output)
RELEASE_MEM(weights)
RELEASE_MEM(activation)
RELEASE_MEM(output)
RETURN(output)
""".format(rows=rows, columns=columns, blocks_per_row=blocks_per_row)
    ast.parse(source)
    return source