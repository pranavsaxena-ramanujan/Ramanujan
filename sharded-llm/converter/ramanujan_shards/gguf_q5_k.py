import ast
import struct
from typing import BinaryIO, Sequence


_VALUES_PER_BLOCK = 256
_BYTES_PER_BLOCK = 176


def decode_q5_k_row(source: BinaryIO, row: int, columns: int) -> list:
    if row < 0 or columns <= 0 or columns % _VALUES_PER_BLOCK:
        raise ValueError("Q5_K requires a nonnegative row and a multiple of 256 columns")
    source.seek(row * (columns // _VALUES_PER_BLOCK) * _BYTES_PER_BLOCK)
    values = []
    for _ in range(columns // _VALUES_PER_BLOCK):
        block = source.read(_BYTES_PER_BLOCK)
        if len(block) != _BYTES_PER_BLOCK:
            raise ValueError("truncated Q5_K row")
        scale, minimum = struct.unpack_from("<ee", block)
        scales = block[4:16]
        high_bits = block[16:48]
        low_bits = block[48:176]
        for group in range(8):
            if group < 4:
                group_scale = scales[group] & 63
                group_minimum = scales[group + 4] & 63
            else:
                group_scale = (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4)
                group_minimum = (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4)
            for index in range(32):
                quant_byte = low_bits[(group // 2) * 32 + index]
                quant = ((quant_byte >> ((group % 2) * 4)) & 15)
                quant += 16 if high_bits[index] & (1 << group) else 0
                values.append(scale * group_scale * quant - minimum * group_minimum)
    return values


def dot_q5_k_row(source: BinaryIO, row: int, vector: Sequence[float]) -> float:
    values = decode_q5_k_row(source, row, len(vector))
    return sum(value * activation for value, activation in zip(values, vector))


def generate_q5_k_matvec_program(rows: int, columns: int) -> str:
    if (rows <= 0 or columns <= 0 or columns % _VALUES_PER_BLOCK or
            rows * columns // _VALUES_PER_BLOCK > 2**31 - 1):
        raise ValueError("Q5_K matvec requires positive, block-aligned dimensions and 32-bit indexes")
    blocks_per_row = columns // _VALUES_PER_BLOCK
    source = """def q5_k_matvec_GPU_1(weights, activation, output, row):
    columns = {columns}
    blocks_per_row = {blocks_per_row}
    total = 0.0
    column = 0
    while column < columns:
        block_index = row * blocks_per_row + column / 256
        position = column % 256
        weight = GGUF_Q5_K_VALUE(weights, block_index, position)
        total = total + activation[column] * weight
        column = column + 1
    output[row] = total

LOAD_MEM(weights)
LOAD_MEM(activation)
LOAD_MEM(output)
q5_k_matvec_GPU_1(weights, activation, output, {rows})
GPU_SYNC(output)
RELEASE_MEM(weights)
RELEASE_MEM(activation)
RELEASE_MEM(output)
RETURN(output)
""".format(rows=rows, columns=columns, blocks_per_row=blocks_per_row)
    ast.parse(source)
    return source