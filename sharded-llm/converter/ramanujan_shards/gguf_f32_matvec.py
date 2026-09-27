import ast
import struct
from typing import BinaryIO, Sequence


def dot_f32_row(source: BinaryIO, row: int, vector: Sequence[float]) -> float:
    if row < 0 or not vector:
        raise ValueError("F32 matvec requires a nonnegative row and nonempty activation")
    source.seek(row * len(vector) * 4)
    data = source.read(len(vector) * 4)
    if len(data) != len(vector) * 4:
        raise ValueError("truncated F32 row")
    weights = struct.unpack("<{0}f".format(len(vector)), data)
    return sum(weight * activation for weight, activation in zip(weights, vector))


def generate_f32_matvec_program(rows: int, columns: int) -> str:
    if rows <= 0 or columns <= 0 or rows * columns > 2**31 - 1:
        raise ValueError("F32 matvec requires positive dimensions and 32-bit indexes")
    source = """def f32_matvec_GPU_1(weights, activation, output, row):
    columns = {columns}
    total = 0.0
    column = 0
    while column < columns:
        total = total + weights[row * columns + column] * activation[column]
        column = column + 1
    output[row] = total

LOAD_MEM(weights)
LOAD_MEM(activation)
LOAD_MEM(output)
f32_matvec_GPU_1(weights, activation, output, {rows})
GPU_SYNC(output)
RELEASE_MEM(weights)
RELEASE_MEM(activation)
RELEASE_MEM(output)
RETURN(output)
""".format(rows=rows, columns=columns)
    ast.parse(source)
    return source