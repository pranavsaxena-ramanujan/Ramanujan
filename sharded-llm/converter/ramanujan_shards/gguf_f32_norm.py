import ast
import math


def generate_f32_rmsnorm_program(dimension: int, epsilon: float) -> str:
    if dimension <= 0 or dimension > 2**31 - 1 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("RMSNorm requires a positive dimension and finite positive epsilon")
    source = """def f32_rmsnorm_GPU_1(hidden, gamma, output, work_item):
    dimension = {dimension}
    squares = 0.0
    index = 0
    while index < dimension:
        squares = squares + hidden[index] * hidden[index]
        index = index + 1
    inv_rms = 1.0 / sqrt(squares / dimension + {epsilon})
    index = 0
    while index < dimension:
        output[index] = hidden[index] * inv_rms * gamma[index]
        index = index + 1

LOAD_MEM(hidden)
LOAD_MEM(gamma)
LOAD_MEM(output)
f32_rmsnorm_GPU_1(hidden, gamma, output, 1)
GPU_SYNC(output)
RELEASE_MEM(hidden)
RELEASE_MEM(gamma)
RELEASE_MEM(output)
RETURN(output)
""".format(dimension=dimension, epsilon=repr(epsilon))
    ast.parse(source)
    return source