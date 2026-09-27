"""Execute generated Ramanujan DSL programs in Python (test oracle for program generators).

Kernels (``*_GPU_1``) run sequentially over their work items, which matches the in-order
OpenCL queue for the data-parallel kernels the generators emit. Integer division between
integers truncates like the translated C code, array indexes are cast to int, and the
GGUF_*_VALUE intrinsics decode through the NumPy reference dequantizers.
"""
import ast
import math

import numpy as np

from .llm_reference import dequantize
from .llm_spec import QUANT

_INTRINSICS = {kind.intrinsic: encoding for encoding, kind in QUANT.items() if kind.intrinsic}
_SKIPPED = {"LOAD_MEM", "GPU_SYNC", "RELEASE_MEM"}


def _div(a, b):
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(a) // int(b)
    return a / b


def _ix(value):
    return int(value)


def _exp(value):
    return math.exp(value) if value < 709.0 else math.inf


class _Rewrite(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Div):
            return ast.copy_location(ast.Call(ast.Name("_div", ast.Load()), [node.left, node.right], []), node)
        return node

    def visit_Subscript(self, node):
        self.generic_visit(node)
        node.slice = ast.Call(ast.Name("_ix", ast.Load()), [node.slice], [])
        return node


def run_program(source, arrays):
    """Run ``source`` with named input arrays; returns {name: array} for RETURN(...)."""
    tree = ast.parse(source)
    arrays = {name: np.array(value, np.float32).reshape(-1) for name, value in arrays.items()}
    decoded = {}

    def intrinsic(encoding):
        kind = QUANT[encoding]

        def value(weights, block, position):
            key = id(weights)
            if key not in decoded:
                raw = weights.view(np.uint8)
                decoded[key] = dequantize(encoding, raw.reshape(-1, kind.block_bytes), kind.block)
            return float(decoded[key][int(block), int(position)])
        return value

    namespace = {"_div": _div, "_ix": _ix, "sqrt": math.sqrt, "exp": _exp, "log1p": math.log1p,
                 "fmax": max, "pow": math.pow, "cos": math.cos, "sin": math.sin}
    namespace.update({name: intrinsic(encoding) for name, encoding in _INTRINSICS.items()})
    returned = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            module = ast.fix_missing_locations(ast.Module([_Rewrite().visit(node)], []))
            exec(compile(module, "<kernel>", "exec"), namespace)
        elif isinstance(node, ast.Assign):
            name = node.targets[0].id
            size = eval(compile(ast.Expression(node.value.generators[0].iter.args[0]), "<size>", "eval"))
            if name in arrays:
                raise ValueError("scratch array shadows an input: {0}".format(name))
            arrays[name] = np.zeros(size, np.float32)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            callee = call.func.id
            args = [arg.id if isinstance(arg, ast.Name) else ast.literal_eval(arg) for arg in call.args]
            if callee in _SKIPPED:
                if args[0] not in arrays:
                    raise ValueError("program uses an unbound array: {0}".format(args[0]))
            elif callee == "RETURN":
                returned = {name: arrays[name] for name in args}
            else:
                kernel = namespace[callee]
                bound = [arrays[name] for name in args[:-1]]
                for work_item in range(args[-1]):
                    kernel(*bound, work_item)
        else:
            raise ValueError("unexpected top-level statement: {0}".format(ast.dump(node)[:80]))
    return returned
