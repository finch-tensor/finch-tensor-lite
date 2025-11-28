"""
This module provides functionality for array fusion and computation using lazy
evaluation.

Overview:
---------
Array fusion allows composing multiple array operations into a single kernel, enabling
significant performance optimizations by letting the compiler optimize the entire
operation at once.

Key Functions:
--------------
- `lazy`: Marks an array as an input to a fused operation.
- `compute`: Executes the fused operation efficiently.
- `fuse`: Combines multiple operations into a single kernel.
- `fused`: A decorator for marking functions as fused.

Examples:
---------
1. Basic Usage:
    >>> C = defer(A)
    >>> D = defer(B)
    >>> E = (C + D) / 2
    >>> compute(E)

    In this example, `E` represents a fused operation that adds `C` and `D` together and
    divides the result by 2. The `compute` function optimizes and executes the operation
    efficiently.

2. Using `fuse` as a higher-order function:
    >>> result = fuse(lambda x, y: (x + y) / 2, A, B)

    Here, `fuse` combines the addition and division operations into a single fused
    kernel.

3. Using the `fused` decorator:
    >>> @fused
    >>> def add_and_divide(x, y):
    >>>     return (x + y) / 2
    >>> result = add_and_divide(A, B)

    The `fused` decorator enables automatic fusion of operations within the function.

Performance:
------------
- Using `lazy` and `compute` results in faster execution due to operation fusion.
- Different optimizers can be used with `compute`, such as the Galley optimizer, which
  adapts to the sparsity patterns of the inputs.
- The optimizer can be set using the `ctx` argument in `compute`, or via `set_scheduler`
  or `with_scheduler`.
"""

from enum import Enum
from typing import Any

import numpy as np

from finchlite.autoschedule import LogicExecutor

from .. import finch_assembly as asm
from .. import finch_notation as ntn
from ..algebra import Tensor, TensorPlaceholder
from ..autoschedule import DefaultLogicOptimizer, LogicCompiler
from ..codegen import NumbaCompiler
from ..compile import BufferizedNDArray, NotationCompiler
from ..finch_logic import (
    Alias,
    Field,
    Literal,
    Plan,
    Produces,
    Query,
    Table,
)
from ..symbolic import Reflector, gensym
from .lazy import defer

_DEFAULT_SCHEDULER = None


class Mode(Enum):
    INTERPRET_LOGIC = 0
    INTERPRET_NOTATION = 1
    INTERPRET_ASSEMBLY = 2
    COMPILE_NUMBA = 3
    COMPILE_C = 4


def set_default_scheduler(
    *,
    ctx=None,
    mode=Mode.INTERPRET_LOGIC,  # TODO: change to NOTATION
):
    global _DEFAULT_SCHEDULER

    if ctx is not None:
        _DEFAULT_SCHEDULER = ctx

    elif mode == Mode.INTERPRET_LOGIC:
        _DEFAULT_SCHEDULER = LogicExecutor()

    elif mode == Mode.INTERPRET_NOTATION:
        optimizer = DefaultLogicOptimizer(LogicCompiler())
        ntn_interp = ntn.NotationInterpreter()

        def fn_compile(plan):
            prgm, tables = optimizer(plan)
            mod = ntn_interp(prgm)
            args = provision_tensors(prgm, tables)
            return (mod.func(*args),)

        _DEFAULT_SCHEDULER = fn_compile

    elif mode == Mode.INTERPRET_ASSEMBLY:
        optimizer = DefaultLogicOptimizer(LogicCompiler())
        notation_compiler = NotationCompiler(Reflector())
        asm_interp = asm.AssemblyInterpreter()

        def fn_compile(plan):
            ntn_prgm, tables = optimizer(plan)
            asm_prgm = notation_compiler(ntn_prgm)
            mod = asm_interp(asm_prgm)
            args = provision_tensors(asm_prgm, tables)
            return (mod.func(*args),)

        _DEFAULT_SCHEDULER = fn_compile

    elif mode == Mode.COMPILE_NUMBA:
        optimizer = DefaultLogicOptimizer(LogicCompiler())
        notation_compiler = NotationCompiler(Reflector())
        numba_compiler = NumbaCompiler()

        def fn_compile(plan):
            # TODO: proper logging
            # print("Logic: \n", plan)
            ntn_prgm, tables = optimizer(plan)
            # print("Notation: \n", ntn_prgm)
            asm_prgm = notation_compiler(ntn_prgm)
            # print("Assembler: \n", asm_prgm)
            mod = numba_compiler(asm_prgm)
            args = provision_tensors(asm_prgm, tables)
            return (mod.func(*args),)

        _DEFAULT_SCHEDULER = fn_compile

    elif mode == Mode.COMPILE_C:
        raise NotImplementedError

    else:
        raise Exception(f"Invalid scheduler mode: {mode}")


set_default_scheduler()


def get_default_scheduler():
    global _DEFAULT_SCHEDULER
    return _DEFAULT_SCHEDULER


def provision_tensors(prgm: Any, tables: dict[Alias, Table]) -> list[Tensor]:
    args: list[Tensor] = []
    dims_dict: dict[Field, int] = {}
    for arg in prgm.funcs[0].args:
        table = tables[Alias(arg.name)]
        match table:
            case Table(Literal(val), idxs):
                if isinstance(val, TensorPlaceholder):
                    shape = tuple(dims_dict[field] for field in idxs)
                    tensor = BufferizedNDArray(np.zeros(dtype=val.dtype, shape=shape))
                else:
                    for idx, field in enumerate(table.idxs):
                        dims_dict[field] = val.shape[idx]
                    tensor = val
            case _:
                raise Exception(f"Invalid table for tensor processing: {table}")

        args.append(tensor)

    return args


def compute(arg, ctx=None):
    """
    Executes a fused operation represented by LazyTensors. This function evaluates the
    entire operation in an optimized manner using the provided scheduler.

    Parameters:
    - arg: A lazy tensor or a tuple of lazy tensors representing the fused operation to
      be computed.
    - ctx: The scheduler to use for computation. Defaults to the result of
      `get_default_scheduler()`.

    Returns:
    - A tensor or a list of tensors computed by the fused operation.
    """
    if ctx is None:
        ctx = get_default_scheduler()

    args = arg if isinstance(arg, tuple) else (arg,)
    vars = tuple(Alias(gensym("A")) for _ in args)
    ctx_2 = args[0].ctx.join(*[x.ctx for x in args[1:]])
    bodies = tuple(map(lambda arg, var: Query(var, arg.data), args, vars))
    prgm = Plan(ctx_2.trace() + bodies + (Produces(vars),))
    res = ctx(prgm)
    if isinstance(arg, tuple):
        return tuple(res)
    return res[0].to_numpy() if hasattr(res[0], "to_numpy") else res[0]


def fuse(f, *args, ctx=None):
    """
    Fuses multiple array operations into a single kernel. This function allows for
    composing operations and executing them efficiently.

    Parameters:
        - f: The function representing the operation to be fused, returning a tensor or
        tuple of tensor results.
        - *args: The input arrays or LazyTensors to be fused.
        - ctx: The scheduler to use for computation. Defaults to the result of
        `get_default_scheduler()`.

    Returns:
        - The result of the fused operation, a tensor or tuple of tensors.
    """
    if ctx is None:
        ctx = get_default_scheduler()

    args = [defer(arg) for arg in args]
    if len(args) == 1:
        return f(args[0])
    return compute(f(*args), ctx=ctx)


def fused(f, /, ctx=None):
    """
    - fused(f):
    A decorator that marks a function as fused. This allows the function to be used with
    the `fuse` function for automatic fusion of operations.

    Parameters:
    - f: The function to be marked as fused.

    Returns:
    - A wrapper function that applies the fusion mechanism to the original function.
    """
    if ctx is None:
        ctx = get_default_scheduler()

    def wrapper(*args):
        return fuse(f, *args, ctx=ctx)

    return wrapper
