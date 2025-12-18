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
    >>> C = lazy(A)
    >>> D = lazy(B)
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

from finchlite.autoschedule import LogicExecutor, LogicNormalizer
from finchlite.autoschedule.formatter import LogicFormatter
from finchlite.finch_logic.stages import LogicEvaluator
from finchlite.finch_notation.interpreter import NotationInterpreter

from ..autoschedule.compiler import LogicCompiler
from ..autoschedule.optimize2 import LogicNormalizer2
from ..codegen import NumbaCompiler
from ..compile import NotationCompiler
from ..finch_assembly import AssemblyInterpreter
from ..finch_logic import (
    Alias,
    LogicInterpreter,
    MockLogicLoader,
    Plan,
    Produces,
    Query,
)
from ..symbolic import gensym
from .lazy import lazy

_DEFAULT_SCHEDULER = None


INTERPRET_LOGIC = LogicInterpreter()
OPTIMIZE_LOGIC = LogicNormalizer(
    LogicExecutor(LogicNormalizer2(LogicFormatter(MockLogicLoader())))
)
INTERPRET_NOTATION = LogicNormalizer(
    LogicExecutor(
        LogicNormalizer2(LogicFormatter(LogicCompiler(NotationInterpreter())))
    )
)
INTERPRET_ASSEMBLY = LogicNormalizer(
    LogicExecutor(
        LogicNormalizer2(
            LogicFormatter(LogicCompiler(NotationCompiler(AssemblyInterpreter())))
        )
    )
)
COMPILE_NUMBA = LogicNormalizer(
    LogicExecutor(
        LogicNormalizer2(
            LogicFormatter(LogicCompiler(NotationCompiler(NumbaCompiler())))
        )
    )
)


def set_default_scheduler(
    *,
    ctx: LogicEvaluator = INTERPRET_NOTATION,
):
    global _DEFAULT_SCHEDULER

    if ctx is not None:
        _DEFAULT_SCHEDULER = ctx


set_default_scheduler()


def get_default_scheduler():
    global _DEFAULT_SCHEDULER
    return _DEFAULT_SCHEDULER


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
        return tuple(tbl.tns for tbl in res)
    return res[0].tns


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

    args = [lazy(arg) for arg in args]
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
