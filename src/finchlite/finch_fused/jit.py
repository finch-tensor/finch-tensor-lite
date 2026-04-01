from ..interface.fuse import get_default_scheduler
from .dataflow import insert_lazy_and_compute
from .parser import fused_function_to_python_function, parse_fused_function


def jit(f, /, ctx=None):
    """
    A decorator that marks a function for just-in-time compilation. This allows the
    function to be compiled and optimized for performance when called.

    Parameters:
    - f: The function to be marked for JIT compilation.

    Returns:
    - A transformed function that inserts lazy and compute statements to do tracing
       and optimization.
    """
    if ctx is None:
        ctx = get_default_scheduler()
    fused_fn = parse_fused_function(f)
    transformed_fn = insert_lazy_and_compute(fused_fn)
    return fused_function_to_python_function(transformed_fn)
