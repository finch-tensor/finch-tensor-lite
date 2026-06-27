from .eager import outer as _outer
from .lazy import matmul, matrix_transpose, tensordot, vecdot

__all__ = ["matmul", "matrix_transpose", "outer", "tensordot", "vecdot"]


def outer(x1, x2, /):
    return _outer(x1, x2)
