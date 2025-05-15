from .lazy import lazy, LazyTensor

# from .tensor import *
from .fuse import compute, fuse, fused
from .eager import (
    EagerTensor,
    add,
    subtract,
    multiply,
    abs,
    positive,
    negative,
    prod,
    permute_dims,
    expand_dims,
    squeeze,
    reduce,
    elementwise,
)

__all__ = [
    "lazy",
    "compute",
    "fuse",
    "fused",
    "permute_dims",
    "expand_dims",
    "squeeze",
    "reduce",
    "elementwise",
    "prod",
    "add",
    "subtract",
    "multiply",
    "abs",
    "positive",
    "negative",
    "EagerTensor",
    "LazyTensor",
]
