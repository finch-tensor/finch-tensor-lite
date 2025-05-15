# from .tensor import *
from .fuse import fuse, fused
from .lazy import (
    compute,
    elementwise,
    expand_dims,
    identify,
    lazy,
    multiply,
    permute_dims,
    prod,
    reduce,
    squeeze,
)

__all__ = [
    "lazy",
    "compute",
    "fuse",
    "fused",
    "permute_dims",
    "expand_dims",
    "squeeze",
    "identify",
    "reduce",
    "elementwise",
    "prod",
    "multiply",
]
