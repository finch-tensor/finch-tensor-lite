from .lazy import *
#from .tensor import *
from .fuse import *

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