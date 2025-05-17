from abc import ABC
from . import lazy
from .lazy import LazyTensor, defer
from .fuse import compute
from typing import Callable, Tuple


class EagerTensor(ABC):
    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return subtract(self, other)

    def __rsub__(self, other):
        return subtract(other, self)

    def __mul__(self, other):
        return multiply(self, other)

    def __rmul__(self, other):
        return multiply(other, self)

    def __abs__(self):
        return abs(self)

    def __pos__(self):
        return positive(self)

    def __neg__(self):
        return negative(self)


def permute_dims(arg, /, axis: Tuple[int, ...]):
    if isinstance(arg, lazy.LazyTensor):
        return lazy.permute_dims(arg, axis=axis)
    else:
        return compute(lazy.permute_dims(defer(arg), axis=axis))


def expand_dims(
    x,
    /,
    axis: int | tuple[int, ...] = 0,
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.expand_dims(x, axis=axis)
    else:
        return compute(lazy.expand_dims(defer(x), axis=axis))


def squeeze(
    x,
    /,
    axis: int | tuple[int, ...],
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.squeeze(x, axis=axis)
    else:
        return compute(lazy.squeeze(defer(x), axis=axis))


def reduce(
    op: Callable,
    x,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype=None,
    keepdims: bool = False,
    init=None,
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.reduce(op, x, axis=axis, dtype=dtype, keepdims=keepdims, init=init)
    else:
        return compute(
            lazy.reduce(
                op, defer(x), axis=axis, dtype=dtype, keepdims=keepdims, init=init
            )
        )


def elementwise(f: Callable, src: LazyTensor, *args):
    if isinstance(src, lazy.LazyTensor):
        return lazy.elementwise(f, src, *args)
    else:
        return compute(lazy.elementwise(f, defer(src), *args))


def prod(arr, /, axis=None):
    if isinstance(arr, lazy.LazyTensor):
        return lazy.prod(arr, axis=axis)
    else:
        return compute(lazy.prod(defer(arr), axis=axis))


def add(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.add(x1, x2)
    else:
        return compute(lazy.add(defer(x1), defer(x2)))


def subtract(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.subtract(x1, x2)
    else:
        return compute(lazy.subtract(defer(x1), defer(x2)))


def multiply(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.multiply(x1, x2)
    else:
        return compute(lazy.multiply(defer(x1), defer(x2)))


def abs(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.abs(x)
    else:
        return compute(lazy.abs(defer(x)))


def positive(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.positive(x)
    else:
        return compute(lazy.positive(defer(x)))


def negative(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.negative(x)
    else:
        return compute(lazy.negative(defer(x)))
