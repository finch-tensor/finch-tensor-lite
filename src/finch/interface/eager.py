import sys
from abc import ABC
from collections.abc import Callable

from . import lazy
from .fuse import compute
from .overrides import AbstractOverrideTensor


class AbstractEagerTensor(AbstractOverrideTensor, ABC):
    def override_module(self):
        return sys.modules[__name__]

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


def permute_dims(arg, /, axis: tuple[int, ...]):
    if isinstance(arg, lazy.LazyTensor):
        return lazy.permute_dims(arg, axis=axis)
    return compute(lazy.permute_dims(arg, axis=axis))


def expand_dims(
    x,
    /,
    axis: int | tuple[int, ...] = 0,
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.expand_dims(x, axis=axis)
    return compute(lazy.expand_dims(x, axis=axis))


def squeeze(
    x,
    /,
    axis: int | tuple[int, ...],
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.squeeze(x, axis=axis)
    return compute(lazy.squeeze(x, axis=axis))


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
    return compute(
        lazy.reduce(op, x, axis=axis, dtype=dtype, keepdims=keepdims, init=init)
    )


def sum(
    x,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype=None,
    keepdims: bool = False,
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)
    return compute(lazy.sum(x, axis=axis, dtype=dtype, keepdims=keepdims))


def prod(
    x,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype=None,
    keepdims: bool = False,
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.prod(x, axis=axis, dtype=dtype, keepdims=keepdims)
    return compute(lazy.prod(x, axis=axis, dtype=dtype, keepdims=keepdims))


def elementwise(f: Callable, *args):
    if any(isinstance(arg, lazy.LazyTensor) for arg in args):
        return lazy.elementwise(f, *args)
    return compute(lazy.elementwise(f, *args))


def add(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.add(x1, x2)
    return compute(lazy.add(x1, x2))


def subtract(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.subtract(x1, x2)
    return compute(lazy.subtract(x1, x2))


def multiply(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.multiply(x1, x2)
    return compute(lazy.multiply(x1, x2))


def abs(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.abs(x)
    return compute(lazy.abs(x))


def positive(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.positive(x)
    return compute(lazy.positive(x))


def negative(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.negative(x)
    return compute(lazy.negative(x))
