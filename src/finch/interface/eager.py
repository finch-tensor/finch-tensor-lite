from abc import ABC, abstractmethod
from . import lazy
from .lazy import LazyTensor, defer
from .fuse import compute
from typing import Callable, Tuple


class EagerTensor(ABC):
    @abstractmethod
    def shape(self):
        """Return the shape of the tensor."""
        pass

    @abstractmethod
    def dtype(self):
        """Return the data type of the tensor."""
        pass

    @abstractmethod
    def to_numpy(self):
        """Convert the tensor to a NumPy array."""
        pass

    def __add__(self, other):
        if isinstance(other, LazyTensor):
            return defer(self).__add__(other)
        else:
            return compute(defer(self).__add__(other))

    def __sub__(self, other):
        """Define subtraction for tensors."""
        if isinstance(other, LazyTensor):
            return defer(self).__sub__(other)
        else:
            return compute(defer(self).__sub__(other))

    def __mul__(self, other):
        """Define multiplication for tensors."""
        if isinstance(other, LazyTensor):
            return defer(self).__mul__(other)
        else:
            return compute(defer(self).__mul__(other))

    def __abs__(self):
        """Define absolute value for tensors."""
        compute(defer(self).__abs__())

    def __pos__(self):
        """Define unary plus for tensors."""
        compute(defer(self).__pos__())

    def __neg__(self):
        """Define negation for tensors."""
        compute(defer(self).__neg__())

    def __complex__(self):
        """Convert the tensor to a complex number."""
        complex(self.__getitem__())

    def __int__(self):
        """Convert the tensor to an integer."""
        int(self.__getitem__())

    def __float__(self):
        """Convert the tensor to a float."""
        float(self.__getitem__())

    def __bool__(self):
        """Convert the tensor to a boolean."""
        bool(self.__getitem__())


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
        return lazy.pos(x)
    else:
        return compute(lazy.pos(defer(x)))


def negative(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.neg(x)
    else:
        return compute(lazy.neg(defer(x)))
