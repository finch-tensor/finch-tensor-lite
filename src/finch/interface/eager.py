from abc import ABC, abstractmethod
from . import lazy
from .lazy import LazyTensor
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
            return lazy.lazy(self).__add__(other)
        else:
            return lazy.compute(lazy.lazy(self).__add__(other))

    def __sub__(self, other):
        """Define subtraction for tensors."""
        if isinstance(other, LazyTensor):
            return lazy.lazy(self).__sub__(other)
        else:
            return lazy.compute(lazy.lazy(self).__sub__(other))

    def __mul__(self, other):
        """Define multiplication for tensors."""
        if isinstance(other, LazyTensor):
            return lazy.lazy(self).__mul__(other)
        else:
            return lazy.compute(lazy.lazy(self).__mul__(other))

    def __abs__(self):
        """Define absolute value for tensors."""
        lazy.compute(lazy.lazy(self).__abs__())

    def __pos__(self):
        """Define unary plus for tensors."""
        lazy.compute(lazy.lazy(self).__pos__())

    def __neg__(self):
        """Define negation for tensors."""
        lazy.compute(lazy.lazy(self).__neg__())

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
        return lazy.compute(lazy.permute_dims(lazy.lazy(arg), axis=axis))


def expand_dims(
    x,
    /,
    axis: int | tuple[int, ...] = 0,
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.expand_dims(x, axis=axis)
    else:
        return lazy.compute(lazy.expand_dims(lazy.lazy(x), axis=axis))


def squeeze(
    x,
    /,
    axis: int | tuple[int, ...],
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.squeeze(x, axis=axis)
    else:
        return lazy.compute(lazy.squeeze(lazy.lazy(x), axis=axis))


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
        return lazy.compute(
            lazy.reduce(
                op, lazy.lazy(x), axis=axis, dtype=dtype, keepdims=keepdims, init=init
            )
        )


def elementwise(f: Callable, src: LazyTensor, *args):
    if isinstance(src, lazy.LazyTensor):
        return lazy.elementwise(f, src, *args)
    else:
        return lazy.compute(lazy.elementwise(f, lazy.lazy(src), *args))


def prod(arr, /, axis=None):
    if isinstance(arr, lazy.LazyTensor):
        return lazy.prod(arr, axis=axis)
    else:
        return lazy.compute(lazy.prod(lazy.lazy(arr), axis=axis))


def add(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.add(x1, x2)
    else:
        return lazy.compute(lazy.add(lazy.lazy(x1), lazy.lazy(x2)))


def subtract(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.subtract(x1, x2)
    else:
        return lazy.compute(lazy.subtract(lazy.lazy(x1), lazy.lazy(x2)))


def multiply(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.multiply(x1, x2)
    else:
        return lazy.compute(lazy.multiply(lazy.lazy(x1), lazy.lazy(x2)))


def abs(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.abs(x)
    else:
        return lazy.compute(lazy.abs(lazy.lazy(x)))


def positive(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.pos(x)
    else:
        return lazy.compute(lazy.pos(lazy.lazy(x)))


def negative(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.neg(x)
    else:
        return lazy.compute(lazy.neg(lazy.lazy(x)))
