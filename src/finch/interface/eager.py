import builtins
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..algebra import register_property
from . import lazy
from .fuse import compute
from .overrides import OverrideTensor


class EagerTensor(OverrideTensor, ABC):
    def override_module(self):
        return sys.modules[__name__]

    @property
    @abstractmethod
    def ndim(self):
        """Number of dimensions of the tensor."""
        ...

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

    def __invert__(self):
        return bitwise_inverse(self)

    def __and__(self, other):
        return bitwise_and(self, other)

    def __rand__(self, other):
        return bitwise_and(other, self)

    def __lshift__(self, other):
        return bitwise_left_shift(self, other)

    def __rlshift__(self, other):
        return bitwise_left_shift(other, self)

    def __or__(self, other):
        return bitwise_or(self, other)

    def __ror__(self, other):
        return bitwise_or(other, self)

    def __rshift__(self, other):
        return bitwise_right_shift(self, other)

    def __rrshift__(self, other):
        return bitwise_right_shift(other, self)

    def __xor__(self, other):
        return bitwise_xor(self, other)

    def __rxor__(self, other):
        return bitwise_xor(other, self)

    def __truediv__(self, other):
        return truediv(self, other)

    def __rtruediv__(self, other):
        return truediv(other, self)

    def __floordiv__(self, other):
        return floordiv(self, other)

    def __rfloordiv__(self, other):
        return floordiv(other, self)

    def __mod__(self, other):
        return mod(self, other)

    def __rmod__(self, other):
        return mod(other, self)

    def __pow__(self, other):
        return pow(self, other)

    def __rpow__(self, other):
        return pow(other, self)

    def __matmul__(self, other):
        return matmul(self, other)

    def __rmatmul__(self, other):
        return matmul(other, self)

    def __sin__(self):
        return sin(self)

    def __sinh__(self):
        return sinh(self)

    def __cos__(self):
        return cos(self)

    def __cosh__(self):
        return cosh(self)

    def __tan__(self):
        return tan(self)

    def __tanh__(self):
        return tanh(self)

    def __asin__(self):
        return asin(self)

    def __asinh__(self):
        return asinh(self)

    def __acos__(self):
        return acos(self)

    def __acosh__(self):
        return acosh(self)

    def __atan__(self):
        return atan(self)

    def __atanh__(self):
        return atanh(self)

    def __atan2__(self, other):
        return atan2(self, other)

    def __complex__(self):
        """
        Converts a zero-dimensional array to a Python `complex` object.
        """
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to complex.")
        # dispatch to the scalar value's `__complex__` method
        return complex(self[()])

    def __float__(self):
        """
        Converts a zero-dimensional array to a Python `float` object.
        """
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to float.")
        # dispatch to the scalar value's `__float__` method
        return float(self[()])

    def __int__(self):
        """
        Converts a zero-dimensional array to a Python `int` object.
        """
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to int.")
        # dispatch to the scalar value's `__int__` method
        return int(self[()])

    def __bool__(self):
        """
        Converts a zero-dimensional array to a Python `bool` object.
        """
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to bool.")
        # dispatch to the scalar value's `__bool__` method
        return bool(self[()])


@dataclass
class Scalar(EagerTensor):
    val: Any
    fill_value: Any
    dtype: Any

    def __init__(self, val: Any):
        self.val = val
        self.fill_value = val
        self.dtype = type(val)

    @property
    def shape(self):
        return ()

    @property
    def ndim(self):
        return 0

    @property
    def element_type(self):
        return type(self.val)

    def __getitem__(self, idx):
        return self.val


register_property(EagerTensor, "asarray", "__attr__", lambda x: x)
register_property(object, "asarray", "__attr__", lambda x: Scalar(x))
register_property(Scalar, "asarray", "__attr__", lambda x: np.asarray(x))


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
    if builtins.any(isinstance(arg, lazy.LazyTensor) for arg in args):
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


def matmul(x1, x2, /):
    """
    Computes the matrix product.

    Returns a LazyTensor if either x1 or x2 is a LazyTensor.
    Otherwise, computes the result eagerly.
    """
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.matmul(x1, x2)
    c = lazy.matmul(x1, x2)
    return compute(c)


def matrix_transpose(x, /):
    """
    Computes the transpose of a matrix or stack of matrices.
    """
    if isinstance(x, lazy.LazyTensor):
        return lazy.matrix_transpose(x)
    return compute(lazy.matrix_transpose(x))


def bitwise_inverse(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.bitwise_inverse(x)
    return compute(lazy.bitwise_inverse(x))


def bitwise_and(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.bitwise_and(x1, x2)
    return compute(lazy.bitwise_and(x1, x2))


def bitwise_left_shift(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.bitwise_left_shift(x1, x2)
    return compute(lazy.bitwise_left_shift(x1, x2))


def bitwise_or(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.bitwise_or(x1, x2)
    return compute(lazy.bitwise_or(x1, x2))


def bitwise_right_shift(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.bitwise_right_shift(x1, x2)
    return compute(lazy.bitwise_right_shift(x1, x2))


def bitwise_xor(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.bitwise_xor(x1, x2)
    return compute(lazy.bitwise_xor(x1, x2))


def truediv(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.truediv(x1, x2)
    return compute(lazy.truediv(x1, x2))


def floordiv(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.floordiv(x1, x2)
    return compute(lazy.floordiv(x1, x2))


def mod(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.mod(x1, x2)
    return compute(lazy.mod(x1, x2))


def pow(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.pow(x1, x2)
    return compute(lazy.pow(x1, x2))


def tensordot(x1, x2, /, *, axes: int | tuple[Sequence[int], Sequence[int]]):
    """
    Computes the tensordot operation.

    Returns a LazyTensor if either x1 or x2 is a LazyTensor.
    Otherwise, computes the result eagerly.
    """
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.tensordot(x1, x2, axes=axes)
    return compute(lazy.tensordot(x1, x2, axes=axes))


def vecdot(x1, x2, /, *, axis=-1):
    """
    Computes the (vector) dot product of two arrays.

    Parameters
    ----------
    x1: array
        The first input tensor.
    x2: array
        The second input tensor.
    axis: int, optional
        The axis along which to compute the dot product. Default is -1 (last axis).

    Returns
    -------
    out: array
        A tensor containing the dot product of `x1` and `x2` along the specified axis.
    """
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.vecdot(x1, x2, axis=axis)
    return compute(lazy.vecdot(x1, x2, axis=axis))


def any(x, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False):
    if isinstance(x, lazy.LazyTensor):
        return lazy.any(x, axis=axis, keepdims=keepdims)
    return compute(lazy.any(x, axis=axis, keepdims=keepdims))


def all(x, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False):
    if isinstance(x, lazy.LazyTensor):
        return lazy.all(x, axis=axis, keepdims=keepdims)
    return compute(lazy.all(x, axis=axis, keepdims=keepdims))


def min(x, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False):
    if isinstance(x, lazy.LazyTensor):
        return lazy.min(x, axis=axis, keepdims=keepdims)
    return compute(lazy.min(x, axis=axis, keepdims=keepdims))


def max(x, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False):
    if isinstance(x, lazy.LazyTensor):
        return lazy.max(x, axis=axis, keepdims=keepdims)
    return compute(lazy.max(x, axis=axis, keepdims=keepdims))


def sin(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.sin(x)
    return compute(lazy.sin(x))


def sinh(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.sinh(x)
    return compute(lazy.sinh(x))


def cos(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.cos(x)
    return compute(lazy.cos(x))


def cosh(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.cosh(x)
    return compute(lazy.cosh(x))


def tan(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.tan(x)
    return compute(lazy.tan(x))


def tanh(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.tanh(x)
    return compute(lazy.tanh(x))


def asin(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.asin(x)
    return compute(lazy.asin(x))


def asinh(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.asinh(x)
    return compute(lazy.asinh(x))


def acos(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.acos(x)
    return compute(lazy.acos(x))


def acosh(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.acosh(x)
    return compute(lazy.acosh(x))


def atan(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.atan(x)
    return compute(lazy.atan(x))


def atanh(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.atanh(x)
    return compute(lazy.atanh(x))


def atan2(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.atan2(x1, x2)
    return compute(lazy.atan2(x1, x2))
