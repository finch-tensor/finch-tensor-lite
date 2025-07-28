import builtins
import sys
from abc import ABC
from collections.abc import Callable, Sequence

from ..algebra import register_property
from . import lazy
from .fuse import compute
from .overrides import OverrideTensor


class EagerTensor(OverrideTensor, ABC):
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

    def __log__(self):
        return log(self)

    def __log1p__(self):
        return log1p(self)

    def __log2__(self):
        return log2(self)

    def __log10__(self):
        return log10(self)

    def __logaddexp__(self, other):
        return logaddexp(self, other)

    def __logical_and__(self, other):
        return logical_and(self, other)

    def __logical_or__(self, other):
        return logical_or(self, other)

    def __logical_xor__(self, other):
        return logical_xor(self, other)

    def __logical_not__(self):
        return logical_not(self)


register_property(EagerTensor, "asarray", "__attr__", lambda x: x)


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


# manipulation functions:
# https://data-apis.org/array-api/2024.12/API_specification/manipulation_functions.html


def broadcast_to(x, /, shape: Sequence[int]):
    """
    Broadcasts an array to a new shape.

    Parameters
    ----------
    x: array
        The input tensor to be broadcasted.
    shape: Sequence[int]
        The target shape to which the input tensor should be broadcasted.

    Returns
    -------
    out: array
        A tensor with the same data as `x`, but with the specified shape.
    """
    shape = tuple(shape)  # Ensure shape is a tuple for consistency
    if isinstance(x, lazy.LazyTensor):
        return lazy.broadcast_to(x, shape=shape)
    return compute(lazy.broadcast_to(x, shape=shape))


def broadcast_arrays(*args):
    """
    Broadcasts one or more arrays against one another.

    Parameters
    ----------
    *args: array
        an arbitrary number of to-be broadcasted arrays.

    Returns
    -------
    out: List[array]
        a list of broadcasted arrays. Each array has the same shape.
        Element types are preserved.
    """
    if builtins.any(isinstance(arg, lazy.LazyTensor) for arg in args):
        return lazy.broadcast_arrays(*args)
    # compute can take in a list of LazyTensors
    return compute(lazy.broadcast_arrays(*args))


def concat(arrays: tuple | list, /, *, axis: int | None = 0):
    """
    Concatenates a sequence of arrays along an existing axis.

    Parameters
    ----------
    arrays: tuple or list
        A sequence of arrays to concatenate. Arrays must have the same shape
        except in the dimension corresponding to the specified axis.
    axis: int, optional
        The axis along which to concatenate the arrays. Default is 0. If None,
        the arrays are flattened before concatenation.

    Returns
    -------
    out: array
        A new concatenated array.
    """
    if builtins.any(isinstance(arr, lazy.LazyTensor) for arr in arrays):
        return lazy.concat(arrays, axis=axis)
    return compute(lazy.concat(arrays, axis=axis))


def moveaxis(x, source: int | tuple[int, ...], destination: int | tuple[int, ...], /):
    """
    Moves array axes (dimensions) to new positions,
    while leaving other axes in their original positions.

    Args
    ---------
    - x (array) - input array.
    - source - Axes to move.
    - destination - indices defining the desired
    positions for each respective source axis index.

    Returns
    --------
    - out (array) - an array containing reordered axes.
    """
    if isinstance(x, lazy.LazyTensor):
        return lazy.moveaxis(x, source, destination)
    return compute(lazy.moveaxis(x, source, destination))


def stack(arrays: Sequence, /, *, axis: int = 0):
    """
    Stacks a sequence of arrays along a new axis.

    Parameters
    ----------
    arrays: Sequence
        A sequence of arrays to stack. All arrays must have the same shape.
    axis: int, optional
        The axis along which to stack the arrays. Default is 0.

    Returns
    -------
    out: array
        A new array with the stacked arrays along the specified axis.
    """
    if builtins.any(isinstance(arr, lazy.LazyTensor) for arr in arrays):
        return lazy.stack(arrays, axis=axis)
    return compute(lazy.stack(arrays, axis=axis))


def split_dims(x, axis: int, shape: tuple):
    """
    Split a dimension into multiple dimensions. The product
    of the sizes in the `shape` tuple must equal the size
    of the dimension being split.

    Parameters
    ----------
    x: array
        The input tensor to split
    axis: int
        The axis to split
    shape: tuple
        The new shape for the split dimensions

    Returns
    -------
    out: array
        A tensor with the specified dimension split into multiple dimensions

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(12).reshape(2, 6)  # shape (2, 6)
    >>> result = split_dims(x, axis=1, shape=(2, 3))
    >>> result.shape
    (2, 2, 3)
    """
    if isinstance(x, lazy.LazyTensor):
        return lazy.split_dims(x, axis, shape)
    return compute(lazy.split_dims(x, axis, shape))


def combine_dims(x, axes: tuple[int, ...]):
    """
    Combine multiple consecutive dimensions into a single dimension.
    The resulting axis will have a size equal to the product of the
    sizes of the combined axes.

    Parameters
    ----------
    x: array
        The input tensor
    axes: tuple[int, ...]
        Consecutive axes to combine.

        The axes will be considered in increasing order.
        So passing axes=(2, 1, 3) will be equivalent to
        passing axes=(1, 2, 3).

    Returns
    -------
    out: array
        A tensor with the specified dimensions combined into one

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(24).reshape(2, 3, 4)  # shape (2, 3, 4)
    >>> result = combine_dims(x, axes=(1, 2))
    >>> result.shape
    (2, 12)
    """
    if isinstance(x, lazy.LazyTensor):
        return lazy.combine_dims(x, axes)
    return compute(lazy.combine_dims(x, axes))


def flatten(x):
    """
    Flattens the input tensor into a 1D tensor.

    Parameters
    ----------
    x: array
        The input tensor to be flattened.

    Returns
    -------
    out: array
        A new tensor that is a flattened version of the input.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(24).reshape(2, 3, 4)  # shape (2, 3, 4)
    >>> result = flatten(x)
    >>> result.shape
    (24,)
    """
    if isinstance(x, lazy.LazyTensor):
        return lazy.flatten(x)
    return compute(lazy.flatten(x))


# trigonometric functions:
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


def log(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.log(x)
    return compute(lazy.log(x))


def log1p(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.log1p(x)
    return compute(lazy.log1p(x))


def log2(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.log2(x)
    return compute(lazy.log2(x))


def log10(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.log10(x)
    return compute(lazy.log10(x))


def logaddexp(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.logaddexp(x1, x2)
    return compute(lazy.logaddexp(x1, x2))


def logical_and(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.logical_and(x1, x2)
    return compute(lazy.logical_and(x1, x2))


def logical_or(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.logical_or(x1, x2)
    return compute(lazy.logical_or(x1, x2))


def logical_xor(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.logical_xor(x1, x2)
    return compute(lazy.logical_xor(x1, x2))


def logical_not(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.logical_not(x)
    return compute(lazy.logical_not(x))


def mean(x, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False):
    if isinstance(x, lazy.LazyTensor):
        return lazy.mean(x, axis=axis, keepdims=keepdims)
    return compute(lazy.mean(x, axis=axis, keepdims=keepdims))


def var(
    x,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: float = 0.0,
    keepdims: bool = False,
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.var(x, axis=axis, correction=correction, keepdims=keepdims)
    return compute(lazy.var(x, axis=axis, correction=correction, keepdims=keepdims))


def std(
    x,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: float = 0.0,
    keepdims: bool = False,
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.std(x, axis=axis, correction=correction, keepdims=keepdims)
    return compute(lazy.std(x, axis=axis, correction=correction, keepdims=keepdims))


def argmin(x, /, *, axis: int | None = None, keepdims: bool = False):
    if isinstance(x, lazy.LazyTensor):
        return lazy.argmin(x, axis=axis, keepdims=keepdims)
    return compute(lazy.argmin(x, axis=axis, keepdims=keepdims))


def argmax(x, /, *, axis: int | None = None, keepdims: bool = False):
    if isinstance(x, lazy.LazyTensor):
        return lazy.argmax(x, axis=axis, keepdims=keepdims)
    return compute(lazy.argmax(x, axis=axis, keepdims=keepdims))
