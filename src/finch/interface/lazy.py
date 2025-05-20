import operator
import builtins
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Tuple
from itertools import accumulate
from numpy.core.numeric import normalize_axis_tuple
from ..algebra import return_type, fixpoint_type, init_value, element_type, fill_value, logical_and, logical_or
import numpy as np
from .overrides import AbstractOverrideTensor
import sys

from ..finch_logic import LogicNode
from ..finch_logic import (
    Immediate,
    Table,
    Alias,
    Field,
    Subquery,
    MapJoin,
    Aggregate,
    Reorder,
    Relabel,
)
from ..symbolic import gensym


def identify(data):
    lhs = Alias(gensym("A"))
    return Subquery(lhs, data)


@dataclass
class LazyTensor(AbstractOverrideTensor):
    data: LogicNode
    shape: Tuple
    fill_value: Any
    element_type: Any

    def override_module(self):
        return sys.modules[__name__]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __add__(self, other):
        return add(self, defer(other))

    def __radd__(self, other):
        return add(defer(other), self)

    def __sub__(self, other):
        return subtract(self, defer(other))

    def __rsub__(self, other):
        return subtract(defer(other), self)

    def __mul__(self, other):
        return multiply(self, defer(other))

    def __rmul__(self, other):
        return multiply(defer(other), self)

    def __abs__(self):
        return abs(self)

    def __pos__(self):
        return positive(self)

    def __neg__(self):
        return negative(self)

def defer(arr) -> LazyTensor:
    """
    - defer(arr) -> LazyTensor:
    Converts an array into a LazyTensor. If the input is already a LazyTensor, it is returned as-is.
    Otherwise, it creates a LazyTensor representation of the input array.

    Parameters:
    - arr: The input array to be converted into a LazyTensor.

    Returns:
    - LazyTensor: A lazy representation of the input array.
    """
    if isinstance(arr, LazyTensor):
        return arr
    name = Alias(gensym("A"))
    idxs = [Field(gensym("i")) for _ in range(arr.ndim)]
    shape = tuple(arr.shape)
    tns = Subquery(name, Table(Immediate(arr), idxs))
    return LazyTensor(tns, shape, fill_value(arr), element_type(arr))


def permute_dims(arg, /, axis: Tuple[int, ...]) -> LazyTensor:
    """
    Permutes the axes (dimensions) of an array ``x``.

    Parameters
    ----------
    x: array
        input array.
    axes: Tuple[int, ...]
        tuple containing a permutation of ``(0, 1, ..., N-1)`` where ``N`` is the number of axes (dimensions) of ``x``.

    Returns
    -------
    out: array
        an array containing the axes permutation. The returned array must have the same data type as ``x``.
    """
    arg = defer(arg)
    axis = normalize_axis_tuple(axis, arg.ndim + len(axis))
    idxs = [Field(gensym("i")) for _ in range(arg.ndim)]
    return LazyTensor(
        Reorder(Relabel(arg.data, idxs), [idxs[i] for i in axis]),
        [arg.shape[i] for i in axis],
        arg.fill_value,
        arg.element_type,
    )


def expand_dims(
    x,
    /,
    axis: int | tuple[int, ...] = 0,
) -> LazyTensor:
    """
    Expands the shape of an array by inserting a new axis (dimension) of size one at the position specified by ``axis``.

    Parameters
    ----------
    x: array
        input array.
    axis: int
        axis position (zero-based). If ``x`` has rank (i.e, number of dimensions) ``N``, a valid ``axis`` must reside on the closed-interval ``[-N-1, N]``. If provided a negative ``axis``, the axis position at which to insert a singleton dimension must be computed as ``N + axis + 1``. Hence, if provided ``-1``, the resolved axis position must be ``N`` (i.e., a singleton dimension must be appended to the input array ``x``). If provided ``-N-1``, the resolved axis position must be ``0`` (i.e., a singleton dimension must be prepended to the input array ``x``).

    Returns
    -------
    out: array
        an expanded output array having the same data type as ``x``.

    Raises
    ------
    IndexError
        If provided an invalid ``axis`` position, an ``IndexError`` should be raised.
    """
    x = defer(x)
    if isinstance(axis, int):
        axis = (axis,)
    axis = normalize_axis_tuple(axis, x.ndim + len(axis))
    assert len(axis) == len(set(axis)), "axis must be unique"
    assert set(axis).issubset(range(x.ndim + len(axis))), "Invalid axis"
    offset = [0] * (x.ndim + len(axis))
    for d in axis:
        offset[d] = 1
    offset = list(accumulate(offset))
    idxs_1 = [Field(gensym("i")) for _ in range(x.ndim)]
    idxs_2 = [
        Field(gensym("i")) if n in axis else idxs_1[n - offset[n]]
        for n in range(x.ndim + len(axis))
    ]
    data_2 = Reorder(Relabel(x.data, idxs_1), idxs_2)
    shape_2 = tuple(
        1 if n in axis else x.shape[n - offset[n]] for n in range(x.ndim + len(axis))
    )
    return LazyTensor(data_2, shape_2, x.fill_value, x.element_type)


def squeeze(
    x,
    /,
    axis: int | tuple[int, ...],
) -> LazyTensor:
    """
    Removes singleton dimensions (axes) from ``x``.

    Parameters
    ----------
    x: array
        input array.
    axis: Union[int, Tuple[int, ...]]
        axis (or axes) to squeeze.

    Returns
    -------
    out: array
        an output array having the same data type and elements as ``x``.

    Raises
    ------
    ValueError
        If a specified axis has a size greater than one (i.e., it is not a
        singleton dimension), a ``ValueError`` should be raised.
    """
    x = defer(x)
    if isinstance(axis, int):
        axis = (axis,)
    axis = normalize_axis_tuple(axis, x.ndim)
    assert len(axis) == len(set(axis)), "axis must be unique"
    assert set(axis).issubset(range(x.ndim)), "Invalid axis"
    assert all(x.shape[d] == 1 for d in axis), "axis to drop must have size 1"
    newaxis = [n for n in range(x.ndim) if n not in axis]
    idxs_1 = [Field(gensym("i")) for _ in range(x.ndim)]
    idxs_2 = [idxs_1[n] for n in newaxis]
    data_2 = Reorder(Relabel(x.data, idxs_1), idxs_2)
    shape_2 = tuple(x.shape[n] for n in newaxis)
    return LazyTensor(data_2, shape_2, x.fill_value, x.element_type)


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
    """
    Reduces the input array ``x`` with the binary operator ``op``. Reduces along
    the specified `axis`, with an initial value `init`.

    Parameters
    ----------
    x: array
        input array. Should have a numeric data type.
    axis: Optional[Union[int, Tuple[int, ...]]]
        axis or axes along which reduction must be computed. By default, the reduction must be computed over the entire array. If a tuple of integers, reductions must be computed over multiple axes. Default: ``None``.

    dtype: Optional[dtype]
        data type of the returned array. If ``None``, a suitable data type will be calculated.

    keepdims: bool
        if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result. Default: ``False``.

    init: Optional
        Initial value for the reduction. If ``None``, a suitable initial value will be calculated. The initial value must be compatible with the operation defined by ``op``. For example, if ``op`` is addition, the initial value should be zero; if ``op`` is multiplication, the initial value should be one.

    Returns
    -------
    out: array
        If the reduction was computed over the entire array, a zero-dimensional array containing the reduction; otherwise, a non-zero-dimensional array containing the reduction. The returned array must have a data type as described by the ``dtype`` parameter above.
    """
    x = defer(x)
    if init is None:
        init = init_value(op, x.element_type)
    if axis is None:
        axis = tuple(range(x.ndim))
    axis = normalize_axis_tuple(axis, x.ndim)
    shape = tuple(x.shape[n] for n in range(x.ndim) if n not in axis)
    fields = [Field(gensym("i")) for _ in range(x.ndim)]
    data = Aggregate(
        Immediate(op),
        Immediate(init),
        Relabel(x.data, fields),
        [fields[i] for i in axis],
    )
    if keepdims:
        keeps = [fields[i] if i in axis else Field(gensym("j")) for i in range(x.ndim)]
        data = Reorder(data, keeps)
        shape = [shape[i] if i in axis else 1 for i in range(x.ndim)]
    if dtype is None:
        dtype = fixpoint_type(op, init, x.element_type)
    return LazyTensor(identify(data), shape, init, dtype)


def elementwise(f: Callable, *args) -> LazyTensor:
    """
        elementwise(f, *args) -> LazyTensor:

    Applies the function f elementwise to the given arguments, following
    [broacasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html).
    The function f should be a callable that takes the same number of arguments
    as the number of tensors passed to `elementwise`.

    The function will automatically handle broadcasting of the input tensors to
    ensure they have compatible shapes.  For example, `elementwise(operator.add,
    x, y)` is equivalent to `x + y`.

    Parameters:
    - f: The function to apply elementwise.
    - *args: The tensors to apply the function to. These tensors should be
        compatible for broadcasting.

    Returns:
    - LazyTensor: The tensor, `out`, of results from applying `f` elementwise to
    the input tensors.  After broadcasting the arguments to the same shape, for
    each index `i`, `out[*i] = f(args[0][*i], args[1][*i], ...)`.
    """
    args = list(map(defer, args))
    ndim = builtins.max([arg.ndim for arg in args])
    shape = tuple(
        builtins.max(
            [
                arg.shape[i - ndim + arg.ndim] if i - ndim + arg.ndim >= 0 else 1
                for arg in args
            ]
        )
        for i in range(ndim)
    )
    idxs = [Field(gensym("i")) for _ in range(ndim)]
    bargs = []
    for arg in args:
        idims = []
        odims = []
        for i in range(ndim - arg.ndim, ndim):
            if arg.shape[i - ndim + arg.ndim] == shape[i]:
                idims.append(idxs[i])
                odims.append(idxs[i])
            else:
                if arg.shape[i - ndim + arg.ndim] != 1:
                    raise ValueError("Invalid shape for broadcasting")
                idims.append(Field(gensym("j")))
        bargs.append(Reorder(Relabel(arg.data, tuple(idims)), tuple(odims)))
    data = MapJoin(Immediate(f), tuple(bargs))
    new_fill_value = f(*[x.fill_value for x in args])
    new_element_type = return_type(f, *[x.element_type for x in args])
    return LazyTensor(identify(data), shape, new_fill_value, new_element_type)


def sum(
    x,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype=None,
    keepdims: bool = False,
):
    x = defer(x)
    return reduce(operator.add, x, axis=axis, dtype=dtype, keepdims=keepdims)


def prod(
    x,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype=None,
    keepdims: bool = False,
):
    x = defer(x)
    return reduce(operator.mul, x, axis=axis, dtype=dtype, keepdims=keepdims)


def add(x1, x2) -> LazyTensor:
    return elementwise(operator.add, defer(x1), defer(x2))


def subtract(x1, x2) -> LazyTensor:
    return elementwise(operator.sub, defer(x1), defer(x2))


def multiply(x1, x2) -> LazyTensor:
    return elementwise(operator.mul, defer(x1), defer(x2))


def abs(x) -> LazyTensor:
    return elementwise(operator.abs, defer(x))


def positive(x) -> LazyTensor:
    return elementwise(operator.pos, defer(x))


def negative(x) -> LazyTensor:
    return elementwise(operator.neg, defer(x))


def any(
        arr: LazyTensor,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        init = None
    ) -> LazyTensor:
    """
    Test whether any element of input array ``arr`` along given axis is True.
    """
    if init is None:
        init = init_value(logical_or, arr)
    return reduce(logical_or, arr, axis=axis, keepdims=keepdims, init=init)

def all(
        arr: LazyTensor,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        init = None
    ) -> LazyTensor:
    """
    Test whether all elements of input array ``arr`` along given axis are True.
    """
    if init is None:
        init = init_value(logical_and, arr)
    return reduce(logical_and, arr, axis=axis, keepdims=keepdims, init=init)

def min(
        arr: LazyTensor,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        init = None
    ) -> LazyTensor:
    """
    Return the minimum of input array ``arr`` along given axis.
    """
    if init is None:
        init = init_value(builtins.min, arr)
    return reduce(builtins.min, arr, axis=axis, keepdims=keepdims, init=init)

def max(
        arr: LazyTensor,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        init = None
    ) -> LazyTensor:
    """
    Return the maximum of input array ``arr`` along given axis.
    """
    if init is None:
        init = init_value(builtins.max, arr)
    return reduce(builtins.max, arr, axis=axis, keepdims=keepdims, init=init)

def mean(
        arr: LazyTensor,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False
    ) -> LazyTensor:
    """
    Calculates the arithmetic mean of the input array ``arr``.
    """
    origin = np.asarray(arr.shape)
    ele_no = origin[axis]
    n = defer(np.full(np.delete(origin, axis), ele_no, dtype=int))
    s = sum(arr, axis=axis, keepdims=keepdims)
    return elementwise(operator.truediv, s, n)

def var(
        arr: LazyTensor,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        correction: int | float = 0.0,
        keepdims: bool = False
    ) -> LazyTensor:
    """
    Calculates the variance of the input array ``arr``.
    """
    m = mean(arr, axis=axis, keepdims=keepdims)
    v = arr - m
    # n = defer(np.full(arr.shape, 2, dtype=int))
    # v2 = elementwise(operator.pow, v, 2)
    v2 = elementwise(operator.mul, v, v)
    # return elementwise(operator.truediv, sum(v, axis=axis, keepdims=keepdims), n)

    return v2

def std(
        arr: LazyTensor,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        correction: int | float = 0.0,
        keepdims: bool = False
    ) -> LazyTensor:
    """
    Calculates the standard deviation of the input array ``arr``.
    """
    d = var(arr, axis=axis, keepdims=keepdims, correction=correction)
    return elementwise(operator.pow, d, 0.5)

def argmin(
        arr: LazyTensor,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False
    ) -> LazyTensor:
    """
    Returns the indices of the minimum values along a specified axis.
    """
    return reduce(operator.add, arr, axis=axis, keepdims=keepdims, init=arr.fill_value)

def argmax(
        arr: LazyTensor,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False
    ) -> LazyTensor:
    """
    Returns the indices of the maximum values along a specified axis.
    """
    return reduce(operator.add, arr, axis=axis, keepdims=keepdims, init=arr.fill_value)
