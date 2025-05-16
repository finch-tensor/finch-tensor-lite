import operator
import builtins
# TODO: Is numpy allowed within lazy tensor interface?
import numpy as np
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Tuple, Iterable
from itertools import accumulate
from numpy.core.numeric import normalize_axis_index, normalize_axis_tuple
from ..algebra import *

from ..finch_logic import *
from ..symbolic import gensym

@dataclass
class LazyTensor:
    data: LogicNode
    shape: Tuple
    fill_value: Any
    element_type: Any

    @property
    def ndim(self) -> int:
        return len(self.shape)

def lazy(arr) -> LazyTensor:
    """
        - lazy(arr) -> LazyTensor:
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

def get_default_scheduler():
    return FinchLogicInterpreter()

def compute(arg, ctx=get_default_scheduler()):
    """
    - compute(arg, ctx=get_default_scheduler()):
    Executes a fused operation represented by LazyTensors. This function evaluates the entire
    operation in an optimized manner using the provided scheduler.

    Parameters:
    - arg: A lazy tensor or a tuple of lazy tensors representing the fused operation to be computed.
    - ctx: The scheduler to use for computation. Defaults to the result of `get_default_scheduler()`.

    Returns:
    - A tensor or a list of tensors computed by the fused operation.
    """
    if isinstance(arg, tuple):
        args = arg
    else:
        args = (arg,)
    vars = tuple(Alias(gensym("A")) for _ in args)
    bodies = tuple(map(lambda arg, var: Query(var, arg.data), args, vars))
    prgm = Plan(bodies + (Produces(vars),))
    res = ctx(prgm)
    if isinstance(arg, tuple):
        return tuple(res)
    else:
        return res[0]

register_property(LazyTensor, "__self__", "fill_value", lambda x: x.fill_value)
register_property(LazyTensor, "__self__", "element_type", lambda x: x.element_type)

def permute_dims(arg: LazyTensor, /, axis: Tuple[int, ...]) -> LazyTensor:
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
    axis = normalize_axis_tuple(axis, arg.ndim + len(axis))
    idxs = [Field(gensym("i")) for _ in range(arg.ndim)]
    return LazyTensor(
        Reorder(Relabel(arg.data, idxs), [idxs[i] for i in axis]),
        [arg.shape[i] for i in axis],
        arg.fill_value,
        arg.element_type,
    )

def identify(data):
    lhs = Alias(gensym("A"))
    return Subquery(lhs, data)

def expand_dims(
    x: LazyTensor,
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
        1 if n in axis else x.shape[n - offset[n]]
        for n in range(x.ndim + len(axis))
    )
    return LazyTensor(data_2, shape_2, x.fill_value, x.element_type)

def squeeze(
    x: LazyTensor,
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
    x: LazyTensor,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype = None,
    keepdims: bool = False,
    init = None):
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
    if init is None:
        init = init_value(op, x.element_type)
    axis = normalize_axis_tuple(axis, x.ndim)
    shape = tuple(x.shape[n] for n in range(x.ndim) if n not in axis)
    fields = [Field(gensym("i")) for _ in range(x.ndim)]
    data = Aggregate(Immediate(op), Immediate(init), Relabel(x.data, fields), [fields[i] for i in axis])
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
    largs = list(map(lazy, args))
    ndim = builtins.max([arg.ndim for arg in largs])
    shape = tuple(
        builtins.max([arg.shape[i - ndim + arg.ndim] if i - ndim + arg.ndim >= 0 else 1 for arg in largs])
        for i in range(ndim)
    )
    idxs = [Field(gensym("i")) for _ in range(ndim)]
    bargs = []
    for arg in largs:
        idims = []
        odims = []
        for i in range(ndim - arg.ndim,ndim):
            if arg.shape[i - ndim + arg.ndim] == shape[i]:
                idims.append(idxs[i])
                odims.append(idxs[i])
            else:
                if arg.shape[i - ndim + arg.ndim] != 1:
                    raise ValueError("Invalid shape for broadcasting")
                idims.append(Field(gensym("j")))
        bargs.append(Reorder(Relabel(arg.data, tuple(idims)), tuple(odims)))
    data = MapJoin(Immediate(f), tuple(bargs))
    new_fill_value = f(*[x.fill_value for x in largs])
    new_element_type = return_type(f, *[x.element_type for x in largs])
    return LazyTensor(identify(data), shape, new_fill_value, new_element_type)

def prod(
        arr: LazyTensor,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype = None,
        keepdims: bool = False
    ) -> LazyTensor:

    """
    Calculates the product of input array ``arr`` elements.

    Parameters
    ----------
    arr: array
        input array. Should have a numeric data type.
    axis: Optional[Union[int, Tuple[int, ...]]]
        axis or axes along which products must be computed. By default, the product must be computed over the entire array. If a tuple of integers, products must be computed over multiple axes. Default: ``None``.

    dtype: Optional[dtype]
        data type of the returned array. If ``None``, the returned array must have the same data type as ``x``, unless ``x`` has an integer data type supporting a smaller range of values than the default integer data type (e.g., ``x`` has an ``int16`` or ``uint32`` data type and the default integer data type is ``int64``). In those latter cases:

        -   if ``x`` has a signed integer data type (e.g., ``int16``), the returned array must have the default integer data type.
        -   if ``x`` has an unsigned integer data type (e.g., ``uint16``), the returned array must have an unsigned integer data type having the same number of bits as the default integer data type (e.g., if the default integer data type is ``int32``, the returned array must have a ``uint32`` data type).

        If the data type (either specified or resolved) differs from the data type of ``x``, the input array should be cast to the specified data type before computing the sum (rationale: the ``dtype`` keyword argument is intended to help prevent overflows). Default: ``None``.

    keepdims: bool
        if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result. Default: ``False``.

    Returns
    -------
    out: array
        if the product was computed over the entire array, a zero-dimensional array containing the product; otherwise, a non-zero-dimensional array containing the products. The returned array must have a data type as described by the ``dtype`` parameter above.

    Notes
    -----

    **Special Cases**

    Let ``N`` equal the number of elements over which to compute the product.

    -   If ``N`` is ``0``, the product is `1` (i.e., the empty product).

    For both real-valued and complex floating-point operands, special cases must be handled as if the operation is implemented by successive application of :func:`~array_api.multiply`.

    .. versionchanged:: 2022.12
       Added complex data type support.

    .. versionchanged:: 2023.12
       Required the function to return a floating-point array having the same data type as the input array when provided a floating-point array.
    """
    return reduce(operator.mul, arr, axis=axis, keepdims=keepdims, init=arr.fill_value)

def multiply(x1: LazyTensor, x2: LazyTensor) -> LazyTensor:
    return elementwise(operator.mul, x1, x2)

def sum(
        arr: LazyTensor,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype = None,
        keepdims: bool = False
    ) -> LazyTensor:
    """
    Calculates the sum of input array ``arr`` elements.
    """
    return reduce(operator.add, arr, axis=axis, keepdims=keepdims, init=arr.fill_value)

def any(
        arr: LazyTensor,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False
    ) -> LazyTensor:
    """
    Test whether any element of input array ``arr`` along given axis is True.
    """
    return reduce(operator.or_, arr, axis=axis, keepdims=keepdims, init=arr.fill_value)

def all(
        arr: LazyTensor,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False
    ) -> LazyTensor:
    """
    Test whether all elements of input array ``arr`` along given axis are True.
    """
    return reduce(operator.and_, arr, axis=axis, keepdims=keepdims, init=arr.fill_value)

# def _flatten(x):
#     for y in x:
#         if isinstance(y, (list, tuple)):
#             yield from _flatten(y)
#         else:
#             yield y

def min(
        arr: LazyTensor,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False    
    ) -> LazyTensor:
    """
    Return the minimum of input array ``arr`` along given axis.
    """
#     num_rows = arr.shape[0]
#     num_cols = arr.shape[1]

#     if axis is None:
#         return min(_flatten(arr))

#     # column‑wise minimum
#     if axis == 0:
#         return [min(arr[r][c] for r in range(num_rows)) for c in range(num_cols)]
#     # row‑wise minimum

#     if axis == 1:
#         return [min(row) for row in arr]

    # TODO: Use operator.min to reduce?
    return reduce(operator.add, arr, axis=axis, keepdims=keepdims, init=arr.fill_value)

def max(
        arr: LazyTensor,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False
    ) -> LazyTensor:
    """
    Return the maximum of input array ``arr`` along given axis.
    """
    # TODO: Use operator.max to reduce?
    return reduce(operator.add, arr, axis=axis, keepdims=keepdims, init=arr.fill_value)


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
    n = lazy(np.full(np.delete(origin, axis), ele_no, dtype=int))

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
    origin = np.asarray(arr.shape)
    ele_no = origin[axis]
    n = lazy(np.full(np.delete(origin, axis), ele_no, dtype=int))

    m = mean(arr, axis=axis, keepdims=keepdims)
    # TODO: Confirm whether we have broadcasting or reshape interface for lazyTensor
    # v = elementwise(operator.sub, arr, m)
    # v2 = elementwise(operator.mul, v, v)
    # return elementwise(operator.truediv, sum(v, axis=axis, keepdims=keepdims), n)

    return reduce(operator.add, arr, axis=axis, keepdims=keepdims, init=arr.fill_value)

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
    # TODO: Need sqrt operator?
    # d = var(arr, axis=axis, keepdims=keepdims, correction=correction)
    # return elementwise(operator.sqrt, d)
    return reduce(operator.add, arr, axis=axis, keepdims=keepdims, init=arr.fill_value)

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
