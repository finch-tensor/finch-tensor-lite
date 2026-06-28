from __future__ import annotations

import builtins
import sys
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate, zip_longest
from typing import Any

import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple

from finchlite import finch_einsum as ein
from finchlite.algebra import (
    FinchOperator,
    FType,
    Tensor,
    TensorFType,
    ffuncs,
    fixpoint_type,
    ftype,
    init_value,
    return_type,
)
from finchlite.algebra.ftypes import (
    FDTypeBoolean,
    FDTypeBuiltin,
    FDTypeNumpy,
)
from finchlite.autoschedule.tensor_stats import StatsInterpreter
from finchlite.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    LogicExpression,
    LogicStatement,
    MapJoin,
    Plan,
    Query,
    Reorder,
    StatsFactory,
    Table,
    TensorStats,
)
from finchlite.symbolic import gensym
from finchlite.tensor import BufferizedNDArray
from finchlite.tensor.override_tensor import OverrideTensor
from finchlite.tensor.scalar import Scalar


class LazyTensorFType(TensorFType):
    _fill_value: Any
    _element_type: FType
    _shape_type: tuple[FType, ...]

    def __init__(
        self,
        _fill_value: Any,
        _element_type: FType | type,
        _shape_type: tuple[FType | type, ...],
    ):
        self._fill_value = _fill_value
        self._element_type = ftype(_element_type)
        self._shape_type = tuple(ftype(dim_t) for dim_t in _shape_type)

    def __eq__(self, other):
        if not isinstance(other, LazyTensorFType):
            return False
        return (
            ffuncs.same(self._fill_value, other._fill_value)
            and self._element_type == other._element_type
            and self._shape_type == other._shape_type
        )

    def __hash__(self):
        return hash(
            (ffuncs.samehash(self._fill_value), self._element_type, self._shape_type)
        )

    def construct(self, shape: tuple) -> LazyTensor:
        idxs = tuple(Field(gensym("i")) for _ in shape)
        ctx = EffectBlob()
        expr = Table(Literal(FillTensor(shape, self._fill_value)), idxs)
        data, ctx = ctx.eval(expr)
        return LazyTensor(
            data=data,
            ctx=ctx,
            shape=shape,
            fill_value=self._fill_value,
            element_type=self._element_type,
        )

    def __call__(self, val: Any) -> LazyTensor:
        """
        Convert a tensor to this tensor type.

        Args:
            val: A tensor to convert to this tensor type.
        Returns:
            A LazyTensor instance of this type.
        """
        raise NotImplementedError(
            f"Tensor conversion not yet implemented for {type(self).__name__}"
        )

    @property
    def fill_value(self):
        return self._fill_value

    @property
    def element_type(self) -> FType:
        return self._element_type

    @property
    def shape_type(self):
        return self._shape_type

    def from_numpy(self, arr):
        raise NotImplementedError


effect_stamp = threading.local()


class EffectBlob:
    """
    The EffectBlob class represents a collection of prior queries in a lazy
    evaluation context, which can be extracted into a linear trace of
    LogicStatements (in the order they were issued). Note that the global
    `effect_stamp` counter is required to maintain the relative ordering of
    effect blobs in the face of in-place mutations to tensors.  Each effect blob
    contains a logic statement and references to other effect blobs. The
    datastructure is designed to be immutable, mergeable in O(1) time, and
    support linear time extraction. To achieve linear time, we use a hash table
    to avoid re-extracting already seen blobs.
    """

    stamp: int
    stmt: LogicStatement
    blobs: tuple[EffectBlob, ...]

    def __init__(
        self,
        stmt: LogicStatement | None = None,
        blobs: tuple[EffectBlob, ...] | None = None,
    ):
        global effect_stamp
        if stmt is None:
            stmt = Plan()
        if blobs is None:
            blobs = ()
        self.stmt = stmt
        self.blobs = blobs

        try:
            curr_stamp = effect_stamp.value
        except AttributeError:
            effect_stamp.value = 0
            curr_stamp = 0

        self.stamp = curr_stamp
        effect_stamp.value += 1

    def exec(self, stmt: LogicStatement) -> EffectBlob:
        return EffectBlob(stmt=stmt, blobs=(self,))

    def eval(self, ex: LogicExpression) -> tuple[Alias, EffectBlob]:
        var = Alias(gensym("A"))
        return var, self.exec(Query(var, ex))

    def join(self, *blobs: EffectBlob) -> EffectBlob:
        return EffectBlob(blobs=(self, *blobs))

    def trace(self) -> tuple[LogicStatement, ...]:
        stmts = list[tuple[int, LogicStatement]]()
        seen = set[int]()
        self._trace(seen, stmts)
        stmts.sort()
        return tuple(stmt for _, stmt in stmts)

    def _trace(self, seen: set[int], stmts: list[tuple[int, LogicStatement]]) -> None:
        if id(self) not in seen:
            seen.add(id(self))
            stmts.append((self.stamp, self.stmt))
            for blob in self.blobs:
                blob._trace(seen, stmts)


class LazyTensor(OverrideTensor):
    def __init__(
        self,
        data: Alias,
        ctx: EffectBlob,
        shape: tuple,
        fill_value: Any,
        element_type: FType,
    ):
        self.data: Alias = data
        self.ctx = ctx
        self._shape = shape
        self._fill_value = fill_value
        self._element_type = element_type

    def override_module(self):
        return sys.modules[__name__]

    @property
    def ftype(self):
        return LazyTensorFType(
            _fill_value=self._fill_value,
            _element_type=self._element_type,
            _shape_type=tuple(ftype(dim) for dim in self._shape),
        )

    @property
    def shape(self) -> tuple:
        """
        Returns the shape of the LazyTensor as a tuple.
        The shape is determined by the data and is a static property.
        """
        return self._shape

    @property
    def fill_value(self) -> Any:
        """Default value to fill the tensor."""
        return self.ftype.fill_value

    @property
    def element_type(self) -> FType:
        """Data type of the tensor elements."""
        return self.ftype.element_type

    @property
    def shape_type(self) -> tuple:
        """Shape type of the tensor."""
        return self.ftype.shape_type

    def item(self):
        raise ValueError(
            "Cannot convert LazyTensor to Python scalar. "
            "Use compute() to evaluate it first."
        )

    # raise ValueError for unsupported operations according to the data-apis spec.
    # NOT tested, since this isn't necessary as it will throw an error anyways.
    def __complex__(self) -> complex:
        """
        Converts the LazyTensor to a complex number.
        This is a no-op for LazyTensors, as they are symbolic representations.
        """
        raise ValueError(
            "Cannot convert LazyTensor to complex. Use compute() to evaluate it first."
        )

    def __float__(self) -> float:
        """
        Converts the LazyTensor to a float.
        This is a no-op for LazyTensors, as they are symbolic representations.
        """
        raise ValueError(
            "Cannot convert LazyTensor to float. Use compute() to evaluate it first."
        )

    def __int__(self) -> int:
        """
        Converts the LazyTensor to an int.
        This is a no-op for LazyTensors, as they are symbolic representations.
        """
        raise ValueError(
            "Cannot convert LazyTensor to int. Use compute() to evaluate it first."
        )

    def __bool__(self) -> bool:
        """
        Converts the LazyTensor to a bool.
        This is a no-op for LazyTensors, as they are symbolic representations.
        """
        raise ValueError(
            "Cannot convert LazyTensor to bool. Use compute() to evaluate it first."
        )


def asarray(
    obj: Any,
    /,
    *,
    dtype=None,
    device=None,
    copy=None,
    format: TensorFType | None = None,
) -> Any:
    """
    Convert given argument and return wrapper type instance.
    If input argument is already array type, return unchanged.
    https://data-apis.org/array-api/latest/API_specification/generated/array_api.asarray.html
    """
    if device is not None:
        raise ValueError(f"device argument is not supported; got {device!r}")

    if format is None:
        if isinstance(obj, BufferizedNDArray):
            if copy is True:
                return BufferizedNDArray.from_numpy(
                    obj.to_numpy().copy(), fill_value=obj.fill_value
                )
            return obj
        if isinstance(obj, np.ndarray):
            if copy is True:
                obj = obj.copy()
            return BufferizedNDArray.from_numpy(obj)
        if np.isscalar(obj) or obj is None:
            if dtype is not None:
                obj = ftype(dtype)(obj)
            elif isinstance(obj, bool | int | float | complex):
                obj = np.asarray(obj)[()]
            return Scalar(obj)
        try:
            np_arr = np.asarray(obj)
            if np_arr.dtype != object:
                if dtype is not None:
                    ft = ftype(dtype)
                    np_dtype = (
                        ft.dtype
                        if hasattr(ft, "dtype")
                        else ft.type
                        if hasattr(ft, "type")
                        else dtype
                    )
                    np_arr = np_arr.astype(np_dtype)
                elif copy is True:
                    np_arr = np_arr.copy()
                return BufferizedNDArray.from_numpy(np_arr)
        except (TypeError, ValueError):
            pass
        return obj

    if isinstance(obj, np.ndarray):
        return format.from_numpy(obj)
    return format(obj)


def _is_convertible_to_array(arg: Any) -> bool:
    return isinstance(arg, np.ndarray) or np.isscalar(arg) or arg is None


def lazy(arr) -> LazyTensor:
    """
    - lazy(arr) -> LazyTensor:
    Converts an array into a LazyTensor. If the input is already a LazyTensor, it is
    returned as-is.
    Otherwise, it creates a LazyTensor representation of the input array.

    Parameters:
    - arr: The input array to be converted into a LazyTensor.

    Returns:
    - LazyTensor: A lazy representation of the input array.
    """
    if not isinstance(arr, Tensor) and not _is_convertible_to_array(arr):
        return arr

    if isinstance(arr, LazyTensor):
        return arr
    arr = Scalar(arr) if isinstance(arr, bool | int | float | complex) else asarray(arr)
    tns = Alias(gensym("A"))
    idxs = tuple(Field(gensym("i")) for _ in range(arr.ndim))
    shape = tuple(arr.shape)
    ctx = EffectBlob(stmt=Query(tns, Table(Literal(arr), idxs)))
    return LazyTensor(tns, ctx, shape, arr.fill_value, arr.element_type)


def _np_dtype(dtype):
    if isinstance(dtype, FDTypeNumpy):
        return dtype.dtype
    if isinstance(dtype, FDTypeBuiltin):
        return dtype.type
    return dtype


def full(
    shape: int | tuple[int, ...],
    fill_value: bool | complex,
    *,
    dtype: Any | None = None,
):
    """
    Returns a new array having a specified shape and filled with fill_value.

    Parameters:
    - shape (Union[int, Tuple[int, ...]]): output array shape.
    - fill_value (Union[bool, int, float, complex]): fill value.
    - dtype (Optional[dtype]): output array data type. If dtype is None, the
    output array data type must be inferred from fill_value according to the
    following rules:
        * If the fill value is an int, the output array data type must be the
            default integer data type.
        * If the fill value is a float, the output array data type must be the
            default real-valued floating-point data type.
        * If the fill value is a complex number, the output array data type must
            be the default complex floating-point data type.
        * If the fill value is a bool, the output array must have a boolean data
            type. Default: None.

    Returns:

    - out (array): an array where every element is equal to fill_value.
    """
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    return broadcast_to(asarray(fill_value, dtype=dtype), shape)


def full_like(x, /, fill_value, *, dtype=None):
    x = lazy(x)
    return full(
        x.shape, fill_value, dtype=dtype if dtype is not None else x.element_type
    )


def linspace(start, stop, /, num, *, dtype=None, endpoint=True):
    return broadcast_to(
        lazy(np.linspace(start, stop, num, endpoint=endpoint, dtype=_np_dtype(dtype))),
        (num,),
    )


def zeros(shape: int | tuple[int, ...], *, dtype=None) -> LazyTensor:
    return full(shape, 0, dtype=dtype if dtype is not None else np.float64)


def ones(shape: int | tuple[int, ...], *, dtype=None) -> LazyTensor:
    return full(shape, 1, dtype=dtype if dtype is not None else np.float64)


def empty(shape: int | tuple[int, ...], *, dtype=None) -> LazyTensor:
    return full(shape, 0, dtype=dtype if dtype is not None else np.float64)


def zeros_like(x, /, *, dtype=None) -> LazyTensor:
    return full_like(x, 0, dtype=dtype)


def ones_like(x, /, *, dtype=None) -> LazyTensor:
    return full_like(x, 1, dtype=dtype)


def arange(
    start: float,
    /,
    stop: float | None = None,
    step: float = 1,
    *,
    dtype=None,
) -> LazyTensor:
    if stop is None:
        start, stop = 0, start
    arr = np.arange(start, stop, step, dtype=_np_dtype(dtype))
    return broadcast_to(lazy(arr), (len(arr),))


def permute_dims(arg, /, axes: tuple[int, ...]) -> LazyTensor:
    """
    Permutes the axes (dimensions) of an array ``x``.

    Parameters
    ----------
    x: array
        input array.
    axes: Tuple[int, ...]
        tuple containing a permutation of ``(0, 1, ..., N-1)`` where ``N`` is the number
        of axes (dimensions) of ``x``.

    Returns
    -------
    out: array
        an array containing the axes permutation. The returned array must have the same
        data type as ``x``.
    """
    arg = lazy(arg)
    axis = normalize_axis_tuple(axes, arg.ndim + len(axes))
    idxs = tuple(Field(gensym("i")) for _ in range(arg.ndim))
    expr = Reorder(Table(arg.data, idxs), tuple(idxs[i] for i in axis))
    data, ctx = arg.ctx.eval(expr)
    return LazyTensor(
        data,
        ctx,
        tuple(arg.shape[i] for i in axis),
        arg.fill_value,
        arg.element_type,
    )


def expand_dims(
    x,
    /,
    axis: int | tuple[int, ...] = 0,
) -> LazyTensor:
    """
    Expands the shape of an array by inserting a new axis (dimension) of size one at the
    position specified by ``axis``.

    Parameters
    ----------
    x: array
        input array.
    axis: int
        axis position (zero-based). If ``x`` has rank (i.e, number of dimensions) ``N``,
        a valid ``axis`` must reside on the closed-interval ``[-N-1, N]``. If provided a
        negative ``axis``, the axis position at which to insert a singleton dimension
        must be computed as ``N + axis + 1``. Hence, if provided ``-1``, the resolved
        axis position must be ``N`` (i.e., a singleton dimension must be appended to the
        input array ``x``). If provided ``-N-1``, the resolved axis position must be
        ``0`` (i.e., a singleton dimension must be prepended to the input array ``x``).

    Returns
    -------
    out: array
        an expanded output array having the same data type as ``x``.

    Raises
    ------
    IndexError
        If provided an invalid ``axis`` position, an ``IndexError`` should be raised.
    """
    x = lazy(x)
    if isinstance(axis, int):
        axis = (axis,)
    axis = normalize_axis_tuple(axis, x.ndim + len(axis))
    if isinstance(axis, int):
        raise IndexError(
            f"Invalid axis: {axis}. Axis must be an integer or a tuple of integers."
        )
    if not len(axis) == len(set(axis)):
        raise IndexError("axis must be unique")
    if not set(axis).issubset(range(x.ndim + len(axis))):
        raise IndexError(
            f"Invalid axis: {axis}. Axis must be unique and must be in the range "
            f"[-{x.ndim + len(axis) - 1}, {x.ndim + len(axis) - 1}]."
        )
    offset = [0] * (x.ndim + len(axis))
    for d in axis:
        offset[d] = 1
    offset = list(accumulate(offset))
    idxs_1 = tuple(Field(gensym("i")) for _ in range(x.ndim))
    idxs_2 = tuple(
        Field(gensym("i")) if n in axis else idxs_1[n - offset[n]]
        for n in range(x.ndim + len(axis))
    )
    expr = Reorder(Table(x.data, idxs_1), idxs_2)
    shape_2 = tuple(
        1 if n in axis else x.shape[n - offset[n]] for n in range(x.ndim + len(axis))
    )
    data_2, ctx = x.ctx.eval(expr)
    return LazyTensor(data_2, ctx, shape_2, x.fill_value, x.element_type)


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
    x = lazy(x)
    if isinstance(axis, int):
        axis = (axis,)
    axis = normalize_axis_tuple(axis, x.ndim)
    if isinstance(axis, int):
        raise ValueError(f"Invalid axis: {axis}. Axis must be a tuple of integers.")
    if len(axis) != len(set(axis)):
        raise ValueError(f"Invalid axis: {axis}. Axis must be unique.")
    if not set(axis).issubset(range(x.ndim)):
        raise ValueError(f"Invalid axis: {axis}. Axis must be within bounds.")
    if not builtins.all(x.shape[d] == 1 for d in axis):
        raise ValueError(f"Invalid axis: {axis}. Axis to drop must have size 1.")
    newaxis = [n for n in range(x.ndim) if n not in axis]
    idxs_1 = tuple(Field(gensym("i")) for _ in range(x.ndim))
    idxs_2 = tuple(idxs_1[n] for n in newaxis)
    expr = Reorder(Table(x.data, idxs_1), idxs_2)
    shape_2 = tuple(x.shape[n] for n in newaxis)
    data_2, ctx = x.ctx.eval(expr)
    return LazyTensor(data_2, ctx, shape_2, x.fill_value, x.element_type)


def reduce(
    op: FinchOperator,
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
        axis or axes along which reduction must be computed. By default, the reduction
        must be computed over the entire array. If a tuple of integers, reductions must
        be computed over multiple axes. Default: ``None``.

    dtype: Optional[dtype]
        data type of the returned array. If ``None``, a suitable data type will be
        calculated.

    keepdims: bool
        if ``True``, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes
        (dimensions) must not be included in the result. Default: ``False``.

    init: Optional
        Initial value for the reduction. If ``None``, a suitable initial value will be
        calculated. The initial value must be compatible with the operation defined by
        ``op``. For example, if ``op`` is addition, the initial value should be zero; if
        ``op`` is multiplication, the initial value should be one.

    Returns
    -------
    out: array
        If the reduction was computed over the entire array, a zero-dimensional array
        containing the reduction; otherwise, a non-zero-dimensional array containing the
        reduction. The returned array must have a data type as described by the
        ``dtype`` parameter above.
    """
    x = lazy(x)
    assert isinstance(op, FinchOperator)
    explicit_dtype = dtype is not None
    if dtype is not None:
        dtype = ftype(dtype)
    if init is None:
        init = init_value(op, dtype or x.element_type)
        if explicit_dtype:
            assert dtype is not None
            init = dtype(init)
    if axis is None:
        axis = tuple(range(x.ndim))
    axis = normalize_axis_tuple(axis, x.ndim)
    if axis is None or isinstance(axis, int):
        raise ValueError("axis must be a tuple")

    shape = tuple(x.shape[n] for n in range(x.ndim) if n not in axis)
    fields = tuple(Field(gensym("i")) for _ in range(x.ndim))
    data: LogicExpression = Aggregate(
        Literal(op),
        Literal(init),
        Table(x.data, fields),
        tuple(fields[i] for i in axis),
    )
    if keepdims:
        keeps = tuple(
            fields[i] if i not in axis else Field(gensym("j")) for i in range(x.ndim)
        )
        data = Reorder(data, keeps)
        shape = tuple(x.shape[i] if i not in axis else 1 for i in range(x.ndim))
    if dtype is None:
        dtype = fixpoint_type(op, init, x.element_type)
    expr, ctx = x.ctx.eval(data)
    return LazyTensor(expr, ctx, shape, init, dtype)


def _broadcast_shape(*args: tuple) -> tuple:
    """
    Computes the broadcasted shape for the given LazyTensor arguments,
    following array_api broadcasting rules.
    Raises ValueError if shapes are not broadcastable.

    Parameters:
    --------------
    - *args: Variable number of tuples representing shapes of LazyTensors.

    Returns:
    --------------
    tuple: The broadcasted shape as a tuple.
    """
    if len(args) < 2:
        return args[0] if args else ()
    shape1, shape2 = args[0], args[1]
    N1, N2 = len(shape1), len(shape2)
    N = builtins.max(N1, N2)
    _shape = [0] * N
    for i in range(N - 1, -1, -1):
        n1, n2 = N1 - N + i, N2 - N + i
        d1 = shape1[n1] if n1 >= 0 else 1
        d2 = shape2[n2] if n2 >= 0 else 1
        if d1 == 1:
            _shape[i] = d2
        elif d2 == 1 or d1 == d2:
            _shape[i] = d1
        else:
            raise ValueError(f"Shapes {shape1} and {shape2} are not broadcastable")
    shape = tuple(_shape)

    if len(args) > 2:
        for arg in args[2:]:
            shape = _broadcast_shape(shape, arg)
    return shape


def elementwise(f: FinchOperator, *args) -> LazyTensor:
    """
        elementwise(f, *args) -> LazyTensor:

    Applies the function f elementwise to the given arguments, following
    [broacasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html).
    The function f should be a callable that takes the same number of arguments
    as the number of tensors passed to `elementwise`.

    The function will automatically handle broadcasting of the input tensors to
    ensure they have compatible shapes.  For example, `elementwise(ffunc.add,
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
    args = tuple(lazy(a) for a in args)
    shape = _broadcast_shape(*(arg.shape for arg in args))
    ndim = len(shape)
    idxs = tuple(Field(gensym("i")) for _ in range(ndim))
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
        bargs.append(Reorder(Table(arg.data, tuple(idims)), tuple(odims)))
    expr = Reorder(MapJoin(Literal(f), tuple(bargs)), idxs)
    new_fill_value = f(*[x.fill_value for x in args])
    new_element_type = return_type(f, *[x.element_type for x in args])
    ctx = args[0].ctx.join(*[x.ctx for x in args[1:]])
    data, ctx = ctx.eval(expr)
    return LazyTensor(data, ctx, shape, new_fill_value, new_element_type)


def round(x) -> LazyTensor:
    return elementwise(ffuncs.round, lazy(x))


def floor(x) -> LazyTensor:
    return elementwise(ffuncs.floor, lazy(x))


def ceil(x) -> LazyTensor:
    return elementwise(ffuncs.ceil, lazy(x))


def astype(x, dtype, /, *, copy=True, device=None) -> LazyTensor:
    if device is not None:
        raise ValueError(f"device argument is not supported; got {device!r}")
    return elementwise(ffuncs.astype(ftype(dtype)), lazy(x))


def trunc(x) -> LazyTensor:
    return elementwise(ffuncs.trunc, lazy(x))


def sum(
    x,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype=None,
    keepdims: bool = False,
):
    x = lazy(x)
    return reduce(ffuncs.add, x, axis=axis, dtype=dtype, keepdims=keepdims)


def prod(
    x,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype=None,
    keepdims: bool = False,
):
    x = lazy(x)
    return reduce(ffuncs.mul, x, axis=axis, dtype=dtype, keepdims=keepdims)


def any(
    x,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    init=None,
):
    """
    Test whether any element of input array ``x`` along given axis is True.
    """
    x = lazy(x)
    return reduce(
        ffuncs.or_,
        elementwise(ffuncs.truth, x),
        axis=axis,
        keepdims=keepdims,
        init=init,
    )


def all(
    x,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    init=None,
):
    """
    Test whether all elements of input array ``x`` along given axis are True.
    """
    x = lazy(x)
    return reduce(
        ffuncs.and_,
        elementwise(ffuncs.truth, x),
        axis=axis,
        keepdims=keepdims,
        init=init,
    )


def real(x) -> LazyTensor:
    return elementwise(ffuncs.real, lazy(x))


def imag(x) -> LazyTensor:
    return elementwise(ffuncs.imag, lazy(x))


def min(
    x,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    init=None,
):
    """
    Return the minimum of input array ``arr`` along given axis.
    """
    x = lazy(x)
    return reduce(ffuncs.min, x, axis=axis, keepdims=keepdims, init=init)


def max(
    x,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    init=None,
):
    """
    Return the maximum of input array ``arr`` along given axis.
    """
    x = lazy(x)
    return reduce(ffuncs.max, x, axis=axis, keepdims=keepdims, init=init)


def minimum(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.min, lazy(x1), lazy(x2))


def maximum(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.max, lazy(x1), lazy(x2))


def clip(x, /, min=None, max=None) -> LazyTensor:
    return elementwise(ffuncs.clip, lazy(x), lazy(min), lazy(max))


def sqrt(x) -> LazyTensor:
    return elementwise(ffuncs.sqrt, lazy(x))


def square(x) -> LazyTensor:
    return elementwise(ffuncs.square, lazy(x))


def sign(x) -> LazyTensor:
    return elementwise(ffuncs.sign, lazy(x))


def add(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.add, lazy(x1), lazy(x2))


def reciprocal(x) -> LazyTensor:
    return elementwise(ffuncs.reciprocal, lazy(x))


def subtract(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.sub, lazy(x1), lazy(x2))


def multiply(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.mul, lazy(x1), lazy(x2))


def divide(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.truediv, lazy(x1), lazy(x2))


def abs(x) -> LazyTensor:
    return elementwise(ffuncs.abs, lazy(x))


def positive(x) -> LazyTensor:
    return elementwise(ffuncs.pos, lazy(x))


def negative(x) -> LazyTensor:
    return elementwise(ffuncs.neg, lazy(x))


def is_broadcastable(shape_a, shape_b):
    """
    Returns True if shape_a and shape_b are broadcastable according to numpy rules.
    """
    for a, b in zip_longest(reversed(shape_a), reversed(shape_b), fillvalue=1):
        if a != b and a != 1 and b != 1:
            return False
    return True


def is_broadcastable_directional(shape_a, shape_b):
    """
    Returns True if shape_a is broadcastable to shape_b according to numpy rules.
    This is a directional check, so it allows only shape_a to be changed
    """
    for a, b in zip_longest(reversed(shape_a), reversed(shape_b), fillvalue=1):
        if a != b and a != 1:
            return False
    return True


def matmul(x1, x2) -> LazyTensor:
    """
    Performs matrix multiplication between two tensors.
    """

    def _matmul_helper(a, b) -> LazyTensor:
        """
        For arrays greater than 1D
        """
        if a.ndim < 2 or b.ndim < 2:
            raise ValueError("Both inputs must be at least 2D arrays")
        if a.shape[-1] != b.shape[-2]:
            raise ValueError("Dimensions mismatch for matrix multiplication")
        # check all preceeding dimensions match
        batch_a, batch_b = a.shape[:-2], b.shape[:-2]
        if not is_broadcastable(batch_a, batch_b):
            raise ValueError(
                "Batch dimensions are not broadcastable for matrix multiplication"
            )
        return reduce(
            ffuncs.add,
            multiply(expand_dims(a, axis=-1), expand_dims(b, axis=-3)),
            axis=-2,
        )

    x1 = lazy(x1)
    x2 = lazy(x2)

    if x1.ndim < 1 or x2.ndim < 1:
        raise ValueError("Both inputs must be at least 1D arrays")

    if x1.ndim == 1 and x2.ndim == 1:
        return reduce(ffuncs.add, multiply(x1, x2), axis=0)

    if x1.ndim == 1:
        x1 = expand_dims(x1, axis=0)  # make it a row vector
        result = _matmul_helper(x1, x2)
        return squeeze(result, axis=-2)  # remove the prepended singleton dimension

    if x2.ndim == 1:
        x2 = expand_dims(x2, axis=1)  # make it a column vector
        result = _matmul_helper(x1, x2)
        return squeeze(result, axis=-1)  # remove the appended singleton dimension

    return _matmul_helper(x1, x2)


def matrix_transpose(x) -> LazyTensor:
    """
    Transposes the input tensor `x`.

    Parameters
    ----------
    x: LazyTensor
        The input tensor to be transposed. Must have at least 2 dimensions.

    Returns
    -------
    LazyTensor
        A new LazyTensor with the axes of `x` transposed.
    """
    x = lazy(x)
    if x.ndim < 2:
        # this is following numpy's behavior.
        # data-apis specification assumes that input is atleast 2D
        raise ValueError(
            "Input tensor must have at least 2 dimensions for transposition"
        )
    # swap the last two axes
    return permute_dims(x, axes=(*range(x.ndim - 2), x.ndim - 1, x.ndim - 2))


def bitwise_invert(x) -> LazyTensor:
    return elementwise(ffuncs.invert, lazy(x))


def bitwise_and(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.and_, lazy(x1), lazy(x2))


def bitwise_left_shift(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.lshift, lazy(x1), lazy(x2))


def bitwise_or(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.or_, lazy(x1), lazy(x2))


def bitwise_right_shift(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.rshift, lazy(x1), lazy(x2))


def bitwise_xor(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.xor, lazy(x1), lazy(x2))


def truediv(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.truediv, lazy(x1), lazy(x2))


def floor_divide(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.floordiv, lazy(x1), lazy(x2))


def mod(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.mod, lazy(x1), lazy(x2))


def pow(x1, x2) -> LazyTensor:
    return power(x1, x2)


def power(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.pow, lazy(x1), lazy(x2))


def remainder(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.remainder, lazy(x1), lazy(x2))


def conj(x) -> LazyTensor:
    """
    Computes the complex conjugate of the input tensor `x`.

    Parameters
    ----------
    x: LazyTensor
        The input tensor to compute the complex conjugate of.

    Returns
    -------
    LazyTensor
        A new LazyTensor with the complex conjugate of `x`.
    """
    return elementwise(ffuncs.conjugate, lazy(x))


def count_nonzero(
    x, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> LazyTensor:
    x = lazy(x)
    element_type = x.element_type
    zero = element_type(False if isinstance(element_type, FDTypeBoolean) else 0)
    return reduce(
        ffuncs.add,
        elementwise(ffuncs.not_equal, x, lazy(zero)),
        axis=axis,
        keepdims=keepdims,
        init=0,
    )


def count_nonfill(
    x, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> LazyTensor:
    x = lazy(x)
    fill = full(x.shape, x.fill_value, dtype=x.element_type)
    return reduce(
        ffuncs.add,
        elementwise(ffuncs.not_same, x, fill),
        axis=axis,
        keepdims=keepdims,
        init=0,
    )


def tensordot(
    x1, x2, /, *, axes: int | tuple[Sequence[int], Sequence[int]]
) -> LazyTensor:
    """
    Computes the tensordot operation.

    Returns a LazyTensor if either x1 or x2 is a LazyTensor.
    Otherwise, computes the result eagerly.
    """
    x1 = lazy(x1)
    x2 = lazy(x2)

    # Parse axes
    if not isinstance(axes, tuple):
        N = int(axes)
        if N < 0:
            raise ValueError("expected axes to be a non-negative integer")
        axes_a = list(range(x1.ndim - N, x1.ndim))
        axes_b = list(range(N))
    else:
        axes_a, axes_b = (list(ax) for ax in axes)

    # Normalize negative axes. We need list
    axes_a = [(a if a >= 0 else x1.ndim + a) for a in axes_a]
    axes_b = [(b if b >= 0 else x2.ndim + b) for b in axes_b]

    # Check axes lengths and shapes
    if len(axes_a) != len(axes_b):
        raise ValueError("shape-mismatch for sum")
    for a, b in zip(axes_a, axes_b, strict=True):
        if x1.shape[a] != x2.shape[b]:
            raise ValueError("shape-mismatch for sum")

    # Move axes to contract to the end of x1 and to the front of x2
    notin_a = [k for k in range(x1.ndim) if k not in axes_a]
    notin_b = [k for k in range(x2.ndim) if k not in axes_b]
    newaxes_a = notin_a + axes_a
    newaxes_b = axes_b + notin_b
    # Permute
    x1p = permute_dims(x1, tuple(newaxes_a))
    x2p = permute_dims(x2, tuple(newaxes_b))

    # Expand x1p and x2p so that their contracted axes align for broadcasting
    # so we can multiply them

    # For x1p, add len(notin_b) singleton dims at the end
    added_dims = tuple(-(i + 1) for i in range(len(notin_b)))
    x1p = expand_dims(x1p, axis=added_dims)

    # For x2p, add len(notin_a) singleton dims at the front
    added_dims = tuple(i for i in range(len(notin_a)))
    x2p = expand_dims(x2p, axis=added_dims)

    # Multiply (broadcasted)
    expanded_product = multiply(x1p, x2p)

    sum_axes = tuple(range(len(notin_a), len(notin_a) + len(axes_a)))
    return sum(expanded_product, axis=sum_axes)


def vecdot(x1, x2, /, *, axis=-1) -> LazyTensor:
    """
    Computes the vector dot product along the specified axis.
    """
    x1 = lazy(x1)
    x2 = lazy(x2)

    # check broadcastability
    if not is_broadcastable(x1.shape, x2.shape):
        raise ValueError("Shapes are not broadcastable for vector dot product")

    # check if dims of axis are the same
    if x1.shape[axis] != x2.shape[axis]:
        raise ValueError(
            "Shapes are not compatible for vector dot product along the specified axis"
        )

    return reduce(
        ffuncs.add,
        multiply(conj(x1), x2),
        axis=axis,
    )


@dataclass(frozen=True, eq=False)
class FillTensorFType(TensorFType):
    _fill_value: Any
    _element_type: FType
    _shape_type: tuple

    @property
    def fill_value(self):
        return self._fill_value

    @property
    def element_type(self) -> FType:
        return self._element_type

    @property
    def shape_type(self):
        return self._shape_type

    def from_numpy(self, arr):
        raise NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, FillTensorFType):
            return False
        return (
            ffuncs.same(self._fill_value, other._fill_value)
            and self._element_type == other._element_type
            and self._shape_type == other._shape_type
        )

    def __hash__(self):
        return hash(
            (ffuncs.samehash(self._fill_value), self._element_type, self._shape_type)
        )

    def construct(self, shape: tuple) -> FillTensor:
        return FillTensor(shape, self.fill_value)

    def __call__(self, val: Any) -> FillTensor:
        """
        Convert a tensor to this fill tensor type.

        Args:
            val: A tensor to convert to this type.
        Returns:
            A FillTensor instance of this type.
        """
        raise NotImplementedError(
            f"Tensor conversion not yet implemented for {type(self).__name__}"
        )


class FillTensor(Tensor):
    """
    A tensor that has a specific shape but contains no actual data.
    Used primarily for broadcasting operations where a tensor of a specific
    shape is needed but the values are irrelevant.
    """

    def __init__(self, shape, fill_value):
        self._shape = shape
        self._fill_value = fill_value

    def __getitem__(self, idxs):
        return Scalar(self._fill_value, fill_value=self._fill_value)

    def item(self):
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to Python scalar.")
        return self._fill_value

    @property
    def shape(self):
        return self._shape

    @property
    def fill_value(self) -> Any:
        """Default fill value."""
        return self.ftype.fill_value

    @property
    def element_type(self) -> FType:
        """Data type of the tensor's elements."""
        return self.ftype.element_type

    @property
    def shape_type(self) -> tuple[FType, ...]:
        """Shape type of the tensor."""
        return self.ftype.shape_type

    @property
    def ftype(self):
        return FillTensorFType(
            self._fill_value,
            ftype(self._fill_value),
            tuple(ftype(dim) for dim in self.shape),
        )


def broadcast_to(tensor, /, shape: tuple) -> LazyTensor:
    """
    Broadcasts a lazy tensor to a specified shape.

    Args:
        tensor: The lazy tensor to broadcast.
        shape: The target shape to broadcast to.

    Returns:
        A new lazy tensor with the specified shape.
    """
    tensor = lazy(tensor)

    if not is_broadcastable_directional(tensor.shape, shape):
        raise ValueError(
            f"Tensor with shape {tensor.shape} is not broadcastable "
            f"to the shape {shape}"
        )
    return elementwise(ffuncs.first_arg, tensor, FillTensor(shape, np.False_))


def broadcast_arrays(*arrays: LazyTensor) -> tuple[LazyTensor, ...]:
    """
    Broadcasts one or more arrays against one another.
    """
    shape = _broadcast_shape(*(array.shape for array in arrays))
    return tuple(broadcast_to(arr, shape) for arr in arrays)


def moveaxis(x, source: int | tuple[int, ...], destination: int | tuple[int, ...], /):
    """
    Moves axes of an array to new positions.
    """
    x = lazy(x)

    # argument validation
    # handles uniqueness, int -> tuple, and bound check
    source = normalize_axis_tuple(source, x.ndim, "source")
    destination = normalize_axis_tuple(destination, x.ndim, "destination")

    if len(source) != len(destination):
        raise ValueError("Source and Destination indices must have the same length")

    final_order = [i for i in range(x.ndim) if i not in source]
    for dest, src in sorted(zip(destination, source, strict=True)):
        final_order.insert(dest, src)

    return permute_dims(x, axes=tuple(final_order))


def sin(x) -> LazyTensor:
    return elementwise(ffuncs.sin, lazy(x))


def sinh(x) -> LazyTensor:
    return elementwise(ffuncs.sinh, lazy(x))


def cos(x) -> LazyTensor:
    return elementwise(ffuncs.cos, lazy(x))


def cosh(x) -> LazyTensor:
    return elementwise(ffuncs.cosh, lazy(x))


def tan(x) -> LazyTensor:
    return elementwise(ffuncs.tan, lazy(x))


def tanh(x) -> LazyTensor:
    return elementwise(ffuncs.tanh, lazy(x))


def asin(x) -> LazyTensor:
    return elementwise(ffuncs.asin, lazy(x))


def asinh(x) -> LazyTensor:
    return elementwise(ffuncs.asinh, lazy(x))


def acos(x) -> LazyTensor:
    return elementwise(ffuncs.acos, lazy(x))


def acosh(x) -> LazyTensor:
    return elementwise(ffuncs.acosh, lazy(x))


def atan(x) -> LazyTensor:
    return elementwise(ffuncs.atan, lazy(x))


def hypot(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.hypot, lazy(x1), lazy(x2))


def atanh(x) -> LazyTensor:
    return elementwise(ffuncs.atanh, lazy(x))


def atan2(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.atan2, lazy(x1), lazy(x2))


def exp(x) -> LazyTensor:
    return elementwise(ffuncs.exp, lazy(x))


def expm1(x) -> LazyTensor:
    return elementwise(ffuncs.expm1, lazy(x))


def log(x) -> LazyTensor:
    return elementwise(ffuncs.log, lazy(x))


def log1p(x) -> LazyTensor:
    return elementwise(ffuncs.log1p, lazy(x))


def log2(x) -> LazyTensor:
    return elementwise(ffuncs.log2, lazy(x))


def log10(x) -> LazyTensor:
    return elementwise(ffuncs.log10, lazy(x))


def logaddexp(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.logaddexp, lazy(x1), lazy(x2))


def signbit(x) -> LazyTensor:
    return elementwise(ffuncs.signbit, lazy(x))


def copysign(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.copysign, lazy(x1), lazy(x2))


def nextafter(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.nextafter, lazy(x1), lazy(x2))


def isfinite(x) -> LazyTensor:
    return elementwise(ffuncs.isfinite, lazy(x))


def isinf(x) -> LazyTensor:
    return elementwise(ffuncs.isinf, lazy(x))


def isnan(x) -> LazyTensor:
    return elementwise(ffuncs.isnan, lazy(x))


def iscomplexobj(x) -> LazyTensor:
    return elementwise(ffuncs.iscomplexobj, lazy(x))


def logical_and(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.logical_and, lazy(x1), lazy(x2))


def logical_or(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.logical_or, lazy(x1), lazy(x2))


def logical_xor(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.logical_xor, lazy(x1), lazy(x2))


def logical_not(x) -> LazyTensor:
    return elementwise(ffuncs.logical_not, lazy(x))


def less(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.less, lazy(x1), lazy(x2))


def less_equal(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.less_equal, lazy(x1), lazy(x2))


def greater(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.greater, lazy(x1), lazy(x2))


def greater_equal(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.greater_equal, lazy(x1), lazy(x2))


def equal(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.equal, lazy(x1), lazy(x2))


def same(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.same, lazy(x1), lazy(x2))


def not_equal(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.not_equal, lazy(x1), lazy(x2))


def not_same(x1, x2) -> LazyTensor:
    return elementwise(ffuncs.not_same, lazy(x1), lazy(x2))


def where(condition, x1, x2) -> LazyTensor:
    return elementwise(ffuncs.where, lazy(condition), lazy(x1), lazy(x2))


def mean(x, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False):
    """
    Calculates the arithmetic mean of the input array ``x``.
    """
    x = lazy(x)
    n = (
        np.prod(tuple(x.shape[i] for i in range(x.ndim) if i in axis))
        if isinstance(axis, tuple)
        else (np.prod(x.shape) if axis is None else x.shape[axis])
    )
    s = sum(x, axis=axis, keepdims=keepdims)
    return truediv(s, x.element_type(n))


def var(
    x,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: float = 0.0,
    keepdims: bool = False,
):
    """
    Calculates the variance of the input array ``x``.
    """
    x = lazy(x)
    n = (
        np.prod(tuple(x.shape[i] for i in range(x.ndim) if i in axis))
        if isinstance(axis, tuple)
        else (np.prod(x.shape) if axis is None else x.shape[axis])
    )
    m = mean(x, axis=axis, keepdims=True)
    v = sum(pow(x - m, x.element_type(2.0)), axis=axis, keepdims=keepdims)
    return truediv(v, x.element_type(n - correction))


def std(
    x,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: float = 0.0,
    keepdims: bool = False,
):
    """
    Calculates the standard deviation of the input array ``x``.
    """
    x = lazy(x)
    d = var(x, axis=axis, correction=correction, keepdims=keepdims)
    return pow(d, 0.5)


def einop(prgm, **kwargs):
    stmt = ein.parse_einop(prgm)
    prgm = ein.Plan((stmt, ein.Produces((stmt.tns,))))
    xp = sys.modules[__name__]
    ctx = ein.EinsumInterpreter(xp)
    bindings = {ein.Alias(k): v for k, v in kwargs.items()}
    return ctx(prgm, bindings)[0]


def einsum(prgm, *args, **kwargs):
    stmt, bindings = ein.parse_einsum(prgm, *args)
    prgm = ein.Plan((stmt, ein.Produces((stmt.tns,))))
    xp = sys.modules[__name__]
    ctx = ein.EinsumInterpreter(xp)
    return ctx(prgm, bindings)[0]


def get_lazy_tensor_stats(
    lazy_tensor: LazyTensor,
    stats_factory: StatsFactory,
) -> TensorStats:
    trace = lazy_tensor.ctx.trace()
    interpreter = StatsInterpreter(stats_factory=stats_factory)
    bindings: OrderedDict[Alias, TensorStats] = OrderedDict()
    last_stats: TensorStats | tuple[TensorStats, ...]
    for stmt in trace:
        last_stats = interpreter(stmt, bindings)

    if last_stats is None:
        raise ValueError("Trace was empty or no stats produced")
    if isinstance(last_stats, tuple):
        return last_stats[0]

    return last_stats


def outer(x1, x2) -> LazyTensor:
    x1 = lazy(x1)
    x2 = lazy(x2)

    if x1.ndim != 1:
        raise ValueError(f"x1 must be a 1D array, got {x1.ndim}D array")
    if x2.ndim != 1:
        raise ValueError(f"x2 must be a 1D array, got {x2.ndim}D array")

    i = Field(gensym("i"))
    j = Field(gensym("j"))
    expr = Reorder(
        MapJoin(
            Literal(ffuncs.mul),
            (Table(x1.data, (i,)), Table(x2.data, (j,))),
        ),
        (i, j),
    )
    ctx = x1.ctx.join(x2.ctx)
    data, ctx = ctx.eval(expr)
    return LazyTensor(
        data,
        ctx,
        (x1.shape[0], x2.shape[0]),
        ffuncs.mul(x1.fill_value, x2.fill_value),
        return_type(ffuncs.mul, x1.element_type, x2.element_type),
    )
