import bisect
import builtins
import operator
import sys
from collections.abc import Callable, Sequence
from itertools import accumulate, zip_longest
from typing import Any

import numpy as np
from numpy.lib.array_utils import normalize_axis_index, normalize_axis_tuple

from ..algebra import (
    TensorFormat,
    element_type,
    fill_value,
    first,
    fixpoint_type,
    identity,
    init_value,
    promote_max,
    promote_min,
    promote_type,
    query_property,
    register_property,
    return_type,
)
from ..algebra import conjugate as conj
from ..finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    LogicNode,
    MapJoin,
    Relabel,
    Reorder,
    Subquery,
    Table,
)
from ..symbolic import gensym
from .overrides import OverrideTensor


def identify(data):
    lhs = Alias(gensym("A"))
    return Subquery(lhs, data)


class LazyTensorFormat(TensorFormat):
    _fill_value: Any
    _element_type: Any
    _shape_type: Any

    def __init__(self, _fill_value: Any, _element_type: Any, _shape_type: tuple):
        self._fill_value = _fill_value
        self._element_type = _element_type
        self._shape_type = _shape_type

    def __eq__(self, other):
        if not isinstance(other, LazyTensorFormat):
            return False
        return (
            self._fill_value == other._fill_value
            and self._element_type == other._element_type
            and self._shape_type == other._shape_type
        )

    def __hash__(self):
        return hash((self._fill_value, self._element_type, self._shape_type))

    @property
    def fill_value(self):
        return self._fill_value

    @property
    def element_type(self):
        return self._element_type

    @property
    def shape_type(self):
        return self._shape_type


class LazyTensor(OverrideTensor):
    def __init__(
        self, data: LogicNode, shape: tuple, fill_value: Any, element_type: Any
    ):
        self.data = data
        self._shape = shape
        self._fill_value = fill_value
        self._element_type = element_type

    @property
    def format(self):
        return LazyTensorFormat(
            _fill_value=self._fill_value,
            _element_type=self._element_type,
            _shape_type=tuple(type(dim) for dim in self.shape),
        )

    @property
    def shape(self) -> tuple:
        """
        Returns the shape of the LazyTensor as a tuple.
        The shape is determined by the data and is a static property.
        """
        return self._shape

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

    def __invert__(self):
        return bitwise_inverse(self)

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


register_property(np.ndarray, "asarray", "__attr__", lambda x: x)
register_property(LazyTensor, "asarray", "__attr__", lambda x: x)


def asarray(arg: Any) -> Any:
    """Convert given argument and return np.asarray(arg) for the scalar type input.
    If input argument is already array type, return unchanged.

    Args:
        arg: The object to be converted.

    Returns:
        The array type result of the given object.
    """
    if hasattr(arg, "asarray"):
        return arg.asarray()

    try:
        return query_property(arg, "asarray", "__attr__")
    except AttributeError:
        return np.asarray(arg)


def defer(arr) -> LazyTensor:
    """
    - defer(arr) -> LazyTensor:
    Converts an array into a LazyTensor. If the input is already a LazyTensor, it is
    returned as-is.
    Otherwise, it creates a LazyTensor representation of the input array.

    Parameters:
    - arr: The input array to be converted into a LazyTensor.

    Returns:
    - LazyTensor: A lazy representation of the input array.
    """
    if isinstance(arr, LazyTensor):
        return arr
    arr = asarray(arr)
    name = Alias(gensym("A"))
    idxs = tuple(Field(gensym("i")) for _ in range(arr.ndim))
    shape = tuple(arr.shape)
    tns = Subquery(name, Table(Literal(arr), idxs))
    return LazyTensor(tns, shape, fill_value(arr), element_type(arr))


def permute_dims(arg, /, axis: tuple[int, ...]) -> LazyTensor:
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
    arg = defer(arg)
    axis = normalize_axis_tuple(axis, arg.ndim + len(axis))
    idxs = tuple(Field(gensym("i")) for _ in range(arg.ndim))
    return LazyTensor(
        Reorder(Relabel(arg.data, idxs), tuple(idxs[i] for i in axis)),
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
    x = defer(x)
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
    x = defer(x)
    if init is None:
        init = init_value(op, x.element_type)
    if axis is None:
        axis = tuple(range(x.ndim))
    axis = normalize_axis_tuple(axis, x.ndim)
    if axis is None or isinstance(axis, int):
        raise ValueError("axis must be a tuple")

    shape = tuple(x.shape[n] for n in range(x.ndim) if n not in axis)
    fields = tuple(Field(gensym("i")) for _ in range(x.ndim))
    data: LogicNode = Aggregate(
        Literal(op),
        Literal(init),
        Relabel(x.data, fields),
        tuple(fields[i] for i in axis),
    )
    if keepdims:
        keeps = tuple(
            fields[i] if i in axis else Field(gensym("j")) for i in range(x.ndim)
        )
        data = Reorder(data, keeps)
        shape = tuple(shape[i] if i in axis else 1 for i in range(x.ndim))
    if dtype is None:
        dtype = fixpoint_type(op, init, x.element_type)
    return LazyTensor(identify(data), shape, init, dtype)


def _broadcast_shape(*args: tuple[int, ...]) -> tuple[int, ...]:
    """
    Computes the broadcasted shape for the given LazyTensor arguments,
    following array_api broadcasting rules.
    Raises ValueError if shapes are not broadcastable.

    Parameters:
    --------------
    - *args: Variable number of tuples representing shapes of LazyTensors.

    Returns:
    --------------
    tuple[int, ...]: The broadcasted shape as a tuple of integers.
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
    args = tuple(defer(a) for a in args)
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
        bargs.append(Reorder(Relabel(arg.data, tuple(idims)), tuple(odims)))
    data = Reorder(MapJoin(Literal(f), tuple(bargs)), idxs)
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
    x = defer(x)
    return reduce(
        operator.or_,
        elementwise(operator.truth, x),
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
    x = defer(x)
    return reduce(
        operator.and_,
        elementwise(operator.truth, x),
        axis=axis,
        keepdims=keepdims,
        init=init,
    )


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
    x = defer(x)
    return reduce(promote_min, x, axis=axis, keepdims=keepdims, init=init)


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
    x = defer(x)
    return reduce(promote_max, x, axis=axis, keepdims=keepdims, init=init)


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
            operator.add,
            multiply(expand_dims(a, axis=-1), expand_dims(b, axis=-3)),
            axis=-2,
        )

    x1 = defer(x1)
    x2 = defer(x2)

    if x1.ndim < 1 or x2.ndim < 1:
        raise ValueError("Both inputs must be at least 1D arrays")

    if x1.ndim == 1 and x2.ndim == 1:
        return reduce(operator.add, multiply(x1, x2), axis=0)

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
    x = defer(x)
    if x.ndim < 2:
        # this is following numpy's behavior.
        # data-apis specification assumes that input is atleast 2D
        raise ValueError(
            "Input tensor must have at least 2 dimensions for transposition"
        )
    # swap the last two axes
    return permute_dims(x, axis=(*range(x.ndim - 2), x.ndim - 1, x.ndim - 2))


def bitwise_inverse(x) -> LazyTensor:
    return elementwise(operator.invert, defer(x))


def bitwise_and(x1, x2) -> LazyTensor:
    return elementwise(operator.and_, defer(x1), defer(x2))


def bitwise_left_shift(x1, x2) -> LazyTensor:
    return elementwise(operator.lshift, defer(x1), defer(x2))


def bitwise_or(x1, x2) -> LazyTensor:
    return elementwise(operator.or_, defer(x1), defer(x2))


def bitwise_right_shift(x1, x2) -> LazyTensor:
    return elementwise(operator.rshift, defer(x1), defer(x2))


def bitwise_xor(x1, x2) -> LazyTensor:
    return elementwise(operator.xor, defer(x1), defer(x2))


def truediv(x1, x2) -> LazyTensor:
    return elementwise(operator.truediv, defer(x1), defer(x2))


def floordiv(x1, x2) -> LazyTensor:
    return elementwise(operator.floordiv, defer(x1), defer(x2))


def mod(x1, x2) -> LazyTensor:
    return elementwise(operator.mod, defer(x1), defer(x2))


def pow(x1, x2) -> LazyTensor:
    return elementwise(operator.pow, defer(x1), defer(x2))


def conjugate(x) -> LazyTensor:
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
    return elementwise(conj, defer(x))


def tensordot(
    x1, x2, /, *, axes: int | tuple[Sequence[int], Sequence[int]]
) -> LazyTensor:
    """
    Computes the tensordot operation.

    Returns a LazyTensor if either x1 or x2 is a LazyTensor.
    Otherwise, computes the result eagerly.
    """
    x1 = defer(x1)
    x2 = defer(x2)

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
    x1 = defer(x1)
    x2 = defer(x2)

    # check broadcastability
    if not is_broadcastable(x1.shape, x2.shape):
        raise ValueError("Shapes are not broadcastable for vector dot product")

    # check if dims of axis are the same
    if x1.shape[axis] != x2.shape[axis]:
        raise ValueError(
            "Shapes are not compatible for vector dot product along the specified axis"
        )

    return reduce(
        operator.add,
        multiply(conjugate(x1), x2),
        axis=axis,
    )


# Manipulation functions
class NoneArray:
    def __init__(self, shape):
        self.shape = shape
        self.fill_value = None
        self.element_type = None
        self.dtype = None
        self.ndim = len(shape)

    def __getitem__(self, idxs):
        return None


register_property(NoneArray, "asarray", "__attr__", lambda x: x)


def broadcast_to(tensor, /, shape: tuple[int, ...]) -> LazyTensor:
    """
    Broadcasts a lazy tensor to a specified shape.

    Args:
        tensor: The lazy tensor to broadcast.
        shape: The target shape to broadcast to.

    Returns:
        A new lazy tensor with the specified shape.
    """
    tensor = defer(tensor)

    if not is_broadcastable_directional(tensor.shape, shape):
        raise ValueError(
            f"Tensor with shape {tensor.shape} is not broadcastable "
            f"to the shape {shape}"
        )
    return elementwise(first, tensor, NoneArray(shape))


def broadcast_arrays(*arrays: LazyTensor) -> tuple[LazyTensor, ...]:
    """
    Broadcasts one or more arrays against one another.
    """
    shape = _broadcast_shape(*(array.shape for array in arrays))
    return tuple(broadcast_to(arr, shape) for arr in arrays)


class ConcatTensor:
    """
    Tensor representing a concatenation of multiple tensors along a specified axis.
    """

    def __init__(self, tensor, *tensors, axis: int = 0):
        """
        Args:
            tensor (ArrayLike):
                The first tensor.
            *tensors (ArrayLike):
                Tensors to concatenate with the first tensor.
        All tensors must support `__getitem__` and have a `shape` attribute.
        `fill_value` is taken from the first tensor.
        `element_type` is casted according to array_api specification.
        """
        self.ndim = len(tensor.shape)

        shape_without_axis = tensor.shape[:axis] + tensor.shape[axis + 1 :]
        # keep track of partial sums of sizes along the concatenation axis
        self.ps_sizes = [0, tensor.shape[axis]]
        for t in tensors:
            if t.ndim != self.ndim:
                raise ValueError("All tensors must have same number of dimensions")
            self.ps_sizes.append(self.ps_sizes[-1] + t.shape[axis])
            if t.shape[:axis] + t.shape[axis + 1 :] != shape_without_axis:
                raise ValueError(
                    "All tensors must have the same shape except "
                    "along the concatenation axis"
                )
        self.shape = (
            tensor.shape[:axis] + (self.ps_sizes[-1],) + tensor.shape[axis + 1 :]
        )
        self.tensors = (tensor,) + tensors
        self.concat_axis = axis
        # find the appropriate element type
        self.element_type = element_type(tensor)
        for t in tensors:
            self.element_type = promote_type(element_type(self), element_type(t))

    def __getitem__(self, idxs: tuple[int, ...]):
        """
        Args:
            idxs: tuple[int, ...]
                Indices to access the concatenated tensor.
        Returns the element at the specified indices.
        """
        # find the tensor to access
        tn = bisect.bisect(self.ps_sizes, idxs[self.concat_axis]) - 1
        if tn < 0 or tn >= len(self.tensors):
            raise IndexError(f"Index {idxs} out of bounds for shape {self.shape}")
        t = self.tensors[tn]
        shifted_idx = idxs[self.concat_axis] - self.ps_sizes[tn]
        result = t[
            idxs[: self.concat_axis] + (shifted_idx,) + idxs[self.concat_axis + 1 :]
        ]
        return self.element_type(result)

    @property
    def fill_value(self):
        """
        Returns the fill value of the first tensor in the concatenation.
        """
        return fill_value(self.tensors[0])

    def asarray(self):
        """
        Returns the concatenated tensor as a numpy array.
        """
        return self

    @property
    def dtype(self):
        """
        Returns the data type of the concatenated tensor.
        For testing reasons and compatibility with numpy.
        """
        return self.element_type


def concat(arrays: tuple | list, /, axis: int | None = 0) -> LazyTensor:
    """
    Concatenates input tensors along the specified axis.

    Parameters:
        arrays: Sequence of tensors to concatenate (tuple or list of LazyTensor)
        axis: Axis along which to concatenate (default=0)

    Returns:
        Concatenated tensor as LazyTensor
    """
    arrays = [defer(arr) for arr in arrays]

    if axis is None:
        # flatten all tensors
        for i, t in enumerate(arrays):
            arrays[i] = flatten(t)
        axis = 0

    # Convert axis to positive index and validate
    ndim = arrays[0].ndim
    axis = normalize_axis_index(axis, ndim)
    from finch import compute

    computed_arrays = tuple(compute(arr) for arr in arrays)
    concat_tensor = ConcatTensor(*computed_arrays, axis=axis)
    # Create a LazyTensor that represents the concatenation
    return elementwise(identity, defer(concat_tensor))


def flatten(x):
    """
    Flattens the input tensor `x` into a 1D tensor along the specified axis.
    Parameters
    ----------
    x: LazyTensor
        The input tensor to be flattened.
    Returns
    -------
    LazyTensor
        A new LazyTensor that is a flattened version of `x`.
    """
    # TODO: This is a stub implementation.
    from finch import compute

    x = defer(x)
    if x.ndim == 0:
        # If x is a scalar, return it as is
        return x
    computed = compute(x)
    if isinstance(computed, OverrideTensor):
        # If the computed tensor is an OverrideTensor, we need to flatten it
        # using its underlying data.
        for possible_name in ["data", "value", "array"]:
            if hasattr(computed, possible_name):
                computed = getattr(computed, possible_name)
                break

    assert isinstance(computed, np.ndarray), "Flatten currently supports numpy arrays"
    # Flatten the computed array
    flattened = computed.flatten()
    return defer(flattened)


def moveaxis(x, source: int | tuple[int, ...], destination: int | tuple[int, ...], /):
    """
    Moves axes of an array to new positions.
    """
    # argument validation
    # handles uniqueness, int -> tuple, and bound check
    source = normalize_axis_tuple(source, x.ndim, "source")
    destination = normalize_axis_tuple(destination, x.ndim, "destination")

    if len(source) != len(destination):
        raise ValueError("Source and Destination indices must have the same length")

    x = defer(x)

    final_order = [i for i in range(x.ndim) if i not in source]
    for dest, src in sorted(zip(destination, source, strict=True)):
        final_order.insert(dest, src)

    return permute_dims(x, axis=tuple(final_order))


def stack(arrays, /, axis: int = 0) -> LazyTensor:
    """
    Stacks input tensors along a new axis.

    Parameters:
        arrays: Sequence of tensors to stack (tuple or list of LazyTensor)
        axis: Axis along which to stack (default=0)
    Returns:
        Stacked tensor as LazyTensor
    """
    if not isinstance(arrays, tuple | list):
        raise TypeError("arrays must be a tuple or list of LazyTensors")
    arrays = [defer(arr) for arr in arrays]
    # add 1-dim at the axis position for stacking
    arrays = tuple(expand_dims(x, axis=axis) for x in arrays)
    # concat, this will also do the shape verification
    return concat(arrays, axis=axis)


def sin(x) -> LazyTensor:
    return elementwise(np.sin, defer(x))


def sinh(x) -> LazyTensor:
    return elementwise(np.sinh, defer(x))


def cos(x) -> LazyTensor:
    return elementwise(np.cos, defer(x))


def cosh(x) -> LazyTensor:
    return elementwise(np.cosh, defer(x))


def tan(x) -> LazyTensor:
    return elementwise(np.tan, defer(x))


def tanh(x) -> LazyTensor:
    return elementwise(np.tanh, defer(x))


def asin(x) -> LazyTensor:
    return elementwise(np.asin, defer(x))


def asinh(x) -> LazyTensor:
    return elementwise(np.asinh, defer(x))


def acos(x) -> LazyTensor:
    return elementwise(np.acos, defer(x))


def acosh(x) -> LazyTensor:
    return elementwise(np.acosh, defer(x))


def atan(x) -> LazyTensor:
    return elementwise(np.atan, defer(x))


def atanh(x) -> LazyTensor:
    return elementwise(np.atanh, defer(x))


def atan2(x1, x2) -> LazyTensor:
    return elementwise(np.atan2, defer(x1), defer(x2))
