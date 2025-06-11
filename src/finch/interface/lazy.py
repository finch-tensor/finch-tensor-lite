import builtins
import operator
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import accumulate, zip_longest
from typing import Any

from numpy.core.numeric import normalize_axis_tuple

from ..algebra import conjugate as conj
from ..algebra import (
    element_type,
    fill_value,
    fixpoint_type,
    init_value,
    promote_max,
    promote_min,
    return_type,
)
from ..finch_logic import (
    Aggregate,
    Alias,
    Field,
    Immediate,
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


@dataclass
class LazyTensor(OverrideTensor):
    data: LogicNode
    shape: tuple
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

    def __and__(self, other):
        return bitwise_and(self, defer(other))

    def __rand__(self, other):
        return bitwise_and(defer(other), self)

    def __lshift__(self, other):
        return bitwise_left_shift(self, defer(other))

    def __rlshift__(self, other):
        return bitwise_left_shift(defer(other), self)

    def __or__(self, other):
        return bitwise_or(self, defer(other))

    def __ror__(self, other):
        return bitwise_or(defer(other), self)

    def __rshift__(self, other):
        return bitwise_right_shift(self, defer(other))

    def __rrshift__(self, other):
        return bitwise_right_shift(defer(other), self)

    def __xor__(self, other):
        return bitwise_xor(self, defer(other))

    def __rxor__(self, other):
        return bitwise_xor(defer(other), self)

    def __truediv__(self, other):
        return truediv(self, defer(other))

    def __rtruediv__(self, other):
        return truediv(defer(other), self)

    def __floordiv__(self, other):
        return floordiv(self, defer(other))

    def __rfloordiv__(self, other):
        return floordiv(defer(other), self)

    def __mod__(self, other):
        return mod(self, defer(other))

    def __rmod__(self, other):
        return mod(defer(other), self)

    def __pow__(self, other):
        return pow(self, defer(other))

    def __rpow__(self, other):
        return pow(defer(other), self)

    def __matmul__(self, other):
        return matmul(self, defer(other))

    def __rmatmul__(self, other):
        return matmul(defer(other), self)

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
    name = Alias(gensym("A"))
    idxs = tuple(Field(gensym("i")) for _ in range(arr.ndim))
    shape = tuple(arr.shape)
    tns = Subquery(name, Table(Immediate(arr), idxs))
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
        Immediate(op),
        Immediate(init),
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
    data = Reorder(MapJoin(Immediate(f), tuple(bargs)), idxs)
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
            raise ValueError("axes must be non-negative")
        axes_a = list(range(x1.ndim - N, x1.ndim))
        axes_b = list(range(N))
    else:
        axes_a, axes_b = (list(ax) for ax in axes)
        axes_a = [axes_a] if isinstance(axes_a, int) else list(axes_a)
        axes_b = [axes_b] if isinstance(axes_b, int) else list(axes_b)

    # Normalize negative axes
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
