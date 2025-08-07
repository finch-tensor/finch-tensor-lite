import operator
import warnings

import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_equal

import finch


# Utility function to generate random complex numpy tensors
def random_array(shape, dtype=np.complex128, rng: np.random.Generator | None = None):
    """Generates a random complex array. Uses integers for both real
    and imaginary parts to avoid floating-point issues in tests.

    Args:
        - shape: A tuple specifying the shape of the array.
        - dtype: The intended dtype of the randomly generated array.
        If nothing is provided, np.complex128 array is generated
        - rng: The random number generator to use. Providing one is strongly recommended
        for reproducibility. If no generator is provided, a new one is instantiated with
        random seed.

    Returns:
        A NumPy array of complex numbers with the given shape.
    """
    if rng is None:
        rng = np.random.default_rng()
    if dtype is bool:
        arr = rng.integers(0, 2, size=shape).astype(dtype)
    elif np.issubdtype(dtype, np.integer):
        arr = rng.integers(-100, 100, size=shape).astype(dtype)
    elif np.issubdtype(dtype, complex):
        real = rng.random(size=shape).astype(np.float32)
        imag = rng.random(size=shape).astype(np.float32)
        arr = (real + 1j * imag).astype(dtype)
    else:
        arr = rng.random(size=shape).astype(dtype)
    return arr


@pytest.mark.parametrize(
    "a, b",
    [
        (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])),
        (np.array([[2, 0], [1, 3]]), np.array([[4, 1], [2, 2]])),
    ],
)
def test_matrix_multiplication(a, b):
    result = finch.fuse(
        lambda a, b: finch.reduce(
            operator.add, finch.multiply(finch.expand_dims(a, 2), b), axis=1
        ),
        a,
        b,
    )

    expected = np.matmul(a, b)

    assert_equal(result, expected)


class TestEagerTensorFType(finch.TensorFType):
    # This class doesn't define any pytests
    __test__ = False

    def __init__(self, fmt):
        self.fmt = fmt

    def __eq__(self, other):
        if not isinstance(other, TestEagerTensorFType):
            return False
        return self.fmt == other.fmt

    def __hash__(self):
        return hash(self.fmt)

    @property
    def fill_value(self):
        return finch.fill_value(self.fmt)

    @property
    def element_type(self):
        return finch.element_type(self.fmt)

    @property
    def shape_type(self):
        return finch.shape_type(self.fmt)


class TestEagerTensor(finch.EagerTensor):
    # This class doesn't define any pytests
    __test__ = False

    def __init__(self, array):
        self.array = np.array(array)

    def __repr__(self):
        return f"TestEagerTensor({self.array})"

    def __getitem__(self, item):
        return self.array[item]

    @property
    def shape(self):
        return self.array.shape

    @property
    def ftype(self):
        return TestEagerTensorFType(finch.ftype(self.array))

    def to_numpy(self):
        return self.array


@pytest.mark.parametrize(
    "a, b",
    [
        (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])),
        (np.array([[2, 0], [1, 3]]), np.array([[4, 1], [2, 2]])),
        (np.array([[2, 0], [1, 3]]), 2),
        (3, np.array([[2, 0], [1, 3]])),
        (np.array([[2, 0], [1, 3]]), True),
    ],
)
@pytest.mark.parametrize(
    "a_wrap",
    [
        lambda x: x,
        TestEagerTensor,
        finch.defer,
    ],
)
@pytest.mark.parametrize(
    "b_wrap",
    [
        lambda x: x,
        TestEagerTensor,
        finch.defer,
    ],
)
@pytest.mark.parametrize(
    "ops, np_op",
    [
        ((operator.add, finch.add, np.add), np.add),
        ((operator.sub, finch.subtract, np.subtract), np.subtract),
        ((operator.mul, finch.multiply, np.multiply), np.multiply),
        ((operator.and_, finch.bitwise_and, np.bitwise_and), np.bitwise_and),
        ((operator.or_, finch.bitwise_or, np.bitwise_or), np.bitwise_or),
        ((operator.xor, finch.bitwise_xor, np.bitwise_xor), np.bitwise_xor),
        (
            (operator.lshift, finch.bitwise_left_shift, np.bitwise_left_shift),
            np.bitwise_left_shift,
        ),
        (
            (operator.rshift, finch.bitwise_right_shift, np.bitwise_right_shift),
            np.bitwise_right_shift,
        ),
        ((operator.truediv, finch.truediv, np.true_divide), np.true_divide),
        ((operator.floordiv, finch.floordiv, np.floor_divide), np.floor_divide),
        ((operator.mod, finch.mod, np.mod), np.mod),
        ((operator.pow, finch.pow, np.pow), np.pow),
        ((finch.atan2, np.atan2), np.atan2),
        ((finch.logaddexp, np.logaddexp), np.logaddexp),
        ((finch.logical_and, np.logical_and), np.logical_and),
        ((finch.logical_or, np.logical_or), np.logical_or),
        ((finch.logical_xor, np.logical_xor), np.logical_xor),
    ],
)
def test_elementwise_operations(a, b, a_wrap, b_wrap, ops, np_op):
    wa = a_wrap(a)
    wb = b_wrap(b)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="divide by zero encountered in",
        )

        expected = np_op(a, b)

        for op in ops:
            result = op(wa, wb)

            if isinstance(wa, finch.LazyTensor) or isinstance(wb, finch.LazyTensor):
                assert isinstance(result, finch.LazyTensor)

                result = finch.compute(result)

            assert_equal(result, expected)


@pytest.mark.parametrize(
    "a",
    [
        (np.array([[1, 2], [3, 4]])),
        (np.array([[2, 0], [1, 3]])),
    ],
)
@pytest.mark.parametrize(
    "a_wrap",
    [
        lambda x: x,
        TestEagerTensor,
        finch.defer,
    ],
)
@pytest.mark.parametrize(
    "ops, np_op",
    [
        ((operator.abs, finch.abs, np.abs), np.abs),
        ((operator.pos, finch.positive, np.positive), np.positive),
        ((operator.neg, finch.negative, np.negative), np.negative),
        (
            (operator.invert, finch.bitwise_inverse, np.bitwise_invert),
            np.bitwise_invert,
        ),
        ((finch.sin, np.sin), np.sin),
        ((finch.sinh, np.sinh), np.sinh),
        ((finch.cos, np.cos), np.cos),
        ((finch.cosh, np.cosh), np.cosh),
        ((finch.tan, np.tan), np.tan),
        ((finch.tanh, np.tanh), np.tanh),
        ((finch.asin, np.asin), np.asin),
        ((finch.asinh, np.asinh), np.asinh),
        ((finch.acos, np.acos), np.acos),
        ((finch.acosh, np.acosh), np.acosh),
        ((finch.atan, np.atan), np.atan),
        ((finch.atanh, np.atanh), np.atanh),
        ((finch.log, np.log), np.log),
        ((finch.log1p, np.log1p), np.log1p),
        ((finch.log2, np.log2), np.log2),
        ((finch.log10, np.log10), np.log10),
        ((finch.logical_not, np.logical_not), np.logical_not),
    ],
)
def test_unary_operations(a, a_wrap, ops, np_op):
    wa = a_wrap(a)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="divide by zero encountered in",
        )

        expected = np_op(a)

        for op in ops:
            result = op(wa)

            if isinstance(wa, finch.LazyTensor):
                assert isinstance(result, finch.LazyTensor)

                result = finch.compute(result)

            assert_equal(result, expected)


@pytest.mark.parametrize(
    "a",
    [
        (np.array([[True, False, True, False], [False, False, False, False]])),
        (np.array([[1, 2], [3, 4]])),
        (np.array([[2, 0], [1, 3]])),
        (np.array([[1.00002, -12.618, 0, 0.001], [-1.414, -5.01, 0, 0]])),
        (np.array([[0, 0.618, 0, 0.001], [0, 0.01, 0, 0]])),
        (np.array([[10000.0, 1.0, -89.0, 78], [401.0, 3, 5, 10.2]])),
    ],
)
@pytest.mark.parametrize(
    "a_wrap",
    [
        lambda x: x,
        TestEagerTensor,
        finch.defer,
    ],
)
@pytest.mark.parametrize(
    "ops, np_op",
    [
        ((finch.prod, np.prod), np.prod),
        ((finch.sum, np.sum), np.sum),
        ((finch.any, np.any), np.any),
        ((finch.all, np.all), np.all),
        ((finch.min, np.min), np.min),
        ((finch.max, np.max), np.max),
        ((finch.mean, np.mean), np.mean),
        ((finch.std, np.std), np.std),
        ((finch.var, np.var), np.var),
    ],
)
@pytest.mark.parametrize(
    "axis",
    [
        None,
        0,
        1,
        (0, 1),
    ],
)
def test_reduction_operations(a, a_wrap, ops, np_op, axis):
    wa = a_wrap(a)

    expected = np_op(a, axis=axis)

    for op in ops:
        result = op(wa, axis=axis)

        if isinstance(wa, finch.LazyTensor):
            assert isinstance(result, finch.LazyTensor)

            result = finch.compute(result)

        if np.issubdtype(expected.dtype, np.floating) or np.issubdtype(
            expected.dtype, np.complexfloating
        ):
            assert_allclose(result, expected, rtol=1e-15, atol=0.0)
        else:
            assert_equal(result, expected)


@pytest.mark.usefixtures(
    "interpreter_scheduler"
)  # batched and broadcasting not supported
@pytest.mark.parametrize(
    "a, b",
    [
        # 1D x 1D (dot product)
        (np.array([1, 2, 3]), np.array([4, 5, 6])),
        # 2D x 2D
        (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])),
        # 2D x 1D
        (np.array([[1, 2], [3, 4]]), np.array([5, 6])),
        # 1D x 2D
        (np.array([1, 2]), np.array([[3, 4], [5, 6]])),
        # 3D x 3D (batched matmul)
        (
            np.arange(2 * 3 * 4).reshape(2, 3, 4),
            np.arange(2 * 4 * 5).reshape(2, 4, 5),
        ),
        # Broadcasting cases
        # 1D x 2D (broadcasting)
        (np.array([1, 2]), np.arange(2 * 4).reshape(2, 4)),
        # 1D x 3D (broadcasting)
        (np.array([1, 2]), np.arange(3 * 2 * 5).reshape(3, 2, 5)),
        #  4D x 3D (broadcasting)
        (
            np.arange(7 * 2 * 4 * 3).reshape(7, 2, 4, 3),
            np.arange(2 * 3 * 4).reshape(2, 3, 4),
        ),
        # 3D x 1D (broadcasting)
        (np.arange(3 * 2 * 4).reshape(3, 2, 4), np.arange(4)),
        # (1, 3, 2) x (5, 2, 3)
        (np.arange(1 * 3 * 2).reshape(1, 3, 2), np.arange(5 * 2 * 3).reshape(5, 2, 3)),
        # Complex numbers, 4D x 5D
        (
            random_array((2, 3, 4, 5)),
            random_array((3, 5, 6)),
        ),
        # mismatch dimensions
        (
            np.arange(7 * 2 * 3 * 4).reshape(7, 2, 3, 4),
            np.arange(2 * 3 * 4).reshape(2, 3, 4),
        ),
        (np.arange(5), np.arange(4)),
    ],
)
@pytest.mark.parametrize(
    "a_wrap",
    [
        lambda x: x,
        TestEagerTensor,
        finch.defer,
    ],
)
@pytest.mark.parametrize(
    "b_wrap",
    [
        lambda x: x,
        TestEagerTensor,
        finch.defer,
    ],
)
def test_matmul(a, b, a_wrap, b_wrap):
    """
    Tests for matrix multiplication using finch's matmul function.
    """
    wa = a_wrap(a)
    wb = b_wrap(b)

    try:
        expected = np.linalg.matmul(a, b)
    except ValueError:
        with pytest.raises(ValueError):
            finch.matmul(wa, wb)  # make sure matmul raises error too
            _ = wa @ wb
        return

    result = finch.matmul(wa, wb)
    result_with_op = wa @ wb  # make sure the operator overload works too
    result_with_np = np.matmul(wa, wb)

    if isinstance(wa, finch.LazyTensor) or isinstance(wb, finch.LazyTensor):
        assert isinstance(result, finch.LazyTensor)
        result = finch.compute(result)
        result_with_op = finch.compute(result_with_op)
        result_with_np = finch.compute(result_with_np)

    assert isinstance(result, np.ndarray)
    assert expected.dtype == result.dtype, (
        f"Expected dtype {expected.dtype}, got {result.dtype}"
    )
    assert_allclose(result, expected)
    assert_allclose(result_with_op, expected)
    assert_allclose(result_with_np, expected)


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
@pytest.mark.parametrize(
    "a",
    [
        np.arange(6).reshape(2, 3),
        np.arange(12).reshape(1, 12),
        np.arange(24).reshape(2, 3, 4),  # 3D array
        # Complex
        random_array((5, 1, 4)),
    ],
)
@pytest.mark.parametrize(
    "a_wrap",
    [
        lambda x: x,
        TestEagerTensor,
        finch.defer,
    ],
)
def test_matrix_transpose(a, a_wrap):
    """
    Tests for matrix transpose
    """
    a = np.array(a)
    wa = a_wrap(a)
    expected = np.linalg.matrix_transpose(a)

    result = finch.matrix_transpose(wa)
    if isinstance(result, finch.LazyTensor):
        result = finch.compute(result)
    assert_equal(result, expected)


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
@pytest.mark.parametrize(
    "a, b, axes",
    [
        # 1D x 1D (dot product)
        (np.array([1, 2, 3]), np.array([4, 5, 6]), 0),
        # axes as int
        (np.arange(6).reshape(2, 3), np.arange(12).reshape(3, 4), 1),
        (np.arange(24).reshape(2, 3, 4), np.arange(12).reshape(4, 3), 1),
        (np.arange(24).reshape(2, 4, 3), np.arange(24).reshape(4, 3, 2), 2),
        # axes as tuple of sequences
        (
            np.arange(24).reshape(2, 3, 4),
            np.arange(24).reshape(4, 3, 2),
            ([1, 2], [1, 0]),
        ),
        (
            np.arange(60).reshape(3, 4, 5),
            np.arange(24).reshape(4, 3, 2),
            ([0, 1], [1, 0]),
        ),
        # axes=0 (outer product)
        (np.arange(3), np.arange(4), 0),
        (np.arange(8 * 7 * 5).reshape(8, 7, 5), np.arange(12).reshape(3, 4, 1), 0),
        # complex
        (random_array((2, 3)), random_array((3, 4)), 1),
        (
            random_array((3, 5, 4, 6)),
            random_array((6, 4, 5, 3)),
            ([2, 1, 3], [1, 2, 0]),
        ),
        # mismatched axes (should raise)
        (np.arange(6).reshape(2, 3), np.arange(8).reshape(2, 4), 1),
    ],
)
@pytest.mark.parametrize(
    "a_wrap",
    [
        lambda x: x,
        TestEagerTensor,
        finch.defer,
    ],
)
@pytest.mark.parametrize(
    "b_wrap",
    [
        lambda x: x,
        TestEagerTensor,
        finch.defer,
    ],
)
def test_tensordot(a, b, axes, a_wrap, b_wrap):
    """
    Tests for tensordot operation according to the Array API specification.
    See: https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.tensordot.html
    """
    wa = a_wrap(a)
    wb = b_wrap(b)
    try:
        expected = np.tensordot(a, b, axes=axes)
    except ValueError:
        # tensordot should raise a ValueError
        with pytest.raises(ValueError):
            finch.tensordot(wa, wb, axes=axes)
        return
    result = finch.tensordot(wa, wb, axes=axes)

    if isinstance(wa, finch.LazyTensor) or isinstance(wb, finch.LazyTensor):
        assert isinstance(result, finch.LazyTensor)
        result = finch.compute(result)
    assert isinstance(result, np.ndarray)  # for type checker
    assert_allclose(result, expected)


@pytest.mark.parametrize(
    "x1, x2, axis",
    [
        # 1D x 1D (scalar result)
        (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]), -1),
        # 2D x 2D (vector result)
        (np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]]), -1),
        # 3D x 3D, axis=-1
        (
            np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4),
            np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4),
            -1,
        ),
        # 3D x 3D, axis=-2
        (
            np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4),
            np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4),
            -2,
        ),
        # Broadcasting: (2, 3, 4) x (4,)
        (
            np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4),
            np.arange(4, dtype=float),
            -1,
        ),
        # Broadcasting: (3, 4) x (1, 4)
        (
            np.arange(3 * 4, dtype=float).reshape(3, 4),
            np.arange(4, dtype=float).reshape(1, 4),
            -1,
        ),
        # Complex numbers
        (np.array([1 + 2j, 3 + 4j]), np.array([5 - 1j, 2 + 2j]), -1),
        # axis=0
        (
            np.arange(2 * 3, dtype=float).reshape(2, 3),
            np.arange(2 * 3, dtype=float).reshape(2, 3),
            0,
        ),
        # Mismatched contracted axis
        (np.ones((2, 3)), np.ones((2, 4)), -1),
        (np.ones((5,)), np.ones((6,)), -1),
        # Broadcasting not allowed on contracted axis
        (np.ones((2, 3)), np.ones((1, 3)), 0),
    ],
)
@pytest.mark.parametrize(
    "x1_wrap",
    [
        lambda x: x,
        TestEagerTensor,
        finch.defer,
    ],
)
@pytest.mark.parametrize(
    "x2_wrap",
    [
        lambda x: x,
        TestEagerTensor,
        finch.defer,
    ],
)
def test_vecdot(x1, x2, axis, x1_wrap, x2_wrap):
    """
    Tests for vector dot product operation according to the Array API specification.
    See: https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.vecdot.html
    """
    wx1 = x1_wrap(x1)
    wx2 = x2_wrap(x2)
    try:
        expected = np.linalg.vecdot(x1, x2, axis=axis)
    except ValueError:
        with pytest.raises(ValueError):
            finch.vecdot(wx1, wx2, axis=axis)
        return

    result = finch.vecdot(wx1, wx2, axis=axis)
    if isinstance(wx1, finch.LazyTensor) or isinstance(wx2, finch.LazyTensor):
        assert isinstance(result, finch.LazyTensor)
        result = finch.compute(result)

    assert isinstance(result, np.ndarray), "Result should be a NumPy array"
    assert_allclose(result, expected)


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
@pytest.mark.parametrize(
    "x, axis, expected",
    [
        (np.array([[1], [2]]), 1, np.array([1, 2])),
        (np.array([[[3]]]), (0, 1), np.array([3])),
        (np.zeros((1, 2, 1, 3)), (0, 2), np.zeros((2, 3))),
        (np.array([[[1, 2, 3]]]), 0, np.array([[1, 2, 3]])),
    ],
)
def test_squeeze_valid(x, axis, expected):
    """
    Tests for squeeze operation
    """
    result = finch.squeeze(x, axis=axis)
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "x, axis",
    [
        (np.array([[1, 2], [3, 4]]), 0),  # axis 0 is not singleton
        (np.zeros((2, 1, 3)), (0, 2)),  # axis 0 and 2 not both singleton
        (np.ones((1, 2, 1)), (1,)),  # axis 1 is not singleton
    ],
)
def test_squeeze_invalid(x, axis):
    with pytest.raises(ValueError):
        finch.squeeze(x, axis=axis)


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
@pytest.mark.parametrize(
    "x, axis",
    [
        (np.array([1, 2, 3]), 0),
        (np.array([1, 2, 3]), 1),
        (np.array([[1, 2], [3, 4]]), 0),
        (np.array([[1, 2], [3, 4]]), 1),
        (np.array([[1, 2], [3, 4]]), 2),
        (np.array([1, 2, 3]), -1),
        (np.array([1, 2, 3]), -2),
        (np.array([[1, 2], [3, 4]]), -1),
        (np.array([[1, 2], [3, 4]]), -3),
    ],
)
def test_expand_dims_valid(x, axis):
    expected = np.expand_dims(x, axis=axis)
    result = finch.expand_dims(x, axis=axis)
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "x, axis",
    [
        (np.array([1, 2, 3]), 3),  # out of bounds
        (np.array([1, 2, 3]), -4),  # out of bounds
        (np.array([[1, 2], [3, 4]]), 4),
        (np.array([[1, 2], [3, 4]]), -4),
    ],
)
def test_expand_dims_invalid(x, axis):
    with pytest.raises(IndexError):
        finch.expand_dims(x, axis=axis)


@pytest.mark.parametrize(
    "x",
    [
        0,
        0.0,
        -4,
        1,
        2.4,
        -1.54,
        True,
        False,
        float("inf"),
        float("-inf"),
        float("nan"),
        complex(1, 2),
        complex(0, 0),
    ],
)
@pytest.mark.parametrize("func", [complex, int, float, bool])
def test_scalar_coerce(x, func):
    """
    Tests for scalar coercion to different types.
    """
    if isinstance(x, complex) and func in [int, float]:
        # no defined behavior in spec
        return

    try:
        expected = func(x)
    except (ValueError, TypeError, OverflowError):
        with pytest.raises((ValueError, TypeError, OverflowError)):
            print(func(TestEagerTensor(np.array(x))))
        return
    result = func(TestEagerTensor(np.array(x)))
    assert isinstance(result, func), f"Result should be of type {func.__name__}"
    works = result == expected or np.isnan(result) and np.isnan(expected)
    assert works, f"Expected {expected}, got {result}"


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
@pytest.mark.parametrize(
    "x, shape",
    [
        # ——— VALID CASES ———
        # scalar → 2×3
        (np.array(5), (2, 3)),
        # 1D int → 2×3
        (np.array([1, 2, 3]), (2, 3)),
        # 1D float → 1×3×2 (prepend one axis)
        (np.array([0.5, 1.5]), (1, 2, 2)),
        # 2D bool → 2×2×3 (prepend one axis)
        (np.array([[True, False, True]]), (2, 1, 3)),
        # 1-element → 2×1×3
        (np.array([7.0 + 4.2j]), (2, 1, 3)),
        # already matching shape (no change)
        (np.arange(6).reshape(2, 3), (2, 3)),
        # zero-length axis: (0,) → (4, 0)
        (np.empty((0,)), (4, 0)),
        # broadcast in middle: (1,4 +1j,1) → (3,4,2)
        (np.ones((1, 4, 1)), (3, 4, 2)),
        (np.arange(4).reshape(2, 2), (2, 2, 2)),
        # 1-dim can be broadcast to 0-dim: (3,1) → (3, 0)
        (np.array([1, 2, 3]).reshape(-1, 1), (3, 0)),
        # 0-dim can be prepended: (3, 2) → (0, 3, 2)
        (np.arange(6).reshape(3, 2), (0, 3, 2)),
        # ——— INVALID CASES ———
        # mismatched non-1 dim at end
        (np.array([1, 2, 3]), (2, 2)),
        # mismatched non-1 dim in middle
        (np.ones((2, 3, 4)), (2, 5, 4)),
        # broadcast on right side
        (np.arange(3).reshape(3, 1), (2, 3, 1, 5)),
    ],
)
@pytest.mark.parametrize(
    "x_wrap",
    [
        lambda x: x,
        TestEagerTensor,
        finch.defer,
    ],
)
def test_broadcast_to(x, shape, x_wrap):
    """
    Tests for broadcasting an array to a specified shape.
    """
    wx = x_wrap(x)
    # try NumPy’s broadcast_to first
    try:
        expected = np.broadcast_to(x, shape)
    except ValueError:
        # if NumPy cannot broadcast, we expect finch to raise
        with pytest.raises(ValueError):
            finch.broadcast_to(wx, shape)

    else:
        out = finch.broadcast_to(wx, shape)
        if isinstance(wx, finch.LazyTensor):
            out = finch.compute(out)
        assert_equal(out, expected, strict=True)


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
@pytest.mark.parametrize(
    "shapes",
    [
        ((1, 2, 3), (4, 2, 3), (3,), (9, 4, 2, 3), (9, 1, 1, 3)),
        ((7, 2, 3, 4), (2, 3, 1), (2, 1, 1), (1, 2, 1, 1), (1,)),
        ((1,), (1,)),
        ((2, 3), (3, 2)),  # error
        ((1,), (4, 0)),
        ((0,), (1, 0)),
    ],
)
@pytest.mark.parametrize(
    "wrapper",
    [
        lambda x: x,
        TestEagerTensor,
        finch.defer,
    ],
)
def test_broadcast_arrays(shapes, wrapper, rng, random_wrapper):
    """
    Tests for broadcasting multiple arrays to a common shape.
    The wrapper is randomly applied to each shape to ensure
    """

    # Generate random arrays for each shape
    arrays = [rng.random(shape) for shape in shapes]
    wrapped_arrays = random_wrapper(arrays, wrapper)
    try:
        expected = np.broadcast_arrays(*arrays)
    except ValueError:
        with pytest.raises(ValueError):
            finch.broadcast_arrays(*wrapped_arrays)
        return
    result = finch.broadcast_arrays(*wrapped_arrays)
    if isinstance(result[0], finch.LazyTensor):
        assert all(isinstance(r, finch.LazyTensor) for r in result)
        result = finch.compute(result)  # compute all lazy tensors

    assert len(result) == len(expected), "Number of results does not match expected"
    for i, (res, exp) in enumerate(zip(result, expected, strict=True)):
        assert res.shape == exp.shape, (
            f"Shape mismatch: got {res.shape}, expected {exp.shape} at index {i}"
        )
        assert_equal(res, exp, "Values mismatch in broadcasted arrays")


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
@pytest.mark.parametrize(
    "shapes_and_types, axis",
    [
        # Basic concatenation along axis 0 - same types
        ([(2, 3, np.float32), (2, 3, np.float32), (2, 3, np.float32)], 0),
        # Different shapes along concat axis
        ([(2, 3, np.int32), (4, 3, np.int32), (3, 3, np.int32)], 0),
        # Concatenation along axis 1
        ([(3, 2, np.float64), (3, 4, np.float64), (3, 1, np.float64)], 1),
        # Mixed types - int and float promotion
        ([(2, 3, np.int32), (2, 3, np.float64), (2, 3, np.float32)], 0),
        # Bool and numeric promotion
        ([(3, 2, bool), (3, 2, np.int8), (3, 2, np.uint8)], 0),
        # Concatenation with complex types
        ([(2, 3, np.complex64), (2, 3, np.float32), (2, 3, np.int32)], 0),
        # 3D arrays with negative axis
        ([(2, 3, 4, np.float32), (5, 3, 4, np.float32), (1, 3, 4, np.int64)], -3),
        # Empty arrays with mixed types
        ([(0, 3, np.float32), (0, 3, np.float64)], 0),
        # Single array (no-op) with special type
        ([(2, 3, np.uint16)], 0),
        # Flattened concatenation with axis=None - mixed types
        ([(2, 3, np.int32), (3, 2, np.float32), (1, 1, np.complex64)], None),
    ],
)
@pytest.mark.parametrize(
    "wrapper",
    [
        lambda x: x,
        TestEagerTensor,
        finch.defer,
    ],
)
def test_concat(shapes_and_types, axis, wrapper, rng, random_wrapper):
    """
    Tests for concatenating arrays along specified axis with various types.
    """
    # Generate arrays for each shape and type
    arrays = []

    for shape_and_type in shapes_and_types:
        shape, type = shape_and_type[:-1], shape_and_type[-1]
        arrays.append(random_array(shape, type, rng))

    # Apply wrapper (randomly to ensure mixed types work)
    wrapped_arrays = random_wrapper(arrays, wrapper)

    expected = np.concatenate(arrays, axis=axis)

    # Test finch's implementation
    result = finch.concat(wrapped_arrays, axis=axis)

    # Evaluate lazy tensors if needed
    if isinstance(result, finch.LazyTensor):
        result = finch.compute(result)

    assert_equal(result, expected, "Values mismatch in concatenated array", strict=True)


@pytest.mark.parametrize(
    "shapes",
    [
        # Incompatible shapes (not matching in non-concatenation dimensions)
        [(2, 3), (2, 4)],
        # Different ndims
        [(2, 3), (2, 3, 4)],
        # Mixed types but incompatible shapes
        [(3, 2), (4, 3)],
    ],
)
def test_concat_invalid(shapes, rng):
    """
    Tests error handling for invalid concatenation cases.
    """
    arrays = [rng.random(shape) for shape in shapes]

    with pytest.raises(ValueError):
        finch.concat(arrays, axis=0)


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
@pytest.mark.parametrize(
    "shape, source, destination",
    [
        ((3, 4, 5), 0, -3),
        ((21, 1, 3, 2, 0), -1, -2),
        ((2, 3, 4), (0, 1), (2, 0)),
        ((5, 4, 3), (0, 1, 2), (-1, -2, -3)),
        ((5, 8, 9, 4, 3), (0, 1), (-1, 4)),  # error case
        ((5, 8, 9, 4, 3), (-9, 1), (-1, 74)),  # error case
    ],
)
@pytest.mark.parametrize(
    "wrapper",
    [
        lambda x: x,
        TestEagerTensor,
        finch.defer,
    ],
)
def test_moveaxis(shape, source, destination, wrapper, rng):
    """
    Tests for moving axes of an array to a new position.
    """
    # Generate a random array with the specified shape
    x = rng.random(shape)
    # Apply wrapper to input
    wrapped_x = wrapper(x)
    # Compute expected result using NumPy
    try:
        expected = np.moveaxis(x, source, destination)
    except ValueError:
        # If NumPy raises a ValueError, we expect finch to raise the same
        with pytest.raises(ValueError):
            finch.moveaxis(wrapped_x, source, destination)
        return

    result = finch.moveaxis(wrapped_x, source, destination)
    if isinstance(result, finch.LazyTensor):
        result = finch.compute(result)

    assert_equal(result, expected, "Values mismatch in moved axis array", strict=True)


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
@pytest.mark.parametrize(
    "shapes_and_types, axis",
    [
        # Basic stacking along axis 0 (default)
        ([(2, 3, np.float32), (2, 3, np.float32), (2, 3, np.float32)], 0),
        # Stacking along axis 1
        ([(2, 3, np.float64), (2, 3, np.float64), (2, 3, np.float64)], 1),
        # Stacking along axis -1 (last dimension)
        ([(3, 2, np.int32), (3, 2, np.int32), (3, 2, np.int32)], -1),
        # Mixed types - should promote
        ([(2, 3, np.int32), (2, 3, np.float64), (2, 3, np.float32)], 0),
        # Stacking complex types
        ([(2, 3, np.complex64), (2, 3, np.float32), (2, 3, np.int32)], 0),
        # Empty arrays
        ([(0, 3, np.float32), (0, 3, np.float32)], 0),
        # Single array case
        ([(2, 3, np.uint16)], 0),
        # Invalid cases - Different shapes
        ([(2, 3, np.float32), (3, 3, np.float32)], 0),
        # Invalid axis (out of bounds)
        ([(2, 3, np.float32), (2, 3, np.float32)], 3),
        ([(2, 3, np.float32), (2, 3, np.float32)], -4),
    ],
)
@pytest.mark.parametrize(
    "wrapper",
    [
        lambda x: x,
        TestEagerTensor,
        finch.defer,
    ],
)
def test_stack(shapes_and_types, axis, wrapper, rng, random_wrapper):
    """
    Tests for stacking arrays along a new axis.
    """
    # Generate arrays for each shape and type
    arrays = []

    for shape_and_type in shapes_and_types:
        shape, dtype = shape_and_type[:-1], shape_and_type[-1]
        arrays.append(random_array(shape, dtype, rng))

    # Apply wrapper (randomly to ensure mixed types work)
    wrapped_arrays = random_wrapper(arrays, wrapper)

    try:
        # Get expected result from NumPy
        expected = np.stack(arrays, axis=axis)
    except ValueError:
        # Check that finch also raises an error
        with pytest.raises(ValueError):
            finch.stack(wrapped_arrays, axis=axis)
        return

    # Test finch's implementation
    result = finch.stack(wrapped_arrays, axis=axis)

    # Evaluate lazy tensors if needed
    if isinstance(result, finch.LazyTensor):
        result = finch.compute(result)

    assert_equal(result, expected, "Values mismatch in stacked array", strict=True)


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
@pytest.mark.parametrize(
    "array_shape, axis, split_shape, expected_shape",
    [
        ((2, 6), 1, (2, 3), (2, 2, 3)),
        ((4, 6), -1, (2, 3), (4, 2, 3)),
        ((2, 24), 1, (2, 3, 4), (2, 2, 3, 4)),
        ((8, 3), 0, (2, 4), (2, 4, 3)),
        ((6,), 0, (1, 6), (1, 6)),
        ((2, 8), 1, (1, 8), (2, 1, 8)),
        # Edge case: 3D tensor with split in middle dimension
        ((3, 6, 4), 1, (2, 3), (3, 2, 3, 4)),
    ],
)
@pytest.mark.parametrize(
    "wrapper",
    [
        lambda x: x,
        TestEagerTensor,
        finch.defer,
    ],
)
def test_split_dims(array_shape, axis, split_shape, expected_shape, wrapper):
    """Test splitting a dimension into multiple dimensions."""
    # Create input tensor with identifiable values
    x = np.arange(np.prod(array_shape)).reshape(array_shape)
    wrapped_x = wrapper(x)
    expected = np.reshape(x, expected_shape)
    # Apply split_dims operation
    result = finch.split_dims(wrapped_x, axis, split_shape)
    # Compute if result is lazy
    if isinstance(result, finch.LazyTensor):
        result = finch.compute(result)

    assert_equal(result, expected, strict=True)


@pytest.mark.parametrize(
    "array_shape, axis, split_shape",
    [
        # Product of split shape doesn't match original dimension size
        ((2, 7), 1, (2, 3)),
        # Axis out of bounds
        ((2, 6), 2, (2, 3)),
        # Negative axis out of bounds
        ((2, 6), -3, (2, 3)),
        # Empty split shape
        ((2, 6), 1, ()),
    ],
)
def test_split_dims_errors(array_shape, axis, split_shape):
    """Test error cases for split_dims."""
    x = np.arange(np.prod(array_shape)).reshape(array_shape)
    with pytest.raises(ValueError):
        # error must be raised before computing
        finch.split_dims(x, axis, split_shape)


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
@pytest.mark.parametrize(
    "array_shape, axes, expected_shape",
    [
        # Basic 2D to 1D combinations
        ((2, 3, 4), (1, 2), (2, 12)),
        # Negative axis indexing
        ((4, 2, 3), (-3, -2), (8, 3)),
        # (combine all dimensions)
        ((2, 3, 4), (0, 1, 2), (24,)),
        # 4D
        ((2, 3, 4, 5), (1, 2), (2, 12, 5)),
        # Edge case: dimensions with size 1
        ((1, 3, 1, 4), (0, 1), (3, 1, 4)),
        # Edge case: zero-size dimensions
        ((0, 3, 4), (1, 2), (0, 12)),
        ((2, 0, 3), (0, 1), (0, 3)),
    ],
)
@pytest.mark.parametrize(
    "wrapper",
    [
        lambda x: x,
        TestEagerTensor,
        finch.defer,
    ],
)
def test_combine_dims(array_shape, axes, expected_shape, wrapper):
    """Test combining multiple consecutive dimensions into one."""
    # Create input tensor with identifiable values
    x = np.arange(np.prod(array_shape)).reshape(array_shape)
    wrapped_x = wrapper(x)
    expected = np.reshape(x, expected_shape)

    # Apply combine_dims operation
    result = finch.combine_dims(wrapped_x, axes)

    # Compute if result is lazy
    if isinstance(result, finch.LazyTensor):
        result = finch.compute(result)

    assert_equal(result, expected, strict=True)


@pytest.mark.parametrize(
    "array_shape, axes",
    [
        # Non-consecutive axes
        ((2, 3, 4, 5), (0, 2)),
        # Axis out of bounds
        ((2, 3, 4), (2, 3)),
        ((2, 3), (-3, -2)),
        # Empty axes tuple
        ((2, 3, 4), ()),
        # Single axis (need at least 2 for consecutive)
        ((2, 3, 4), (1,)),
    ],
)
def test_combine_dims_errors(array_shape, axes):
    """Test error cases for combine_dims."""
    x = np.arange(np.prod(array_shape)).reshape(array_shape)
    with pytest.raises(ValueError):
        # error must be raised before computing
        finch.combine_dims(x, axes)


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
@pytest.mark.parametrize(
    "array_shape, expected_shape",
    [
        # 2D array
        ((2, 3), (6,)),
        # 3D array
        ((2, 3, 4), (24,)),
        # 1D array (no change)
        ((6,), (6,)),
        # Scalar (0D) - should become 1D
        ((), (1,)),
        # Edge case: zero-size array
        ((0, 3), (0,)),
        # 4D array
        ((2, 1, 3, 2), (12,)),
    ],
)
@pytest.mark.parametrize(
    "wrapper",
    [
        lambda x: x,
        TestEagerTensor,
        finch.defer,
    ],
)
def test_flatten(array_shape, expected_shape, wrapper):
    """Test flattening arrays to 1D."""
    if array_shape == ():
        # Scalar case
        x = np.array(5)
    else:
        x = np.arange(np.prod(array_shape)).reshape(array_shape)

    wrapped_x = wrapper(x)
    expected = x.flatten() if array_shape != () else np.array([5])

    # Apply flatten operation
    result = finch.flatten(wrapped_x)

    # Compute if result is lazy
    if isinstance(result, finch.LazyTensor):
        result = finch.compute(result)

    assert_equal(result, expected, strict=True)
