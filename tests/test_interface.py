import operator

import pytest

import numpy as np
from numpy.testing import assert_equal

import finch


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


class TestEagerTensor(finch.AbstractEagerTensor):
    def __init__(self, array):
        self.array = np.array(array)

    def __repr__(self):
        return f"TestEagerTensor({self.array})"

    def __getitem__(self, item):
        return self.array[item]

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def fill_value(self):
        return finch.fill_value(self.array)

    @property
    def element_type(self):
        return finch.element_type(self.array)

    def to_numpy(self):
        return self.array


@pytest.mark.parametrize(
    "a, b",
    [
        (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])),
        (np.array([[2, 0], [1, 3]]), np.array([[4, 1], [2, 2]])),
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
    ],
)
def test_elementwise_operations(a, b, a_wrap, b_wrap, ops, np_op):
    wa = a_wrap(a)
    wb = b_wrap(b)

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
    ],
)
def test_unary_operations(a, a_wrap, ops, np_op):
    wa = a_wrap(a)

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
        ((finch.sum, np.sum), np.sum),
        ((finch.prod, np.prod), np.prod),
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

        assert_equal(result, expected)


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
        # mismatch dimensions
        (
            np.arange(7 * 2 * 4 * 3).reshape(7, 2, 3, 4),
            np.arange(2 * 3 * 4).reshape(2, 3, 4),
        ),
        (
            np.arange(5), np.arange(4)
        ),
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
        with pytest.raises(Exception):
            finch.matmul(wa, wb) # make sure matmul raises error too
        return

    result = finch.matmul(wa, wb)

    if isinstance(wa, finch.LazyTensor) or isinstance(wb, finch.LazyTensor):
        assert isinstance(result, finch.LazyTensor)
        result = finch.compute(result)

    assert_equal(result, expected)


@pytest.mark.parametrize(
    "a",
    [
        np.arange(6).reshape(2, 3),
        np.arange(12).reshape(1, 12),
        np.arange(24).reshape(2, 3, 4), # 3D array
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
    expected = np.linalg.matrix_transpose(a)

    result = finch.matrix_transpose(a)
    if isinstance(result, finch.LazyTensor):
        result = finch.compute(result)
    assert_equal(result, expected)

