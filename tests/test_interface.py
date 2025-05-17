import numpy as np
from numpy.testing import assert_equal
import pytest
import finch
import operator


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


class TestEagerTensor(finch.EagerTensor):
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
    "op, finch_op, np_op",
    [
        (operator.add, finch.add, np.add),
        (operator.sub, finch.subtract, np.subtract),
        (operator.mul, finch.multiply, np.multiply),
    ],
)
def test_elementwise_operations(a, b, op, finch_op, np_op):
    ea = TestEagerTensor(a)
    la = finch.defer(a)
    eb = TestEagerTensor(b)
    lb = finch.defer(b)

    expected = np_op(a, b)

    result = op(ea, b)

    assert_equal(result, expected)

    result = op(a, eb)

    assert_equal(result, expected)

    result = finch_op(ea, b)

    assert_equal(result, expected)

    result = finch_op(a, eb)

    assert_equal(result, expected)

    result = finch.compute(op(la, eb))

    assert_equal(result, expected)

    result = finch.compute(op(ea, lb))

    assert_equal(result, expected)


@pytest.mark.parametrize(
    "a",
    [
        (np.array([[1, 2], [3, 4]])),
        (np.array([[2, 0], [1, 3]])),
    ],
)
@pytest.mark.parametrize(
    "op, finch_op, np_op",
    [
        (operator.abs, finch.abs, np.abs),
        (operator.pos, finch.positive, np.positive),
        (operator.neg, finch.negative, np.negative),
    ],
)
def test_unary_operations(a, op, finch_op, np_op):
    ea = TestEagerTensor(a)
    la = finch.defer(a)

    expected = np_op(a)

    result = op(ea)

    assert_equal(result, expected)

    result = op(la)

    assert_equal(result, expected)

    result = finch_op(ea)

    assert_equal(result, expected)

    result = finch_op(la)

    assert_equal(result, expected)
