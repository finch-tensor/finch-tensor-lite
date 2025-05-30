import pytest

import numpy as np
from numpy.testing import assert_equal

import finch

verbose = False
# verbose = True


def output_term(args):
    if verbose:
        print(args)


# TODO: (Overall) 3-D or higher dimension tensor testcases


@pytest.mark.parametrize(
    "a",
    [
        (np.array([[True, False, True, False], [False, False, False, False]])),
    ],
)
@pytest.mark.parametrize(
    "finch_op, np_op",
    [
        (finch.prod, np.prod),
        # (finch.sum,  np.sum),
        (finch.any, np.any),
        (finch.all, np.all),
        (finch.min, np.min),
        (finch.max, np.max),
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
def test_reduction_api_bitwise(a, finch_op, np_op, axis):
    result = finch_op(a, axis=axis)
    expected = np_op(a, axis=axis)
    assert_equal(result, expected)


@pytest.mark.parametrize(
    "a",
    [
        (np.array([[2, 0], [1, 3]])),
        (np.array([[2, 3, 4], [5, 6, 7]])),
        (np.array([[1, 0, 3, 8], [0, 0, 10, 0]])),
        (np.array([[100, 14, 9, 78], [44, 3, 5, 10]])),
    ],
)
@pytest.mark.parametrize(
    "finch_op, np_op",
    [
        (finch.prod, np.prod),
        (finch.sum, np.sum),
        (finch.any, np.any),
        (finch.all, np.all),
        (finch.min, np.min),
        (finch.max, np.max),
    ],
)
@pytest.mark.parametrize(
    "axis",
    [None, 0, 1, (0, 1)],
)
def test_reduction_api_integer(a, finch_op, np_op, axis):
    result = finch_op(a, axis=axis)
    expected = np_op(a, axis=axis)
    assert_equal(result, expected)


@pytest.mark.parametrize(
    "a",
    [
        (np.array([[0, 0.618, 0, 0.001], [0, 0.01, 0, 0]])),
        (np.array([[10000.0, 1.0, 89.0, 78], [401.0, 3, 5, 10.2]])),
    ],
)
@pytest.mark.parametrize(
    "finch_op, np_op",
    [
        (finch.prod, np.prod),
        (finch.sum, np.sum),
        (finch.any, np.any),
        (finch.all, np.all),
        (finch.min, np.min),
        (finch.max, np.max),
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
def test_reduction_api_floating(a, finch_op, np_op, axis):
    result = finch_op(a, axis=axis)
    expected = np_op(a, axis=axis)
    assert_equal(result, expected)
