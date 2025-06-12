import pytest

import numpy as np
from numpy.testing import assert_equal

import finch

# verbose = False
verbose = True


def output_term(args):
    if verbose:
        print(args)


# TODO: Combine these test cases into test_interface on final commit


@pytest.mark.parametrize(
    "a",
    [
        (np.array([[2, 4, 6, 8], [1, 3, 5, 7]])),
    ],
)
def test_lazyTensor_mean(a):
    result = finch.mean(a, axis=0)
    expected = np.mean(a, axis=0)
    assert_equal(result, expected)

    result = finch.mean(a, axis=1)
    expected = np.mean(a, axis=1)
    assert_equal(result, expected)


@pytest.mark.parametrize(
    "a",
    [
        (np.array([[2, 4, 6, 8], [1, 3, 5, 7]])),
    ],
)
def test_lazyTensor_var(a):
    result = finch.var(a, axis=0, correction=1.0)
    expected = np.var(a, axis=0, correction=1.0)
    assert_equal(result, expected)

    # result = finch.var(a, axis=1, correction=1.0); output_term(result)
    # expected = np.var(a, axis=1, correction=1.0); output_term(expected)
    # assert_equal(result, expected)


@pytest.mark.parametrize(
    "a",
    [
        (np.array([[2, 4, 6, 8], [1, 3, 5, 7]])),
    ],
)
def test_lazyTensor_std(a):
    result = finch.std(a, axis=0, correction=1.0)
    expected = np.std(a, axis=0, correction=1.0)
    assert_equal(result, expected)

    # result = finch.std(a, axis=1, correction=1.0)
    # expected = np.std(a, axis=1, correction=1.0)
    # assert_equal(result, expected)
