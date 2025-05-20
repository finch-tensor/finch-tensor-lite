import numpy as np
from numpy.testing import assert_equal
import pytest
import finch

# verbose = False
verbose = True

def output_term(args):
    if verbose:
        print(args)


# TODO: (Overall) 3-D or higher dimension tensor testcases

@pytest.mark.parametrize(
    "a",
    [
        (np.array([[2, 3, 4], [5, 6, 7]])),
    ],
)
def test_lazyTensor_prod(a):
    output_term("")
    output_term("***** prod *****")

    # TODO: axis=None is not supported well in all interfaces?

    # result = finch.compute(finch.prod(finch.lazy(a), axis=None))
    # output_term(result)
    # expected = np.prod(a)
    # output_term(expected)
    # assert_equal(result, expected)
    # output_term("")

    # result = finch.compute(finch.prod(a, axis=0)); output_term(result)
    result = finch.prod(a, axis=0); output_term(result)
    expected = np.prod(a, axis=0); output_term(expected)
    assert_equal(result, expected)
    output_term("")

    # result = finch.compute(finch.prod(a, axis=1)); output_term(result)
    result = finch.prod(a, axis=1); output_term(result)
    expected = np.prod(a, axis=1); output_term(expected)
    assert_equal(result, expected)
    output_term("")


@pytest.mark.parametrize(
    "a",
    [
        (np.array([[2, 4, 6, 8], [1, 3, 5, 7]])),
    ],
)
def test_lazyTensor_sum(a):
    output_term("***** sum *****")

    # result = finch.compute(finch.sum(finch.lazy(a), axis=0)); output_term(result)
    result = finch.sum(a, axis=0); output_term(result)
    expected = np.sum(a, axis=0); output_term(expected)
    assert_equal(result, expected)
    output_term("")

    # result = finch.compute(finch.sum(finch.lazy(a), axis=1)); output_term(result)
    result = finch.sum(a, axis=1); output_term(result)
    expected = np.sum(a, axis=1); output_term(expected)
    assert_equal(result, expected)
    output_term("")


@pytest.mark.parametrize(
    # (np.array([[0, 0.618, 0, 0.001], [0, 0.01, 0, 0]])),
    # (np.array([[True, False, True, False], [False, False, False, False]])),
    "a",
    [
        (np.array([[0, 4, 0, 8], [0, 0, 0, 0]])),
    ],
)
def test_lazyTensor_any(a):
    output_term("***** any *****")

    # TODO: init_value set to all False.

    # result = np.asarray(finch.compute(finch.any(finch.lazy(a), axis=0)), dtype=bool); output_term(result)
    result = finch.any(a, axis=0); output_term(result)
    expected = np.any(a, axis=0); output_term(expected)
    assert_equal(result, expected)
    output_term("")

    # result = np.asarray(finch.compute(finch.any(finch.lazy(a), axis=1)), dtype=bool); output_term(result)
    result = finch.any(a, axis=1); output_term(result)
    expected = np.any(a, axis=1); output_term(expected)
    assert_equal(result, expected)
    output_term("")


@pytest.mark.parametrize(
    "a",
    [
        (np.array([[1, 4, 3, 8], [0, 0, 10, 0]])),
    ],
)
def test_lazyTensor_all(a):
    output_term("***** all *****")

    # TODO: init_value set to all True.

    # result = finch.compute(finch.all(finch.lazy(a), axis=0)); output_term(result)
    result = finch.all(a, axis=0); output_term(result)
    expected = np.all(a, axis=0); output_term(expected)
    assert_equal(result, expected)
    output_term("")

    # result = finch.compute(finch.all(finch.lazy(a), axis=1)); output_term(result)
    result = finch.all(a, axis=1); output_term(result)
    expected = np.all(a, axis=1); output_term(expected)
    assert_equal(result, expected)
    output_term("")


@pytest.mark.parametrize(
    "a",
    [
        (np.array([[100, 14, 9, 78], [44, 3, 5, 10]])),
    ],
)
def test_lazyTensor_min(a):
    output_term("***** min *****")

    # TODO: init_value set to dtype.max.

    # result = finch.compute(finch.min(finch.lazy(a), axis=0)); output_term(result)
    result = finch.min(a, axis=0); output_term(result)
    expected = np.min(a, axis=0); output_term(expected)
    assert_equal(result, expected)
    output_term("")

    # result = finch.compute(finch.min(finch.lazy(a), axis=1)); output_term(result)
    result = finch.min(a, axis=1); output_term(result)
    expected = np.min(a, axis=1); output_term(expected)
    assert_equal(result, expected)
    output_term("")


@pytest.mark.parametrize(
    "a",
    [
        (np.array([[100, 14, 9, 78], [44, 3, 5, 10]])),
    ],
)
def test_lazyTensor_max(a):
    output_term("***** max *****")

    # TODO: init_value set to dtype.max.

    # result = finch.compute(finch.max(finch.lazy(a), axis=0)); output_term(result)
    result = finch.max(a, axis=0); output_term(result)
    expected = np.max(a, axis=0); output_term(expected)
    assert_equal(result, expected)
    output_term("")

    # result = finch.compute(finch.max(finch.lazy(a), axis=1)); output_term(result)
    result = finch.max(a, axis=1); output_term(result)
    expected = np.max(a, axis=1); output_term(expected)
    assert_equal(result, expected)
    output_term("")


@pytest.mark.parametrize(
    "a",
    [
        (np.array([[100, 14, 9, 78], [44, 3, 5, 10]])),
    ],
)
def test_lazyTensor_mean(a):
    output_term("***** mean *****")

    # result = finch.compute(finch.mean(finch.lazy(a), axis=0)); output_term(result)
    result = finch.mean(a, axis=0); output_term(result)
    expected = np.mean(a, axis=0); output_term(expected)
    # assert_equal(result, expected)
    output_term("")

    # result = finch.compute(finch.mean(finch.lazy(a), axis=1)); output_term(result)
    result = finch.mean(a, axis=1); output_term(result)
    expected = np.mean(a, axis=1); output_term(expected)
    # assert_equal(result, expected)
    output_term("")

@pytest.mark.parametrize(
    "a",
    [
        (np.array([[100, 14, 9, 78], [44, 3, 5, 10]])),
    ],
)
def test_lazyTensor_var(a):
    output_term("***** var *****")

    # result = finch.compute(finch.var(finch.lazy(a), axis=0)); output_term(result)
    result = finch.var(a, axis=0, correction=0); output_term(result)
    expected = np.var(a, axis=0); output_term(expected)
    # assert_equal(result, expected)
    output_term("")

    # result = finch.compute(finch.var(finch.lazy(a), axis=1)); output_term(result)
    result = finch.var(a, axis=1, correction=0); output_term(result)
    expected = np.var(a, axis=1); output_term(expected)
    # assert_equal(result, expected)
    output_term("")

# @pytest.mark.parametrize(
#     "a",
#     [
#         (np.array([[100, 14, 9, 78], [44, 3, 5, 10]])),
#     ],
# )
# def test_lazyTensor_std(a):
#     output_term("***** std *****")

#     result = finch.compute(finch.std(finch.lazy(a), axis=0)); output_term(result)
#     expected = np.std(a, axis=0); output_term(expected)
#     # assert_equal(result, expected)
#     output_term("")

#     result = finch.compute(finch.std(finch.lazy(a), axis=1)); output_term(result)
#     expected = np.std(a, axis=1); output_term(expected)
#     # assert_equal(result, expected)
#     output_term("")
