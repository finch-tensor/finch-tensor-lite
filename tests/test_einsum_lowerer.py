import pytest

import numpy as np

import finch
from finch.autoschedule.einsum import LogicEinsumLoader
from finch.autoschedule.executor import LogicExecutor
from finch.autoschedule.formatter import DefaultLogicFormatter
from finch.autoschedule.loop_ordering import DefaultLoopOrderer
from finch.autoschedule.optimize import DefaultLogicOptimizer
from finch.finch_einsum import MockEinsumLoader
from finch.interface.fuse import compute
from finch.interface.lazy import lazy

from .conftest import finch_assert_allclose


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def ctx():
    return LogicExecutor(
        DefaultLogicOptimizer(
            DefaultLoopOrderer(
                DefaultLogicFormatter(LogicEinsumLoader(ctx_load=MockEinsumLoader()))
            )
        )
    )


def test_simple_addition(rng, ctx):
    """Test lowering of simple addition A + B"""
    A = lazy(rng.random((3, 3)))
    B = lazy(rng.random((3, 3)))

    C = finch.add(A, B)

    # Execute the plan
    result = compute(C, ctx=ctx)

    # Compare with expected
    expected = compute(A + B)
    finch_assert_allclose(result, expected)


def test_scalar_multiplication(rng, ctx):
    """Test lowering of scalar multiplication 2 * A"""
    A = lazy(rng.random((4, 4)))

    B = finch.multiply(2, A)

    result = compute(B, ctx=ctx)

    expected = compute(B)
    finch_assert_allclose(result, expected)


def test_element_wise_operations(rng, ctx):
    """Test lowering of element-wise operations"""
    A = lazy(rng.random((3, 3)))
    B = lazy(rng.random((3, 3)))
    C = lazy(rng.random((3, 3)))

    D = finch.add(finch.multiply(A, B), C)

    result = compute(D, ctx=ctx)

    expected = compute(D)
    finch_assert_allclose(result, expected)


def test_sum_reduction(rng, ctx):
    """Test sum reduction using +="""
    A = lazy(rng.random((3, 4)))

    B = finch.sum(A, axis=1)

    result = compute(B, ctx=ctx)

    expected = compute(B)
    finch_assert_allclose(result, expected)


def test_maximum_reduction(rng, ctx):
    """Test maximum reduction using max="""
    A = lazy(rng.random((3, 4)))

    B = finch.max(A, axis=1)

    result = compute(B, ctx=ctx)
    expected = compute(B)
    finch_assert_allclose(result, expected)


def test_batch_matrix_multiplication(rng, ctx):
    """Test batch matrix multiplication using +="""
    A = lazy(rng.random((2, 3, 4)))
    B = lazy(rng.random((2, 4, 5)))

    C = finch.matmul(A, B)

    result = compute(C, ctx=ctx)
    expected = compute(C)
    finch_assert_allclose(result, expected)


def test_minimum_reduction(rng, ctx):
    """Test minimum reduction using min="""
    A = lazy(rng.random((3, 4)))

    B = finch.min(A, axis=1)

    result = compute(B, ctx=ctx)
    expected = compute(B)
    finch_assert_allclose(result, expected)
