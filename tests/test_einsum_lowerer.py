from typing import cast

import pytest

import numpy as np

import finchlite
from finchlite.autoschedule import optimize
from finchlite.autoschedule.einsum import EinsumLowerer
from finchlite.finch_einsum import EinsumInterpreter
from finchlite.finch_logic import Alias, Plan, Produces, Query
from finchlite.interface.fuse import compute
from finchlite.interface.lazy import LazyTensor, lazy
from finchlite.symbolic import gensym

from .conftest import finch_assert_allclose


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def lower_and_execute(tns: LazyTensor):
    """
    Helper function to optimize, lower, and execute a Logic IR plan.

    Args:
        plan: The Logic IR plan to execute

    Returns:
        The result of executing the einsum plan
    """
    ir = tns.data
    # Optimize into a plan
    var = Alias(gensym("result"))
    plan = Plan(tns.ctx.trace() + (Query(var, ir), Produces((var,))))
    optimized_plan = cast(Plan, optimize(plan))

    # Lower to einsum IR
    lowerer = EinsumLowerer()
    einsum_plan, bindings = lowerer(optimized_plan)

    for k, v in bindings.items():
        if hasattr(v, "to_numpy"):
            bindings[k] = v.to_numpy()
        elif isinstance(v, finchlite.interface.Scalar):
            bindings[k] = v.val

    # Interpret and execute
    interpreter = EinsumInterpreter()
    return interpreter(einsum_plan, bindings)[0]


def test_simple_addition(rng):
    """Test lowering of simple addition A + B"""
    A = lazy(rng.random((3, 3)))
    B = lazy(rng.random((3, 3)))

    C = finchlite.add(A, B)

    # Execute the plan
    result = lower_and_execute(C)

    # Compare with expected
    expected = compute(A + B)
    finch_assert_allclose(result, expected)


def test_scalar_multiplication(rng):
    """Test lowering of scalar multiplication 2 * A"""
    A = lazy(rng.random((4, 4)))

    B = finchlite.multiply(2, A)

    result = lower_and_execute(B)

    expected = compute(B)
    finch_assert_allclose(result, expected)


def test_element_wise_operations(rng):
    """Test lowering of element-wise operations"""
    A = lazy(rng.random((3, 3)))
    B = lazy(rng.random((3, 3)))
    C = lazy(rng.random((3, 3)))

    D = finchlite.add(finchlite.multiply(A, B), C)

    result = lower_and_execute(D)

    expected = compute(D)
    finch_assert_allclose(result, expected)


def test_sum_reduction(rng):
    """Test sum reduction using +="""
    A = lazy(rng.random((3, 4)))

    B = finchlite.sum(A, axis=1)

    result = lower_and_execute(B)

    expected = compute(B)
    finch_assert_allclose(result, expected)


def test_maximum_reduction(rng):
    """Test maximum reduction using max="""
    A = lazy(rng.random((3, 4)))

    B = finchlite.max(A, axis=1)

    result = lower_and_execute(B)
    expected = compute(B)
    finch_assert_allclose(result, expected)


def test_batch_matrix_multiplication(rng):
    """Test batch matrix multiplication using +="""
    A = lazy(rng.random((2, 3, 4)))
    B = lazy(rng.random((2, 4, 5)))

    C = finchlite.matmul(A, B)

    result = lower_and_execute(C)
    expected = compute(C)
    finch_assert_allclose(result, expected)


def test_minimum_reduction(rng):
    """Test minimum reduction using min="""
    A = lazy(rng.random((3, 4)))

    B = finchlite.min(A, axis=1)

    result = lower_and_execute(B)
    expected = compute(B)
    finch_assert_allclose(result, expected)
