import pytest
import numpy as np
import operator
from finchlite.autoschedule.einsum import EinsumLowerer
from finchlite.finch_logic import (
    Plan,
    Produces,
    Query,
    Alias,
    Table,
    MapJoin,
    Literal,
    Aggregate,
    Relabel,
    Field,
    Reorder,
)
from finchlite.finch_einsum import EinsumInterpreter
from finchlite.algebra import promote_max, promote_min


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_simple_addition(rng):
    """Test lowering of simple addition A + B"""
    A = rng.random((3, 3))
    B = rng.random((3, 3))

    # Create logic IR for C[i,j] = A[i,j] + B[i,j]
    plan = Plan((
        Query(Alias("A"), Table(A, (Field("i"), Field("j")))),
        Query(Alias("B"), Table(B, (Field("i"), Field("j")))),
        Query(
            Alias("C"),
            MapJoin(
                Literal(operator.add),
                (
                    Relabel(Alias("A"), (Field("i"), Field("j"))),
                    Relabel(Alias("B"), (Field("i"), Field("j"))),
                ),
            ),
        ),
        Produces((Alias("C"),)),
    ))

    # Lower to einsum IR
    lowerer = EinsumLowerer()
    einsum_plan, parameters = lowerer(plan)

    # Interpret einsum
    interpreter = EinsumInterpreter(bindings=parameters)
    result = interpreter(einsum_plan)

    # Compare with expected
    expected = A + B
    assert np.allclose(result, expected)


def test_scalar_multiplication(rng):
    """Test lowering of scalar multiplication 2 * A"""
    A = rng.random((4, 4))

    plan = Plan((
        Query(Alias("A"), Table(A, (Field("i"), Field("j")))),
        Query(
            Alias("B"),
            MapJoin(
                Literal(operator.mul),
                (
                    Literal(2),
                    Relabel(Alias("A"), (Field("i"), Field("j"))),
                ),
            ),
        ),
        Produces((Alias("B"),)),
    ))

    lowerer = EinsumLowerer()
    einsum_plan, parameters = lowerer(plan)
    interpreter = EinsumInterpreter(bindings=parameters)
    result = interpreter(einsum_plan)

    expected = 2 * A
    assert np.allclose(result, expected)


def test_element_wise_operations(rng):
    """Test lowering of element-wise operations"""
    A = rng.random((3, 3))
    B = rng.random((3, 3))
    C = rng.random((3, 3))

    # D[i,j] = A[i,j] * B[i,j] + C[i,j]
    plan = Plan((
        Query(Alias("A"), Table(A, (Field("i"), Field("j")))),
        Query(Alias("B"), Table(B, (Field("i"), Field("j")))),
        Query(Alias("C"), Table(C, (Field("i"), Field("j")))),
        Query(
            Alias("D"),
            MapJoin(
                Literal(operator.add),
                (
                    MapJoin(
                        Literal(operator.mul),
                        (
                            Relabel(Alias("A"), (Field("i"), Field("j"))),
                            Relabel(Alias("B"), (Field("i"), Field("j"))),
                        ),
                    ),
                    Relabel(Alias("C"), (Field("i"), Field("j"))),
                ),
            ),
        ),
        Produces((Alias("D"),)),
    ))

    lowerer = EinsumLowerer()
    einsum_plan, parameters = lowerer(plan)
    interpreter = EinsumInterpreter(bindings=parameters)
    result = interpreter(einsum_plan)

    expected = A * B + C
    assert np.allclose(result, expected)


def test_sum_reduction(rng):
    """Test lowering of sum reduction C[i] = sum_j A[i,j]"""
    A = rng.random((3, 4))

    plan = Plan((
        Query(Alias("A"), Table(A, (Field("i"), Field("j")))),
        Query(
            Alias("C"),
            Aggregate(
                Literal(operator.add),
                Literal(0),  # init value
                Relabel(Alias("A"), (Field("i"), Field("j"))),
                (Field("j"),),  # sum over j
            ),
        ),
        Produces((Alias("C"),)),
    ))

    lowerer = EinsumLowerer()
    einsum_plan, parameters = lowerer(plan)
    interpreter = EinsumInterpreter(bindings=parameters)
    result = interpreter(einsum_plan)

    expected = np.sum(A, axis=1)
    assert np.allclose(result, expected)


def test_max_reduction(rng):
    """Test lowering of max reduction C[i] = max_j A[i,j]"""
    A = rng.random((3, 4))

    plan = Plan((
        Query(Alias("A"), Table(A, (Field("i"), Field("j")))),
        Query(
            Alias("C"),
            Aggregate(
                Literal(promote_max),
                Literal(-np.inf),  # init value for max
                Relabel(Alias("A"), (Field("i"), Field("j"))),
                (Field("j"),),  # max over j
            ),
        ),
        Produces((Alias("C"),)),
    ))

    lowerer = EinsumLowerer()
    einsum_plan, parameters = lowerer(plan)
    interpreter = EinsumInterpreter(bindings=parameters)
    result = interpreter(einsum_plan)

    expected = np.max(A, axis=1)
    assert np.allclose(result, expected)


def test_min_reduction(rng):
    """Test lowering of min reduction C[i] = min_j A[i,j]"""
    A = rng.random((3, 4))

    plan = Plan((
        Query(Alias("A"), Table(A, (Field("i"), Field("j")))),
        Query(
            Alias("C"),
            Aggregate(
                Literal(promote_min),
                Literal(np.inf),  # init value for min
                Relabel(Alias("A"), (Field("i"), Field("j"))),
                (Field("j"),),  # min over j
            ),
        ),
        Produces((Alias("C"),)),
    ))

    lowerer = EinsumLowerer()
    einsum_plan, parameters = lowerer(plan)
    interpreter = EinsumInterpreter(bindings=parameters)
    result = interpreter(einsum_plan)

    expected = np.min(A, axis=1)
    assert np.allclose(result, expected)


def test_matrix_multiplication(rng):
    """Test lowering of matrix multiplication C[i,j] = sum_k A[i,k] * B[k,j]"""
    A = rng.random((3, 4))
    B = rng.random((4, 5))

    plan = Plan((
        Query(Alias("A"), Table(A, (Field("i"), Field("k")))),
        Query(Alias("B"), Table(B, (Field("k"), Field("j")))),
        Query(
            Alias("C"),
            Aggregate(
                Literal(operator.add),
                Literal(0),
                MapJoin(
                    Literal(operator.mul),
                    (
                        Relabel(Alias("A"), (Field("i"), Field("k"))),
                        Relabel(Alias("B"), (Field("k"), Field("j"))),
                    ),
                ),
                (Field("k"),),  # sum over k
            ),
        ),
        Produces((Alias("C"),)),
    ))

    lowerer = EinsumLowerer()
    einsum_plan, parameters = lowerer(plan)
    interpreter = EinsumInterpreter(bindings=parameters)
    result = interpreter(einsum_plan)

    expected = A @ B
    assert np.allclose(result, expected)


def test_nested_operations(rng):
    """Test nested operations: D = (A + B) * C"""
    A = rng.random((3, 3))
    B = rng.random((3, 3))
    C = rng.random((3, 3))

    plan = Plan((
        Query(Alias("A"), Table(A, (Field("i"), Field("j")))),
        Query(Alias("B"), Table(B, (Field("i"), Field("j")))),
        Query(Alias("C"), Table(C, (Field("i"), Field("j")))),
        Query(
            Alias("D"),
            MapJoin(
                Literal(operator.mul),
                (
                    MapJoin(
                        Literal(operator.add),
                        (
                            Relabel(Alias("A"), (Field("i"), Field("j"))),
                            Relabel(Alias("B"), (Field("i"), Field("j"))),
                        ),
                    ),
                    Relabel(Alias("C"), (Field("i"), Field("j"))),
                ),
            ),
        ),
        Produces((Alias("D"),)),
    ))

    lowerer = EinsumLowerer()
    einsum_plan, parameters = lowerer(plan)
    interpreter = EinsumInterpreter(bindings=parameters)
    result = interpreter(einsum_plan)

    expected = (A + B) * C
    assert np.allclose(result, expected)


def test_multiple_aggregations(rng):
    """Test multiple aggregations in sequence"""
    A = rng.random((3, 4, 5))

    # First sum over k: B[i,j] = sum_k A[i,j,k]
    # Then sum over j: C[i] = sum_j B[i,j]
    plan = Plan((
        Query(Alias("A"), Table(A, (Field("i"), Field("j"), Field("k")))),
        Query(
            Alias("B"),
            Aggregate(
                Literal(operator.add),
                Literal(0),
                Relabel(Alias("A"), (Field("i"), Field("j"), Field("k"))),
                (Field("k"),),
            ),
        ),
        Query(
            Alias("C"),
            Aggregate(
                Literal(operator.add),
                Literal(0),
                Relabel(Alias("B"), (Field("i"), Field("j"))),
                (Field("j"),),
            ),
        ),
        Produces((Alias("C"),)),
    ))

    lowerer = EinsumLowerer()
    einsum_plan, parameters = lowerer(plan)
    interpreter = EinsumInterpreter(bindings=parameters)
    result = interpreter(einsum_plan)

    expected = np.sum(np.sum(A, axis=2), axis=1)
    assert np.allclose(result, expected)


def test_aggregate_with_pointwise(rng):
    """Test aggregation combined with pointwise operations"""
    A = rng.random((3, 4))
    B = rng.random((3, 4))

    # C[i] = sum_j (A[i,j] * B[i,j])
    plan = Plan((
        Query(Alias("A"), Table(A, (Field("i"), Field("j")))),
        Query(Alias("B"), Table(B, (Field("i"), Field("j")))),
        Query(
            Alias("C"),
            Aggregate(
                Literal(operator.add),
                Literal(0),
                MapJoin(
                    Literal(operator.mul),
                    (
                        Relabel(Alias("A"), (Field("i"), Field("j"))),
                        Relabel(Alias("B"), (Field("i"), Field("j"))),
                    ),
                ),
                (Field("j"),),
            ),
        ),
        Produces((Alias("C"),)),
    ))

    lowerer = EinsumLowerer()
    einsum_plan, parameters = lowerer(plan)
    interpreter = EinsumInterpreter(bindings=parameters)
    result = interpreter(einsum_plan)

    expected = np.sum(A * B, axis=1)
    assert np.allclose(result, expected)


def test_transpose(rng):
    """Test lowering of transpose operation"""
    A = rng.random((3, 4))

    plan = Plan((
        Query(Alias("A"), Table(A, (Field("i"), Field("j")))),
        Query(
            Alias("B"),
            Reorder(
                Relabel(Alias("A"), (Field("i"), Field("j"))),
                (Field("j"), Field("i")),
            ),
        ),
        Produces((Alias("B"),)),
    ))

    lowerer = EinsumLowerer()
    einsum_plan, parameters = lowerer(plan)
    interpreter = EinsumInterpreter(bindings=parameters)
    result = interpreter(einsum_plan)

    expected = A.T
    assert np.allclose(result, expected)


def test_permutation_3d(rng):
    """Test permutation of 3D tensor"""
    A = rng.random((2, 3, 4))

    # Permute from [i,j,k] to [k,i,j]
    plan = Plan((
        Query(Alias("A"), Table(A, (Field("i"), Field("j"), Field("k")))),
        Query(
            Alias("B"),
            Reorder(
                Relabel(Alias("A"), (Field("i"), Field("j"), Field("k"))),
                (Field("k"), Field("i"), Field("j")),
            ),
        ),
        Produces((Alias("B"),)),
    ))

    lowerer = EinsumLowerer()
    einsum_plan, parameters = lowerer(plan)
    interpreter = EinsumInterpreter(bindings=parameters)
    result = interpreter(einsum_plan)

    expected = np.transpose(A, (2, 0, 1))
    assert np.allclose(result, expected)


def test_multiple_outputs(rng):
    """Test lowering with multiple output tensors"""
    A = rng.random((3, 3))
    B = rng.random((3, 3))

    plan = Plan((
        Query(Alias("A"), Table(A, (Field("i"), Field("j")))),
        Query(Alias("B"), Table(B, (Field("i"), Field("j")))),
        Query(
            Alias("C"),
            MapJoin(
                Literal(operator.add),
                (
                    Relabel(Alias("A"), (Field("i"), Field("j"))),
                    Relabel(Alias("B"), (Field("i"), Field("j"))),
                ),
            ),
        ),
        Query(
            Alias("D"),
            MapJoin(
                Literal(operator.mul),
                (
                    Relabel(Alias("A"), (Field("i"), Field("j"))),
                    Relabel(Alias("B"), (Field("i"), Field("j"))),
                ),
            ),
        ),
        Produces((Alias("C"), Alias("D"))),
    ))

    lowerer = EinsumLowerer()
    einsum_plan, parameters = lowerer(plan)
    interpreter = EinsumInterpreter(bindings=parameters)
    result_c, result_d = interpreter(einsum_plan)

    expected_c = A + B
    expected_d = A * B
    assert np.allclose(result_c, expected_c)
    assert np.allclose(result_d, expected_d)


def test_empty_plan():
    """Test lowering of empty plan"""
    plan = Plan(())
    lowerer = EinsumLowerer()
    einsum_plan, parameters = lowerer(plan)
    assert len(einsum_plan.bodies) == 0
    assert len(einsum_plan.returnValues) == 0


def test_scalar_operations():
    """Test operations with scalar results"""
    A = np.array([[1, 2], [3, 4]])

    # Total sum: result = sum_{i,j} A[i,j]
    plan = Plan((
        Query(Alias("A"), Table(A, (Field("i"), Field("j")))),
        Query(
            Alias("result"),
            Aggregate(
                Literal(operator.add),
                Literal(0),
                Relabel(Alias("A"), (Field("i"), Field("j"))),
                (Field("i"), Field("j")),  # sum over all dimensions
            ),
        ),
        Produces((Alias("result"),)),
    ))

    lowerer = EinsumLowerer()
    einsum_plan, parameters = lowerer(plan)
    interpreter = EinsumInterpreter(bindings=parameters)
    result = interpreter(einsum_plan)

    expected = np.sum(A)
    assert np.allclose(result, expected)


def test_nested_aggregate_in_pointwise(rng):
    """Test aggregate inside a pointwise expression"""
    A = rng.random((3, 4))

    # C[i,j] = A[i,j] + (sum_k A[i,k])
    # This requires the aggregate to be computed separately
    plan = Plan((
        Query(Alias("A"), Table(A, (Field("i"), Field("j")))),
        Query(
            Alias("C"),
            MapJoin(
                Literal(operator.add),
                (
                    Relabel(Alias("A"), (Field("i"), Field("j"))),
                    Aggregate(
                        Literal(operator.add),
                        Literal(0),
                        Relabel(Alias("A"), (Field("i"), Field("k"))),
                        (Field("k"),),
                    ),
                ),
            ),
        ),
        Produces((Alias("C"),)),
    ))

    lowerer = EinsumLowerer()
    einsum_plan, parameters = lowerer(plan)
    interpreter = EinsumInterpreter(bindings=parameters)
    result = interpreter(einsum_plan)

    row_sums = np.sum(A, axis=1, keepdims=True)
    expected = A + row_sums
    assert np.allclose(result, expected)


def test_commutative_flattening(rng):
    """Test that commutative operations are flattened"""
    A = rng.random((3, 3))
    B = rng.random((3, 3))
    C = rng.random((3, 3))
    D = rng.random((3, 3))

    # (A + B) + (C + D) should be flattened to A + B + C + D
    plan = Plan((
        Query(Alias("A"), Table(A, (Field("i"), Field("j")))),
        Query(Alias("B"), Table(B, (Field("i"), Field("j")))),
        Query(Alias("C"), Table(C, (Field("i"), Field("j")))),
        Query(Alias("D"), Table(D, (Field("i"), Field("j")))),
        Query(
            Alias("E"),
            MapJoin(
                Literal(operator.add),
                (
                    MapJoin(
                        Literal(operator.add),
                        (
                            Relabel(Alias("A"), (Field("i"), Field("j"))),
                            Relabel(Alias("B"), (Field("i"), Field("j"))),
                        ),
                    ),
                    MapJoin(
                        Literal(operator.add),
                        (
                            Relabel(Alias("C"), (Field("i"), Field("j"))),
                            Relabel(Alias("D"), (Field("i"), Field("j"))),
                        ),
                    ),
                ),
            ),
        ),
        Produces((Alias("E"),)),
    ))

    lowerer = EinsumLowerer()
    einsum_plan, parameters = lowerer(plan)
    interpreter = EinsumInterpreter(bindings=parameters)
    result = interpreter(einsum_plan)

    expected = A + B + C + D
    assert np.allclose(result, expected)


def test_non_commutative_order():
    """Test that non-commutative operations preserve order"""
    A = np.array([[4.0, 6.0], [8.0, 10.0]])
    B = np.array([[2.0, 2.0], [2.0, 2.0]])
    C = np.array([[1.0, 1.0], [1.0, 1.0]])

    # (A / B) / C should NOT be flattened
    plan = Plan((
        Query(Alias("A"), Table(A, (Field("i"), Field("j")))),
        Query(Alias("B"), Table(B, (Field("i"), Field("j")))),
        Query(Alias("C"), Table(C, (Field("i"), Field("j")))),
        Query(
            Alias("D"),
            MapJoin(
                Literal(operator.truediv),
                (
                    MapJoin(
                        Literal(operator.truediv),
                        (
                            Relabel(Alias("A"), (Field("i"), Field("j"))),
                            Relabel(Alias("B"), (Field("i"), Field("j"))),
                        ),
                    ),
                    Relabel(Alias("C"), (Field("i"), Field("j"))),
                ),
            ),
        ),
        Produces((Alias("D"),)),
    ))

    lowerer = EinsumLowerer()
    einsum_plan, parameters = lowerer(plan)
    interpreter = EinsumInterpreter(bindings=parameters)
    result = interpreter(einsum_plan)

    expected = (A / B) / C
    assert np.allclose(result, expected)


def test_nested_plan(rng):
    """Test lowering of nested plans"""
    A = rng.random((3, 3))
    B = rng.random((3, 3))

    inner_plan = Plan((
        Query(
            Alias("temp"),
            MapJoin(
                Literal(operator.add),
                (
                    Relabel(Alias("A"), (Field("i"), Field("j"))),
                    Relabel(Alias("B"), (Field("i"), Field("j"))),
                ),
            ),
        ),
        Produces((Alias("temp"),)),
    ))

    outer_plan = Plan((
        Query(Alias("A"), Table(A, (Field("i"), Field("j")))),
        Query(Alias("B"), Table(B, (Field("i"), Field("j")))),
        inner_plan,
    ))

    lowerer = EinsumLowerer()
    einsum_plan, parameters = lowerer(outer_plan)
    interpreter = EinsumInterpreter(bindings=parameters)
    result = interpreter(einsum_plan)

    expected = A + B
    assert np.allclose(result, expected)