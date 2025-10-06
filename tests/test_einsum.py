
import pytest

import numpy as np

import finchlite


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_basic_addition_with_transpose(rng):
    """Test basic addition with transpose"""
    A = rng.random((5, 5))
    B = rng.random((5, 5))

    C = finchlite.einop("C[i,j] = A[i,j] + B[j,i]", A=A, B=B)
    C_ref = A + B.T

    assert np.allclose(C, C_ref)


def test_matrix_multiplication(rng):
    """Test matrix multiplication using += (increment/accumulation)"""
    A = rng.random((3, 4))
    B = rng.random((4, 5))

    C = finchlite.einop("C[i,j] += A[i,k] * B[k,j]", A=A, B=B)
    C_ref = A @ B

    assert np.allclose(C, C_ref)


def test_element_wise_multiplication(rng):
    """Test element-wise multiplication"""
    A = rng.random((4, 4))
    B = rng.random((4, 4))

    C = finchlite.einop("C[i,j] = A[i,j] * B[i,j]", A=A, B=B)
    C_ref = A * B

    assert np.allclose(C, C_ref)


def test_sum_reduction(rng):
    """Test sum reduction using +="""
    A = rng.random((3, 4))

    C = finchlite.einop("C[i] += A[i,j]", A=A)
    C_ref = np.sum(A, axis=1)

    assert np.allclose(C, C_ref)


def test_maximum_reduction(rng):
    """Test maximum reduction using max="""
    A = rng.random((3, 4))

    C = finchlite.einop("C[i] max= A[i,j]", A=A)
    C_ref = np.max(A, axis=1)

    assert np.allclose(C, C_ref)


def test_outer_product(rng):
    """Test outer product"""
    A = rng.random(3)
    B = rng.random(4)

    C = finchlite.einop("C[i,j] = A[i] * B[j]", A=A, B=B)
    C_ref = np.outer(A, B)

    assert np.allclose(C, C_ref)


def test_batch_matrix_multiplication(rng):
    """Test batch matrix multiplication using +="""
    A = rng.random((2, 3, 4))
    B = rng.random((2, 4, 5))

    C = finchlite.einop("C[b,i,j] += A[b,i,k] * B[b,k,j]", A=A, B=B)
    C_ref = np.matmul(A, B)

    assert np.allclose(C, C_ref)


def test_minimum_reduction(rng):
    """Test minimum reduction using min="""
    A = rng.random((3, 4))

    C = finchlite.einop("C[i] min= A[i,j]", A=A)
    C_ref = np.min(A, axis=1)

    assert np.allclose(C, C_ref)


@pytest.mark.parametrize("axis", [(0, 2, 1), (3, 0, 1), (1, 0, 3, 2), (1, 0, 3, 2)])
@pytest.mark.parametrize(
    "idxs",
    [
        ("i", "j", "k", "l"),
        ("l", "j", "k", "i"),
        ("l", "k", "j", "i"),
    ],
)
def test_swizzle_in(rng, axis, idxs):
    """Test transpositions with einop"""
    A = rng.random((4, 4, 4, 4))

    jdxs = [idxs[p] for p in axis]
    xp_idxs = ", ".join(idxs)
    np_idxs = "".join(idxs)
    xp_jdxs = ", ".join(jdxs)
    np_jdxs = "".join(jdxs)

    C = finchlite.einop(f"C[{xp_jdxs}] += A[{xp_idxs}]", A=A)
    C_ref = np.einsum(f"{np_idxs}->{np_jdxs}", A)

    assert np.allclose(C, C_ref)


def test_operator_precedence_arithmetic(rng):
    """Test that arithmetic operator precedence follows Python rules"""
    A = rng.random((3, 3))
    B = rng.random((3, 3))
    C = rng.random((3, 3))

    # Test: A + B * C should be A + (B * C), not (A + B) * C
    result = finchlite.einop("D[i,j] = A[i,j] + B[i,j] * C[i,j]", A=A, B=B, C=C)
    expected = A + (B * C)

    assert np.allclose(result, expected)


def test_operator_precedence_power_and_multiplication(rng):
    """Test that power has higher precedence than multiplication"""
    A = rng.random((3, 3)) + 1  # Add 1 to avoid numerical issues with powers

    # Test: A * A ** 2 should be A * (A ** 2), not (A * A) ** 2
    result = finchlite.einop("B[i,j] = A[i,j] * A[i,j] ** 2", A=A)
    expected = A * (A**2)

    assert np.allclose(result, expected)


def test_operator_precedence_addition_and_multiplication(rng):
    """Test complex arithmetic precedence: A + B * C ** 2"""
    A = rng.random((3, 3))
    B = rng.random((3, 3))
    C = rng.random((3, 3)) + 1  # Add 1 to avoid numerical issues

    # Test: A + B * C ** 2 should be A + (B * (C ** 2))
    result = finchlite.einop("D[i,j] = A[i,j] + B[i,j] * C[i,j] ** 2", A=A, B=B, C=C)
    expected = A + (B * (C**2))

    assert np.allclose(result, expected)


def test_operator_precedence_logical_and_or(rng):
    """Test that 'and' has higher precedence than 'or'"""
    A = (rng.random((3, 3)) > 0.3).astype(float)  # Boolean-like arrays
    B = (rng.random((3, 3)) > 0.3).astype(float)
    C = (rng.random((3, 3)) > 0.3).astype(float)

    # Test: A or B and C should be A or (B and C), not (A or B) and C
    result = finchlite.einop("D[i,j] = A[i,j] or B[i,j] and C[i,j]", A=A, B=B, C=C)
    expected = np.logical_or(A, np.logical_and(B, C)).astype(float)

    assert np.allclose(result, expected)


def test_operator_precedence_bitwise_operations(rng):
    """Test bitwise operator precedence.

    | has lower precedence than ^ which has lower than &
    """
    # Use integer arrays for bitwise operations
    A = rng.integers(0, 8, size=(3, 3))
    B = rng.integers(0, 8, size=(3, 3))
    C = rng.integers(0, 8, size=(3, 3))
    D = rng.integers(0, 8, size=(3, 3))

    # Test: A | B ^ C & D should be A | (B ^ (C & D))
    result = finchlite.einop("E[i,j] = A[i,j] | B[i,j] ^ C[i,j] & D[i,j]", A=A, B=B, C=C, D=D)
    expected = A | (B ^ (C & D))

    assert np.allclose(result, expected)


def test_operator_precedence_shift_operations(rng):
    """Test shift operator precedence with arithmetic"""
    # Use small integer arrays to avoid overflow in shifts
    A = rng.integers(1, 4, size=(3, 3))

    # Test: A << 1 + 1 should be A << (1 + 1), not (A << 1) + 1
    # Since shift has lower precedence than addition
    result = finchlite.einop("B[i,j] = A[i,j] << 1 + 1", A=A)
    expected = A << (1 + 1)  # A << 2

    assert np.allclose(result, expected)


def test_operator_precedence_comparison_with_arithmetic(rng):
    """Test that arithmetic has higher precedence than comparison"""
    A = rng.random((3, 3))
    B = rng.random((3, 3))
    C = rng.random((3, 3))

    # Test: A + B == C should be (A + B) == C, not A + (B == C)
    result = finchlite.einop("D[i,j] = A[i,j] + B[i,j] == C[i,j]", A=A, B=B, C=C)
    expected = ((A + B) == C).astype(float)

    assert np.allclose(result, expected)


def test_operator_precedence_with_parentheses(rng):
    """Test that parentheses override operator precedence"""
    A = rng.random((3, 3))
    B = rng.random((3, 3))
    C = rng.random((3, 3))

    # Test: (A + B) * C should be different from A + B * C
    result_with_parens = finchlite.einop("D[i,j] = (A[i,j] + B[i,j]) * C[i,j]", A=A, B=B, C=C)
    result_without_parens = finchlite.einop(
        "E[i,j] = A[i,j] + B[i,j] * C[i,j]", A=A, B=B, C=C
    )

    expected_with_parens = (A + B) * C
    expected_without_parens = A + (B * C)

    assert np.allclose(result_with_parens, expected_with_parens)
    assert np.allclose(result_without_parens, expected_without_parens)

    # Verify they're different (unless by coincidence)
    if not np.allclose(expected_with_parens, expected_without_parens):
        assert not np.allclose(result_with_parens, result_without_parens)


def test_operator_precedence_unary_operators(rng):
    """Test unary operator precedence"""
    A = rng.random((3, 3)) - 0.5  # Some negative values

    # Test: -A ** 2 should be -(A ** 2), not (-A) ** 2
    result = finchlite.einop("B[i,j] = -A[i,j] ** 2", A=A)
    expected = -(A**2)

    assert np.allclose(result, expected)


def test_numeric_literals(rng):
    """Test that numeric literals work correctly"""
    A = rng.random((3, 3))

    # Test simple addition with literal
    result = finchlite.einop("B[i,j] = A[i,j] + 1", A=A)
    expected = A + 1

    assert np.allclose(result, expected)

    # Test complex expression with literals
    result2 = finchlite.einop("C[i,j] = A[i,j] * 2 + 3", A=A)
    expected2 = A * 2 + 3

    assert np.allclose(result2, expected2)


def test_comparison_chaining(rng):
    """Test that comparison chaining works like Python.

    a < b < c becomes (a < b) and (b < c)
    """
    A = rng.random((3, 3)) * 10  # Scale to get variety in comparisons
    B = rng.random((3, 3)) * 10
    C = rng.random((3, 3)) * 10

    # Test: A < B < C should be (A < B) and (B < C), not (A < B) < C
    result = finchlite.einop("D[i,j] = A[i,j] < B[i,j] < C[i,j]", A=A, B=B, C=C)
    expected = np.logical_and(A < B, B < C).astype(float)

    assert np.allclose(result, expected)


def test_comparison_chaining_three_way(rng):
    """Test three-way comparison chaining with different operators"""
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[2, 3], [4, 5]])
    C = np.array([[3, 4], [5, 6]])

    # Test: A <= B < C should be (A <= B) and (B < C)
    result = finchlite.einop("D[i,j] = A[i,j] <= B[i,j] < C[i,j]", A=A, B=B, C=C)
    expected = np.logical_and(A <= B, B < C).astype(float)

    assert np.allclose(result, expected)


def test_comparison_chaining_four_way(rng):
    """Test four-way comparison chaining"""
    A = np.array([[1]])
    B = np.array([[2]])
    C = np.array([[3]])
    D = np.array([[4]])

    # Test: A < B < C < D should be ((A < B) and (B < C)) and (C < D)
    result = finchlite.einop("E[i,j] = A[i,j] < B[i,j] < C[i,j] < D[i,j]", A=A, B=B, C=C, D=D)
    expected = np.logical_and(np.logical_and(A < B, B < C), C < D).astype(float)

    assert np.allclose(result, expected)


def test_single_comparison_vs_chained(rng):
    """Test that single comparison and chained comparison work differently"""
    A = np.array([[2]])
    B = np.array([[3]])
    C = np.array([[1]])  # Intentionally make C < A to show difference

    # Single comparison: A < B should be True
    result_single = finchlite.einop("D[i,j] = A[i,j] < B[i,j]", A=A, B=B)
    expected_single = (A < B).astype(float)

    # Chained comparison: A < B < C should be (A < B) and (B < C)
    # = True and False = False
    result_chained = finchlite.einop("E[i,j] = A[i,j] < B[i,j] < C[i,j]", A=A, B=B, C=C)
    expected_chained = np.logical_and(A < B, B < C).astype(float)

    assert np.allclose(result_single, expected_single)
    assert np.allclose(result_chained, expected_chained)

    # Verify they're different
    assert not np.allclose(result_single, result_chained)


def test_alphanumeric_tensor_names(rng):
    """Test that tensor names with numbers work correctly"""
    A1 = rng.random((2, 2))
    B2 = rng.random((2, 2))
    C3_test = rng.random((2, 2))

    # Test basic arithmetic with alphanumeric names
    result = finchlite.einop(
        "result_1[i,j] = A1[i,j] + B2[i,j] * C3_test[i,j]",
        A1=A1,
        B2=B2,
        C3_test=C3_test,
    )
    expected = A1 + (B2 * C3_test)

    assert np.allclose(result, expected)

    # Test comparison chaining with alphanumeric names
    X1 = np.array([[1, 2]])
    Y2 = np.array([[3, 4]])
    Z3 = np.array([[5, 6]])

    result2 = finchlite.einop(
        "chain_result[i,j] = X1[i,j] < Y2[i,j] < Z3[i,j]", X1=X1, Y2=Y2, Z3=Z3
    )
    expected2 = np.logical_and(X1 < Y2, Y2 < Z3).astype(float)

    assert np.allclose(result2, expected2)


def test_bool_literals(rng):
    """Test that boolean literals work correctly"""
    A = rng.random((2, 2))

    # Test True literal
    result_true = finchlite.einop("B[i,j] = A[i,j] and True", A=A)
    expected_true = np.logical_and(A, True).astype(float)
    assert np.allclose(result_true, expected_true)

    # Test False literal
    result_false = finchlite.einop("C[i,j] = A[i,j] or False", A=A)
    expected_false = np.logical_or(A, False).astype(float)
    assert np.allclose(result_false, expected_false)

    # Test boolean operations with literals
    A_bool = rng.random((2, 2)) > 0.5
    result_and = finchlite.einop("D[i,j] = A_bool[i,j] and True and False", A_bool=A_bool)
    expected_and = np.logical_and(np.logical_and(A_bool, True), False)
    assert np.allclose(result_and, expected_and)


def test_int_literals(rng):
    """Test that integer literals work correctly"""
    A = rng.random((2, 2))

    # Test positive integer
    result_pos = finchlite.einop("B[i,j] = A[i,j] + 42", A=A)
    expected_pos = A + 42
    assert np.allclose(result_pos, expected_pos)

    # Test negative integer
    result_neg = finchlite.einop("C[i,j] = A[i,j] * -5", A=A)
    expected_neg = A * (-5)
    assert np.allclose(result_neg, expected_neg)

    # Test zero
    result_zero = finchlite.einop("D[i,j] = A[i,j] + 0", A=A)
    expected_zero = A + 0
    assert np.allclose(result_zero, expected_zero)

    # Test large integer
    result_large = finchlite.einop("E[i,j] = A[i,j] + 123456789", A=A)
    expected_large = A + 123456789
    assert np.allclose(result_large, expected_large)


def test_float_literals(rng):
    """Test that float literals work correctly"""
    A = rng.random((2, 2))

    # Test positive float
    result_pos = finchlite.einop("B[i,j] = A[i,j] + 3.14159", A=A)
    expected_pos = A + 3.14159
    assert np.allclose(result_pos, expected_pos)

    # Test negative float
    result_neg = finchlite.einop("C[i,j] = A[i,j] * -2.71828", A=A)
    expected_neg = A * (-2.71828)
    assert np.allclose(result_neg, expected_neg)

    # Test scientific notation
    result_sci = finchlite.einop("D[i,j] = A[i,j] + 1.5e-3", A=A)
    expected_sci = A + 1.5e-3
    assert np.allclose(result_sci, expected_sci)

    # Test very small float
    result_small = finchlite.einop("E[i,j] = A[i,j] + 0.000001", A=A)
    expected_small = A + 0.000001
    assert np.allclose(result_small, expected_small)


def test_complex_literals(rng):
    """Test that complex literals work correctly"""
    A = rng.random((2, 2)).astype(complex)  # Use complex arrays

    # Test complex with real and imaginary parts
    result_complex = finchlite.einop("B[i,j] = A[i,j] + (3+4j)", A=A)
    expected_complex = A + (3 + 4j)
    assert np.allclose(result_complex, expected_complex)

    # Test pure imaginary
    result_imag = finchlite.einop("C[i,j] = A[i,j] * 2j", A=A)
    expected_imag = A * 2j
    assert np.allclose(result_imag, expected_imag)

    # Test complex with negative parts
    result_neg = finchlite.einop("D[i,j] = A[i,j] + (-1-2j)", A=A)
    expected_neg = A + (-1 - 2j)
    assert np.allclose(result_neg, expected_neg)


def test_mixed_literal_types(rng):
    """Test expressions mixing different literal types"""
    A = rng.random((2, 2))

    # Test int + float
    result_int_float = finchlite.einop("B[i,j] = A[i,j] + 5 + 3.14", A=A)
    expected_int_float = A + 5 + 3.14
    assert np.allclose(result_int_float, expected_int_float)

    # Test operator precedence with literals
    result_precedence = finchlite.einop("C[i,j] = A[i,j] + 2 * 3", A=A)
    expected_precedence = A + (2 * 3)  # Should be A + 6, not (A + 2) * 3
    assert np.allclose(result_precedence, expected_precedence)

    # Test power with literals
    result_power = finchlite.einop("D[i,j] = A[i,j] + 2 ** 3", A=A)
    expected_power = A + (2**3)  # Should be A + 8
    assert np.allclose(result_power, expected_power)


def test_literal_edge_cases(rng):
    """Test edge cases with literals"""
    A = rng.random((2, 2))

    # Test multiple literals in sequence
    result_multi = finchlite.einop("B[i,j] = A[i,j] + 1 + 2 + 3", A=A)
    expected_multi = A + 1 + 2 + 3  # Should be A + 6
    assert np.allclose(result_multi, expected_multi)

    # Test literals in comparisons
    result_comp = finchlite.einop("C[i,j] = A[i,j] > 0.5", A=A)
    expected_comp = (A > 0.5).astype(float)
    assert np.allclose(result_comp, expected_comp)

    # Test literals with parentheses
    result_parens = finchlite.einop("D[i,j] = A[i,j] * (2 + 3)", A=A)
    expected_parens = A * (2 + 3)  # Should be A * 5
    assert np.allclose(result_parens, expected_parens)
