import operator
from typing import Any

import pytest

import numpy as np

import finchlite
import finchlite.finch_einsum as ein
from finchlite.algebra import overwrite
from finchlite.tensor import SparseTensor


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_pass_through(rng):
    """Test pass through of a tensor"""
    A = rng.random((5, 5))

    B = finchlite.einop("B[i,j] = A[i,j]", A=A)

    assert np.allclose(B, A)


def test_transpose(rng):
    """Test basic addition with transpose"""
    A = rng.random((5, 5))

    B = finchlite.einop("B[i,j] = A[j, i]", A=A)
    B_ref = A.T

    assert np.allclose(B, B_ref)


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
    result = finchlite.einop(
        "E[i,j] = A[i,j] | B[i,j] ^ C[i,j] & D[i,j]", A=A, B=B, C=C, D=D
    )
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
    result_with_parens = finchlite.einop(
        "D[i,j] = (A[i,j] + B[i,j]) * C[i,j]", A=A, B=B, C=C
    )
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
    result = finchlite.einop(
        "E[i,j] = A[i,j] < B[i,j] < C[i,j] < D[i,j]", A=A, B=B, C=C, D=D
    )
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
    result_and = finchlite.einop(
        "D[i,j] = A_bool[i,j] and True and False", A_bool=A_bool
    )
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


# =============================================================================
# Einsum tests comparing finchlite.einsum to np.einsum
# =============================================================================


class TestEinsumImplicitMode:
    """Test einsum in implicit mode (no -> in subscripts)"""

    @pytest.mark.skip(reason="Repeated indices not yet supported")
    def test_trace(self, rng):
        """Test trace of a matrix"""
        A = rng.random((5, 5))

        result = finchlite.einsum("ii", A)
        expected = np.einsum("ii", A)

        assert np.allclose(result, expected)
        assert np.allclose(result, np.trace(A))

    def test_element_wise_multiplication(self, rng):
        """Test element-wise multiplication"""
        A = rng.random((4, 3))
        B = rng.random((4, 3))

        result = finchlite.einsum("ij,ij", A, B)
        expected = np.einsum("ij,ij", A, B)

        assert np.allclose(result, expected)
        assert np.allclose(result, np.sum(A * B))

    def test_matrix_multiplication(self, rng):
        """Test matrix multiplication"""
        A = rng.random((3, 4))
        B = rng.random((4, 5))

        result = finchlite.einsum("ij,jk", A, B)
        expected = np.einsum("ij,jk", A, B)

        assert np.allclose(result, expected)
        assert np.allclose(result, A @ B)

    def test_transpose(self, rng):
        """Test transpose via einsum"""
        A = rng.random((3, 4))

        result = finchlite.einsum("ji", A)
        expected = np.einsum("ji", A)

        assert np.allclose(result, expected)
        assert np.allclose(result, A.T)

    def test_vector_inner_product(self, rng):
        """Test vector inner product"""
        a = rng.random(5)
        b = rng.random(5)

        result = finchlite.einsum("i,i", a, b)
        expected = np.einsum("i,i", a, b)

        assert np.allclose(result, expected)
        assert np.allclose(result, np.inner(a, b))

    def test_vector_outer_product(self, rng):
        """Test vector outer product"""
        a = rng.random(3)
        b = rng.random(4)

        result = finchlite.einsum("i,j", a, b)
        expected = np.einsum("i,j", a, b)

        assert np.allclose(result, expected)
        assert np.allclose(result, np.outer(a, b))

    def test_tensor_contraction(self, rng):
        """Test tensor contraction"""
        A = rng.random((3, 4, 5))
        B = rng.random((4, 3, 2))

        result = finchlite.einsum("ijk,jil", A, B)
        expected = np.einsum("ijk,jil", A, B)

        assert np.allclose(result, expected)

    def test_bilinear_transformation(self, rng):
        """Test bilinear transformation"""
        A = rng.random((3, 4))
        B = rng.random((4, 5, 6))
        C = rng.random((6, 7))

        result = finchlite.einsum("ij,jkl,lm", A, B, C)
        expected = np.einsum("ij,jkl,lm", A, B, C)

        assert np.allclose(result, expected)


class TestEinsumExplicitMode:
    """Test einsum in explicit mode (with -> in subscripts)"""

    @pytest.mark.skip(reason="Repeated indices not yet supported")
    def test_extract_diagonal(self, rng):
        """Test extracting diagonal"""
        A = rng.random((5, 5))

        result = finchlite.einsum("ii->i", A)
        expected = np.einsum("ii->i", A)

        assert np.allclose(result, expected)
        assert np.allclose(result, np.diag(A))

    def test_sum_over_axis(self, rng):
        """Test sum over specific axis"""
        A = rng.random((4, 5))

        # Sum over axis 1
        result = finchlite.einsum("ij->i", A)
        expected = np.einsum("ij->i", A)

        assert np.allclose(result, expected)
        assert np.allclose(result, np.sum(A, axis=1))

        # Sum over axis 0
        result2 = finchlite.einsum("ij->j", A)
        expected2 = np.einsum("ij->j", A)

        assert np.allclose(result2, expected2)
        assert np.allclose(result2, np.sum(A, axis=0))

    def test_total_sum(self, rng):
        """Test total sum"""
        A = rng.random((3, 4))

        result = finchlite.einsum("ij->", A)
        expected = np.einsum("ij->", A)

        assert np.allclose(result, expected)
        assert np.allclose(result, np.sum(A))

    def test_transpose_explicit(self, rng):
        """Test transpose with explicit output"""
        A = rng.random((3, 4))

        result = finchlite.einsum("ij->ji", A)
        expected = np.einsum("ij->ji", A)

        assert np.allclose(result, expected)
        assert np.allclose(result, A.T)

    def test_matrix_vector_explicit(self, rng):
        """Test matrix-vector multiplication with explicit output"""
        A = rng.random((3, 4))
        b = rng.random(4)

        result = finchlite.einsum("ij,j->i", A, b)
        expected = np.einsum("ij,j->i", A, b)

        assert np.allclose(result, expected)
        assert np.allclose(result, A @ b)

    def test_custom_contraction(self, rng):
        """Test custom tensor contraction with explicit output"""
        A = rng.random((2, 3, 4))
        B = rng.random((4, 5))

        result = finchlite.einsum("ijk,kl->ijl", A, B)
        expected = np.einsum("ijk,kl->ijl", A, B)

        assert np.allclose(result, expected)

    def test_reorder_axes(self, rng):
        """Test reordering axes with explicit output"""
        A = rng.random((2, 3, 4, 5))

        result = finchlite.einsum("ijkl->ljik", A)
        expected = np.einsum("ijkl->ljik", A)

        assert np.allclose(result, expected)


class TestEinsumVariableOrdering:
    """Test different variable orderings in einsum"""

    def test_alphabetical_ordering_implicit(self, rng):
        """Test that implicit mode respects alphabetical ordering"""
        A = rng.random((3, 4))
        B = rng.random((4, 5))

        # Standard order
        result1 = finchlite.einsum("ij,jk", A, B)
        expected1 = np.einsum("ij,jk", A, B)
        assert np.allclose(result1, expected1)

        # Different variable names but same meaning
        result2 = finchlite.einsum("ab,bc", A, B)
        expected2 = np.einsum("ab,bc", A, B)
        assert np.allclose(result2, expected2)
        assert np.allclose(result1, result2)

    def test_non_alphabetical_ordering(self, rng):
        """Test non-alphabetical variable ordering"""
        A = rng.random((3, 4))
        B = rng.random((4, 5))

        # Non-alphabetical order should transpose result in implicit mode
        result = finchlite.einsum("ij,jl", A, B)
        expected = np.einsum("ij,jl", A, B)

        assert np.allclose(result, expected)
        # This should be different from alphabetical ij,jk due to l < k

    def test_variable_ordering_explicit_override(self, rng):
        """Test that explicit mode overrides alphabetical ordering"""
        A = rng.random((3, 4))
        B = rng.random((4, 5))

        # Force specific output order
        result = finchlite.einsum("ij,jk->ki", A, B)
        expected = np.einsum("ij,jk->ki", A, B)

        assert np.allclose(result, expected)
        # This should be transpose of standard matrix multiplication
        assert np.allclose(result, (A @ B).T)

    @pytest.mark.skip(reason="Repeated indices not yet supported")
    def test_repeated_indices(self, rng):
        """Test repeated indices for diagonal operations"""
        A = rng.random((4, 4, 4))

        # Repeated index in same tensor
        result = finchlite.einsum("iji", A)
        expected = np.einsum("iji", A)

        assert np.allclose(result, expected)

    def test_mixed_variable_types(self, rng):
        """Test mixing different variable names"""
        A = rng.random((3, 4))
        B = rng.random((4, 5))

        # Mix letters from different parts of alphabet
        result = finchlite.einsum("az,zb", A, B)
        expected = np.einsum("az,zb", A, B)

        assert np.allclose(result, expected)


class TestEinsumSpecialSyntax:
    """Test einsum special syntax: einsum(op1, op1inds, op2, op2inds, ...)"""

    def test_alternative_syntax_matrix_mult(self, rng):
        """Test alternative syntax for matrix multiplication"""
        A = rng.random((3, 4))
        B = rng.random((4, 5))

        # Standard syntax
        result1 = finchlite.einsum("ij,jk", A, B)

        # Alternative syntax
        result2 = finchlite.einsum(A, [0, 1], B, [1, 2])

        expected = np.einsum(A, [0, 1], B, [1, 2])

        assert np.allclose(result1, result2)
        assert np.allclose(result2, expected)

    def test_alternative_syntax_with_output(self, rng):
        """Test alternative syntax with explicit output specification"""
        A = rng.random((3, 4))
        B = rng.random((4, 5))

        # Explicit output indices
        result = finchlite.einsum(A, [0, 1], B, [1, 2], [0, 2])
        expected = np.einsum(A, [0, 1], B, [1, 2], [0, 2])

        assert np.allclose(result, expected)
        assert np.allclose(result, A @ B)

    def test_alternative_syntax_transpose(self, rng):
        """Test alternative syntax for transpose"""
        A = rng.random((3, 4))

        # Implicit transpose
        result1 = finchlite.einsum(A, [1, 0])
        expected1 = np.einsum(A, [1, 0])

        assert np.allclose(result1, expected1)
        assert np.allclose(result1, A.T)

        # Explicit transpose
        result2 = finchlite.einsum(A, [0, 1], [1, 0])
        expected2 = np.einsum(A, [0, 1], [1, 0])

        assert np.allclose(result2, expected2)
        assert np.allclose(result2, A.T)

    @pytest.mark.skip(reason="Repeated indices not yet supported")
    def test_alternative_syntax_trace(self, rng):
        """Test alternative syntax for trace"""
        A = rng.random((5, 5))

        # Trace using alternative syntax
        result = finchlite.einsum(A, [0, 0])
        expected = np.einsum(A, [0, 0])

        assert np.allclose(result, expected)
        assert np.allclose(result, np.trace(A))

    @pytest.mark.skip(reason="Repeated indices not yet supported")
    def test_alternative_syntax_diagonal(self, rng):
        """Test alternative syntax for diagonal extraction"""
        A = rng.random((5, 5))

        result = finchlite.einsum(A, [0, 0], [0])
        expected = np.einsum(A, [0, 0], [0])

        assert np.allclose(result, expected)
        assert np.allclose(result, np.diag(A))

    def test_alternative_syntax_sum(self, rng):
        """Test alternative syntax for sum operations"""
        A = rng.random((3, 4))

        # Sum over axis 1
        result1 = finchlite.einsum(A, [0, 1], [0])
        expected1 = np.einsum(A, [0, 1], [0])

        assert np.allclose(result1, expected1)
        assert np.allclose(result1, np.sum(A, axis=1))

        # Total sum
        result2 = finchlite.einsum(A, [0, 1], [])
        expected2 = np.einsum(A, [0, 1], [])

        assert np.allclose(result2, expected2)
        assert np.allclose(result2, np.sum(A))

    def test_alternative_syntax_outer_product(self, rng):
        """Test alternative syntax for outer product"""
        a = rng.random(3)
        b = rng.random(4)

        result = finchlite.einsum(a, [0], b, [1])
        expected = np.einsum(a, [0], b, [1])

        assert np.allclose(result, expected)
        assert np.allclose(result, np.outer(a, b))

    def test_alternative_syntax_complex_contraction(self, rng):
        """Test alternative syntax for complex tensor contraction"""
        A = rng.random((2, 3, 4))
        B = rng.random((4, 5, 6))
        C = rng.random((6, 2))

        result = finchlite.einsum(A, [0, 1, 2], B, [2, 3, 4], C, [4, 0], [1, 3])
        expected = np.einsum(A, [0, 1, 2], B, [2, 3, 4], C, [4, 0], [1, 3])

        assert np.allclose(result, expected)


class TestEinsumEdgeCases:
    """Test edge cases and special scenarios"""

    def test_single_tensor_operations(self, rng):
        """Test operations on single tensors"""
        A = rng.random((3, 4, 5))

        # Identity operation
        result1 = finchlite.einsum("ijk", A)
        expected1 = np.einsum("ijk", A)
        assert np.allclose(result1, expected1)
        assert np.allclose(result1, A)

        # Permute dimensions
        result2 = finchlite.einsum("ikj", A)
        expected2 = np.einsum("ikj", A)
        assert np.allclose(result2, expected2)

    def test_scalar_operations(self, rng):
        """Test operations involving scalars"""
        scalar = rng.random()
        A = rng.random((3, 4))

        # Scalar multiplication (broadcasting)
        result = finchlite.einsum(",ij", scalar, A)
        expected = np.einsum(",ij", scalar, A)

        assert np.allclose(result, expected)
        assert np.allclose(result, scalar * A)

    def test_empty_dimensions(self, rng):
        """Test with empty dimensions"""
        A = rng.random((0, 3))
        B = rng.random((3, 4))

        result = finchlite.einsum("ij,jk", A, B)
        expected = np.einsum("ij,jk", A, B)

        assert np.allclose(result, expected)
        assert result.shape == (0, 4)

    def test_1d_tensors(self, rng):
        """Test operations on 1D tensors"""
        a = rng.random(5)
        b = rng.random(5)

        # Element-wise product and sum
        result = finchlite.einsum("i,i", a, b)
        expected = np.einsum("i,i", a, b)

        assert np.allclose(result, expected)
        assert np.isscalar(result) or result.shape == ()

    def test_high_dimensional(self, rng):
        """Test high-dimensional tensors"""
        A = rng.random((2, 2, 2, 2, 2))
        B = rng.random((2, 2, 2, 2, 2))

        result = finchlite.einsum("abcde,abcde", A, B)
        expected = np.einsum("abcde,abcde", A, B)

        assert np.allclose(result, expected)


class TestEinsumEllipses:
    """Test einsum with ellipses (...) notation"""

    def test_basic_ellipses(self, rng):
        """Test basic ellipses usage for identity operations"""
        A = rng.random((3, 4, 5))

        # Identity with ellipses
        result = finchlite.einsum("...", A)
        expected = np.einsum("...", A)

        assert np.allclose(result, expected)
        assert result.shape == expected.shape

    def test_ellipses_with_named_indices(self, rng):
        """Test ellipses combined with named indices"""
        A = rng.random((2, 3, 4, 5))

        # Sum over last dimension, keeping others
        result = finchlite.einsum("...i->...", A)
        expected = np.einsum("...i->...", A)

        assert np.allclose(result, expected)
        assert result.shape == expected.shape

    def test_ellipses_transpose(self, rng):
        """Test ellipses with transpose operations"""
        A = rng.random((2, 3, 4, 5))

        # Transpose last two dimensions
        result = finchlite.einsum("...ij->...ji", A)
        expected = np.einsum("...ij->...ji", A)

        assert np.allclose(result, expected)
        assert result.shape == expected.shape

    def test_ellipses_matrix_multiply(self, rng):
        """Test batch matrix multiplication with ellipses"""
        A = rng.random((2, 3, 4, 5))
        B = rng.random((2, 3, 5, 6))

        # Batch matrix multiplication
        result = finchlite.einsum("...ij,...jk->...ik", A, B)
        expected = np.einsum("...ij,...jk->...ik", A, B)

        assert np.allclose(result, expected)
        assert result.shape == expected.shape

    def test_ellipses_different_batch_dims(self, rng):
        """Test ellipses with different numbers of batch dimensions"""
        A = rng.random((2, 3, 4))  # 1 batch dim + 2x4 matrix
        B = rng.random((5, 2, 4, 6))  # 2 batch dims + 4x6 matrix

        # Broadcasting should work
        result = finchlite.einsum("...ij,...jk->...ik", A, B)
        expected = np.einsum("...ij,...jk->...ik", A, B)

        assert np.allclose(result, expected)
        assert result.shape == expected.shape

    @pytest.mark.skip(reason="Repeated indices not yet supported")
    def test_ellipses_trace(self, rng):
        """Test computing trace with ellipses"""
        A = rng.random((2, 3, 4, 4))

        # Trace of last two dimensions for each batch
        result = finchlite.einsum("...ii->...", A)
        expected = np.einsum("...ii->...", A)

        assert np.allclose(result, expected)
        assert result.shape == expected.shape

    @pytest.mark.skip(reason="Repeated indices not yet supported")
    def test_ellipses_diagonal(self, rng):
        """Test extracting diagonal with ellipses"""
        A = rng.random((2, 3, 4, 4))

        # Extract diagonal of last two dimensions
        result = finchlite.einsum("...ii->...i", A)
        expected = np.einsum("...ii->...i", A)

        assert np.allclose(result, expected)
        assert result.shape == expected.shape

    def test_ellipses_element_wise(self, rng):
        """Test element-wise operations with ellipses"""
        A = rng.random((2, 3, 4, 5))
        B = rng.random((2, 3, 4, 5))

        # Element-wise multiplication
        result = finchlite.einsum("...,...->...", A, B)
        expected = np.einsum("...,...->...", A, B)

        assert np.allclose(result, expected)
        assert result.shape == expected.shape

    def test_ellipses_sum_product(self, rng):
        """Test sum of element-wise product with ellipses"""
        A = rng.random((2, 3, 4, 5))
        B = rng.random((2, 3, 4, 5))

        # Sum of element-wise product
        result = finchlite.einsum("...,...", A, B)
        expected = np.einsum("...,...", A, B)

        assert np.allclose(result, expected)

    def test_ellipses_outer_product(self, rng):
        """Test outer product with ellipses"""
        A = rng.random((2, 3))
        B = rng.random((2, 5))

        # Outer product with batch dimensions
        result = finchlite.einsum("...i,...j->...ij", A, B)
        expected = np.einsum("...i,...j->...ij", A, B)

        assert np.allclose(result, expected)
        assert result.shape == expected.shape

    def test_ellipses_multiple_contractions(self, rng):
        """Test multiple contractions with ellipses"""
        A = rng.random((2, 3, 4, 5))
        B = rng.random((2, 3, 5, 6))
        C = rng.random((2, 3, 6, 7))

        # Chain of matrix multiplications
        result = finchlite.einsum("...ij,...jk,...kl->...il", A, B, C)
        expected = np.einsum("...ij,...jk,...kl->...il", A, B, C)

        assert np.allclose(result, expected)
        assert result.shape == expected.shape

    def test_ellipses_broadcasting_edge_cases(self, rng):
        """Test edge cases with broadcasting and ellipses"""
        # Test with single dimension arrays
        A = rng.random((1, 3, 4))
        B = rng.random((2, 1, 4, 5))

        result = finchlite.einsum("...ij,...jk->...ik", A, B)
        expected = np.einsum("...ij,...jk->...ik", A, B)

        assert np.allclose(result, expected)
        assert result.shape == expected.shape

    def test_ellipses_with_scalars(self, rng):
        """Test ellipses operations with scalar inputs"""
        A = rng.random((2, 3, 4))
        scalar = 2.5

        # Multiply tensor by scalar using ellipses
        result = finchlite.einsum("...,...->...", A, scalar)
        expected = np.einsum("...,...->...", A, scalar)

        assert np.allclose(result, expected)
        assert result.shape == expected.shape

    def test_ellipses_reduction_patterns(self, rng):
        """Test various reduction patterns with ellipses"""
        A = rng.random((2, 3, 4, 5, 6))

        # Sum over last dimension
        result1 = finchlite.einsum("...i->...", A)
        expected1 = np.einsum("...i->...", A)
        assert np.allclose(result1, expected1)

        # Sum over last two dimensions
        result2 = finchlite.einsum("...ij->...", A)
        expected2 = np.einsum("...ij->...", A)
        assert np.allclose(result2, expected2)

        # Sum over specific dimensions while keeping others
        result3 = finchlite.einsum("...ijk->...ik", A)
        expected3 = np.einsum("...ijk->...ik", A)
        assert np.allclose(result3, expected3)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
class TestEinsumDataTypes:
    """Test einsum with different data types"""

    def test_matrix_multiplication_dtypes(self, rng, dtype):
        """Test matrix multiplication with different dtypes"""
        A = rng.random((3, 4)).astype(dtype)
        B = rng.random((4, 5)).astype(dtype)

        result = finchlite.einsum("ij,jk", A, B)
        expected = np.einsum("ij,jk", A, B)

        assert np.allclose(result, expected)
        # Check dtype preservation (may depend on implementation)

    def test_complex_operations(self, rng, dtype):
        """Test operations with complex numbers"""
        if not np.issubdtype(dtype, np.complexfloating):
            pytest.skip("Test only for complex dtypes")

        A = (rng.random((3, 3)) + 1j * rng.random((3, 3))).astype(dtype)

        result = finchlite.einsum("ij", A)
        expected = np.einsum("ij", A)

        assert np.allclose(result, expected)


class TestEinsumIndirectAccess:
    """Test einsum with indirect access"""

    def run_einsum_plan(
        self, prgm: ein.Plan, bindings: dict[str, Any], expected: np.ndarray
    ):
        """Runs an einsum plan and returns the result"""
        interpreter = ein.EinsumInterpreter(bindings=bindings)
        result = interpreter(prgm)[0]

        import sys

        import numpy as np

        np.set_printoptions(threshold=sys.maxsize)

        assert np.allclose(result, expected)

    def test_indirect_elementwise_multiplication(self, rng):
        """Test indirect elementwise multiplication but no indirect assignment"""

        A = rng.random((5, 5))
        B = rng.random((5, 5))

        sparse_A = SparseTensor.from_dense_tensor(A)

        # A is sparse
        # C[i] = AElems[i] * B[ACoords[i]]
        prgm = ein.Plan(
            (
                ein.Einsum(
                    op=ein.Literal(overwrite),
                    tns=ein.Alias("C"),
                    idxs=(ein.Index("i"),),
                    arg=ein.Call(
                        op=ein.Literal(operator.mul),
                        args=(
                            ein.Access(
                                tns=ein.GetAttr(
                                    obj=ein.Alias("A"),
                                    attr=ein.Literal("elems"),
                                    dim=None,
                                ),
                                idxs=(ein.Index("i"),),
                            ),
                            ein.Access(
                                tns=ein.Alias("B"),
                                idxs=(
                                    ein.Access(
                                        tns=ein.GetAttr(
                                            obj=ein.Alias("A"),
                                            attr=ein.Literal("coords"),
                                            dim=None,
                                        ),
                                        idxs=(ein.Index("i"),),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
                ein.Produces((ein.Alias("C"),)),
            )
        )

        result = finchlite.multiply(A, B).flatten()
        self.run_einsum_plan(prgm, {"A": sparse_A, "B": B}, result)

    def test_indirect_elementwise_addition(self, rng):
        """Test indirect elementwise addition"""

        A = rng.random((4, 4))
        B = rng.random((4, 4))

        sparse_A = SparseTensor.from_dense_tensor(A)

        # C[i] = AElems[i] + B[ACoords[i]]
        prgm = ein.Plan(
            (
                ein.Einsum(
                    op=ein.Literal(overwrite),
                    tns=ein.Alias("C"),
                    idxs=(ein.Index("i"),),
                    arg=ein.Call(
                        op=ein.Literal(operator.add),
                        args=(
                            ein.Access(
                                tns=ein.GetAttr(
                                    obj=ein.Alias("A"),
                                    attr=ein.Literal("elems"),
                                    dim=None,
                                ),
                                idxs=(ein.Index("i"),),
                            ),
                            ein.Access(
                                tns=ein.Alias("B"),
                                idxs=(
                                    ein.Access(
                                        tns=ein.GetAttr(
                                            obj=ein.Alias("A"),
                                            attr=ein.Literal("coords"),
                                            dim=None,
                                        ),
                                        idxs=(ein.Index("i"),),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
                ein.Produces((ein.Alias("C"),)),
            )
        )

        result = (A + B).flatten()
        self.run_einsum_plan(prgm, {"A": sparse_A, "B": B}, result)

    def test_indirect_multiple_reads(self, rng):
        """Test multiple indirect reads from the same tensor"""

        A = rng.random((3, 3))
        B = rng.random((3, 3))

        sparse_A = SparseTensor.from_dense_tensor(A)
        sparse_B = SparseTensor.from_dense_tensor(B)

        # C[i] = AElems[i] * BElems[i]
        # Both A and B are sparse, reading their elements directly
        prgm = ein.Plan(
            (
                ein.Einsum(
                    op=ein.Literal(overwrite),
                    tns=ein.Alias("C"),
                    idxs=(ein.Index("i"),),
                    arg=ein.Call(
                        op=ein.Literal(operator.mul),
                        args=(
                            ein.Access(
                                tns=ein.GetAttr(
                                    obj=ein.Alias("A"),
                                    attr=ein.Literal("elems"),
                                    dim=None,
                                ),
                                idxs=(ein.Index("i"),),
                            ),
                            ein.Access(
                                tns=ein.GetAttr(
                                    obj=ein.Alias("B"),
                                    attr=ein.Literal("elems"),
                                    dim=None,
                                ),
                                idxs=(ein.Index("i"),),
                            ),
                        ),
                    ),
                ),
                ein.Produces((ein.Alias("C"),)),
            )
        )

        result = finchlite.multiply(A, B).flatten()
        self.run_einsum_plan(prgm, {"A": sparse_A, "B": sparse_B}, result)

    def test_indirect_with_constant(self, rng):
        """Test indirect access combined with constant"""

        A = rng.random((4, 4))
        B = rng.random((4, 4))

        sparse_A = SparseTensor.from_dense_tensor(A)

        # C[i] = AElems[i] * B[ACoords[i]] + 5.0
        prgm = ein.Plan(
            (
                ein.Einsum(
                    op=ein.Literal(overwrite),
                    tns=ein.Alias("C"),
                    idxs=(ein.Index("i"),),
                    arg=ein.Call(
                        op=ein.Literal(operator.add),
                        args=(
                            ein.Call(
                                op=ein.Literal(operator.mul),
                                args=(
                                    ein.Access(
                                        tns=ein.GetAttr(
                                            obj=ein.Alias("A"),
                                            attr=ein.Literal("elems"),
                                            dim=None,
                                        ),
                                        idxs=(ein.Index("i"),),
                                    ),
                                    ein.Access(
                                        tns=ein.Alias("B"),
                                        idxs=(
                                            ein.Access(
                                                tns=ein.GetAttr(
                                                    obj=ein.Alias("A"),
                                                    attr=ein.Literal("coords"),
                                                    dim=None,
                                                ),
                                                idxs=(ein.Index("i"),),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                            ein.Literal(5.0),
                        ),
                    ),
                ),
                ein.Produces((ein.Alias("C"),)),
            )
        )

        result = (A * B + 5.0).flatten()
        self.run_einsum_plan(prgm, {"A": sparse_A, "B": B}, result)

    def test_indirect_nested_operations(self, rng):
        """Test nested operations with indirect access"""

        A = rng.random((3, 3))
        B = rng.random((3, 3))
        C = rng.random((3, 3))

        sparse_A = SparseTensor.from_dense_tensor(A)

        # D[i] = (AElems[i] + B[ACoords[i]]) * C[ACoords[i]]
        prgm = ein.Plan(
            (
                ein.Einsum(
                    op=ein.Literal(overwrite),
                    tns=ein.Alias("D"),
                    idxs=(ein.Index("i"),),
                    arg=ein.Call(
                        op=ein.Literal(operator.mul),
                        args=(
                            ein.Call(
                                op=ein.Literal(operator.add),
                                args=(
                                    ein.Access(
                                        tns=ein.GetAttr(
                                            obj=ein.Alias("A"),
                                            attr=ein.Literal("elems"),
                                            dim=None,
                                        ),
                                        idxs=(ein.Index("i"),),
                                    ),
                                    ein.Access(
                                        tns=ein.Alias("B"),
                                        idxs=(
                                            ein.Access(
                                                tns=ein.GetAttr(
                                                    obj=ein.Alias("A"),
                                                    attr=ein.Literal("coords"),
                                                    dim=None,
                                                ),
                                                idxs=(ein.Index("i"),),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                            ein.Access(
                                tns=ein.Alias("C"),
                                idxs=(
                                    ein.Access(
                                        tns=ein.GetAttr(
                                            obj=ein.Alias("A"),
                                            attr=ein.Literal("coords"),
                                            dim=None,
                                        ),
                                        idxs=(ein.Index("i"),),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
                ein.Produces((ein.Alias("D"),)),
            )
        )

        result = ((A + B) * C).flatten()
        self.run_einsum_plan(prgm, {"A": sparse_A, "B": B, "C": C}, result)

    def test_indirect_direct_access_only(self, rng):
        """Test accessing only the indirect coordinates"""

        A = rng.random((8,))
        B = rng.random((8,))

        sparse_A = SparseTensor.from_dense_tensor(A)

        # C[i] = B[ACoords[i]] (read B indirectly, without using A's elements)
        prgm = ein.Plan(
            (
                ein.Einsum(
                    op=ein.Literal(overwrite),
                    tns=ein.Alias("C"),
                    idxs=(ein.Index("i"),),
                    arg=ein.Access(
                        tns=ein.Alias("B"),
                        idxs=(
                            ein.Access(
                                tns=ein.GetAttr(
                                    obj=ein.Alias("A"),
                                    attr=ein.Literal("coords"),
                                    dim=None,
                                ),
                                idxs=(ein.Index("i"),),
                            ),
                        ),
                    ),
                ),
                ein.Produces((ein.Alias("C"),)),
            )
        )

        # Result should be B's values at A's coordinates
        expected = B[A != 0]
        self.run_einsum_plan(prgm, {"A": sparse_A, "B": B}, expected)

    def test_double_indirection(self, rng):
        """Test double indirection: A[B[CCoords[i]]]"""

        # Create small arrays to ensure valid indexing
        A = rng.random((8,))
        # B contains integer indices into A (0-7)
        B = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        rng.shuffle(B)
        # C is sparse
        C = rng.random((8,))

        sparse_C = SparseTensor.from_dense_tensor(C)

        # D[i] = A[B[CCoords[i]]]
        # First get CCoords[i], then use that to index B, then use B's value to index A
        prgm = ein.Plan(
            (
                ein.Einsum(
                    op=ein.Literal(overwrite),
                    tns=ein.Alias("D"),
                    idxs=(ein.Index("i"),),
                    arg=ein.Access(
                        tns=ein.Alias("A"),
                        idxs=(
                            ein.Access(
                                tns=ein.Alias("B"),
                                idxs=(
                                    ein.Access(
                                        tns=ein.GetAttr(
                                            obj=ein.Alias("C"),
                                            attr=ein.Literal("coords"),
                                            dim=None,
                                        ),
                                        idxs=(ein.Index("i"),),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
                ein.Produces((ein.Alias("D"),)),
            )
        )

        # Expected: for each non-zero position in C, get its coord,
        # index into B, then index into A
        c_coords = sparse_C.coords
        expected = A[B[c_coords]].flatten()
        self.run_einsum_plan(prgm, {"A": A, "B": B, "C": sparse_C}, expected)

    def test_triple_indirection(self, rng):
        """Test triple indirection: A[B[C[DCoords[i]]]]"""

        # Create arrays with valid indexing ranges
        A = rng.random((8,))
        B = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        rng.shuffle(B)
        C = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        rng.shuffle(C)
        D = rng.random((8,))

        sparse_D = SparseTensor.from_dense_tensor(D)

        # E[i] = A[B[C[DCoords[i]]]]
        prgm = ein.Plan(
            (
                ein.Einsum(
                    op=ein.Literal(overwrite),
                    tns=ein.Alias("E"),
                    idxs=(ein.Index("i"),),
                    arg=ein.Access(
                        tns=ein.Alias("A"),
                        idxs=(
                            ein.Access(
                                tns=ein.Alias("B"),
                                idxs=(
                                    ein.Access(
                                        tns=ein.Alias("C"),
                                        idxs=(
                                            ein.Access(
                                                tns=ein.GetAttr(
                                                    obj=ein.Alias("D"),
                                                    attr=ein.Literal("coords"),
                                                    dim=None,
                                                ),
                                                idxs=(ein.Index("i"),),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
                ein.Produces((ein.Alias("E"),)),
            )
        )

        # Expected: chain of indirections
        d_coords = sparse_D.coords
        expected = A[B[C[d_coords]]].flatten()
        self.run_einsum_plan(prgm, {"A": A, "B": B, "C": C, "D": sparse_D}, expected)

    def test_mixed_direct_indirect_indexing_2d(self, rng):
        """Test mixed indexing: A[BCoords[i], j] - one indirect, one direct"""

        A = rng.random((5, 4))
        B = rng.random((5,))

        sparse_B = SparseTensor.from_dense_tensor(B)

        # C[i, j] = A[BCoords[i], j]
        # First index is indirect (from B's coords), second is direct
        prgm = ein.Plan(
            (
                ein.Einsum(
                    op=ein.Literal(overwrite),
                    tns=ein.Alias("C"),
                    idxs=(ein.Index("i"), ein.Index("j")),
                    arg=ein.Access(
                        tns=ein.Alias("A"),
                        idxs=(
                            ein.Access(
                                tns=ein.GetAttr(
                                    obj=ein.Alias("B"),
                                    attr=ein.Literal("coords"),
                                    dim=None,
                                ),
                                idxs=(ein.Index("i"),),
                            ),
                            ein.Index("j"),
                        ),
                    ),
                ),
                ein.Produces((ein.Alias("C"),)),
            )
        )

        # Expected: A rows indexed by B's coords, all columns
        b_coords = sparse_B.coords
        expected = A[b_coords.flatten(), :]
        self.run_einsum_plan(prgm, {"A": A, "B": sparse_B}, expected)

    def test_mixed_direct_indirect_indexing_reversed(self, rng):
        """
        Test mixed indexing reversed
        A[i, BCoords[j]] - first direct, second indirect
        """

        A = rng.random((4, 6))
        B = rng.random((6,))

        sparse_B = SparseTensor.from_dense_tensor(B)

        # C[i, j] = A[i, BCoords[j]]
        # First index is direct, second is indirect
        prgm = ein.Plan(
            (
                ein.Einsum(
                    op=ein.Literal(overwrite),
                    tns=ein.Alias("C"),
                    idxs=(ein.Index("i"), ein.Index("j")),
                    arg=ein.Access(
                        tns=ein.Alias("A"),
                        idxs=(
                            ein.Index("i"),
                            ein.Access(
                                tns=ein.GetAttr(
                                    obj=ein.Alias("B"),
                                    attr=ein.Literal("coords"),
                                    dim=None,
                                ),
                                idxs=(ein.Index("j"),),
                            ),
                        ),
                    ),
                ),
                ein.Produces((ein.Alias("C"),)),
            )
        )

        # Expected: all rows of A, columns indexed by B's coords
        b_coords = sparse_B.coords
        expected = A[:, b_coords.flatten()]
        self.run_einsum_plan(prgm, {"A": A, "B": sparse_B}, expected)

    def test_both_indices_indirect_same_source(self, rng):
        """Test both indices indirect from same source: A[BCoords[i], BCoords[i]]"""

        A = rng.random((6, 6))
        B = rng.random((6,))

        sparse_B = SparseTensor.from_dense_tensor(B)

        # C[i] = A[BCoords[i], BCoords[i]]
        # Extracting diagonal-like elements using indirect coordinates
        prgm = ein.Plan(
            (
                ein.Einsum(
                    op=ein.Literal(overwrite),
                    tns=ein.Alias("C"),
                    idxs=(ein.Index("i"),),
                    arg=ein.Access(
                        tns=ein.Alias("A"),
                        idxs=(
                            ein.Access(
                                tns=ein.GetAttr(
                                    obj=ein.Alias("B"),
                                    attr=ein.Literal("coords"),
                                    dim=None,
                                ),
                                idxs=(ein.Index("i"),),
                            ),
                            ein.Access(
                                tns=ein.GetAttr(
                                    obj=ein.Alias("B"),
                                    attr=ein.Literal("coords"),
                                    dim=None,
                                ),
                                idxs=(ein.Index("i"),),
                            ),
                        ),
                    ),
                ),
                ein.Produces((ein.Alias("C"),)),
            )
        )

        # Expected: A[coords, coords] - pseudo-diagonal at indirect positions
        b_coords = sparse_B.coords
        expected = A[b_coords.flatten(), b_coords.flatten()]
        self.run_einsum_plan(prgm, {"A": A, "B": sparse_B}, expected)

    def test_both_indices_indirect_different_sources(self, rng):
        """
        Test both indices indirect from different sources:
        A[BCoords[i], CCoords[i]]
        """

        A = rng.random((6, 6))
        B = rng.random((6,))
        C = rng.random((6,))

        sparse_B = SparseTensor.from_dense_tensor(B)
        sparse_C = SparseTensor.from_dense_tensor(C)

        # D[i] = A[BCoords[i], CCoords[i]]
        prgm = ein.Plan(
            (
                ein.Einsum(
                    op=ein.Literal(overwrite),
                    tns=ein.Alias("D"),
                    idxs=(ein.Index("i"),),
                    arg=ein.Access(
                        tns=ein.Alias("A"),
                        idxs=(
                            ein.Access(
                                tns=ein.GetAttr(
                                    obj=ein.Alias("B"),
                                    attr=ein.Literal("coords"),
                                    dim=None,
                                ),
                                idxs=(ein.Index("i"),),
                            ),
                            ein.Access(
                                tns=ein.GetAttr(
                                    obj=ein.Alias("C"),
                                    attr=ein.Literal("coords"),
                                    dim=None,
                                ),
                                idxs=(ein.Index("i"),),
                            ),
                        ),
                    ),
                ),
                ein.Produces((ein.Alias("D"),)),
            )
        )

        # Expected: A indexed by pairs of coords from B and C
        # This requires both to have the same number of non-zero elements
        b_coords = sparse_B.coords.flatten()
        c_coords = sparse_C.coords.flatten()
        expected = A[b_coords, c_coords]
        self.run_einsum_plan(prgm, {"A": A, "B": sparse_B, "C": sparse_C}, expected)

    def test_double_indirection_with_operation(self, rng):
        """Test double indirection combined with arithmetic operation"""

        A = rng.random((8,))
        B = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        rng.shuffle(B)
        C = rng.random((8,))

        sparse_C = SparseTensor.from_dense_tensor(C)

        # E[i] = A[B[CCoords[i]]] * CElems[i]
        # Double indirection plus multiplication with sparse elements
        prgm = ein.Plan(
            (
                ein.Einsum(
                    op=ein.Literal(overwrite),
                    tns=ein.Alias("E"),
                    idxs=(ein.Index("i"),),
                    arg=ein.Call(
                        op=ein.Literal(operator.mul),
                        args=(
                            ein.Access(
                                tns=ein.Alias("A"),
                                idxs=(
                                    ein.Access(
                                        tns=ein.Alias("B"),
                                        idxs=(
                                            ein.Access(
                                                tns=ein.GetAttr(
                                                    obj=ein.Alias("C"),
                                                    attr=ein.Literal("coords"),
                                                    dim=None,
                                                ),
                                                idxs=(ein.Index("i"),),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                            ein.Access(
                                tns=ein.GetAttr(
                                    obj=ein.Alias("C"),
                                    attr=ein.Literal("elems"),
                                    dim=None,
                                ),
                                idxs=(ein.Index("i"),),
                            ),
                        ),
                    ),
                ),
                ein.Produces((ein.Alias("E"),)),
            )
        )

        c_coords = sparse_C.coords.flatten()
        c_elems = sparse_C.data
        expected = A[B[c_coords]] * c_elems

        self.run_einsum_plan(prgm, {"A": A, "B": B, "C": sparse_C}, expected)

    def test_mixed_indexing_with_computation(self, rng):
        """Test mixed direct/indirect indexing with computation"""

        A = rng.random((5, 5))
        B = rng.random((4,))

        sparse_B = SparseTensor.from_dense_tensor(B)

        # D[i, j] = A[BCoords[i], j] + BElems[i]
        # Mixed indexing plus addition with sparse elements
        prgm = ein.Plan(
            (
                ein.Einsum(
                    op=ein.Literal(overwrite),
                    tns=ein.Alias("D"),
                    idxs=(ein.Index("i"), ein.Index("j")),
                    arg=ein.Call(
                        op=ein.Literal(operator.add),
                        args=(
                            ein.Access(
                                tns=ein.Alias("A"),
                                idxs=(
                                    ein.Access(
                                        tns=ein.GetAttr(
                                            obj=ein.Alias("B"),
                                            attr=ein.Literal("coords"),
                                            dim=None,
                                        ),
                                        idxs=(ein.Index("i"),),
                                    ),
                                    ein.Index("j"),
                                ),
                            ),
                            ein.Access(
                                tns=ein.GetAttr(
                                    obj=ein.Alias("B"),
                                    attr=ein.Literal("elems"),
                                    dim=None,
                                ),
                                idxs=(ein.Index("i"),),
                            ),
                        ),
                    ),
                ),
                ein.Produces((ein.Alias("D"),)),
            )
        )

        b_coords = sparse_B.coords.flatten()
        b_elems = sparse_B.data
        expected = A[b_coords, :] + b_elems[:, np.newaxis]
        self.run_einsum_plan(prgm, {"A": A, "B": sparse_B}, expected)

    def test_indirect_3d_tensor_access(self, rng):
        """Test indirect access on 3D tensor with mixed indices"""

        A = rng.random((3, 4, 5))
        B = rng.random((4,))

        sparse_B = SparseTensor.from_dense_tensor(B)

        # C[i, j, k] = A[i, BCoords[j], k]
        # Middle dimension is indirectly indexed
        prgm = ein.Plan(
            (
                ein.Einsum(
                    op=ein.Literal(overwrite),
                    tns=ein.Alias("C"),
                    idxs=(ein.Index("i"), ein.Index("j"), ein.Index("k")),
                    arg=ein.Access(
                        tns=ein.Alias("A"),
                        idxs=(
                            ein.Index("i"),
                            ein.Access(
                                tns=ein.GetAttr(
                                    obj=ein.Alias("B"),
                                    attr=ein.Literal("coords"),
                                    dim=None,
                                ),
                                idxs=(ein.Index("j"),),
                            ),
                            ein.Index("k"),
                        ),
                    ),
                ),
                ein.Produces((ein.Alias("C"),)),
            )
        )

        b_coords = sparse_B.coords.flatten()
        expected = A[:, b_coords, :]
        self.run_einsum_plan(prgm, {"A": A, "B": sparse_B}, expected)
