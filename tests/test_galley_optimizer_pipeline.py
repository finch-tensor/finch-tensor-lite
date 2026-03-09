"""
Galley optimizer pipeline tests.
"""

import numpy as np

import finchlite.interface as fl_interface


# --- TEST 1: out = a * b via frontend ---
def test_elementwise_mul():
    """Running out = a * b with Finch/Galley pipeline using the frontend."""
    a = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    out = fl_interface.compute(
        fl_interface.lazy(a) * fl_interface.lazy(b),
        ctx=fl_interface.INTERPRET_NOTATION_GALLEY,
    )
    expected = np.array([[1.0, 2.0], [3.0, 4.0]]) * np.array([[1.0, 1.0], [1.0, 1.0]])
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 2: out = a * b + c * d via frontend ---
def test_add_of_elementwise():
    """Running out = a * b + c * d with Finch/Galley frontend."""
    a = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    c = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    d = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    out = fl_interface.compute(
        fl_interface.lazy(a) * fl_interface.lazy(b)
        + fl_interface.lazy(c) * fl_interface.lazy(d),
        ctx=fl_interface.INTERPRET_NOTATION_GALLEY,
    )
    expected = np.array(a) * np.array(b) + np.array(c) * np.array(d)
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 3: sum(A @ B, axis=0) via frontend ---
def test_matmul_sum_axis0():
    """
    Running out = sum_i sum_j (A[i,j] * B[j,k]) with Finch/Galley frontend.
    """
    A = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    out = fl_interface.compute(
        fl_interface.sum(fl_interface.lazy(A) @ fl_interface.lazy(B), axis=0),
        ctx=fl_interface.INTERPRET_NOTATION_GALLEY,
    )
    expected = np.sum(np.array(A) @ np.array(B), axis=0)
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 4: sum(A@B, axis=0) + sum(C@D, axis=1) ---
def test_sum_axis0_plus_sum_axis1():
    """
    Running out = sum(A @ B, axis=0) + sum(C @ D, axis=1). Correctness vs NumPy.
    """
    A = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    C = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    D = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))

    out = fl_interface.compute(
        fl_interface.sum(fl_interface.lazy(A) @ fl_interface.lazy(B), axis=0)
        + fl_interface.sum(fl_interface.lazy(C) @ fl_interface.lazy(D), axis=1),
        ctx=fl_interface.INTERPRET_NOTATION_GALLEY,
    )

    left = np.sum(np.array(A) @ np.array(B), axis=0)
    right = np.sum(np.array(C) @ np.array(D), axis=1)
    expected = left + right
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 5: Nested aggregates sum(A @ B) ---
def test_nested_aggregates_full_sum():
    """
    Nested aggregates: out = sum_i sum_j (A[i,j] * B[j,k]).
    Verified against NumPy.
    """
    A = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))

    out = fl_interface.compute(
        fl_interface.sum(fl_interface.lazy(A) @ fl_interface.lazy(B)),
        ctx=fl_interface.INTERPRET_NOTATION_GALLEY,
    )

    expected = np.sum(np.array(A) @ np.array(B))
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 6: sum((A @ B) @ C) ---
def test_deeper_nesting():
    """Deeper nesting: out = sum((A @ B) @ C).
    Verified against NumPy."""
    A = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    C = fl_interface.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))

    out = fl_interface.compute(
        fl_interface.sum(
            (fl_interface.lazy(A) @ fl_interface.lazy(B)) @ fl_interface.lazy(C)
        ),
        ctx=fl_interface.INTERPRET_NOTATION_GALLEY,
    )

    expected = np.sum((np.array(A) @ np.array(B)) @ np.array(C))
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 7: expand_dims + sum over singleton (extra axis padding) ---
def test_expand_dims_sum_singleton():
    """
    Exercises extra axis padding in logic_to_stats: sum(expand_dims(A, 2), axis=2).
    """
    A = fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))

    expanded = fl_interface.expand_dims(fl_interface.lazy(A), axis=2)
    out = fl_interface.compute(
        fl_interface.sum(expanded, axis=2),
        ctx=fl_interface.INTERPRET_NOTATION_GALLEY,
    )

    expected = np.sum(np.expand_dims(np.array(A), axis=2), axis=2)
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 8: Alias Matmul ---
def test_alias_matmul():
    """
    Exercises alias-based query merging and reordering.
    """
    A_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    A = fl_interface.lazy(fl_interface.asarray(np.array([[1.0, 2.0], [3.0, 4.0]])))
    B = A @ A
    C = B @ B @ B
    out = fl_interface.compute(
        C,
        ctx=fl_interface.INTERPRET_NOTATION_GALLEY,
    )

    expected = A_np @ A_np @ A_np @ A_np @ A_np @ A_np
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 9: Performance optimization (cost-based reduction order) ---
def test_galley_performance_optimization_chain_matmul():
    """
    Test Galley performing a real performance optimization based on
    input data. For a chain matmul A @ B @ C, the greedy optimizer chooses
    reduction order by cost (flops). With asymmetric shapes, one order is
    cheaper than the other; Galley picks the cheaper one.

    A(2,10) @ B(10,3) @ C(3,4) -> (2,4)
    - Reducing middle (j) first: 2*10*3 + 2*3*4 = 60 + 24 = 84 computations
    - Reducing k first: 10*3*4 + 2*10*4 = 120 + 80 = 200 computations

    Galley's cost model favors the cheaper reduction order.
    """
    A = fl_interface.asarray(np.arange(2 * 10, dtype=float).reshape(2, 10))
    B = fl_interface.asarray(np.arange(10 * 3, dtype=float).reshape(10, 3))
    C = fl_interface.asarray(np.arange(3 * 4, dtype=float).reshape(3, 4))

    out = fl_interface.compute(
        fl_interface.lazy(A) @ fl_interface.lazy(B) @ fl_interface.lazy(C),
        ctx=fl_interface.INTERPRET_NOTATION_GALLEY,
    )

    expected = np.array(A) @ np.array(B) @ np.array(C)
    assert np.allclose(np.array(out), np.array(expected))


# --- TEST 10: Chain matmul A(10,2) @ B(2,10) @ C(10,2) -> (10,2) ---
def test_galley_chain_matmul_10_2_2_10_10_2():
    """
    Chain matmul A(10,2) @ B(2,10) @ C(10,2) -> (10,2)
    - (A @ B) @ C: (10,2)@(2,10)@(10,2) -> contracts 2, then 10
    - A @ (B @ C): (10,2)@(2,10)@(10,2) -> contracts 10, then 2
    """
    A = fl_interface.asarray(np.arange(10 * 2, dtype=float).reshape(10, 2))
    B = fl_interface.asarray(np.arange(2 * 10, dtype=float).reshape(2, 10))
    C = fl_interface.asarray(np.arange(10 * 2, dtype=float).reshape(10, 2))

    out = fl_interface.compute(
        fl_interface.lazy(A) @ fl_interface.lazy(B) @ fl_interface.lazy(C),
        ctx=fl_interface.INTERPRET_NOTATION_GALLEY,
    )

    expected = np.array(A) @ np.array(B) @ np.array(C)
    assert np.allclose(np.array(out), np.array(expected))