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
