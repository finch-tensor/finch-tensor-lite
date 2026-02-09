from operator import add, eq, ge, le

from finchlite.finch_notation.nodes import Call
from finchlite.finch_notation.nodes import Literal as L
from finchlite.finch_notation.proves import (
    rule_add_with_max,
    rule_add_with_min,
    rule_all_literals,
    rule_associative_flatten,
    rule_disjoint_flat_pair_max_min,
    rule_disjoint_flat_pair_min_max,
    rule_disjoint_flat_single_max_min,
    rule_disjoint_flat_single_min_max,
    rule_disjoint_nested_max_min,
    rule_disjoint_nested_min_max,
    rule_equal_same,
    rule_ge,
    rule_idempotent_unique,
    rule_le,
)

# NOTE: These test cases were AI generated based on proves.py.


def test_rule_all_literals():
    """Test that rule_all_literals evaluates ops on literal arguments."""
    # Simple case: add(1, 2) => 3
    expr = Call(L(add), (L(1), L(2)))
    result = rule_all_literals(expr)
    assert result == L(3)

    # Should not match non-literals
    expr = Call(L(add), (L(1), Call(L(max), (L(2), L(3)))))
    result = rule_all_literals(expr)
    assert result is None


def test_rule_idempotent_unique():
    """Test that rule_idempotent_unique removes duplicates from idempotent ops."""
    # max(a, a, b) => max(a, b)
    a, b = L("a"), L("b")
    expr = Call(L(max), (a, a, b))
    result = rule_idempotent_unique(expr)
    assert isinstance(result, Call)
    assert result.op == L(max)
    assert set(result.args) == {a, b}

    # Should not match when all unique
    expr = Call(L(max), (a, b))
    result = rule_idempotent_unique(expr)
    assert result is None


def test_rule_associative_flatten():
    """Test that rule_associative_flatten flattens nested associative ops."""
    # max(a, max(b, c)) => max(a, b, c)
    a, b, c = L("a"), L("b"), L("c")
    expr = Call(L(max), (a, Call(L(max), (b, c))))
    result = rule_associative_flatten(expr)
    assert result == Call(L(max), (a, b, c))

    # Should not match when no nested call
    expr = Call(L(max), (a, b))
    result = rule_associative_flatten(expr)
    assert result is None


def test_rule_equal_same():
    """Test that rule_equal_same matches eq(a, a) => True."""
    # eq(a, a) => True
    a = L("a")
    expr = Call(L(eq), (a, a))
    result = rule_equal_same(expr)
    assert result == L(True)

    # Should not match when different
    b = L("b")
    expr = Call(L(eq), (a, b))
    result = rule_equal_same(expr)
    assert result is None


def test_rule_ge():
    """Test that rule_ge transforms ge(a, b) => eq(a, max(a, b))."""
    # ge(a, b) => eq(a, max(a, b))
    a, b = L("a"), L("b")
    expr = Call(L(ge), (a, b))
    result = rule_ge(expr)
    assert result == Call(L(eq), (a, Call(L(max), (a, b))))


def test_rule_le():
    """Test that rule_le transforms le(a, b) => eq(max(a, b), b)."""
    # le(a, b) => eq(max(a, b), b)
    a, b = L("a"), L("b")
    expr = Call(L(le), (a, b))
    result = rule_le(expr)
    assert result == Call(L(eq), (Call(L(max), (a, b)), b))


def test_rule_add_with_max():
    """Test that rule_add_with_max distributes max over add."""
    # add(a, max(b, c)) => max(add(a, b), add(a, c))
    a, b, c = L("a"), L("b"), L("c")
    expr = Call(L(add), (a, Call(L(max), (b, c))))
    result = rule_add_with_max(expr)
    assert result == Call(L(max), (Call(L(add), (a, b)), Call(L(add), (a, c))))

    # Should not match when no max
    expr = Call(L(add), (a, b))
    result = rule_add_with_max(expr)
    assert result is None


def test_rule_add_with_min():
    """Test that rule_add_with_min distributes min over add."""
    # add(a, min(b, c)) => min(add(a, b), add(a, c))
    a, b, c = L("a"), L("b"), L("c")
    expr = Call(L(add), (a, Call(L(min), (b, c))))
    result = rule_add_with_min(expr)
    assert result == Call(L(min), (Call(L(add), (a, b)), Call(L(add), (a, c))))

    # Should not match when no min
    expr = Call(L(add), (a, b))
    result = rule_add_with_min(expr)
    assert result is None


def test_rule_disjoint_nested_max_min():
    """Test that rule_disjoint_nested_max_min handles nested max in min."""
    # max(a, min(b, max(a, c))) with non-disjoint sets
    a, b, c = L("a"), L("b"), L("c")
    expr = Call(L(max), (a, Call(L(min), (b, Call(L(max), (a, c))))))
    result = rule_disjoint_nested_max_min(expr)
    # Should return max(a, min(b, c)) since 'a' appears in both outer and inner max
    assert result == Call(L(max), (a, Call(L(min), (b, c))))


def test_rule_disjoint_nested_min_max():
    """Test that rule_disjoint_nested_min_max handles nested min in max."""
    # min(a, max(b, min(a, c))) with non-disjoint sets
    a, b, c = L("a"), L("b"), L("c")
    expr = Call(L(min), (a, Call(L(max), (b, Call(L(min), (a, c))))))
    result = rule_disjoint_nested_min_max(expr)
    # Should return min(a, max(b, c)) since 'a' appears in both outer and inner min
    assert result == Call(L(min), (a, Call(L(max), (b, c))))


def test_rule_disjoint_flat_single_max_min():
    """
    Test that rule_disjoint_flat_single_max_min removes min from max
    when disjoint.
    """
    # max(a, min(a, b)) with non-disjoint sets => max(a)
    a, b = L("a"), L("b")
    expr = Call(L(max), (a, Call(L(min), (a, b))))
    result = rule_disjoint_flat_single_max_min(expr)
    # Since 'a' is in both max args and min args (not disjoint), min is removed
    assert result == Call(L(max), (a,))

    # Should not match when disjoint
    c = L("c")
    expr = Call(L(max), (a, Call(L(min), (b, c))))
    result = rule_disjoint_flat_single_max_min(expr)
    assert result is None


def test_rule_disjoint_flat_single_min_max():
    """
    Test that rule_disjoint_flat_single_min_max removes max from min
    when disjoint.
    """
    # min(a, max(a, b)) with non-disjoint sets => min(a)
    a, b = L("a"), L("b")
    expr = Call(L(min), (a, Call(L(max), (a, b))))
    result = rule_disjoint_flat_single_min_max(expr)
    # Since 'a' is in both min args and max args (not disjoint), max is removed
    assert result == Call(L(min), (a,))

    # Should not match when disjoint
    c = L("c")
    expr = Call(L(min), (a, Call(L(max), (b, c))))
    result = rule_disjoint_flat_single_min_max(expr)
    assert result is None


def test_rule_disjoint_flat_pair_max_min():
    """Test that rule_disjoint_flat_pair_max_min handles two mins in max."""
    # max(min(a, b), min(a, c)) with non-disjoint mins
    a, b, c = L("a"), L("b"), L("c")
    expr = Call(L(max), (Call(L(min), (a, b)), Call(L(min), (a, c))))
    result = rule_disjoint_flat_pair_max_min(expr)
    # Should create nested structure with intersection and differences
    assert result == Call(
        L(max),
        (Call(L(min), (a, Call(L(max), (Call(L(min), (b,)), Call(L(min), (c,)))))),),
    )

    # Should not match when disjoint
    d = L("d")
    expr = Call(L(max), (Call(L(min), (a, b)), Call(L(min), (c, d))))
    result = rule_disjoint_flat_pair_max_min(expr)
    assert result is None


def test_rule_disjoint_flat_pair_min_max():
    """Test that rule_disjoint_flat_pair_min_max handles two maxs in min."""
    # min(max(a, b), max(a, c)) with non-disjoint maxs
    a, b, c = L("a"), L("b"), L("c")
    expr = Call(L(min), (Call(L(max), (a, b)), Call(L(max), (a, c))))
    result = rule_disjoint_flat_pair_min_max(expr)
    # Should create nested structure with intersection and differences
    assert result == Call(
        L(min),
        (Call(L(max), (a, Call(L(min), (Call(L(max), (b,)), Call(L(max), (c,)))))),),
    )

    # Should not match when disjoint
    d = L("d")
    expr = Call(L(min), (Call(L(max), (a, b)), Call(L(max), (c, d))))
    result = rule_disjoint_flat_pair_min_max(expr)
    assert result is None
