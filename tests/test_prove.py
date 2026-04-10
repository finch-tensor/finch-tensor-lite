from finchlite.algebra import ffuncs
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


def test_rule_all_literals():
    # Simple case: add(1, 2) => 3
    expr = Call(L(ffuncs.add), (L(1), L(2)))
    result = rule_all_literals(expr)
    assert result == L(3)

    # Should not match non-literals
    expr = Call(L(ffuncs.add), (L(1), Call(L(ffuncs.max), (L(2), L(3)))))
    result = rule_all_literals(expr)
    assert result is None


def test_rule_idempotent_unique():
    # max(a, a, b) => max(a, b)
    a, b = L("a"), L("b")
    expr = Call(L(ffuncs.max), (a, a, b))
    result = rule_idempotent_unique(expr)
    assert isinstance(result, Call)
    assert result.op == L(ffuncs.max)
    assert set(result.args) == {a, b}

    # Should not match when all unique
    expr = Call(L(ffuncs.max), (a, b))
    result = rule_idempotent_unique(expr)
    assert result is None


def test_rule_associative_flatten():
    # max(a, max(b, c)) => max(a, b, c)
    a, b, c = L("a"), L("b"), L("c")
    expr = Call(L(ffuncs.max), (a, Call(L(ffuncs.max), (b, c))))
    result = rule_associative_flatten(expr)
    assert result == Call(L(ffuncs.max), (a, b, c))

    # Should not match when no nested call
    expr = Call(L(ffuncs.max), (a, b))
    result = rule_associative_flatten(expr)
    assert result is None


def test_rule_equal_same():
    # eq(a, a) => True
    a = L("a")
    expr = Call(L(ffuncs.eq), (a, a))
    result = rule_equal_same(expr)
    assert result == L(True)

    # Should not match when different
    b = L("b")
    expr = Call(L(ffuncs.eq), (a, b))
    result = rule_equal_same(expr)
    assert result is None


def test_rule_ge():
    # ge(a, b) => eq(a, max(a, b))
    a, b = L("a"), L("b")
    expr = Call(L(ffuncs.ge), (a, b))
    result = rule_ge(expr)
    assert result == Call(L(ffuncs.eq), (a, Call(L(ffuncs.max), (a, b))))


def test_rule_le():
    # le(a, b) => eq(max(a, b), b)
    a, b = L("a"), L("b")
    expr = Call(L(ffuncs.le), (a, b))
    result = rule_le(expr)
    assert result == Call(L(ffuncs.eq), (Call(L(ffuncs.max), (a, b)), b))


def test_rule_add_with_max():
    # add(a, max(b, c)) => max(add(a, b), add(a, c))
    a, b, c = L("a"), L("b"), L("c")
    expr = Call(L(ffuncs.add), (a, Call(L(ffuncs.max), (b, c))))
    result = rule_add_with_max(expr)
    assert result == Call(
        L(ffuncs.max), (Call(L(ffuncs.add), (a, b)), Call(L(ffuncs.add), (a, c)))
    )

    # Should not match when no max
    expr = Call(L(ffuncs.add), (a, b))
    result = rule_add_with_max(expr)
    assert result is None


def test_rule_add_with_min():
    # add(a, min(b, c)) => min(add(a, b), add(a, c))
    a, b, c = L("a"), L("b"), L("c")
    expr = Call(L(ffuncs.add), (a, Call(L(ffuncs.min), (b, c))))
    result = rule_add_with_min(expr)
    assert result == Call(
        L(ffuncs.min), (Call(L(ffuncs.add), (a, b)), Call(L(ffuncs.add), (a, c)))
    )

    # Should not match when no min
    expr = Call(L(ffuncs.add), (a, b))
    result = rule_add_with_min(expr)
    assert result is None


def test_rule_disjoint_nested_max_min():
    # max(a, min(b, max(a, c))) with non-disjoint sets
    a, b, c = L("a"), L("b"), L("c")
    expr = Call(L(ffuncs.max), (a, Call(L(ffuncs.min), (b, Call(L(ffuncs.max), (a, c))))))
    result = rule_disjoint_nested_max_min(expr)
    # Should return max(a, min(b, c)) since 'a' appears in both outer and inner max
    assert result == Call(L(ffuncs.max), (a, Call(L(ffuncs.min), (b, c))))


def test_rule_disjoint_nested_min_max():
    # min(a, max(b, min(a, c))) with non-disjoint sets
    a, b, c = L("a"), L("b"), L("c")
    expr = Call(L(ffuncs.min), (a, Call(L(ffuncs.max), (b, Call(L(ffuncs.min), (a, c))))))
    result = rule_disjoint_nested_min_max(expr)
    # Should return min(a, max(b, c)) since 'a' appears in both outer and inner min
    assert result == Call(L(ffuncs.min), (a, Call(L(ffuncs.max), (b, c))))


def test_rule_disjoint_flat_single_max_min():
    # max(a, min(a, b)) with non-disjoint sets => max(a)
    a, b = L("a"), L("b")
    expr = Call(L(ffuncs.max), (a, Call(L(ffuncs.min), (a, b))))
    result = rule_disjoint_flat_single_max_min(expr)
    # Since 'a' is in both max args and min args (not disjoint), min is removed
    assert result == Call(L(ffuncs.max), (a,))

    # Should not match when disjoint
    c = L("c")
    expr = Call(L(ffuncs.max), (a, Call(L(ffuncs.min), (b, c))))
    result = rule_disjoint_flat_single_max_min(expr)
    assert result is None


def test_rule_disjoint_flat_single_min_max():
    # min(a, max(a, b)) with non-disjoint sets => min(a)
    a, b = L("a"), L("b")
    expr = Call(L(ffuncs.min), (a, Call(L(ffuncs.max), (a, b))))
    result = rule_disjoint_flat_single_min_max(expr)
    # Since 'a' is in both min args and max args (not disjoint), max is removed
    assert result == Call(L(ffuncs.min), (a,))

    # Should not match when disjoint
    c = L("c")
    expr = Call(L(ffuncs.min), (a, Call(L(ffuncs.max), (b, c))))
    result = rule_disjoint_flat_single_min_max(expr)
    assert result is None


def test_rule_disjoint_flat_pair_max_min():
    # max(min(a, b), min(a, c)) with non-disjoint mins
    a, b, c = L("a"), L("b"), L("c")
    expr = Call(L(ffuncs.max), (Call(L(ffuncs.min), (a, b)), Call(L(ffuncs.min), (a, c))))
    result = rule_disjoint_flat_pair_max_min(expr)
    # Should create nested structure with intersection and differences
    assert result == Call(
        L(ffuncs.max),
        (
            Call(
                L(ffuncs.min),
                (
                    a,
                    Call(
                        L(ffuncs.max),
                        (Call(L(ffuncs.min), (b,)), Call(L(ffuncs.min), (c,))),
                    ),
                ),
            ),
        ),
    )

    # Should not match when disjoint
    d = L("d")
    expr = Call(L(ffuncs.max), (Call(L(ffuncs.min), (a, b)), Call(L(ffuncs.min), (c, d))))
    result = rule_disjoint_flat_pair_max_min(expr)
    assert result is None


def test_rule_disjoint_flat_pair_min_max():
    # min(max(a, b), max(a, c)) with non-disjoint maxs
    a, b, c = L("a"), L("b"), L("c")
    expr = Call(L(ffuncs.min), (Call(L(ffuncs.max), (a, b)), Call(L(ffuncs.max), (a, c))))
    result = rule_disjoint_flat_pair_min_max(expr)
    # Should create nested structure with intersection and differences
    assert result == Call(
        L(ffuncs.min),
        (
            Call(
                L(ffuncs.max),
                (
                    a,
                    Call(
                        L(ffuncs.min),
                        (Call(L(ffuncs.max), (b,)), Call(L(ffuncs.max), (c,))),
                    ),
                ),
            ),
        ),
    )

    # Should not match when disjoint
    d = L("d")
    expr = Call(L(ffuncs.min), (Call(L(ffuncs.max), (a, b)), Call(L(ffuncs.max), (c, d))))
    result = rule_disjoint_flat_pair_min_max(expr)
    assert result is None
