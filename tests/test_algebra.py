import math

from finchlite.algebra import (
    cansplitpush,
    ffunc,
    init_value,
    is_annihilator,
    is_associative,
    is_distributive,
    is_idempotent,
    is_identity,
    repeat_operator,
)


def test_algebra_selected():
    assert is_distributive(ffunc.mul, ffunc.add)
    assert is_distributive(ffunc.mul, ffunc.sub)
    assert is_distributive(ffunc.and_, ffunc.or_)
    assert is_distributive(ffunc.and_, ffunc.xor)
    assert is_distributive(ffunc.or_, ffunc.and_)
    assert is_distributive(ffunc.logical_and, ffunc.logical_or)
    assert is_distributive(ffunc.logical_and, ffunc.logical_xor)
    assert is_distributive(ffunc.logical_or, ffunc.logical_and)
    assert is_annihilator(ffunc.add, math.inf)
    assert is_annihilator(ffunc.mul, 0)
    assert is_annihilator(ffunc.or_, True)
    assert is_annihilator(ffunc.and_, False)
    assert is_annihilator(ffunc.logaddexp, math.inf)
    assert is_annihilator(ffunc.logical_or, True)
    assert is_annihilator(ffunc.logical_and, False)
    assert is_identity(ffunc.add, 0)
    assert is_identity(ffunc.mul, 1)
    assert is_identity(ffunc.or_, False)
    assert is_identity(ffunc.and_, True)
    assert is_identity(ffunc.truediv, 1)
    assert is_identity(ffunc.lshift, 0)
    assert is_identity(ffunc.rshift, 0)
    assert is_identity(ffunc.pow, 1)
    assert is_identity(ffunc.truediv, 1)
    assert is_identity(ffunc.logaddexp, -math.inf)
    assert is_identity(ffunc.logical_or, False)
    assert is_identity(ffunc.logical_and, True)
    assert is_identity(ffunc.max, -math.inf)
    assert is_identity(ffunc.min, math.inf)
    assert is_associative(ffunc.add)
    assert is_associative(ffunc.mul)
    assert is_associative(ffunc.logical_and)
    assert is_associative(ffunc.logical_xor)
    assert is_associative(ffunc.logical_or)
    assert is_associative(ffunc.logaddexp)
    assert init_value(ffunc.and_, bool) is True
    assert init_value(ffunc.or_, bool) is False
    assert init_value(ffunc.xor, bool) is False
    assert init_value(ffunc.logaddexp, float) == -math.inf
    assert init_value(ffunc.logical_and, bool) is True
    assert init_value(ffunc.logical_or, bool) is False
    assert init_value(ffunc.logical_xor, bool) is False
    assert is_idempotent(ffunc.and_)
    assert is_idempotent(ffunc.or_)
    assert is_idempotent(ffunc.logical_and)
    assert is_idempotent(ffunc.logical_or)
    assert is_idempotent(ffunc.min)
    assert is_idempotent(ffunc.max)
    assert is_idempotent(ffunc.add) is False
    assert is_idempotent(ffunc.mul) is False
    assert is_idempotent(ffunc.xor) is False
    assert is_idempotent(ffunc.logical_xor) is False
    assert is_idempotent(ffunc.logaddexp) is False
    assert repeat_operator(ffunc.add) is ffunc.mul
    assert repeat_operator(ffunc.mul) is ffunc.pow
    assert repeat_operator(ffunc.and_) is None
    assert cansplitpush(ffunc.add, ffunc.add) is True
    assert cansplitpush(ffunc.add, ffunc.mul) is False
    assert cansplitpush(ffunc.and_, ffunc.and_) is False
