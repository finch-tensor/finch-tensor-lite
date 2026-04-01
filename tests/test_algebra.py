import math

import numpy as np

from finchlite.algebra import (
    cansplitpush,
    init_value,
    is_annihilator,
    is_associative,
    is_distributive,
    is_idempotent,
    is_identity,
    repeat_operator,
    operator
)


def test_algebra_selected():
    assert is_distributive(operator.mul, operator.add)
    assert is_distributive(operator.mul, operator.sub)
    assert is_distributive(operator.and_, operator.or_)
    assert is_distributive(operator.and_, operator.xor)
    assert is_distributive(operator.or_, operator.and_)
    assert is_distributive(operator.logical_and, operator.logical_or)
    assert is_distributive(operator.logical_and, operator.logical_xor)
    assert is_distributive(operator.logical_or, operator.logical_and)
    assert is_annihilator(operator.add, math.inf)
    assert is_annihilator(operator.mul, 0)
    assert is_annihilator(operator.or_, True)
    assert is_annihilator(operator.and_, False)
    assert is_annihilator(operator.logaddexp, math.inf)
    assert is_annihilator(operator.logical_or, True)
    assert is_annihilator(operator.logical_and, False)
    assert is_identity(operator.add, 0)
    assert is_identity(operator.mul, 1)
    assert is_identity(operator.or_, False)
    assert is_identity(operator.and_, True)
    assert is_identity(operator.truediv, 1)
    assert is_identity(operator.lshift, 0)
    assert is_identity(operator.rshift, 0)
    assert is_identity(operator.pow, 1)
    assert is_identity(operator.truediv, 1)
    assert is_identity(operator.logaddexp, -math.inf)
    assert is_identity(operator.logical_or, False)
    assert is_identity(operator.logical_and, True)
    assert is_identity(operator.max, -math.inf)
    assert is_identity(operator.min, math.inf)
    assert is_associative(operator.add)
    assert is_associative(operator.mul)
    assert is_associative(operator.logical_and)
    assert is_associative(operator.logical_xor)
    assert is_associative(operator.logical_or)
    assert is_associative(operator.logaddexp)
    assert init_value(operator.and_, bool) is True
    assert init_value(operator.or_, bool) is False
    assert init_value(operator.xor, bool) is False
    assert init_value(operator.logaddexp, float) == -math.inf
    assert init_value(operator.logical_and, bool) is True
    assert init_value(operator.logical_or, bool) is False
    assert init_value(operator.logical_xor, bool) is False
    assert is_idempotent(operator.and_)
    assert is_idempotent(operator.or_)
    assert is_idempotent(operator.logical_and)
    assert is_idempotent(operator.logical_or)
    assert is_idempotent(operator.min)
    assert is_idempotent(operator.max)
    assert is_idempotent(operator.add) is False
    assert is_idempotent(operator.mul) is False
    assert is_idempotent(operator.xor) is False
    assert is_idempotent(operator.logical_xor) is False
    assert is_idempotent(operator.logaddexp) is False
    assert repeat_operator(operator.add) is operator.mul
    assert repeat_operator(operator.mul) is operator.pow
    assert repeat_operator(operator.and_) is None
    assert cansplitpush(operator.add, operator.add) is True
    assert cansplitpush(operator.add, operator.mul) is False
    assert cansplitpush(operator.and_, operator.and_) is False
