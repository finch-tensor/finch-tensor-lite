import math
import operator

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
)


def test_algebra_selected():
    assert is_distributive(operator.mul, operator.add)
    assert is_distributive(operator.mul, operator.sub)
    assert is_distributive(operator.and_, operator.or_)
    assert is_distributive(operator.and_, operator.xor)
    assert is_distributive(operator.or_, operator.and_)
    assert is_distributive(np.logical_and, np.logical_or)
    assert is_distributive(np.logical_and, np.logical_xor)
    assert is_distributive(np.logical_or, np.logical_and)
    assert is_annihilator(operator.add, math.inf)
    assert is_annihilator(operator.mul, 0)
    assert is_annihilator(operator.or_, True)
    assert is_annihilator(operator.and_, False)
    assert is_annihilator(np.logaddexp, math.inf)
    assert is_annihilator(np.logical_or, True)
    assert is_annihilator(np.logical_and, False)
    assert is_identity(operator.add, 0)
    assert is_identity(operator.mul, 1)
    assert is_identity(operator.or_, False)
    assert is_identity(operator.and_, True)
    assert is_identity(operator.truediv, 1)
    assert is_identity(operator.lshift, 0)
    assert is_identity(operator.rshift, 0)
    assert is_identity(operator.pow, 1)
    assert is_identity(np.divide, 1)
    assert is_identity(np.logaddexp, -math.inf)
    assert is_identity(np.logical_or, False)
    assert is_identity(np.logical_and, True)
    assert is_identity(max, -math.inf)
    assert is_identity(min, math.inf)
    assert is_associative(operator.add)
    assert is_associative(operator.mul)
    assert is_associative(np.logical_and)
    assert is_associative(np.logical_xor)
    assert is_associative(np.logical_or)
    assert is_associative(np.logaddexp)
    assert init_value(operator.and_, bool) is True
    assert init_value(operator.or_, bool) is False
    assert init_value(operator.xor, bool) is False
    assert init_value(np.logaddexp, float) == -math.inf
    assert init_value(np.logical_and, bool) is True
    assert init_value(np.logical_or, bool) is False
    assert init_value(np.logical_xor, bool) is False
    assert is_idempotent(operator.and_)
    assert is_idempotent(operator.or_)
    assert is_idempotent(np.logical_and)
    assert is_idempotent(np.logical_or)
    assert is_idempotent(min)
    assert is_idempotent(max)
    assert is_idempotent(operator.add) is False
    assert is_idempotent(operator.mul) is False
    assert is_idempotent(operator.xor) is False
    assert is_idempotent(np.logical_xor) is False
    assert is_idempotent(np.logaddexp) is False
    assert repeat_operator(operator.add) is operator.mul
    assert repeat_operator(operator.mul) is math.exp
    assert repeat_operator(operator.and_) is None
    assert cansplitpush(operator.add, operator.add) is True
    assert cansplitpush(operator.add, operator.mul) is False
    assert cansplitpush(operator.and_, operator.and_) is False
