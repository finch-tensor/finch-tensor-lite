import math

from finchlite.algebra import (
    cansplitpush,
    ffuncs,
    init_value,
    is_annihilator,
    is_associative,
    is_distributive,
    is_idempotent,
    is_identity,
    repeat_operator,
)
import finchlite
import numpy as np


def test_algebra_selected():
    assert is_distributive(ffuncs.mul, ffuncs.add)
    assert is_distributive(ffuncs.mul, ffuncs.sub)
    assert is_distributive(ffuncs.and_, ffuncs.or_)
    assert is_distributive(ffuncs.and_, ffuncs.xor)
    assert is_distributive(ffuncs.or_, ffuncs.and_)
    assert is_distributive(ffuncs.logical_and, ffuncs.logical_or)
    assert is_distributive(ffuncs.logical_and, ffuncs.logical_xor)
    assert is_distributive(ffuncs.logical_or, ffuncs.logical_and)
    assert is_annihilator(ffuncs.add, math.inf)
    assert is_annihilator(ffuncs.mul, 0)
    assert is_annihilator(ffuncs.or_, True)
    assert is_annihilator(ffuncs.and_, False)
    assert is_annihilator(ffuncs.logaddexp, math.inf)
    assert is_annihilator(ffuncs.logical_or, True)
    assert is_annihilator(ffuncs.logical_and, False)
    assert is_identity(ffuncs.add, 0)
    assert is_identity(ffuncs.mul, 1)
    assert is_identity(ffuncs.or_, False)
    assert is_identity(ffuncs.and_, True)
    assert is_identity(ffuncs.truediv, 1)
    assert is_identity(ffuncs.lshift, 0)
    assert is_identity(ffuncs.rshift, 0)
    assert is_identity(ffuncs.pow, 1)
    assert is_identity(ffuncs.truediv, 1)
    assert is_identity(ffuncs.logaddexp, -math.inf)
    assert is_identity(ffuncs.logical_or, False)
    assert is_identity(ffuncs.logical_and, True)
    assert is_identity(ffuncs.max, -math.inf)
    assert is_identity(ffuncs.min, math.inf)
    assert is_associative(ffuncs.add)
    assert is_associative(ffuncs.mul)
    assert is_associative(ffuncs.logical_and)
    assert is_associative(ffuncs.logical_xor)
    assert is_associative(ffuncs.logical_or)
    assert is_associative(ffuncs.logaddexp)
    assert init_value(ffuncs.and_, finchlite.bool_) is np.True_
    assert init_value(ffuncs.or_, finchlite.bool_) is np.False_
    assert init_value(ffuncs.xor, finchlite.bool_) is np.False_
    assert init_value(ffuncs.logaddexp, finchlite.float64) == -math.inf
    assert init_value(ffuncs.logical_and, finchlite.bool_) is True
    assert init_value(ffuncs.logical_or, finchlite.bool_) is False
    assert init_value(ffuncs.logical_xor, finchlite.bool_) is False
    assert is_idempotent(ffuncs.and_)
    assert is_idempotent(ffuncs.or_)
    assert is_idempotent(ffuncs.logical_and)
    assert is_idempotent(ffuncs.logical_or)
    assert is_idempotent(ffuncs.min)
    assert is_idempotent(ffuncs.max)
    assert is_idempotent(ffuncs.add) is False
    assert is_idempotent(ffuncs.mul) is False
    assert is_idempotent(ffuncs.xor) is False
    assert is_idempotent(ffuncs.logical_xor) is False
    assert is_idempotent(ffuncs.logaddexp) is False
    assert repeat_operator(ffuncs.add) is ffuncs.mul
    assert repeat_operator(ffuncs.mul) is ffuncs.pow
    assert repeat_operator(ffuncs.and_) is None
    assert cansplitpush(ffuncs.add, ffuncs.add) is True
    assert cansplitpush(ffuncs.add, ffuncs.mul) is False
    assert cansplitpush(ffuncs.and_, ffuncs.and_) is False