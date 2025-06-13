import math
import operator

from finch.algebra import is_annihilator, is_associative, is_distributive, is_identity


def test_algebra_selected():
    assert is_distributive(operator.mul, operator.add)
    assert is_distributive(operator.mul, operator.sub)
    assert is_distributive(operator.and_, operator.or_)
    assert is_distributive(operator.and_, operator.xor)
    assert is_distributive(operator.or_, operator.and_)
    assert is_annihilator(operator.add, math.inf)
    assert is_annihilator(operator.mul, 0)
    assert is_annihilator(operator.or_, True)
    assert is_annihilator(operator.and_, False)
    assert is_identity(operator.add, 0)
    assert is_identity(operator.mul, 1)
    assert is_identity(operator.or_, False)
    assert is_identity(operator.and_, True)
    assert is_identity(operator.truediv, 1)
    assert is_identity(operator.floordiv, 1)
    assert is_identity(operator.lshift, 0)
    assert is_identity(operator.rshift, 0)
    assert is_identity(operator.pow, 1)
    assert is_associative(operator.add)
    assert is_associative(operator.mul)
    assert is_associative(operator.and_)
    assert is_associative(operator.xor)
    assert is_associative(operator.or_)
