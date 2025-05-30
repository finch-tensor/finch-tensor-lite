import math
import operator

from finch.algebra import is_annihilator, is_distributive, is_identity


def test_algebra_selected():
    assert is_distributive(operator.mul, operator.add)
    assert is_annihilator(operator.add, math.inf)
    assert is_annihilator(operator.mul, 0)
    assert is_identity(operator.add, 0)
    assert is_identity(operator.mul, 1)
