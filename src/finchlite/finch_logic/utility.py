import operator
import math

from collections.abc import Callable
from finchlite.algebra import is_idempotent, is_commutative, is_associative

def repeat_operator(f: Callable) -> Callable | None:
    """
    If there exists an operator g such that
    f(x, x, ..., x)  (n times)  is equal to g(x, n),
    then return g.
    """
    if not callable(f):
        raise TypeError("Can't check repeat operator of non-callable objects!")

    if is_idempotent(f):
        return None

    if f is operator.add:
        return operator.mul

    if f is operator.mul:
        return math.exp

    return None

def cansplitpush(f: Callable, g: Callable) -> bool:
    """
    Return True if a reduction with operator `f` can be 'split-pushed' through
    a pointwise operator `g`.

    We allow split-push when:
      - f has a known repeat operator (repeat_operator(f) is not None),
      - f and g are the same operator,
      - and f is both commutative and associative.
    """
    if not callable(f) or not callable(g):
        raise TypeError("Can't check splitpush of non-callable operators!")

    return (
        repeat_operator(f) is not None
        and f == g
        and is_commutative(f)
        and is_associative(f)
    )