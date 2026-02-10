import math
import operator
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import algebra
from .algebra import is_associative, is_commutative, is_idempotent


def and_test(a, b):
    return a & b


def or_test(a, b):
    return a | b


def not_test(a):
    return not a


def ifelse(a, b, c):
    return a if c else b


def promote_min(a, b):
    cast = algebra.promote_type(a, b)
    return cast(min(a, b))


def promote_max(a, b):
    cast = algebra.promote_type(a, b)
    return max(cast(a), cast(b))


def conjugate(x):
    """
    Computes the complex conjugate of the input number

    Parameters
    ----------
    x: Any
        The input number to compute the complex conjugate of.

    Returns
    ----------
    Any
        The complex conjugate of the input number. If the input is not a complex number,
        it returns the input unchanged.
    """
    if hasattr(x, "conjugate"):
        return x.conjugate()
    return x


# register the conjugate operation return type. The conjugate operation
# preserves the element type of the input tensor.
algebra.register_property(
    conjugate,
    "__call__",
    "return_type",
    lambda obj, x: x,
)

algebra.register_property(
    promote_min,
    "__call__",
    "return_type",
    lambda op, a, b: algebra.promote_type(a, b),
)


algebra.register_property(
    promote_max,
    "__call__",
    "return_type",
    lambda op, a, b: algebra.promote_type(a, b),
)

algebra.register_property(
    promote_min, "__call__", "init_value", lambda op, arg: algebra.type_max(arg)
)
algebra.register_property(
    promote_max, "__call__", "init_value", lambda op, arg: algebra.type_min(arg)
)


@dataclass(eq=True, frozen=True)
class InitWrite:
    """
    InitWrite may assert that its first argument is equal to `value`, and returns
    its second argument. This is useful when you want to communicate to the compiler
    that the tensor has already been initialized to a specific value.
    """

    value: Any

    def __call__(self, x, y):
        assert x == self.value, f"Expected {self.value}, got {x}"
        return y

    def __str__(self):
        return f"InitWrite({self.value})"


algebra.register_property(
    InitWrite,
    "__call__",
    "return_type",
    lambda op, x, y: y,
)


algebra.register_property(
    InitWrite,
    "__call__",
    "is_annihilator",
    lambda op, val: False,
)


def overwrite(x, y):
    """
    overwrite(x, y) returns y always.
    """
    return y


algebra.register_property(
    overwrite,
    "__call__",
    "return_type",
    lambda op, x, y: y,
)


def first_arg(*args):
    """
    Returns the first argument passed to it.
    """
    return args[0] if args else None


algebra.register_property(
    first_arg,
    "__call__",
    "return_type",
    # args[0] is the function name
    lambda *args: args[1],
)


def identity(x):
    """
    Returns the input value unchanged.
    """
    return x


algebra.register_property(
    identity,
    "__call__",
    "return_type",
    lambda op, x: x,
)


def repeat_operator(x):
    """
    If there exists an operator g such that
    f(x, x, ..., x)  (n times)  is equal to g(x, n),
    then return g.
    """
    if not callable(x):
        raise TypeError("Can't check repeat operator of non-callable objects!")

    if is_idempotent(x):
        return None

    if x is operator.add:
        return operator.mul

    if x is operator.mul:
        return math.exp

    return None


for fn in [
    operator.and_,
    operator.or_,
    min,
    max,
]:
    algebra.register_property(
        fn,
        "__call__",
        "repeat_operator",
        lambda op: None,
    )

algebra.register_property(
    operator.add,
    "__call__",
    "repeat_operator",
    lambda op: operator.mul,
)

algebra.register_property(
    operator.mul,
    "__call__",
    "repeat_operator",
    lambda op: math.exp,
)


def cansplitpush(x, y):
    """
    Return True if a reduction with operator `x` can be 'split-pushed' through
    a pointwise operator `y`.

    We allow split-push when:
      - x has a known repeat operator (repeat_operator(x) is not None),
      - x and y are the same operator,
      - and x is both commutative and associative.
    """
    if not callable(x) or not callable(y):
        raise TypeError("Can't check splitpush of non-callable operators!")

    return (
        repeat_operator(x) is not None
        and x == y
        and is_commutative(x)
        and is_associative(x)
    )


def scansearch(
    arr: np.ndarray, x: np.integer, lo: np.integer, hi: np.integer
) -> np.integer:
    dtype = lo.dtype.type
    u = dtype(1)
    d = dtype(1)
    p = lo

    # searching for binary search bounds
    while p < hi and arr[p] < x:
        d <<= 0x01
        p += d
    lo = p - d
    hi = min(p, hi) + u

    # binary searching within those bounds
    while lo < hi - u:
        m = lo + ((hi - lo) >> 0x01)
        if arr[m] < x:
            lo = m
        else:
            hi = m

    return hi


def make_tuple(*args):
    return tuple(args)
