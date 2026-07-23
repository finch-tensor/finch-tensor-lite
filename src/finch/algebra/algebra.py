"""Algebraic interfaces and helpers used by Finch operators."""

from abc import ABC, abstractmethod
from typing import Any

from .ftypes import FDTypeOrdered, FType, ftype


class FinchOperator(ABC):
    is_associative: bool = False
    is_commutative: bool = False
    is_idempotent: bool = False

    @abstractmethod
    def __call__(self, *args: Any) -> Any:
        pass

    @abstractmethod
    def return_type(self, *args: FType) -> FType:
        pass

    def is_distributive(self, other_op: "FinchOperator") -> bool:
        return False

    def is_identity(self, val: Any) -> bool:
        return False

    def is_annihilator(self, val: Any) -> bool:
        return False

    def init_value(self, type_: FType) -> Any:
        raise AttributeError(f"{type(self)} has no init_value")

    def repeat_operator(self) -> Any:
        if self.is_idempotent:
            return None
        raise AttributeError(f"{type(self)} has no repeat_operator")

    def __qual_str__(self) -> str:
        """Return qualified string for printing/display purposes."""
        # Display as just the lowercase name
        return repr(self)


def is_associative(op: FinchOperator) -> bool:
    return op.is_associative


def is_commutative(op: FinchOperator) -> bool:
    return op.is_commutative


def is_idempotent(op: FinchOperator) -> bool:
    return op.is_idempotent


def is_identity(op: FinchOperator, val: Any) -> bool:
    return op.is_identity(val)


def is_annihilator(op: FinchOperator, val: Any) -> bool:
    return op.is_annihilator(val)


def is_distributive(op: FinchOperator, other_op: FinchOperator) -> bool:
    return op.is_distributive(other_op)


def return_type(op: FinchOperator, *args: FType) -> FType:
    arg_types = tuple(ftype(arg) for arg in args)
    return op.return_type(*arg_types)


def init_value(op: FinchOperator, arg: Any) -> Any:
    return op.init_value(arg)


def fixpoint_type(op: FinchOperator, z: Any, t: FType) -> FType:
    """
    Determines the fixpoint type after repeated calling the given operation.

    Args:
        op: The operation to evaluate.
        z: The initial value.
        t: The type to evaluate against.

    Returns:
        The fixpoint type.
    """
    s = set()
    z_type = ftype(z)
    r = z_type
    while r not in s:
        s.add(r)
        r = return_type(op, z_type, t)
    return r


def type_min(type_: FDTypeOrdered) -> Any:
    """
    Returns the minimum value of the given type.

    Args:
        type_: The type to determine the minimum value for.

    Returns:
        The minimum value of the given type.

    Raises:
        AttributeError: If the minimum value is not implemented for the given type.
    """
    return type_.type_min


def type_max(type_: FDTypeOrdered) -> Any:
    """
    Returns the maximum value of the given type.

    Args:
        type_: The type to determine the maximum value for.

    Returns:
        The maximum value of the given type.

    Raises:
        AttributeError: If the maximum value is not implemented for the given type.
    """
    return type_.type_max


def repeat_operator(op: FinchOperator):
    """
    If there exists an operator g such that
    f(x, x, ..., x)  (n times)  is equal to g(x, n),
    then return g.
    """
    return op.repeat_operator()


def cansplitpush(x: FinchOperator, y: FinchOperator):
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
