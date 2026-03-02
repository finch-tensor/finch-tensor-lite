"""
Finch performs extensive rewriting and defining of functions.  The Finch
compiler is designed to inspect objects and functions defined by other
frameworks, such as NumPy. The Finch compiler is designed to be extensible, so that
users can define their own properties and behaviors for objects and functions in
their own code or in third-party libraries.

Finch tracks properties of attributes/methods of objects or classes. Properties
of the object/class itself are accessed with the `__attr__` property.
Properties of functions are properties of their `__call__` method.

You can query a property with `query_property(obj, attr, prop, *args)`. You can
set the property with `register_property(obj, attr, prop, f)`, where `f` is a
function of the form `property(obj, *args)`, where `obj` is the object and
`args` are the arguments to the property.

For example, we might declare that the `__add__` method of a complex number
is associative with the following code:

```python
from finchlite import register_property

register_property(complex, "__add__", "is_associative", lambda obj: True)
```

Finch includes a convenience functions to query each property as well,
for example:
```python
from finchlite import query_property
from operator import add

query_property(complex, "__add__", "is_associative")
# True
is_associative(add, complex, complex)
# True
```

Properties can be inherited in the same way as methods. First we check whether
properties have been defined for the object itself (in the case of functions),
then we check ancestors of that class. For example, if you register a property
for a class, all subclasses of that class will inherit that property. This
allows you to define properties for a class and have them automatically apply to
all subclasses, without having to register the property for each subclass
individually.


Only use the '__attr__' property for attributes which may be overridden by the
user defining an attribute or method of an object or class.  For example, the
`fill_value` attribute of a tensor is defined with the `__attr__` property, so
that if a user defines a custom tensor class, they can override the `__attr__`
property of the `fill_value` attribute by defining a `fill_value` in the class
itself.
"""

import math
import operator
from abc import ABC, abstractmethod
from collections.abc import Hashable
from typing import Any, TypeVar

import numpy as np

_properties: dict[tuple[type | Hashable, str, str], Any] = {}

StableNumber = bool | int | float | complex | np.generic


def query_property(obj: type | Hashable, attr: str, prop: str, *args) -> Any:
    """Queries a property of an attribute of an object or class.  Properties can
    be overridden by calling register_property on the object or it's class.

    Args:
        obj: The object or class of object to query.
        attr: The attribute to query.
        prop: The property to query.
        args: Additional arguments to pass to the property.

    Returns:
        The value of the queried property.

    Raises:
        AttributeError: If the property is not implemented for the given type.
    """
    if not isinstance(obj, type):
        # Only catch TypeError for hashability check
        try:
            hash(obj)
        except TypeError:
            t = type(obj)
        else:
            query_fn = _properties.get((obj, attr, prop))
            if query_fn is not None:
                return query_fn(obj, *args)
            t = type(obj)
    else:
        t = obj

    for ti in t.__mro__:
        query_fn = _properties.get((ti, attr, prop))
        if query_fn is not None:
            return query_fn(obj, *args)

    msg = ""
    obj_name = obj.__name__ if isinstance(obj, type) else type(obj).__name__
    if prop == "__attr__":
        if isinstance(obj, type):
            msg += f"type object '{obj_name}' has no attribute or property '{attr}'. "
        else:
            msg += f"'{obj_name}' object has no attribute or property '{attr}'. "
        msg += "Hint: You may need to register the property by calling "
        if isinstance(obj, Hashable) and not isinstance(obj, type):
            msg += (
                f"`finchlite.register_property({repr(obj)}, '{attr}', '{prop}', "
                "lambda ...)` or "
            )
        msg += (
            f"`finchlite.register_property({obj_name}, '{attr}', '{prop}', lambda ...)`"
        )
        msg += f"or you may define `{obj_name}.{attr}`. "
    elif attr == "__call__":
        msg += f"function '{repr(obj)}' has no property '{prop}'. "
        msg += "Hint: You may need to register the property by calling "
        if isinstance(obj, Hashable) and not isinstance(obj, type):
            msg += (
                f"`finchlite.register_property({repr(obj)}, '{attr}', '{prop}',"
                " lambda ...)` or "
            )
        msg += (
            f"`finchlite.register_property({obj_name},{attr}', '{prop}', lambda ...)`."
        )
    else:
        msg += f"attribute '{obj_name}.{attr}' has no property '{prop}'. "
        msg += "You may need to register the property by calling "
        if isinstance(obj, Hashable) and not isinstance(obj, type):
            msg += (
                f"finchlite.register_property({repr(obj)}, '{attr}', '{prop}'"
                ", lambda ...) or "
            )
        msg += (
            f"`finchlite.register_property({obj_name},"
            f" '{attr}', '{prop}', lambda ...)`."
        )
    msg += (
        " See https://github.com/finch-tensor/finch-tensor-lite/blob/main/src/finch-lite/"
        "algebra/algebra.py for more information."
    )
    raise AttributeError(msg)


def register_property(cls, attr, prop, f):
    """Registers a property for a class or object.

    Args:
        cls: The class or object to register the property for.
        prop: The property to register.
        f: The function to register as the property, which should take the
            object and any additional arguments as input.
    """
    _properties[(cls, attr, prop)] = f


def promote_type(a: Any, b: Any) -> type:
    """Returns the data type with the smallest size and smallest scalar kind to
    which both type1 and type2 may be safely cast.

    Args:
        *args: The types to promote.

    Returns:
        The common type of the given arguments.
    """
    if hasattr(a, "promote_type"):
        res = a.promote_type(b)
        if res is not NotImplemented:
            return res
    if hasattr(b, "promote_type"):
        res = b.promote_type(a)
        if res is not NotImplemented:
            return res
    try:
        return query_property(a, "promote_type", "__attr__", b)
    except AttributeError:
        return query_property(b, "promote_type", "__attr__", a)


def promote_type_stable(a, b) -> type:
    a = type(a) if not isinstance(a, type) else a
    b = type(b) if not isinstance(b, type) else b
    if issubclass(a, np.generic) or issubclass(b, np.generic):
        return np.promote_types(a, b).type
    return type(a(False) + b(False))


for t in StableNumber.__args__:
    register_property(
        t,
        "promote_type",
        "__attr__",
        lambda a, b: promote_type_stable(a, b),
    )


class FinchOperator(ABC):
    is_associative: bool = False
    is_commutative: bool = False
    is_idempotent: bool = False

    @abstractmethod
    def __call__(self, *args: Any) -> Any:
        pass

    @abstractmethod
    def return_type(self, *args: Any) -> Any:
        pass

    def is_distributive(self, other_op: "FinchOperator") -> bool:
        return False

    def is_identity(self, val: Any) -> bool:
        return False

    def is_annihilator(self, val: Any) -> bool:
        return False

    def init_value(self, arg: type) -> Any:
        raise AttributeError(f"{type(self)} has no init_value")

    def repeat_operator(self) -> Any:
        if self.is_idempotent:
            return None
        raise AttributeError(f"{type(self)} has no repeat_operator")


# accessor functions for properties


def is_associative(op: Any) -> bool:
    if op in _operator_map:
        return as_finch_operator(op).is_associative
    return query_property(op, "__call__", "is_associative")


def is_commutative(op: Any) -> bool:
    if op in _operator_map:
        return as_finch_operator(op).is_commutative
    return query_property(op, "__call__", "is_commutative")


def is_idempotent(op: Any) -> bool:
    if op in _operator_map:
        return as_finch_operator(op).is_idempotent
    return query_property(op, "__call__", "is_idempotent")


def is_identity(op: Any, val: Any) -> bool:
    if op in _operator_map:
        return as_finch_operator(op).is_identity(val)
    return query_property(op, "__call__", "is_identity", val)


def is_annihilator(op: Any, val: Any) -> bool:
    if op in _operator_map:
        return as_finch_operator(op).is_annihilator(val)
    return query_property(op, "__call__", "is_annihilator", val)


def is_distributive(op: Any, other_op: Any) -> bool:
    if op in _operator_map:
        return as_finch_operator(op).is_distributive(as_finch_operator(other_op))
    return query_property(op, "__call__", "is_distributive", other_op)


def return_type(op: Any, *args: Any) -> Any:
    if op in _operator_map:
        return as_finch_operator(op).return_type(*args)
    return query_property(op, "__call__", "return_type", *args)


def init_value(op: Any, arg: Any) -> Any:
    if op in _operator_map:
        return as_finch_operator(op).init_value(arg)
    return query_property(op, "__call__", "init_value", arg)


class ReflexiveFinchOperator(FinchOperator):
    def return_type(self, a: Any, b: Any) -> type:
        return type(self(a(True), b(True)))


class UnaryFinchOperator(FinchOperator):
    def return_type(self, a: Any) -> type:
        return type(self(a(True)))


class ComparisonFinchOperator(FinchOperator):
    def return_type(self, a: Any, b: Any) -> type:
        return bool


class Add(ReflexiveFinchOperator):
    is_associative = True
    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return operator.add(a, b)

    def is_identity(self, arg: Any) -> bool:
        return arg == 0

    def is_annihilator(self, arg: Any) -> bool:
        return np.isinf(arg)

    def repeat_operator(self):
        return operator.mul

    def init_value(self, arg: type[Any]) -> Any:
        if arg is bool:
            return 0
        if arg is np.bool_:
            return np.int_(0)
        if issubclass(arg, np.integer):
            return np.int_(0) if issubclass(arg, np.signedinteger) else np.uint(0)
        return arg(0)


class Mul(ReflexiveFinchOperator):
    is_associative = True
    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return operator.mul(a, b)

    def is_identity(self, arg: Any) -> bool:
        return arg == 1

    def repeat_operator(self):
        return operator.pow

    def is_distributive(self, other_op: "FinchOperator") -> bool:
        return isinstance(other_op, (Add, Sub))

    def is_annihilator(self, val):
        return val == 0

    def init_value(self, arg: type[Any]) -> Any:
        return arg(1)


class Sub(ReflexiveFinchOperator):
    def __call__(self, a: Any, b: Any):
        return operator.sub(a, b)


class MatMul(ReflexiveFinchOperator):
    is_associative = True

    def __call__(self, a: Any, b: Any):
        return operator.matmul(a, b)


class TrueDiv(ReflexiveFinchOperator):
    def __call__(self, a: Any, b: Any):
        return operator.truediv(a, b)

    def is_identity(self, arg):
        return arg == 1


class FloorDiv(ReflexiveFinchOperator):
    def __call__(self, a: Any, b: Any):
        return operator.floordiv(a, b)


class Mod(ReflexiveFinchOperator):
    def __call__(self, a: Any, b: Any):
        return operator.mod(a, b)


class DivMod(ReflexiveFinchOperator):
    def __call__(self, a: Any, b: Any):
        return divmod(a, b)


class Pow(ReflexiveFinchOperator):
    def __call__(self, a: Any, b: Any):
        return operator.pow(a, b)

    def is_identity(self, arg):
        return arg == 1

    def is_annihilator(self, arg):
        return arg == 0


class LShift(ReflexiveFinchOperator):
    def __call__(self, a: Any, b: Any):
        return operator.lshift(a, b)

    def is_identity(self, arg):
        return arg == 0


class RShift(ReflexiveFinchOperator):
    def __call__(self, a: Any, b: Any):
        return operator.rshift(a, b)

    def is_identity(self, arg):
        return arg == 0


class And(ReflexiveFinchOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __call__(self, a: Any, b: Any):
        return operator.and_(a, b)

    def is_identity(self, arg):
        return bool(arg)

    def is_annihilator(self, arg):
        return not bool(arg)

    def is_distributive(self, other_op: "FinchOperator") -> bool:
        return isinstance(other_op, (Or, Xor))

    def init_value(self, arg: type[Any]) -> Any:
        return arg(True)


class Xor(ReflexiveFinchOperator):
    is_associative = True
    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return operator.xor(a, b)

    def is_identity(self, arg):
        return arg == 0

    def init_value(self, arg: type[Any]) -> Any:
        return arg(False)


class Or(ReflexiveFinchOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __call__(self, a: Any, b: Any):
        return operator.or_(a, b)

    def is_identity(self, arg):
        return not bool(arg)

    def is_annihilator(self, arg):
        return bool(arg)

    def is_distributive(self, other_op: "FinchOperator") -> bool:
        return isinstance(other_op, And)

    def init_value(self, arg: type[Any]) -> Any:
        return arg(False)


class Abs(UnaryFinchOperator):
    is_idempotent = True

    def __call__(self, a: Any):
        return operator.abs(a)


class Pos(UnaryFinchOperator):
    is_idempotent = True

    def __call__(self, a: Any):
        return operator.pos(a)


class Neg(UnaryFinchOperator):
    def __call__(self, a: Any):
        return operator.neg(a)


class Invert(UnaryFinchOperator):
    def __call__(self, a: Any):
        return operator.invert(a)


class Eq(ComparisonFinchOperator):
    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return operator.eq(a, b)


class Ne(ComparisonFinchOperator):
    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return operator.ne(a, b)


class Gt(ComparisonFinchOperator):
    def __call__(self, a: Any, b: Any):
        return operator.gt(a, b)


class Lt(ComparisonFinchOperator):
    def __call__(self, a: Any, b: Any):
        return operator.lt(a, b)


class Ge(ComparisonFinchOperator):
    def __call__(self, a: Any, b: Any):
        return operator.ge(a, b)


class Le(ComparisonFinchOperator):
    def __call__(self, a: Any, b: Any):
        return operator.le(a, b)


class BinaryFloatOperator(FinchOperator):
    def return_type(self, a: Any, b: Any) -> type:
        return float


class UnaryOperator(FinchOperator):
    def return_type(self, a: Any) -> type:
        if a is np.float16:
            return a
        if a is np.float32:
            return a
        if np.can_cast(a, np.float64):
            return np.float64
        if a is np.complex64:
            return a
        if a is np.complex128:
            return a
        raise TypeError(f"Unsupported operand type for {self}: {a}")


class UnaryBoolOperator(FinchOperator):
    def return_type(self, a: Any) -> type:
        return bool


class BinaryBoolOperator(FinchOperator):
    def return_type(self, a: Any, b: Any) -> type:
        return bool


class LogicalBinaryOperator(BinaryBoolOperator):
    is_associative = True
    is_commutative = True


class Divide(BinaryFloatOperator):
    def __call__(self, a, b):
        return np.divide(a, b)

    def is_identity(self, val) -> bool:
        return val == 1


class LogAddExp(BinaryFloatOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = False

    def __call__(self, a, b):
        return np.logaddexp(a, b)

    def is_identity(self, val) -> bool:
        return val == -math.inf

    def is_annihilator(self, val) -> bool:
        return val == math.inf

    def init_value(self, arg: type[Any]) -> Any:
        return -math.inf


class LogicalAnd(LogicalBinaryOperator):
    is_idempotent = True

    def __call__(self, a, b):
        return np.logical_and(a, b)

    def is_identity(self, val) -> bool:
        return bool(val)

    def is_annihilator(self, val) -> bool:
        return not bool(val)

    def is_distributive(self, other_op: FinchOperator) -> bool:
        return isinstance(other_op, (LogicalOr, LogicalXor))

    def init_value(self, arg: type[Any]) -> Any:
        return True


class LogicalOr(LogicalBinaryOperator):
    is_idempotent = True

    def __call__(self, a, b):
        return np.logical_or(a, b)

    def is_identity(self, val) -> bool:
        return not bool(val)

    def is_annihilator(self, val) -> bool:
        return bool(val)

    def is_distributive(self, other_op: FinchOperator) -> bool:
        return isinstance(other_op, LogicalAnd)

    def init_value(self, arg: type[Any]) -> Any:
        return False


class LogicalXor(LogicalBinaryOperator):
    is_idempotent = False

    def __call__(self, a, b):
        return np.logical_xor(a, b)

    def is_identity(self, val) -> bool:
        return not bool(val)

    def init_value(self, arg: type[Any]) -> Any:
        return False


class LogicalNot(UnaryBoolOperator):
    def __call__(self, a):
        return np.logical_not(a)


class Truth(UnaryBoolOperator):
    def __call__(self, a: Any):
        return bool(a)


class Min(FinchOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __call__(self, a, b):
        return min(a, b)

    def return_type(self, a: Any, b: Any) -> type:
        return type(min(a(True), b(True)))

    def is_identity(self, val) -> bool:
        return val == math.inf

    def init_value(self, arg: type[Any]):
        return type_max(arg)


class Max(FinchOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __call__(self, a, b):
        return max(a, b)

    def return_type(self, a: Any, b: Any) -> type:
        return type(max(a(True), b(True)))

    def is_identity(self, val) -> bool:
        return val == -math.inf

    def init_value(self, arg: type[Any]):
        return type_min(arg)


class Remainder(BinaryFloatOperator):
    def __call__(self, a, b):
        return np.remainder(a, b)


class Hypot(BinaryFloatOperator):
    is_commutative = True

    def __call__(self, a, b):
        return np.hypot(a, b)


class Atan2(BinaryFloatOperator):
    def __call__(self, a, b):
        return np.atan2(a, b)


class Copysign(BinaryFloatOperator):
    def __call__(self, a, b):
        return np.copysign(a, b)


class Nextafter(BinaryFloatOperator):
    def __call__(self, a, b):
        return np.nextafter(a, b)


class IsFinite(UnaryBoolOperator):
    def __call__(self, a):
        return np.isfinite(a)


class IsInf(UnaryBoolOperator):
    def __call__(self, a):
        return np.isinf(a)


class IsNan(UnaryBoolOperator):
    def __call__(self, a):
        return np.isnan(a)


class Real(UnaryOperator):
    def __call__(self, a):
        return np.real(a)

    def return_type(self, a: Any) -> type:
        return float


class Imag(UnaryOperator):
    def __call__(self, a: Any):
        return np.imag(a)

    def return_type(self, a: Any) -> type:
        return float


class Clip(FinchOperator):
    def __call__(self, a: Any, b: Any, c: Any):
        return np.clip(a, b, c)

    def return_type(self, a: Any, b: Any, c: Any) -> type:
        return float


class Equal(BinaryBoolOperator):
    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return np.equal(a, b)


class NotEqual(BinaryBoolOperator):
    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return np.not_equal(a, b)


class Less(BinaryBoolOperator):
    def __call__(self, a: Any, b: Any):
        return np.less(a, b)


class LessEqual(BinaryBoolOperator):
    def __call__(self, a: Any, b: Any):
        return np.less_equal(a, b)


class Greater(BinaryBoolOperator):
    def __call__(self, a: Any, b: Any):
        return np.greater(a, b)


class GreaterEqual(BinaryBoolOperator):
    def __call__(self, a: Any, b: Any):
        return np.greater_equal(a, b)


class Reciprocal(UnaryOperator):
    def __call__(self, a: Any):
        return np.reciprocal(a)


class Sin(UnaryOperator):
    def __call__(self, a: Any):
        return np.sin(a)


class Cos(UnaryOperator):
    def __call__(self, a: Any):
        return np.cos(a)


class Tan(UnaryOperator):
    def __call__(self, a: Any):
        return np.tan(a)


class Sinh(UnaryOperator):
    def __call__(self, a: Any):
        return np.sinh(a)


class Cosh(UnaryOperator):
    def __call__(self, a: Any):
        return np.cosh(a)


class Tanh(UnaryOperator):
    def __call__(self, a: Any):
        return np.tanh(a)


class Atan(UnaryOperator):
    def __call__(self, a: Any):
        return np.atan(a)


class Asinh(UnaryOperator):
    def __call__(self, a: Any):
        return np.asinh(a)


class Asin(UnaryOperator):
    def __call__(self, a: Any):
        return np.asin(a)


class Acos(UnaryOperator):
    def __call__(self, a: Any):
        return np.acos(a)


class Acosh(UnaryOperator):
    def __call__(self, a: Any):
        return np.acosh(a)


class Atanh(UnaryOperator):
    def __call__(self, a: Any):
        return np.atanh(a)


class Round(UnaryOperator):
    is_idempotent = True

    def __call__(self, a: Any):
        return np.round(a)


class Floor(UnaryOperator):
    is_idempotent = True

    def __call__(self, a: Any):
        return np.floor(a)


class Ceil(UnaryOperator):
    is_idempotent = True

    def __call__(self, a: Any):
        return np.ceil(a)


class Trunc(UnaryOperator):
    is_idempotent = True

    def __call__(self, a: Any):
        return np.trunc(a)


class Exp(UnaryOperator):
    def __call__(self, a: Any):
        return np.exp(a)


class Expm1(UnaryOperator):
    def __call__(self, a: Any):
        return np.expm1(a)


class Log(UnaryOperator):
    def __call__(self, a: Any):
        return np.log(a)


class Log1p(UnaryOperator):
    def __call__(self, a: Any):
        return np.log1p(a)


class Log2(UnaryOperator):
    def __call__(self, a: Any):
        return np.log2(a)


class Log10(UnaryOperator):
    def __call__(self, a: Any):
        return np.log10(a)


class Signbit(UnaryBoolOperator):
    def __call__(self, a: Any):
        return np.signbit(a)


class Sqrt(UnaryOperator):
    def __call__(self, a: Any):
        return np.sqrt(a)


class Square(UnaryOperator):
    def __call__(self, a: Any):
        return np.square(a)


class Sign(UnaryOperator):
    def __call__(self, a: Any):
        return np.sign(a)


def fixpoint_type(op: Any, z: Any, t: type) -> type:
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
    r = type(z)
    while r not in s:
        s.add(r)
        r = return_type(op, type(z), t)
    return r


# below is used for types, not for operators

T = TypeVar("T")


def type_min(t: type[T]) -> T:
    """
    Returns the minimum value of the given type.

    Args:
        t: The type to determine the minimum value for.

    Returns:
        The minimum value of the given type.

    Raises:
        AttributeError: If the minimum value is not implemented for the given type.
    """
    if hasattr(t, "type_min"):
        return t.type_min()  # type: ignore[attr-defined]
    return query_property(t, "type_min", "__attr__")


for t, tn in [
    (bool, lambda x: -math.inf),
    (int, lambda x: -math.inf),
    (float, lambda x: -math.inf),
    (np.bool_, lambda x: x(False)),
    (np.integer, lambda x: np.iinfo(x).min),
    (np.floating, lambda x: np.finfo(x).min),
]:
    register_property(t, "type_min", "__attr__", tn)


def type_max(t: type[T]) -> T:
    """
    Returns the maximum value of the given type.

    Args:
        t: The type to determine the maximum value for.

    Returns:
        The maximum value of the given type.

    Raises:
        AttributeError: If the maximum value is not implemented for the given type.
    """
    if hasattr(t, "type_max"):
        return t.type_max()  # type: ignore[attr-defined]
    return query_property(t, "type_max", "__attr__")


for t, tn in [
    (bool, lambda x: math.inf),
    (int, lambda x: math.inf),
    (float, lambda x: math.inf),
    (np.bool_, lambda x: x(True)),
    (np.integer, lambda x: np.iinfo(x).max),
    (np.floating, lambda x: np.finfo(x).max),
]:
    register_property(t, "type_max", "__attr__", tn)


def sum_init_value(t):
    if t is bool:
        return 0
    if t is np.bool_:
        return np.int_(0)
    if issubclass(t, np.integer):
        if issubclass(t, np.signedinteger):
            return np.int_(0)
        return np.uint(0)
    return t(0)


# functions ported from ops.py


def and_test(a, b):
    return a & b


def or_test(a, b):
    return a | b


def not_test(a):
    return not a


def ifelse(a, b, c):
    return a if c else b


def make_tuple(*args):
    return tuple(args)


def identity(x):
    """
    Returns the input value unchanged.
    """
    return x


def first_arg(*args):
    """
    Returns the first argument passed to it.
    """
    return args[0] if args else None


def overwrite(x, y):
    """
    overwrite(x, y) returns y always.
    """
    return y


def promote_min(a, b):
    cast = promote_type(a, b)
    return cast(min(a, b))


def promote_max(a, b):
    cast = promote_type(a, b)
    return max(cast(a), cast(b))


class PromoteMin(FinchOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __call__(self, a: Any, b: Any):
        cast = promote_type(type(a), type(b))
        return cast(min(a, b))

    def return_type(self, a: Any, b: Any) -> type:
        return promote_type(a, b)

    def init_value(self, arg):
        return type_max(arg)


class PromoteMax(FinchOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __call__(self, a: Any, b: Any):
        cast = promote_type(type(a), type(b))
        return max(cast(a), cast(b))

    def return_type(self, a: Any, b: Any) -> type:
        return promote_type(a, b)

    def init_value(self, arg):
        return type_min(arg)


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


class InitWrite(FinchOperator):
    """
    InitWrite may assert that its first argument is
    equal to z, and returns its second argument. This is useful when you want to
    communicate to the compiler that the tensor has already been initialized to
    a specific value.
    """

    def __init__(self, value):
        self.value = value

    def __call__(self, x: Any, y: Any):
        assert x == self.value, f"Expected {self.value}, got {x}"
        return y

    def return_type(self, x: Any, y: Any) -> type:
        return y


class Overwrite(FinchOperator):
    """
    Overwrite(x, y) returns y always.
    """

    def __call__(self, x: Any, y: Any):
        return y

    def return_type(self, x: Any, y: Any) -> type:
        return y


class FirstArg(FinchOperator):
    """
    Returns the first argument passed to it.
    """

    def __call__(self, *args):
        return args[0] if args else None

    def return_type(self, *args) -> type:
        return args[0]


class Identity(FinchOperator):
    """
    Returns the input value unchanged.
    """

    is_idempotent = True

    def __call__(self, x: Any):
        return x

    def return_type(self, x: Any) -> type:
        return x


class Conjugate(FinchOperator):
    """
    Returns the input value unchanged.
    """

    def __call__(self, x: Any):
        return np.conjugate(x)

    def return_type(self, x: Any) -> type:
        return x


class MakeTuple(FinchOperator):
    is_commutative = False
    is_associative = False

    def __call__(self, *args: Any) -> Any:
        return tuple(args)

    def return_type(self, *args: Any) -> Any:
        from finchlite.finch_assembly.struct import TupleFType

        return TupleFType.from_tuple(args)


def repeat_operator(x):
    """
    If there exists an operator g such that
    f(x, x, ..., x)  (n times)  is equal to g(x, n),
    then return g.
    """
    return as_finch_operator(x).repeat_operator()


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


_operator_map: dict[Any, FinchOperator] = {
    # Python Operators
    operator.add: Add(),
    operator.mul: Mul(),
    operator.sub: Sub(),
    operator.matmul: MatMul(),
    operator.truediv: TrueDiv(),
    operator.floordiv: FloorDiv(),
    operator.truth: Truth(),
    operator.mod: Mod(),
    operator.pow: Pow(),
    operator.lshift: LShift(),
    operator.rshift: RShift(),
    operator.and_: And(),
    operator.xor: Xor(),
    operator.or_: Or(),
    operator.abs: Abs(),
    operator.pos: Pos(),
    operator.neg: Neg(),
    operator.invert: Invert(),
    operator.eq: Eq(),
    operator.ne: Ne(),
    operator.gt: Gt(),
    operator.lt: Lt(),
    operator.ge: Ge(),
    operator.le: Le(),
    min: Min(),
    max: Max(),
    divmod: DivMod(),
    abs: Abs(),
    # NumPy Functions
    np.divide: Divide(),
    np.remainder: Remainder(),
    np.hypot: Hypot(),
    np.atan2: Atan2(),
    np.copysign: Copysign(),
    np.nextafter: Nextafter(),
    np.logaddexp: LogAddExp(),
    np.logical_and: LogicalAnd(),
    np.logical_or: LogicalOr(),
    np.logical_xor: LogicalXor(),
    np.logical_not: LogicalNot(),
    np.isfinite: IsFinite(),
    np.isinf: IsInf(),
    np.isnan: IsNan(),
    np.real: Real(),
    np.imag: Imag(),
    np.clip: Clip(),
    np.equal: Equal(),
    np.not_equal: NotEqual(),
    np.less: Less(),
    np.less_equal: LessEqual(),
    np.greater: Greater(),
    np.greater_equal: GreaterEqual(),
    np.reciprocal: Reciprocal(),
    np.sin: Sin(),
    np.cos: Cos(),
    np.tan: Tan(),
    np.sinh: Sinh(),
    np.cosh: Cosh(),
    np.tanh: Tanh(),
    np.atan: Atan(),
    np.asinh: Asinh(),
    np.asin: Asin(),
    np.acos: Acos(),
    np.acosh: Acosh(),
    np.atanh: Atanh(),
    np.round: Round(),
    np.floor: Floor(),
    np.ceil: Ceil(),
    np.trunc: Trunc(),
    np.exp: Exp(),
    np.expm1: Expm1(),
    np.log: Log(),
    np.log1p: Log1p(),
    np.log2: Log2(),
    np.log10: Log10(),
    np.signbit: Signbit(),
    np.sqrt: Sqrt(),
    np.square: Square(),
    np.sign: Sign(),
    conjugate: Conjugate(),
    promote_min: PromoteMin(),
    overwrite: Overwrite(),
    first_arg: FirstArg(),
    identity: Identity(),
    promote_max: PromoteMax(),
    make_tuple: MakeTuple(),
}


def as_finch_operator(f: Any) -> FinchOperator:
    # Given an operator, returns its FinchOperator equivalent by
    # looking up the operator map.
    if isinstance(f, FinchOperator):
        return f
    if f in _operator_map:
        return _operator_map[f]
    raise TypeError(f"No FinchOperator registered for {f}. ")
