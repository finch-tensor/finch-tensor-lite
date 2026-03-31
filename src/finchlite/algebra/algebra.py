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

import operator
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Hashable
from functools import reduce
from typing import Any

import numpy as np

_properties: dict[tuple[type | Hashable, str, str], Any] = {}


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
            f"`finchlite.register_property({obj_name}, '{attr}', '{prop}', "
            "lambda ...)`."
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
    return promote_type_stable(a, b)


def promote_type_stable(a: Any, b: Any) -> type:
    a = type(a) if not isinstance(a, type) else a
    b = type(b) if not isinstance(b, type) else b
    if issubclass(a, np.generic) or issubclass(b, np.generic):
        return np.promote_types(a, b).type
    return type(a(False) + b(False))


class COperator(ABC):
    @property
    @abstractmethod
    def c_symbol(self) -> str:
        pass

    @abstractmethod
    def c_function_call(self, ctx: Any, *args: Any) -> Any:
        pass


class CNAryOperator(COperator):
    def c_function_call(self, ctx: Any, *args: Any) -> Any:
        assert len(args) > 0

        if len(args) == 1:
            return f"{self.c_symbol}{ctx(args[0])}"
        return f" {self.c_symbol} ".join(map(ctx, args))


class CBinaryOperator(COperator):
    def c_function_call(self, ctx: Any, *args: Any) -> Any:
        a, b = args
        return f"{ctx(a)} {self.c_symbol} {ctx(b)}"


class CNUnaryOperator(COperator):
    def c_function_call(self, ctx: Any, *args: Any) -> Any:
        return f"{self.c_symbol}{ctx(args[0])}"


class NumbaOperator:
    def numba_literal(self, val: Any, ctx: Any, *args: Any) -> Any:
        return f"({f' {self.numba_name()} '.join(map(ctx, args))})"

    def numba_name(self) -> str:
        raise NotImplementedError(f"{type(self)} must implement numba_name")


# Abstract Base Class for Algebraic Properties
class SingletonMeta(ABCMeta):
    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__call__(*args, **kwargs)
            cls._instance.__qualname__ = cls.__qualname__
        return cls._instance


class FinchOperator(ABC):
    is_associative: bool = False
    is_commutative: bool = False
    is_idempotent: bool = False

    @abstractmethod
    def __call__(self, *args: Any) -> Any:
        pass

    @abstractmethod
    def return_type(self, *args: Any) -> type:
        pass

    def is_distributive(self, other_op: "FinchOperator") -> bool:
        return False

    def is_identity(self, val: Any) -> bool:
        return False

    def is_annihilator(self, val: Any) -> bool:
        return False

    def init_value(self, type_: type) -> Any:
        raise AttributeError(f"{type(self)} has no init_value")

    def repeat_operator(self) -> Any:
        if self.is_idempotent:
            return None
        raise AttributeError(f"{type(self)} has no repeat_operator")


# accessor functions for properties


def is_associative(op: Any) -> bool:
    return as_finch_operator(op).is_associative


def is_commutative(op: Any) -> bool:
    return as_finch_operator(op).is_commutative


def is_idempotent(op: Any) -> bool:
    return as_finch_operator(op).is_idempotent


def is_identity(op: Any, val: Any) -> bool:
    return as_finch_operator(op).is_identity(val)


def is_annihilator(op: Any, val: Any) -> bool:
    return as_finch_operator(op).is_annihilator(val)


def is_distributive(op: Any, other_op: Any) -> bool:
    return as_finch_operator(op).is_distributive(as_finch_operator(other_op))


def return_type(op: Any, *args: Any) -> Any:
    return as_finch_operator(op).return_type(*args)


def init_value(op: Any, arg: Any) -> Any:
    return as_finch_operator(op).init_value(arg)


class ReflexiveFinchOperator(FinchOperator):
    def return_type(self, a: Any, b: Any) -> type:
        return type(self(a(True), b(True)))


class UnaryFinchOperator(FinchOperator):
    def return_type(self, a: Any) -> type:
        return type(self(a(True)))


class ComparisonFinchOperator(FinchOperator):
    def return_type(self, a: Any, b: Any) -> type:
        return bool


class Add(
    ReflexiveFinchOperator, CNAryOperator, NumbaOperator, metaclass=SingletonMeta
):
    is_associative = True
    is_commutative = True

    @property
    def c_symbol(self) -> str:
        return "+"

    def __call__(self, *args: Any) -> Any:
        return reduce(operator.add, args)

    def is_identity(self, arg: Any) -> bool:
        return arg == 0

    def is_annihilator(self, arg: Any) -> bool:
        return np.isinf(arg)

    def repeat_operator(self):
        return _operator_map[operator.mul]

    def init_value(self, type_: type) -> Any:
        return type_(0)

    def numba_name(self) -> str:
        return "+"


class Mul(
    ReflexiveFinchOperator, CNAryOperator, NumbaOperator, metaclass=SingletonMeta
):
    is_associative = True
    is_commutative = True

    @property
    def c_symbol(self) -> str:
        return "*"

    def __call__(self, *args: Any) -> Any:
        return reduce(operator.mul, args)

    def is_identity(self, arg: Any) -> bool:
        return arg == 1

    def repeat_operator(self):
        return _operator_map[operator.pow]

    def is_distributive(self, other_op: "FinchOperator") -> bool:
        return isinstance(other_op, (Add, Sub))

    def is_annihilator(self, val):
        return val == 0

    def init_value(self, type_: type) -> Any:
        return type_(1)

    def numba_name(self) -> str:
        return "*"


class Sub(
    ReflexiveFinchOperator, CBinaryOperator, NumbaOperator, metaclass=SingletonMeta
):
    @property
    def c_symbol(self) -> str:
        return "-"

    def __call__(self, a: Any, b: Any):
        return operator.sub(a, b)

    def numba_name(self) -> str:
        return "-"


class MatMul(ReflexiveFinchOperator, metaclass=SingletonMeta):
    is_associative = True

    def __call__(self, a: Any, b: Any):
        return operator.matmul(a, b)


class TrueDiv(ReflexiveFinchOperator, CBinaryOperator, metaclass=SingletonMeta):
    @property
    def c_symbol(self) -> str:
        return "/"

    def __call__(self, a: Any, b: Any):
        return operator.truediv(a, b)

    def is_identity(self, arg):
        return arg == 1


class FloorDiv(ReflexiveFinchOperator, CBinaryOperator, metaclass=SingletonMeta):
    @property
    def c_symbol(self) -> str:
        return "/"

    def __call__(self, a: Any, b: Any):
        return operator.floordiv(a, b)


class Mod(ReflexiveFinchOperator, CBinaryOperator, metaclass=SingletonMeta):
    @property
    def c_symbol(self) -> str:
        return "%"

    def __call__(self, a: Any, b: Any):
        return operator.mod(a, b)


class DivMod(ReflexiveFinchOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any, b: Any):
        return divmod(a, b)


class Pow(ReflexiveFinchOperator, COperator, metaclass=SingletonMeta):
    @property
    def c_symbol(self) -> str:
        return "pow"

    def c_function_call(self, ctx: Any, *args: Any) -> Any:
        a, b = args
        return f"pow({ctx(a)}, {ctx(b)})"

    def __call__(self, a: Any, b: Any):
        return operator.pow(a, b)

    def is_identity(self, arg):
        return arg == 1

    def is_annihilator(self, arg):
        return arg == 0


class LShift(ReflexiveFinchOperator, CBinaryOperator, metaclass=SingletonMeta):
    @property
    def c_symbol(self) -> str:
        return "<<"

    def __call__(self, a: Any, b: Any):
        return operator.lshift(a, b)

    def is_identity(self, arg):
        return arg == 0


class RShift(ReflexiveFinchOperator, CBinaryOperator, metaclass=SingletonMeta):
    @property
    def c_symbol(self) -> str:
        return ">>"

    def __call__(self, a: Any, b: Any):
        return operator.rshift(a, b)

    def is_identity(self, arg):
        return arg == 0


class And(ReflexiveFinchOperator, CNAryOperator, metaclass=SingletonMeta):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    @property
    def c_symbol(self) -> str:
        return "&"

    def __call__(self, *args: Any) -> Any:
        return reduce(operator.and_, args)

    def is_identity(self, arg):
        return bool(arg)

    def is_annihilator(self, arg):
        return not bool(arg)

    def is_distributive(self, other_op: "FinchOperator") -> bool:
        return isinstance(other_op, (Or, Xor))

    def init_value(self, type_: type) -> Any:
        return type_(True)


class Xor(ReflexiveFinchOperator, CNAryOperator, metaclass=SingletonMeta):
    is_associative = True
    is_commutative = True

    @property
    def c_symbol(self) -> str:
        return "^"

    def __call__(self, *args: Any) -> Any:
        return reduce(operator.xor, args)

    def is_identity(self, arg):
        return arg == 0

    def init_value(self, type_: type) -> Any:
        return type_(False)


class Or(ReflexiveFinchOperator, CNAryOperator, metaclass=SingletonMeta):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    @property
    def c_symbol(self) -> str:
        return "|"

    def __call__(self, *args: Any) -> Any:
        return reduce(operator.or_, args)

    def is_identity(self, arg):
        return not bool(arg)

    def is_annihilator(self, arg):
        return bool(arg)

    def is_distributive(self, other_op: "FinchOperator") -> bool:
        return isinstance(other_op, And)

    def init_value(self, type_: type) -> Any:
        return type_(False)


class Not(CNUnaryOperator):
    @property
    def c_symbol(self) -> str:
        return "!"


class Abs(UnaryFinchOperator, metaclass=SingletonMeta):
    is_idempotent = True

    def __call__(self, a: Any):
        return operator.abs(a)


class Pos(UnaryFinchOperator, metaclass=SingletonMeta):
    is_idempotent = True

    def __call__(self, a: Any):
        return operator.pos(a)


class Neg(UnaryFinchOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any):
        return operator.neg(a)


class Invert(UnaryFinchOperator, CNUnaryOperator, metaclass=SingletonMeta):
    @property
    def c_symbol(self) -> str:
        return "~"

    def __call__(self, a: Any):
        return operator.invert(a)


class Eq(
    ComparisonFinchOperator, CBinaryOperator, NumbaOperator, metaclass=SingletonMeta
):
    @property
    def c_symbol(self) -> str:
        return "=="

    def numba_name(self) -> str:
        return "=="

    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return operator.eq(a, b)


class Ne(ComparisonFinchOperator, CBinaryOperator, metaclass=SingletonMeta):
    @property
    def c_symbol(self) -> str:
        return "!="

    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return operator.ne(a, b)


class Gt(ComparisonFinchOperator, CBinaryOperator, metaclass=SingletonMeta):
    @property
    def c_symbol(self) -> str:
        return ">"

    def __call__(self, a: Any, b: Any):
        return operator.gt(a, b)


class Lt(
    ComparisonFinchOperator, CBinaryOperator, NumbaOperator, metaclass=SingletonMeta
):
    @property
    def c_symbol(self) -> str:
        return "<"

    def numba_name(self) -> str:
        return "<"

    def __call__(self, a: Any, b: Any):
        return operator.lt(a, b)


class Ge(ComparisonFinchOperator, CBinaryOperator, metaclass=SingletonMeta):
    @property
    def c_symbol(self) -> str:
        return ">="

    def __call__(self, a: Any, b: Any):
        return operator.ge(a, b)


class Le(ComparisonFinchOperator, CBinaryOperator, metaclass=SingletonMeta):
    @property
    def c_symbol(self) -> str:
        return "<="

    def __call__(self, a: Any, b: Any):
        return operator.le(a, b)


class BinaryFloatOperator(FinchOperator):
    def return_type(self, a: Any, b: Any) -> type:
        return float


class UnaryOperator(FinchOperator):
    def return_type(self, a: Any) -> type:
        # TODO: Temporary implementation
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


class Divide(BinaryFloatOperator, metaclass=SingletonMeta):
    def __call__(self, a, b):
        return np.divide(a, b)

    def is_identity(self, val) -> bool:
        return val == 1


class LogAddExp(BinaryFloatOperator, metaclass=SingletonMeta):
    is_associative = True
    is_commutative = True
    is_idempotent = False

    def __call__(self, a, b):
        return np.logaddexp(a, b)

    def is_identity(self, val) -> bool:
        return val == -np.inf

    def is_annihilator(self, val) -> bool:
        return val == np.inf

    def init_value(self, type_: type) -> Any:
        return -np.inf


class LogicalAnd(LogicalBinaryOperator, metaclass=SingletonMeta):
    is_idempotent = True

    def __call__(self, a, b):
        return np.logical_and(a, b)

    def is_identity(self, val) -> bool:
        return bool(val)

    def is_annihilator(self, val) -> bool:
        return not bool(val)

    def is_distributive(self, other_op: FinchOperator) -> bool:
        return isinstance(other_op, (LogicalOr, LogicalXor))

    def init_value(self, type_: type) -> Any:
        return True


class LogicalOr(LogicalBinaryOperator, metaclass=SingletonMeta):
    is_idempotent = True

    def __call__(self, a, b):
        return np.logical_or(a, b)

    def is_identity(self, val) -> bool:
        return not bool(val)

    def is_annihilator(self, val) -> bool:
        return bool(val)

    def is_distributive(self, other_op: FinchOperator) -> bool:
        return isinstance(other_op, LogicalAnd)

    def init_value(self, type_: type) -> Any:
        return False


class LogicalXor(LogicalBinaryOperator, metaclass=SingletonMeta):
    is_idempotent = False

    def __call__(self, a, b):
        return np.logical_xor(a, b)

    def is_identity(self, val) -> bool:
        return not bool(val)

    def init_value(self, type_: type) -> Any:
        return False


class LogicalNot(UnaryBoolOperator, metaclass=SingletonMeta):
    def __call__(self, a):
        return np.logical_not(a)


class Truth(UnaryBoolOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any):
        return bool(a)


class Min(FinchOperator, NumbaOperator, metaclass=SingletonMeta):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __call__(self, a, b):
        return min(a, b)

    def return_type(self, a: Any, b: Any) -> type:
        return type(min(a(True), b(True)))

    def is_identity(self, val) -> bool:
        return val == np.inf

    def init_value(self, type_: type):
        return type_max(type_)

    def numba_name(self) -> str:
        return "min"

    def numba_literal(self, val: Any, ctx: Any, *args: Any) -> Any:
        return f"min({', '.join(map(ctx, args))})"


class Max(FinchOperator, NumbaOperator, metaclass=SingletonMeta):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __call__(self, a, b):
        return max(a, b)

    def return_type(self, a: Any, b: Any) -> type:
        return type(max(a(True), b(True)))

    def is_identity(self, val) -> bool:
        return val == -np.inf

    def init_value(self, type_: type):
        return type_min(type_)

    def numba_name(self) -> str:
        return "max"

    def numba_literal(self, val: Any, ctx: Any, *args: Any) -> Any:
        return f"max({', '.join(map(ctx, args))})"


class Remainder(BinaryFloatOperator, metaclass=SingletonMeta):
    def __call__(self, a, b):
        return np.remainder(a, b)


class Hypot(BinaryFloatOperator, metaclass=SingletonMeta):
    is_commutative = True

    def __call__(self, a, b):
        return np.hypot(a, b)


class Atan2(BinaryFloatOperator, metaclass=SingletonMeta):
    def __call__(self, a, b):
        return np.atan2(a, b)


class Copysign(BinaryFloatOperator, metaclass=SingletonMeta):
    def __call__(self, a, b):
        return np.copysign(a, b)


class Nextafter(BinaryFloatOperator, metaclass=SingletonMeta):
    def __call__(self, a, b):
        return np.nextafter(a, b)


class IsFinite(UnaryBoolOperator, metaclass=SingletonMeta):
    def __call__(self, a):
        return np.isfinite(a)


class IsInf(UnaryBoolOperator, metaclass=SingletonMeta):
    def __call__(self, a):
        return np.isinf(a)


class IsNan(UnaryBoolOperator, metaclass=SingletonMeta):
    def __call__(self, a):
        return np.isnan(a)


class Real(UnaryOperator, metaclass=SingletonMeta):
    def __call__(self, a):
        return np.real(a)

    def return_type(self, a: Any) -> type:
        return float


class Imag(UnaryOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any):
        return np.imag(a)

    def return_type(self, a: Any) -> type:
        return float


class Clip(FinchOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any, b: Any, c: Any):
        return np.clip(a, b, c)

    def return_type(self, a: Any, b: Any, c: Any) -> type:
        return float


class Equal(BinaryBoolOperator, metaclass=SingletonMeta):
    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return np.equal(a, b)


class NotEqual(BinaryBoolOperator, metaclass=SingletonMeta):
    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return np.not_equal(a, b)


class Less(BinaryBoolOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any, b: Any):
        return np.less(a, b)


class LessEqual(BinaryBoolOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any, b: Any):
        return np.less_equal(a, b)


class Greater(BinaryBoolOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any, b: Any):
        return np.greater(a, b)


class GreaterEqual(BinaryBoolOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any, b: Any):
        return np.greater_equal(a, b)


class Reciprocal(UnaryOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any):
        return np.reciprocal(a)


class Sin(UnaryOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any):
        return np.sin(a)


class Cos(UnaryOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any):
        return np.cos(a)


class Tan(UnaryOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any):
        return np.tan(a)


class Sinh(UnaryOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any):
        return np.sinh(a)


class Cosh(UnaryOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any):
        return np.cosh(a)


class Tanh(UnaryOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any):
        return np.tanh(a)


class Atan(UnaryOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any):
        return np.atan(a)


class Asinh(UnaryOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any):
        return np.asinh(a)


class Asin(UnaryOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any):
        return np.asin(a)


class Acos(UnaryOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any):
        return np.acos(a)


class Acosh(UnaryOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any):
        return np.acosh(a)


class Atanh(UnaryOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any):
        return np.atanh(a)


class Round(UnaryOperator, metaclass=SingletonMeta):
    is_idempotent = True

    def __call__(self, a: Any):
        return np.round(a)


class Floor(UnaryOperator, metaclass=SingletonMeta):
    is_idempotent = True

    def __call__(self, a: Any):
        return np.floor(a)


class Ceil(UnaryOperator, metaclass=SingletonMeta):
    is_idempotent = True

    def __call__(self, a: Any):
        return np.ceil(a)


class Trunc(UnaryOperator, metaclass=SingletonMeta):
    is_idempotent = True

    def __call__(self, a: Any):
        return np.trunc(a)


class Exp(UnaryOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any):
        return np.exp(a)


class Expm1(UnaryOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any):
        return np.expm1(a)


class Log(UnaryOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any):
        return np.log(a)


class Log1p(UnaryOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any):
        return np.log1p(a)


class Log2(UnaryOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any):
        return np.log2(a)


class Log10(UnaryOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any):
        return np.log10(a)


class Signbit(UnaryBoolOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any):
        return np.signbit(a)


class Sqrt(UnaryOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any):
        return np.sqrt(a)


class Square(UnaryOperator, metaclass=SingletonMeta):
    def __call__(self, a: Any):
        return np.square(a)


class Sign(UnaryOperator, metaclass=SingletonMeta):
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


def type_min(type_: type) -> Any:
    """
    Returns the minimum value of the given type.

    Args:
        type_: The type to determine the minimum value for.

    Returns:
        The minimum value of the given type.

    Raises:
        AttributeError: If the minimum value is not implemented for the given type.
    """
    if type_ in (bool, np.bool_):
        return np.bool_(False)
    if issubclass(type_, (int, np.integer)):
        return np.iinfo(type_).min
    if issubclass(type_, (float, np.floating)):
        return np.finfo(type_).min
    raise Exception(f"Unsupported type for type_min: {type_}")


def type_max(type_: type) -> Any:
    """
    Returns the maximum value of the given type.

    Args:
        type_: The type to determine the maximum value for.

    Returns:
        The maximum value of the given type.

    Raises:
        AttributeError: If the maximum value is not implemented for the given type.
    """
    if type_ in (bool, np.bool_):
        return np.bool_(True)
    if issubclass(type_, (int, np.integer)):
        return np.iinfo(type_).max
    if issubclass(type_, (float, np.floating)):
        return np.finfo(type_).max
    raise Exception(f"Unsupported type for type_max: {type_}")


# functions ported from ops.py


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


class PromoteMin(FinchOperator, metaclass=SingletonMeta):
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


class PromoteMax(FinchOperator, metaclass=SingletonMeta):
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


class InitWrite(FinchOperator, NumbaOperator):
    """
    InitWrite may assert that its first argument is
    equal to z, and returns its second argument. This is useful when you want to
    communicate to the compiler that the tensor has already been initialized to
    a specific value.
    """

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, InitWrite) and self.value == other.value

    def __hash__(self):
        return hash((self.value,))

    def __call__(self, x: Any, y: Any):
        assert x == self.value, f"Expected {self.value}, got {x}"
        return y

    def return_type(self, x: Any, y: Any) -> type:
        return y

    def numba_literal(self, val: Any, ctx: Any, *args: Any) -> Any:
        return ctx(args[1])


class Overwrite(FinchOperator, metaclass=SingletonMeta):
    """
    Overwrite(x, y) returns y always.
    """

    def __call__(self, x: Any, y: Any):
        return y

    def return_type(self, x: Any, y: Any) -> type:
        return y


class FirstArg(FinchOperator, metaclass=SingletonMeta):
    """
    Returns the first argument passed to it.
    """

    def __call__(self, *args):
        return args[0] if args else None

    def return_type(self, *args) -> type:
        return args[0]


class Identity(FinchOperator, metaclass=SingletonMeta):
    """
    Returns the input value unchanged.
    """

    is_idempotent = True

    def __call__(self, x: Any):
        return x

    def return_type(self, x: Any) -> type:
        return x


class Conjugate(FinchOperator, metaclass=SingletonMeta):
    """
    Returns the complex conjugate of the input value.
    """

    def __call__(self, x: Any):
        return np.conjugate(x)

    def return_type(self, x: Any) -> type:
        return x


class MakeTuple(FinchOperator, NumbaOperator, metaclass=SingletonMeta):
    is_commutative = False
    is_associative = False

    def __call__(self, *args: Any) -> tuple:
        return tuple(args)

    def return_type(self, *args: Any) -> Any:
        from finchlite.finch_assembly.struct import TupleFType

        return TupleFType.from_tuple(args)

    def numba_literal(self, val: Any, ctx: Any, *args: Any):
        return f"({','.join([ctx(arg) for arg in args])},)"


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


class Scansearch(FinchOperator, NumbaOperator, metaclass=SingletonMeta):
    """
    Scansearch is a search operator that performs a scan search on a sorted array.

    It takes an array `arr`, a value `x`, and search bounds `lo` and `hi`, and returns
    the index of the smallest element in `arr` that is greater than or equal to `x`.
    If all elements in `arr` are less than `x`, it returns `hi`.
    """

    @staticmethod
    def _func(
        arr: np.ndarray, x: np.integer, lo: np.integer, hi: np.integer
    ) -> np.integer:
        dtype = np.array(lo).dtype.type
        u = dtype(1)
        d = dtype(1)
        p = lo

        # searching for binary search bounds
        while p < hi and arr[p] < x:
            d <<= 0x01
            p += d
        lo = p - d
        hi = min(p, hi) + u  # type: ignore[call-overload]

        # binary searching within those bounds
        while lo < hi - u:
            m = lo + ((hi - lo) >> 0x01)
            if arr[m] < x:
                lo = m
            else:
                hi = m

        return hi

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def return_type(self, arr, x, lo, hi) -> type:
        return hi

    def numba_literal(self, val: Any, ctx: Any, *args: Any) -> Any:
        arr = args[0]
        x = args[1]
        lo = args[2]
        hi = args[3]
        return f"scansearch({ctx(arr)}, {ctx(x)}, {ctx(lo)}, {ctx(hi)})"


scansearch = Scansearch()

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
    # Finch Functions
    conjugate: Conjugate(),
    promote_min: PromoteMin(),
    promote_max: PromoteMax(),
    overwrite: Overwrite(),
    first_arg: FirstArg(),
    identity: Identity(),
    make_tuple: MakeTuple(),
    scansearch: Scansearch(),
}


def as_finch_operator(f: Any) -> FinchOperator:
    """
    Given an operator, returns its FinchOperator equivalent by
    looking up the operator map.
    """
    if isinstance(f, FinchOperator):
        return f
    if f in _operator_map:
        return _operator_map[f]
    raise TypeError(f"No FinchOperator registered for {f}. ")
