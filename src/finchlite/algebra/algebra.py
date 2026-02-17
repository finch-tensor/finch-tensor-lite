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
from collections.abc import Callable, Hashable
from typing import Any, TypeVar
from abc import ABC, abstractmethod


import numpy as np

_properties: dict[tuple[type | Hashable, str, str], Any] = {}

StableNumber = bool | int | float | complex | np.generic

def return_type(op: Any, *args: Any) -> Any:
    """The return type of the given function on the given argument types.

    Args:
        op: The function or operator to infer the type for.
        *args: The types of the arguments.

    Returns:
        The return type of op(*args: arg_types)
    """
    return query_property(op, "__call__", "return_type", *args)

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

    method_name: str

    is_associative: bool = False
    is_commutative: bool = False
    is_idempotent: bool = False

    @abstractmethod
    def __call__(self, *args: Any) -> Any:
        pass

    @abstractmethod
    def return_type(self, *arg_types: type) -> Any:
        pass

    def is_distributive(self, other_op: "FinchOperator") -> bool:
        return False

    def is_identity(self):
        return False

    def is_annihilator(self, arg: Any):
        return False
    
class ReflexiveFinchOperator(FinchOperator):

    reflected_method: str

    def return_type(self, a, b):
        if hasattr(a, self.method_name):
            try:
                result = type(getattr(a(True), self.method_name)(b(True)))
                if result is not type(NotImplemented):
                    return result
                
            except (TypeError, AttributeError):
                pass
        if hasattr(b, self.reflected_method):
            return type(getattr(b(True), self.reflected_method)(a(True)))
        raise TypeError(f"Unsupported operand types: {a}, {b}")

    
class UnaryFinchOperator(FinchOperator):
    def return_type(self, a):
        return type(getattr(a(True), self.method_name)())
    

class ComparisonFinchOperator(FinchOperator):
    def return_type(self, a, b) -> type:
        return bool
    
class Add(ReflexiveFinchOperator):
    method_name = "__add__"
    reflected_method = "__radd__"

    is_associative = True
    is_commutative = True

    def __call__(self, a, b):
        return operator.add(a, b)
    
    def is_identity(self, arg: Any) -> bool:
        return arg == 0
    
    def is_annihilator(self, arg: Any) ->bool:
        return np.isinf(arg)
    
    def is_distributive(self, other_op: "FinchOperator") -> bool:
        return isinstance(other_op, (Add, Sub))
    

class Mul(ReflexiveFinchOperator):
    method_name = "__mul__"
    reflected_method = "__rmul__"

    def __call__(self, a, b):
        return operator.mul(a, b)

    def is_identity(self, arg: Any) -> bool:
        return arg == 1
    

class Sub(ReflexiveFinchOperator):
    method_name = "__sub__"
    reflected_method = "__rsub__"

    def __call__(self, a, b):
        return operator.sub(a, b)
    
class MatMul(ReflexiveFinchOperator):
    method_name = "__matmul__"
    reflected_method = "__rmatmul__"

    is_associative = True

    def __call__(self, a, b):
        return operator.matmul(a, b)


class TrueDiv(ReflexiveFinchOperator):
    method_name = "__truediv__"
    reflected_method = "__rtruediv__"

    def __call__(self, a, b):
        return operator.truediv(a, b)
    
    def is_identity(self, arg):
        return arg == 1


class FloorDiv(ReflexiveFinchOperator):
    method_name = "__floordiv__"
    reflected_method = "__rfloordiv__"

    def __call__(self, a, b):
        return operator.floordiv(a, b)


class Mod(ReflexiveFinchOperator):
    method_name = "__mod__"
    reflected_method = "__rmod__"

    def __call__(self, a, b):
        return operator.mod(a, b)


class DivMod(ReflexiveFinchOperator):
    method_name = "__divmod__"
    reflected_method = "__rdivmod__"

    def __call__(self, a, b):
        return divmod(a, b)


class Pow(ReflexiveFinchOperator):
    method_name = "__pow__"
    reflected_method = "__rpow__"

    def __call__(self, a, b):
        return operator.pow(a, b)

    def is_identity(self, arg):
        return arg == 1

    def is_annihilator(self, arg):
        return arg == 0


class LShift(ReflexiveFinchOperator):
    method_name = "__lshift__"
    reflected_method = "__rlshift__"

    def __call__(self, a, b):
        return operator.lshift(a, b)
    
    def is_identity(self, arg):
        return arg == 1


class RShift(ReflexiveFinchOperator):
    method_name = "__rshift__"
    reflected_method = "__rrshift__"

    def __call__(self, a, b):
        return operator.rshift(a, b)
    
    def is_identity(self, arg):
        return arg == 1


class And(ReflexiveFinchOperator):
    method_name = "__and__"
    reflected_method = "__rand__"

    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __call__(self, a, b):
        return operator.and_(a, b)
    
    def is_identity(self, arg):
        return bool(arg)

    def is_annihilator(self, arg):
        return not bool(arg)
    
    def is_distributive(self, other_op: "FinchOperator") -> bool:
        return isinstance(other_op, (Or, Xor))


class Xor(ReflexiveFinchOperator):
    method_name = "__xor__"
    reflected_method = "__rxor__"

    is_associative = True
    is_commutative = True

    def __call__(self, a, b):
        return operator.xor(a, b)

    def is_identity(self, arg):
        return arg == 0


class Or(ReflexiveFinchOperator):
    method_name = "__or__"
    reflected_method = "__ror__"

    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __call__(self, a, b):
        return operator.or_(a, b)

    def is_identity(self, arg):
        return not bool(arg)
    
    def is_annihilator(self, arg):
        return bool(arg)
    
    def is_distributive(self, other_op: "FinchOperator") -> bool:
        return isinstance(other_op, And)

class Abs(UnaryFinchOperator):    
    method_name = "__abs__"    
    is_idempotent = True
    
    def __call__(self, a):
        return operator.abs(a)      


class Pos(UnaryFinchOperator):

    method_name = "__pos__"
    is_idempotent = True
    
    def __call__(self, a):
        return operator.pos(a)


class Neg(UnaryFinchOperator):
    method_name = "__neg__"

    def __call__(self, a):
        return operator.neg(a)


class Invert(UnaryFinchOperator):
    method_name = "__invert__"

    def __call__(self, a):
        return operator.invert(a)


class Eq(ComparisonFinchOperator):
    method_name = "__eq__"
    is_commutative = True

    def __call__(self, a, b):
        return operator.eq(a, b)

class Ne(ComparisonFinchOperator):
    method_name = "__ne__"
    is_commutative = True

    def __call__(self, a, b):
        return operator.ne(a, b)

class Gt(ComparisonFinchOperator):
    method_name = "__gt__"

    def __call__(self, a, b):
        return operator.gt(a, b)


class Lt(ComparisonFinchOperator):
    method_name = "__lt__"

    def __call__(self, a, b):
        return operator.lt(a, b)

class Ge(ComparisonFinchOperator):
    method_name = "__ge__"

    def __call__(self, a, b):
        return operator.ge(a, b)

class Le(ComparisonFinchOperator):
    method_name = "__le__"

    def __call__(self, a, b):
        return operator.le(a, b)






def is_identity(op: Any, val: Any) -> bool:
    """
    Returns whether the given object is an identity for the given function, that is,
    whether the `op(a, val) == a for all a`.

    Args:
        op: The function to check.
        val: The value to check for identity.

    Returns:
        True if the value can be proven to be an identity, False otherwise.
    """
    return query_property(op, "__call__", "is_identity", val)


for fn, func in [
    (np.divide, lambda op, val: val == 1),
    (np.logaddexp, lambda op, val: val == -math.inf),
    (np.logical_and, lambda op, val: bool(val)),
    (np.logical_or, lambda op, val: not val),
    (min, lambda op, val: val == math.inf),
    (max, lambda op, val: val == -math.inf),
]:
    register_property(fn, "__call__", "is_identity", func)


def is_distributive(op, other_op):
    """
    Returns whether the given pair of functions are distributive, that is,
    whether the `f(a, g(b, c)) = g(f(a, b), f(a, c))` for all a, b, c`.

    Args:
        op: The function to check.
        other_op: The other function to check for distributiveness.

    Returns:
        True if the pair of functions can be proven to be distributive, False otherwise.
    """
    return query_property(op, "__call__", "is_distributive", other_op)


for fn, func in [
    (np.logical_and, lambda op, other_op: other_op in (np.logical_or, np.logical_xor)),
    (np.logical_or, lambda op, other_op: other_op == np.logical_and),

    (max, lambda op, other_op: False),
    (min, lambda op, other_op: False),
]:
    register_property(fn, "__call__", "is_distributive", func)


def is_annihilator(op, val):
    """
    Returns whether the given object is an annihilator for the given function, that is,
    whether the `op(a, val) == val for all a`.

    Args:
        op: The function to check.
        val: The value to check for annihilator.

    Returns:
        True if the value can be proven to be an annihilator, False otherwise.
    """
    return query_property(op, "__call__", "is_annihilator", val)


for fn, func in [
    (np.logaddexp, lambda op, val: val == math.inf),
    (np.logical_and, lambda op, val: not bool(val)),
    (np.logical_or, lambda op, val: bool(val)),
]:
    register_property(fn, "__call__", "is_annihilator", func)


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
        r = return_type(
            op, type(z), t
        )  # Assuming `op` is a callable that takes `z` and `t` as arguments
    return r


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
    (bool, lambda x: +math.inf),
    (int, lambda x: +math.inf),
    (float, lambda x: +math.inf),
    (np.bool_, lambda x: x(True)),
    (np.integer, lambda x: np.iinfo(x).max),
    (np.floating, lambda x: np.finfo(x).max),
]:
    register_property(t, "type_max", "__attr__", tn)


def init_value(op, arg) -> Any:
    """Returns the initial value for a reduction operation on the given type.

    Args:
        op: The reduction operation to determine the initial value for.
        arg: The type of arguments to be reduced.

    Returns:
        The initial value for the given operation and type.

    Raises:
        AttributeError: If the initial value is not implemented for the given type
        and operation.
    """
    return query_property(op, "__call__", "init_value", arg)


for op in [operator.add, operator.mul, operator.and_, operator.xor, operator.or_]:
    (meth, rmeth) = _reflexive_operators[op]
    register_property(
        op,
        "__call__",
        "init_value",
        lambda op, arg, meth=meth: query_property(arg, meth, "init_value"),
    )

for fn, func in [
    (np.logaddexp, lambda op, val: -math.inf),
    (np.logical_and, lambda op, val: True),
    (np.logical_or, lambda op, val: False),
    (np.logical_xor, lambda op, val: False),
    (min, lambda op, val: type_max(val)),
    (max, lambda op, val: type_min(val)),
]:
    register_property(fn, "__call__", "init_value", func)


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


for t in StableNumber.__args__:
    register_property(t, "__add__", "init_value", sum_init_value)
    register_property(t, "__mul__", "init_value", lambda a: a(True))
    register_property(t, "__and__", "init_value", lambda a: a(True))
    register_property(t, "__xor__", "init_value", lambda a: a(False))
    register_property(t, "__or__", "init_value", lambda a: a(False))


def is_idempotent(op: Any) -> bool:
    """
    Returns whether the given operator is idempotent over the argument domain,
    i.e., op(x, x) == x.

    Args:
        op: The operator/function to check.

    Returns:
        True if the operator is known to be idempotent, False otherwise.
    """
    return query_property(op, "__call__", "is_idempotent")


for fn in [
    operator.and_,
    operator.or_,
    np.logical_and,
    np.logical_or,
    min,
    max,
]:
    register_property(fn, "__call__", "is_idempotent", lambda op: True)

for fn in [
    operator.add,
    operator.mul,
    operator.xor,
    np.logical_xor,
    np.logaddexp,
]:
    register_property(fn, "__call__", "is_idempotent", lambda op: False)

for unary in (
    np.reciprocal,
    np.sin,
    np.cos,
    np.tan,
    np.sinh,
    np.cosh,
    np.tanh,
    np.atan,
    np.asinh,
    np.asin,
    np.acos,
    np.acosh,
    np.atanh,
    np.round,
    np.floor,
    np.ceil,
    np.trunc,
    np.exp,
    np.expm1,
    np.log,
    np.log1p,
    np.log2,
    np.log10,
    np.signbit,
    np.sqrt,
    np.square,
    np.sign,
):

    def unary_type(op, a):
        # TODO: Determine a better way to do this
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
        raise TypeError(f"Unsupported operand type for {op}:  {a}")

    register_property(
        unary,
        "__call__",
        "return_type",
        unary_type,
    )

for binary_op in (
    np.divide,
    np.remainder,
    np.hypot,
    np.atan2,
    np.logaddexp,
    np.copysign,
    np.nextafter,
):
    register_property(
        binary_op,
        "__call__",
        "return_type",
        lambda op, a, b, _binary_op=binary_op: float,
    )

register_property(np.isfinite, "__call__", "return_type", lambda op, a: bool)
register_property(np.isinf, "__call__", "return_type", lambda op, a: bool)
register_property(np.isnan, "__call__", "return_type", lambda op, a: bool)

for logical in (
    np.logical_and,
    np.logical_or,
    np.logical_xor,
):
    register_property(
        logical,
        "__call__",
        "return_type",
        lambda op, a, b, _logical=logical: bool,
    )
register_property(np.logical_not, "__call__", "return_type", lambda op, a: bool)


for complex_op in (np.real, np.imag):
    register_property(
        complex_op,
        "__call__",
        "return_type",
        lambda op, a, _complex_op=complex_op: float,
    )


for ternary_op in (np.clip,):
    register_property(
        ternary_op,
        "__call__",
        "return_type",
        lambda op, a, b, c, _ternary_op=ternary_op: float,
    )

# Register return types for numpy comparison ufuncs
for comparison in (
    np.equal,
    np.not_equal,
    np.less,
    np.less_equal,
    np.greater,
    np.greater_equal,
):
    register_property(
        comparison,
        "__call__",
        "return_type",
        lambda op, a, b, _comparison=comparison: bool,
    )
