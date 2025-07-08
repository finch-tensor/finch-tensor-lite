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
from finch import register_property

register_property(complex, "__add__", "is_associative", lambda obj: True)
```

Finch includes a convenience functions to query each property as well,
for example:
```python
from finch import query_property
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
            msg += f"`finch.register_property({repr(obj)}, '{attr}', '{prop}', "
            "lambda ...)` or "
        msg += f"`finch.register_property({obj_name}, '{attr}', '{prop}', lambda ...)`"
        msg += f"or you may define `{obj_name}.{attr}`. "
    elif attr == "__call__":
        msg += f"function '{repr(obj)}' has no property '{prop}'. "
        msg += "Hint: You may need to register the property by calling "
        if isinstance(obj, Hashable) and not isinstance(obj, type):
            msg += f"`finch.register_property({repr(obj)}, '{attr}', '{prop}',"
            ", lambda ...)` or "
        msg += f"`finch.register_property({obj_name}, '{attr}', '{prop}', lambda ...)`."
    else:
        msg += f"attribute '{obj_name}.{attr}' has no property '{prop}'. "
        msg += "You may need to register the property by calling "
        if isinstance(obj, Hashable) and not isinstance(obj, type):
            msg += f"finch.register_property({repr(obj)}, '{attr}', '{prop}'"
            ", lambda ...) or "
        msg += f"`finch.register_property({obj_name}, '{attr}', '{prop}', lambda ...)`."
    msg += " See https://github.com/finch-tensor/finch-tensor-lite/blob/main/src/finch/"
    "algebra/algebra.py for more information."
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


def return_type(op: Any, *args: Any) -> Any:
    """The return type of the given function on the given argument types.

    Args:
        op: The function or operator to infer the type for.
        *args: The types of the arguments.

    Returns:
        The return type of op(*args: arg_types)
    """
    return query_property(op, "__call__", "return_type", *args)


_reflexive_operators = {
    operator.add: ("__add__", "__radd__"),
    operator.sub: ("__sub__", "__rsub__"),
    operator.mul: ("__mul__", "__rmul__"),
    operator.matmul: ("__matmul__", "__rmatmul__"),
    operator.truediv: ("__truediv__", "__rtruediv__"),
    operator.floordiv: ("__floordiv__", "__rfloordiv__"),
    operator.mod: ("__mod__", "__rmod__"),
    divmod: ("__divmod__", "__rdivmod__"),
    operator.pow: ("__pow__", "__rpow__"),
    operator.lshift: ("__lshift__", "__rlshift__"),
    operator.rshift: ("__rshift__", "__rrshift__"),
    operator.and_: ("__and__", "__rand__"),
    operator.xor: ("__xor__", "__rxor__"),
    operator.or_: ("__or__", "__ror__"),
}


def _return_type_reflexive_method(meth):
    def _return_type_closure(a, b):
        if issubclass(b, StableNumber):
            return type(getattr(a(True), meth)(b(True)))
        raise TypeError(f"Unsupported operand type for {type(a)}.{meth}:  {type(b)}")

    return _return_type_closure


def _return_type_reflexive_operator(meth, rmeth):
    def _return_type_closure(op, a, b):
        if hasattr(a, meth):
            try:
                res = query_property(a, meth, "return_type", b)
                if res is not type(NotImplemented):
                    return res
            except AttributeError:
                pass
        return query_property(b, rmeth, "return_type", a)

    return _return_type_closure


op: Callable


for op, (meth, rmeth) in _reflexive_operators.items():
    (
        register_property(
            op,
            "__call__",
            "return_type",
            _return_type_reflexive_operator(meth, rmeth),
        ),
    )

    for t in StableNumber.__args__:
        register_property(t, meth, "return_type", _return_type_reflexive_method(meth))
        register_property(t, rmeth, "return_type", _return_type_reflexive_method(rmeth))


_unary_operators: dict[Callable, str] = {
    operator.abs: "__abs__",
    operator.pos: "__pos__",
    operator.neg: "__neg__",
    operator.invert: "__invert__",
}


_comparison_operators: dict[Callable, str] = {
    operator.eq: "__eq__",
    operator.ne: "__ne__",
    operator.gt: "__gt__",
    operator.lt: "__lt__",
    operator.ge: "__ge__",
    operator.le: "__le__",
}


for op, meth in _comparison_operators.items():
    (
        register_property(
            op,
            "__call__",
            "return_type",
            lambda op, a, b, meth=meth: bool,
        ),
    )


def _return_type_unary(meth):
    def _return_type_closure(a):
        return type(getattr(a(True), meth)())

    return _return_type_closure


for op, meth in _unary_operators.items():
    (
        register_property(
            op,
            "__call__",
            "return_type",
            lambda op, a, meth=meth: query_property(a, meth, "return_type"),
        ),
    )

    for t in StableNumber.__args__:
        register_property(t, meth, "return_type", _return_type_unary(meth))
register_property(operator.truth, "__call__", "return_type", lambda op, a: bool)


def is_associative(op: Any) -> bool:
    """
    Returns whether the given function is associative, that is, whether the
    op(op(a, b), c) == op(a, op(b, c)) for all a, b, c.

    Args:
        op: The function to check.

    Returns:
        True if the function can be proven to be associative, False otherwise.
    """
    return query_property(op, "__call__", "is_associative")


for op in [operator.add, operator.mul, operator.and_, operator.xor, operator.or_]:
    register_property(op, "__call__", "is_associative", lambda op: True)

register_property(np.logaddexp, "__call__", "is_associative", lambda op: True)
register_property(np.logical_and, "__call__", "is_associative", lambda op: True)
register_property(np.logical_or, "__call__", "is_associative", lambda op: True)
register_property(np.logical_xor, "__call__", "is_associative", lambda op: True)


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


register_property(operator.add, "__call__", "is_identity", lambda op, val: val == 0)
register_property(operator.mul, "__call__", "is_identity", lambda op, val: val == 1)
register_property(
    operator.or_, "__call__", "is_identity", lambda op, val: not bool(val)
)
register_property(operator.and_, "__call__", "is_identity", lambda op, val: bool(val))
register_property(operator.truediv, "__call__", "is_identity", lambda op, val: val == 1)
register_property(
    operator.floordiv, "__call__", "is_identity", lambda op, val: val == 1
)
register_property(operator.lshift, "__call__", "is_identity", lambda op, val: val == 0)
register_property(operator.rshift, "__call__", "is_identity", lambda op, val: val == 0)
register_property(operator.pow, "__call__", "is_identity", lambda op, val: val == 1)
register_property(
    np.logaddexp, "__call__", "is_identity", lambda op, val: val == -math.inf
)
register_property(np.logical_and, "__call__", "is_identity", lambda op, val: bool(val))
register_property(np.logical_or, "__call__", "is_identity", lambda op, val: not val)


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


register_property(
    operator.mul,
    "__call__",
    "is_distributive",
    lambda op, other_op: other_op in (operator.add, operator.sub),
)
register_property(
    operator.and_,
    "__call__",
    "is_distributive",
    lambda op, other_op: other_op in (operator.or_, operator.xor),
)
register_property(
    operator.or_,
    "__call__",
    "is_distributive",
    lambda op, other_op: other_op == operator.and_,
)
register_property(
    np.logical_and,
    "__call__",
    "is_distributive",
    lambda op, other_op: other_op in (np.logical_or, np.logical_xor),
)
register_property(
    np.logical_or,
    "__call__",
    "is_distributive",
    lambda op, other_op: other_op == np.logical_and,
)


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


for op, func in [
    (operator.add, lambda op, val: np.isinf(val)),
    (operator.mul, lambda op, val: val == 0),
    (operator.or_, lambda op, val: bool(val)),
    (operator.and_, lambda op, val: not bool(val)),
]:
    register_property(op, "__call__", "is_annihilator", func)

register_property(
    np.logaddexp, "__call__", "is_annihilator", lambda op, val: val == math.inf
)
register_property(
    np.logical_and, "__call__", "is_annihilator", lambda op, val: not bool(val)
)
register_property(
    np.logical_or, "__call__", "is_annihilator", lambda op, val: bool(val)
)


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


for t in [bool, int, float]:
    register_property(t, "type_min", "__attr__", lambda x: -math.inf)
    register_property(t, "type_max", "__attr__", lambda x: +math.inf)

register_property(np.bool_, "type_min", "__attr__", lambda x: x(False))
register_property(np.bool_, "type_max", "__attr__", lambda x: x(True))
register_property(np.integer, "type_min", "__attr__", lambda x: np.iinfo(x).min)
register_property(np.integer, "type_max", "__attr__", lambda x: np.iinfo(x).max)
register_property(np.floating, "type_min", "__attr__", lambda x: np.finfo(x).min)
register_property(np.floating, "type_max", "__attr__", lambda x: np.finfo(x).max)


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

register_property(np.logaddexp, "__call__", "init_value", lambda op, arg: -math.inf)
register_property(np.logical_and, "__call__", "init_value", lambda op, arg: True)
register_property(np.logical_or, "__call__", "init_value", lambda op, arg: False)
register_property(np.logical_xor, "__call__", "init_value", lambda op, arg: False)


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

register_property(min, "__call__", "init_value", lambda op, arg: type_max(arg))
register_property(max, "__call__", "init_value", lambda op, arg: type_min(arg))

for unary in (
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
    np.log,
    np.log1p,
    np.log2,
    np.log10,
):
    register_property(
        unary, "__call__", "return_type", lambda op, a, _unary=unary: float
    )

for binary_op in (
    np.atan2,
    np.logaddexp,
):
    register_property(
        binary_op,
        "__call__",
        "return_type",
        lambda op, a, b, _binary_op=binary_op: float,
    )

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
