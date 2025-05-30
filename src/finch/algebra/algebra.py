"""
Finch performs extensive rewriting and defining of functions.  The Finch
compiler is designed to inspect objects and functions defined by other
frameworks, such as NumPy. The Finch compiler is designed to be extensible, so that
users can define their own properties and behaviors for objects and functions in
their own code or in third-party libraries.

Finch tracks properties of attributes/methods of objects or classes. Properties
of the object/class itself are properties of the `__self__` attribute.
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
properties have been defined for the object itself (in the case of functions), then we
check For example, if you register a property for a class, all subclasses of that class
will inherit that property. This allows you to define properties for a class and have
them automatically apply to all subclasses, without having to register the
property for each subclass individually.
"""

import math
import operator
from collections.abc import Callable, Hashable
from typing import Any

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
        NotImplementedError: If the property is not implemented for the given type.
    """
    if not isinstance(obj, type):
        if isinstance(obj, Hashable):
            query_fn = _properties.get((obj, attr, prop))
            if query_fn is not None:
                return query_fn(obj, *args)
        T = type(obj)
    else:
        T = obj

    for Ti in T.__mro__:
        query_fn = _properties.get((Ti, attr, prop))
        if query_fn is not None:
            return query_fn(obj, *args)

    raise NotImplementedError(f"Property {prop} not implemented for {obj}")


def register_property(cls, attr, prop, f):
    """Registers a property for a class or object.

    Args:
        cls: The class or object to register the property for.
        prop: The property to register.
        f: The function to register as the property, which should take the
            object and any additional arguments as input.
    """
    _properties[(cls, attr, prop)] = f


def fill_value(arg: Any) -> Any:
    """The fill value for the given argument.  The fill value is the
    default value for a tensor when it is created with a given shape and dtype,
    as well as the background value for sparse tensors.

    Args:
        arg: The argument to determine the fill value for.

    Returns:
        The fill value for the given argument.

    Raises:
        NotImplementedError: If the fill value is not implemented for the given type.
    """
    if hasattr(arg, "fill_value"):
        return arg.fill_value
    return query_property(arg, "__self__", "fill_value")


register_property(
    np.ndarray, "__self__", "fill_value", lambda x: np.zeros((), dtype=x.dtype)[()]
)


def element_type(arg: Any) -> type:
    """The element type of the given argument.  The element type is the scalar type of
    the elements in a tensor, which may be different from the data type of the
    tensor.

    Args:
        arg: The tensor to determine the element type for.

    Returns:
        The element type of the given tensor.

    Raises:
        NotImplementedError: If the element type is not implemented for the given type.
    """
    if hasattr(arg, "element_type"):
        return arg.element_type
    return query_property(arg, "__self__", "element_type")


register_property(
    np.ndarray,
    "__self__",
    "element_type",
    lambda x: x.dtype.type,
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


def _return_type_reflexive(meth):
    def _return_type_closure(a, b):
        if issubclass(b, StableNumber):
            return type(getattr(a(True), meth)(b(True)))
        raise TypeError(f"Unsupported operand type for {type(a)}.{meth}:  {type(b)}")

    return _return_type_closure


op: Callable

for op, (meth, rmeth) in _reflexive_operators.items():
    (
        register_property(
            op,
            "__call__",
            "return_type",
            lambda op, a, b, meth=meth, rmeth=rmeth: query_property(
                a, meth, "return_type", b
            )
            if hasattr(a, meth)
            else query_property(b, rmeth, "return_type", a),
        ),
    )

    for T in StableNumber.__args__:
        register_property(T, meth, "return_type", _return_type_reflexive(meth))
        register_property(T, rmeth, "return_type", _return_type_reflexive(rmeth))

register_property(
    min,
    "__call__",
    "return_type",
    lambda op, a, b: query_property(a, "__add__", "return_type", b),
)
register_property(
    max,
    "__call__",
    "return_type",
    lambda op, a, b: query_property(a, "__add__", "return_type", b),
)
register_property(any, "__call__", "return_type", lambda op, a, b: bool)
register_property(all, "__call__", "return_type", lambda op, a, b: bool)


_unary_operators: dict[Callable, str] = {
    operator.abs: "__abs__",
    operator.pos: "__pos__",
    operator.neg: "__neg__",
}


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

    for T in StableNumber.__args__:
        register_property(T, meth, "return_type", _return_type_unary(meth))


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
    lambda op, other_op: other_op == operator.add,
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


def fixpoint_type(op: Any, z: Any, T: type) -> type:
    """
    Determines the fixpoint type after repeated calling the given operation.

    Args:
        op: The operation to evaluate.
        z: The initial value.
        T: The type to evaluate against.

    Returns:
        The fixpoint type.
    """
    S = set()
    R = type(z)
    while R not in S:
        S.add(R)
        R = return_type(
            op, type(z), T
        )  # Assuming `op` is a callable that takes `z` and `T` as arguments
    return R


def init_value(op, arg) -> Any:
    """Returns the initial value for a reduction operation on the given type.

    Args:
        op: The reduction operation to determine the initial value for.
        arg: The type of arguments to be reduced.

    Returns:
        The initial value for the given operation and type.

    Raises:
        NotImplementedError: If the initial value is not implemented for the given type
        and operation.
    """
    return query_property(op, "__call__", "init_value", arg)


for op in [operator.add, operator.mul, operator.and_, operator.xor, operator.or_]:
    (meth, rmeth) = _reflexive_operators[op]
    register_property(
        op,
        "__call__",
        "init_value",
        lambda op, arg, meth=meth: query_property(arg, meth, "init_value", arg),
    )

for T in StableNumber.__args__:
    register_property(T, "__add__", "init_value", lambda a, b: a(False))
    register_property(T, "__mul__", "init_value", lambda a, b: a(True))
    register_property(T, "__and__", "init_value", lambda a, b: a(True))
    register_property(T, "__xor__", "init_value", lambda a, b: a(False))
    register_property(T, "__or__", "init_value", lambda a, b: a(False))


def _min_init(arg):
    dtype = np.dtype(arg) if isinstance(arg, np.ndarray) else np.dtype(arg)
    if np.issubdtype(dtype, np.floating):
        return math.inf
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).max
    if np.issubdtype(dtype, np.bool_):
        return True
    raise TypeError("Unsupported dtype for min")


def _max_init(arg):
    dtype = np.dtype(arg) if isinstance(arg, np.ndarray) else np.dtype(arg)
    if np.issubdtype(dtype, np.floating):
        return -math.inf
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).min
    if np.issubdtype(dtype, np.bool_):
        return False
    raise TypeError("Unsupported dtype for max")


register_property(
    min, "__call__", "init_value", lambda op, arg: _min_init(element_type(arg))
)
register_property(
    max, "__call__", "init_value", lambda op, arg: _max_init(element_type(arg))
)
register_property(any, "__call__", "init_value", lambda op, arg: False)
register_property(all, "__call__", "init_value", lambda op, arg: True)
register_property(operator.truth, "__call__", "init_value", lambda op, arg: True)
