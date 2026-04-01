# AI modified: 2026-04-01T17:18:51Z 0de216cc18e91710a9b1a0328f5b181137d8901b
# AI modified: 2026-04-01T17:18:51Z 0de216cc18e91710a9b1a0328f5b181137d8901b
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

from abc import ABC, abstractmethod
from collections.abc import Hashable
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


class NumbaOperator:
    def numba_literal(self, val: Any, ctx: Any, *args: Any) -> Any:
        return f"({f' {self.numba_name()} '.join(map(ctx, args))})"

    def numba_name(self) -> str:
        raise NotImplementedError(f"{type(self)} must implement numba_name")


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


def return_type(op: FinchOperator, *args: Any) -> Any:
    return op.return_type(*args)


def init_value(op: FinchOperator, arg: Any) -> Any:
    return op.init_value(arg)


def fixpoint_type(op: FinchOperator, z: Any, t: type) -> type:
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
