from abc import ABC, abstractmethod
from collections import namedtuple
from functools import lru_cache
from textwrap import dedent
from typing import Any

import numba

from ..algebra import register_property
from ..symbolic import FType, ftype


class AssemblyStructFType(FType, ABC):
    @property
    @abstractmethod
    def struct_name(self) -> str: ...

    @property
    @abstractmethod
    def struct_fields(self) -> list[tuple[str, Any]]: ...

    @abstractmethod
    def __call__(self, *args): ...

    def numba_type(self):
        """
        Method for registering and caching Numba jitclass.
        """
        from ..codegen.numba_backend import (
            numba_globals,
            numba_jitclass_type,
            numba_structnames,
            numba_structs,
        )

        if self in numba_structs:
            return numba_structs[self]

        spec = [
            (name, numba_jitclass_type(field_type))
            for (name, field_type) in self.struct_fields
        ]
        class_name = numba_structnames.freshen("Numba", self.struct_name)
        # Dynamically define __init__ based on spec, unrolling the arguments
        field_names = [name for name, _ in spec]
        # Build the argument list for __init__
        arg_list = ", ".join(field_names)
        # Build the body of __init__ to assign each argument to self
        body = "; ".join([f"self.{name} = {name}" for name in field_names])
        # Compose the full class source
        class_src = dedent(
            f"""\
            class {class_name}:
                def __init__(self, {arg_list}):
                    {body if body else "pass"}
                @staticmethod
                def numba_name():
                    return '{class_name}'
            """
        )
        ns: dict[str, object] = {}
        exec(class_src, ns)
        new_struct = numba.experimental.jitclass(ns[class_name], spec)
        numba_structs[self] = new_struct
        numba_globals[new_struct.__name__] = new_struct
        return new_struct

    def numba_jitclass_type(self) -> numba.types.Type:
        return self.numba_type().class_type.instance_type

    @property
    def is_mutable(self) -> bool:
        return False

    def struct_getattr(self, obj, attr) -> Any:
        return getattr(obj, attr)

    def struct_setattr(self, obj, attr, value) -> None:
        setattr(obj, attr, value)
        return

    @property
    def struct_fieldnames(self) -> list[str]:
        return [name for (name, _) in self.struct_fields]

    @property
    def struct_fieldformats(self) -> list[Any]:
        return [type_ for (_, type_) in self.struct_fields]

    def struct_hasattr(self, attr: str) -> bool:
        return attr in dict(self.struct_fields)

    def struct_attrtype(self, attr: str) -> Any:
        return dict(self.struct_fields)[attr]


class NamedTupleFType(AssemblyStructFType):
    def __init__(self, struct_name, struct_fields):
        self._struct_name = struct_name
        self._struct_fields = struct_fields

    def __eq__(self, other):
        return (
            isinstance(other, NamedTupleFType)
            and self.struct_name == other.struct_name
            and self.struct_fields == other.struct_fields
        )

    def __len__(self):
        return len(self._struct_fields)

    def __hash__(self):
        return hash((self.struct_name, tuple(self.struct_fields)))

    @property
    def struct_name(self):
        return self._struct_name

    @property
    def struct_fields(self):
        return self._struct_fields

    def __call__(self, *args):
        assert all(
            isinstance(a, f)
            for a, f in zip(args, self.struct_fieldformats, strict=False)
        )
        return namedtuple(self.struct_name, self.struct_fieldnames)(args)


class TupleFType(AssemblyStructFType):
    def __init__(self, struct_name, struct_formats):
        self._struct_name = struct_name
        self._struct_formats = struct_formats

    def __eq__(self, other):
        return (
            isinstance(other, TupleFType)
            and self.struct_name == other.struct_name
            and self._struct_formats == other._struct_formats
        )

    def __len__(self):
        return len(self._struct_formats)

    def struct_getattr(self, obj, attr):
        index = list(self.struct_fieldnames).index(attr)
        return obj[index]

    def struct_setattr(self, obj, attr, value):
        index = list(self.struct_fieldnames).index(attr)
        obj[index] = value
        return

    def __hash__(self):
        return hash((self.struct_name, tuple(self.struct_fieldformats)))

    @property
    def struct_name(self):
        return self._struct_name

    @property
    def struct_fields(self):
        return [(f"element_{i}", fmt) for i, fmt in enumerate(self._struct_formats)]

    def __call__(self, *args):
        assert all(
            isinstance(a, f)
            for a, f in zip(args, self.struct_fieldformats, strict=False)
        )
        return tuple(args)

    @staticmethod
    @lru_cache
    def from_tuple(types: tuple[Any, ...]) -> "TupleFType":
        return TupleFType("tuple", types)


def tupleformat(x):
    if hasattr(type(x), "_fields"):
        return NamedTupleFType(
            type(x).__name__,
            [
                (fieldname, ftype(getattr(x, fieldname)))
                for fieldname in type(x)._fields
            ],
        )
    return TupleFType.from_tuple(tuple([type(elem) for elem in x]))


register_property(tuple, "ftype", "__attr__", tupleformat)
