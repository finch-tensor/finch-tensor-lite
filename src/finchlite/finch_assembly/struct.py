from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any

from ..algebra import FType, fisinstance, ftype, register_property
from ..algebra.ftype import TupleFType


class StructFType(FType, ABC):
    @property
    @abstractmethod
    def struct_name(self) -> str: ...

    @property
    @abstractmethod
    def struct_fields(self) -> list[tuple[str, Any]]: ...

    @abstractmethod
    def from_fields(self, *args): ...

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
    def struct_fieldtypes(self) -> list[Any]:
        return [type_ for (_, type_) in self.struct_fields]

    def struct_hasattr(self, attr: str) -> bool:
        return attr in dict(self.struct_fields)

    def struct_attrtype(self, attr: str) -> Any:
        return dict(self.struct_fields)[attr]


class ImmutableStructFType(StructFType):
    @property
    def is_mutable(self) -> bool:
        return False


class MutableStructFType(StructFType):
    """
    Class for a mutable assembly struct type.
    It is currently not used anywhere, but maybe it will be useful in the future?
    """

    @property
    def is_mutable(self) -> bool:
        return True


class NamedTupleFType(ImmutableStructFType):
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

    def fisinstance(self, other):
        if not isinstance(other, tuple) or not hasattr(other, "_fields"):
            return False
        if tuple(other._fields) != tuple(self.struct_fieldnames):
            return False

        return all(
            fisinstance(elt, format)
            for elt, format in zip(other, self.struct_fieldtypes, strict=False)
        )

    def from_fields(self, *args):
        assert all(
            fisinstance(a, f)
            for a, f in zip(args, self.struct_fieldtypes, strict=False)
        )
        return namedtuple(self.struct_name, self.struct_fieldnames)(args)

    def __call__(self, *args):
        return self.from_fields(*args)


def tupleformat(x):
    if hasattr(type(x), "_fields"):
        return NamedTupleFType(
            type(x).__name__,
            [
                (fieldname, ftype(getattr(x, fieldname)))
                for fieldname in type(x)._fields
            ],
        )
    return TupleFType.from_tuple(tuple([ftype(elem) for elem in x]))


register_property(tuple, "ftype", "__attr__", tupleformat)
