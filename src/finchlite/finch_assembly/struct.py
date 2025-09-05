from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

from ..algebra import register_property
from ..symbolic import FType, ftype


class AssemblyStructFType(FType, ABC):
    @property
    @abstractmethod
    def struct_name(self) -> str: ...

    @property
    @abstractmethod
    def struct_fields(self) -> list[tuple[str, Any]]: ...

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
