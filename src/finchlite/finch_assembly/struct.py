from abc import ABC, abstractmethod
from collections import namedtuple
from functools import lru_cache
from typing import Any

from ..algebra import register_property, make_tuple
from ..symbolic import FType, ftype


class AssemblyStructFType(FType, ABC):
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

    @abstractmethod
    def from_kwargs(self, **kwargs) -> "AssemblyStructFType":
        """
        Protocol for constructing Finch tensors from keyword arguments.
        Here are currently supported arguments. They are all optional,
        each implementor decides which fields to select:
        - lvl_t: LevelFType
        - fill_value: np.number
        - element_type: type
        - position_type: type
        - dimension_type: type
        - shape_type: tuple[type, ...]
        - buffer_factory: type
        - buffer_type: BufferFType
        - ndim: int
        """
        ...

    @abstractmethod
    def to_kwargs(self) -> dict: ...

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


class ImmutableStructFType(AssemblyStructFType):
    @property
    def is_mutable(self) -> bool:
        return False


class MutableStructFType(AssemblyStructFType):
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

    def from_kwargs(self, **kwargs) -> "TupleFType":
        raise NotImplementedError

    def to_kwargs(self) -> dict:
        raise NotImplementedError

    def from_fields(self, *args):
        assert all(
            isinstance(a, f)
            for a, f in zip(args, self.struct_fieldformats, strict=False)
        )
        return namedtuple(self.struct_name, self.struct_fieldnames)(args)

    def __call__(self, *args):
        assert all(
            isinstance(a, f)
            for a, f in zip(args, self.struct_fieldformats, strict=False)
        )
        return namedtuple(self.struct_name, self.struct_fieldnames)(args)
    

class TupleFType(ImmutableStructFType):
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

    def from_kwargs(self, **kwargs) -> "TupleFType":
        raise NotImplementedError

    def to_kwargs(self) -> dict:
        raise NotImplementedError

    def __call__(self, **kwargs):
        assert all(
            isinstance(a, f)
            for a, f in zip(kwargs.values(), self.struct_fieldformats, strict=False)
        )
        return tuple(kwargs.values())

    def from_fields(self, *args):
        assert all(
            isinstance(a, f)
            for a, f in zip(args, self.struct_fieldformats, strict=False)
        )
        return tuple(args)

    @staticmethod
    @lru_cache
    def from_tuple(types: tuple[Any, ...]) -> "TupleFType":
        return TupleFType("tuple", types)

    def __str__(self):
        return f"{self.struct_name}({', '.join(map(str, self._struct_formats))})"


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

def tuple_return_type(fmt, *args):
    return TupleFType(
        "tuple",
        args
    )

register_property(
    make_tuple,
    "__call__",
    "return_type",
    tuple_return_type,
)