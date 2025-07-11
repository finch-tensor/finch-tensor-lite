from abc import ABC, abstractmethod
from typing import NamedTuple

from ..algebra import register_property
from ..symbolic import Format, format


class AssemblyStructFormat(Format, ABC):
    @property
    @abstractmethod
    def struct_name(self): ...

    @property
    @abstractmethod
    def struct_fields(self): ...

    @property
    def is_mutable(self):
        return False

    def struct_getattr(self, obj, attr):
        return getattr(obj, attr)

    def struct_setattr(self, obj, attr, value):
        setattr(obj, attr, value)
        return None

    @property
    def struct_fieldnames(self):
        return [name for (name, _) in self.struct_fields]

    @property
    def struct_fieldformats(self):
        return [type_ for (_, type_) in self.struct_fields]

    def struct_hasattr(self, attr):
        return attr in dict(self.struct_fields)

    def struct_attrtype(self, attr):
        return dict(self.struct_fields)[attr]


class NamedTupleFormat(AssemblyStructFormat):
    def __init__(self, struct_name, struct_fields):
        self._struct_name = struct_name
        self._struct_fields = struct_fields

    def __eq__(self, other):
        return (
            isinstance(other, NamedTupleFormat) and
            self.struct_name == other.struct_name and
            self.struct_fields == other.struct_fields
        )
    
    def __hash__(self):
        return hash((self.struct_name, tuple(self.struct_fields)))

    @property
    def struct_name(self):
        return self._struct_name

    @property
    def struct_fields(self):
        return self._struct_fields


class TupleFormat(AssemblyStructFormat):
    def __init__(self, name, struct_fieldformats):
        self._struct_name = name
        self._struct_formats = struct_fieldformats

    def __eq__(self, other):
        return (
            isinstance(other, TupleFormat) and
            self.struct_name == other.struct_name and
            self._struct_formats == other._struct_formats
        )


    def struct_getattr(self, obj, attr):
        index = list(self.struct_fieldnames).index(attr)
        return obj[index]

    def struct_setattr(self, obj, attr, value):
        index = list(self.struct_fieldnames).index(attr)
        obj[index] = value
        return None
    
    def __hash__(self):
        return hash((self.struct_name, tuple(self.struct_fieldformats)))

    @property
    def struct_name(self):
        return self._struct_name

    @property
    def struct_fields(self):
        return [
            (f"element_{i}", fmt) for i, fmt in enumerate(self._struct_formats)
        ]

def tupleformat(x):
    if hasattr(type(x), "_fields"):
        return NamedTupleFormat(type(x).__name__, [(fieldname, format(getattr(x, fieldname))) for fieldname in type(x)._fields])
    else:
        print(type(x).__name__)
        return TupleFormat(type(x).__name__, [type(elem) for elem in x])


register_property(
    tuple, "format", "__attr__", tupleformat
)
