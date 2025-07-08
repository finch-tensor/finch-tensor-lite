from abc import ABC, abstractmethod
from typing import NamedTuple

from ..algebra import register_property


class AssemblyStructFormat(ABC):
    @property
    @abstractmethod
    def struct_name(self): ...

    @property
    @abstractmethod
    def struct_fields(self): ...

    @property
    def struct_fieldnames(self):
        return [name for (name, _) in self.struct_fields]

    @property
    def struct_types(self):
        return [type_ for (_, type_) in self.struct_fields]

    def struct_hasattr(self, attr):
        return attr in dict(self.struct_fields)

    def struct_attrtype(self, attr):
        return dict(self.struct_fields)[attr]


class NamedTupleFormat(AssemblyStructFormat):
    def __init__(self, struct_name, struct_fields):
        self._struct_name = struct_name
        self._struct_fields = struct_fields

    @property
    def struct_name(self):
        return self._struct_name

    @property
    def struct_fields(self):
        return self._struct_fields


register_property(
    NamedTuple, "format", "__attr__", lambda x: NamedTupleFormat(x.__name__, x._fields)
)


class TupleFormat(AssemblyStructFormat):
    def __init__(self, name, struct_formats):
        self._struct_name = name
        self._struct_formats = struct_formats

    @property
    def struct_name(self):
        return self._struct_name

    @property
    def struct_fields(self):
        return [
            ("element_" + str(i), fmt) for i, fmt in enumerate(self._struct_formats)
        ]


register_property(
    tuple, "format", "__attr__", lambda x: TupleFormat(x.__name__, len(x))
)
