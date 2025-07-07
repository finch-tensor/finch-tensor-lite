from abc import ABC, abstractmethod

class AssemblyStructFormat(ABC):
    @property
    @abstractmethod
    def struct_name(self): ...

    @property
    @abstractmethod
    def struct_fields(self): ...

    @property
    def struct_names(self):
        return [name for (name, _) in self.struct_fields]

    @property
    def struct_types(self):
        return [type_ for (_, type_) in self.struct_fields]

    def struct_hasattr(self, attr):
        return attr in dict(self.struct_fields)

    def struct_attrtype(self, attr):
        return dict(self.struct_fields)[attr]

