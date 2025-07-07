from abc import ABC, abstractmethod
from typing import Any
from ..compile import CArgument, CStackFormat, c_type
from ..compile import NumbaArgument, NumbaStackFormat, numba_type
from ..symbolic import Namespace
import ctypes
from dataclasses import make_dataclass
import numba
from collections import namedtuple

# ------------------ C Struct Classes ------------------

c_structs = {}
c_structnames = Namespace()

class CAssemblyStruct(CArgument, ABC):
    """
    An abstract base class for structures that can be used in C assembly code.
    Provides methods to convert the structure to C formats and to unpack/repack.
    """

    def serialize_to_c(self) -> Any:
        args = [getattr(self, name) for (name, _) in self.fieldnames]
        return self.c_type(*args)
    
    def deserialize_from_c(self, c_struct: Any) -> None:
        for (name, _) in self.fieldnames:
            setattr(self, name, getattr(c_struct, name))
        return

class CAssemblyStructFormat(CStackFormat, ABC):
    @property
    @abstractmethod
    def struct_name(self):
        ...

    @property
    @abstractmethod
    def struct_fields(self):
        ...

    def c_type(self):
        res = c_structs.get(self)
        if res:
            return res
        else:
            fields = [(name, c_type(fmt)) for name, fmt in self.struct_fields]
            new_struct = type(
                c_structnames.freshen("C", self.struct_name),
                (ctypes.Structure,),
                {"_fields_": fields}
            )
            c_structs[self] = new_struct
            return ctypes.POINTER(new_struct)

    def c_unpack(self, ctx, var_n, val):
        var_names = [ctx.freshen(name) for (name, _) in self.fieldnames]
        for var_name, (name, fmt) in zip(var_names, self.fieldnames):
            t = ctx.ctype_name(c_type(fmt))
            ctx.exec(f"{ctx.feed}{t} {var_name} = ({t}){ctx(val)}->{name};")

        StructTuple = namedtuple(f"{self.struct_name}Tuple", [name for name, _ in self.fieldnames])
        return StructTuple(*var_names)

    def c_repack(self, ctx, lhs, obj):
        for (name, fmt) in self.fieldnames:
            t = ctx.ctype_name(c_type(fmt))
            ctx.exec(f"{ctx.feed}{lhs}->{name} = ({t}){getattr(obj, name)};")
        return

    def construct_from_c(self, c_struct):
        args = [getattr(c_struct, name) for (name, _) in self.fieldnames]
        return self.__class__(*args)

# ------------------ Numba Struct Classes ------------------

numba_structs = {}
numba_structnames = Namespace()

class NumbaAssemblyStruct(NumbaArgument, ABC):
    """
    An abstract base class for structures that can be used in Numba assembly code.
    Provides methods to convert the structure to Numba formats and to unpack/repack.
    """

    def serialize_to_numba(self) -> Any:
        args = [getattr(self, name) for (name, _) in self.fieldnames]
        return self.numba_type(*args)
    
    def deserialize_from_numba(self, numba_buffer: Any) -> None:
        for (name, _) in self.fieldnames:
            setattr(self, name, getattr(numba_buffer, name))
        return

class NumbaAssemblyStructFormat(NumbaStackFormat, ABC):
    @property
    @abstractmethod
    def struct_name(self):
        ...

    @property
    @abstractmethod
    def struct_fields(self):
        ...

    @property
    @abstractmethod
    def numba_type(self):
        res = numba_structs.get(self)
        if res:
            return res
        else:
            spec = [(name, numba_type(fmt)) for name, fmt in self.struct_fields]
            new_struct = make_dataclass(
                numba_structnames.freshen("Numba", self.struct_name),
                fields = spec,
            )
            new_struct = numba.jitclass(spec)(new_struct)
            numba_structs[self] = new_struct
            return new_struct

    def numba_unpack(self, ctx, var_n, val):
        var_names = [ctx.freshen(name) for (name, _) in self.fieldnames]
        for var_name, (name, fmt) in zip(var_names, self.fieldnames):
            ctx.exec(f"{var_name} = {ctx(val)}.{name}")

        StructTuple = namedtuple(f"{self.struct_name}Tuple", [name for name, _ in self.fieldnames])
        return StructTuple(*var_names)

    def numba_repack(self, ctx, lhs, obj):
        for (name, _) in self.fieldnames:
            ctx.exec(f"{ctx.feed}{lhs}.{name} = {getattr(obj, name)};")
        return

    def construct_from_numba(self, numba_struct):
        args = [getattr(numba_struct, name) for (name, _) in self.fieldnames]
        return self.__class__(*args)
