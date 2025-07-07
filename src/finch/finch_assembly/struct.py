from abc import ABC, abstractmethod
from typing import Any
from ..compile import CArgument, CStackFormat, c_type
from ..compile import NumbaArgument, NumbaStackFormat, numba_type
from ..symbolic import Namespace
import ctypes
from dataclasses import make_dataclass
import numba
from collections import namedtuple

class AssemblyStruct(CArgument, NumbaArgument, ABC):
    """
    An abstract base class for structures that can be used in assembly code.
    This class provides methods to convert the structure to C and Numba formats,
    and to unpack and repack the structure in C and Numba contexts.
    """

    def serialize_to_c(self) -> Any:
        """
        Serialize the structure to a C-compatible format.
        This should return a ctypes structure or pointer to it.
        """
        args = [getattr(self, name) for (name, _) in self.fieldnames]
        return self.c_type(*args)
    
    def deserialize_from_c(self, c_struct: Any) -> None:
        """
        Update this structure based on how the C call modified the C-compatible structure.
        This should set the attributes of the structure based on the fields of c_struct.
        """
        for (name, _) in self.fieldnames:
            setattr(self, name, getattr(c_struct, name))
        return
    
    def serialize_to_numba(self) -> Any:
        """
        Serialize the structure to a Numba-compatible format.
        This should return a Numba-compatible object.
        """
        args = [getattr(self, name) for (name, _) in self.fieldnames]
        return self.numba_type(*args)
    
    def deserialize_from_numba(self, numba_buffer: Any) -> None:
        """
        Update this structure based on how the Numba call modified the Numba-compatible object.
        This should set the attributes of the structure based on the fields of numba_buffer.
        """
        for (name, _) in self.fieldnames:
            setattr(self, name, getattr(numba_buffer, name))
        return

c_structs = {}
c_structnames = Namespace()

numba_structs = {}
numba_structnames = Namespace()

        
class AssemblyStructFormat(CStackFormat, NumbaStackFormat, ABC):
    @property
    @abstractmethod
    def struct_name(self):
        """
        Return the name of the structure.
        This should be unique across all structures.
        """
        ...
    
    @property
    @abstractmethod
    def struct_fields(self):
        """
        Return the fields of the structure as a list of tuples (name, format).
        Each format should be a valid assembly format.
        """
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
            
    def c_unpack(self, ctx, var_n, val):
        """
        Unpack the struct into C context.
        """
        var_names = [ctx.freshen(name) for (name, _) in self.fieldnames]
        for var_name, (name, fmt) in zip(var_names, self.fieldnames):
            t = ctx.ctype_name(c_type(fmt))
            ctx.exec(f"{ctx.feed}{t} {var_name} = ({t}){ctx(val)}->{name};")

        StructTuple = namedtuple(f"{self.struct_name}Tuple", [name for name, _ in self.fieldnames])
        return StructTuple(*var_names)

    def c_repack(self, ctx, lhs, obj):
        """
        Repack the buffer from C context.
        """
        for (name, fmt) in self.fieldnames:
            t = ctx.ctype_name(c_type(fmt))
            ctx.exec(f"{ctx.feed}{lhs}->{name} = ({t}){getattr(obj, name)};")
        return

    def construct_from_c(self, c_struct):
        """
        Construct a NumpyBuffer from a C-compatible structure.
        """
        args = [getattr(c_struct, name) for (name, _) in self.fieldnames]
        return self.__class__(*args)

    def numba_unpack(self, ctx, var_n, val):
        """
        Unpack the buffer into Numba context.
        """
        var_names = [ctx.freshen(name) for (name, _) in self.fieldnames]
        for var_name, (name, fmt) in zip(var_names, self.fieldnames):
            ctx.exec(f"{var_name} = {ctx(val)}.{name}")

        StructTuple = namedtuple(f"{self.struct_name}Tuple", [name for name, _ in self.fieldnames])
        return StructTuple(*var_names)

    def numba_repack(self, ctx, lhs, obj):
        """
        Repack the buffer from Numba context.
        """
        for (name, _) in self.fieldnames:
            ctx.exec(f"{ctx.feed}{lhs}.{name} = {getattr(obj, name)};")
        return

    def construct_from_numba(self, numba_struct):
        """
        Construct a NumpyBuffer from a Numba-compatible object.
        """
        args = [getattr(numba_struct, name) for (name, _) in self.fieldnames]
        return self.__class__(*args)