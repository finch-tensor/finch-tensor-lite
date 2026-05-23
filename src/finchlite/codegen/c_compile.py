import ctypes
import logging
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from collections import namedtuple
from collections.abc import Hashable
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from finchlite import algebra
from finchlite import finch_assembly as asm
from finchlite.algebra import (
    COperator,
    FType,
    ImmutableStructFType,
    MutableStructFType,
    NamedTupleFType,
    StructFType,
    TupleFType,
    ffuncs,
    fisinstance,
    ftype,
    query_property,
    register_property,
)
from finchlite.algebra.algebra import FinchOperator
from finchlite.finch_assembly import BufferFType, DictFType
from finchlite.symbolic import Context, Namespace, ScopedDict
from finchlite.util import config, file_cache
from finchlite.util.logging import LOG_BACKEND_C

from .stages import CCode, CLowerer
from .c_codegen import (
    CBufferFType,
    CContext,
    CDictFType,
    CKernel,
    CLibrary,
    c_getattr,
    c_literal,
    c_setattr,
    c_function_call,
    c_type,
    load_shared_lib,
)

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_BACKEND_C)

class CKernel(asm.AssemblyKernel):
    """
    A class to represent a C kernel.
    """

    def __init__(self, c_function, ret_type, argtypes):
        self.c_function = c_function
        self.ret_type = ret_type
        self.argtypes = argtypes
        self.c_function.restype = c_type(ret_type)
        self.c_function.argtypes = tuple(c_type(argtype) for argtype in argtypes)

    def __call__(self, *args):
        """
        Calls the C function with the given arguments.
        """
        if len(args) != len(self.argtypes):
            raise ValueError(
                f"Expected {len(self.argtypes)} arguments, got {len(args)}"
            )
        for argtype, arg in zip(self.argtypes, args, strict=False):
            if not fisinstance(arg, argtype):
                raise TypeError(f"Expected argument of type {argtype}, got {type(arg)}")
        serial_args = list(map(serialize_to_c, self.argtypes, args))
        res = self.c_function(*serial_args)
        for type_, arg, serial_arg in zip(
            self.argtypes, args, serial_args, strict=False
        ):
            deserialize_from_c(type_, arg, serial_arg)
        if self.ret_type is algebra.none_:
            return None
        return construct_from_c(self.ret_type, res)


class CLibrary(asm.AssemblyLibrary):
    """
    A class to represent a C module.
    """

    def __init__(self, c_module, kernels):
        self.c_module = c_module
        self.kernels = kernels

    def __getattr__(self, name):
        # Allow attribute access to kernels by name
        if name in self.kernels:
            return self.kernels[name]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


class CCompiler(asm.AssemblyLoader):
    """
    A class to compile and run FinchAssembly.
    """

    def __init__(
        self, ctx: CLowerer | None = None, cc=None, cflags=None, shared_cflags=None
    ):
        if cc is None:
            cc = config.get("cc")
        if cflags is None:
            cflags = config.get("cflags").split()
        if shared_cflags is None:
            shared_cflags = config.get("shared_cflags").split()
        self.cc = cc
        self.cflags = cflags
        self.shared_cflags = shared_cflags
        self.ctx: CLowerer = CGenerator() if ctx is None else ctx

    def __call__(self, prgm: asm.Module) -> CLibrary:
        c_code = self.ctx(prgm).code
        logger.debug(f"Compiling C code:\n{c_code}")
        lib = load_shared_lib(
            c_code=c_code,
            cc=self.cc,
            cflags=(*self.cflags, *self.shared_cflags),
        )
        kernels = {}
        if prgm.head() != asm.Module:
            raise ValueError(
                "CCompiler expects a Module as the head of the program, "
                f"got {type(prgm.head())}"
            )
        for func in prgm.funcs:
            match func:
                case asm.Function(asm.Variable(func_name, return_t), args, _):
                    # return_t = c_type(return_t)
                    arg_ts = [arg.result_type for arg in args]
                    kern = CKernel(getattr(lib, func_name), return_t, arg_ts)
                    kernels[func_name] = kern
                case _:
                    raise NotImplementedError(
                        f"Unrecognized function type: {type(func)}"
                    )
        return CLibrary(lib, kernels)

class CGenerator(CLowerer):
    def __call__(self, prgm: asm.AssemblyNode):
        ctx = CContext()
        CGeneratorContext(ctx)(prgm)
        return CCode(ctx.emit_global())

class CGeneratorContext:
    """
    Lowers Finch Assembly into C code while keeping assembly scope state.

    The held CContext owns C emission details, such as headers, indentation,
    fresh names, emitted statements, and datastructure metadata.
    """

    def __init__(self, ctx: CContext | None = None, types=None, slots=None):
        self.ctx = CContext() if ctx is None else ctx
        self.types = ScopedDict() if types is None else types
        self.slots = ScopedDict() if slots is None else slots

    def __getattr__(self, name):
        return getattr(self.ctx, name)

    @property
    def feed(self) -> str:
        return self.ctx.feed

    def block(self) -> "CGeneratorContext":
        return CGeneratorContext(self.ctx.block(), self.types, self.slots)

    def subblock(self):
        return CGeneratorContext(
            self.ctx.subblock(), self.types.scope(), self.slots.scope()
        )

    def resolve(self, node):
        match node:
            case asm.Slot(var_n, var_t):
                if var_n in self.slots:
                    var_o = self.slots[var_n]
                    return asm.Stack(var_o, var_t)
                raise KeyError(f"Slot {var_n} not found in context")
            case asm.Stack(_, _):
                return node
            case _:
                raise ValueError(f"Expected Slot or Stack, got: {type(node)}")

    def resolve_fields(self, node):
        stack = self.resolve(node)
        return stack.result_type, stack.obj

    def emit(self):
        return self.ctx.emit()

    def cache(self, name, val):
        if isinstance(val, asm.Literal | asm.Variable | asm.Stack):
            return val
        var_n = self.freshen(name)
        var_t = val.result_type
        var_t_code = self.ctype_name(c_type(var_t))
        self.exec(f"{self.feed}{var_t_code} {var_n} = {self(val)};")
        return asm.Variable(var_n, var_t)

    def __call__(self, prgm: asm.AssemblyNode):
        feed = self.feed
        """
        lower the program to C code.
        """
        match prgm:
            case asm.Literal(value):
                # in the future, would be nice to be able to pass in constants that
                # are more complex than C literals, maybe as globals.
                return c_literal(self, value)
            case asm.Variable(name, t):
                return name
            case asm.Assign(asm.Variable(var_n, var_t), val):
                val_code = self(val)
                if val.result_type != var_t:
                    raise TypeError(f"Type mismatch: {val.result_type} != {var_t}")
                if var_n in self.types:
                    assert var_t == self.types[var_n]
                    self.exec(f"{feed}{var_n} = {val_code};")
                else:
                    self.types[var_n] = var_t
                    var_t_code = self.ctype_name(c_type(var_t))
                    self.exec(f"{feed}{var_t_code} {var_n} = {val_code};")
                return None
            case asm.GetAttr(obj, attr):
                obj_t = obj.result_type
                if not isinstance(obj_t, StructFType):
                    raise TypeError(f"Expected struct type, got: {obj_t}")
                if not obj_t.struct_hasattr(attr.val):
                    raise ValueError("trying to get missing attr")
                return c_getattr(obj_t, self, self(obj), attr.val)
            case asm.SetAttr(obj, attr, val):
                obj = self.cache("obj", obj)
                obj_t = obj.result_type
                if not isinstance(obj_t, StructFType):
                    raise TypeError(f"Expected struct type, got: {obj_t}")
                if not fisinstance(val, obj_t.struct_attrtype(attr.val)):
                    raise TypeError(
                        f"Type mismatch: {val.result_type} != "
                        f"{obj_t.struct_attrtype(attr.val)}"
                    )
                val_code = self(val)
                c_setattr(obj_t, self, self(obj), attr.val, val_code)
                return None
            case asm.Call(f, args):
                return c_function_call(f.val, self, *args)
            # case asm.Slot(var_n, var_t) as ref:
            #    return self(self.deref(ref))
            # case asm.Stack(obj, var_t) as ref:
            #    return var_t.c_lower(self, obj)
            case asm.Unpack(asm.Slot(var_n, var_t), val):
                val_code = self(val)
                if val.result_type != var_t:
                    raise TypeError(f"Type mismatch: {val.result_type} != {var_t}")
                if var_n in self.slots:
                    raise KeyError(
                        f"Slot {var_n} already exists in context, cannot unpack"
                    )
                if var_n in self.types:
                    raise KeyError(
                        f"Variable '{var_n}' is already defined in the current"
                        f" context, cannot overwrite with slot."
                    )
                var_t_code = self.ctype_name(c_type(var_t))
                self.exec(f"{feed}{var_t_code} {var_n} = {val_code};")
                self.types[var_n] = var_t
                self.slots[var_n] = var_t.c_unpack(self, var_n, var_t, var_n, var_t)
                return None
            case asm.Repack(asm.Slot(var_n, var_t)):
                if var_n not in self.slots or var_n not in self.types:
                    raise KeyError(f"Slot {var_n} not found in context, cannot repack")
                if var_t != self.types[var_n]:
                    raise TypeError(f"Type mismatch: {var_t} != {self.types[var_n]}")
                obj = self.slots[var_n]
                var_t.c_repack(self, var_n, var_t, obj)
                return None
            case asm.Load(buf, idx):
                buf_t, buf_fields = self.resolve_fields(buf)
                if not isinstance(buf_t, CBufferFType):
                    raise TypeError(f"Expected C buffer type, got: {buf_t}")
                idx_symbol = self(idx)
                return buf_t.c_load(
                    self, buf_t, buf_fields, idx_symbol, idx.result_type
                )
            case asm.Store(buf, idx, val):
                buf_t, buf_fields = self.resolve_fields(buf)
                if not isinstance(buf_t, CBufferFType):
                    raise TypeError(f"Expected C buffer type, got: {buf_t}")
                idx_symbol = self(idx)
                val_symbol = self(val)
                return buf_t.c_store(
                    self,
                    buf_t,
                    buf_fields,
                    idx_symbol,
                    idx.result_type,
                    val_symbol,
                    val.result_type,
                )
            case asm.Resize(buf, len):
                buf_t, buf_fields = self.resolve_fields(buf)
                if not isinstance(buf_t, CBufferFType):
                    raise TypeError(f"Expected C buffer type, got: {buf_t}")
                len_symbol = self(len)
                return buf_t.c_resize(
                    self, buf_t, buf_fields, len_symbol, len.result_type
                )
            case asm.Length(buf):
                buf_t, buf_fields = self.resolve_fields(buf)
                if not isinstance(buf_t, CBufferFType):
                    raise TypeError(f"Expected C buffer type, got: {buf_t}")
                return buf_t.c_length(self, buf_t, buf_fields)
            case asm.LoadDict(map, idx):
                map_t, map_fields = self.resolve_fields(map)
                if not isinstance(map_t, CDictFType):
                    raise TypeError(f"Expected C dict type, got: {map_t}")
                idx_symbol = self(idx)
                return map_t.c_loaddict(
                    self, map_t, map_fields, idx_symbol, idx.result_type
                )
            case asm.ExistsDict(map, idx):
                map_t, map_fields = self.resolve_fields(map)
                if not isinstance(map_t, CDictFType):
                    raise TypeError(f"Expected C dict type, got: {map_t}")
                idx_symbol = self(idx)
                return map_t.c_existsdict(
                    self, map_t, map_fields, idx_symbol, idx.result_type
                )
            case asm.StoreDict(map, idx, val):
                map_t, map_fields = self.resolve_fields(map)
                if not isinstance(map_t, CDictFType):
                    raise TypeError(f"Expected C dict type, got: {map_t}")
                idx_symbol = self(idx)
                val_symbol = self(val)
                return map_t.c_storedict(
                    self,
                    map_t,
                    map_fields,
                    idx_symbol,
                    idx.result_type,
                    val_symbol,
                    val.result_type,
                )
            case asm.Block(bodies):
                ctx_2 = self.block()
                for body in bodies:
                    ctx_2(body)
                self.exec(ctx_2.emit())
                return None
            case asm.ForLoop(asm.Variable(_, _) as var, start, end, body):
                var_t = self.ctype_name(c_type(var.result_type))
                var_2 = self(var)
                start = self(start)
                end = self(end)
                ctx_2 = self.subblock()
                ctx_2(body)
                ctx_2.types[var.name] = var.result_type
                body_code = ctx_2.emit()
                self.exec(
                    f"{feed}for ({var_t} {var_2} = {start}; "
                    f"{var_2} < {end}; {var_2}++) {{\n"
                    f"{body_code}"
                    f"\n{feed}}}"
                )
                return None
            case asm.BufferLoop(buf, asm.Variable(_, t) as var, body):
                if not isinstance(buf.result_type, BufferFType):
                    raise TypeError(f"Expected buffer type, got: {buf.result_type}")
                idx = asm.Variable(
                    self.freshen(var.name + "_i"), buf.result_type.length_type
                )
                start = asm.Literal(t(0))
                stop = asm.Call(
                    asm.Literal(ffuncs.sub), (asm.Length(buf), asm.Literal(t(1)))
                )
                body_2 = asm.Block((asm.Assign(var, asm.Load(buf, idx)), body))
                return self(asm.ForLoop(idx, start, stop, body_2))
            case asm.WhileLoop(cond, body):
                if not isinstance(cond, asm.Literal | asm.Variable):
                    cond_var = asm.Variable(self.freshen("cond"), cond.result_type)
                    new_prgm = asm.Block(
                        (
                            asm.Assign(cond_var, cond),
                            asm.WhileLoop(
                                cond_var,
                                asm.Block(
                                    (
                                        body,
                                        asm.Assign(cond_var, cond),
                                    )
                                ),
                            ),
                        )
                    )
                    return self(new_prgm)
                cond_code = self(cond)
                ctx_2 = self.subblock()
                ctx_2(body)
                body_code = ctx_2.emit()
                self.exec(f"{feed}while ({cond_code}) {{\n{body_code}\n{feed}}}")
                return None
            case asm.If(cond, body):
                cond_code = self(cond)
                ctx_2 = self.subblock()
                ctx_2(body)
                body_code = ctx_2.emit()
                self.exec(f"{feed}if ({cond_code}) {{\n{body_code}\n{feed}}}")
                return None
            case asm.IfElse(cond, body, else_body):
                cond_code = self(cond)
                ctx_2 = self.subblock()
                ctx_2(body)
                body_code = ctx_2.emit()
                ctx_3 = self.subblock()
                ctx_3(else_body)
                else_body_code = ctx_3.emit()
                self.exec(
                    f"{feed}if ({cond_code}) {{\n{body_code}\n{feed}}} "
                    f"else {{\n{else_body_code}\n{feed}}}"
                )
                return None
            case asm.Function(asm.Variable(func_name, return_t), args, body):
                ctx_2 = self.subblock()
                arg_decls = []
                for arg in args:
                    match arg:
                        case asm.Variable(name, t):
                            t_name = self.ctype_name(c_type(t))
                            arg_decls.append(f"{t_name} {name}")
                            ctx_2.types[name] = t
                        case _:
                            raise NotImplementedError(
                                f"Unrecognized argument type: {arg}"
                            )
                ctx_2(body)
                body_code = ctx_2.emit()
                return_t_name = self.ctype_name(c_type(return_t))
                feed = self.feed
                self.exec(
                    f"{feed}{return_t_name} {func_name}({', '.join(arg_decls)}) {{\n"
                    f"{body_code}\n"
                    f"{feed}}}"
                )
                return None
            case asm.Return(value):
                value = self(value)
                self.exec(f"{feed}return {value};")
                return None
            case asm.Break():
                self.exec(f"{feed}break;")
                return None
            case asm.Module(funcs):
                for func in funcs:
                    if not isinstance(func, asm.Function):
                        raise NotImplementedError(
                            f"Unrecognized function type: {type(func)}"
                        )
                    self(func)
                return None
            case _:
                raise NotImplementedError(
                    f"Unrecognized assembly node type: {type(prgm)}"
                )
