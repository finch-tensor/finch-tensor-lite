# AI modified: 2026-04-10T21:26:00Z 9e1552f1
import logging
from abc import ABC, abstractmethod
from textwrap import dedent
from typing import Any

import numpy as np

import numba

from .. import algebra
from .. import finch_assembly as asm
from ..algebra import (
    ImmutableStructFType,
    MutableStructFType,
    NumbaOperator,
    StructFType,
    TupleFType,
    ffuncs,
    fisinstance,
    ftype,
    query_property,
    register_property,
)
from ..finch_assembly import BufferFType
from ..finch_assembly.dct import DictFType
from ..symbolic import Context, Namespace, ScopedDict
from ..util.logging import LOG_BACKEND_NUMBA
from .stages import NumbaCode, NumbaLowerer

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_BACKEND_NUMBA)


# Cache for Numba structs
numba_structs: dict[Any, Any] = {}
numba_structnames = Namespace()
numba_globals: dict[str, Any] = {"scansearch": numba.njit(ffuncs.scansearch._func)}


def _normalize_fmt(fmt):
    try:
        return ftype(fmt)
    except NotImplementedError:
        return fmt


def to_numpy_type(t: Any) -> np.dtype:
    """Return a NumPy dtype for a Finch scalar/data type."""
    if isinstance(t, np.dtype):
        return t

    t = _normalize_fmt(t)
    if isinstance(t, algebra.ftypes.FDTypeNumpy):
        return np.dtype(t.dtype)
    if isinstance(t, algebra.ftypes.FDTypeBuiltin):
        return np.dtype(t.type)
    return np.dtype(t)


def numba_type(t):
    """
    Returns the Numba type/ftype after serialization.

    Args:
        t: The Python type/ftype.

    Returns:
        The corresponding Numba type.
    """
    t = _normalize_fmt(t)
    if isinstance(t, algebra.ftypes.FDTypeNumpy):
        return numba.from_dtype(t.dtype)
    if isinstance(t, algebra.ftypes.FDTypeBuiltin):
        return t.type
    if hasattr(t, "numba_type"):
        return t.numba_type()
    try:
        return query_property(t, "numba_type", "__attr__")
    except AttributeError:
        return t


def numba_jitclass_type(t):
    """
    Returns the Numba jitclass spec type/ftype after serialization.

    Args:
        t: The Python type/ftype.

    Returns:
        The corresponding Numba jitclass spec type.
    """
    t = _normalize_fmt(t)
    if hasattr(t, "numba_jitclass_type"):
        return t.numba_jitclass_type()
    return query_property(t, "numba_jitclass_type", "__attr__")


register_property(
    algebra.ftypes.FDTypeNumpy,
    "numba_type",
    "__attr__",
    lambda t: numba.from_dtype(t.dtype),
)
register_property(
    algebra.ftypes.FDTypeBuiltin,
    "numba_type",
    "__attr__",
    lambda t: t.type,
)


def immutable_struct_jitclass_type(fmt: ImmutableStructFType):
    return numba.types.Tuple(
        tuple([numba_jitclass_type(t) for t in fmt.struct_fieldtypes])
    )


register_property(
    ImmutableStructFType,
    "numba_jitclass_type",
    "__attr__",
    immutable_struct_jitclass_type,
)
register_property(
    ImmutableStructFType,
    "numba_type",
    "__attr__",
    lambda t: tuple,
)

register_property(
    TupleFType,
    "numba_jitclass_type",
    "__attr__",
    immutable_struct_jitclass_type,
)
register_property(
    TupleFType,
    "numba_type",
    "__attr__",
    lambda t: tuple,
)


def assembly_struct_numba_type(ftype_: Any) -> type:
    """
    Method for registering and caching Numba jitclass.
    """
    if ftype_ in numba_structs:
        return numba_structs[ftype_]

    spec = [
        (name, numba_jitclass_type(field_type))
        for (name, field_type) in ftype_.struct_fields
    ]
    class_name = numba_structnames.freshen("Numba", ftype_.struct_name)
    # Dynamically define __init__ based on spec, unrolling the arguments
    field_names = [name for name, _ in spec]
    # Build the argument list for __init__
    arg_list = ", ".join(field_names)
    # Build the body of __init__ to assign each argument to ftype_
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
    numba_structs[ftype_] = new_struct
    numba_globals[new_struct.__name__] = new_struct
    # logger.debug(f"Numba class:\njitclass({spec})\n{class_src}")
    return new_struct


register_property(
    StructFType,
    "numba_type",
    "__attr__",
    assembly_struct_numba_type,
)


def assembly_struct_numba_jitclass_type(ftype_) -> numba.types.Type:
    return numba_type(ftype_).class_type.instance_type


register_property(
    algebra.ftypes.FDTypeNumpy,
    "numba_jitclass_type",
    "__attr__",
    lambda t: numba.from_dtype(t.dtype),
)

register_property(
    algebra.int_,
    "numba_jitclass_type",
    "__attr__",
    lambda t: numba.int32,
)

register_property(
    algebra.float_,
    "numba_jitclass_type",
    "__attr__",
    lambda t: numba.float64,
)

register_property(
    StructFType,
    "numba_jitclass_type",
    "__attr__",
    assembly_struct_numba_jitclass_type,
)


class NumbaArgumentFType(ABC):
    @abstractmethod
    def serialize_to_numba(self, obj):
        """
        Return a Numba-compatible object to be used in place of this argument
        for the Numba backend.
        """
        ...

    @abstractmethod
    def deserialize_from_numba(self, obj, numba_buffer):
        """
        Return an object from Numba returned value.
        """
        ...

    @abstractmethod
    def construct_from_numba(self, numba_buffer):
        """
        Construct and return an object from Numba returned value.
        """
        ...


def serialize_to_numba(fmt, obj):
    """
    Serialize an object to a Numba-compatible ftype.

    Args:
        fmt: FType of obj
        obj: The object to serialize.

    Returns:
        A Numba-compatible object.
    """
    fmt = _normalize_fmt(fmt)
    if fmt is type(None):
        return None
    if hasattr(fmt, "serialize_to_numba"):
        return fmt.serialize_to_numba(obj)
    return query_property(fmt, "serialize_to_numba", "__attr__", obj)


register_property(
    algebra.ftypes.FDTypeNumpy,
    "serialize_to_numba",
    "__attr__",
    lambda fmt, numba_obj: numba_obj,
)


def serialize_immutable_to_numba(fmt: ImmutableStructFType, obj):
    return tuple(
        serialize_to_numba(childfmt, fmt.struct_getattr(obj, attr))
        for attr, childfmt in fmt.struct_fields
    )


register_property(
    ImmutableStructFType, "serialize_to_numba", "__attr__", serialize_immutable_to_numba
)

register_property(
    TupleFType, "serialize_to_numba", "__attr__", serialize_immutable_to_numba
)


def immutable_construct_from_numba(fmt: StructFType, numba_tuple):
    return fmt.from_fields(
        *[
            construct_from_numba(field_type, field_value)
            for field_value, (name, field_type) in zip(
                numba_tuple, fmt.struct_fields, strict=False
            )
        ]
    )


register_property(
    ImmutableStructFType,
    "construct_from_numba",
    "__attr__",
    immutable_construct_from_numba,
)

register_property(
    TupleFType,
    "construct_from_numba",
    "__attr__",
    immutable_construct_from_numba,
)


def deserialize_from_numba(fmt, obj, numba_obj):
    """
    Deserialize a Numba-compatible object back to the original ftype.

    Args:
        fmt: FType of obj
        obj: The original object to update.
        numba_obj: The Numba-compatible object to deserialize from.

    Returns:
        None
    """
    fmt = _normalize_fmt(fmt)
    if fmt is type(None):
        return
    if hasattr(fmt, "deserialize_from_numba"):
        fmt.deserialize_from_numba(obj, numba_obj)
    else:
        try:
            query_property(fmt, "deserialize_from_numba", "__attr__", obj, numba_obj)
        except AttributeError:
            return


register_property(
    algebra.ftypes.FDTypeNumpy,
    "deserialize_from_numba",
    "__attr__",
    lambda fmt, obj, numba_obj: None,
)


def construct_from_numba(fmt, numba_obj):
    """
    Construct an object from a Numba-compatible ftype.

    Args:
        fmt: The ftype of the object.
        numba_obj: The Numba-compatible object to construct from.

    Returns:
        An instance of the original object type.
    """
    fmt = _normalize_fmt(fmt)
    if fmt is type(None):
        return None
    if hasattr(fmt, "construct_from_numba"):
        return fmt.construct_from_numba(numba_obj)
    try:
        return query_property(fmt, "construct_from_numba", "__attr__", numba_obj)
    except NotImplementedError:
        return fmt(numba_obj)


register_property(
    algebra.ftypes.FDTypeNumpy,
    "construct_from_numba",
    "__attr__",
    lambda fmt, numba_obj: fmt(numba_obj),
)


class NumbaDictFType(DictFType, NumbaArgumentFType, ABC):
    """
    Abstract base class for the ftype of datastructures. The ftype defines how
    the data in a Map is organized and accessed.
    """

    @abstractmethod
    def numba_existsdict(self, ctx: "NumbaContext", map, idx):
        """
        Return numba code which checks whether a given key exists in a map.
        """
        ...

    @abstractmethod
    def numba_loaddict(self, ctx, buffer, idx):
        """
        Return numba code which gets a value corresponding to a certain key.
        """
        ...

    @abstractmethod
    def numba_storedict(self, ctx, buffer, idx, value):
        """
        Return C code which stores a certain value given a certain integer tuple key.
        """
        ...


class NumbaBufferFType(BufferFType, NumbaArgumentFType, ABC):
    @abstractmethod
    def numba_length(self, ctx: "NumbaContext", buffer):
        """
        Return a Numba-compatible expression to get the length of the buffer.
        """
        ...

    @abstractmethod
    def numba_resize(self, ctx: "NumbaContext", buffer, size):
        """
        Return a Numba-compatible expression to resize the buffer to the given size.
        """
        ...

    @abstractmethod
    def numba_load(self, ctx: "NumbaContext", buffer, idx):
        """
        Return a Numba-compatible expression to load an element from the buffer
        at the given index.
        """
        ...

    @abstractmethod
    def numba_store(self, ctx: "NumbaContext", buffer, idx, value=None):
        """
        Return a Numba-compatible expression to store an element in the buffer
        at the given index. If value is None, it should store the length of the
        buffer.
        """
        ...

    @staticmethod
    def numba_name():
        return "list[numpy.ndarray]"


class NumbaLibrary(asm.AssemblyLibrary):
    """
    A class to represent a Numba module.
    """

    def __init__(self, kernels):
        self.kernels = kernels

    def __getattr__(self, name):
        # Allow attribute access to kernels by name
        if name in self.kernels:
            return self.kernels[name]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


class NumbaKernel(asm.AssemblyKernel):
    def __init__(self, numba_func, ret_type: Any, arg_types):
        self.numba_func = numba_func
        self.ret_type = ret_type
        self.arg_types = arg_types

    def __call__(self, *args):
        for arg_type, arg in zip(self.arg_types, args, strict=False):
            if not fisinstance(arg, arg_type):
                raise TypeError(
                    f"Expected argument of type {arg_type}, got {type(arg)}"
                )
        serial_args = list(map(serialize_to_numba, self.arg_types, args))
        res = self.numba_func(*serial_args)
        for arg_type, arg, serial_arg in zip(
            self.arg_types, args, serial_args, strict=False
        ):
            deserialize_from_numba(arg_type, arg, serial_arg)
        if self.ret_type is type(None):
            return None
        return construct_from_numba(self.ret_type, res)


class NumbaCompiler(asm.AssemblyLoader):
    def __init__(self, ctx: NumbaLowerer | None = None):
        if ctx is None:
            ctx = NumbaGenerator()
        self.ctx: NumbaLowerer = ctx

    def __call__(self, prgm: asm.Module) -> NumbaLibrary:
        numba_code = self.ctx(prgm).code
        logger.debug(f"Executing Numba code:\n{numba_code}")
        _globals = globals()
        _globals |= numba_globals
        try:
            exec(numba_code, _globals, None)
        except Exception as e:
            logger.error(
                f"Numba compilation failed on the following code:\n"
                f"{numba_code}\n"
                f"Error message: {e}"
            )
            raise e

        kernels = {}
        for func in prgm.funcs:
            match func:
                case asm.Function(asm.Variable(func_name, ret_type), args, _):
                    kern = _globals[func_name]
                    arg_ts = [arg.result_type for arg in args]
                    kernels[func_name] = NumbaKernel(kern, ret_type, arg_ts)
                case _:
                    raise NotImplementedError(
                        f"Unrecognized function type: {type(func)}"
                    )

        return NumbaLibrary(kernels)


class NumbaGenerator(NumbaLowerer):
    def __call__(self, prgm: asm.AssemblyNode):
        ctx = NumbaContext()
        ctx(prgm)
        return NumbaCode(ctx.emit_global())


class NumbaContext(Context):
    def __init__(self, tab="    ", indent=0, types=None, slots=None):
        if types is None:
            types = ScopedDict()
        if slots is None:
            slots = ScopedDict()

        super().__init__()

        self.tab = tab
        self.indent = indent
        self.types = types
        self.slots = slots

        self.imports = [
            "import _operator, builtins",
            "from numba import njit",
            "import numpy",
            "from numpy import int64, float64",
            "\n",
        ]

    @property
    def feed(self) -> str:
        return self.tab * self.indent

    def emit_global(self):
        """
        Emit the headers for the C code.
        """
        return "\n".join([*self.imports, self.emit()])

    def emit(self):
        return "\n".join([*self.preamble, *self.epilogue])

    def block(self) -> "NumbaContext":
        blk = super().block()
        blk.indent = self.indent
        blk.tab = self.tab
        blk.types = self.types
        blk.slots = self.slots
        return blk

    def subblock(self):
        blk = self.block()
        blk.indent = self.indent + 1
        blk.types = self.types.scope()
        blk.slots = self.slots.scope()
        return blk

    def cache(self, name, val):
        if isinstance(val, asm.Literal | asm.Variable | asm.Stack):
            return val
        var_n = self.freshen(name)
        var_t = val.result_type
        self.exec(f"{self.feed}{var_n} = {self(val)}")
        return asm.Variable(var_n, var_t)

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

    @staticmethod
    def full_name(val: Any) -> str:
        if isinstance(val, algebra.ftypes.FType):
            val = numba_type(val)
        if hasattr(val, "numba_name"):
            return val.numba_name()
        if hasattr(val, "name"):
            return val.name
        if not hasattr(val, "__module__") or not hasattr(val, "__name__"):
            return str(val)
        return f"{val.__module__}.{val.__name__}"

    def __call__(self, prgm: asm.AssemblyNode):
        feed = self.feed
        match prgm:
            case asm.Literal(value):
                return str(value)
            case asm.Variable(name, _) | asm.Slot(name, _):
                return name.replace("#", "_")
            case asm.Assign(asm.Variable(var_n, var_t) as var, val):
                val_code = self(val)
                var_code = self(var)
                if val.result_type != var_t:
                    raise TypeError(f"Type mismatch: {val.result_type} != {var_t}")
                if var_n in self.types:
                    assert var_t == self.types[var_n]
                    self.exec(f"{feed}{var_code} = {val_code}")
                else:
                    self.types[var_n] = var_t
                    self.exec(
                        f"{feed}{var_code}: "
                        f"{self.full_name(numba_type(var_t))} = {val_code}"
                    )
                return None
            case asm.GetAttr(obj, attr):
                obj_code = self(obj)
                obj_t = obj.result_type
                if not isinstance(obj_t, StructFType):
                    raise TypeError(f"Expected struct type, got: {obj_t}")
                if not obj_t.struct_hasattr(attr.val):
                    raise ValueError(f"trying to get missing attr: {attr}")
                return query_property(
                    obj_t,
                    "numba_getattr",
                    "__attr__",
                    self,
                    obj_code,
                    attr.val,
                )
            case asm.SetAttr(obj, attr, val):
                obj_code = self(obj)
                obj_t = obj.result_type
                if not isinstance(obj_t, StructFType):
                    raise TypeError(f"Expected struct type, got: {obj_t}")
                if not fisinstance(val, obj_t.struct_attrtype(attr.val)):
                    raise TypeError(
                        f"Type mismatch: {val.result_type} != "
                        f"{obj_t.struct_attrtype(attr.val)}"
                    )
                val_code = self(val)
                query_property(
                    obj_t,
                    "numba_setattr",
                    "__attr__",
                    self,
                    obj_code,
                    attr.val,
                    val_code,
                )
                return None
            case asm.Call(asm.Literal(op), args):
                if not isinstance(op, NumbaOperator):
                    raise TypeError(f"{op} has no Numba representation.")
                return op.numba_literal(op, self, *args)

            case asm.Unpack(asm.Slot(var_n, var_t) as slot, val):
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
                var_code = self(slot)
                self.exec(f"{feed}{var_code} = {self(val)}")
                self.types[var_n] = var_t
                self.slots[var_n] = var_t.numba_unpack(
                    self, var_code, asm.Variable(var_n, var_t)
                )
                return None
            case asm.Repack(asm.Slot(var_n, var_t) as slot):
                if var_n not in self.slots or var_n not in self.types:
                    raise KeyError(f"Slot {var_n} not found in context, cannot repack")
                if var_t != self.types[var_n]:
                    raise TypeError(f"Type mismatch: {var_t} != {self.types[var_n]}")
                obj = self.slots[var_n]
                var_code = self(slot)
                var_t.numba_repack(self, var_code, obj)
                return None
            case asm.Load(buf, idx):
                buf = self.resolve(buf)
                buf_t = buf.result_type
                if not isinstance(buf_t, NumbaBufferFType):
                    raise TypeError(f"Expected numba buffer type, got: {buf_t}")
                return buf_t.numba_load(self, buf, idx)
            case asm.Store(buf, idx, val):
                buf = self.resolve(buf)
                buf_t = buf.result_type
                if not isinstance(buf_t, NumbaBufferFType):
                    raise TypeError(f"Expected numba buffer type, got: {buf_t}")
                buf_t.numba_store(self, buf, idx, val)
                return None
            case asm.Resize(buf, size):
                buf = self.resolve(buf)
                buf_t = buf.result_type
                if not isinstance(buf_t, NumbaBufferFType):
                    raise TypeError(f"Expected numba buffer type, got: {buf_t}")
                buf_t.numba_resize(self, buf, size)
                return None
            case asm.Length(buf):
                buf = self.resolve(buf)
                buf_t = buf.result_type
                if not isinstance(buf_t, NumbaBufferFType):
                    raise TypeError(f"Expected numba buffer type, got: {buf_t}")
                return buf_t.numba_length(self, buf)
            case asm.LoadDict(dct, idx):
                dct = self.resolve(dct)
                dct_t = dct.result_type
                if not isinstance(dct_t, NumbaDictFType):
                    raise TypeError(f"Expected numba dict type, got: {dct_t}")
                return dct_t.numba_loaddict(self, dct, idx)
            case asm.ExistsDict(dct, idx):
                dct = self.resolve(dct)
                dct_t = dct.result_type
                if not isinstance(dct_t, NumbaDictFType):
                    raise TypeError(f"Expected numba dict type, got: {dct_t}")
                return dct_t.numba_existsdict(self, dct, idx)
            case asm.StoreDict(dct, idx, val):
                dct = self.resolve(dct)
                dct_t = dct.result_type
                if not isinstance(dct_t, NumbaDictFType):
                    raise TypeError(f"Expected numba dict type, got: {dct_t}")
                return dct_t.numba_storedict(self, dct, idx, val)
            case asm.Block(bodies):
                ctx_2 = self.block()
                if bodies == ():
                    self.exec(f"{feed}pass")
                else:
                    for body in bodies:
                        ctx_2(body)
                    self.exec(ctx_2.emit())
                return None
            case asm.ForLoop(asm.Variable(_, _) as var, start, end, body):
                var_2 = self(var)
                start = self(start)
                end = self(end)
                ctx_2 = self.subblock()
                ctx_2(body)
                ctx_2.types[var.name] = var.result_type
                body_code = ctx_2.emit()
                self.exec(f"{feed}for {var_2} in range({start}, {end}):\n{body_code}")
                return None
            case asm.BufferLoop(buf, var, body):
                raise NotImplementedError
            case asm.WhileLoop(cond, body):
                cond_code = self(cond)
                ctx_2 = self.subblock()
                ctx_2(body)
                body_code = ctx_2.emit()
                self.exec(f"{feed}while {cond_code}:\n{body_code}")
                return None
            case asm.If(cond, body):
                cond_code = self(cond)
                ctx_2 = self.subblock()
                ctx_2(body)
                body_code = ctx_2.emit()
                self.exec(f"{feed}if {cond_code}:\n{body_code}")
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
                    f"{feed}if {cond_code}:\n{body_code}\n{feed}else:\n{else_body_code}"
                )
                return None
            case asm.Function(asm.Variable(func_name, return_t), args, body):
                ctx_2 = self.subblock()
                arg_decls = []
                for arg in args:
                    match arg:
                        case asm.Variable(name, t) as var:
                            name_code = self(var)
                            arg_decls.append(
                                f"{name_code}: {self.full_name(numba_type(t))}"
                            )
                            ctx_2.types[name] = t
                        case _:
                            raise NotImplementedError(
                                f"Unrecognized argument type: {arg}"
                            )
                ctx_2(body)
                body_code = ctx_2.emit()
                feed = self.feed
                self.exec(
                    f"{feed}@njit\n"
                    f"{feed}def {func_name}({', '.join(arg_decls)}) -> "
                    f"{self.full_name(numba_type(return_t))}:\n"
                    f"{body_code}\n"
                )
                return None
            case asm.Return(value):
                value = self(value)
                self.exec(f"{feed}return {value}")
                return None
            case asm.Break():
                self.exec(f"{feed}break")
                return None
            case asm.Module(funcs):
                for func in funcs:
                    if not isinstance(func, asm.Function):
                        raise NotImplementedError(
                            f"Unrecognized function type: {type(func)}"
                        )
                    self(func)
                return None
            case node:
                raise NotImplementedError(f"Unrecognized node: {node}")


class NumbaStackFType(ABC):
    """
    Abstract base class for symbolic formats in Numba. Stack formats must also
    support other functions with symbolic inputs in addition to variable ones.
    """

    @abstractmethod
    def numba_unpack(self, ctx, lhs, rhs):
        """
        Convert a value to a symbolic representation in Numba. Returns a NamedTuple
        of unpacked variable names, etc. The `lhs` is the variable namespace to
        assign to.
        """
        ...

    @abstractmethod
    def numba_repack(self, ctx, lhs, rhs):
        """
        Update an object based on a symbolic representation. The `rhs` is the
        symbolic representation to update from, and `lhs` is a variable name referring
        to the original object to update.
        """
        ...


def _serialize_asm_struct_to_numba(fmt: StructFType, obj) -> Any:
    args = [
        serialize_to_numba(fmt, getattr(obj, name)) for (name, fmt) in fmt.struct_fields
    ]
    return numba_type(fmt)(*args)


register_property(
    StructFType,
    "serialize_to_numba",
    "__attr__",
    _serialize_asm_struct_to_numba,
)


def _deserialize_asm_struct_from_numba(
    fmt: StructFType, obj, numba_struct: Any
) -> None:
    if fmt.is_mutable:
        for name in fmt.struct_fieldnames:
            setattr(obj, name, getattr(numba_struct, name))
        return


register_property(
    StructFType,
    "deserialize_from_numba",
    "__attr__",
    _deserialize_asm_struct_from_numba,
)

register_property(
    TupleFType,
    "deserialize_from_numba",
    "__attr__",
    _deserialize_asm_struct_from_numba,
)


def struct_numba_getattr(fmt: StructFType, ctx, obj, attr):
    return f"{obj}.{attr}"


register_property(
    StructFType,
    "numba_getattr",
    "__attr__",
    struct_numba_getattr,
)


def immutable_struct_numba_getattr(fmt: StructFType, ctx, obj, attr):
    index = list(fmt.struct_fieldnames).index(attr)
    return f"{obj}[{index}]"


register_property(
    ImmutableStructFType,
    "numba_getattr",
    "__attr__",
    immutable_struct_numba_getattr,
)

register_property(
    TupleFType,
    "numba_getattr",
    "__attr__",
    immutable_struct_numba_getattr,
)


def struct_numba_setattr(fmt: StructFType, ctx, obj, attr, val):
    ctx.emit(f"{ctx.feed}{obj}.{attr} = {val}")
    return


register_property(
    MutableStructFType,
    "numba_setattr",
    "__attr__",
    struct_numba_setattr,
)


def struct_construct_from_numba(fmt: StructFType, numba_struct):
    args = [
        construct_from_numba(field_type, getattr(numba_struct, name))
        for (name, field_type) in fmt.struct_fields
    ]
    return fmt.from_fields(*args)


register_property(
    StructFType,
    "construct_from_numba",
    "__attr__",
    struct_construct_from_numba,
)


# trivial ser/deser
for t in (algebra.int_, algebra.bool_, algebra.float_):
    register_property(
        t,
        "construct_from_numba",
        "__attr__",
        lambda fmt, obj: obj,
    )

    register_property(
        t,
        "serialize_to_numba",
        "__attr__",
        lambda fmt, obj: obj,
    )
