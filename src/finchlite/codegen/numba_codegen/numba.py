import logging
from abc import ABC, abstractmethod
from textwrap import dedent
from typing import Any

import numpy as np

import numba

from finchlite import algebra
from finchlite import finch_assembly as asm
from finchlite.algebra import (
    FType,
    ImmutableStructFType,
    MutableStructFType,
    StructFType,
    TupleFType,
    ffuncs,
    fisinstance,
)
from finchlite.finch_assembly import BufferFType
from finchlite.finch_assembly.dct import DictFType
from finchlite.symbolic import Context, Namespace, ScopedDict
from finchlite.util.logging import LOG_BACKEND_NUMBA

from .stages import NumbaCode, NumbaLowerer

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_BACKEND_NUMBA)


# Cache for Numba structs
numba_structs: dict[Any, Any] = {}
numba_structnames = Namespace()
numba_globals: dict[str, Any] = {
    "scansearch": numba.njit(ffuncs.scansearch._func),
    "resize_if_smaller": numba.njit(ffuncs.resize_if_smaller._func),
}


class NumbaOperator(ABC):
    @abstractmethod
    def numba_name(self) -> str: ...

    @abstractmethod
    def numba_function_call(self, val: Any, ctx: Any, *args: Any) -> Any: ...


class NumbaNAryOperator(NumbaOperator):
    def numba_function_call(self, val: Any, ctx: Any, *args: Any) -> Any:
        return numba_nary_function_call(self.numba_name(), ctx, *args)


class NumbaBinaryOperator(NumbaOperator):
    def numba_function_call(self, val: Any, ctx: Any, *args: Any) -> Any:
        return numba_binary_function_call(self.numba_name(), ctx, *args)


class NumbaUnaryOperator(NumbaOperator):
    def numba_function_call(self, val: Any, ctx: Any, *args: Any) -> Any:
        return numba_unary_function_call(self.numba_name(), ctx, *args)


def numba_function_name(op, ctx, *args: Any) -> str:
    match op:
        case ffuncs.add:
            return "+"
        case ffuncs.mul:
            return "*"
        case ffuncs.sub:
            return "-"
        case ffuncs.truediv:
            return "/"
        case ffuncs.floordiv:
            return "//"
        case ffuncs.mod:
            return "%"
        case ffuncs.pow:
            return "**"
        case ffuncs.lshift:
            return "<<"
        case ffuncs.rshift:
            return ">>"
        case ffuncs.and_:
            return "&"
        case ffuncs.xor:
            return "^"
        case ffuncs.or_:
            return "|"
        case ffuncs.not_:
            return "not "
        case ffuncs.invert:
            return "~"
        case ffuncs.eq:
            return "=="
        case ffuncs.ne:
            return "!="
        case ffuncs.gt:
            return ">"
        case ffuncs.lt:
            return "<"
        case ffuncs.ge:
            return ">="
        case ffuncs.le:
            return "<="
        case ffuncs.min:
            return "min"
        case ffuncs.max:
            return "max"
        case ffuncs.scansearch:
            return "scansearch"
        case ffuncs.resize_if_smaller:
            return "resize_if_smaller"
        case NumbaOperator():
            return op.numba_name()
        case _:
            raise TypeError(f"{op} has no Numba representation.")


def numba_nary_function_call(numba_name: str, ctx: Any, *args: Any) -> str:
    assert len(args) > 0
    if len(args) == 1:
        return f"({ctx(args[0])})"
    return f"({f' {numba_name} '.join(map(ctx, args))})"


def numba_binary_function_call(numba_name: str, ctx: Any, *args: Any) -> str:
    a, b = args
    return f"({ctx(a)} {numba_name} {ctx(b)})"


def numba_unary_function_call(numba_name: str, ctx: Any, *args: Any) -> str:
    return f"({numba_name}{ctx(args[0])})"


def numba_call_function_call(numba_name: str, ctx: Any, *args: Any) -> str:
    return f"{numba_name}({', '.join(map(ctx, args))})"


def numba_function_call(op, ctx, *args: Any) -> str:
    match op:
        case ffuncs._InitWrite():
            return ctx(args[1])
        case ffuncs.make_tuple:
            return f"({','.join([ctx(arg) for arg in args])},)"
        case NumbaOperator():
            return op.numba_function_call(op, ctx, *args)

    numba_name = numba_function_name(op, ctx, *args)
    match op:
        case ffuncs.add | ffuncs.mul | ffuncs.and_ | ffuncs.xor | ffuncs.or_:
            return numba_nary_function_call(numba_name, ctx, *args)
        case (
            ffuncs.sub
            | ffuncs.truediv
            | ffuncs.floordiv
            | ffuncs.mod
            | ffuncs.pow
            | ffuncs.lshift
            | ffuncs.rshift
            | ffuncs.eq
            | ffuncs.ne
            | ffuncs.gt
            | ffuncs.lt
            | ffuncs.ge
            | ffuncs.le
        ):
            return numba_binary_function_call(numba_name, ctx, *args)
        case ffuncs.not_ | ffuncs.invert:
            return numba_unary_function_call(numba_name, ctx, *args)
        case ffuncs.min | ffuncs.max | ffuncs.scansearch | ffuncs.resize_if_smaller:
            return numba_call_function_call(numba_name, ctx, *args)
        case _:
            raise TypeError(f"{op} has no Numba representation.")


class NumbaArgumentFType(ABC):
    @abstractmethod
    def numba_type(self):
        """
        Return a Numba-compatible type for this ftype.
        """
        ...

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


def to_numpy_type(t: FType) -> np.dtype:
    """Return a NumPy dtype for a Finch scalar/data type."""
    if isinstance(t, algebra.ftypes.FDTypeNumpy):
        return np.dtype(t.dtype)
    if isinstance(t, algebra.ftypes.FDTypeBuiltin):
        return np.dtype(t.type)
    raise NotImplementedError(f"No NumPy dtype mapping for {t}")


def numba_type(t: FType):
    """
    Returns the Numba type/ftype after serialization.

    Args:
        t: The Python type/ftype.

    Returns:
        The corresponding Numba type.
    """
    match t:
        case NumbaArgumentFType():
            return t.numba_type()
        case algebra.ftypes.FDTypeNumpy():
            return numba.from_dtype(t.dtype)
        case algebra.ftypes.FDTypeBuiltin():
            return t.type
        case TupleFType() | ImmutableStructFType():
            return tuple
        case StructFType():
            return assembly_struct_numba_type(t)
        case _:
            return t


def numba_jitclass_type(t: FType):
    """
    Returns the Numba jitclass spec type/ftype after serialization.

    Args:
        t: The Python type/ftype.

    Returns:
        The corresponding Numba jitclass spec type.
    """
    match t:
        case _ if hasattr(t, "numba_jitclass_type"):
            return t.numba_jitclass_type()
        case algebra.ftypes.FDTypeNumpy():
            return numba.from_dtype(t.dtype)
        case algebra.int_:
            return numba.int32
        case algebra.bool_:
            return numba.boolean
        case algebra.float_:
            return numba.float64
        case TupleFType() | ImmutableStructFType():
            return immutable_struct_jitclass_type(t)
        case StructFType():
            return assembly_struct_numba_jitclass_type(t)
        case _:
            raise NotImplementedError(f"No Numba jitclass type mapping for {t}")


def immutable_struct_jitclass_type(fmt: ImmutableStructFType):
    return numba.types.Tuple(
        tuple([numba_jitclass_type(t) for t in fmt.struct_fieldtypes])
    )


def assembly_struct_numba_type(ftype_: StructFType) -> type:
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


def assembly_struct_numba_jitclass_type(ftype_: StructFType) -> numba.types.Type:
    return numba_type(ftype_).class_type.instance_type


def serialize_to_numba(fmt: FType, obj):
    """
    Serialize an object to a Numba-compatible ftype.

    Args:
        fmt: FType of obj
        obj: The object to serialize.

    Returns:
        A Numba-compatible object.
    """
    match fmt:
        case NumbaArgumentFType():
            return fmt.serialize_to_numba(obj)
        case algebra.none_:
            return None
        case (
            algebra.ftypes.FDTypeNumpy() | algebra.int_ | algebra.bool_ | algebra.float_
        ):
            return obj
        case TupleFType() | ImmutableStructFType():
            return serialize_immutable_to_numba(fmt, obj)
        case StructFType():
            return _serialize_asm_struct_to_numba(fmt, obj)
        case _:
            raise NotImplementedError(f"No Numba serialization mapping for {fmt}")


def serialize_immutable_to_numba(fmt: ImmutableStructFType, obj):
    return tuple(
        serialize_to_numba(childfmt, fmt.struct_getattr(obj, attr))
        for attr, childfmt in fmt.struct_fields
    )


def immutable_construct_from_numba(fmt: ImmutableStructFType, numba_tuple):
    return fmt.from_fields(
        *[
            construct_from_numba(field_type, field_value)
            for field_value, (name, field_type) in zip(
                numba_tuple, fmt.struct_fields, strict=False
            )
        ]
    )


def deserialize_from_numba(fmt: FType, obj, numba_obj):
    """
    Deserialize a Numba-compatible object back to the original ftype.

    Args:
        fmt: FType of obj
        obj: The original object to update.
        numba_obj: The Numba-compatible object to deserialize from.

    Returns:
        None
    """
    match fmt:
        case NumbaArgumentFType():
            fmt.deserialize_from_numba(obj, numba_obj)
        case algebra.none_ | algebra.ftypes.FDTypeNumpy():
            return
        case StructFType():
            _deserialize_asm_struct_from_numba(fmt, obj, numba_obj)
        case _:
            return


def construct_from_numba(fmt: FType, numba_obj):
    """
    Construct an object from a Numba-compatible ftype.

    Args:
        fmt: The ftype of the object.
        numba_obj: The Numba-compatible object to construct from.

    Returns:
        An instance of the original object type.
    """
    match fmt:
        case NumbaArgumentFType():
            return fmt.construct_from_numba(numba_obj)
        case algebra.none_:
            return None
        case algebra.ftypes.FDTypeNumpy():
            return fmt(numba_obj)
        case algebra.int_ | algebra.bool_ | algebra.float_:
            return numba_obj
        case TupleFType() | ImmutableStructFType():
            return immutable_construct_from_numba(fmt, numba_obj)
        case StructFType():
            return struct_construct_from_numba(fmt, numba_obj)
        case _:
            return fmt(numba_obj)


def _serialize_asm_struct_to_numba(fmt: StructFType, obj) -> Any:
    args = [
        serialize_to_numba(fmt, getattr(obj, name)) for (name, fmt) in fmt.struct_fields
    ]
    return numba_type(fmt)(*args)


def _deserialize_asm_struct_from_numba(
    fmt: StructFType, obj, numba_struct: Any
) -> None:
    if fmt.is_mutable:
        for name in fmt.struct_fieldnames:
            setattr(obj, name, getattr(numba_struct, name))
        return


def struct_construct_from_numba(fmt: StructFType, numba_struct):
    args = [
        construct_from_numba(field_type, getattr(numba_struct, name))
        for (name, field_type) in fmt.struct_fields
    ]
    return fmt.from_fields(*args)


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
                return numba_getattr(obj_t, self, obj_code, attr.val)
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
                numba_setattr(obj_t, self, obj_code, attr.val, val_code)
                return None
            case asm.Call(asm.Literal(op), args):
                return numba_function_call(op, self, *args)

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


def struct_numba_getattr(fmt: StructFType, ctx, obj, attr):
    return f"{obj}.{attr}"


def numba_getattr(fmt: StructFType, ctx, obj, attr):
    match fmt:
        case _ if hasattr(fmt, "numba_getattr"):
            return fmt.numba_getattr(ctx, obj, attr)
        case TupleFType() | ImmutableStructFType():
            return immutable_struct_numba_getattr(fmt, ctx, obj, attr)
        case StructFType():
            return struct_numba_getattr(fmt, ctx, obj, attr)
        case _:
            raise NotImplementedError(f"No Numba getattr mapping for {fmt}")


def immutable_struct_numba_getattr(fmt: StructFType, ctx, obj, attr):
    index = list(fmt.struct_fieldnames).index(attr)
    return f"{obj}[{index}]"


def struct_numba_setattr(fmt: StructFType, ctx, obj, attr, val):
    ctx.exec(f"{ctx.feed}{obj}.{attr} = {val}")
    return


def numba_setattr(fmt: StructFType, ctx, obj, attr, val):
    match fmt:
        case _ if hasattr(fmt, "numba_setattr"):
            return fmt.numba_setattr(ctx, obj, attr, val)
        case MutableStructFType():
            return struct_numba_setattr(fmt, ctx, obj, attr, val)
        case _:
            raise NotImplementedError(f"No Numba setattr mapping for {fmt}")
