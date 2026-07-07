import ctypes
import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

import numpy as np

from mlir import ir
from mlir.execution_engine import ExecutionEngine
from mlir.passmanager import PassManager
from mlir.runtime import make_nd_memref_descriptor

from finchlite import algebra
from finchlite import finch_assembly as asm
from finchlite.algebra import (
    FType,
    StructFType,
    TupleFType,
    ffuncs,
    fisinstance,
)
from finchlite.finch_assembly import BufferFType
from finchlite.symbolic import Context, ScopedDict, UnvalidatedForm
from finchlite.util.logging import LOG_BACKEND_MLIR

from .stages import MLIRCode, MLIRLowerer

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_BACKEND_MLIR)


mlir_memrefs: dict[Any, Any] = {}
mlir_structs: dict[Any, Any] = {}
MLIR_PIPELINE = (
    "builtin.module("
    "expand-strided-metadata,"
    "convert-scf-to-cf,"
    "convert-cf-to-llvm,"
    "convert-arith-to-llvm,"
    "finalize-memref-to-llvm,"
    "convert-func-to-llvm,"
    "reconcile-unrealized-casts)"
)


class MLIROperator(ABC):
    @abstractmethod
    def mlir_name(self) -> str: ...

    @abstractmethod
    def mlir_function_call(self, ctx: Any, *args: Any) -> Any: ...

    @staticmethod
    def is_float(arg) -> bool:
        return algebra.isdtype(arg, "real floating")

    @staticmethod
    def is_unsigned(arg) -> bool:
        return algebra.isdtype(arg, "unsigned integer")


class MLIRNAryOperator(MLIROperator):
    def mlir_function_call(self, ctx: Any, *args: Any) -> Any:
        return mlir_nary_function_call(self.mlir_name(), ctx, *args)


class MLIRBinaryOperator(MLIROperator):
    def mlir_function_call(self, ctx: Any, *args: Any) -> Any:
        return mlir_binary_function_call(self.mlir_name(), ctx, *args)


def aliased_return_args(func):
    match func:
        case asm.Function(_, args, asm.Block(bodies)) if bodies:
            pass
        case _:
            return None
    params = []
    for a in args:
        match a:
            case asm.Variable(name, _):
                params.append(name)
            case _:
                return None
    defs = {}
    for stmt in bodies[:-1]:
        match stmt:
            case asm.Assign(asm.Variable(name, _), val):
                defs[name] = val

    def resolve(node):
        seen = set()
        while True:
            match node:
                case asm.Variable(name, _) if name in defs and name not in seen:
                    seen.add(name)
                    node = defs[name]
                case asm.Variable(name, _) if name in params:
                    return params.index(name)
                case _:
                    return None

    match bodies[-1]:
        case asm.Return(val):
            pass
        case _:
            return None
    seen = set()
    while True:
        match val:
            case asm.Variable(name, _) if name in defs and name not in seen:
                seen.add(name)
                val = defs[name]
            case _:
                break
    match val:
        case asm.Call(asm.Literal(op), tuple_args) if op == ffuncs.make_tuple:
            idxs = [resolve(a) for a in tuple_args]
            if all(i is not None for i in idxs):
                return tuple(idxs)
    return None


class MLIRKernel(asm.AssemblyKernel):
    def __init__(self, engine, func_name, ret_type, argtypes, ret_alias=None):
        self.engine = engine
        self.func_name = func_name
        self.ret_type = ret_type
        self.argtypes = argtypes
        self.ret_alias = ret_alias

    def __call__(self, *args):
        if len(args) != len(self.argtypes):
            raise ValueError(
                f"Expected {len(self.argtypes)} arguments, got {len(args)}"
            )
        for argtype, arg in zip(self.argtypes, args, strict=False):
            if not fisinstance(arg, argtype):
                raise TypeError(f"Expected argument of type {argtype}, got {type(arg)}")
        serial_args = list(map(serialize_to_mlir, self.argtypes, args))

        packed = []
        for t, sa in zip(self.argtypes, serial_args, strict=False):
            if isinstance(t, BufferFType):
                packed.append(ctypes.pointer(ctypes.pointer(sa)))
            else:
                packed.append(ctypes.pointer(sa))

        if self.ret_type == algebra.none_ or self.ret_alias is not None:
            res = None
            self.engine.invoke(self.func_name, *packed)
        elif isinstance(self.ret_type, StructFType):
            res = mlir_field_ctype(self.ret_type)()
            self.engine.invoke(
                self.func_name, *packed, ctypes.pointer(ctypes.pointer(res))
            )
        else:
            res = mlir_field_ctype(self.ret_type)()
            self.engine.invoke(self.func_name, *packed, ctypes.pointer(res))

        for arg_type, arg, serial_arg in zip(
            self.argtypes, args, serial_args, strict=False
        ):
            deserialize_from_mlir(arg_type, arg, serial_arg)

        if self.ret_alias is not None:
            return tuple(args[i] for i in self.ret_alias)
        if self.ret_type is type(None):
            return None
        return construct_from_mlir(self.ret_type, res)


class MLIRLibrary(asm.AssemblyLibrary):
    def __init__(self, mlir_module, kernels):
        self.mlir_module = mlir_module
        self.kernels = kernels

    def __getattr__(self, name):
        if name in self.kernels:
            return self.kernels[name]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


@lru_cache(maxsize=10_000)
def load_mlir_engine(mlir_code: str):
    context = ir.Context()
    with context:
        module = ir.Module.parse(mlir_code)
        pm = PassManager.parse(MLIR_PIPELINE)
        try:
            pm.run(module.operation)
        except Exception as e:
            logger.error(
                f"MLIR pass pipeline failed on the following code:\n{mlir_code}"
                f"\nError message: {e}"
            )
            raise
        engine = ExecutionEngine(module)
    return context, module, engine


class MLIRCompiler(UnvalidatedForm, asm.AssemblyLoader):
    def __init__(self, ctx: MLIRLowerer | None = None):
        self.ctx: MLIRLowerer = MLIRGenerator() if ctx is None else ctx

    def lower(self, prgm: asm.Module) -> MLIRLibrary:
        if prgm.head() != asm.Module:
            raise ValueError(
                "MLIRCompiler expects a Module as the head of the program, "
                f"got {type(prgm.head())}"
            )
        mlir_code = self.ctx(prgm).code
        logger.debug(f"Compiling MLIR code:\n{mlir_code}")
        context, module, engine = load_mlir_engine(mlir_code)
        kernels = {}
        for func in prgm.funcs:
            match func:
                case asm.Function(asm.Variable(func_name, return_t), args, _):
                    arg_ts = [arg.result_type for arg in args]
                    kernels[func_name] = MLIRKernel(
                        engine,
                        func_name,
                        return_t,
                        arg_ts,
                        ret_alias=aliased_return_args(func),
                    )
                case _:
                    raise NotImplementedError(
                        f"Unrecognized function type: {type(func)}"
                    )
        return MLIRLibrary((context, module, engine), kernels)


def mlir_function_name(op, arg: FType) -> str:
    match op:
        case ffuncs.add:
            if MLIROperator.is_float(arg):
                return "arith.addf"
            return "arith.addi"
        case ffuncs.sub:
            if MLIROperator.is_float(arg):
                return "arith.subf"
            return "arith.subi"
        case ffuncs.mul:
            if MLIROperator.is_float(arg):
                return "arith.mulf"
            return "arith.muli"
        case ffuncs.truediv:
            if MLIROperator.is_float(arg):
                return "arith.divf"
            if MLIROperator.is_unsigned(arg):
                return "arith.divui"
            return "arith.divsi"
        case ffuncs.floordiv:
            if MLIROperator.is_float(arg):
                return "arith.divf"
            if MLIROperator.is_unsigned(arg):
                return "arith.divui"
            return "arith.floordivsi"
        case ffuncs.mod:
            if MLIROperator.is_float(arg):
                return "arith.remf"
            if MLIROperator.is_unsigned(arg):
                return "arith.remui"
            return "arith.remsi"
        case ffuncs.min:
            if MLIROperator.is_float(arg):
                return "arith.minimumf"
            if MLIROperator.is_unsigned(arg):
                return "arith.minui"
            return "arith.minsi"
        case ffuncs.max:
            if MLIROperator.is_float(arg):
                return "arith.maximumf"
            if MLIROperator.is_unsigned(arg):
                return "arith.maxui"
            return "arith.maxsi"
        case ffuncs.and_:
            return "arith.andi"
        case ffuncs.or_:
            return "arith.ori"
        case ffuncs.xor:
            return "arith.xori"
        case ffuncs.lshift:
            return "arith.shli"
        case ffuncs.rshift:
            if MLIROperator.is_unsigned(arg):
                return "arith.shrui"
            return "arith.shrsi"
        case ffuncs.eq:
            if MLIROperator.is_float(arg):
                return "arith.cmpf oeq,"
            return "arith.cmpi eq,"
        case ffuncs.ne:
            if MLIROperator.is_float(arg):
                return "arith.cmpf one,"
            return "arith.cmpi ne,"
        case ffuncs.lt:
            if MLIROperator.is_float(arg):
                return "arith.cmpf olt,"
            if MLIROperator.is_unsigned(arg):
                return "arith.cmpi ult,"
            return "arith.cmpi slt,"
        case ffuncs.le:
            if MLIROperator.is_float(arg):
                return "arith.cmpf ole,"
            if MLIROperator.is_unsigned(arg):
                return "arith.cmpi ule,"
            return "arith.cmpi sle,"
        case ffuncs.gt:
            if MLIROperator.is_float(arg):
                return "arith.cmpf ogt,"
            if MLIROperator.is_unsigned(arg):
                return "arith.cmpi ugt,"
            return "arith.cmpi sgt,"
        case ffuncs.ge:
            if MLIROperator.is_float(arg):
                return "arith.cmpf oge,"
            if MLIROperator.is_unsigned(arg):
                return "arith.cmpi uge,"
            return "arith.cmpi sge,"
        case ffuncs.invert:
            return "arith.xori -1"
        case ffuncs.not_:
            return "arith.xori 1"
        case MLIROperator():
            return op.mlir_name()
        case _:
            raise NotImplementedError(f"{op} has no MLIR representation.")


def mlir_nary_function_call(mlir_name: str, ctx: Any, *args: Any) -> str:
    t = mlir_type(args[0].result_type)
    acc = ctx(args[0])
    for a in args[1:]:
        rhs = ctx(a)
        res = ctx.new_ssa()
        ctx.exec(f"{ctx.feed}{res} = {mlir_name} {acc}, {rhs} : {t}")
        acc = res
    return acc


def mlir_binary_function_call(mlir_name: str, ctx: Any, *args: Any) -> str:
    a, b = args
    av, bv = ctx(a), ctx(b)
    res = ctx.new_ssa()
    ctx.exec(f"{ctx.feed}{res} = {mlir_name} {av}, {bv} : {mlir_type(a.result_type)}")
    return res


def mlir_new_function_call(mlir_name: str, ctx: Any, *args: Any) -> str:
    name, const = mlir_name.split()
    (a,) = args
    av = ctx(a)
    c = ctx.new_ssa()
    ctx.exec(f"{ctx.feed}{c} = arith.constant {const} : {mlir_type(a.result_type)}")
    res = ctx.new_ssa()
    ctx.exec(f"{ctx.feed}{res} = {name} {av}, {c} : {mlir_type(a.result_type)}")
    return res


def mlir_function_call(op, ctx, *args: Any) -> str:
    match op:
        case MLIROperator():
            return op.mlir_function_call(ctx, *args)

    match op:
        case ffuncs._InitWrite():
            return ctx(args[1])
        case ffuncs.make_tuple:
            t = TupleFType.from_tuple(tuple(a.result_type for a in args))
            st = mlir_struct_type(t)
            acc = ctx.new_ssa()
            ctx.exec(f"{ctx.feed}{acc} = llvm.mlir.undef : {st}")
            for k, a in enumerate(args):
                v = ctx(a)
                nxt = ctx.new_ssa()
                ctx.exec(f"{ctx.feed}{nxt} = llvm.insertvalue {v}, {acc}[{k}] : {st}")
                acc = nxt
            return acc
        case ffuncs.add | ffuncs.mul | ffuncs.and_ | ffuncs.xor | ffuncs.or_:
            return mlir_nary_function_call(
                mlir_function_name(op, args[0].result_type), ctx, *args
            )
        case (
            ffuncs.sub
            | ffuncs.truediv
            | ffuncs.floordiv
            | ffuncs.mod
            | ffuncs.lshift
            | ffuncs.rshift
            | ffuncs.min
            | ffuncs.max
            | ffuncs.eq
            | ffuncs.ne
            | ffuncs.gt
            | ffuncs.lt
            | ffuncs.ge
            | ffuncs.le
        ):
            return mlir_binary_function_call(
                mlir_function_name(op, args[0].result_type), ctx, *args
            )
        case ffuncs.not_ | ffuncs.invert:
            return mlir_new_function_call(
                mlir_function_name(op, args[0].result_type), ctx, *args
            )
        case _:
            raise NotImplementedError(f"{op} has no MLIR representation.")


def mlir_identifier(name: str) -> str:
    """
    Sanitize a finch variable name (which may contain characters like '#')
    into a valid MLIR identifier.
    """
    return "".join(c if c.isalnum() or c in "_.$-" else "_" for c in name)


def mlir_getattr(fmt: FType, ctx, obj, attrs):
    """
    Emit a single llvm.extractvalue for the chain of field accesses `attrs`
    rooted at the struct value `obj`, converting the result back from its
    lowered form. Results are memoized per scope, so repeated accesses reuse
    the first extraction.
    """
    idxs = []
    t = fmt
    for a in attrs:
        if not isinstance(t, StructFType):
            raise TypeError(f"Expected struct type, got: {t}")
        if not t.struct_hasattr(a):
            raise ValueError(f"trying to get missing attr {a!r}")
        idxs.append(t.struct_fieldnames.index(a))
        t = t.struct_attrtype(a)
    key = f".getattr:{obj}[{','.join(map(str, idxs))}]"
    if key in ctx.bindings:
        return ctx.bindings[key][0]
    res = ctx.new_ssa()
    ctx.exec(
        f"{ctx.feed}{res} = llvm.extractvalue {obj}"
        f"[{', '.join(map(str, idxs))}] : {mlir_struct_type(fmt)}"
    )
    # Struct fields are stored in their LLVM-lowered form; convert back to
    # the type the rest of the program expects.
    s = mlir_type(t)
    lowered = llvm_lowered_type(s)
    if lowered != s:
        cast = ctx.new_ssa()
        if s == "index":
            ctx.exec(f"{ctx.feed}{cast} = arith.index_cast {res} : i64 to index")
        else:
            ctx.exec(
                f"{ctx.feed}{cast} = builtin.unrealized_conversion_cast {res} "
                f": {lowered} to {s}"
            )
        res = cast
    ctx.bindings[key] = (res, s)
    return res


def llvm_lowered_type(s: str) -> str:
    """
    Return the LLVM-dialect spelling of an MLIR type, i.e. how a value of
    that type is stored as a field inside an !llvm.struct.
    """
    if s.startswith("memref<") and s.endswith(">"):
        *dims, _ = s[len("memref<") : -1].split("x")
        return (
            f"!llvm.struct<(ptr, ptr, i64, "
            f"array<{len(dims)} x i64>, array<{len(dims)} x i64>)>"
        )
    if s == "index":
        return "i64"
    return s


def mlir_struct_type(fmt: StructFType) -> str:
    fields = []
    for _, t in fmt.struct_fields:
        match t:
            case StructFType():
                fields.append(mlir_struct_type(t))
            case _:
                fields.append(llvm_lowered_type(mlir_type(t)))
    return f"!llvm.struct<({', '.join(fields)})>"


class MLIRGenerator(UnvalidatedForm, MLIRLowerer):
    def lower(self, prgm: asm.AssemblyNode) -> MLIRCode:
        ctx = MLIRContext()
        ctx(prgm)
        return MLIRCode(ctx.emit_global())


class MLIRArgumentFType(ABC):
    @abstractmethod
    def mlir_type(self): ...

    @abstractmethod
    def serialize_to_mlir(self, obj): ...

    @abstractmethod
    def deserialize_from_mlir(self, obj, mlir_buffer): ...

    @abstractmethod
    def construct_from_mlir(self, mlir_buffer): ...


def mlir_type(t: FType):
    match t:
        case MLIRArgumentFType():
            return t.mlir_type()
        case algebra.bool_:
            return "i1"
        case algebra.intp:
            return "index"
        case algebra.int_:
            return "i64"
        case algebra.float_:
            return "f64"
        case algebra.ftypes.FDTypeNumpy():
            return numpy_to_mlir_types(t)
        case StructFType():
            return mlir_struct_type(t)
        case _:
            raise NotImplementedError(f"No MLIR type mapping for {t}")


def numpy_to_mlir_types(t):
    dt = np.dtype(t.dtype)
    if dt.kind == "b":
        return "i1"

    bits = dt.itemsize * 8
    if dt.kind in ("i", "u"):
        return f"i{bits}"
    if dt.kind == "f":
        return f"f{bits}"

    raise NotImplementedError(f"No MLIR type for numpy dtype {dt}")


def mlir_ctype(s: str):
    if s.startswith("memref<") and s.endswith(">"):
        res = mlir_memrefs.get(s)
        if res is None:
            *dims, elem = s[len("memref<") : -1].split("x")
            res = make_nd_memref_descriptor(len(dims), mlir_ctype(elem))
            mlir_memrefs[s] = res
        return res

    match s:
        case "i1":
            return ctypes.c_bool
        case "i8":
            return ctypes.c_int8
        case "i16":
            return ctypes.c_int16
        case "i32":
            return ctypes.c_int32
        case "index" | "i64":
            return ctypes.c_int64
        case "f32":
            return ctypes.c_float
        case "f64":
            return ctypes.c_double
        case _:
            raise NotImplementedError(f"No ctypes mapping for MLIR type {s}")


def mlir_struct_ctype(fmt: StructFType):
    res = mlir_structs.get(fmt)
    if res is None:
        fields = [(name, mlir_field_ctype(t)) for name, t in fmt.struct_fields]
        res = type("MLIR" + fmt.struct_name, (ctypes.Structure,), {"_fields_": fields})
        mlir_structs[fmt] = res
    return res


def mlir_field_ctype(t: FType):
    match t:
        case StructFType():
            return mlir_struct_ctype(t)
        case _:
            return mlir_ctype(mlir_type(t))


def serialize_to_mlir(fmt: FType, obj):
    match fmt:
        case MLIRArgumentFType():
            return fmt.serialize_to_mlir(obj)
        case algebra.ftypes.FDTypeNumpy():
            return np.ctypeslib.as_ctypes(np.array(obj, dtype=fmt.dtype))
        case algebra.int_ | algebra.float_ | algebra.bool_:
            return mlir_ctype(mlir_type(fmt))(obj)
        case algebra.none_:
            return None
        case StructFType():
            return serialize_struct_to_mlir(fmt, obj)
        case _:
            raise NotImplementedError(f"No MLIR serialization mapping for {fmt}")


def serialize_struct_to_mlir(fmt: StructFType, obj):
    args = [
        serialize_to_mlir(t, fmt.struct_getattr(obj, name))
        for name, t in fmt.struct_fields
    ]
    return mlir_struct_ctype(fmt)(*args)


def deserialize_from_mlir(fmt: FType, obj, c_obj):
    match fmt:
        case MLIRArgumentFType():
            fmt.deserialize_from_mlir(obj, c_obj)
        case StructFType():
            deserialize_struct_from_mlir(fmt, obj, c_obj)
        case _:
            return


def deserialize_struct_from_mlir(fmt: StructFType, obj, c_obj):
    if not fmt.is_mutable:
        return
    for name, t in fmt.struct_fields:
        fmt.struct_setattr(obj, name, construct_from_mlir(t, getattr(c_obj, name)))


def construct_from_mlir(fmt: FType, mlir_obj):
    match fmt:
        case MLIRArgumentFType():
            return fmt.construct_from_mlir(mlir_obj)
        case algebra.ftypes.FDTypeNumpy():
            return fmt(mlir_obj.value if hasattr(mlir_obj, "value") else mlir_obj)
        case algebra.int_:
            return int(mlir_obj.value if hasattr(mlir_obj, "value") else mlir_obj)
        case algebra.float_:
            return float(mlir_obj.value if hasattr(mlir_obj, "value") else mlir_obj)
        case algebra.bool_:
            return bool(mlir_obj.value if hasattr(mlir_obj, "value") else mlir_obj)
        case algebra.none_:
            return None
        case StructFType():
            return construct_struct_from_mlir(fmt, mlir_obj)
        case _:
            return fmt(mlir_obj)


def construct_struct_from_mlir(fmt: StructFType, mlir_obj):
    return fmt.from_fields(
        *(
            construct_from_mlir(t, getattr(mlir_obj, name))
            for name, t in fmt.struct_fields
        )
    )


class MLIRBufferFType(BufferFType, MLIRArgumentFType, ABC):
    """
    Abstract base class for the ftype of datastructures. The ftype defines how
    the data in an Buffer is organized and accessed.
    """

    @abstractmethod
    def mlir_length(self, ctx: "MLIRContext", buffer):
        """
        Return MLIR code which loads a named buffer at the given index.
        """
        ...

    @abstractmethod
    def mlir_load(self, ctx: "MLIRContext", buffer, index):
        """
        Return MLIR code which loads a named buffer at the given index.
        """
        ...

    @abstractmethod
    def mlir_store(self, ctx: "MLIRContext", buffer, index, value=None):
        """
        Return MLIR code which stores a named buffer to the given index.
        """
        ...

    @abstractmethod
    def mlir_resize(self, ctx: "MLIRContext", buffer, new_length):
        """
        Return MLIR code which resizes a named buffer to the given length.
        """
        ...


class MLIRStackFType(ABC):
    """
    Abstract base class for symbolic formats in MLIR. Stack formats must also
    support other functions with symbolic inputs in addition to variable ones.
    """

    @abstractmethod
    def mlir_unpack(self, ctx, lhs, rhs):
        """
        Convert a value to a symbolic representation in MLIR. Returns a NamedTuple
        of unpacked variable names, etc. The `lhs` is the variable namespace to
        assign to.
        """
        ...

    @abstractmethod
    def mlir_repack(self, ctx, lhs, rhs):
        """
        Update an object based on a symbolic representation. The `rhs` is the
        symbolic representation to update from, and `lhs` is a variable name referring
        to the original object to update.
        """
        ...


class MLIRContext(Context):
    def __init__(
        self,
        tab="  ",
        indent=1,
        bindings=None,
        slots=None,
    ):
        if bindings is None:
            bindings = ScopedDict()
        if slots is None:
            slots = ScopedDict()

        super().__init__()
        self.tab = tab
        self.indent = indent
        self.bindings = bindings
        self.slots = slots

    @property
    def feed(self) -> str:
        return self.tab * self.indent

    def new_ssa(self) -> str:
        return "%" + self.freshen("v")

    def resolve(self, node):
        match node:
            case asm.Slot(var_n, _):
                if var_n in self.slots:
                    return self.slots[var_n]
                raise KeyError(f"Slot {var_n} not found in context")
            case _:
                raise ValueError(f"Expected Slot, got: {type(node)}")

    def emit(self):
        return "\n".join([*self.preamble, *self.epilogue])

    def emit_global(self):
        return f"module {{\n{self.emit()}\n}}\n"

    def block(self) -> "MLIRContext":
        blk = super().block()
        blk.tab = self.tab
        blk.indent = self.indent
        blk.bindings = self.bindings
        blk.slots = self.slots
        return blk

    def subblock(self):
        blk = self.block()
        blk.indent = self.indent + 1
        blk.bindings = self.bindings.scope()
        blk.slots = self.slots.scope()
        return blk

    def __call__(self, prgm: asm.AssemblyNode):
        feed = self.feed
        match prgm:
            case asm.Literal(value):
                t = mlir_type(prgm.result_type)
                new = (
                    float(value)
                    if MLIROperator.is_float(prgm.result_type)
                    else int(value)
                )
                s = self.new_ssa()
                self.exec(f"{self.feed}{s} = arith.constant {new} : {t}")
                return s

            case asm.Variable(name, _):
                if name not in self.bindings:
                    raise ValueError(f"Variable does not exist: {name!r}")
                return self.bindings[name][0]

            case asm.Assign(asm.Variable(var_n, var_t), val):
                v = self(val)
                self.bindings[var_n] = (v, mlir_type(var_t))
                return None

            case asm.Call(asm.Literal(op), args):
                return mlir_function_call(op, self, *args)

            case asm.Length(buffer):
                buf_t = buffer.result_type
                if not isinstance(buf_t, MLIRBufferFType):
                    raise TypeError(f"Expected MLIR buffer type, got: {buf_t}")
                return buf_t.mlir_length(self, self.resolve(buffer))

            case asm.Load(buffer, index):
                buf_t = buffer.result_type
                if not isinstance(buf_t, MLIRBufferFType):
                    raise TypeError(f"Expected MLIR buffer type, got: {buf_t}")
                return buf_t.mlir_load(self, self.resolve(buffer), index)

            case asm.Store(buffer, index, value):
                buf_t = buffer.result_type
                if not isinstance(buf_t, MLIRBufferFType):
                    raise TypeError(f"Expected MLIR buffer type, got: {buf_t}")
                return buf_t.mlir_store(self, self.resolve(buffer), index, value)

            case asm.Resize(buffer, size):
                buf_t = buffer.result_type
                if not isinstance(buf_t, MLIRBufferFType):
                    raise TypeError(f"Expected MLIR buffer type, got: {buf_t}")
                return buf_t.mlir_resize(self, self.resolve(buffer), size)

            case asm.GetAttr(obj, attr):
                # Fold chained accesses (a.lvl.lvl.val) into one extraction
                # path rooted at the innermost non-GetAttr object.
                attrs = [attr.val]
                base = obj
                while True:
                    match base:
                        case asm.GetAttr(inner, a):
                            attrs.append(a.val)
                            base = inner
                        case _:
                            break
                attrs.reverse()
                obj_t = base.result_type
                if not isinstance(obj_t, StructFType):
                    raise TypeError(f"Expected struct type, got: {obj_t}")
                return mlir_getattr(obj_t, self, self(base), attrs)

            case asm.Unpack(asm.Slot(var_n, var_t), val):
                if val.result_type != var_t:
                    raise TypeError(f"Type mismatch: {val.result_type} != {var_t}")
                self.slots[var_n] = var_t.mlir_unpack(self, var_n, val)
                return None

            case asm.Repack(asm.Slot(var_n, var_t)):
                obj = self.slots[var_n]
                var_t.mlir_repack(self, var_n, obj)
                return None

            case asm.Block(bodies):
                ctx_2 = self.block()
                for body in bodies:
                    ctx_2(body)
                self.exec(ctx_2.emit())
                return None

            case asm.ForLoop(asm.Variable(var_n, var_t), start, end, body):
                lo, hi = self(start), self(end)
                for bound_t, name in (
                    (start.result_type, "lo"),
                    (end.result_type, "hi"),
                ):
                    if mlir_type(bound_t) != "index":
                        cast = self.new_ssa()
                        old = lo if name == "lo" else hi
                        self.exec(
                            f"{feed}{cast} = arith.index_cast {old} "
                            f": {mlir_type(bound_t)} to index"
                        )
                        if name == "lo":
                            lo = cast
                        else:
                            hi = cast
                step = self.new_ssa()
                self.exec(f"{feed}{step} = arith.constant 1 : index")
                iv = self.new_ssa()
                ctx_2 = self.subblock()
                if mlir_type(var_t) == "index":
                    ctx_2.bindings[var_n] = (iv, "index")
                else:
                    iv_cast = ctx_2.new_ssa()
                    ctx_2.exec(
                        f"{ctx_2.feed}{iv_cast} = arith.index_cast {iv} "
                        f": index to {mlir_type(var_t)}"
                    )
                    ctx_2.bindings[var_n] = (iv_cast, mlir_type(var_t))
                ctx_2(body)
                self.exec(
                    f"{feed}scf.for {iv} = {lo} to {hi} step {step} {{\n"
                    f"{ctx_2.emit()}\n"
                    f"{feed}}}"
                )
                return None

            case asm.Function(asm.Variable(func_name, return_t), args, body):
                ctx_2 = self.subblock()
                statement = []
                for arg in args:
                    match arg:
                        case asm.Variable(name, t):
                            ssa = "%" + mlir_identifier(name)
                            statement.append(f"{ssa}: {mlir_type(t)}")
                            ctx_2.bindings[name] = (ssa, mlir_type(t))
                        case _:
                            raise NotImplementedError(
                                f"Unrecognized argument type: {arg}"
                            )
                if isinstance(return_t, StructFType):
                    if aliased_return_args(prgm) is not None:
                        ctx_2.bindings[".ret_alias"] = ("", "")
                        ret = ""
                    else:
                        statement.append("%_ret: !llvm.ptr")
                        ctx_2.bindings[".ret_ptr"] = ("%_ret", "!llvm.ptr")
                        ret = ""
                else:
                    ret = (
                        ""
                        if return_t == algebra.none_
                        else f" -> {mlir_type(return_t)}"
                    )
                ctx_2(body)
                body_code = ctx_2.emit()
                feed = self.feed

                self.exec(
                    f"{feed}func.func @{func_name}({', '.join(statement)}){ret} "
                    f"attributes {{llvm.emit_c_interface}} {{\n"
                    f"{body_code}\n"
                    f"{feed}}}"
                )
                return None

            case asm.Return(value):
                if value.result_type == algebra.none_:
                    self.exec(f"{feed}func.return")
                elif isinstance(value.result_type, StructFType):
                    if ".ret_alias" in self.bindings:
                        self.exec(f"{feed}func.return")
                    else:
                        v = self(value)
                        ptr = self.bindings[".ret_ptr"][0]
                        self.exec(
                            f"{feed}llvm.store {v}, {ptr} "
                            f": {mlir_type(value.result_type)}, !llvm.ptr"
                        )
                        self.exec(f"{feed}func.return")
                else:
                    v = self(value)
                    self.exec(f"{feed}func.return {v} : {mlir_type(value.result_type)}")
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
