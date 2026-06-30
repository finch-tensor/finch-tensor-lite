from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from finchlite import algebra
from finchlite import finch_assembly as asm
from finchlite.algebra import FType, ffuncs
from finchlite.finch_assembly import BufferFType
from finchlite.symbolic import Context, ScopedDict


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


class MLIRArgumentFType(ABC):
    @abstractmethod
    def mlir_type(self): ...

    @abstractmethod
    def serialize_to_mlir(self, obj): ...

    @abstractmethod
    def deserialize_from_mlir(self, obj, mlir_buffer): ...

    @abstractmethod
    def construct_from_mlir(self, mlir_buffer): ...


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
        case _:
            raise NotImplementedError(f"No MLIR type mapping for {t}")


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


class MLIRContext(Context):
    def __init__(
        self,
        tab="  ",
        indent=1,
        bindings=None,
    ):
        if bindings is None:
            bindings = ScopedDict()

        super().__init__()
        self.tab = tab
        self.indent = indent
        self.bindings = bindings

    @property
    def feed(self) -> str:
        return self.tab * self.indent

    def new_ssa(self) -> str:
        return "%" + self.freshen("v")

    def emit(self):
        return "\n".join([*self.preamble, *self.epilogue])

    def emit_global(self):
        return f"module {{\n{self.emit()}\n}}\n"

    def block(self) -> "MLIRContext":
        blk = super().block()
        blk.tab = self.tab
        blk.indent = self.indent
        blk.bindings = self.bindings
        return blk

    def subblock(self):
        blk = self.block()
        blk.indent = self.indent + 1
        blk.bindings = self.bindings.scope()
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

            case asm.Load(buffer, index):
                buf_t = buffer.result_type
                if not isinstance(buf_t, MLIRBufferFType):
                    raise TypeError(f"Expected MLIR buffer type, got: {buf_t}")
                return buf_t.mlir_load(self, buffer, index)

            case asm.Store(buffer, index, value):
                buf_t = buffer.result_type
                if not isinstance(buf_t, MLIRBufferFType):
                    raise TypeError(f"Expected MLIR buffer type, got: {buf_t}")
                return buf_t.mlir_store(self, buffer, index, value)

            case asm.Block(bodies):
                ctx_2 = self.block()
                for body in bodies:
                    ctx_2(body)
                self.exec(ctx_2.emit())
                return None

            case asm.ForLoop(asm.Variable(var_n, var_t), start, end, body):
                lo, hi = self(start), self(end)
                step = self.new_ssa()
                self.exec(f"{feed}{step} = arith.constant 1 : index")
                iv = self.new_ssa()
                ctx_2 = self.subblock()
                ctx_2.bindings[var_n] = (iv, mlir_type(var_t))
                ctx_2(body)
                self.exec(
                    f"{feed}scf.for {iv} = {lo} to {hi} step {step} {{\n"
                    f"{ctx_2.emit()}\n"
                    f"{feed}}}"
                )
                return None

            # case asm.WhileLoop(cond, body):
            #     ...

            # case asm.BufferLoop(buf, var, body):
            #     ...

            # case asm.If(cond, body):
            #     ...

            # case asm.IfElse(cond, body, else_body):
            #     ...

            case asm.Function(asm.Variable(func_name, return_t), args, body):
                ctx_2 = self.subblock()
                statement = []
                for arg in args:
                    match arg:
                        case asm.Variable(name, t):
                            statement.append(f"%{name}: {mlir_type(t)}")
                            ctx_2.bindings[name] = (f"%{name}", mlir_type(t))
                        case _:
                            raise NotImplementedError(
                                f"Unrecognized argument type: {arg}"
                            )
                ctx_2(body)
                body_code = ctx_2.emit()
                feed = self.feed
                ret = "" if return_t == algebra.none_ else f" -> {mlir_type(return_t)}"

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
