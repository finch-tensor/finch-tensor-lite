from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from finchlite import algebra
from finchlite import finch_assembly as asm
from finchlite.algebra import FType, ffuncs
from finchlite.symbolic import Context, ScopedDict

from .mlir_scansearch import MLIR_HELPERS


class MLIROperator(ABC):
    @abstractmethod
    def mlir_name(self) -> str: ...

    @abstractmethod
    def mlir_function_call(self, ctx: Any, *args: Any) -> Any: ...


class MLIRNAryOperator(MLIROperator):
    def mlir_function_call(self, ctx: Any, *args: Any) -> Any:
        return mlir_nary_function_call(self.mlir_name(), ctx, *args)


class MLIRBinaryOperator(MLIROperator):
    def mlir_function_call(self, ctx: Any, *args: Any) -> Any:
        return mlir_binary_function_call(self.mlir_name(), ctx, *args)


class MLIRUnaryOperator(MLIROperator):
    def mlir_function_call(self, ctx: Any, *args: Any) -> Any:
        return mlir_unary_function_call(self.mlir_name(), ctx, *args)


def is_float(arg):
    return arg == algebra.float_ or (
        isinstance(arg, algebra.ftypes.FDTypeNumpy) and np.dtype(arg.dtype).kind == "f"
    )


def is_unsigned(arg):
    return (
        isinstance(arg, algebra.ftypes.FDTypeNumpy) and np.dtype(arg.dtype).kind == "u"
    )


def mlir_function_name(op, arg: FType) -> str:
    match op:
        case ffuncs.add:
            if is_float(arg):
                return "arith.addf"
            return "arith.addi"
        case ffuncs.sub:
            if is_float(arg):
                return "arith.subf"
            return "arith.subi"
        case ffuncs.mul:
            if is_float(arg):
                return "arith.mulf"
            return "arith.muli"
        case ffuncs.truediv:
            if is_float(arg):
                return "arith.divf"
            if is_unsigned(arg):
                return "arith.divui"
            return "arith.divsi"
        case ffuncs.floordiv:
            if is_float(arg):
                return "arith.divf"
            if is_unsigned(arg):
                return "arith.divui"
            return "arith.floordivsi"
        case ffuncs.mod:
            if is_float(arg):
                return "arith.remf"
            if is_unsigned(arg):
                return "arith.remui"
            return "arith.remsi"
        case ffuncs.min:
            if is_float(arg):
                return "arith.minimumf"
            if is_unsigned(arg):
                return "arith.minui"
            return "arith.minsi"
        case ffuncs.max:
            if is_float(arg):
                return "arith.maximumf"
            if is_unsigned(arg):
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
            if is_unsigned(arg):
                return "arith.shrui"
            return "arith.shrsi"
        case ffuncs.trunc:
            if is_float(arg):
                return "arith.truncf"
            return "arith.trunci"
        case ffuncs.eq:
            if is_float(arg):
                return "arith.cmpf oeq,"
            return "arith.cmpi eq,"
        case ffuncs.ne:
            if is_float(arg):
                return "arith.cmpf one,"
            return "arith.cmpi ne,"
        case ffuncs.lt:
            if is_float(arg):
                return "arith.cmpf olt,"
            if is_unsigned(arg):
                return "arith.cmpi ult,"
            return "arith.cmpi slt,"
        case ffuncs.le:
            if is_float(arg):
                return "arith.cmpf ole,"
            if is_unsigned(arg):
                return "arith.cmpi ule,"
            return "arith.cmpi sle,"
        case ffuncs.gt:
            if is_float(arg):
                return "arith.cmpf ogt,"
            if is_unsigned(arg):
                return "arith.cmpi ugt,"
            return "arith.cmpi sgt,"
        case ffuncs.ge:
            if is_float(arg):
                return "arith.cmpf oge,"
            if is_unsigned(arg):
                return "arith.cmpi uge,"
            return "arith.cmpi sge,"
        case ffuncs.scansearch:
            return "scansearch"
        case ffuncs.invert:
            return "-1"
        case ffuncs.not_:
            return "1"
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
    t = mlir_type(a.result_type)
    av, bv = ctx(a), ctx(b)
    res = ctx.new_ssa()
    ctx.exec(f"{ctx.feed}{res} = {mlir_name} {av}, {bv} : {t}")
    return res


def mlir_unary_function_call(mlir_name: str, ctx: Any, *args: Any) -> str:
    (a,) = args
    t = mlir_type(a.result_type)
    av = ctx(a)
    res = ctx.new_ssa()
    ctx.exec(f"{ctx.feed}{res} = {mlir_name} {av} : {t}")
    return res


def mlir_call_function_call(mlir_name: str, ctx: Any, ret_t: str, *args: Any) -> str:
    vs = [ctx(a) for a in args]
    ts = [mlir_type(a.result_type) for a in args]
    res = ctx.new_ssa()
    ctx.exec(
        f"{ctx.feed}{res} = func.call @{mlir_name}({', '.join(vs)}) "
        f": ({', '.join(ts)}) -> {ret_t}"
    )
    return res


def mlir_new_function_call(const: str, ctx: Any, *args: Any) -> str:
    (a,) = args
    t = mlir_type(a.result_type)
    av = ctx(a)
    c = ctx.new_ssa()
    ctx.exec(f"{ctx.feed}{c} = arith.constant {const} : {t}")
    res = ctx.new_ssa()
    ctx.exec(f"{ctx.feed}{res} = arith.xori {av}, {c} : {t}")
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
        case ffuncs.scansearch:
            return mlir_call_function_call(
                mlir_function_name(op, args[0].result_type),
                ctx,
                mlir_type(args[-1].result_type),
                *args,
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
        case algebra.int_:
            return "i64"
        case algebra.float_:
            return "f64"
        case algebra.ftypes.FDTypeNumpy():
            return numpy_to_mlir_types(t)
        case _:
            raise NotImplementedError(f"No MLIR type mapping for {t}")


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

    def new_ssa(self):
        n = self.val_counter[0]
        self.val_counter[0] = n + 1
        return f"%v{n}"

    def emit(self):
        return "\n".join([*self.preamble, *self.epilogue])

    def emit_global(self):
        body = "\n".join([*MLIR_HELPERS.values(), self.emit()])
        return f"module {{\n{body}\n}}\n"

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
            # case asm.Literal(value):
            #     ...

            # case asm.Variable(name, _):
            #     ...

            # case asm.Assign(asm.Variable(var_n, var_t) as var, val):
            #     ...

            case asm.Call(asm.Literal(op), args):
                return mlir_function_call(op, self, *args)

            # case asm.Load(buffer, index):
            #     ...

            # case asm.Store(buffer, index, value):
            #     ...

            # case asm.Block(bodies):
            #     ...

            # case asm.ForLoop(asm.Variable(var_n, var_t), start, end, body):
            #     ...

            # case asm.WhileLoop(cond, body):
            #     ...

            # case asm.BufferLoop(buf, var, body):
            #     ...

            # case asm.If(cond, body):
            #     ...

            # case asm.IfElse(cond, body, else_body):
            #     ...

            # case asm.Function(asm.Variable(func_name, return_t), args, body):
            #     ...

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
