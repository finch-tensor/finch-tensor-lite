from inspect import isbuiltin, isclass, isfunction
from typing import Any

from finch.codegen.numpy_buffer import NumpyBuffer, NumpyBufferFType

from ..symbolic import Context
from .nodes import (
    AssemblyNode,
    Assign,
    Block,
    Break,
    BufferLoop,
    Call,
    ForLoop,
    Function,
    GetAttr,
    If,
    IfElse,
    Length,
    Literal,
    Load,
    Module,
    Repack,
    Resize,
    Return,
    SetAttr,
    Slot,
    Store,
    Unpack,
    Variable,
    WhileLoop,
)


def _get_str(val: Any) -> str:
    if isbuiltin(val) or isclass(val) or isfunction(val):
        return str(val.__qualname__)
    if isinstance(val, NumpyBuffer):
        arr_str = str(val.arr).replace("\n", "")
        return f"np_buf({arr_str})"
    if isinstance(val, NumpyBufferFType):
        return f"ftype({_get_str(val._dtype)})"
    return str(val)


class PrinterCompiler:
    def __call__(self, prgm: Module):
        ctx = PrinterContext()
        ctx(prgm)
        return ctx.emit()


class PrinterContext(Context):
    def __init__(self, tab="    ", indent=0):
        super().__init__()
        self.tab = tab
        self.indent = indent

    @property
    def feed(self) -> str:
        return self.tab * self.indent

    def emit(self):
        return "\n".join([*self.preamble, *self.epilogue])

    def block(self) -> "PrinterContext":
        blk = super().block()
        blk.indent = self.indent
        blk.tab = self.tab
        return blk

    def subblock(self):
        blk = self.block()
        blk.indent = self.indent + 1
        return blk

    def __call__(self, prgm: AssemblyNode):
        feed = self.feed
        match prgm:
            case Literal(value):
                return _get_str(value)
            case Variable(name, _):
                return str(name)
            case Assign(Variable(var_n, var_t), val):
                self.exec(f"{feed}{var_n}: {_get_str(var_t)} = {self(val)}")
                return None
            case GetAttr(obj, attr):
                return f"getattr({obj}, {attr})"
            case SetAttr(obj, attr, val):
                return f"setattr({obj}, {attr})"
            case Call(Literal(_) as lit, args):
                return f"{self(lit)}({', '.join(self(arg) for arg in args)})"
            case Unpack(Slot(var_n, var_t), val):
                self.exec(f"{feed}{var_n}: {_get_str(var_t)} = unpack({self(val)})")
                return None
            case Repack(Slot(var_n, var_t)):
                self.exec(f"{feed}repack({var_n})")
                return None
            case Load(buf, idx):
                return f"load({self(buf)}, {self(idx)})"
            case Slot(name, type_):
                return f"slot({name}, {_get_str(type_)})"
            case Store(buf, idx, val):
                self.exec(f"{feed}store({self(buf)}, {self(idx)})")
                return None
            case Resize(buf, size):
                self.exec(f"{feed}resize({self(buf)}, {self(size)})")
                return None
            case Length(buf):
                return f"length({self(buf)})"
            case Block(bodies):
                ctx_2 = self.block()
                for body in bodies:
                    ctx_2(body)
                self.exec(ctx_2.emit())
                return None
            case ForLoop(var, start, end, body):
                var_2 = self(var)
                start = self(start)
                end = self(end)
                ctx_2 = self.subblock()
                ctx_2(body)
                body_code = ctx_2.emit()
                self.exec(f"{feed}for {var_2} in range({start}, {end}):\n{body_code}")
                return None
            case BufferLoop(buf, var, body):
                raise NotImplementedError
            case WhileLoop(cond, body):
                cond_code = self(cond)
                ctx_2 = self.subblock()
                ctx_2(body)
                body_code = ctx_2.emit()
                self.exec(f"{feed}while {cond_code}:\n{body_code}")
                return None
            case If(cond, body):
                cond_code = self(cond)
                ctx_2 = self.subblock()
                ctx_2(body)
                body_code = ctx_2.emit()
                self.exec(f"{feed}if {cond_code}:\n{body_code}")
                return None
            case IfElse(cond, body, else_body):
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
            case Function(Variable(func_name, return_t), args, body):
                ctx_2 = self.subblock()
                arg_decls = []
                for arg in args:
                    match arg:
                        case Variable(name, t):
                            arg_decls.append(f"{name}: {_get_str(t)}")
                        case _:
                            raise NotImplementedError(
                                f"Unrecognized argument type: {arg}"
                            )
                ctx_2(body)
                body_code = ctx_2.emit()
                feed = self.feed
                self.exec(
                    f"{feed}def {func_name}({', '.join(arg_decls)}) -> "
                    f"{_get_str(return_t)}:\n"
                    f"{body_code}\n"
                )
                return None
            case Return(value):
                self.exec(f"{feed}return {self(value)}")
                return None
            case Break():
                self.exec(f"{feed}break")
                return None
            case Module(funcs):
                for func in funcs:
                    if not isinstance(func, Function):
                        raise NotImplementedError(
                            f"Unrecognized function type: {type(func)}"
                        )
                    self(func)
                return None
            case _:
                raise NotImplementedError
