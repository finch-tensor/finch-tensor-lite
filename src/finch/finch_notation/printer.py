from inspect import isbuiltin, isclass, isfunction
from typing import Any

from ..symbolic import Context
from .nodes import (
    Access,
    Assign,
    Block,
    Call,
    Declare,
    Freeze,
    Function,
    If,
    IfElse,
    Increment,
    Literal,
    Loop,
    Module,
    NotationNode,
    Read,
    Return,
    Thaw,
    Unwrap,
    Update,
    Variable,
)


def _get_str(val: Any) -> str:
    if isbuiltin(val) or isclass(val) or isfunction(val):
        return str(val.__qualname__)
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

    def __call__(self, prgm: NotationNode):
        feed = self.feed
        match prgm:
            case Literal(value):
                return _get_str(value)
            case Variable(name, _):
                return str(name)
            case Call(f, args):
                return f"{self(f)}({', '.join(self(arg) for arg in args)})"
            case Unwrap(tns):
                return f"unwrap({self(tns)})"
            case Assign(Variable(var_n, var_t), val):
                self.exec(f"{feed}{var_n}: {_get_str(var_t)} = {self(val)}")
                return None
            case Access(tns, mode, idxs):
                tns_e = self(tns)
                idxs_e = [self(idx) for idx in idxs]
                match mode:
                    case Read():
                        return f"read({tns_e}, {idxs_e})"
                    case Update(op):
                        op_e = self(op)
                        return f"update({tns_e}, {idxs_e}, {op_e})"
                    case _:
                        raise NotImplementedError(f"Unrecognized access mode: {mode}")
                return None
            case Increment(tns, val):
                tns_e = self(tns)
                val_e = self(val)
                self.exec(f"{feed}increment({tns_e}, {val_e})")
                return None
            case Block(bodies):
                ctx_2 = self.block()
                for body in bodies:
                    ctx_2(body)
                self.exec(ctx_2.emit())
                return None
            case Loop(idx, ext, body):
                idx_e = self(idx)
                ext_e = self(ext)
                ctx_2 = self.subblock()
                ctx_2(body)
                body_code = ctx_2.emit()
                self.exec(f"{feed}loop({idx_e}, {ext_e}):\n{body_code}")
                return None
            case Declare(tns, init, op, shape):
                shape_e = [self(s) for s in shape]
                return f"declare({self(tns)}, {self(init)}, {self(op)}, {shape_e})"
            case Freeze(tns, op):
                return f"freeze({self(tns)}, {self(op)})"
            case Thaw(tns, op):
                return f"thaw({self(tns)}, {self(op)})"
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
            case Function(Variable(func_n, ret_t), args, body):
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
                    f"{feed}def {func_n}({', '.join(arg_decls)}) -> "
                    f"{_get_str(ret_t)}:\n"
                    f"{body_code}\n"
                )
                return None
            case Return(value):
                self.exec(f"{feed}return {self(value)}")
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
                raise NotImplementedError(
                    f"Unrecognized assembly node type: {type(prgm)}"
                )
