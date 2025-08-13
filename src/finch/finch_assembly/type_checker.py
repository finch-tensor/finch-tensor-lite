from dataclasses import dataclass

import numpy as np

from .. import algebra
from ..symbolic import FType, ScopedDict, ftype
from . import nodes as asm
from .buffer import BufferFType
from .struct import AssemblyStructFType


class AssemblyTypeError(Exception):
    pass


@dataclass(eq=True)
class FunctionState:
    def __init__(self, return_type=None):
        self.return_type = return_type


@dataclass(eq=True)
class LoopState:
    def __init__(self):
        pass


@dataclass(eq=True)
class StmtReturnType:
    def __init__(self, return_type=None):
        self.return_type = return_type


class AssemblyTypeChecker:
    """
    A type checker for FinchAssembly
    """

    def __init__(
        self,
        ctxt=None,
        loop_state=None,
        function_state=None,
    ):
        if ctxt is None:
            ctxt = ScopedDict()
        self.ctxt = ctxt
        self.loop_state = loop_state
        self.function_state = function_state

    def scope(
        self,
        ctxt=None,
        loop_state=None,
        function_state=None,
    ):
        if ctxt is None:
            ctxt = self.ctxt.scope()
        if loop_state is None:
            loop_state = self.loop_state
        if function_state is None:
            function_state = self.function_state
        return AssemblyTypeChecker(
            ctxt=ctxt,
            loop_state=loop_state,
            function_state=function_state,
        )

    def check_in_loop(self):
        if not self.loop_state:
            raise AssemblyTypeError("not in loop")

    def check_return_type(self, return_type):
        if self.function_state:
            check_type_match(
                self.function_state.return_type,
                return_type,
            )
        else:
            raise AssemblyTypeError("Cannot return outside of function")

    def check_returns(self, return_type):
        if not isinstance(return_type, StmtReturnType):
            raise AssemblyTypeError("Missing return statement.")

    def check_in_ctxt(self, var_n, var_t):
        try:
            check_type_match(self.ctxt[var_n], var_t)
        except KeyError:
            raise AssemblyTypeError(
                f"'{var_n}' is not defined in the current context."
            ) from KeyError

    def check_buffer(self, buffer):
        buffer_type = self(buffer)
        if isinstance(buffer_type, BufferFType):
            return buffer_type
        raise AssemblyTypeError("Expected buffer.")

    def check_struct(self, struct):
        struct_type = self(struct)
        if isinstance(struct_type, AssemblyStructFType):
            return struct_type
        raise AssemblyTypeError("Expected struct.")

    def check_cond(self, cond):
        cond_type = self(cond)
        if (
            not isinstance(cond_type, type)
            or not np.issubdtype(cond_type, np.number)
            and not np.issubdtype(cond_type, np.bool_)
        ):
            raise AssemblyTypeError("Expected number or boolean for conditional")

    def __call__(self, prgm: asm.AssemblyNode):
        match prgm:
            case asm.Literal(value):
                return ftype(value)
            case asm.Variable(var_n, var_t):
                self.check_in_ctxt(var_n, var_t)
                return var_t
            case asm.Slot(var_n, var_t):
                self.check_in_ctxt(var_n, var_t)
                return var_t
            case asm.Stack(obj, obj_t):
                check_is_type(obj_t)
                return obj_t
            case asm.Assign(asm.Variable(var_n, var_t), rhs):
                rhs_type = self(rhs)  # rhs must be an expression
                check_type_match(var_t, rhs_type)
                if var_n in self.ctxt:
                    check_type_match(self.ctxt[var_n], var_t)
                else:
                    self.ctxt[var_n] = var_t
                return None
            case asm.Assign(asm.Stack(_obj, _obj_t), rhs):
                return None  # TODO
            case asm.GetAttr(obj, asm.Literal(attr)):
                obj_type = self.check_struct(obj)
                return check_attrtype(obj_type, attr)
            case asm.SetAttr(obj, asm.Literal(attr), value):
                obj_type = self.check_struct(obj)
                attrtype = check_attrtype(obj_type, attr)
                value_type = self(value)  # value must be an expression
                check_type_match(attrtype, value_type)
                return None
            case asm.Call(asm.Literal(op), args):
                arg_types = [self(arg) for arg in args]
                try:
                    # TODO: need to check definedness
                    return algebra.return_type(op, *arg_types)
                except (AttributeError, TypeError):
                    raise AssemblyTypeError(
                        "return type of function is not registered"
                    ) from AttributeError
                    raise AssemblyTypeError(
                        "operation on defined on given types"
                    ) from TypeError
            case asm.Load(buffer, index):
                buffer_type = self.check_buffer(buffer)
                index_type = self(index)
                check_type_match(buffer_type.length_type, index_type)
                return buffer_type.element_type
            case asm.Store(buffer, index, value):
                buffer_type = self.check_buffer(buffer)
                index_type = self(index)
                check_type_match(buffer_type.length_type, index_type)
                value_type = self(value)
                check_type_match(buffer_type.element_type, value_type)
                return None
            case asm.Resize(buffer, new_size):
                buffer_type = self.check_buffer(buffer)
                new_size_type = self(new_size)
                check_type_match(buffer_type.length_type, new_size_type)
                return None
            case asm.Length(buffer):
                buffer_type = self.check_buffer(buffer)
                return buffer_type.length_type
            case asm.ForLoop(asm.Variable(var_n, var_t), start, end, body):
                start_type = self(start)
                end_type = self(end)
                check_is_index_type(var_t)
                check_type_match(var_t, start_type)
                check_type_match(var_t, end_type)
                loop = self.scope(loop_state=LoopState())
                loop.ctxt[var_n] = var_t
                return None
            case asm.BufferLoop(buffer, asm.Variable(var_n, var_t), body):
                buffer_type = self.check_buffer(buffer)
                check_type_match(buffer_type.element_type, var_t)
                loop = self.scope(loop_state=LoopState())
                loop.ctxt[var_n] = var_t
                return None
            case asm.WhileLoop(cond, body):
                self.check_cond(cond)
                self(body)
                return None
            case asm.If(cond, body):
                self.check_cond(cond)
                self(body)
                return None
            case asm.IfElse(cond, body, else_body):
                self.check_cond(cond)
                body_type = self(body)
                else_body_type = self(else_body)
                if body_type is None or else_body_type is None:
                    return None
                check_type_match(body_type, else_body_type)
                return StmtReturnType(body_type)
            case asm.Function(asm.Variable(_, return_type), args, body):
                body_scope = self.scope(function_state=FunctionState(return_type))
                for arg in args:
                    body_scope.ctxt[arg.name] = arg.type
                body_type = body_scope(body)
                self.check_returns(body_type)
                # TODO: add to self.ctxt
                return None
            case asm.Return(arg):
                return_type = self(arg)  # arg must be an expression
                self.check_return_type(return_type)
                return StmtReturnType(return_type)
            case asm.Break():
                self.check_in_loop()
                return None
            case asm.Block(bodies):
                block = self.scope()
                return_type = None
                left_already = False
                for body in bodies:
                    if left_already:
                        raise AssemblyTypeError("unreachable statements in block")
                    body_type = block(body)
                    if body is asm.Break:
                        left_already = True
                    if isinstance(body_type, StmtReturnType):
                        return_type = body_type
                        left_already = True
                return return_type
            case asm.Unpack(asm.Slot(var_n, var_t), rhs):
                rhs_type = self(rhs)
                check_type_match(var_t, rhs_type)
                if var_n in self.ctxt:
                    raise AssemblyTypeError(
                        f"Slot {var_n} is already defined in the current "
                        f"context, cannot overwrite with slot."
                    )
                self.ctxt[var_n] = var_t
                return None
            case asm.Repack(asm.Slot(var_n, var_t)):
                # TODO: check in ctxt
                return None
            case asm.Module(funcs):
                for func in funcs:
                    self(func)
                return None
            case _:
                raise AssemblyTypeError("Invalid AssemblyNode.")


def check_is_index_type(index_type):
    if not np.issubdtype(index_type, np.integer):
        raise AssemblyTypeError("invalid index type")


def check_type_match(expected_type, actual_type):
    if expected_type != actual_type:
        raise AssemblyTypeError(f"Expected {expected_type}, found {actual_type}")


def check_is_type(type_):
    if not isinstance(type_, type | FType):
        raise AssemblyTypeError(f"Expected type, {type_} is not a type.")


def check_attrtype(obj_type, attr):
    try:
        return obj_type.struct_attrtype(attr)
    except KeyError:
        raise AssemblyTypeError("{attr.val} is not attribute.") from KeyError
