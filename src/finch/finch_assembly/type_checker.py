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


def stmt_return(type_=None):
    if type_ is None:
        return None
    if isinstance(type_, StmtReturnType):
        type_ = type_.return_type
    return StmtReturnType(type_)


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
            raise (AssemblyTypeError("not in loop"))

    def check_return_type(self, return_type):
        if self.function_state:
            check_type_match(self.function_state.return_type, return_type)
            return
        raise AssemblyTypeError("Cannot return outside of function")

    def check_returns(self, return_type):
        if isinstance(return_type, StmtReturnType):
            return_type = return_type.return_type
        check_type_match(self.function_state.return_type, return_type)

    def check_in_ctxt(self, var_n, var_t):
        try:
            check_type_match(self.ctxt[var_n], var_t)
        except KeyError:
            raise AssemblyTypeError(
                f"'{var_n}' is not defined in the current context."
            ) from KeyError

    def check_buffer(self, buffer):
        if not isinstance(buffer, asm.Slot | asm.Stack):
            raise AssemblyTypeError("Buffer expression must be slot or stack")
        buffer_type = self(buffer)
        check_is_buffer_type(buffer_type)
        return buffer_type

    def __call__(self, prgm: asm.AssemblyNode):
        match prgm:
            case asm.Literal(value) as lit:
                return ftype(lit.val)
            case asm.Variable(var_n, var_t) as var:
                check_var(var)
                self.check_in_ctxt(var_n, var_t)
                return var_t
            case asm.Slot(var_n, var_t) as slot:
                check_slot(slot)
                self.check_in_ctxt(var_n, var_t)
                return var_t
            case asm.Stack(obj, obj_t):
                check_is_type(obj_t)
                return obj_t
            case asm.Assign(lhs, rhs):
                match lhs:
                    case asm.Variable(var_n, var_t) as var:
                        check_var(var)
                        rhs_type = self(rhs)
                        check_is_type(rhs_type)  # rhs must be expression with type
                        check_type_match(var_t, rhs_type)
                        if var_n in self.ctxt:
                            check_type_match(self.ctxt[var_n], var_t)
                        self.ctxt[var_n] = var_t
                        return None
                    case asm.Stack(_obj, _obj_t):
                        return None  # TODO
            case asm.GetAttr(obj, attr):
                obj_type = self(obj)
                check_is_struct_type(obj_type)
                attr = check_is_literal(attr)
                return check_attrtype(obj_type, attr)
            case asm.SetAttr(obj, attr, value):
                obj_type = self(obj)
                check_is_struct_type(obj_type)
                attr = check_is_literal(attr)
                attrtype = check_attrtype(obj_type, attr)
                value_type = self(value)
                check_is_type(value_type)  # rhs must be expression with type
                check_type_match(attrtype, value_type)
                return None
            case asm.Call(op, args):
                op = check_is_literal(op)
                arg_types = [self(arg) for arg in args]
                try:
                    return algebra.return_type(op, *arg_types)
                except AttributeError:
                    raise AssemblyTypeError(
                        "return type of function is not registered"
                    ) from AttributeError
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
            case asm.ForLoop(var, start, end, body):
                (var_n, var_t) = check_is_var(var)
                start_type = self(start)
                end_type = self(end)
                check_index(start_type)
                check_type_match(var_t, start_type)
                check_type_match(var_t, end_type)
                loop = self.scope(loop_state=LoopState())
                loop.ctxt[var_n] = var_t
                return None
            case asm.BufferLoop(buffer, var, body):
                buffer_type = self.check_buffer(buffer)
                (var_n, var_t) = check_is_var(var)
                check_type_match(buffer_type.element_type, var_t)
                loop = self.scope(loop_state=LoopState())
                loop.ctxt[var_n] = var_t
                return None
            case asm.WhileLoop(cond, body):
                cond_type = self(cond)
                check_condition_type(cond_type)
                return None
            case asm.If(cond, body):
                cond_type = self(cond)
                check_condition_type(cond_type)
                return None
            case asm.IfElse(cond, body, else_body):
                cond_type = self(cond)
                check_condition_type(cond_type)
                body_type = self(body)
                else_body_type = self(else_body)
                if body_type is None or else_body_type is None:
                    return None
                check_type_match(body_type, else_body_type)
                return stmt_return(body_type)
            case asm.Function(name, args, body):
                (name, return_type) = check_is_var(name)
                body_scope = self.scope(function_state=FunctionState(return_type))
                for arg in args:
                    (arg_n, arg_t) = check_is_var(arg)
                    body_scope.ctxt[arg_n] = arg_t
                body_type = body_scope(body)
                self.check_returns(body_type)
                # TODO: add to self.ctxt
                return None
            case asm.Return(arg):
                return_type = self(arg)
                check_is_type(return_type)
                self.check_return_type(return_type)
                return stmt_return(return_type)
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
            case asm.Unpack(slot, rhs):
                # TODO: find corresponding repack
                # TODO: rhs cannot be accessed or modified until repack
                check_is_slot(slot)
                rhs_type = self(rhs)
                check_type_match(slot.type, rhs_type)
                if slot.name in self.ctxt:
                    raise AssemblyTypeError(
                        f"Slot {slot.name} is already defined in the current "
                        f"context, cannot overwrite with slot."
                    )
                self.ctxt[slot.name] = slot.type
                return None
            case asm.Repack(slot):
                # TODO: allow rhs of unpack to be modified
                check_is_slot(slot)
                return None


def check_is_buffer_type(buffer_type):
    if not isinstance(buffer_type, BufferFType):
        raise AssemblyTypeError("expected buffer")


def check_index(index_type):
    if not np.issubdtype(index_type, np.integer):
        raise AssemblyTypeError("invalid index type")


def check_condition_type(type_):
    if (
        not isinstance(type_, type)
        or not np.issubdtype(type_, np.number)
        and not np.issubdtype(type_, np.bool_)
    ):
        raise AssemblyTypeError("Expected number or boolean for conditional")


def check_type_match(expected_type, actual_type):
    if expected_type != actual_type:
        raise AssemblyTypeError(f"Expected {expected_type}, found {actual_type}")


def check_is_literal(lit):
    if not isinstance(lit, asm.Literal):
        raise AssemblyTypeError("Expected literal.")
    return lit.val


def check_is_str(var_n):
    if not isinstance(var_n, str):
        raise AssemblyTypeError("Identifier must be a string.")


def check_is_expr(expr):
    if not isinstance(expr, asm.AssemblyExpr):
        raise AssemblyTypeError("Expected expression")


def check_is_type(type_):
    if not isinstance(type_, type | FType):
        raise AssemblyTypeError(f"Expected type, {type_} is not a type.")


def check_var(var):
    check_is_str(var.name)
    check_is_type(var.type)
    return (var.name, var.type)


def check_slot(slot):
    check_is_str(slot.name)
    check_is_type(slot.type)
    return (slot.name, slot.type)


def check_is_slot(slot):
    if isinstance(slot, asm.Slot):
        return check_slot(slot)
    raise AssemblyTypeError("Expected slot.")


def check_is_var(var):
    if isinstance(var, asm.Variable):
        return check_var(var)
    raise AssemblyTypeError("Expected variable.")


def check_is_struct_type(type_):
    if not isinstance(type_, AssemblyStructFType):
        raise AssemblyTypeError("Expected struct.")


def check_attrtype(obj_type, attr):
    try:
        return obj_type.struct_attrtype(attr)
    except KeyError:
        raise AssemblyTypeError("{attr.val} is not attribute.") from KeyError
