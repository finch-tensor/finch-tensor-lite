from dataclasses import dataclass

import numpy as np

from .. import algebra
from ..symbolic import FType, ScopedDict, ftype
from . import nodes as asm
from .buffer import BufferFType
from .struct import AssemblyStructFType


@dataclass(eq=True)
class AutoType:
    def __init__(self):
        pass


class AssemblyTypeError(Exception):
    pass


@dataclass(eq=True)
class FunctionState:
    def __init__(self, return_type):
        self.return_type = return_type


@dataclass(eq=True)
class LoopState:
    def __init__(self):
        pass


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

    def check_return_type(self, return_type):
        if self.function_state:
            check_type_match(
                self.function_state.return_type,
                return_type,
            )
        else:
            raise AssemblyTypeError("Cannot return outside of function.")

    def check_in_ctxt(self, var_n, var_t):
        try:
            type_in_ctxt = self.ctxt[var_n]
            check_type_match(var_t, self.ctxt[var_n])
            return type_in_ctxt
        except KeyError:
            raise AssemblyTypeError(
                f"The variable '{var_n}' is not defined in the current context."
            ) from KeyError

    def check_buffer(self, buffer):
        buffer_type = self.check_expr(buffer)
        if isinstance(buffer_type, BufferFType):
            return buffer_type
        raise AssemblyTypeError(f"Expected buffer, got {buffer_type}.")

    def check_struct(self, struct):
        struct_type = self.check_expr(struct)
        if isinstance(struct_type, AssemblyStructFType):
            return struct_type
        raise AssemblyTypeError(f"Expected struct, got {struct_type}.")

    def check_cond(self, cond):
        cond_type = self.check_expr(cond)
        if isinstance(cond_type, type) and (
            np.issubdtype(cond_type, np.number) or np.issubdtype(cond_type, np.bool_)
        ):
            return
        raise AssemblyTypeError("Conditional must be number or boolean.")

    def check_expr(self, expr: asm.AssemblyExpression):
        match expr:
            case asm.Literal(value):
                return ftype(value)
            case asm.Variable(var_n, var_t) as var:
                check_type(var_t)
                var.type = self.check_in_ctxt(var_n, var_t)
                return var.type
            case asm.Slot(var_n, var_t) as slot:
                check_type(var_t)
                slot.type = self.check_in_ctxt(var_n, var_t)
                return slot.type
            case asm.Stack(obj, obj_t):
                check_type(obj_t)
                return obj_t
            case asm.GetAttr(obj, asm.Literal(attr)):
                obj_type = self.check_struct(obj)
                return check_attrtype(obj_type, attr)
            case asm.Call(asm.Literal(op), args):
                arg_types = [self.check_expr(arg) for arg in args]
                try:
                    return algebra.return_type(op, *arg_types)
                except (AttributeError, TypeError):
                    raise AssemblyTypeError(
                        "Return type of function is not registered."
                    ) from AttributeError
                    raise AssemblyTypeError(
                        "Operation not defined on given types."
                    ) from TypeError
            case asm.Load(buffer, index):
                buffer_type = self.check_buffer(buffer)
                index_type = self.check_expr(index)
                check_type_match(buffer_type.length_type, index_type)
                return buffer_type.element_type
            case asm.Length(buffer):
                buffer_type = self.check_buffer(buffer)
                return buffer_type.length_type
            case _:
                raise ValueError(f"Ill-formed AssemblyExpression:  {type(expr)}.")

    def check_stmt(self, stmt: asm.AssemblyNode):
        if isinstance(stmt, asm.AssemblyExpression):
            self.check_expr(stmt)
            return None
        match stmt:
            case asm.Unpack(asm.Slot(var_n, var_t) as slot, rhs):
                check_type(var_t)
                rhs_type = self(rhs)
                check_type_match(var_t, rhs_type)
                slot.type = rhs_type
                if var_n in self.ctxt:
                    raise AssemblyTypeError(
                        f"Slot {var_n} is already defined in the current "
                        f"context, cannot overwrite with slot."
                    )
                self.ctxt[var_n] = rhs_type
                return None
            case asm.Repack(asm.Slot(var_n, var_t) as slot):
                check_type(var_t)
                slot.type = self.check_in_ctxt(var_n, var_t)
                return None
            case asm.Assign(asm.Variable(var_n, var_t) as var, rhs):
                check_type(var_t)
                rhs_type = self.check_expr(rhs)
                check_type_match(var_t, rhs_type)
                var.type = rhs_type
                if var_n in self.ctxt:
                    check_type_match(self.ctxt[var_n], rhs_type)
                else:
                    self.ctxt[var_n] = rhs_type
                return None
            case asm.Assign(asm.Stack(_obj, _obj_t), _rhs):
                raise NotImplementedError("Cannot assign to stack currently.")
            case asm.SetAttr(obj, asm.Literal(attr), value):
                obj_type = self.check_struct(obj)
                attrtype = check_attrtype(obj_type, attr)
                value_type = self.check_expr(value)
                check_type_match(attrtype, value_type)
                return None
            case asm.Store(buffer, index, value):
                buffer_type = self.check_buffer(buffer)
                index_type = self.check_expr(index)
                check_type_match(buffer_type.length_type, index_type)
                value_type = self.check_expr(value)
                check_type_match(buffer_type.element_type, value_type)
                return None
            case asm.Resize(buffer, new_size):
                buffer_type = self.check_buffer(buffer)
                new_size_type = self.check_expr(new_size)
                check_type_match(buffer_type.length_type, new_size_type)
                return None
            case asm.ForLoop(asm.Variable(var_n, var_t) as var, start, end, body):
                check_type(var_t)
                start_type = self.check_expr(start)
                end_type = self.check_expr(end)
                check_type_match(var_t, start_type)
                check_type_match(var_t, end_type)
                check_is_index_type(start_type)
                var.type = start_type
                loop = self.scope(loop_state=LoopState())
                loop.ctxt[var_n] = start_type
                loop.check_stmt(body)
                return None
            case asm.BufferLoop(buffer, asm.Variable(var_n, var_t) as var, body):
                check_type(var_t)
                buffer_type = self.check_buffer(buffer)
                check_type_match(var_t, buffer_type.element_type)
                var.type = buffer_type.element_type
                loop = self.scope(loop_state=LoopState())
                loop.ctxt[var_n] = buffer_type.element_type
                loop.check_stmt(body)
                return None
            case asm.WhileLoop(cond, body):
                self.check_cond(cond)
                self.loop_state = LoopState()
                self.check_stmt(body)
                self.loop_state = None
                return None
            case asm.If(cond, body):
                self.check_cond(cond)
                self.check_stmt(body)
                return None
            case asm.IfElse(cond, body, else_body):
                self.check_cond(cond)
                body_type = self.check_stmt(body)
                else_body_type = self.check_stmt(else_body)
                if body_type is None or else_body_type is None:
                    return None
                return body_type
            case asm.Function(asm.Variable(func_name, return_type), args, body):
                check_type(return_type, auto=False)
                if self.function_state:
                    raise AssemblyTypeError(
                        f"Cannot nest function definitions:  '{func_name}'."
                    )
                body_scope = self.scope(function_state=FunctionState(return_type))
                for arg in args:
                    check_type(arg.type, auto=False)
                    body_scope.ctxt[arg.name] = arg.type
                body_type = body_scope.check_stmt(body)
                if body_type is None:
                    raise AssemblyTypeError(
                        f"Function '{func_name}' is not guaranteed to return."
                    )
                check_type_match(return_type, body_type)
                return None
            case asm.Return(arg):
                return_type = self.check_expr(arg)
                self.check_return_type(return_type)
                return return_type
            case asm.Break():
                if self.loop_state is None:
                    raise AssemblyTypeError("Cannot break outside of loop.")
                return None
            case asm.Block(bodies):
                block = self.scope()
                return_type = None
                left_already = False
                for body in bodies:
                    if left_already:
                        raise AssemblyTypeError("Unreachable statements in block.")
                    return_type = block.check_stmt(body)
                    left_already = return_type is not None or body is asm.Break
                return return_type
            case _:
                raise ValueError(f"Ill-formed statement:  {type(stmt)}.")

    def check_module(self, mod: asm.AssemblyNode):
        defined_funcs = []
        match mod:
            case asm.Module(funcs):
                for func in funcs:
                    func_name = check_is_func(func)  # can be removed later
                    if func_name in defined_funcs:
                        raise AssemblyTypeError(
                            f"Two functions defined with the name '{func_name}'."
                        )
                    self.check_stmt(func)
                    defined_funcs.append(func_name)
                return
            case _:
                raise AssemblyTypeError(f"Expected module, got {type(mod)}.")

    def __call__(self, prgm: asm.AssemblyNode):
        if isinstance(prgm, asm.Module):
            self.check_module(prgm)
            return None
        if isinstance(prgm, asm.AssemblyExpression):
            return self.check_expr(prgm)
        self.check_stmt(prgm)
        return None


def check_is_index_type(index_type):
    if not np.issubdtype(index_type, np.integer):
        raise AssemblyTypeError(f"Expected index, got {index_type}.")


def check_type_match(expected_type, actual_type):
    if expected_type is not AutoType and expected_type != actual_type:
        raise AssemblyTypeError(f"Expected {expected_type}, got {actual_type}.")


def check_type(type_, auto=True):
    if auto:
        if not isinstance(type_, type | FType | AutoType):
            raise AssemblyTypeError(f"Expected type, got {type_}.")
    else:
        if isinstance(type_, AutoType):
            raise AssemblyTypeError("Expected concrete type, got {type_}.")
        if not isinstance(type_, type | FType):
            raise AssemblyTypeError(f"Expected type, got {type_}.")


def check_attrtype(obj_type, attr):
    try:
        return obj_type.struct_attrtype(attr)
    except KeyError:
        raise AssemblyTypeError(f"'{attr}' is not valid attribute.") from KeyError


def check_is_func(func):
    match func:
        case asm.Function(asm.Variable(func_name, _), _, _):
            return func_name
        case _:
            raise AssemblyTypeError(f"Expected function definition, got {type(func)}.")
