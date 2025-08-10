from ..symbolic import FType, ScopedDict
from . import nodes as asm


class AssemblyTypeChecker:
    """
    A type checker for FinchAssembly
    """

    def __init__(self, ctxt=None):
        if ctxt is None:
            ctxt = ScopedDict()
        self.ctxt = ctxt

    def check_in_ctxt(self, var_n, var_t):
        try:
            check_type_match(self.ctxt[var_n], var_t)
        except KeyError:
            raise TypeError(
                f"'{var_n}' is not defined in the current context."
            ) from KeyError

    def scope(self):
        return None  # TODO

    def __call__(self, prgm: asm.AssemblyNode):
        match prgm:
            case asm.Literal(value):
                return prgm.result_format
            case asm.Variable(var_n, var_t) | asm.Slot(var_n, var_t):
                check_is_str(var_n)
                check_is_type(var_t)
                self.check_in_ctxt(var_n, var_t)
                return var_t
            case asm.Unpack(lhs, rhs):
                # TODO: make sure rhs is not modified
                # TODO: make sure there is an unpack
                # TODO: is the rhs always a variable
                match lhs:
                    case asm.Slot(var_n, var_t):
                        check_is_str(var_n)
                        check_is_type(var_t)
                        rhs_type = self(rhs)
                        check_is_type(rhs_type)
                        check_type_match(var_t, rhs_type)
                        if var_n in self.ctxt:
                            raise TypeError(
                                f"Slot {var_n} is already defined in the current "
                                f"context, cannot overwrite with slot."
                            )
                        self.ctxt[var_n] = var_t
                        return None
                    case _:
                        raise TypeError("Left-hand side of unpack must be a slot.")
            case asm.Repack(val):
                # TODO: allow variable to be modified after
                if not isinstance(val, asm.Slot):
                    raise TypeError("Left-hand side of repack must be a slot.")
                return None
            case asm.Assign(lhs, rhs):
                match lhs:
                    case asm.Variable(var_n, var_t):
                        check_is_str(var_n)
                        check_is_type(var_t)
                        rhs_type = self(var_t)
                        check_type_match(var_t, rhs_type)
                        self.ctxt[var_n] = var_t
                        return None
                    case _:
                        raise TypeError(
                            "Left-hand side of assignment must be a variable."
                        )
            case asm.GetAttr(obj, attr):
                obj_type = self(obj)
                check_is_struct_type(obj_type)
                check_is_literal(attr)
                return obj_type.struct_attrtype(attr.val)
            case asm.SetAttr(obj, attr, value):
                obj_type = self(obj)
                check_is_struct_type(obj_type)
                check_is_literal(attr)
                value_type = self(value)
                check_type_match(obj_type.struct_attrtype(attr.val), value_type)
                return None
            case asm.Call(_op, _args):
                # DOUBLE CHECK
                return None  # TODO
            case asm.Load(buffer, index):
                buffer_type = check_is_buffer_expr(buffer)
                index_type = self(index)
                check_type_match(buffer_type.length_type, index_type)
                return buffer_type.element_type
            case asm.Store(buffer, index, value):
                buffer_type = check_is_buffer_expr(buffer)
                index_type = self(index)
                check_type_match(buffer_type.length_type, index_type)
                check_type_match(buffer_type.element_type, index_type)
                return buffer_type.length_type
            case asm.Resize(buffer, new_size):
                buffer_type = check_is_buffer_expr(buffer)
                new_size_type = self(new_size)
                check_type_match(buffer_type.length_type, new_size_type)  # DOUBLE CHECK
                return buffer_type
            case asm.Length(buffer):
                buffer_type = check_is_buffer_expr(buffer)
                return buffer_type.length_type
            case asm.ForLoop(var, start, end, body):
                check_is_var(var)
                start_type = self(start)
                end_type = self(end)
                check_type_match(var.type, start_type)
                check_type_match(var.type, end_type)
                check_is_block(body)  # DOUBLE CHECK
                loop = self.scope()
                loop.ctxt[var.name] = var.type
                loop(body)
                return None
            case asm.BufferLoop(buffer, var, body):
                buffer_type = check_is_buffer_expr(buffer)
                check_is_var(var)
                check_type_match(buffer_type.length_type, var.type)
                check_is_block(body)  # DOUBLE CHECK
                loop = self.scope()
                loop.ctxt[var.name] = var.type
                loop(body)
                return None
            case asm.WhileLoop(cond, body):
                cond_type = self(cond)
                check_is_bool(cond_type)  # DOUBLE CHECK
                check_is_block(body)  # DOUBLE CHECK
                self(body)
                return None
            case asm.If(cond, body):
                cond_type = self(cond)
                check_is_bool(cond_type)  # DOUBLE CHECK
                check_is_block(body)  # DOUBLE CHECK
                self(body)
                return None
            case asm.IfElse(cond, body, else_body):
                cond_type = self(cond)
                check_is_bool(cond_type)  # DOUBLE CHECK
                body_type = self(body)
                else_body_type = self(else_body)
                if body_type is None or else_body_type is None:
                    return None
                check_type_match(body_type, else_body_type)
                return body_type
            case asm.Function(_name, _args, _body):
                return None  # TODO
            case asm.Return(arg):
                # check if in function
                return self(arg)
            case asm.Break():
                # check if in loop
                return None
            case asm.Block(stmts):
                for stmt in stmts:
                    self(stmt)
                return None


def check_is_struct_type(type_):
    return None  # TODO


def check_is_literal(expr):
    return None  # TODO


def check_is_buffer_expr(expr):
    return None  # TODO


def check_is_var(expr):
    return None  # TODO


def check_is_block(expr):
    return None  # TODO


def check_is_bool(expr):
    return None  # TODO


def check_is_type(type_):
    if not isinstance(type_, type | FType):
        raise TypeError(f"Expected type, {type_} is not a type.")


def check_is_str(var_n):
    if not isinstance(var_n, str):
        raise TypeError("Variable/slot identifier must be a string.")


def check_type_match(expected_type, actual_type):
    if expected_type != actual_type:
        raise TypeError(f"Expected {expected_type}, found {actual_type}")
