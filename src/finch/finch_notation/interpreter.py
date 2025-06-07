from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..algebra import element_type, query_property
from ..codegen import isinstanceorformat
from ..symbolic import ScopedDict
from . import nodes as ntn


@dataclass(eq=True, frozen=True)
class TensorView:
    idxs: tuple[Any, ...]
    tns: ntn.NotationNode

    @property
    def shape(self):
        """
        Get the shape of the tensor view.
        This is the shape of the tensor at the specified indices.
        """
        return self.tns.shape[len(self.idxs) : -1]

    @property
    def ndims(self):
        """
        Get the number of dimensions of the tensor view.
        """
        return len(self.shape)

    @property
    def element_type(self):
        """
        Get the element type of the tensor view.
        This is the type of the elements in the tensor at the specified indices.
        """
        return element_type(self.tns)

    @property
    def fill_value(self):
        """
        Get the fill value of the tensor view.
        This is the value used to fill the tensor at the specified indices.
        """
        return self.tns.fill_value

    def access(self, idxs):
        """
        Unfurl the tensor view along a specific index.
        This creates a new tensor view with the specified index unfurled.
        """
        return TensorView(idxs=self.idxs + idxs, tns=self.tns)

    def unwrap(self):
        """
        Unwrap the tensor view to get the underlying tensor.
        This returns the original tensor from which the view was created.
        """
        return self.tns[self.idxs]

    def increment(self, op, val):
        """
        Increment the value in the tensor view.
        This updates the tensor at the specified index with the operation and value.
        """
        return self.tns[self.idxs] + op(self.tns[self.idxs], val)


def access(tns, mode, idxs):
    """
    Unfurl a tensor along an index.
    This is used to create a tensor view for a specific slice of the tensor.
    """
    if hasattr(tns, "access"):
        return tns.access(idxs)
    try:
        return query_property(tns, "__self__", "access", mode, idxs)
    except AttributeError:
        return TensorView(idxs=idxs, tns=tns)


def unwrap(tns):
    """
    Unwrap a tensor view to get the underlying tensor.
    This is used to get the original tensor from a tensor view.
    """
    if hasattr(tns, "unwrap"):
        return tns.unwrap()
    try:
        return query_property(tns, "__self__", "unwrap")
    except AttributeError:
        return tns[()]


def increment(tns, op, val):
    """
    Increment a tensor view with an operation and value.
    This updates the tensor at the specified index with the operation and value.
    """
    if hasattr(tns, "increment"):
        return tns.increment(op, val)
    try:
        return query_property(tns, "__self__", "increment", op, val)
    except AttributeError:
        return tns + op(tns, val)


@dataclass(eq=True, frozen=True)
class ExtentValue:
    """
    A class to represent the extent of a loop variable.
    This is used to define the start and end values of a loop.
    """

    start: Any
    end: Any

    def loop(self, ctx, idx, body):
        for idx_e in range(self.start, self.end):
            # Create a new scope for each iteration
            ctx_2 = ctx.scope(loop_state=HaltState())
            # Assign the loop variable
            ctx_2.bindings[idx] = idx.type_(idx_e)
            # Execute the body of the loop
            ctx_2(body)
            if ctx_2.should_halt():
                break


class NotationInterpreterKernel:
    """
    A kernel for interpreting FinchNotation code.
    This is a simple interpreter that executes the assembly code.
    """

    def __init__(self, ctx, func_n, ret_t):
        self.ctx = ctx
        self.func = ntn.Variable(func_n, ret_t)

    def __call__(self, *args):
        args_i = (ntn.Literal(arg) for arg in args)
        return self.ctx(ntn.Call(self.func, args_i))


class NotationInterpreterModule:
    """
    A class to represent an interpreted module of FinchNotation.
    """

    def __init__(self, ctx, kernels):
        self.ctx = ctx
        self.kernels = kernels

    def __getattr__(self, name):
        # Allow attribute access to kernels by name
        if name in self.kernels:
            return self.kernels[name]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


@dataclass(eq=True)
class HaltState:
    """
    A class to represent the halt state of an assembly program.
    This is used to indicate whether we should break or return, and
    what the return value is if applicable.
    """

    should_halt: bool = False
    return_value: Any = None


class NotationInterpreter:
    """
    An interpreter for FinchNotation.
    """

    def __init__(self, bindings=None, types=None, loop_state=None, function_state=None):
        if bindings is None:
            bindings = ScopedDict()
        if types is None:
            types = ScopedDict()
        self.bindings = bindings
        self.types = types
        self.loop_state = loop_state
        self.function_state = function_state

    def scope(self, bindings=None, types=None, loop_state=None, function_state=None):
        """
        Create a new scope for the interpreter.
        This allows for nested scopes and variable shadowing.
        """
        if bindings is None:
            bindings = self.bindings.scope()
        if types is None:
            types = self.types.scope()
        if loop_state is None:
            loop_state = self.loop_state
        if function_state is None:
            function_state = self.function_state
        return NotationInterpreter(
            bindings=bindings,
            types=types,
            loop_state=loop_state,
            function_state=function_state,
        )

    def should_halt(self):
        """
        Check if the interpreter should halt execution.
        This is used to stop execution in loops or when a return
        statement is encountered.
        """
        return (
            self.loop_state
            and self.loop_state.should_halt
            or self.function_state
            and self.function_state.should_halt
        )

    def __call__(self, prgm: ntn.NotationNode):
        """
        Run the program.
        """
        match prgm:
            case ntn.Literal(value):
                return value
            case ntn.Variable(var_n, var_t):
                if var_n in self.types:
                    def_t = self.types[var_n]
                    if def_t != var_t:
                        raise TypeError(
                            f"Variable '{var_n}' is declared as type {def_t}, "
                            f"but used as type {var_t}."
                        )
                if var_n in self.bindings:
                    return self.bindings[var_n]
                raise KeyError(
                    f"Variable '{var_n}' is not defined in the current context."
                )
            case ntn.Call(f, args):
                f_e = self(f)
                args_e = [self(arg) for arg in args]
                return f_e(*args_e)
            case ntn.Unwrap(tns):
                return unwrap(self(tns))
            case ntn.Access(tns, mode, idxs):
                tns_e = self(tns)
                idxs_e = [self(idx) for idx in idxs]
                return access(tns_e, mode, idxs)
            case ntn.Increment(tns, op, val):
                tns_e = self(tns)
                val_e = self(val)
                op_e = self(op)
                tns_e[idxs_e] = op_e(tns_e[idxs_e], val_e)
                return None
            case ntn.Block(bodies):
                for body in bodies:
                    if self.should_halt():
                        break
                    self(body)
                return None
            case ntn.Loop(idx, ext, body):
                ext_e = self(ext)
                ext_e.loop(self, idx, body)
                return None
            case ntn.If(cond, body):
                if self(cond):
                    ctx_2 = self.scope()
                    ctx_2(body)
                return None
            case ntn.IfElse(cond, body, else_body):
                if not self(cond):
                    body = else_body
                ctx_2 = self.scope()
                ctx_2(body)
                return None
            case ntn.Function(ntn.Variable(func_n, ret_t), args, body):

                def my_func(*args_e):
                    ctx_2 = self.scope(function_state=HaltState())
                    if len(args_e) != len(args):
                        raise ValueError(
                            f"Function '{func_n}' expects {len(args)} arguments, "
                            f"but got {len(args_e)}."
                        )
                    for arg, arg_e in zip(args, args_e, strict=False):
                        match arg:
                            case ntn.Variable(arg_n, arg_t):
                                if not isinstanceorformat(arg_e, arg_t):
                                    raise TypeError(
                                        f"Argument '{arg_n}' is expected to be of type "
                                        f"{arg_t}, but got {type(arg_e)}."
                                    )
                                ctx_2.bindings[arg_n] = arg_e
                            case _:
                                raise NotImplementedError(
                                    f"Unrecognized argument type: {arg}"
                                )
                    ctx_2(body)
                    if ctx_2.function_state.should_halt:
                        ret_e = ctx_2.function_state.return_value
                        if not isinstance(ret_e, ret_t):
                            raise TypeError(
                                f"Return value {ret_e} is not of type {ret_t} "
                                f"for function '{func_n}'."
                            )
                        return ret_e
                    raise ValueError(
                        f"Function '{func_n}' did not return a value, "
                        f"but expected type {ret_t}."
                    )

                self.bindings[func_n] = my_func
                return None
            case ntn.Return(value):
                self.function_state.return_value = self(value)
                self.function_state.should_halt = True
                return None
            case ntn.Break():
                self.loop_state.should_halt = True
                return None
            case ntn.Module(funcs):
                for func in funcs:
                    self(func)
                kernels = {}
                for func in funcs:
                    match func:
                        case ntn.Function(ntn.Variable(func_n, ret_t), args, _):
                            kernel = NotationInterpreterKernel(self, func_n, ret_t)
                            kernels[func_n] = kernel
                        case _:
                            raise NotImplementedError(
                                f"Unrecognized function definition: {func}"
                            )
                return NotationInterpreterModule(self, kernels)
            case _:
                raise NotImplementedError(
                    f"Unrecognized assembly node type: {type(prgm)}"
                )
