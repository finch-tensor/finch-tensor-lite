from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..algebra import (
    Tensor,
    TensorFType,
    element_type,
    fill_value,
    query_property,
    register_property,
    shape_type,
)
from ..finch_assembly import nodes as asm
from ..symbolic import ScopedDict, fisinstance, ftype
from . import nodes as ntn


class TensorViewFType(TensorFType):
    """
    A ftype for tensor views.
    This is used to represent the ftype of a tensor at specific indices.
    It is a subclass of ntn.FType to allow for custom formatting.
    """

    def __init__(self, idxs: tuple[Any, ...], tns: Any, op: Any = None):
        """
        Initialize the TensorViewFType with the specified indices, tensor, and
        operation.

        :param idxs: Tuple of index types for the tensor view.
        :param tns: The underlying tensor ftype.
        :param op: The operation applied to the tensor view, if any.
        """
        self.idxs = idxs
        self.tns = tns
        self.op = op

    def __eq__(self, other):
        return (
            isinstance(other, TensorViewFType)
            and self.idxs == other.idxs
            and self.tns == other.tns
            and self.op == other.op
        )

    def __hash__(self):
        return hash((self.idxs, self.tns, self.op))

    @property
    def element_type(self):
        return element_type(self.tns)

    @property
    def fill_value(self):
        """
        Get the fill value of the tensor view.
        This is the value used to fill the tensor at the specified indices.
        """
        return fill_value(self.tns)

    @property
    def shape_type(self):
        """
        Get the shape type of the tensor view.
        This is the shape type of the tensor at the specified indices.
        """
        return shape_type(self.tns)[len(self.idxs) : -1]


class TensorView(Tensor):
    def __init__(self, idxs: tuple[Any, ...], tns: ntn.NotationNode, op: Any = None):
        """
        Initialize the TensorView with the specified indices, tensor, and operation.

        :param idxs: Tuple of index types for the tensor view.
        :param tns: The underlying tensor node.
        :param op: The operation applied to the tensor view, if any.
        """
        self.idxs = idxs
        self.tns = tns
        self.op = op

    @property
    def ftype(self):
        """
        Get the ftype of the tensor view.
        This is the ftype of the tensor at the specified indices.
        """
        return TensorViewFType(map(ftype, self.idxs), self.tns.ftype, self.op)

    @property
    def shape(self):
        """
        Get the shape of the tensor view.
        This is the shape of the tensor at the specified indices.
        """
        return self.tns.shape[len(self.idxs) : -1]

    @property
    def ndim(self):
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

    def access(self, idxs, op=None):
        """
        Unfurl the tensor view along a specific index.
        This creates a new tensor view with the specified index unfurled.
        """
        return TensorView(idxs=self.idxs + idxs, tns=self.tns, op=op)

    def unwrap(self):
        """
        Unwrap the tensor view to get the underlying tensor.
        This returns the original tensor from which the view was created.
        """
        return self.tns[*self.idxs]

    def increment(self, val):
        """
        Increment the value in the tensor view.
        This updates the tensor at the specified index with the operation and value.
        """
        self.tns[*self.idxs] = self.op(self.tns[*self.idxs], val)
        return


def access(tns, idxs, op=None):
    """
    Unfurl a tensor along an index.
    This is used to create a tensor view for a specific slice of the tensor.
    """
    if hasattr(tns, "access"):
        return tns.access(idxs, op)
    try:
        return query_property(tns, "access", "__attr__", idxs, op)
    except AttributeError:
        return TensorView(idxs=idxs, tns=tns, op=op)


def unwrap(tns):
    """
    Unwrap a tensor view to get the underlying tensor.
    This is used to get the original tensor from a tensor view.
    """
    if hasattr(tns, "unwrap"):
        return tns.unwrap()
    return query_property(tns, "unwrap", "__attr__")


def increment(tns, val):
    """
    Increment a tensor view with an operation and value.
    This updates the tensor at the specified index with the operation and value.
    """
    if hasattr(tns, "increment"):
        return tns.increment(val)
    return query_property(tns, "increment", "__attr__", val)


def declare(tns, init, op, shape):
    """
    Declare a tensor.
    """
    if hasattr(tns, "declare"):
        return tns.declare(init, op, shape)
    return query_property(tns, "declare", "__attr__", init, op, shape)


def np_declare(tns, init, op, shape):
    for dim in shape:
        if dim.start != 0:
            raise ValueError(
                f"Invalid dimension start value {dim.start} for ndarray declaration."
            )
    shape = tuple(dim.end for dim in shape)
    tns.fill(init)
    if tns.shape != shape:
        tns = np.broadcast_to(tns, shape).copy()
    return tns


register_property(np.ndarray, "declare", "__attr__", np_declare)


def freeze(tns, op):
    """
    Freeze a tensor.
    """
    if hasattr(tns, "freeze"):
        return tns.freeze(op)
    try:
        query_property(tns, "freeze", "__attr__", op)
    except AttributeError:
        return tns


def thaw(tns, op):
    """
    Thaw a tensor.
    """
    if hasattr(tns, "freeze"):
        return tns.freeze(op)
    try:
        return query_property(tns, "freeze", "__attr__", op)
    except AttributeError:
        return tns


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
    A class to represent the halt state of a notation program.
    These programs can't break, but calling return sets a special return value.
    """

    has_returned: bool = False
    return_value: Any = None


class NotationInterpreter:
    """
    An interpreter for FinchNotation.
    """

    def __init__(
        self,
        bindings=None,
        slots=None,
        types=None,
        loop_state=None,
        function_state=None,
    ):
        if bindings is None:
            bindings = ScopedDict()
        if slots is None:
            slots = ScopedDict()
        if types is None:
            types = ScopedDict()
        self.bindings = bindings
        self.slots = slots
        self.types = types
        self.loop_state = loop_state
        self.function_state = function_state

    def scope(
        self,
        bindings=None,
        slots=None,
        types=None,
        loop_state=None,
        function_state=None,
    ):
        """
        Create a new scope for the interpreter.
        This allows for nested scopes and variable shadowing.
        """
        if bindings is None:
            bindings = self.bindings.scope()
        if slots is None:
            slots = self.slots.scope()
        if types is None:
            types = self.types.scope()
        if loop_state is None:
            loop_state = self.loop_state
        if function_state is None:
            function_state = self.function_state
        return NotationInterpreter(
            bindings=bindings,
            slots=slots,
            types=types,
            loop_state=loop_state,
            function_state=function_state,
        )

    def __call__(self, prgm: ntn.NotationNode | asm.AssemblyNode):
        """
        Run the program.
        """
        match prgm:
            case ntn.Literal(val) | asm.Literal(val):
                return val
            case ntn.Value(val, val_t):
                val_e = self(val)
                if type(val_e) is not val_t:
                    raise TypeError(
                        f"Value '{val_e}' is expected to be of type {val_t}, "
                        f"but is a type {type(val_e)}."
                    )
                return val_e
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
            case ntn.Assign(var, val):
                val_e = self(val)
                if isinstance(var, ntn.Variable):
                    var_n = var.name
                    self.bindings[var_n] = val_e
                    return None
                raise NotImplementedError(f"Unrecognized assignment target: {var}")
            case ntn.Slot(var_n, var_t):
                if var_n in self.types:
                    def_t = self.types[var_n]
                    if def_t != var_t:
                        raise TypeError(
                            f"Slot '{var_n}' is declared as type {def_t}, "
                            f"but used as type {var_t}."
                        )
                if var_n in self.slots:
                    return self.slots[var_n]
                raise KeyError(f"Slot '{var_n}' is not defined in the current context.")
            case ntn.Unpack(ntn.Slot(var_n, var_t), val):
                val_e = self(val)
                if not fisinstance(val_e, var_t):
                    raise TypeError(
                        f"Assigned value {val_e} is not of type {var_t} for "
                        f"variable '{var_n}'."
                    )
                assert var_n not in self.types, (
                    f"Variable '{var_n}' is already defined in the current"
                    f" context, cannot overwrite with slot."
                )
                self.types[var_n] = var_t
                self.slots[var_n] = val_e
                return None
            case ntn.Repack(ntn.Slot(var_n, var_t), val):
                if isinstance(val, ntn.Variable):
                    val_n = val.name
                    self.bindings[val_n] = self.slots[var_n]
                    del self.types[var_n]
                    del self.slots[var_n]
                    return None
                raise NotImplementedError(f"Unrecognized repack obj target: {val}")
            case ntn.Access(tns, mode, idxs):
                assert isinstance(tns, ntn.Slot)
                tns_e = self(tns)
                idxs_e = [self(idx) for idx in idxs]
                match mode:
                    case ntn.Read():
                        return access(tns_e, idxs_e)
                    case ntn.Update(op):
                        op_e = self(op)
                        return access(tns_e, idxs_e, op=op_e)
                    case _:
                        raise NotImplementedError(f"Unrecognized access mode: {mode}")
            case ntn.Increment(tns, val):
                tns_e = self(tns)
                val_e = self(val)
                increment(tns_e, val_e)
                return None
            case ntn.Block(bodies):
                for body in bodies:
                    self(body)
                return None
            case ntn.Loop(idx, ext, body):
                ext_e = self(ext)
                ext_e.loop(self, idx, body)
                return None
            case ntn.Declare(tns, init, op, shape):
                assert isinstance(tns, ntn.Slot)
                tns_e = self(tns)
                init_e = self(init)
                op_e = self(op)
                shape_e = [self(s) for s in shape]
                self.slots[tns.name] = declare(tns_e, init_e, op_e, shape_e)
                return None
            case ntn.Freeze(tns, op):
                assert isinstance(tns, ntn.Slot)
                tns_e = self(tns)
                op_e = self(op)
                self.slots[tns.name] = freeze(tns_e, op_e)
                return None
            case ntn.Thaw(tns, op):
                assert isinstance(tns, ntn.Slot)
                tns_e = self(tns)
                op_e = self(op)
                self.slots[tns.name] = thaw(tns_e, op_e)
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
                            case ntn.Variable(arg_n, _):
                                ctx_2.bindings[arg_n] = arg_e
                            case _:
                                raise NotImplementedError(
                                    f"Unrecognized argument type: {arg}"
                                )
                    ctx_2(body)
                    if ctx_2.function_state.has_returned:
                        ret_e = ctx_2.function_state.return_value
                        if not fisinstance(ret_e, ret_t):
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
                self.function_state.has_returned = True
                self.function_state.return_value = self(value)
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
            case ntn.Stack(val):
                raise NotImplementedError(
                    "NotationInterpreter does not support symbolic, no target language"
                )
            case _:
                raise NotImplementedError(
                    f"Unrecognized notation node type: {type(prgm)}"
                )
